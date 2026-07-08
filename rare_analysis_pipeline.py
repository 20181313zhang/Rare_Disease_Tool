"""
rare_analysis_pipeline.py - 完整业务流水线
串行执行罕见病筛查分析的全部5个步骤：
  步骤1: LLM结构化解析病历
  步骤2: LLM生成检索Query -> 批量调用4类证据检索接口 -> 汇总证据
  步骤3: LLM综合推理（结构化病历 + 证据）-> 罕见病筛查分析
  步骤4: LLM整理排版报告 + 参考文献列表
  步骤5: 返回完整报告（供MCP异步任务取用）
"""

import json
import logging
from typing import Dict, Any, Callable, Optional, List

from llm_engine import (
    extract_medical_structure,
    generate_search_queries,
    rare_disease_reasoning,
    format_screening_report,
    sanitize_output,
)
from knows_api import batch_search_evidences
from config import COMPLIANCE_DISCLAIMER, REPORT_FOOTER

logger = logging.getLogger(__name__)


class RareDiseasePipeline:
    """
    罕见病筛查分析完整流水线
    支持进度回调，供前端/异步任务实时获取执行状态
    """

    def __init__(self, progress_callback: Optional[Callable[[str, float], None]] = None):
        """
        Args:
            progress_callback: 进度回调函数 callback(message: str, progress: float)
                               progress范围 0.0~1.0
        """
        self.progress_callback = progress_callback
        self._current_step = ""
        self._step_logs = []  # 记录每步执行日志

    def _report_progress(self, message: str, progress: float):
        """上报进度"""
        self._current_step = message
        self._step_logs.append(f"[{progress*100:.0f}%] {message}")
        logger.info(f"[Pipeline] {message} ({progress*100:.0f}%)")
        if self.progress_callback:
            try:
                self.progress_callback(message, progress)
            except Exception as e:
                logger.warning(f"[Pipeline] 进度回调异常: {e}")

    def run(
        self,
        records_text: str,
        family_history: str = "",
        excluded_diseases: str = "",
        persistent_symptoms: str = "",
        abnormal_labs_summary: str = "",
    ) -> Dict[str, Any]:
        """
        执行完整罕见病筛查分析流水线
        
        Args:
            records_text: 多条带时间顺序的病历自由文本
            family_history: 家族史描述
            excluded_diseases: 已排除的常见病列表
            persistent_symptoms: 持续性异常症状描述
            abnormal_labs_summary: 异常检验指标汇总
        
        Returns:
            {
                "success": bool,
                "report": str,          # Markdown格式完整报告
                "references": str,      # 参考文献列表文本
                "structured_data": {},  # 结构化病历数据
                "evidences": {},        # 检索到的循证资料
                "reasoning": {},        # 推理分析结果
                "logs": [],             # 执行日志
                "error": str            # 错误信息（仅失败时）
            }
        """
        logger.info("=" * 60)
        logger.info("[Pipeline] 开始罕见病筛查分析")
        logger.info("=" * 60)

        # ============================================================
        # 步骤1: LLM结构化解析病历
        # ============================================================
        self._report_progress("步骤1/5: 正在解析多条病历，提取结构化信息...", 0.05)
        try:
            structured_data = extract_medical_structure(records_text)
            if structured_data.get("error"):
                return self._fail(f"病历结构化失败: {structured_data['error']}")
            logger.info(f"[Pipeline] 步骤1完成, 抽取到 {len(structured_data)} 个字段")
        except Exception as e:
            logger.error(f"[Pipeline] 步骤1异常: {e}")
            return self._fail(f"步骤1异常: {e}")

        # 将补充信息合并入结构化数据
        if family_history:
            structured_data["family_history_extra"] = family_history
        if excluded_diseases:
            structured_data["excluded_diseases_extra"] = excluded_diseases
        if persistent_symptoms:
            structured_data["persistent_symptoms"] = persistent_symptoms
        if abnormal_labs_summary:
            structured_data["abnormal_labs_summary"] = abnormal_labs_summary

        self._report_progress("步骤1完成: 病历结构化解析完毕", 0.15)

        # ============================================================
        # 步骤2: 两阶段检索 —— 第一阶段：常见病排除
        # ============================================================
        self._report_progress("步骤2/5: 正在生成检索策略并检索循证文献...", 0.20)
        try:
            query_plan = generate_search_queries(structured_data)
            if query_plan.get("error"):
                logger.warning(f"[Pipeline] Query生成异常，使用默认策略: {query_plan['error']}")
                query_plan = self._fallback_queries(structured_data)
        except Exception as e:
            logger.warning(f"[Pipeline] Query生成异常，使用默认策略: {e}")
            query_plan = self._fallback_queries(structured_data)

        # 第一阶段：常见病排除检索
        common_queries_cn = query_plan.get("common_queries_cn", [])
        common_queries_en = query_plan.get("common_queries_en", [])

        self._report_progress(
            f"步骤2-A: 常见病排除检索 (中文{len(common_queries_cn)}组/英文{len(common_queries_en)}组)...",
            0.25
        )

        try:
            common_evidences = batch_search_evidences(common_queries_cn, common_queries_en, [], [])
            total_common = (
                len(common_evidences.get("papers_cn", []))
                + len(common_evidences.get("papers_en", []))
            )
            logger.info(f"[Pipeline] 常见病检索完成, 共{total_common}条")
        except Exception as e:
            logger.warning(f"[Pipeline] 常见病检索异常: {e}")
            common_evidences = {"papers_cn": [], "papers_en": [], "guides": [], "package_inserts": []}

        self._report_progress(f"步骤2-A完成: 常见病检索到{total_common}条文献", 0.35)

        # 第二阶段：罕见病定向检索
        rare_queries_cn = query_plan.get("rare_queries_cn", [])
        rare_queries_en = query_plan.get("rare_queries_en", [])
        queries_guide = query_plan.get("queries_guide", [])
        queries_drug = query_plan.get("queries_drug", [])

        self._report_progress(
            f"步骤2-B: 罕见病定向检索 (中文{len(rare_queries_cn)}组/英文{len(rare_queries_en)}组/"
            f"指南{len(queries_guide)}组/药品{len(queries_drug)}组)...",
            0.40
        )

        try:
            rare_evidences = batch_search_evidences(rare_queries_cn, rare_queries_en, queries_guide, queries_drug)
            total_rare = (
                len(rare_evidences.get("papers_cn", []))
                + len(rare_evidences.get("papers_en", []))
                + len(rare_evidences.get("guides", []))
                + len(rare_evidences.get("package_inserts", []))
            )
            logger.info(f"[Pipeline] 罕见病检索完成, 共{total_rare}条")
        except Exception as e:
            logger.warning(f"[Pipeline] 罕见病检索异常: {e}")
            rare_evidences = {"papers_cn": [], "papers_en": [], "guides": [], "package_inserts": []}

        # 合并全部证据（供报告参考文献使用）
        evidences = {
            "papers_cn": common_evidences.get("papers_cn", []) + rare_evidences.get("papers_cn", []),
            "papers_en": common_evidences.get("papers_en", []) + rare_evidences.get("papers_en", []),
            "guides": rare_evidences.get("guides", []),
            "package_inserts": rare_evidences.get("package_inserts", []),
        }
        total_evidence = total_common + total_rare

        self._report_progress(
            f"步骤2完成: 共检索到{total_evidence}条循证资料 (常见病{total_common}+罕见病{total_rare})", 0.50
        )

        # ============================================================
        # 步骤3: 综合推理分析
        # ============================================================
        self._report_progress("步骤3/5: 正在进行罕见病综合推理分析...", 0.55)

        # 构建补充上下文
        extra_parts = []
        if family_history:
            extra_parts.append(f"家族史: {family_history}")
        if excluded_diseases:
            extra_parts.append(f"已排除的常见病: {excluded_diseases}")
        if persistent_symptoms:
            extra_parts.append(f"持续性异常症状: {persistent_symptoms}")
        if abnormal_labs_summary:
            extra_parts.append(f"异常检验指标汇总: {abnormal_labs_summary}")
        extra_context = "\n".join(extra_parts) if extra_parts else "无额外补充信息"

        try:
            reasoning_result = rare_disease_reasoning(structured_data, common_evidences, rare_evidences, extra_context)
            if reasoning_result.get("error"):
                return self._fail(f"罕见病推理分析失败: {reasoning_result['error']}")
            logger.info("[Pipeline] 步骤3完成, 推理分析结束")
        except Exception as e:
            logger.error(f"[Pipeline] 步骤3异常: {e}")
            return self._fail(f"步骤3异常: {e}")

        self._report_progress("步骤3完成: 罕见病综合推理分析完毕", 0.70)

        # ============================================================
        # 步骤4: 格式化报告
        # ============================================================
        self._report_progress("步骤4/5: 正在生成排版清晰的筛查报告...", 0.75)
        try:
            report = format_screening_report(reasoning_result, structured_data, evidences)
            if report.startswith("[LLM"):
                return self._fail(f"报告生成失败: {report}")
            # 附加合规声明
            report = report + REPORT_FOOTER
            # 合规清洗
            report = sanitize_output(report)
            logger.info(f"[Pipeline] 步骤4完成, 报告长度={len(report)}")
        except Exception as e:
            logger.error(f"[Pipeline] 步骤4异常: {e}")
            return self._fail(f"步骤4异常: {e}")

        self._report_progress("步骤4完成: 筛查报告生成完毕", 0.90)

        # ============================================================
        # 步骤5: 整理参考文献列表
        # ============================================================
        self._report_progress("步骤5/5: 正在整理参考文献列表...", 0.92)
        references_text = self._build_references(evidences)
        self._report_progress("全部步骤完成!", 1.0)

        logger.info("[Pipeline] 全部步骤执行完毕")

        return {
            "success": True,
            "report": report,
            "references": references_text,
            "structured_data": structured_data,
            "evidences": evidences,
            "reasoning": reasoning_result,
            "logs": self._step_logs.copy(),
        }

    def _fail(self, error_msg: str) -> Dict[str, Any]:
        """构造失败返回结果"""
        logger.error(f"[Pipeline] 失败: {error_msg}")
        self._step_logs.append(f"[失败] {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "report": f"**分析失败**: {error_msg}\n\n{COMPLIANCE_DISCLAIMER}",
            "references": "",
            "structured_data": {},
            "evidences": {},
            "reasoning": {},
            "logs": self._step_logs.copy(),
        }

    def _fallback_queries(self, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        当LLM生成Query失败时，基于结构化数据构造默认检索策略
        """
        key_features = structured_data.get("key_clinical_features", [])
        symptoms = structured_data.get("symptoms", [])

        # 提取症状名称用于检索
        symptom_names = []
        if isinstance(symptoms, list):
            for s in symptoms[:5]:
                if isinstance(s, dict):
                    symptom_names.append(s.get("name", ""))
                elif isinstance(s, str):
                    symptom_names.append(s)

        feature_text = " ".join(key_features[:3]) if key_features else ""
        symptom_text = " ".join([s for s in symptom_names if s][:3])

        queries_cn = []
        queries_en = []

        if feature_text:
            queries_cn.append(f"罕见病 {feature_text}")
            queries_en.append(f"rare disease {feature_text}")
        if symptom_text:
            queries_cn.append(f"{symptom_text} 罕见病 鉴别诊断")
            queries_en.append(f"{symptom_text} rare disease differential diagnosis")
        if not queries_cn:
            queries_cn = ["罕见病 多系统受累 鉴别诊断"]
            queries_en = ["rare disease multisystem involvement differential diagnosis"]

        return {
            "queries_cn": queries_cn,
            "queries_en": queries_en,
            "common_queries_cn": queries_cn,
            "common_queries_en": queries_en,
            "rare_queries_cn": queries_cn,
            "rare_queries_en": queries_en,
            "queries_guide": ["罕见病诊疗指南"],
            "queries_drug": [],
            "suspected_diseases": [],
            "common_diseases_to_exclude": [],
        }

    def _build_references(self, evidences: Dict[str, List]) -> str:
        """构建参考文献列表Markdown文本"""
        parts = ["## 参考文献列表\n"]
        ref_num = 0

        sections = [
            ("papers_cn", "中文医学文献"),
            ("papers_en", "英文医学文献"),
            ("guides", "诊疗指南"),
            ("package_inserts", "药品说明书"),
        ]

        for key, label in sections:
            items = evidences.get(key, [])
            if items:
                parts.append(f"\n### {label}\n")
                for item in items[:20]:
                    ref_num += 1
                    if isinstance(item, dict):
                        title = item.get("title", item.get("name", str(item)))
                        source = item.get("source", item.get("journal", item.get("author", "")))
                        year = item.get("year", item.get("date", ""))
                        ref_line = f"{ref_num}. {title}"
                        if source:
                            ref_line += f". *{source}*"
                        if year:
                            ref_line += f" ({year})"
                        parts.append(ref_line)
                    else:
                        parts.append(f"{ref_num}. {str(item)[:200]}")

        if ref_num == 0:
            parts.append("（本次分析未检索到相关循证文献）")

        return "\n".join(parts)

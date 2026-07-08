"""
llm_engine.py - 大模型调用封装
基于StepFun（阶跃星辰）OpenAI兼容API，封装三套专用Prompt：
  1. 病历结构化抽取 Prompt
  2. 检索Query生成 Prompt
  3. 罕见病汇总推理 Prompt
所有调用均含超时、重试、异常捕获
"""

import json
import time
import logging
import requests
from typing import Dict, Any, List, Optional

from config import (
    LLM_API_KEY,
    LLM_CHAT_ENDPOINT,
    LLM_MODEL,
    LLM_TIMEOUT,
    LLM_MAX_RETRIES,
    LLM_RETRY_DELAY,
    FORBIDDEN_PHRASES,
    REQUIRED_PHRASE,
)
from api_logger import record_api_call

logger = logging.getLogger(__name__)


# ============================================================
# 通用大模型调用函数
# ============================================================
def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = LLM_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 8192,
    task_label: str = "大模型调用",
) -> str:
    """
    调用StepFun大模型（OpenAI兼容格式）
    
    Args:
        system_prompt: 系统提示词
        user_prompt: 用户输入
        model: 模型名称
        temperature: 采样温度
        max_tokens: 最大输出token数
    
    Returns:
        模型输出文本，失败时返回错误描述
    """
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            logger.info(f"[LLM] 调用 {model} (attempt {attempt}/{LLM_MAX_RETRIES}), "
                        f"user_prompt长度={len(user_prompt)}")
            resp = requests.post(
                LLM_CHAT_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=LLM_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            logger.info(f"[LLM] 成功, 输出长度={len(content)}")
            # 记录API调用日志
            record_api_call(
                task_label, LLM_CHAT_ENDPOINT, "POST",
                request_payload={"model": model, "system_prompt": system_prompt[:200] + "...", "user_prompt": user_prompt[:500] + "..."},
                response_data={"content": content[:1000] + ("..." if len(content) > 1000 else ""), "length": len(content)},
                status="success",
            )
            return content

        except requests.exceptions.Timeout:
            logger.warning(f"[LLM] 超时 (attempt {attempt}/{LLM_MAX_RETRIES})")
            if attempt == LLM_MAX_RETRIES:
                record_api_call(task_label, LLM_CHAT_ENDPOINT, "POST", request_payload={"model": model}, status="error", error_msg="LLM调用超时")
                return f"[LLM调用超时] 经过{LLM_MAX_RETRIES}次重试仍然超时"
            time.sleep(LLM_RETRY_DELAY)

        except requests.exceptions.HTTPError as e:
            logger.error(f"[LLM] HTTP错误: {e} (attempt {attempt}/{LLM_MAX_RETRIES})")
            # 尝试打印响应体便于调试
            try:
                logger.error(f"[LLM] 响应体: {resp.text[:500]}")
            except:
                pass
            if attempt == LLM_MAX_RETRIES:
                record_api_call(task_label, LLM_CHAT_ENDPOINT, "POST", request_payload={"model": model}, status="error", error_msg=str(e))
                return f"[LLM调用失败] HTTP错误: {e}"
            time.sleep(LLM_RETRY_DELAY)

        except (KeyError, IndexError) as e:
            logger.error(f"[LLM] 解析响应失败: {e}")
            try:
                logger.error(f"[LLM] 原始响应: {json.dumps(data, ensure_ascii=False)[:500]}")
            except:
                pass
            return f"[LLM解析失败] 响应格式异常: {e}"

        except Exception as e:
            logger.error(f"[LLM] 未知异常: {e}")
            return f"[LLM异常] {e}"

    return "[LLM调用失败] 所有重试均失败"


def _parse_json_response(text: str) -> Dict[str, Any]:
    """
    从LLM输出中提取JSON对象
    支持 ```json ... ``` 包裹或纯JSON文本
    
    Args:
        text: LLM输出文本
    
    Returns:
        解析后的字典，失败时返回包含原始文本的字典
    """
    # 尝试提取 ```json ... ``` 块
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            json_str = text[start:end].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            json_str = text[start:end].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

    # 尝试直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        logger.warning(f"[JSON解析失败] 返回原始文本, 长度={len(text)}")
        return {"raw_text": text, "parse_error": True}


def sanitize_output(text: str) -> str:
    """
    合规性输出清洗：替换禁止出现的确定性诊断措辞
    
    Args:
        text: 原始输出文本
    
    Returns:
        清洗后的文本
    """
    result = text
    for phrase in FORBIDDEN_PHRASES:
        if phrase in result:
            result = result.replace(phrase, REQUIRED_PHRASE)
    return result


# ============================================================
# Prompt 1: 病历结构化抽取
# ============================================================
PROMPT_EXTRACT_STRUCTURE = """你是一位资深临床医学信息学专家。你的任务是对多条时序病历进行结构化信息抽取。

请严格按以下JSON格式输出，不要添加任何额外解释文字：
```json
{
  "timeline": [
    {"date": "时间", "event": "就诊事件描述", "key_findings": ["关键发现"]}
  ],
  "symptoms": [
    {"name": "症状名称", "onset": "出现时间", "duration": "持续时间", "severity": "严重程度", "progression": "进展趋势"}
  ],
  "abnormal_labs": [
    {"name": "检验项目", "value": "检测值", "reference": "参考范围", "trend": "变化趋势（升高/降低/波动）", "dates": ["检测日期"]}
  ],
  "imaging_findings": [
    {"date": "检查日期", "modality": "检查方式", "findings": "影像所见"}
  ],
  "medications": [
    {"name": "药物名称", "dosage": "剂量", "start_date": "开始日期", "indication": "适应症"}
  ],
  "family_history": "家族史描述",
  "past_history": "既往病史描述",
  "excluded_diseases": ["已排除的疾病列表"],
  "persistent_abnormalities": ["持续性异常表现汇总"],
  "key_clinical_features": ["核心临床特征汇总，用于后续罕见病匹配"]
}
```

要求：
1. 按时间顺序梳理所有就诊事件，不遗漏任何病历信息
2. 症状需标注出现时间、持续时长、严重程度及变化趋势
3. 异常检验指标需标注具体数值、参考范围和变化趋势
4. 提取所有影像检查发现
5. 汇总核心临床特征，便于后续罕见病匹配
6. 如果某字段在病历中未提及，填写"未提及"
"""


def extract_medical_structure(records_text: str) -> Dict[str, Any]:
    """
    步骤1：调用大模型对多条病历进行结构化信息抽取
    
    Args:
        records_text: 多条病历自由文本（含时间标记）
    
    Returns:
        结构化病历信息字典
    """
    user_input = f"请对以下多条时序病历进行结构化信息抽取：\n\n{records_text}"
    raw_output = call_llm(PROMPT_EXTRACT_STRUCTURE, user_input, temperature=0.2, max_tokens=8192, task_label="LLM-病历结构化抽取")
    
    if raw_output.startswith("[LLM"):
        # 调用失败，返回错误信息
        return {"error": raw_output}
    
    return _parse_json_response(raw_output)


# ============================================================
# Prompt 2: 检索Query生成（两阶段：先常见病排除 → 再罕见病定向）
# ============================================================
PROMPT_GENERATE_QUERIES = """你是一位精通罕见病诊断的临床医学研究员。你的任务是基于结构化病历信息，设计**两阶段**文献检索策略。

## 核心原则
- 所有检索Query**严禁包含任何罕见病病名**，只使用症状、体征、异常指标的组合
- 第一阶段：用症状组合检索常见病文献，目的是**排除常见病**
- 第二阶段：用更精准的症状组合检索罕见病文献，目的是**发现罕见病线索**
- 诊疗指南和药品说明书仅针对第二阶段发现的疑似罕见病方向

## 输出格式
请严格按以下JSON格式输出：
```json
{
  "common_queries_cn": [
    "症状组合1（如：蛋白尿 水肿 高血压 肾功能减退）",
    "症状组合2（不同器官系统组合）",
    "症状组合3"
  ],
  "common_queries_en": [
    "English MeSH terms for common disease search 1",
    "English MeSH terms for common disease search 2",
    "English MeSH terms for common disease search 3"
  ],
  "rare_queries_cn": [
    "更精准的症状/体征/指标组合1（侧重罕见病特征）",
    "多系统受累特征组合2",
    "特殊体征组合3"
  ],
  "rare_queries_en": [
    "Precise symptom combination in English MeSH 1",
    "Multisystem feature combination 2",
    "Distinctive clinical sign combination 3"
  ],
  "queries_guide": [
    "疑似罕见病方向1 诊疗指南/专家共识",
    "疑似罕见病方向2 诊疗指南"
  ],
  "queries_drug": [
    "疑似罕见病方向1 特效药物/酶替代治疗药物",
    "疑似罕见病方向2 治疗药物"
  ],
  "suspected_diseases": [
    {"name": "疑似罕见病名", "name_en": "English name", "reason": "基于症状匹配的怀疑理由"}
  ],
  "common_diseases_to_exclude": [
    {"name": "常见病名", "reason": "该常见病需要被排除的理由"}
  ]
}
```

## 生成策略详解

### 第一阶段：常见病排除检索
- common_queries_cn/en：每组由2-4个核心症状/体征组合，**不含任何病名**
- 目标：检索到该症状组合下最常见的疾病文献，用于后续排除
- 至少生成3组不同器官系统维度的组合
- common_diseases_to_exclude：预判3-5个该症状组合下最可能的常见病

### 第二阶段：罕见病定向检索
- rare_queries_cn/en：在第一阶段基础上，加入更精准的特征组合（如特殊体征、罕见指标异常、多系统交叉受累模式）
- 仍然**不含病名**，但组合更特异、更指向罕见病特征
- 至少生成3组不同鉴别方向

### 指南与药品
- queries_guide：仅当第二阶段有疑似罕见病方向时才生成，针对病名+指南
- queries_drug：针对疑似罕见病的特效药/酶替代治疗药物

### 注意事项
- 所有检索词使用标准医学术语(MeSH)，精准专业
- 中英文query一一对应
- 每组query聚焦不同鉴别方向，避免重复
- 疑似罕见病3-5个，常见病排除3-5个
"""


def generate_search_queries(structured_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    步骤2a：基于结构化病历，由大模型生成两阶段检索Query
    第一阶段：常见病排除检索（症状组合，不含病名）
    第二阶段：罕见病定向检索（更精准的特异症状组合）
    
    Args:
        structured_data: 步骤1输出的结构化病历字典
    
    Returns:
        包含两阶段检索query、疑似疾病、待排除常见病的字典
    """
    data_str = json.dumps(structured_data, ensure_ascii=False, indent=2)
    user_input = f"请基于以下结构化病历信息生成两阶段检索Query：\n\n{data_str}"
    raw_output = call_llm(PROMPT_GENERATE_QUERIES, user_input, temperature=0.3, max_tokens=4096, task_label="LLM-检索Query生成")
    
    if raw_output.startswith("[LLM"):
        return {
            "error": raw_output,
            "common_queries_cn": [], "common_queries_en": [],
            "rare_queries_cn": [], "rare_queries_en": [],
            "queries_guide": [], "queries_drug": [],
            "suspected_diseases": [], "common_diseases_to_exclude": [],
        }
    
    result = _parse_json_response(raw_output)
    
    # 确保所有必要字段存在
    for key in [
        "common_queries_cn", "common_queries_en",
        "rare_queries_cn", "rare_queries_en",
        "queries_guide", "queries_drug",
        "suspected_diseases", "common_diseases_to_exclude",
    ]:
        if key not in result:
            result[key] = []
    
    return result


# ============================================================
# Prompt 3: 罕见病汇总推理
# ============================================================
PROMPT_RARE_DISEASE_REASONING = """你是一位资深罕见病临床专家与医学遗传学家。你将收到以下信息：
1. 结构化病历摘要
2. 常见病排除检索结果（第一阶段）
3. 罕见病定向检索结果（第二阶段）

请基于全部信息进行**两阶段递进式**罕见病综合推理分析。

## 分析逻辑

### 第一步：常见病排除
- 根据第一阶段常见病检索结果，逐一评估该患者症状是否可被常见病解释
- 对每个常见病明确判断：可排除 / 不能完全排除（说明理由）
- 已被常见病充分解释的症状不再纳入罕见病分析

### 第二步：罕见病定向筛查
- 仅针对**不能被常见病解释的剩余症状/体征/指标异常**
- 结合第二阶段罕见病定向检索的文献证据
- 从证据出发，匹配可能的罕见病方向

### 第三步：三级分层
- 将候选罕见病分为：高度可疑 / 中度可疑 / 低度可疑

【强制措辞约束】
- 严禁使用"确诊""患病""就是XX病"等确定性诊断表述
- 统一使用"该罕见病为重点排查可疑方向"
- 所有结论均为排查参考，不是诊断结论

请严格按以下JSON格式输出分析报告：
```json
{
  "common_disease_exclusion": [
    {
      "disease": "常见病名称",
      "assessment": "可排除 / 不能完全排除",
      "reason": "判断理由（结合文献证据）"
    }
  ],
  "unexplained_features": [
    "不能被常见病解释的症状/体征/指标异常列表"
  ],
  "high_suspicion": [
    {
      "disease": "罕见病名称",
      "disease_en": "英文名",
      "matching_evidence": [
        {"feature": "匹配的临床特征", "support": "支撑依据（标注来源：文献/指南/药品说明书）"}
      ],
      "doubts": ["现存疑点或不符合之处"],
      "recommendations": ["建议的进一步检查/检测"]
    }
  ],
  "moderate_suspicion": [
    {
      "disease": "罕见病名称",
      "disease_en": "英文名",
      "matching_evidence": [
        {"feature": "匹配的临床特征", "support": "支撑依据（标注来源）"}
      ],
      "doubts": ["现存疑点"],
      "recommendations": ["建议检查"]
    }
  ],
  "low_suspicion": [
    {
      "disease": "罕见病名称",
      "disease_en": "英文名",
      "matching_evidence": [
        {"feature": "匹配的临床特征", "support": "支撑依据"}
      ],
      "doubts": ["现存疑点"],
      "recommendations": ["建议检查"]
    }
  ],
  "next_steps": {
    "specialist_referral": ["建议就诊专科"],
    "genetic_testing": ["建议基因检测项目"],
    "targeted_tests": ["建议的针对性检查"],
    "other": ["其他建议"]
  },
  "summary": "总体排查思路总结（含常见病排除结论 + 罕见病方向，300字以内）"
}
```

分析要求：
1. 常见病排除要客观严谨，有文献支撑
2. 仅对无法用常见病解释的特征进行罕见病匹配
3. 高度可疑：至少3个核心特征匹配，且有文献/指南支撑
4. 中度可疑：2个核心特征匹配，需进一步验证
5. 低度可疑：仅部分特征提示，作为鉴别诊断参考
6. 每条支撑依据必须标注循证来源（中文文献/英文文献/诊疗指南/药品说明书）
7. 疑点要客观指出不符合之处
8. 下一步建议要具体到科室、检测项目、检查名称
"""


def rare_disease_reasoning(
    structured_data: Dict[str, Any],
    common_evidences: Dict[str, List],
    rare_evidences: Dict[str, List],
    extra_context: str = "",
) -> Dict[str, Any]:
    """
    步骤3：两阶段递进式综合推理分析
    先排除常见病，再定向筛查罕见病
    
    Args:
        structured_data: 结构化病历数据
        common_evidences: 第一阶段常见病排除检索结果
        rare_evidences: 第二阶段罕见病定向检索结果
        extra_context: 补充上下文（家族史、已排除疾病等）
    
    Returns:
        罕见病分析结果字典（常见病排除 + 三级可疑等级 + 建议）
    """
    common_text = _format_evidences(common_evidences, label_prefix="[常见病]")
    rare_text = _format_evidences(rare_evidences, label_prefix="[罕见病]")
    data_str = json.dumps(structured_data, ensure_ascii=False, indent=2)
    
    user_input = (
        f"## 结构化病历摘要\n{data_str}\n\n"
        f"## 补充信息\n{extra_context}\n\n"
        f"## 第一阶段：常见病排除检索结果\n{common_text}\n\n"
        f"## 第二阶段：罕见病定向检索结果\n{rare_text}\n\n"
        f"请先排除常见病，再基于剩余未解释特征进行罕见病定向筛查。"
    )
    
    raw_output = call_llm(PROMPT_RARE_DISEASE_REASONING, user_input, temperature=0.3, max_tokens=8192, task_label="LLM-罕见病综合推理")
    
    if raw_output.startswith("[LLM"):
        return {"error": raw_output}
    
    result = _parse_json_response(raw_output)
    return result


# ============================================================
# Prompt 4: 报告格式化
# ============================================================
PROMPT_FORMAT_REPORT = """你是一位专业的医学报告撰写专家。请将以下罕见病筛查分析结果整理为一份排版清晰、专业规范的Markdown格式筛查报告。

格式要求：
1. 标题：多病历纵向罕见病筛查分析报告
2. 第一节：病程时间线概述（用表格或有序列表展示）
3. 第二节：核心临床特征汇总
4. 第三节：常见病排除分析
   - 列出每个被评估的常见病及排除/不排除的判断
   - 说明判断依据
5. 第四节：未被常见病解释的临床特征
   - 列出无法用常见病解释的症状/体征/指标异常
6. 第五节：罕见病排查分析
   - 高度可疑（用红色标记/加粗）
   - 中度可疑（用橙色标记）
   - 低度可疑（用普通标记）
   - 每个疾病列出：名称、匹配证据、现存疑点、建议检查
7. 第六节：下一步建议（专科就诊、基因检测、针对性检查）
8. 第七节：参考文献列表（列出所有引用的文献标题和来源）
9. 末尾：合规声明

【强制措辞约束】
- 严禁使用"确诊""患病""就是XX病"等确定性诊断表述
- 统一使用"该罕见病为重点排查可疑方向"
- 全文语气为"排查参考"而非"诊断结论"

请直接输出Markdown格式报告全文。
"""


def format_screening_report(
    reasoning_result: Dict[str, Any],
    structured_data: Dict[str, Any],
    evidences: Dict[str, List],
) -> str:
    """
    步骤4：调用大模型整理排版清晰的完整筛查报告
    
    Args:
        reasoning_result: 步骤3的推理分析结果
        structured_data: 结构化病历数据
        evidences: 循证资料
    
    Returns:
        Markdown格式的完整筛查报告文本
    """
    reasoning_str = json.dumps(reasoning_result, ensure_ascii=False, indent=2)
    data_str = json.dumps(structured_data, ensure_ascii=False, indent=2)
    evidence_summary = _format_evidences_brief(evidences)
    
    user_input = (
        f"## 分析结果\n{reasoning_str}\n\n"
        f"## 结构化病历\n{data_str}\n\n"
        f"## 参考文献汇总\n{evidence_summary}\n\n"
        f"请整理为完整的Markdown格式筛查报告。"
    )
    
    report = call_llm(PROMPT_FORMAT_REPORT, user_input, temperature=0.3, max_tokens=8192, task_label="LLM-报告格式化")
    
    # 合规清洗
    report = sanitize_output(report)
    
    return report


# ============================================================
# 辅助函数：格式化证据资料
# ============================================================
def _format_evidences(evidences: Dict[str, List], label_prefix: str = "") -> str:
    """将证据列表格式化为文本摘要，供大模型阅读
    Args:
        evidences: 证据字典
        label_prefix: 标签前缀（如"[常见病]"或"[罕见病]"）
    """
    parts = []
    
    # 中文文献
    if evidences.get("papers_cn"):
        parts.append(f"### {label_prefix}中文医学文献")
        for i, paper in enumerate(evidences["papers_cn"][:30], 1):
            if isinstance(paper, dict):
                title = paper.get("title", paper.get("name", str(paper)))
                snippet = paper.get("snippet", paper.get("abstract", paper.get("content", "")))
                parts.append(f"{i}. {title}")
                if snippet:
                    parts.append(f"   摘要: {str(snippet)[:300]}")
            else:
                parts.append(f"{i}. {str(paper)[:300]}")
    
    # 英文文献
    if evidences.get("papers_en"):
        parts.append(f"\n### {label_prefix}英文医学文献")
        for i, paper in enumerate(evidences["papers_en"][:30], 1):
            if isinstance(paper, dict):
                title = paper.get("title", paper.get("name", str(paper)))
                snippet = paper.get("snippet", paper.get("abstract", paper.get("content", "")))
                parts.append(f"{i}. {title}")
                if snippet:
                    parts.append(f"   Abstract: {str(snippet)[:300]}")
            else:
                parts.append(f"{i}. {str(paper)[:300]}")
    
    # 诊疗指南
    if evidences.get("guides"):
        parts.append(f"\n### {label_prefix}诊疗指南")
        for i, guide in enumerate(evidences["guides"][:5], 1):
            if isinstance(guide, dict):
                title = guide.get("title", guide.get("name", str(guide)))
                content = guide.get("content", guide.get("snippet", ""))
                parts.append(f"{i}. {title}")
                if content:
                    parts.append(f"   内容: {str(content)[:500]}")
            else:
                parts.append(f"{i}. {str(guide)[:500]}")
    
    # 药品说明书
    if evidences.get("package_inserts"):
        parts.append(f"\n### {label_prefix}药品说明书")
        for i, drug in enumerate(evidences["package_inserts"][:5], 1):
            if isinstance(drug, dict):
                title = drug.get("title", drug.get("name", drug.get("drug_name", str(drug))))
                content = drug.get("content", drug.get("snippet", ""))
                parts.append(f"{i}. {title}")
                if content:
                    parts.append(f"   内容: {str(content)[:500]}")
            else:
                parts.append(f"{i}. {str(drug)[:500]}")
    
    return "\n".join(parts) if parts else "（未检索到循证资料）"


def _format_evidences_brief(evidences: Dict[str, List]) -> str:
    """简短格式化证据列表，用于报告参考文献部分"""
    parts = []
    
    sections = [
        ("papers_cn", "中文医学文献"),
        ("papers_en", "英文医学文献"),
        ("guides", "诊疗指南"),
        ("package_inserts", "药品说明书"),
    ]
    
    for key, label in sections:
        items = evidences.get(key, [])
        if items:
            parts.append(f"### {label}")
            for i, item in enumerate(items[:20], 1):
                if isinstance(item, dict):
                    title = item.get("title", item.get("name", str(item)))
                    source = item.get("source", item.get("journal", ""))
                    parts.append(f"{i}. {title}" + (f" [{source}]" if source else ""))
                else:
                    parts.append(f"{i}. {str(item)[:200]}")
            parts.append("")
    
    return "\n".join(parts) if parts else "（无引用文献）"

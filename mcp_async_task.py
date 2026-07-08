"""
mcp_async_task.py - MCP异步任务管理
实现MCP扩展工具设计规范的异步任务模式：
  - 提交分析任务获得jobId
  - 后台线程串行执行完整Pipeline
  - 轮询jobId查询任务状态与最终结果
  - 内存存储，不持久化，用完即销毁
同时支持桥接KNOWS平台的/jobs接口
"""

import uuid
import time
import threading
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from rare_analysis_pipeline import RareDiseasePipeline
from knows_api import create_job, get_job_status

logger = logging.getLogger(__name__)


# ============================================================
# 任务状态常量
# ============================================================
STATUS_PENDING = "pending"       # 已提交，等待执行
STATUS_RUNNING = "running"       # 正在执行中
STATUS_COMPLETED = "completed"   # 执行完成
STATUS_FAILED = "failed"         # 执行失败


# ============================================================
# 任务存储（内存字典，进程级，不持久化）
# ============================================================
_task_store: Dict[str, Dict[str, Any]] = {}
_store_lock = threading.Lock()


def _generate_job_id() -> str:
    """生成唯一任务ID"""
    return f"rare-screen-{uuid.uuid4().hex[:12]}"


# ============================================================
# 核心：提交分析任务
# ============================================================
def submit_analysis(
    records_text: str,
    family_history: str = "",
    excluded_diseases: str = "",
    persistent_symptoms: str = "",
    abnormal_labs_summary: str = "",
) -> str:
    """
    提交罕见病筛查分析任务（异步）
    创建后台线程执行Pipeline，立即返回job_id
    
    Args:
        records_text: 多条时序病历文本
        family_history: 家族史
        excluded_diseases: 已排除疾病
        persistent_symptoms: 持续性异常症状
        abnormal_labs_summary: 异常检验指标汇总
    
    Returns:
        job_id: 任务唯一标识，用于后续轮询查询
    """
    job_id = _generate_job_id()

    # 初始化任务记录
    task_record = {
        "job_id": job_id,
        "status": STATUS_PENDING,
        "progress": 0.0,
        "progress_message": "任务已提交，等待执行...",
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "result": None,         # Pipeline完整返回结果
        "report": "",           # 最终报告文本
        "references": "",       # 参考文献列表
        "error": None,          # 错误信息
        "logs": [],             # 执行日志
    }

    with _store_lock:
        _task_store[job_id] = task_record

    # 启动后台线程执行Pipeline
    thread = threading.Thread(
        target=_execute_pipeline,
        args=(
            job_id,
            records_text,
            family_history,
            excluded_diseases,
            persistent_symptoms,
            abnormal_labs_summary,
        ),
        daemon=True,
        name=f"pipeline-{job_id}",
    )
    thread.start()

    logger.info(f"[MCP Task] 任务已提交: {job_id}")
    return job_id


def _execute_pipeline(
    job_id: str,
    records_text: str,
    family_history: str,
    excluded_diseases: str,
    persistent_symptoms: str,
    abnormal_labs_summary: str,
):
    """
    后台线程：执行完整Pipeline并更新任务状态
    
    Args:
        job_id: 任务ID
        records_text: 病历文本
        family_history: 家族史
        excluded_diseases: 已排除疾病
        persistent_symptoms: 持续性异常症状
        abnormal_labs_summary: 异常指标汇总
    """
    # 更新状态为运行中
    with _store_lock:
        if job_id in _task_store:
            _task_store[job_id]["status"] = STATUS_RUNNING
            _task_store[job_id]["started_at"] = datetime.now().isoformat()

    def progress_callback(message: str, progress: float):
        """Pipeline进度回调，更新任务记录"""
        with _store_lock:
            if job_id in _task_store:
                _task_store[job_id]["progress"] = progress
                _task_store[job_id]["progress_message"] = message

    try:
        # 创建并执行Pipeline
        pipeline = RareDiseasePipeline(progress_callback=progress_callback)
        result = pipeline.run(
            records_text=records_text,
            family_history=family_history,
            excluded_diseases=excluded_diseases,
            persistent_symptoms=persistent_symptoms,
            abnormal_labs_summary=abnormal_labs_summary,
        )

        # 更新任务结果
        with _store_lock:
            if job_id in _task_store:
                if result.get("success"):
                    _task_store[job_id]["status"] = STATUS_COMPLETED
                    _task_store[job_id]["progress"] = 1.0
                    _task_store[job_id]["progress_message"] = "分析完成!"
                    _task_store[job_id]["report"] = result.get("report", "")
                    _task_store[job_id]["references"] = result.get("references", "")
                else:
                    _task_store[job_id]["status"] = STATUS_FAILED
                    _task_store[job_id]["error"] = result.get("error", "未知错误")
                _task_store[job_id]["result"] = result
                _task_store[job_id]["logs"] = result.get("logs", [])
                _task_store[job_id]["completed_at"] = datetime.now().isoformat()

        logger.info(f"[MCP Task] 任务完成: {job_id}, 状态={result.get('success')}")

    except Exception as e:
        logger.error(f"[MCP Task] 任务异常: {job_id}, error={e}")
        with _store_lock:
            if job_id in _task_store:
                _task_store[job_id]["status"] = STATUS_FAILED
                _task_store[job_id]["error"] = str(e)
                _task_store[job_id]["completed_at"] = datetime.now().isoformat()


# ============================================================
# 查询任务状态与结果
# ============================================================
def query_result(job_id: str) -> Dict[str, Any]:
    """
    查询异步任务状态与结果
    
    Args:
        job_id: 任务ID
    
    Returns:
        {
            "job_id": str,
            "status": "pending|running|completed|failed",
            "progress": float,          # 0.0~1.0
            "progress_message": str,    # 当前步骤描述
            "report": str,              # 完成时的完整报告
            "references": str,          # 参考文献列表
            "error": str|None,          # 错误信息
            "created_at": str,
            "completed_at": str|None,
        }
    """
    with _store_lock:
        task = _task_store.get(job_id)

    if task is None:
        return {
            "job_id": job_id,
            "status": STATUS_FAILED,
            "progress": 0.0,
            "progress_message": "",
            "report": "",
            "references": "",
            "error": f"任务不存在: {job_id}",
            "created_at": "",
            "completed_at": None,
        }

    return {
        "job_id": task["job_id"],
        "status": task["status"],
        "progress": task["progress"],
        "progress_message": task["progress_message"],
        "report": task.get("report", ""),
        "references": task.get("references", ""),
        "error": task.get("error"),
        "created_at": task["created_at"],
        "completed_at": task.get("completed_at"),
    }


def get_all_tasks() -> list:
    """
    获取所有任务列表（用于调试/管理）
    
    Returns:
        任务简要信息列表
    """
    with _store_lock:
        tasks = []
        for job_id, task in _task_store.items():
            tasks.append({
                "job_id": job_id,
                "status": task["status"],
                "progress": task["progress"],
                "progress_message": task["progress_message"],
                "created_at": task["created_at"],
            })
    return tasks


def cleanup_old_tasks(max_age_hours: int = 24):
    """
    清理超过指定时间的已完成/失败任务（释放内存）
    
    Args:
        max_age_hours: 最大保留小时数
    """
    cutoff = time.time() - max_age_hours * 3600
    to_remove = []

    with _store_lock:
        for job_id, task in _task_store.items():
            if task["status"] in (STATUS_COMPLETED, STATUS_FAILED):
                created = task.get("completed_at") or task.get("created_at", "")
                try:
                    created_ts = datetime.fromisoformat(created).timestamp()
                    if created_ts < cutoff:
                        to_remove.append(job_id)
                except (ValueError, TypeError):
                    pass

        for job_id in to_remove:
            del _task_store[job_id]

    if to_remove:
        logger.info(f"[MCP Task] 清理了 {len(to_remove)} 个过期任务")


# ============================================================
# KNOWS平台异步任务桥接
# ============================================================
def submit_knows_job(
    job_type: str = "rare_disease_screening",
    requested_by: str = "mcp_rare_screen_tool",
    input_data: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    向KNOWS平台提交异步任务（桥接外部MCP任务系统）
    
    Args:
        job_type: 任务类型
        requested_by: 请求方标识
        input_data: 任务输入
    
    Returns:
        KNOWS平台返回的任务信息（含jobId）
    """
    if input_data is None:
        input_data = {}
    return create_job(job_type, requested_by, input_data)


def query_knows_job(job_id: str) -> Dict[str, Any]:
    """
    查询KNOWS平台异步任务状态
    
    Args:
        job_id: KNOWS平台的jobId
    
    Returns:
        任务状态与结果
    """
    return get_job_status(job_id)

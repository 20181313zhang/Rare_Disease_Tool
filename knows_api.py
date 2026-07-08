"""
knows_api.py - KNOWS官方API接口封装
封装所有KNOWS平台提供的HTTP接口：
  - 异步任务创建与查询（POST /jobs, GET /jobs/{jobId}）
  - 证据检索：中文文献、英文文献、诊疗指南、药品说明书
统一处理认证头、超时、异常捕获与重试逻辑
"""

import time
import logging
import requests
from typing import Dict, Any, Optional, List

from config import (
    KNOWS_API_KEY,
    KNOWS_JOBS_CREATE,
    KNOWS_JOBS_QUERY,
    KNOWS_EVIDENCE_PAPER_CN,
    KNOWS_EVIDENCE_PAPER_EN,
    KNOWS_EVIDENCE_GUIDE,
    KNOWS_EVIDENCE_PACKAGE_INSERT,
    HTTP_TIMEOUT,
    EVIDENCE_TIMEOUT,
)
from api_logger import record_api_call

# ============================================================
# 日志配置
# ============================================================
logger = logging.getLogger(__name__)

# ============================================================
# URL转可读标签
# ============================================================
def _url_to_label(url: str) -> str:
    """将URL转为可读的API名称标签"""
    mapping = {
        "ai_search_paper_cn": "中文医学文献检索",
        "ai_search_paper_en": "英文医学文献检索",
        "ai_search_guide": "诊疗指南检索",
        "ai_search_package_insert": "药品说明书检索",
        "/jobs": "KNOWS异步任务",
    }
    for key, label in mapping.items():
        if key in url:
            return label
    return f"KNOWS API ({url.split('/')[-1]})"


# ============================================================
# 通用请求头
# ============================================================
def _get_headers() -> Dict[str, str]:
    """构造KNOWS API认证请求头"""
    return {
        "Authorization": f"Bearer {KNOWS_API_KEY}",
        "Content-Type": "application/json",
    }


def _safe_request(
    method: str,
    url: str,
    payload: Optional[Dict] = None,
    timeout: int = HTTP_TIMEOUT,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    统一HTTP请求封装，含超时、异常捕获、自动重试
    
    Args:
        method: HTTP方法 (GET/POST)
        url: 请求地址
        payload: POST请求体
        timeout: 超时秒数
        max_retries: 最大重试次数
    
    Returns:
        响应JSON字典，失败时返回 {"error": "错误描述"}
    """
    headers = _get_headers()
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[KNOWS API] {method} {url} (attempt {attempt}/{max_retries})")
            if method.upper() == "GET":
                resp = requests.get(url, headers=headers, timeout=timeout)
            else:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            
            resp.raise_for_status()
            result = resp.json()
            logger.info(f"[KNOWS API] 成功, status={resp.status_code}")
            # 记录API调用日志
            api_label = _url_to_label(url)
            record_api_call(api_label, url, method, request_payload=payload, response_data=result, status="success")
            return result

        except requests.exceptions.Timeout:
            logger.warning(f"[KNOWS API] 请求超时 (attempt {attempt}/{max_retries})")
            if attempt == max_retries:
                record_api_call(_url_to_label(url), url, method, request_payload=payload, status="error", error_msg="请求超时")
                return {"error": f"请求超时: {url}"}
            time.sleep(3)

        except requests.exceptions.HTTPError as e:
            logger.error(f"[KNOWS API] HTTP错误: {e} (attempt {attempt}/{max_retries})")
            if attempt == max_retries:
                record_api_call(_url_to_label(url), url, method, request_payload=payload, status="error", error_msg=str(e))
                return {"error": f"HTTP错误: {e}"}
            time.sleep(3)

        except requests.exceptions.RequestException as e:
            logger.error(f"[KNOWS API] 网络异常: {e} (attempt {attempt}/{max_retries})")
            if attempt == max_retries:
                record_api_call(_url_to_label(url), url, method, request_payload=payload, status="error", error_msg=str(e))
                return {"error": f"网络异常: {e}"}
            time.sleep(3)

        except Exception as e:
            logger.error(f"[KNOWS API] 未知异常: {e}")
            record_api_call(_url_to_label(url), url, method, request_payload=payload, status="error", error_msg=str(e))
            return {"error": f"未知异常: {e}"}

    return {"error": "所有重试均失败"}


# ============================================================
# 异步任务接口
# ============================================================
def create_job(job_type: str, requested_by: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建异步任务，返回jobId
    
    Args:
        job_type: 任务类型标识
        requested_by: 请求方标识
        input_data: 任务输入数据
    
    Returns:
        包含jobId的响应字典
    """
    payload = {
        "jobType": job_type,
        "requestedBy": requested_by,
        "input": input_data,
    }
    return _safe_request("POST", KNOWS_JOBS_CREATE, payload=payload)


def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    查询异步任务状态与结果
    
    Args:
        job_id: 任务ID
    
    Returns:
        任务状态与结果字典
    """
    url = f"{KNOWS_JOBS_QUERY}/{job_id}"
    return _safe_request("GET", url)


# ============================================================
# 证据检索接口
# ============================================================
def search_paper_cn(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    检索中文医学文献（最多40条）
    
    Args:
        query: 检索关键词
        top_k: 返回条数上限（最大40）
    
    Returns:
        文献列表，失败时返回空列表
    """
    payload = {"query": query, "top_k": min(top_k, 40)}
    result = _safe_request("POST", KNOWS_EVIDENCE_PAPER_CN, payload=payload, timeout=EVIDENCE_TIMEOUT)
    if "error" in result:
        logger.warning(f"[中文文献检索失败] query={query}, error={result['error']}")
        return []
    # 适配返回结构：可能是列表或字典中包含列表
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        # 尝试常见字段名
        for key in ["data", "results", "items", "records", "evidences"]:
            if key in result and isinstance(result[key], list):
                return result[key]
    return [result] if result else []


def search_paper_en(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    检索英文医学文献（最多40条）
    
    Args:
        query: 检索关键词（英文）
        top_k: 返回条数上限（最大40）
    
    Returns:
        文献列表，失败时返回空列表
    """
    payload = {"query": query, "top_k": min(top_k, 40)}
    result = _safe_request("POST", KNOWS_EVIDENCE_PAPER_EN, payload=payload, timeout=EVIDENCE_TIMEOUT)
    if "error" in result:
        logger.warning(f"[英文文献检索失败] query={query}, error={result['error']}")
        return []
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in ["data", "results", "items", "records", "evidences"]:
            if key in result and isinstance(result[key], list):
                return result[key]
    return [result] if result else []


def search_guide(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    检索诊疗指南（最多5条）
    
    Args:
        query: 检索关键词
        top_k: 返回条数上限（最大5）
    
    Returns:
        指南列表，失败时返回空列表
    """
    payload = {"query": query, "top_k": min(top_k, 5)}
    result = _safe_request("POST", KNOWS_EVIDENCE_GUIDE, payload=payload, timeout=EVIDENCE_TIMEOUT)
    if "error" in result:
        logger.warning(f"[诊疗指南检索失败] query={query}, error={result['error']}")
        return []
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in ["data", "results", "items", "records", "evidences"]:
            if key in result and isinstance(result[key], list):
                return result[key]
    return [result] if result else []


def search_package_insert(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    检索药品说明书（最多5条）
    
    Args:
        query: 检索关键词
        top_k: 返回条数上限（最大5）
    
    Returns:
        药品说明书列表，失败时返回空列表
    """
    payload = {"query": query, "top_k": min(top_k, 5)}
    result = _safe_request("POST", KNOWS_EVIDENCE_PACKAGE_INSERT, payload=payload, timeout=EVIDENCE_TIMEOUT)
    if "error" in result:
        logger.warning(f"[药品说明书检索失败] query={query}, error={result['error']}")
        return []
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in ["data", "results", "items", "records", "evidences"]:
            if key in result and isinstance(result[key], list):
                return result[key]
    return [result] if result else []


# ============================================================
# 批量证据检索（供Pipeline调用）
# ============================================================
def batch_search_evidences(queries_cn: List[str], queries_en: List[str],
                           queries_guide: List[str], queries_drug: List[str]) -> Dict[str, List]:
    """
    批量调用四类证据检索接口，汇总全部结果
    
    Args:
        queries_cn: 中文文献检索词列表
        queries_en: 英文文献检索词列表
        queries_guide: 诊疗指南检索词列表
        queries_drug: 药品说明书检索词列表
    
    Returns:
        {
            "papers_cn": [...],
            "papers_en": [...],
            "guides": [...],
            "package_inserts": [...]
        }
    """
    papers_cn = []
    papers_en = []
    guides = []
    package_inserts = []

    # 中文文献：逐query检索并合并去重
    for q in queries_cn:
        results = search_paper_cn(q)
        papers_cn.extend(results)
        logger.info(f"[中文文献] query='{q}' 返回 {len(results)} 条")

    # 英文文献
    for q in queries_en:
        results = search_paper_en(q)
        papers_en.extend(results)
        logger.info(f"[英文文献] query='{q}' 返回 {len(results)} 条")

    # 诊疗指南
    for q in queries_guide:
        results = search_guide(q)
        guides.extend(results)
        logger.info(f"[诊疗指南] query='{q}' 返回 {len(results)} 条")

    # 药品说明书
    for q in queries_drug:
        results = search_package_insert(q)
        package_inserts.extend(results)
        logger.info(f"[药品说明书] query='{q}' 返回 {len(results)} 条")

    return {
        "papers_cn": papers_cn,
        "papers_en": papers_en,
        "guides": guides,
        "package_inserts": package_inserts,
    }

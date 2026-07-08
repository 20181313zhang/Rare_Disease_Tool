"""
api_logger.py - API调用日志收集器
线程安全地记录所有API请求/响应数据，供前端调试查看
"""

import json
import threading
from datetime import datetime
from typing import List, Dict, Any

_log_store: List[Dict[str, Any]] = []
_log_lock = threading.Lock()
MAX_LOGS = 200  # 最多保留200条日志


def record_api_call(
    api_name: str,
    url: str,
    method: str,
    request_payload: Any = None,
    response_data: Any = None,
    status: str = "success",
    error_msg: str = "",
):
    """
    记录一次API调用
    
    Args:
        api_name: API名称（如"中文文献检索"、"大模型调用-病历解析"等）
        url: 请求URL
        method: HTTP方法
        request_payload: 请求体
        response_data: 响应数据
        status: "success" 或 "error"
        error_msg: 错误信息
    """
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "api_name": api_name,
        "url": url,
        "method": method,
        "request": _truncate(_safe_json(request_payload), 2000),
        "response": _truncate(_safe_json(response_data), 2000),
        "status": status,
        "error": error_msg,
    }

    with _log_lock:
        _log_store.append(entry)
        # 超过上限时移除最早的
        if len(_log_store) > MAX_LOGS:
            _log_store.pop(0)


def get_logs() -> List[Dict[str, Any]]:
    """获取全部日志（副本）"""
    with _log_lock:
        return list(_log_store)


def clear_logs():
    """清空日志"""
    with _log_lock:
        _log_store.clear()


def format_logs_markdown() -> str:
    """将日志格式化为Markdown文本，供Gradio展示"""
    logs = get_logs()
    if not logs:
        return "*暂无API调用记录。提交分析任务后，所有API请求和响应将显示在此处。*"

    parts = [f"共 {len(logs)} 条API调用记录（最新在前）\n"]

    # 倒序展示，最新的在前
    for i, entry in enumerate(reversed(logs), 1):
        status_icon = "✅" if entry["status"] == "success" else "❌"
        parts.append(f"---\n### {status_icon} #{i} {entry['api_name']}")
        parts.append(f"**时间**: {entry['time']} | **方法**: {entry['method']} | **URL**: `{entry['url']}`")

        if entry.get("request"):
            req_str = json.dumps(entry["request"], ensure_ascii=False, indent=2) if isinstance(entry["request"], (dict, list)) else str(entry["request"])
            parts.append(f"<details><summary>📤 请求体（点击展开）</summary>\n\n```json\n{req_str}\n```\n</details>")

        if entry.get("response"):
            resp_str = json.dumps(entry["response"], ensure_ascii=False, indent=2) if isinstance(entry["response"], (dict, list)) else str(entry["response"])
            parts.append(f"<details><summary>📥 响应数据（点击展开）</summary>\n\n```json\n{resp_str}\n```\n</details>")

        if entry.get("error"):
            parts.append(f"**错误**: {entry['error']}")

    return "\n\n".join(parts)


def _safe_json(obj: Any) -> Any:
    """安全转换，避免不可序列化对象"""
    if obj is None:
        return None
    if isinstance(obj, (dict, list, str, int, float, bool)):
        return obj
    try:
        return json.loads(json.dumps(obj, ensure_ascii=False, default=str))
    except:
        return str(obj)


def _truncate(obj: Any, max_len: int) -> Any:
    """截断过长的响应数据"""
    if isinstance(obj, str) and len(obj) > max_len:
        return obj[:max_len] + f"... (已截断，总长{len(obj)})"
    if isinstance(obj, dict):
        s = json.dumps(obj, ensure_ascii=False)
        if len(s) > max_len:
            return s[:max_len] + "... (已截断)"
    if isinstance(obj, list) and len(obj) > 20:
        return obj[:20]  # 列表最多保留20条
    return obj

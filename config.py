"""
config.py - 全局配置文件
多病历纵向罕见病筛查鉴别MCP工具
包含所有API地址、密钥、模型配置、合规声明等全局常量
"""

# ============================================================
# 1. KNOWS 官方API配置（病历检索、证据检索、异步任务）
# ============================================================
KNOWS_API_KEY = "sk-knows-rszSuuyS-A8vsRIzANlxafYwkMSXA-dz"
KNOWS_BASE_URL = "https://api.nullht.com/v1"

# KNOWS 各接口路径
KNOWS_JOBS_CREATE = f"{KNOWS_BASE_URL}/jobs"                        # POST 创建异步任务
KNOWS_JOBS_QUERY = f"{KNOWS_BASE_URL}/jobs"                          # GET /jobs/{jobId}
KNOWS_EVIDENCE_PAPER_CN = f"{KNOWS_BASE_URL}/evidences/ai_search_paper_cn"         # 中文医学文献
KNOWS_EVIDENCE_PAPER_EN = f"{KNOWS_BASE_URL}/evidences/ai_search_paper_en"         # 英文医学文献
KNOWS_EVIDENCE_GUIDE = f"{KNOWS_BASE_URL}/evidences/ai_search_guide"               # 诊疗指南
KNOWS_EVIDENCE_PACKAGE_INSERT = f"{KNOWS_BASE_URL}/evidences/ai_search_package_insert"  # 药品说明书

# ============================================================
# 2. 自有大模型配置（StepFun 阶跃星辰）
# ============================================================
LLM_API_KEY = "1yxMRkViXMrr1FVsNYFzpnymOgjsWpxgBs8outTtZWDqQYD3KMlZaPYzOSkIpFLb9"
LLM_BASE_URL = "https://api.stepfun.com/step_plan/v1"
LLM_CHAT_ENDPOINT = f"{LLM_BASE_URL}/chat/completions"  # OpenAI兼容格式

# 主力模型：step-3.5-flash（196B MoE，11B激活，专为Agent和代码任务优化）
LLM_MODEL = "step-3.5-flash"

# ============================================================
# 3. 请求超时与重试配置
# ============================================================
HTTP_TIMEOUT = 120          # 普通HTTP请求超时（秒）
LLM_TIMEOUT = 300           # 大模型调用超时（秒），推理可能较慢
LLM_MAX_RETRIES = 3         # 大模型调用最大重试次数
LLM_RETRY_DELAY = 5         # 重试间隔（秒）
EVIDENCE_TIMEOUT = 60       # 证据检索接口超时（秒）

# ============================================================
# 4. 医疗合规声明（全局固定文本）
# ============================================================
COMPLIANCE_DISCLAIMER = (
    "【医疗合规免责声明】\n"
    "本工具仅为罕见病排查辅助参考工具，不构成任何医学诊断或治疗建议。\n"
    "所有分析结果均基于公开文献与AI模型推理，仅供临床医生作为排查思路参考。\n"
    "严禁将本工具输出作为确诊依据。患者如有健康问题，请务必前往正规医疗机构就诊，\n"
    "由具有资质的专业医师进行面诊、检查与诊断。\n"
    "本工具严禁录入真实患者隐私病历信息，所有演示数据均为虚构模拟病历。"
)

REPORT_FOOTER = (
    "\n\n---\n"
    "【合规声明】本报告由AI辅助生成，仅作为罕见病排查方向参考，不构成确诊结论或医疗建议。"
    "所有分析均基于循证文献检索与大模型推理，具体诊断须由专业医师结合临床实际作出。"
    "严禁将本报告用于任何确诊或治疗决策的唯一依据。"
)

# 强制措辞约束：输出中不得出现的词汇
FORBIDDEN_PHRASES = ["确诊", "患病", "就是"]
# 统一替换用语
REQUIRED_PHRASE = "该罕见病为重点排查可疑方向"

# ============================================================
# 5. Gradio / 应用配置
# ============================================================
APP_TITLE = "多病历纵向罕见病筛查鉴别MCP工具"
APP_DESCRIPTION = (
    "基于大模型理解 + 官方文献检索循证的罕见病线索筛查工具。\n"
    "支持多条时序病历输入，结合中英文文献、诊疗指南、药品说明书进行综合推理分析。\n"
    "**请勿录入真实患者隐私病历信息，所有输入应为模拟/脱敏数据。**"
)
APP_SERVER_PORT = 7860  # 魔搭Space默认端口

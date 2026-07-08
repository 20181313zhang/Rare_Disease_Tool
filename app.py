"""
app.py - Gradio前端入口
多病历纵向罕见病筛查鉴别MCP工具的Web界面
可部署至魔搭ModelScope Space
"""

import time
import gradio as gr

from config import COMPLIANCE_DISCLAIMER, APP_TITLE, APP_DESCRIPTION
from mock_data import get_case_names, fill_case
from mcp_async_task import submit_analysis, query_result
from api_logger import format_logs_markdown, clear_logs


# ============================================================
# 核心交互逻辑
# ============================================================

def run_analysis(records_text, family_history, excluded_diseases,
                 persistent_symptoms, abnormal_labs_summary):
    """
    提交罕见病筛查分析任务并轮询结果
    
    Args:
        records_text: 病历文本
        family_history: 家族史
        excluded_diseases: 已排除疾病
        persistent_symptoms: 持续性异常症状
        abnormal_labs_summary: 异常检验指标汇总
    
    Yields:
        (report_markdown, references_text) 用于Gradio输出更新
    """
    if not records_text or not records_text.strip():
        yield "**错误**: 请输入病历文本", ""
        return

    # 提交异步分析任务
    job_id = submit_analysis(
        records_text=records_text,
        family_history=family_history,
        excluded_diseases=excluded_diseases,
        persistent_symptoms=persistent_symptoms,
        abnormal_labs_summary=abnormal_labs_summary,
    )

    # 轮询等待结果
    yield f"任务已提交 (ID: {job_id})，正在分析中...", ""

    max_poll_time = 600  # 最大轮询等待10分钟
    poll_interval = 3    # 每3秒查询一次
    start_time = time.time()

    while time.time() - start_time < max_poll_time:
        result = query_result(job_id)
        status = result.get("status", "")

        if status == "completed":
            report = result.get("report", "")
            references = result.get("references", "")
            yield report, references
            return
        elif status == "failed":
            error_msg = result.get("error", "分析失败")
            yield f"**分析失败**: {error_msg}", ""
            return

        time.sleep(poll_interval)

    yield "**分析超时**: 任务执行时间过长，请稍后重试", ""
    return



# ============================================================
# Gradio界面构建
# ============================================================

def build_ui():
    """构建Gradio界面"""

    # 合规声明展示文本
    disclaimer_html = f"""
    <div style="background-color:#fff3cd; border:px solid #ffc107; padding:15px; margin:10px 0; border-radius:5px;">
        <strong>⚠️ {COMPLIANCE_DISCLAIMER}</strong>
    </div>
    """

    with gr.Blocks(
        title=APP_TITLE,
    ) as demo:
        # 顶部标题和说明
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESCRIPTION)
        gr.Markdown(disclaimer_html)

        gr.HTML("<hr>")

        with gr.Row():
            with gr.Column(scale=3):
                # 输入区域
                gr.Markdown("### 📋 病历输入区（请勿录入真实患者隐私信息）")

                records_input = gr.Textbox(
                    label="多条时序病历文本",
                    placeholder="请按时间顺序粘贴多条病历内容...\n\n【2024年1月 首次就诊】\n...\n【2024年3月 复诊】\n...",
                    lines=15,
                    info="输入多条带时间标记的病历自由文本，系统将自动解析和结构化处理"
                )

                with gr.Row():
                    family_input = gr.Textbox(
                        label="家族史",
                        placeholder="描述家族中相关疾病史...",
                        lines=3,
                    )
                    excluded_input = gr.Textbox(
                        label="已排除的常见病",
                        placeholder="列出已排除的疾病...",
                        lines=3,
                    )

                with gr.Row():
                    symptoms_input = gr.Textbox(
                        label="持续性异常症状",
                        placeholder="描述持续存在的异常表现...",
                        lines=3,
                    )
                    labs_input = gr.Textbox(
                        label="异常检验指标汇总",
                        placeholder="汇总异常检验指标及变化趋势...",
                        lines=3,
                    )

                # 功能按钮
                with gr.Row():
                    submit_btn = gr.Button(
                        "🔍 提交罕见病筛查分析",
                        variant="primary",
                        size="lg",
                    )

                # 示例病历下拉选择（选中即填充）
                case_dropdown = gr.Dropdown(
                    label="选择示例病历一键填充",
                    choices=get_case_names(),
                    value=None,
                    info="选中即自动填充所有输入框",
                )

            with gr.Column(scale=2):
                # 输出区域
                gr.Markdown("### 📊 分析报告输出区")
                report_output = gr.Markdown(
                    label="筛查分析报告",
                    value="*等待提交分析任务...*",
                )
                references_output = gr.Markdown(
                    label="参考文献列表",
                    value="*等待生成...*",
                )

        # 状态栏
        status_bar = gr.Textbox(
            label="执行状态",
            value="就绪",
            interactive=False,
        )

        # API调用日志区域
        with gr.Accordion("📝 API调用日志（点击查看所有请求/响应数据）", open=False):
            log_output = gr.Markdown(
                value="*暂无API调用记录。提交分析任务后，所有API请求和响应将显示在此处。*",
            )
            with gr.Row():
                refresh_log_btn = gr.Button("🔄 刷新日志", size="sm")
                clear_log_btn = gr.Button("🗑️ 清空日志", size="sm", variant="stop")

        # ============================================================
        # 事件绑定
        # ============================================================

        # 提交分析按钮
        submit_btn.click(
            fn=run_analysis,
            inputs=[records_input, family_input, excluded_input,
                    symptoms_input, labs_input],
            outputs=[report_output, references_output],
            show_progress=True,
        )

        # 示例病历填充（下拉框选中即触发）
        case_dropdown.change(
            fn=fill_case,
            inputs=[case_dropdown],
            outputs=[records_input, family_input, excluded_input,
                     symptoms_input, labs_input],
        )

        # 日志刷新按钮
        refresh_log_btn.click(
            fn=lambda: format_logs_markdown(),
            inputs=[],
            outputs=[log_output],
        )

        # 日志清空按钮
        clear_log_btn.click(
            fn=lambda: (clear_logs(), "*日志已清空*")[-1],
            inputs=[],
            outputs=[log_output],
        )

        # 底部合规声明
        gr.HTML(f"<hr><div style='text-align:center; padding:10px; color:#666; font-size:12px;'>{COMPLIANCE_DISCLAIMER}</div>")

    return demo


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )

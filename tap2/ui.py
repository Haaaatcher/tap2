import gradio as gr
import click
from tap2.utils import _beta_get_TA_prop, _beta_get_AA_prop


@click.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=7860)
def run_gradio(host, port):
    with gr.Blocks(title="TAP2") as index:
        # 定义界面布局
        gr.Markdown("## 钛合金/铝合金性能预测")
        with gr.Tabs():
            with gr.Tab("钛合金"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 元素组成")
                        with gr.Row():
                            TA_Ti = gr.Number(label='Ti (wt%)', value=100, minimum=0, maximum=100, interactive=True)
                            TA_Al = gr.Number(label='Al (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                            TA_V = gr.Number(label='V (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                            TA_Cr = gr.Number(label='Cr (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                            TA_Cu = gr.Number(label='Cu (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                            TA_Zr = gr.Number(label='Zr (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                            TA_Mo = gr.Number(label='Mo (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                        gr.Markdown("### 工艺参数")
                        TA_proc = gr.Text(label="工艺参数", placeholder='820℃/1h-500℃/6h', interactive=True)
                        with gr.Row():
                            TA_clear_btn = gr.Button("重置", interactive=True)
                            TA_run_btn = gr.Button("提交", interactive=True)
                    with gr.Column():
                        gr.Markdown('### 物理性能')
                        TA_TE = gr.Number(label="热膨胀系数 (10^-6/K)", value=0, interactive=False, precision=3)
                        TA_TC = gr.Number(label="热导率 (W/m·K)", value=0, interactive=False, precision=3)
                        gr.Markdown('### 力学性能')
                        TA_YS = gr.Number(label="屈服强度 (MPa)", value=0, interactive=False, precision=3)
                        TA_TS = gr.Number(label="抗拉强度 (MPa)", value=0, interactive=False, precision=3)
                        gr.Markdown('### 相比例')
                        TA_alpha1_ratio = gr.Number(label="等轴α相比例 (%)", value=0, interactive=False, precision=3)
                        TA_alpha2_ratio = gr.Number(label="(次生)α相比例 (%)", value=0, interactive=False, precision=3)
                        TA_beta_ratio = gr.Number(label="β相比例 (%)", value=0, interactive=False, precision=3)
                        gr.Markdown('### 相尺寸')
                        TA_alpha1_size = gr.Number(label="等轴α相直径 (μm)", value=0, interactive=False, precision=3)
                        TA_alpha2_size = gr.Number(label="(次生)片层α相厚度 (μm)", value=0, interactive=False, precision=3)
                        TA_beta_size = gr.Number(label="片层β相厚度 (μm)", value=0, interactive=False, precision=3)
            with gr.Tab("铝合金"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 元素组成")
                        with gr.Row():
                            AA_Al = gr.Number(label='Al (wt%)', value=100, minimum=0, maximum=100, interactive=True)
                            AA_Mg = gr.Number(label='Mg (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                            AA_Si = gr.Number(label='Si (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                            AA_Cr = gr.Number(label='Cr (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                            AA_Mn = gr.Number(label='Mn (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                            AA_Fe = gr.Number(label='Fe (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                            AA_Cu = gr.Number(label='Cu (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                            AA_Zn = gr.Number(label='Zn (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                            AA_Zr = gr.Number(label='Zr (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                            AA_Ag = gr.Number(label='Ag (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                        gr.Markdown("### 工艺参数")
                        AA_proc = gr.Text(label="工艺参数", placeholder='495℃/1h-25℃/7days', interactive=True)
                        with gr.Row():
                            AA_clear_btn = gr.Button("重置", interactive=True)
                            AA_run_btn = gr.Button("提交", interactive=True)
                    with gr.Column():
                        gr.Markdown('### 物理性能')
                        AA_TE = gr.Number(label="热膨胀系数 (10^-6/K)", value=0, interactive=False, precision=3)
                        AA_TC = gr.Number(label="热导率 (W/m·K)", value=0, interactive=False, precision=3)
                        gr.Markdown('### 力学性能')
                        AA_YS = gr.Number(label="屈服强度 (MPa)", value=0, interactive=False, precision=3)
                        AA_TS = gr.Number(label="抗拉强度 (MPa)", value=0, interactive=False, precision=3)
                        gr.Markdown('### 相比例')
                        AA_phase1_ratio = gr.Number(label="第二相1体积分数 (%)", value=0, interactive=False, precision=3)
                        AA_phase2_ratio = gr.Number(label="第二相2体积分数 (%)", value=0, interactive=False, precision=3)
                        gr.Markdown('### 相尺寸')
                        AA_phase1_size = gr.Number(label="第二相1平均粒子半径 (nm)", value=0, interactive=False, precision=3)
                        AA_phase2_size = gr.Number(label="第二相2平均粒子半径 (nm)", value=0, interactive=False, precision=3)
        # 定义交互逻辑
        TA_clear_btn.click(
            fn=lambda: (100, 0, 0, 0, 0, 0, 0, '', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            outputs=[TA_Ti, TA_Al, TA_V, TA_Cr, TA_Cu, TA_Zr, TA_Mo, TA_proc, TA_TE, TA_TC, TA_YS, TA_TS, TA_alpha1_ratio, TA_alpha2_ratio, TA_beta_ratio, TA_alpha1_size, TA_alpha2_size, TA_beta_size]
        )
        for elem_num in (TA_Al, TA_V, TA_Cr, TA_Cu, TA_Zr, TA_Mo):
            elem_num.change(
                fn=lambda *args: 100 - sum(_ for _ in args if _ is not None),
                inputs=[TA_Al, TA_V, TA_Cr, TA_Cu, TA_Zr, TA_Mo],
                outputs=TA_Ti
            )
        TA_run_btn.click(
            fn=_beta_get_TA_prop,
            inputs=[TA_Ti, TA_Al, TA_V, TA_Cr, TA_Cu, TA_Zr, TA_Mo, TA_proc],
            outputs=[TA_TE, TA_TC, TA_YS, TA_TS, TA_alpha1_ratio, TA_alpha2_ratio, TA_beta_ratio, TA_alpha1_size, TA_alpha2_size, TA_beta_size]
        )
        AA_clear_btn.click(
            fn=lambda: (100, 0, 0, 0, 0, 0, 0, 0, 0, 0, '', 0, 0, 0, 0, 0, 0, 0, 0),
            outputs=[AA_Al, AA_Mg, AA_Si, AA_Cr, AA_Mn, AA_Fe, AA_Cu, AA_Zn, AA_Zr, AA_Ag, AA_proc, AA_TE, AA_TC, AA_YS, AA_TS, AA_phase1_ratio, AA_phase2_ratio, AA_phase1_size, AA_phase2_size]
        )
        for elem_num in (AA_Mg, AA_Si, AA_Cr, AA_Mn, AA_Fe, AA_Cu, AA_Zn, AA_Zr, AA_Ag):
            elem_num.change(
                fn=lambda *args: 100 - sum(_ for _ in args if _ is not None),
                inputs=[AA_Mg, AA_Si, AA_Cr, AA_Mn, AA_Fe, AA_Cu, AA_Zn, AA_Zr, AA_Ag],
                outputs=AA_Al
            )
        AA_run_btn.click(
            fn=_beta_get_AA_prop,
            inputs=[AA_Al, AA_Mg, AA_Si, AA_Cr, AA_Mn, AA_Fe, AA_Cu, AA_Zn, AA_Zr, AA_Ag, AA_proc],
            outputs=[AA_TE, AA_TC, AA_YS, AA_TS, AA_phase1_ratio, AA_phase2_ratio, AA_phase1_size, AA_phase2_size]
        )
    index.queue(max_size=32, default_concurrency_limit=4).launch(server_name=host, server_port=port, share=False)

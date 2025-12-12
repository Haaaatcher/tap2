import gradio as gr
import click
from importlib import resources
from tap2.utils import _get_ti_alloy_phys_prop, _get_ti_alloy_mech_prop, _get_ti_alloy_prop, _get_ti_alloy_wf


@click.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=7860)
def run_gradio(host, port):
    with (gr.Blocks(title="钛合金性能预测模块") as index):
        # 定义界面布局
        gr.Markdown("## 钛合金性能预测模块")
        with gr.Tabs():
            with gr.Tab("物理性能"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 元素组成")
                        with gr.Row():
                            phys_Ti = gr.Number(label=f"Ti (wt%)", value=100, minimum=0, maximum=100, interactive=False)
                            phys_H = gr.Number(label=f"H (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_B = gr.Number(label=f"B (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_C = gr.Number(label=f"C (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_N = gr.Number(label=f"N (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_O = gr.Number(label=f"O (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Al = gr.Number(label=f"Al (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Si = gr.Number(label=f"Si (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Cr = gr.Number(label=f"Cr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Fe = gr.Number(label=f"Fe (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Ni = gr.Number(label=f"Ni (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Cu = gr.Number(label=f"Cu (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Zr = gr.Number(label=f"Zr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Nb = gr.Number(label=f"Nb (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Mo = gr.Number(label=f"Mo (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_V = gr.Number(label=f"V (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Sn = gr.Number(label=f"Sn (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                        gr.Markdown("### 工艺参数")
                        phys_htt = gr.Number(label="热处理温度 (℃)", value=600, minimum=-273.15, interactive=True)
                        with gr.Row():
                            phys_clear_btn = gr.Button("重置", interactive=True)
                            phys_run_btn = gr.Button("提交", interactive=True)
                    with gr.Column():
                        gr.Markdown("### 物理性能")
                        phys_te = gr.Number(label="热膨胀系数 (10^-6/K)", value=0, interactive=False, precision=3)
                        phys_d = gr.Number(label="密度 (g/cm^3)", value=0, interactive=False, precision=3)
                        phys_tc = gr.Number(label="热导率 (W/m·K)", value=0, interactive=False, precision=3)
                        phys_ec = gr.Number(label="电导率 (10^6 S/m)", value=0, interactive=False, precision=3)
                        phys_ym = gr.Number(label="杨氏模量 (GPa)", value=0, interactive=False, precision=3)
                        phys_bm = gr.Number(label="体积模量 (GPa)", value=0, interactive=False, precision=3)
                        phys_sm = gr.Number(label="剪切模量 (GPa)", value=0, interactive=False, precision=3)
                        phys_pr = gr.Number(label="泊松比", value=0, interactive=False, precision=3)
                        phys_se = gr.Number(label="比焓 (J/g)", value=0, interactive=False, precision=3)
                        phys_shc = gr.Number(label="比热容 (J/g·K)", value=0, interactive=False, precision=3)
            with gr.Tab("力学性能"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 元素组成")
                        with gr.Row():
                            mech_Ti = gr.Number(label=f"Ti (wt%)", value=100, minimum=0, maximum=100, interactive=False)
                            mech_H = gr.Number(label=f"H (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_B = gr.Number(label=f"B (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_C = gr.Number(label=f"C (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_N = gr.Number(label=f"N (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_O = gr.Number(label=f"O (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Al = gr.Number(label=f"Al (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Si = gr.Number(label=f"Si (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Cr = gr.Number(label=f"Cr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Fe = gr.Number(label=f"Fe (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Ni = gr.Number(label=f"Ni (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Cu = gr.Number(label=f"Cu (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Zr = gr.Number(label=f"Zr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Nb = gr.Number(label=f"Nb (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Mo = gr.Number(label=f"Mo (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_V = gr.Number(label=f"V (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Sn = gr.Number(label=f"Sn (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                        gr.Markdown("### 工艺参数")
                        mech_proc_mode = gr.State(value="simple")
                        with gr.Tabs():
                            with gr.Tab("快捷模式", id="simple") as mech_simple_mode:
                                mech_htt = gr.Number(label="热处理温度 (℃)", value=600, minimum=-273.15, interactive=True)
                                mech_gs = gr.Number(label="晶粒尺寸 (μm)", value=10, minimum=0, interactive=True)
                            with gr.Tab("高级模式", id="advanced") as mech_advanced_mode:
                                with gr.Accordion("热变形", open=True):
                                    mech_td_htt = gr.Number(label="温度（℃）", info="Required", value=960, interactive=True)
                                    mech_td_ts = gr.Number(label="真实应变", info="Required", value=0.8, interactive=True)
                                    mech_td_sr = gr.Number(label="应变速率（1/s）", info="Required", value=0.1, interactive=True)
                                    mech_td_init_gs = gr.Number(label="初始晶粒尺寸（μm）", info="Optional", value=100, interactive=True)
                                    mech_td_btt = gr.Number(label="β相变温度（℃）", info="Optional", value=1000, interactive=True)
                        with gr.Row():
                            mech_clear_btn = gr.Button("重置", interactive=True)
                            mech_run_btn = gr.Button("提交", interactive=True)
                    with gr.Column():
                        gr.Markdown("### 力学性能")
                        mech_ys = gr.Number(label="屈服强度 (MPa)", value=0, interactive=False, precision=3)
                        mech_ts = gr.Number(label="抗拉强度 (MPa)", value=0, interactive=False, precision=3)
                        mech_h = gr.Number(label="硬度 (VPN)", value=0, interactive=False, precision=3)
                        mech_hp = gr.Number(label="霍尔佩奇系数 (MPa·m^(1/2))", value=0, interactive=False, precision=3)
            with gr.Tab("相比例"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 元素组成")
                        with gr.Row():
                            wf_Ti = gr.Number(label=f"Ti (wt%)", value=100, minimum=0, maximum=100, interactive=False)
                            wf_Al = gr.Number(label=f"Al (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            wf_Si = gr.Number(label=f"Si (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            wf_Cr = gr.Number(label=f"Cr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            wf_Fe = gr.Number(label=f"Fe (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            wf_Ni = gr.Number(label=f"Ni (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            wf_Cu = gr.Number(label=f"Cu (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            wf_Zr = gr.Number(label=f"Zr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            wf_Nb = gr.Number(label=f"Nb (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            wf_Mo = gr.Number(label=f"Mo (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            wf_V = gr.Number(label=f"V (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            wf_Sn = gr.Number(label=f"Sn (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                        gr.Markdown("### 工艺参数")
                        wf_htt = gr.Number(label="热处理温度 (℃)", value=600, minimum=-273.15, interactive=True)
                        with gr.Row():
                            wf_clear_btn = gr.Button("重置", interactive=True)
                            wf_run_btn = gr.Button("提交", interactive=True)
                    with gr.Column():
                        gr.Markdown("### 相比例")
                        with gr.Row():
                            wf_alpha = gr.Number(label="ALPHA (wt%)", value=0, interactive=False, precision=3)
                            wf_beta = gr.Number(label="BETA (wt%)", value=0, interactive=False, precision=3)
                            wf_laves = gr.Number(label="LAVES (wt%)", value=0, interactive=False, precision=3)
                            wf_ti3al = gr.Number(label="TI3AL (wt%)", value=0, interactive=False, precision=3)
                            wf_ti2cu = gr.Number(label="TI2CU (wt%)", value=0, interactive=False, precision=3)
                            wf_ti5si3 = gr.Number(label="TI5SI3 (wt%)", value=0, interactive=False, precision=3)
                            wf_tizrsi = gr.Number(label="TIZRSI (wt%)", value=0, interactive=False, precision=3)
                            wf_ti2ni = gr.Number(label="TI2NI (wt%)", value=0, interactive=False, precision=3)
                            wf_timb2 = gr.Number(label="TIM_B2 (wt%)", value=0, interactive=False, precision=3)
                            wf_liquid = gr.Number(label="LIQUID (wt%)", value=0, interactive=False, precision=3)
                            wf_c15fcc = gr.Number(label="C15_FCC (wt%)", value=0, interactive=False, precision=3)
                            wf_mc = gr.Number(label="MC (wt%)", value=0, interactive=False, precision=3)
                        wf_plot = gr.Plot()
            with gr.Tab("β-转变温度"):
                gr.Markdown("即将到来...")
            with gr.Tab("批量模式"):
                with gr.Row():
                    with gr.Column():
                        up_files = gr.File(label="上传 CSV 或 XLSX 文件",
                                           file_types=[".csv", ".xlsx"],
                                           file_count="multiple",
                                           type="filepath",
                                           interactive=True)
                        checkboxes = gr.CheckboxGroup(choices=["热膨胀系数", "密度", "热导率", "电导率", "杨氏模量",
                                                               "体积模量", "剪切模量", "泊松比", "比焓", "比热容",
                                                               "屈服强度", "抗拉强度", "硬度", "霍尔佩奇系数"],
                                                      label="选择性能",
                                                      interactive=True)
                        with gr.Row():
                            batch_clear_btn = gr.Button("重置", interactive=True)
                            batch_run_btn = gr.Button("提交", interactive=True)
                            batch_all_btn = gr.Button("全选", interactive=True)
                    with gr.Column():
                        down_files = gr.File(label="下载 CSV 或 XLSX 文件",
                                             file_types=[".csv", ".xlsx"],
                                             file_count="multiple",
                                             type="filepath",
                                             interactive=False)
            with gr.Tab("模板下载"):
                gr.File(value=[str(resources.files("tap2.resource").joinpath("tapp_template.csv")),
                               str(resources.files("tap2.resource").joinpath("tapp_template.xlsx"))],
                        label="下载批处理模板文件",
                        file_count="multiple",
                        type="filepath",
                        interactive=False)
        # 定义交互逻辑
        phys_clear_btn.click(
            fn=lambda: [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            outputs=[phys_Ti, phys_H, phys_B, phys_C, phys_N, phys_O, phys_Al, phys_Si, phys_Cr, phys_Fe, phys_Ni,
                     phys_Cu, phys_Zr, phys_Nb, phys_Mo, phys_V, phys_Sn, phys_htt, phys_te, phys_d, phys_tc, phys_ec,
                     phys_ym, phys_bm, phys_sm, phys_pr, phys_se, phys_shc]
        )
        phys_run_btn.click(
            fn=_get_ti_alloy_phys_prop,
            inputs=[phys_Ti, phys_H, phys_B, phys_C, phys_N, phys_O, phys_Al, phys_Si, phys_Cr, phys_Fe, phys_Ni,
                    phys_Cu, phys_Zr, phys_Nb, phys_Mo, phys_V, phys_Sn, phys_htt],
            outputs=[phys_te, phys_d, phys_tc, phys_ec, phys_ym, phys_bm, phys_sm, phys_pr, phys_se, phys_shc]
        )
        for phys_elem in [phys_H, phys_B, phys_C, phys_N, phys_O, phys_Al, phys_Si, phys_Cr, phys_Fe, phys_Ni, phys_Cu,
                          phys_Zr, phys_Nb, phys_Mo, phys_V, phys_Sn]:
            phys_elem.change(
                fn=lambda *args: 100 - sum(arg for arg in args if arg is not None),
                inputs=[phys_H, phys_B, phys_C, phys_N, phys_O, phys_Al, phys_Si, phys_Cr, phys_Fe, phys_Ni, phys_Cu,
                        phys_Zr, phys_Nb, phys_Mo, phys_V, phys_Sn],
                outputs=phys_Ti
            )
        mech_simple_mode.select(fn=lambda: "simple", outputs=mech_proc_mode)
        mech_advanced_mode.select(fn=lambda: "advanced", outputs=mech_proc_mode)
        mech_clear_btn.click(
            fn=lambda: [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600, 10, 960, 0.8, 0.1, 100, 1000, 0, 0, 0, 0],
            outputs=[mech_Ti, mech_H, mech_B, mech_C, mech_N, mech_O, mech_Al, mech_Si, mech_Cr, mech_Fe, mech_Ni,
                     mech_Cu, mech_Zr, mech_Nb, mech_Mo, mech_V, mech_Sn, mech_htt, mech_gs, mech_td_htt, mech_td_ts,
                     mech_td_sr, mech_td_init_gs, mech_td_btt, mech_ys, mech_ts, mech_h, mech_hp]
        )
        mech_run_btn.click(
            fn=_get_ti_alloy_mech_prop,
            inputs=[mech_Ti, mech_H, mech_B, mech_C, mech_N, mech_O, mech_Al, mech_Si, mech_Cr, mech_Fe, mech_Ni,
                    mech_Cu, mech_Zr, mech_Nb, mech_Mo, mech_V, mech_Sn, mech_proc_mode, mech_htt, mech_gs, mech_td_htt,
                    mech_td_ts, mech_td_sr, mech_td_init_gs, mech_td_btt],
            outputs=[mech_ys, mech_ts, mech_h, mech_hp]
        )
        for mech_elem in [mech_H, mech_B, mech_C, mech_N, mech_O, mech_Al, mech_Si, mech_Cr, mech_Fe, mech_Ni, mech_Cu,
                          mech_Zr, mech_Nb, mech_Mo, mech_V, mech_Sn]:
            mech_elem.change(
                fn=lambda *args: 100 - sum(arg for arg in args if arg is not None),
                inputs=[mech_H, mech_B, mech_C, mech_N, mech_O, mech_Al, mech_Si, mech_Cr, mech_Fe, mech_Ni, mech_Cu,
                        mech_Zr, mech_Nb, mech_Mo, mech_V, mech_Sn],
                outputs=mech_Ti
            )
        batch_run_btn.click(
            fn=_get_ti_alloy_prop,
            inputs=[checkboxes, up_files],
            outputs=down_files
        )
        batch_all_btn.click(
            fn=lambda sel_props: list({"热膨胀系数", "密度", "热导率", "电导率", "杨氏模量", "体积模量", "剪切模量", "泊松比",
                                       "比焓","比热容", "屈服强度", "抗拉强度", "硬度", "霍尔佩奇系数"} - set(sel_props)),
            inputs=checkboxes,
            outputs=checkboxes
        )
        batch_clear_btn.click(
            fn=lambda: [None, None, None],
            outputs=[up_files, down_files, checkboxes]
        )
        wf_clear_btn.click(
            fn=lambda: [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None],
            outputs=[wf_Ti, wf_Al, wf_Si, wf_Cr, wf_Fe, wf_Ni, wf_Cu, wf_Zr, wf_Nb, wf_Mo, wf_V, wf_Sn, wf_htt,
                     wf_alpha, wf_beta, wf_laves, wf_ti3al, wf_ti2cu, wf_ti5si3, wf_tizrsi, wf_ti2ni, wf_timb2,
                     wf_liquid, wf_c15fcc, wf_mc, wf_plot]
        )
        wf_run_btn.click(
            fn=_get_ti_alloy_wf,
            inputs=[wf_Ti, wf_Al, wf_Si, wf_Cr, wf_Fe, wf_Ni, wf_Cu, wf_Zr, wf_Nb, wf_Mo, wf_V, wf_Sn, wf_htt],
            outputs=[wf_alpha, wf_beta, wf_laves, wf_ti3al, wf_ti2cu, wf_ti5si3, wf_tizrsi, wf_ti2ni, wf_timb2,
                     wf_liquid, wf_c15fcc, wf_mc, wf_plot]
        )
        for wf_elem in [wf_Al, wf_Si, wf_Cr, wf_Fe, wf_Ni, wf_Cu, wf_Zr, wf_Nb, wf_Mo, wf_V, wf_Sn]:
            wf_elem.change(
                fn=lambda *args: 100 - sum(arg for arg in args if arg is not None),
                inputs=[wf_Al, wf_Si, wf_Cr, wf_Fe, wf_Ni, wf_Cu, wf_Zr, wf_Nb, wf_Mo, wf_V, wf_Sn],
                outputs=wf_Ti
            )
    index.queue(max_size=32, default_concurrency_limit=4).launch(server_name=host, server_port=port, share=False)

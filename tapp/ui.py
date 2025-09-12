import gradio as gr
from importlib import resources
from tapp.utils import _beta_get_ti_alloy_phys_prop, _beta_get_ti_alloy_mech_prop, _beta_get_al_alloy_prop, _beta_get_ti_alloy_prop


def run_gradio():
    with gr.Blocks(title="钛合金性能预测模块") as index:
        # 定义界面布局
        gr.Markdown("## 钛合金性能预测模块")
        with gr.Tabs():
            with gr.Tab("物理性能"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 元素组成")
                        with gr.Row():
                            phys_Ti = gr.Number(label=f"Ti (wt%)", value=100, minimum=0, maximum=100, interactive=True)
                            phys_H = gr.Number(label=f"H (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_B = gr.Number(label=f"B (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_C = gr.Number(label=f"C (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_N = gr.Number(label=f"N (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_O = gr.Number(label=f"O (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Al = gr.Number(label=f"Al (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Si = gr.Number(label=f"Si (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Cr = gr.Number(label=f"Cr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Mn = gr.Number(label=f"Mn (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Fe = gr.Number(label=f"Fe (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Co = gr.Number(label=f"Co (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Ni = gr.Number(label=f"Ni (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Cu = gr.Number(label=f"Cu (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Nb = gr.Number(label=f"Nb (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Mo = gr.Number(label=f"Mo (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Ta = gr.Number(label=f"Ta (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_V = gr.Number(label=f"V (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Zr = gr.Number(label=f"Zr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Sn = gr.Number(label=f"Sn (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            phys_Bi = gr.Number(label=f"Bi (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                        gr.Markdown("### 工艺参数")
                        phys_proc = gr.Textbox(lines=3, placeholder="在这里输入工艺参数文本...",
                                               show_label=False, interactive=True)
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
                            mech_Ti = gr.Number(label=f"Ti (wt%)", value=100, minimum=0, maximum=100, interactive=True)
                            mech_H = gr.Number(label=f"H (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_B = gr.Number(label=f"B (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_C = gr.Number(label=f"C (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_N = gr.Number(label=f"N (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_O = gr.Number(label=f"O (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Al = gr.Number(label=f"Al (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Si = gr.Number(label=f"Si (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Cr = gr.Number(label=f"Cr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Mn = gr.Number(label=f"Mn (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Fe = gr.Number(label=f"Fe (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Co = gr.Number(label=f"Co (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Ni = gr.Number(label=f"Ni (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Cu = gr.Number(label=f"Cu (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Zr = gr.Number(label=f"Zr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Nb = gr.Number(label=f"Nb (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Mo = gr.Number(label=f"Mo (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Ta = gr.Number(label=f"Ta (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_V = gr.Number(label=f"V (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Sn = gr.Number(label=f"Sn (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                            mech_Bi = gr.Number(label=f"Bi (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                        gr.Markdown("### 工艺参数")
                        mech_proc = gr.Textbox(lines=3, placeholder="在这里输入工艺参数文本...",
                                               show_label=False, interactive=True)
                        with gr.Row():
                            mech_clear_btn = gr.Button("重置", interactive=True)
                            mech_run_btn = gr.Button("提交", interactive=True)
                    with gr.Column():
                        gr.Markdown("### 力学性能")
                        mech_ys = gr.Number(label="屈服强度 (MPa)", value=0, interactive=False, precision=3)
                        mech_ts = gr.Number(label="抗拉强度 (MPa)", value=0, interactive=False, precision=3)
                        mech_h = gr.Number(label="硬度 (VPN)", value=0, interactive=False, precision=3)
                        mech_hp = gr.Number(label="霍尔佩奇系数 (MPa·m^(1/2))", value=0,
                                            interactive=False, precision=3)
            with gr.Tab("批量模式"):
                with gr.Row():
                    with gr.Column():
                        up_files = gr.File(label="上传 CSV 或 XLSX 文件",
                                           file_types=[".csv", ".xlsx"],
                                           file_count="multiple",
                                           type="filepath",
                                           interactive=True)
                        down_files = gr.File(label="下载 CSV 或 XLSX 文件",
                                             file_types=[".csv", ".xlsx"],
                                             file_count="multiple",
                                             type="filepath",
                                             interactive=False)
                    with gr.Column():
                        checkboxes = gr.CheckboxGroup(choices=["热膨胀系数", "密度", "热导率", "电导率", "杨氏模量",
                                                               "体积模量","剪切模量", "泊松比", "比焓", "比热容",
                                                               "屈服强度", "抗拉强度", "硬度", "霍尔佩奇系数"],
                                                      label="选择性能",
                                                      interactive=True)
                        with gr.Row():
                            batch_clear_btn = gr.Button("重置", interactive=True)
                            batch_run_btn = gr.Button("提交", interactive=True)
                            batch_all_btn = gr.Button("全选", interactive=True)
            with gr.Tab("模板下载"):
                gr.File(value=[str(resources.files("tapp.resource").joinpath("tapp_template.csv")),
                               str(resources.files("tapp.resource").joinpath("tapp_template.xlsx"))],
                        label="下载批处理模板文件",
                        file_count="multiple",
                        type="filepath",
                        interactive=False)
            with gr.Tab("铝合金性能"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 元素组成")
                        with gr.Row():
                            al_alloy_Al = gr.Number(label=f"Al (wt%)", value=100, minimum=0, maximum=100,
                                                    interactive=True)
                            al_alloy_Mg = gr.Number(label=f"Mg (wt%)", value=0, minimum=0, maximum=100,
                                                    interactive=True)
                            al_alloy_Si = gr.Number(label=f"Si (wt%)", value=0, minimum=0, maximum=100,
                                                    interactive=True)
                            al_alloy_Ca = gr.Number(label=f"Ca (wt%)", value=0, minimum=0, maximum=100,
                                                    interactive=True)
                            al_alloy_Cr = gr.Number(label=f"Cr (wt%)", value=0, minimum=0, maximum=100,
                                                    interactive=True)
                            al_alloy_Mn = gr.Number(label=f"Mn (wt%)", value=0, minimum=0, maximum=100,
                                                    interactive=True)
                            al_alloy_Fe = gr.Number(label=f"Fe (wt%)", value=0, minimum=0, maximum=100,
                                                    interactive=True)
                            al_alloy_Ni = gr.Number(label=f"Ni (wt%)", value=0, minimum=0, maximum=100,
                                                    interactive=True)
                            al_alloy_Cu = gr.Number(label=f"Cu (wt%)", value=0, minimum=0, maximum=100,
                                                    interactive=True)
                            al_alloy_Zn = gr.Number(label=f"Zn (wt%)", value=0, minimum=0, maximum=100,
                                                    interactive=True)
                            al_alloy_Ce = gr.Number(label=f"Ce (wt%)", value=0, minimum=0, maximum=100,
                                                    interactive=True)
                        gr.Markdown("### 工艺参数")
                        al_alloy_proc = gr.Textbox(lines=3, placeholder="在这里输入工艺参数文本...", show_label=False,
                                                   interactive=True)
                        with gr.Row():
                            al_alloy_clear_btn = gr.Button("重置", interactive=True)
                            al_alloy_run_btn = gr.Button("提交", interactive=True)
                    with gr.Column():
                        gr.Markdown("### 物理性能")
                        al_alloy_te = gr.Number(label="热膨胀系数 (10^-6/K)", value=0, interactive=False, precision=3)
                        al_alloy_tc = gr.Number(label="热导率 (W/m·K)", value=0, interactive=False, precision=3)
                        gr.Markdown("### 力学性能")
                        al_alloy_ys = gr.Number(label="屈服强度 (MPa)", value=0, interactive=False, precision=3)
                        al_alloy_ts = gr.Number(label="抗拉强度 (MPa)", value=0, interactive=False, precision=3)
        # 定义交互逻辑
        phys_clear_btn.click(
            fn=lambda: [100., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., "", 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0.],
            outputs=[phys_Ti, phys_H, phys_B, phys_C, phys_N, phys_O, phys_Al, phys_Si, phys_Cr, phys_Mn, phys_Fe,
                     phys_Co, phys_Ni, phys_Cu, phys_Zr, phys_Nb, phys_Mo, phys_Ta, phys_V, phys_Sn, phys_Bi, phys_proc,
                     phys_te, phys_d, phys_tc, phys_ec, phys_ym, phys_bm, phys_sm, phys_pr, phys_se, phys_shc]
        )
        phys_run_btn.click(
            fn=_beta_get_ti_alloy_phys_prop,
            inputs=[phys_Ti, phys_H, phys_B, phys_C, phys_N, phys_O, phys_Al, phys_Si, phys_Cr, phys_Mn, phys_Fe,
                    phys_Co, phys_Ni, phys_Cu, phys_Zr, phys_Nb, phys_Mo, phys_Ta, phys_V, phys_Sn, phys_Bi, phys_proc],
            outputs=[phys_te, phys_d, phys_tc, phys_ec, phys_ym, phys_bm, phys_sm, phys_pr, phys_se, phys_shc]
        )
        for phys_elem in [phys_H, phys_B, phys_C, phys_N, phys_O, phys_Al, phys_Si, phys_Cr, phys_Mn, phys_Fe, phys_Co,
                          phys_Ni, phys_Cu, phys_Zr, phys_Nb, phys_Mo, phys_Ta, phys_V, phys_Sn, phys_Bi]:
            phys_elem.change(
                fn=lambda *args: 100 - sum(args),
                inputs=[phys_H, phys_B, phys_C, phys_N, phys_O, phys_Al, phys_Si, phys_Cr, phys_Mn, phys_Fe, phys_Co,
                        phys_Ni, phys_Cu, phys_Zr, phys_Nb, phys_Mo, phys_Ta, phys_V, phys_Sn, phys_Bi],
                outputs=phys_Ti
            )
        mech_clear_btn.click(
            fn=lambda: [100., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., "", 0.,
                        0., 0., 0.],
            outputs=[mech_Ti, mech_H, mech_B, mech_C, mech_N, mech_O, mech_Al, mech_Si, mech_Cr, mech_Mn, mech_Fe,
                     mech_Co, mech_Ni, mech_Cu, mech_Zr, mech_Nb, mech_Mo, mech_Ta, mech_V, mech_Sn, mech_Bi, mech_proc,
                     mech_ys, mech_ts, mech_h, mech_hp]
        )
        mech_run_btn.click(
            fn=_beta_get_ti_alloy_mech_prop,
            inputs=[mech_Ti, mech_H, mech_B, mech_C, mech_N, mech_O, mech_Al, mech_Si, mech_Cr, mech_Mn, mech_Fe,
                    mech_Co, mech_Ni, mech_Cu, mech_Zr, mech_Nb, mech_Mo, mech_Ta, mech_V, mech_Sn, mech_Bi, phys_proc],
            outputs=[mech_ys, mech_ts, mech_h, mech_hp]
        )
        for mech_elem in [mech_H, mech_B, mech_C, mech_N, mech_O, mech_Al, mech_Si, mech_Cr, mech_Mn, mech_Fe, mech_Co,
                          mech_Ni, mech_Cu, mech_Zr, mech_Nb, mech_Mo, mech_Ta, mech_V, mech_Sn, mech_Bi]:
            mech_elem.change(
                fn=lambda *args: 100 - sum(args),
                inputs=[mech_H, mech_B, mech_C, mech_N, mech_O, mech_Al, mech_Si, mech_Cr, mech_Mn, mech_Fe, mech_Co,
                        mech_Ni, mech_Cu, mech_Zr, mech_Nb, mech_Mo, mech_Ta, mech_V, mech_Sn, mech_Bi],
                outputs=mech_Ti
            )
        batch_run_btn.click(
            fn=_beta_get_ti_alloy_prop,
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
        al_alloy_clear_btn.click(
            fn=lambda: [100., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., "", 0., 0., 0., 0.],
            outputs=[al_alloy_Al, al_alloy_Mg, al_alloy_Si, al_alloy_Ca, al_alloy_Cr, al_alloy_Mn, al_alloy_Fe,
                     al_alloy_Ni, al_alloy_Cu, al_alloy_Zn, al_alloy_Ce, al_alloy_proc, al_alloy_te, al_alloy_tc,
                     al_alloy_ys, al_alloy_ts]
        )
        al_alloy_run_btn.click(
            fn=_beta_get_al_alloy_prop,
            inputs=[al_alloy_Al, al_alloy_Mg, al_alloy_Si, al_alloy_Ca, al_alloy_Cr, al_alloy_Mn, al_alloy_Fe,
                    al_alloy_Ni, al_alloy_Cu, al_alloy_Zn, al_alloy_Ce, al_alloy_proc],
            outputs=[al_alloy_te, al_alloy_tc, al_alloy_ys, al_alloy_ts]
        )
        for al_alloy_elem in [al_alloy_Mg, al_alloy_Si, al_alloy_Ca, al_alloy_Cr, al_alloy_Mn, al_alloy_Fe, al_alloy_Ni,
                              al_alloy_Cu, al_alloy_Zn, al_alloy_Ce]:
            al_alloy_elem.change(
                fn=lambda *args: 100 - sum(args),
                inputs=[al_alloy_Mg, al_alloy_Si, al_alloy_Ca, al_alloy_Cr, al_alloy_Mn, al_alloy_Fe, al_alloy_Ni,
                        al_alloy_Cu, al_alloy_Zn, al_alloy_Ce],
                outputs=al_alloy_Al
            )
    index.launch()

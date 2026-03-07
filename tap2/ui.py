import gradio as gr
import click
import numpy as np
import ncn

from advanced_module import predict_titanium_yield_strength
from tap2.core import *
from tap2.utils import *
from pathlib import Path
from gradio.utils import NamedString
from art import text2art
from rich import print as rprint


_ta_infer_ = TAInfer()


_aa_infer_ = AAInfer()


def _get_ta_phys_props_(*args):
    """
    Gradio interface: Obtain the physical properties of titanium alloys
    based on their elemental composition and processing route.
    """
    # Unpacking
    Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, HTT = args
    # Check if there is a null value
    if any(arg is None for arg in args):
        gr.Warning('请输入有效的数值！')
        return [None] * 10
    # Calculation
    vals = []
    for prop in (TAPropAbbr.TE, TAPropAbbr.DS, TAPropAbbr.TC, TAPropAbbr.EC, TAPropAbbr.YM,
                 TAPropAbbr.BM, TAPropAbbr.SM, TAPropAbbr.PR, TAPropAbbr.SE, TAPropAbbr.SHC):
        ta_input = TAInput(prop=prop, Ti=Ti, H=H, B=B, C=C, N=N, O=O, Al=Al, Si=Si, Cr=Cr,
                           Fe=Fe, Ni=Ni, Cu=Cu, Zr=Zr, Nb=Nb, Mo=Mo, V=V, Sn=Sn, HTT=HTT)
        ta_output = _ta_infer_(ta_input)
        vals.append(ta_output.value)
    # Correction unit
    vals[0] *= 1e6 # TE
    vals[3] *= 1e-6 # EC
    return vals


def _get_ta_mech_props_(*args) -> list[float | None]:
    """
    Gradio interface: Obtain the mechanical properties of titanium alloy
    according to its elemental composition, processing route, and grain size.
    """
    # Unpacking
    proc_mode, Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, \
        HTT, GS, proc_temp, proc_deform, ht_temp, ht_time, ht_cool = args
    # Check if there is a null value
    if any(_ is None for _ in args[:18]):
        gr.Warning('请输入有效的数值！')
        return [None] * 4
    if proc_mode == 'simple':
        # Simple mode
        if HTT is None or GS is None:
            gr.Warning('请输入有效的数值！')
            return [None] * 4
        vals = []
        for prop in (TAPropAbbr.YS, TAPropAbbr.TS, TAPropAbbr.HD, TAPropAbbr.HP):
            ta_input = TAInput(prop=prop, Ti=Ti, H=H, B=B, C=C, N=N, O=O, Al=Al, Si=Si, Cr=Cr,
                               Fe=Fe, Ni=Ni, Cu=Cu, Zr=Zr, Nb=Nb, Mo=Mo, V=V, Sn=Sn, HTT=HTT, GS=GS)
            ta_output = _ta_infer_(ta_input)
            vals.append(ta_output.value)
        return vals
    elif args[0] == 'advanced':
        # Advanced mode
        if any(_ is None for _ in args[-5:]):
            gr.Warning('请输入有效的数值！')
        vals = []
        for prop in (TAPropAbbr.YS, TAPropAbbr.TS, TAPropAbbr.HD, TAPropAbbr.HP):
            ta_input = TAInput(prop=prop, Ti=Ti, H=H, B=B, C=C, N=N, O=O, Al=Al, Si=Si, Cr=Cr,
                               Fe=Fe, Ni=Ni, Cu=Cu, Zr=Zr, Nb=Nb, Mo=Mo, V=V, Sn=Sn, HTT=ht_temp, GS=10)
            ta_output = _ta_infer_(ta_input)
            vals.append(ta_output.value)
        trace_elements = {}
        if H > 0:
            trace_elements['H'] = H
        if B > 0:
            trace_elements['B'] = B
        if C > 0:
            trace_elements['C'] = C
        if N > 0:
            trace_elements['N'] = N
        if O > 0:
            trace_elements['O'] = O
        if len(trace_elements) == 0:
            trace_elements = None
        advanced_ys = predict_titanium_yield_strength(
            composition=ncn.name(hyphen=True, Ti='balance', Al=Al, Si=Si, Cr=Cr, Fe=Fe,
                                 Ni=Ni, Cu=Cu, Zr=Zr, Nb=Nb, Mo=Mo, V=V, Sn=Sn),
            processing_params={'temperature': proc_temp, 'deformation': proc_deform},
            heat_treatment_params={'temperature': ht_temp, 'time': ht_time, 'cooling': ht_cool},
            trace_elements=trace_elements
        )['yield_strength']
        advanced_ts = advanced_ys + (vals[1] - vals[0])
        vals[0] = advanced_ys
        vals[1] = advanced_ts
        return vals
    else:
        return [None] * 4


def _batch_get_ta_prop_(prop_names: list[str], input_files: list[NamedString] | None) -> list[str]:
    """
    Gradio interface: Obtain the properties of titanium alloys in batches
    based on their elemental composition, processing route, and grain size,
    and support CSV and XLSX file formats.
    """
    _zh_abbr_map_ = {
        "热膨胀系数": TAPropAbbr.TE,
        "密度": TAPropAbbr.DS,
        "热导率": TAPropAbbr.TC,
        "电导率": TAPropAbbr.EC,
        "杨氏模量": TAPropAbbr.YM,
        "体积模量": TAPropAbbr.BM,
        "剪切模量": TAPropAbbr.SM,
        "泊松比": TAPropAbbr.PR,
        "比焓": TAPropAbbr.SE,
        "比热容": TAPropAbbr.SHC,
        "屈服强度": TAPropAbbr.YS,
        "抗拉强度": TAPropAbbr.TS,
        "硬度": TAPropAbbr.HD,
        "霍尔佩奇系数": TAPropAbbr.HP
    }
    _zh_unit_map_ = {
        "热膨胀系数": "10^-6/K",
        "密度": "g/cm^3",
        "热导率": "W/m·K",
        "电导率": "10^6 S/m",
        "杨氏模量": "GPa",
        "体积模量": "GPa",
        "剪切模量": "GPa",
        "泊松比": None,
        "比焓": "J/g",
        "比热容": "J/g·K",
        "屈服强度": "MPa",
        "抗拉强度": "MPa",
        "硬度": "VPN",
        "霍尔佩奇系数": "MPa·m^(1/2)"
    }
    prop_abbrs = [_zh_abbr_map_[prop_name] for prop_name in prop_names]
    append_header = [f"{prop_name} ({_zh_unit_map_[prop_name]})"
                     if _zh_unit_map_[prop_name] is not None else prop_name
                     for prop_name in prop_names]
    # 检查：至少选择一种性能
    if len(prop_names) == 0:
        gr.Warning("请至少选择一个性能！")
        return []
    # 检查：至少上传一个文件
    if input_files is None:
        gr.Warning("请至少上传一个文件！")
        return []
    output_paths = []
    for input_path in input_files:
        input_path = Path(input_path)
        if input_path.match("*.csv"):
            reader, writer = read_inputs_from_csv, write_outputs_to_csv
        elif input_path.match("*.xlsx"):
            reader, writer = read_inputs_from_xlsx, write_outputs_to_xlsx
        else:
            gr.Warning(f"不支持的文件格式：\"{input_path.name}\"！")
            continue
        dict_input = reader(input_path)
        input_size = min(len(_) for _ in dict_input.values())
        if input_size < 1:
            gr.Warning(f"空的 CSV 文件：\"{input_path.name}\"！")
            continue
        outputs_col = []
        for prop_abbr in prop_abbrs:
            ta_input = TABatchInput(prop=prop_abbr, **dict_input)
            ta_output = _ta_infer_(ta_input)
            # 单位修正
            if prop_abbr is TAPropAbbr.TE:
                outputs = [_ * 1e6 for _ in ta_output.value]
            elif prop_abbr is TAPropAbbr.EC:
                outputs = [_ * 1e-6 for _ in ta_output.value]
            else:
                outputs = ta_output.value
            outputs_col.append(outputs)
        # 转置
        outputs_col = np.array(outputs_col)
        outputs_row = outputs_col.T
        outputs_row = outputs_row.tolist()
        output_path = str(writer(input_path, append_header, outputs_row))
        output_paths.append(output_path)
    return output_paths


def _get_ta_wf_(*args) -> list[float | None]:
    """
    Gradio interface: Obtain the phase ratio of titanium alloys
    based on their elemental composition and processing route.
    """
    # Unpacking
    Ti, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, HTT = args
    # Check if there is a null value
    if any(arg is None for arg in args):
        gr.Warning('请输入有效的数值！')
        return [None] * 12
    # Calculation
    ta_input = TAInput(prop=TAPropAbbr.WF, Ti=Ti, Al=Al, Si=Si, Cr=Cr, Fe=Fe, Ni=Ni,
                       Cu=Cu, Zr=Zr, Nb=Nb, Mo=Mo, V=V, Sn=Sn, HTT=HTT)
    ta_output = _ta_infer_(ta_input)
    return [ta_output.value.ALPHA, ta_output.value.BETA, ta_output.value.LAVES, ta_output.value.TI3AL,
            ta_output.value.TI2CU, ta_output.value.TI5SI3, ta_output.value.TIZRSI, ta_output.value.TI2NI,
            ta_output.value.TIM_B2, ta_output.value.LIQUID, ta_output.value.C15_FCC, ta_output.value.MC]


def _get_ta_btt_(*args) -> float | None:
    """
    Gradio interface: Obtain the beta transus temperature of titanium alloys based on their elemental composition.
    """
    # Unpacking
    Ti, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn = args
    # Check if there is a null value
    if any(arg is None for arg in args):
        gr.Warning('请输入有效的数值！')
        return None
    # Calculation
    ta_input = TAInput(prop=TAPropAbbr.BTT, Ti=Ti, Al=Al, Si=Si, Cr=Cr, Fe=Fe,
                       Ni=Ni, Cu=Cu, Zr=Zr, Nb=Nb, Mo=Mo, V=V, Sn=Sn)
    ta_output = _ta_infer_(ta_input)
    btt_value = ta_output.value
    return btt_value


def _get_aa_phys_props_(*args) -> list[float | None]:
    """
    Gradio interface: Obtain the physical properties of aluminum alloys
    based on their elemental composition and processing route.
    """
    # Unpacking
    Al, Li, Mg, Si, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Sn, La, HTT = args
    # Check if there is a null value
    if any(arg is None for arg in args):
        gr.Warning('请输入有效的数值！')
        return [None] * 2
    # Calculation
    vals = []
    for prop in (AAPropAbbr.TE, AAPropAbbr.TC):
        aa_input = AAInput(prop=prop, Al=Al, Li=Li, Mg=Mg, Si=Si, Ca=Ca, Sc=Sc, Ti=Ti, V=V, Cr=Cr,
                           Mn=Mn, Fe=Fe, Co=Co, Ni=Ni, Cu=Cu, Zn=Zn, Zr=Zr, Sn=Sn, La=La, HTT=HTT)
        aa_output = _aa_infer_(aa_input)
        vals.append(aa_output.value)
    return vals


def _build_ta_phys_tab_():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 元素组成")
            with gr.Row():
                Ti = gr.Number(label=f"Ti (wt%)", value=100, minimum=0, maximum=100, interactive=False)
                H = gr.Number(label=f"H (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                B = gr.Number(label=f"B (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                C = gr.Number(label=f"C (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                N = gr.Number(label=f"N (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                O = gr.Number(label=f"O (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Al = gr.Number(label=f"Al (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Si = gr.Number(label=f"Si (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Cr = gr.Number(label=f"Cr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Fe = gr.Number(label=f"Fe (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Ni = gr.Number(label=f"Ni (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Cu = gr.Number(label=f"Cu (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Zr = gr.Number(label=f"Zr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Nb = gr.Number(label=f"Nb (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Mo = gr.Number(label=f"Mo (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                V = gr.Number(label=f"V (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Sn = gr.Number(label=f"Sn (wt%)", value=0, minimum=0, maximum=100, interactive=True)
            gr.Markdown("### 工艺参数")
            HTT = gr.Number(label="热处理温度 (℃)", value=600, minimum=-273.15, interactive=True)
            with gr.Row():
                clear_btn = gr.Button("重置", interactive=True)
                run_btn = gr.Button("提交", interactive=True)
        with gr.Column():
            gr.Markdown("### 物理性能")
            TE = gr.Number(label="热膨胀系数 (10^-6/K)", value=0, interactive=False, precision=3)
            DS = gr.Number(label="密度 (g/cm^3)", value=0, interactive=False, precision=3)
            TC = gr.Number(label="热导率 (W/m·K)", value=0, interactive=False, precision=3)
            EC = gr.Number(label="电导率 (10^6 S/m)", value=0, interactive=False, precision=3)
            YM = gr.Number(label="杨氏模量 (GPa)", value=0, interactive=False, precision=3)
            BM = gr.Number(label="体积模量 (GPa)", value=0, interactive=False, precision=3)
            SM = gr.Number(label="剪切模量 (GPa)", value=0, interactive=False, precision=3)
            PR = gr.Number(label="泊松比", value=0, interactive=False, precision=3)
            SE = gr.Number(label="比焓 (J/g)", value=0, interactive=False, precision=3)
            SHC = gr.Number(label="比热容 (J/g·K)", value=0, interactive=False, precision=3)
    clear_btn.click(
        fn=lambda: [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        outputs=[Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, HTT,
                 TE, DS, TC, EC, YM, BM, SM, PR, SE, SHC]
    )
    run_btn.click(
        fn=_get_ta_phys_props_,
        inputs=[Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, HTT],
        outputs=[TE, DS, TC, EC, YM, BM, SM, PR, SE, SHC]
    )
    for elem_num in [H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn]:
        elem_num.change(
            fn=get_balance_comp,
            inputs=[H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn],
            outputs=Ti
        )


def _build_ta_mech_tab_():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 元素组成")
            with gr.Row():
                Ti = gr.Number(label=f"Ti (wt%)", value=100, minimum=0, maximum=100, interactive=False)
                H = gr.Number(label=f"H (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                B = gr.Number(label=f"B (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                C = gr.Number(label=f"C (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                N = gr.Number(label=f"N (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                O = gr.Number(label=f"O (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Al = gr.Number(label=f"Al (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Si = gr.Number(label=f"Si (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Cr = gr.Number(label=f"Cr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Fe = gr.Number(label=f"Fe (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Ni = gr.Number(label=f"Ni (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Cu = gr.Number(label=f"Cu (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Zr = gr.Number(label=f"Zr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Nb = gr.Number(label=f"Nb (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Mo = gr.Number(label=f"Mo (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                V = gr.Number(label=f"V (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Sn = gr.Number(label=f"Sn (wt%)", value=0, minimum=0, maximum=100, interactive=True)
            gr.Markdown("### 工艺参数")
            proc_mode = gr.State(value="simple")
            with gr.Tabs():
                with gr.Tab("快捷模式", id="simple") as simple_mode:
                    with gr.Row():
                        HTT = gr.Number(label="热处理温度 (℃)", value=600, interactive=True)
                        GS = gr.Number(label="晶粒尺寸 (μm)", value=10, interactive=True)
                with gr.Tab("高级模式", id="advanced") as advanced_mode:
                    gr.Markdown("#### 加工工艺")
                    with gr.Row():
                        proc_temp = gr.Number(label="温度（℃）", value=860, interactive=True)
                        proc_deform = gr.Number(label="变形量（%）", value=80, interactive=True)
                    gr.Markdown("#### 热处理工艺")
                    with gr.Row():
                        ht_temp = gr.Number(label="温度（℃）", value=760, interactive=True)
                        ht_time = gr.Number(label="时间（min）", value=30, interactive=True)
                        ht_cool = gr.Dropdown(label="冷却方式", choices=["炉冷", "空冷", "水淬"], value="空冷",
                                              interactive=True)
            with gr.Row():
                clear_btn = gr.Button("重置", interactive=True)
                run_btn = gr.Button("提交", interactive=True)
        with gr.Column():
            gr.Markdown("### 力学性能")
            YS = gr.Number(label="屈服强度 (MPa)", value=0, interactive=False, precision=3)
            TS = gr.Number(label="抗拉强度 (MPa)", value=0, interactive=False, precision=3)
            HD = gr.Number(label="硬度 (VPN)", value=0, interactive=False, precision=3)
            HP = gr.Number(label="霍尔佩奇系数 (MPa·m^(1/2))", value=0, interactive=False, precision=3)
    simple_mode.select(fn=lambda: "simple", outputs=proc_mode)
    advanced_mode.select(fn=lambda: "advanced", outputs=proc_mode)
    clear_btn.click(
        fn=lambda: [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600, 10, 860, 80, 760, 30, "空冷"],
        outputs=[Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, HTT, GS, proc_temp, proc_deform,
                 ht_temp, ht_time, ht_cool, YS, TS, HD, HP]
    )
    run_btn.click(
        fn=_get_ta_mech_props_,
        inputs=[proc_mode, Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, HTT, GS,
                proc_temp, proc_deform, ht_temp, ht_time, ht_cool],
        outputs=[YS, TS, HD, HP]
    )
    for elem in [H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn]:
        elem.change(
            fn=get_balance_comp,
            inputs=[H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn],
            outputs=Ti
        )


def _build_ta_wf_tab_():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 元素组成")
            with gr.Row():
                Ti = gr.Number(label=f"Ti (wt%)", value=100, minimum=0, maximum=100, interactive=False)
                Al = gr.Number(label=f"Al (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Si = gr.Number(label=f"Si (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Cr = gr.Number(label=f"Cr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Fe = gr.Number(label=f"Fe (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Ni = gr.Number(label=f"Ni (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Cu = gr.Number(label=f"Cu (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Zr = gr.Number(label=f"Zr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Nb = gr.Number(label=f"Nb (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Mo = gr.Number(label=f"Mo (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                V = gr.Number(label=f"V (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Sn = gr.Number(label=f"Sn (wt%)", value=0, minimum=0, maximum=100, interactive=True)
            gr.Markdown("### 工艺参数")
            HTT = gr.Number(label="热处理温度 (℃)", value=600, minimum=-273.15, interactive=True)
            with gr.Row():
                clear_btn = gr.Button("重置", interactive=True)
                run_btn = gr.Button("提交", interactive=True)
        with gr.Column():
            gr.Markdown("### 相比例")
            with gr.Row():
                ALPHA = gr.Number(label="ALPHA (wt%)", value=0, interactive=False, precision=3)
                BETA = gr.Number(label="BETA (wt%)", value=0, interactive=False, precision=3)
                LAVES = gr.Number(label="LAVES (wt%)", value=0, interactive=False, precision=3)
                TI3AL = gr.Number(label="TI3AL (wt%)", value=0, interactive=False, precision=3)
                TI2CU = gr.Number(label="TI2CU (wt%)", value=0, interactive=False, precision=3)
                TI5SI3 = gr.Number(label="TI5SI3 (wt%)", value=0, interactive=False, precision=3)
                TIZRSI = gr.Number(label="TIZRSI (wt%)", value=0, interactive=False, precision=3)
                TI2NI = gr.Number(label="TI2NI (wt%)", value=0, interactive=False, precision=3)
                TIM_B2 = gr.Number(label="TIM_B2 (wt%)", value=0, interactive=False, precision=3)
                LIQUID = gr.Number(label="LIQUID (wt%)", value=0, interactive=False, precision=3)
                C15_FCC = gr.Number(label="C15_FCC (wt%)", value=0, interactive=False, precision=3)
                MC = gr.Number(label="MC (wt%)", value=0, interactive=False, precision=3)
    clear_btn.click(
        fn=lambda: [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        outputs=[Ti, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, HTT,
                 ALPHA, BETA, LAVES, TI3AL, TI2CU, TI5SI3, TIZRSI, TI2NI, TIM_B2, LIQUID, C15_FCC, MC]
    )
    run_btn.click(
        fn=_get_ta_wf_,
        inputs=[Ti, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, HTT],
        outputs=[ALPHA, BETA, LAVES, TI3AL, TI2CU, TI5SI3, TIZRSI, TI2NI, TIM_B2, LIQUID, C15_FCC, MC]
    )
    for elem in [Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn]:
        elem.change(
            fn=get_balance_comp,
            inputs=[Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn],
            outputs=Ti
        )


def _build_ta_btt_tab_():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 元素组成")
            with gr.Row():
                Ti = gr.Number(label=f"Ti (wt%)", value=100, minimum=0, maximum=100, interactive=False)
                Al = gr.Number(label=f"Al (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Si = gr.Number(label=f"Si (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Cr = gr.Number(label=f"Cr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Fe = gr.Number(label=f"Fe (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Ni = gr.Number(label=f"Ni (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Cu = gr.Number(label=f"Cu (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Zr = gr.Number(label=f"Zr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Nb = gr.Number(label=f"Nb (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Mo = gr.Number(label=f"Mo (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                V = gr.Number(label=f"V (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Sn = gr.Number(label=f"Sn (wt%)", value=0, minimum=0, maximum=100, interactive=True)
            with gr.Row():
                clear_btn = gr.Button("重置", interactive=True)
                run_btn = gr.Button("提交", interactive=True)
        with gr.Column():
            gr.Markdown("### β-转变温度")
            BTT = gr.Number(label="β-转变温度（℃）", value=0, interactive=False, precision=3)
    clear_btn.click(
        fn=lambda: [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        outputs=[Ti, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, BTT]
    )
    run_btn.click(
        fn=_get_ta_btt_,
        inputs=[Ti, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn],
        outputs=BTT
    )
    for elem in [Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn]:
        elem.change(
            fn=get_balance_comp,
            inputs=[Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn],
            outputs=Ti
        )


def _build_ta_batch_tab_():
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
            gr.Markdown('### 批处理模板文件下载')
            gr.File(value=[str(resources.files("tap2.resource").joinpath("tapp_template.csv")),
                           str(resources.files("tap2.resource").joinpath("tapp_template.xlsx"))],
                    label="下载批处理模板文件",
                    file_count="multiple",
                    type="filepath",
                    interactive=False)
        with gr.Column():
            down_files = gr.File(label="下载 CSV 或 XLSX 文件",
                                 file_types=[".csv", ".xlsx"],
                                 file_count="multiple",
                                 type="filepath",
                                 interactive=False)
    # 定义交互逻辑
    batch_run_btn.click(
        fn=_batch_get_ta_prop_,
        inputs=[checkboxes, up_files],
        outputs=down_files
    )
    batch_all_btn.click(
        fn=lambda sel_props: list(
            {"热膨胀系数", "密度", "热导率", "电导率", "杨氏模量", "体积模量", "剪切模量", "泊松比",
             "比焓", "比热容", "屈服强度", "抗拉强度", "硬度", "霍尔佩奇系数"} - set(sel_props)),
        inputs=checkboxes,
        outputs=checkboxes
    )
    batch_clear_btn.click(
        fn=lambda: [None, None, None],
        outputs=[up_files, down_files, checkboxes]
    )


def _build_ta_tab_():
    with gr.Tabs():
        with gr.Tab("物理性能"):
            _build_ta_phys_tab_()
        with gr.Tab("力学性能"):
            _build_ta_mech_tab_()
        with gr.Tab("相比例"):
            _build_ta_wf_tab_()
        with gr.Tab("β-转变温度"):
            _build_ta_btt_tab_()
        with gr.Tab("批量模式"):
            _build_ta_batch_tab_()


def _build_aa_phys_tab_():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 元素组成")
            with gr.Row():
                Al = gr.Number(label=f"Al (wt%)", value=100, minimum=0, maximum=100, interactive=False)
                Li = gr.Number(label=f"Li (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Mg = gr.Number(label=f"Mg (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Si = gr.Number(label=f"Si (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Ca = gr.Number(label=f"Ca (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Sc = gr.Number(label=f"Sc (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Ti = gr.Number(label=f"Ti (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                V = gr.Number(label=f"V (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Cr = gr.Number(label=f"Cr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Mn = gr.Number(label=f"Mn (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Fe = gr.Number(label=f"Fe (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Co = gr.Number(label=f"Co (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Ni = gr.Number(label=f"Ni (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Cu = gr.Number(label=f"Cu (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Zn = gr.Number(label=f"Zn (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Zr = gr.Number(label=f"Zr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Sn = gr.Number(label=f"Sn (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                La = gr.Number(label=f"La (wt%)", value=0, minimum=0, maximum=100, interactive=True)
            gr.Markdown("### 工艺参数")
            HTT = gr.Number(label="热处理温度 (℃)", value=500, minimum=-273.15, interactive=True)
            with gr.Row():
                clear_btn = gr.Button("重置", interactive=True)
                run_btn = gr.Button("提交", interactive=True)
        with gr.Column():
            gr.Markdown("### 物理性能")
            TE = gr.Number(label="热膨胀系数 (10^-6/K)", value=0, interactive=False, precision=3)
            TC = gr.Number(label="热导率 (W/m·K)", value=0, interactive=False, precision=3)
    clear_btn.click(
        fn=lambda: [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600, 0, 0],
        outputs=[Al, Li, Mg, Si, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Sn, La, HTT, TE, TC]
    )
    run_btn.click(
        fn=_get_aa_phys_props_,
        inputs=[Al, Li, Mg, Si, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Sn, La, HTT],
        outputs=[TE, TC]
    )
    for elem in [Li, Mg, Si, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Sn, La]:
        elem.change(
            fn=get_balance_comp,
            inputs=[Li, Mg, Si, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Sn, La],
            outputs=Al
        )


def _build_aa_mech_tab_():
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 元素组成")
            with gr.Row():
                Al = gr.Number(label=f"Al (wt%)", value=100, minimum=0, maximum=100, interactive=False)
                Li = gr.Number(label=f"Li (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Mg = gr.Number(label=f"Mg (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Si = gr.Number(label=f"Si (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Ca = gr.Number(label=f"Ca (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Sc = gr.Number(label=f"Sc (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Ti = gr.Number(label=f"Ti (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                V = gr.Number(label=f"V (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Cr = gr.Number(label=f"Cr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Mn = gr.Number(label=f"Mn (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Fe = gr.Number(label=f"Fe (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Co = gr.Number(label=f"Co (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Ni = gr.Number(label=f"Ni (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Cu = gr.Number(label=f"Cu (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Zn = gr.Number(label=f"Zn (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Zr = gr.Number(label=f"Zr (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                Sn = gr.Number(label=f"Sn (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                La = gr.Number(label=f"La (wt%)", value=0, minimum=0, maximum=100, interactive=True)
            gr.Markdown("### 工艺参数")
            HTT = gr.Number(label="热处理温度 (℃)", value=500, minimum=-273.15, interactive=True)
            with gr.Row():
                clear_btn = gr.Button("重置", interactive=True)
                run_btn = gr.Button("提交", interactive=True)
        with gr.Column():
            gr.Markdown("### 力学性能")
            YS = gr.Number(label="屈服强度 (MPa)", value=0, interactive=False, precision=3)
            TS = gr.Number(label="抗拉强度 (MPa)", value=0, interactive=False, precision=3)
    clear_btn.click(
        fn=lambda: [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600, 0, 0],
        outputs=[Al, Li, Mg, Si, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Sn, La, HTT, YS, TS]
    )
    for elem in [Li, Mg, Si, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Sn, La]:
        elem.change(
            fn=get_balance_comp,
            inputs=[Li, Mg, Si, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Sn, La],
            outputs=Al
        )


def _build_aa_tab_():
    with gr.Tab('物理性能'):
        _build_aa_phys_tab_()
    with gr.Tab('力学性能'):
        _build_aa_mech_tab_()


@click.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=7860)
def run_gradio(host, port):
    # Print the configuration
    print(text2art('TAP2'), end='')
    print(f'Device: {_ta_infer_.device}')
    print(f'Batch Size: {_ta_infer_.batch_size}')
    rprint('Silence: [green]YES[/green]' if _ta_infer_.silence else 'Silence: [red]NO[/red]')
    rprint('Correction: [red]NO[/red]' if _ta_infer_.skip else 'Correction: [green]YES[/green]')
    with gr.Blocks(title="TAP2") as index:
        with gr.Tabs():
            with gr.Tab('钛合金'):
                _build_ta_tab_()
            with gr.Tab('铝合金'):
                _build_aa_tab_()
    index.queue(max_size=32, default_concurrency_limit=4).launch(server_name=host, server_port=port, share=False)

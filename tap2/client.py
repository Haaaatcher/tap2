import gradio as gr
import ncn
from strengthml import predict_titanium_yield_strength_enhanced
from tap2.utils import *
from tap2.server import *
from art import text2art



class TAPP:
    def __init__(self, device: str='cpu', max_batch_size: int=64):
        self.device = device
        self.max_batch_size = max_batch_size
        self.route_api_map = {
            '/titanium-alloy/physical/density': TADSLitAPI(),
            '/titanium-alloy/physical/thermal-conductivity': TATCLitAPI(),
            '/titanium-alloy/physical/electrical-conductivity': TAECLitAPI(),
            '/titanium-alloy/physical/youngs-modulus': TAYMLitAPI(),
            '/titanium-alloy/physical/bulk-modulus': TABMLitAPI(),
            '/titanium-alloy/physical/shear-modulus': TASMLitAPI(),
            '/titanium-alloy/physical/poisson-ratio': TAPRLitAPI(),
            '/titanium-alloy/physical/specific-enthalpy': TASELitAPI(),
            '/titanium-alloy/physical/specific-heat-capacity': TASHCLitAPI(),
            '/titanium-alloy/physical/thermal-expansivity': TATELitAPI(),
            '/titanium-alloy/physical/beta-transus-temperature': TABTTLitAPI(),
            '/titanium-alloy/mechanical/yield-stress': TAYSLitAPI(),
            '/titanium-alloy/mechanical/tensile-stress': TATSLitAPI(),
            '/titanium-alloy/mechanical/hardness': TAHDLitAPI(),
            '/titanium-alloy/mechanical/hall-petch-coefficient': TAHPLitAPI(),
            '/titanium-alloy/weight-fraction': TAWFLitAPI(),
            '/aluminum-alloy/physical/thermal-conductivity': AATCLitAPI(),
            '/aluminum-alloy/physical/thermal-expansivity': AATELitAPI(),
            '/aluminum-alloy/mechanical/yield-stress': AAYSLitAPI()
        }
        for lit_api in self.route_api_map.values():
            lit_api._device = device
            lit_api.setup(device)
        self.route_base_model_map = {
            '/titanium-alloy/physical/density': TAPhysBaseModel,
            '/titanium-alloy/physical/thermal-conductivity': TAPhysBaseModel,
            '/titanium-alloy/physical/electrical-conductivity': TAPhysBaseModel,
            '/titanium-alloy/physical/youngs-modulus': TAPhysBaseModel,
            '/titanium-alloy/physical/bulk-modulus': TAPhysBaseModel,
            '/titanium-alloy/physical/shear-modulus': TAPhysBaseModel,
            '/titanium-alloy/physical/poisson-ratio': TAPhysBaseModel,
            '/titanium-alloy/physical/specific-enthalpy': TAPhysBaseModel,
            '/titanium-alloy/physical/specific-heat-capacity': TAPhysBaseModel,
            '/titanium-alloy/physical/thermal-expansivity': TAPhysBaseModel,
            '/titanium-alloy/physical/beta-transus-temperature': TAPhysBaseModel,
            '/titanium-alloy/mechanical/yield-stress': TAMechBaseModel,
            '/titanium-alloy/mechanical/tensile-stress': TAMechBaseModel,
            '/titanium-alloy/mechanical/hardness': TAMechBaseModel,
            '/titanium-alloy/mechanical/hall-petch-coefficient': TAMechBaseModel,
            '/titanium-alloy/weight-fraction': TAWFBaseModel,
            '/aluminum-alloy/physical/thermal-conductivity': AAPhysBaseModel,
            '/aluminum-alloy/physical/thermal-expansivity': AAPhysBaseModel,
            '/aluminum-alloy/mechanical/yield-stress': AAMechBaseModel
        }

    def __call__(self, route: str, requests: list[dict]):
        if route in self.route_api_map:
            api = self.route_api_map[route]
            base_model = self.route_base_model_map[route]
            responses = []
            for i in range(0, len(requests), self.max_batch_size):
                batch_requests = requests[i: i + self.max_batch_size]
                batch_inputs = [api.decode_request(base_model(**r)) for r in batch_requests]
                batch_inputs = api.batch(batch_inputs)
                batch_outputs = api.predict(batch_inputs)
                batch_responses = [api.encode_response(o) for o in batch_outputs]
                responses.extend(batch_responses)
            return responses
        else:
            logger.warning(f'Unknown route {route}')
            return None


tapp: TAPP | None = None


def get_ta_phys_props(Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, HTT):
    global tapp
    if any(v is None for v in (Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, HTT)):
        gr.Warning('请输入有效的数值！')
        return [None] * 11
    prop_vals = []
    for route in (
        '/titanium-alloy/physical/thermal-expansivity',
        '/titanium-alloy/physical/density',
        '/titanium-alloy/physical/thermal-conductivity',
        '/titanium-alloy/physical/electrical-conductivity',
        '/titanium-alloy/physical/youngs-modulus',
        '/titanium-alloy/physical/bulk-modulus',
        '/titanium-alloy/physical/shear-modulus',
        '/titanium-alloy/physical/poisson-ratio',
        '/titanium-alloy/physical/specific-enthalpy',
        '/titanium-alloy/physical/specific-heat-capacity',
        '/titanium-alloy/physical/beta-transus-temperature'
    ):
        requests = [{
            'Ti': Ti, 'H': H, 'B': B, 'C': C, 'N': N, 'O': O, 'Al': Al, 'Si': Si, 'Cr': Cr, 'Fe': Fe, 'Ni': Ni,
            'Cu': Cu, 'Zr': Zr, 'Nb': Nb, 'Mo': Mo, 'V': V, 'Sn': Sn, 'HTT': HTT
        }]
        responses = tapp(route, requests)
        assert responses is not None
        prop_vals.append(responses[0]['value'])
    return prop_vals


def simple_get_ta_mech_props(Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, HTT, GS):
    global tapp
    if any(v is None for v in (Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, HTT, GS)):
        gr.Warning('请输入有效的数值！')
        return [None] * 6
    prop_vals = []
    for route in (
        '/titanium-alloy/mechanical/yield-stress',
        '/titanium-alloy/mechanical/tensile-stress',
        '/titanium-alloy/mechanical/hardness',
        '/titanium-alloy/mechanical/hall-petch-coefficient'
    ):
        requests = [{
            'Ti': Ti, 'H': H, 'B': B, 'C': C, 'N': N, 'O': O, 'Al': Al, 'Si': Si, 'Cr': Cr, 'Fe': Fe, 'Ni': Ni,
            'Cu': Cu, 'Zr': Zr, 'Nb': Nb, 'Mo': Mo, 'V': V, 'Sn': Sn, 'HTT': HTT, 'GS': GS
        }]
        responses = tapp(route, requests)
        assert responses is not None
        prop_vals.append(responses[0]['value'])
    prop_vals.append(plot_gs_stress_curve(prop_vals[3], GS, prop_vals[0], '屈服强度'))
    prop_vals.append(plot_gs_stress_curve(prop_vals[3], GS, prop_vals[1], '抗拉强度'))
    return prop_vals


def advanced_get_ta_mech_props(Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, proc_temperature,
                               proc_deformation, ht_checkbox_1, ht_temperature_1, ht_time_1, ht_cooling_method_1,
                               ht_checkbox_2, ht_temperature_2, ht_time_2, ht_cooling_method_2):
    global tapp
    cooling_zh2en_map = {
        '炉冷': 'FC',
        '空冷': 'AC',
        '水淬': 'WQ'
    }
    if any(v is None
        for v in (Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, proc_temperature, proc_deformation)
    ):
        gr.Warning('请输入有效的数值！')
        return [None] * 6
    if ht_checkbox_1 and any(v is None for v in (ht_temperature_1, ht_time_1, ht_cooling_method_1)):
        gr.Warning('请输入有效的数值！')
        return [None] * 6
    if ht_checkbox_2 and any(v is None for v in (ht_temperature_2, ht_time_2, ht_cooling_method_2)):
        gr.Warning('请输入有效的数值！')
        return [None] * 6
    if not ht_checkbox_1 and not ht_checkbox_2:
        gr.Warning('请输入有效的数值！')
        return [None] * 6
    composition = ncn.name(hyphen=True, Ti='balance', Al=Al, Si=Si, Cr=Cr, Fe=Fe, Ni=Ni, Cu=Cu, Zr=Zr, Nb=Nb, Mo=Mo,
                           V=V, Sn=Sn)
    processing_params = f'{proc_temperature}/{proc_deformation}'
    heat_treatment_params = []
    ht_temperatures = []
    if ht_checkbox_1:
        heat_treatment_params.append(f'{ht_temperature_1}/{ht_time_1}/{cooling_zh2en_map[ht_cooling_method_1]}')
        ht_temperatures.append(ht_temperature_1)
    if ht_checkbox_2:
        heat_treatment_params.append(f'{ht_temperature_2}/{ht_time_2}/{cooling_zh2en_map[ht_cooling_method_2]}')
        ht_temperatures.append(ht_temperature_2)
    max_ht_temperature = max(ht_temperatures)
    heat_treatment_params = '+'.join(heat_treatment_params)
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
    result = predict_titanium_yield_strength_enhanced(
        composition=composition,
        processing_params=processing_params,
        heat_treatment_params=heat_treatment_params,
        trace_elements=trace_elements
    )
    advanced_ys = result['yield_strength']
    response = tapp('/titanium-alloy/mechanical/yield-stress', [{
        'Ti': Ti, 'H': H, 'B': B, 'C': C, 'N': N, 'O': O, 'Al': Al, 'Si': Si, 'Cr': Cr, 'Fe': Fe, 'Ni': Ni, 'Cu': Cu,
        'Zr': Zr, 'Nb': Nb, 'Mo': Mo, 'V': V, 'Sn': Sn, 'HTT': max_ht_temperature, 'GS': 10
    }])
    assert response is not None
    simple_ys = response[0]['value']
    response = tapp('/titanium-alloy/mechanical/tensile-stress', [{
        'Ti': Ti, 'H': H, 'B': B, 'C': C, 'N': N, 'O': O, 'Al': Al, 'Si': Si, 'Cr': Cr, 'Fe': Fe, 'Ni': Ni, 'Cu': Cu,
        'Zr': Zr, 'Nb': Nb, 'Mo': Mo, 'V': V, 'Sn': Sn, 'HTT': max_ht_temperature, 'GS': 10
    }])
    assert response is not None
    simple_ts = response[0]['value']
    ts_ys_diff = simple_ts - simple_ys
    advanced_ts = advanced_ys + ts_ys_diff
    return [advanced_ys, advanced_ts, None, None, None, None]


def batch_get_ta_prop(prop_names, input_files):
    global tapp
    zh_route_map = {
        "热膨胀系数": '/titanium-alloy/physical/thermal-expansivity',
        "密度": '/titanium-alloy/physical/density',
        "热导率": '/titanium-alloy/physical/thermal-conductivity',
        "电导率": '/titanium-alloy/physical/electrical-conductivity',
        "杨氏模量": '/titanium-alloy/physical/youngs-modulus',
        "体积模量": '/titanium-alloy/physical/bulk-modulus',
        "剪切模量": '/titanium-alloy/physical/shear-modulus',
        "泊松比": '/titanium-alloy/physical/poisson-ratio',
        "比焓": '/titanium-alloy/physical/specific-enthalpy',
        "比热容": '/titanium-alloy/physical/specific-heat-capacity',
        'β-转变温度': '/titanium-alloy/physical/beta-transus-temperature',
        "屈服强度": '/titanium-alloy/mechanical/yield-stress',
        "抗拉强度": '/titanium-alloy/mechanical/tensile-stress',
        "硬度": '/titanium-alloy/mechanical/hardness',
        "霍尔佩奇系数": '/titanium-alloy/mechanical/hall-petch-coefficient',
        '相比例': '/titanium-alloy/weight-fraction'
    }
    zh_header_map = {
        "热膨胀系数": ['热膨胀系数 (10^-6/K)'],
        "密度": ['密度 (g/cm^3)'],
        "热导率": ['热导率 (W/m·K)'],
        "电导率": ['电导率 (10^6 S/m)'],
        "杨氏模量": ['杨氏模量 (GPa)'],
        "体积模量": ['体积模量 (GPa)'],
        "剪切模量": ['剪切模量 (GPa)'],
        "泊松比": ['泊松比'],
        "比焓": ['比焓 (J/g)'],
        "比热容": ['比热容 (J/g·K)'],
        'β-转变温度': ['β-转变温度 (℃)'],
        "屈服强度": ['屈服强度 (MPa)'],
        "抗拉强度": ['抗拉强度 (MPa)'],
        "硬度": ['硬度 (VPN)'],
        "霍尔佩奇系数": ['霍尔佩奇系数 (MPa·m^(1/2))'],
        '相比例': ['ALPHA (wt%)', 'BETA (wt%)', 'LAVES (wt%)', 'TI3AL (wt%)', 'TI2CU (wt%)', 'TI5SI3 (wt%)',
                   'TIZRSI (wt%)', 'TI2NI (wt%)', 'TIM_B2 (wt%)', 'LIQUID (wt%)', 'C15_FCC (wt%)', 'MC (wt%)']
    }
    prop_names = sorted(prop_names)
    if len(prop_names) == 0:
        gr.Warning("请至少选择一个性能！")
        return []
    if input_files is None:
        gr.Warning("请至少上传一个文件！")
        return []
    output_files = []
    for input_file in input_files:
        requests = read_input_data_from_csv(input_file)
        if not requests:
            gr.Warning(f"空的 CSV 文件：\"{input_file}\"！")
            continue
        output_data = [{} for _ in range(len(requests))]
        for prop_name in prop_names:
            responses = tapp(zh_route_map[prop_name], requests)
            assert responses is not None
            header = zh_header_map[prop_name]
            if prop_name == '相比例':
                for response, outputs in zip(responses, output_data):
                    for phase in header:
                        outputs[phase] = response['value'][phase[:-6]]
            else:
                for response, outputs in zip(responses, output_data):
                    outputs[header[0]] = response['value']
        output_file = str(write_output_data_to_csv(
            input_csv_path=input_file,
            output_header=[header for zh in prop_names for header in zh_header_map[zh]],
            output_data=output_data
        ))
        output_files.append(output_file)
    return output_files


def get_ta_wf(Ti, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, HTT):
    global tapp
    if any(v is None for v in (Ti, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, HTT)):
        gr.Warning('请输入有效的数值！')
        return [None] * 12
    requests = [{
        'Ti': Ti, 'Al': Al, 'Si': Si, 'Cr': Cr, 'Fe': Fe, 'Ni': Ni, 'Cu': Cu, 'Zr': Zr, 'Nb': Nb, 'Mo': Mo, 'V': V,
        'Sn': Sn, 'HTT': HTT
    }]
    responses = tapp('/titanium-alloy/weight-fraction', requests)
    assert responses is not None
    return [responses[0]['value']['ALPHA'],
            responses[0]['value']['BETA'],
            responses[0]['value']['LAVES'],
            responses[0]['value']['TI3AL'],
            responses[0]['value']['TI2CU'],
            responses[0]['value']['TI5SI3'],
            responses[0]['value']['TIZRSI'],
            responses[0]['value']['TI2NI'],
            responses[0]['value']['TIM_B2'],
            responses[0]['value']['LIQUID'],
            responses[0]['value']['C15_FCC'],
            responses[0]['value']['MC']]


def get_aa_phys_props(Al, Li, Mg, Si, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Sn, La, HTT):
    global tapp
    if any(v is None for v in (Al, Li, Mg, Si, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Sn, La, HTT)):
        gr.Warning('请输入有效的数值！')
        return [None] * 2
    prop_vals = []
    for route in (
        '/aluminum-alloy/physical/thermal-expansivity',
        '/aluminum-alloy/physical/thermal-conductivity'
    ):
        requests = [{
            'Al': Al, 'Li': Li, 'Mg': Mg, 'Si': Si, 'Ca': Ca, 'Sc': Sc, 'Ti': Ti, 'V': V, 'Cr': Cr, 'Mn': Mn, 'Fe': Fe,
            'Co': Co, 'Ni': Ni, 'Cu': Cu, 'Zn': Zn, 'Zr': Zr, 'Sn': Sn, 'La': La, 'HTT': HTT
        }]
        responses = tapp(route, requests)
        assert responses is not None
        prop_vals.append(responses[0]['value'])
    return prop_vals


def get_aa_mech_props(Al, Li, Mg, Si, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Mo, Sn, La, Ce):
    global tapp
    if any(v is None for v in (Al, Li, Mg, Si, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Mo, Sn, La, Ce)):
        gr.Warning('请输入有效的数值！')
        return None
    requests = [{
        'Al': Al, 'Li': Li, 'Mg': Mg, 'Si': Si, 'Ca': Ca, 'Sc': Sc, 'Ti': Ti, 'V': V, 'Cr': Cr, 'Mn': Mn, 'Fe': Fe,
        'Co': Co, 'Ni': Ni, 'Cu': Cu, 'Zn': Zn, 'Zr': Zr, 'Mo': Mo, 'Sn': Sn, 'La': La, 'Ce': Ce
    }]
    responses = tapp('/aluminum-alloy/mechanical/yield-stress', requests)
    assert responses is not None
    return responses[0]['value']


def build_ta_phys_tab():
    with gr.Row():
        with gr.Column():
            with gr.Accordion('元素组成'):
                with gr.Row():
                    Ti = gr.Number(label=f"Ti (wt%)", value=99.655, minimum=0, maximum=100, interactive=False)
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
            with gr.Accordion('微量元素', open=False):
                with gr.Row():
                    H = gr.Number(label=f"H (wt%)", value=0.015, minimum=0, maximum=1, interactive=True)
                    B = gr.Number(label=f"B (wt%)", value=0, minimum=0, maximum=1, interactive=True)
                    C = gr.Number(label=f"C (wt%)", value=0.08, minimum=0, maximum=1, interactive=True)
                    N = gr.Number(label=f"N (wt%)", value=0.05, minimum=0, maximum=1, interactive=True)
                    O = gr.Number(label=f"O (wt%)", value=0.2, minimum=0, maximum=1, interactive=True)
            with gr.Accordion('工艺参数'):
                HTT = gr.Number(label="热处理温度 (℃)", value=600, minimum=200, maximum=1800, interactive=True)
            with gr.Row():
                reset_btn = gr.Button("重置", interactive=True)
                run_btn = gr.Button("提交", interactive=True)
        with gr.Column():
            with gr.Accordion('物理性能'):
                TE = gr.Number(label='热膨胀系数 (10^-6/K)', interactive=False, precision=3)
                DS = gr.Number(label='密度 (g/cm^3)', interactive=False, precision=3)
                TC = gr.Number(label='热导率 (W/m·K)', interactive=False, precision=3)
                EC = gr.Number(label='电导率 (10^6 S/m)', interactive=False, precision=3)
                YM = gr.Number(label='杨氏模量 (GPa)', interactive=False, precision=3)
                BM = gr.Number(label='体积模量 (GPa)', interactive=False, precision=3)
                SM = gr.Number(label='剪切模量 (GPa)', interactive=False, precision=3)
                PR = gr.Number(label='泊松比', interactive=False, precision=3)
                SE = gr.Number(label='比焓 (J/g)', interactive=False, precision=3)
                SHC = gr.Number(label='比热容 (J/g·K)', interactive=False, precision=3)
                BTT = gr.Number(label='β-转变温度 (℃)', interactive=False, precision=3)
    reset_btn.click(
        fn=lambda: [99.655, 0.015, 0, 0.08, 0.05, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600,
                    None, None, None, None, None, None, None, None, None, None, None],
        outputs=[Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, HTT,
                 TE, DS, TC, EC, YM, BM, SM, PR, SE, SHC, BTT]
    )
    run_btn.click(
        fn=get_ta_phys_props,
        inputs=[Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, HTT],
        outputs=[TE, DS, TC, EC, YM, BM, SM, PR, SE, SHC, BTT]
    )
    for comp in [H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn]:
        comp.change(
            fn=lambda *args: 100.0 - sum(0 if arg is None else arg for arg in args),
            inputs=[H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn],
            outputs=Ti
        )


def build_ta_mech_tab():
    with gr.Row():
        with gr.Column():
            with gr.Accordion('元素组成'):
                with gr.Row():
                    Ti = gr.Number(label=f'Ti (wt%)', value=99.655, minimum=0, maximum=100, interactive=False)
                    Al = gr.Number(label=f'Al (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Si = gr.Number(label=f'Si (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Cr = gr.Number(label=f'Cr (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Fe = gr.Number(label=f'Fe (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Ni = gr.Number(label=f'Ni (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Cu = gr.Number(label=f'Cu (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Zr = gr.Number(label=f'Zr (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Nb = gr.Number(label=f'Nb (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Mo = gr.Number(label=f'Mo (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    V = gr.Number(label=f'V (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Sn = gr.Number(label=f'Sn (wt%)', value=0, minimum=0, maximum=100, interactive=True)
            with gr.Accordion('微量元素', open=False):
                with gr.Row():
                    H = gr.Number(label=f'H (wt%)', value=0.015, minimum=0, maximum=100, interactive=True)
                    B = gr.Number(label=f'B (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    C = gr.Number(label=f'C (wt%)', value=0.08, minimum=0, maximum=100, interactive=True)
                    N = gr.Number(label=f'N (wt%)', value=0.05, minimum=0, maximum=100, interactive=True)
                    O = gr.Number(label=f'O (wt%)', value=0.2, minimum=0, maximum=100, interactive=True)
            with gr.Accordion('工艺参数'):
                proc_mode = gr.State(value="simple")
                with gr.Tabs():
                    with gr.Tab("快捷模式", id="simple") as simple_mode_tab:
                        HTT = gr.Number(label="热处理温度 (℃)", value=600, minimum=200, maximum=1800, interactive=True)
                        GS = gr.Number(label="晶粒尺寸 (μm)", value=10, minimum=1, maximum=200, interactive=True)
                    with gr.Tab("高级模式", id="advanced") as advanced_mode_tab:
                        with gr.Accordion('加工工艺'):
                            proc_temperature = gr.Number(label="温度（℃）", value=900, interactive=True)
                            proc_deformation = gr.Number(label="变形量（%）", value=70, interactive=True)
                        with gr.Accordion('热处理工艺'):
                            with gr.Row():
                                ht_checkbox_1 = gr.Checkbox(value=True, label='1')
                                ht_temperature_1 = gr.Number(label='温度 (℃)', value=800, interactive=True)
                                ht_time_1 = gr.Number(label='时间 (min)', value=60, interactive=True)
                                ht_cooling_method_1 = gr.Dropdown(label='冷却方式', choices=['炉冷', '空冷', '水淬'],
                                                                  value='水淬', interactive=True)
                            with gr.Row():
                                ht_checkbox_2 = gr.Checkbox(value=True, label='2')
                                ht_temperature_2 = gr.Number(label='温度 (℃)', value=500, interactive=True)
                                ht_time_2 = gr.Number(label='时间 (min)', value=120, interactive=True)
                                ht_cooling_method_2 = gr.Dropdown(label='冷却方式', choices=['炉冷', '空冷', '水淬'],
                                                                  value='空冷', interactive=True)
            with gr.Row():
                reset_btn = gr.Button('重置', interactive=True)
                run_btn = gr.Button('提交', interactive=True)
        with gr.Column():
            with gr.Accordion('力学性能'):
                YS = gr.Number(label='屈服强度 (MPa)', interactive=False, precision=3)
                TS = gr.Number(label='抗拉强度 (MPa)', interactive=False, precision=3)
                HD = gr.Number(label='硬度 (VPN)', interactive=False, precision=3)
                HP = gr.Number(label='霍尔佩奇系数 (MPa·m^(1/2))', interactive=False, precision=3)
                YS_GS_curve = gr.Plot()
                TS_GS_curve = gr.Plot()
    simple_mode_tab.select(fn=lambda: 'simple', outputs=proc_mode)
    advanced_mode_tab.select(fn=lambda: 'advanced', outputs=proc_mode)
    reset_btn.click(
        fn=lambda: [99.655, 0.015, 0, 0.08, 0.05, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600, 10, 900, 70, True, 800, 60,
                    '水淬', True, 500, 120, '空冷', None, None, None, None, None, None],
        outputs=[Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, HTT, GS, proc_temperature,
                 proc_deformation, ht_checkbox_1, ht_temperature_1, ht_time_1, ht_cooling_method_1, ht_checkbox_2,
                 ht_temperature_2, ht_time_2, ht_cooling_method_2, YS, TS, HD, HP, YS_GS_curve, TS_GS_curve]
    )
    run_btn.click(
        fn=lambda *args: simple_get_ta_mech_props(
            Ti=args[1], H=args[2], B=args[3], C=args[4], N=args[5], O=args[6], Al=args[7], Si=args[8], Cr=args[9],
            Fe=args[10], Ni=args[11], Cu=args[12], Zr=args[13], Nb=args[14], Mo=args[15], V=args[16], Sn=args[17],
            HTT=args[18], GS=args[19]
        ) if args[0] == 'simple' else advanced_get_ta_mech_props(
            Ti=args[1], H=args[2], B=args[3], C=args[4], N=args[5], O=args[6], Al=args[7], Si=args[8], Cr=args[9],
            Fe=args[10], Ni=args[11], Cu=args[12], Zr=args[13], Nb=args[14], Mo=args[15], V=args[16], Sn=args[17],
            proc_temperature=args[20], proc_deformation=args[21], ht_checkbox_1=args[22], ht_temperature_1=args[23],
            ht_time_1=args[24], ht_cooling_method_1=args[25], ht_checkbox_2=args[26], ht_temperature_2=args[27],
            ht_time_2=args[28], ht_cooling_method_2=args[29]
        ),
        inputs=[proc_mode, Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, HTT, GS, proc_temperature,
                proc_deformation, ht_checkbox_1, ht_temperature_1, ht_time_1, ht_cooling_method_1, ht_checkbox_2,
                ht_temperature_2, ht_time_2, ht_cooling_method_2],
        outputs=[YS, TS, HD, HP, YS_GS_curve, TS_GS_curve]
    )
    for elem in [H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn]:
        elem.change(
            fn=lambda *args: 100.0 - sum(0 if arg is None else arg for arg in args),
            inputs=[H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn],
            outputs=Ti
        )


def build_ta_wf_tab():
    with gr.Row():
        with gr.Column():
            with gr.Accordion('元素组成'):
                with gr.Row():
                    Ti = gr.Number(label=f'Ti (wt%)', value=100, minimum=0, maximum=100, interactive=False)
                    Al = gr.Number(label=f'Al (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Si = gr.Number(label=f'Si (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Cr = gr.Number(label=f'Cr (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Fe = gr.Number(label=f'Fe (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Ni = gr.Number(label=f'Ni (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Cu = gr.Number(label=f'Cu (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Zr = gr.Number(label=f'Zr (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Nb = gr.Number(label=f'Nb (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Mo = gr.Number(label=f'Mo (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    V = gr.Number(label=f'V (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Sn = gr.Number(label=f'Sn (wt%)', value=0, minimum=0, maximum=100, interactive=True)
            with gr.Accordion('工艺参数'):
                HTT = gr.Number(label='热处理温度 (℃)', value=600, minimum=200, maximum=1800, interactive=True)
            with gr.Row():
                reset_btn = gr.Button('重置', interactive=True)
                run_btn = gr.Button('提交', interactive=True)
        with gr.Column():
            with gr.Accordion('相比例'):
                ALPHA = gr.Number(label='ALPHA (wt%)', interactive=False, precision=3)
                BETA = gr.Number(label='BETA (wt%)', interactive=False, precision=3)
                LAVES = gr.Number(label='LAVES (wt%)', interactive=False, precision=3)
                TI3AL = gr.Number(label='TI3AL (wt%)', interactive=False, precision=3)
                TI2CU = gr.Number(label='TI2CU (wt%)', interactive=False, precision=3)
                TI5SI3 = gr.Number(label='TI5SI3 (wt%)', interactive=False, precision=3)
                TIZRSI = gr.Number(label='TIZRSI (wt%)', interactive=False, precision=3)
                TI2NI = gr.Number(label='TI2NI (wt%)', interactive=False, precision=3)
                TIM_B2 = gr.Number(label='TIM_B2 (wt%)', interactive=False, precision=3)
                LIQUID = gr.Number(label='LIQUID (wt%)', interactive=False, precision=3)
                C15_FCC = gr.Number(label='C15_FCC (wt%)', interactive=False, precision=3)
                MC = gr.Number(label='MC (wt%)', interactive=False, precision=3)
    reset_btn.click(
        fn=lambda: [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600, None, None, None, None, None, None, None, None, None,
                    None, None, None],
        outputs=[Ti, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, HTT, ALPHA, BETA, LAVES, TI3AL, TI2CU, TI5SI3, TIZRSI,
                 TI2NI, TIM_B2, LIQUID, C15_FCC, MC]
    )
    run_btn.click(
        fn=get_ta_wf,
        inputs=[Ti, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn, HTT],
        outputs=[ALPHA, BETA, LAVES, TI3AL, TI2CU, TI5SI3, TIZRSI, TI2NI, TIM_B2, LIQUID, C15_FCC, MC]
    )
    for elem in [Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn]:
        elem.change(
            fn=lambda *args: 100.0 - sum(0 if arg is None else arg for arg in args),
            inputs=[Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn],
            outputs=Ti
        )


def build_ta_batch_tab():
    with gr.Row():
        with gr.Column():
            upload_files = gr.File(label='上传 CSV 文件', file_types=['.csv'], file_count='multiple', type='filepath',
                               interactive=True)
            checkboxes = gr.CheckboxGroup(choices=['热膨胀系数', '密度', '热导率', '电导率', '杨氏模量', '体积模量', '剪切模量',
                                                   '泊松比', '比焓', '比热容', 'β-转变温度', '屈服强度', '抗拉强度', '硬度',
                                                   '霍尔佩奇系数', '相比例'],
                                          label='选择性能', interactive=True)
            with gr.Row():
                reset_btn = gr.Button('重置', interactive=True)
                run_btn = gr.Button('提交', interactive=True)
                select_all_btn = gr.Button('全选', interactive=True)
            gr.Markdown('### 批处理模板文件下载')
            gr.File(value=[str(resources.files('tap2.resource').joinpath('tapp_template.csv'))],
                    label='下载批处理模板文件', file_count='multiple', type='filepath', interactive=False)
        with gr.Column():
            download_files = gr.File(label='下载 CSV 文件', file_types=['.csv'], file_count='multiple', type='filepath',
                                 interactive=False)
    run_btn.click(
        fn=batch_get_ta_prop,
        inputs=[checkboxes, upload_files],
        outputs=download_files
    )
    select_all_btn.click(
        fn=lambda sel_props: list(
            {'热膨胀系数', '密度', '热导率', '电导率', '杨氏模量', '体积模量', '剪切模量', '泊松比', '比焓', '比热容', 'β-转变温度',
             '屈服强度', '抗拉强度', '硬度', '霍尔佩奇系数', '相比例'} - set(sel_props)
        ),
        inputs=checkboxes,
        outputs=checkboxes
    )
    reset_btn.click(
        fn=lambda: [None, None, None],
        outputs=[upload_files, download_files, checkboxes]
    )


def build_ta_tab():
    with gr.Tabs():
        with gr.Tab("物理性能"):
            build_ta_phys_tab()
        with gr.Tab("力学性能"):
            build_ta_mech_tab()
        with gr.Tab("相比例"):
            build_ta_wf_tab()
        with gr.Tab("批量模式"):
            build_ta_batch_tab()


def build_aa_phys_tab():
    with gr.Row():
        with gr.Column():
            with gr.Accordion('元素组成'):
                with gr.Row():
                    Al = gr.Number(label=f'Al (wt%)', value=100, minimum=0, maximum=100, interactive=False)
                    Li = gr.Number(label=f'Li (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Mg = gr.Number(label=f'Mg (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Si = gr.Number(label=f'Si (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Ca = gr.Number(label=f'Ca (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Sc = gr.Number(label=f'Sc (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Ti = gr.Number(label=f'Ti (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    V = gr.Number(label=f'V (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Cr = gr.Number(label=f'Cr (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Mn = gr.Number(label=f'Mn (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Fe = gr.Number(label=f'Fe (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Co = gr.Number(label=f'Co (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Ni = gr.Number(label=f'Ni (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Cu = gr.Number(label=f'Cu (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Zn = gr.Number(label=f'Zn (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Zr = gr.Number(label=f'Zr (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    Sn = gr.Number(label=f'Sn (wt%)', value=0, minimum=0, maximum=100, interactive=True)
                    La = gr.Number(label=f'La (wt%)', value=0, minimum=0, maximum=100, interactive=True)
            with gr.Accordion('工艺参数'):
                HTT = gr.Number(label='热处理温度 (℃)', value=500, minimum=400, maximum=700, interactive=True)
            with gr.Row():
                reset_btn = gr.Button('重置', interactive=True)
                run_btn = gr.Button('提交', interactive=True)
        with gr.Column():
            with gr.Accordion('物理性能'):
                TE = gr.Number(label='热膨胀系数 (10^-6/K)',interactive=False, precision=3)
                TC = gr.Number(label='热导率 (W/m·K)',interactive=False, precision=3)
    reset_btn.click(
        fn=lambda: [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600, None, None],
        outputs=[Al, Li, Mg, Si, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Sn, La, HTT, TE, TC]
    )
    run_btn.click(
        fn=get_aa_phys_props,
        inputs=[Al, Li, Mg, Si, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Sn, La, HTT],
        outputs=[TE, TC]
    )
    for elem in [Li, Mg, Si, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Sn, La]:
        elem.change(
            fn=lambda *args: 100.0 - sum(0 if arg is None else arg for arg in args),
            inputs=[Li, Mg, Si, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Sn, La],
            outputs=Al
        )


def build_aa_mech_tab():
    with gr.Row():
        with gr.Column():
            with gr.Accordion('元素组成'):
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
                    Mo = gr.Number(label=f"Mo (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                    Sn = gr.Number(label=f"Sn (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                    La = gr.Number(label=f"La (wt%)", value=0, minimum=0, maximum=100, interactive=True)
                    Ce = gr.Number(label=f"Ce (wt%)", value=0, minimum=0, maximum=100, interactive=True)
            with gr.Row():
                reset_btn = gr.Button("重置", interactive=True)
                run_btn = gr.Button("提交", interactive=True)
        with gr.Column():
            with gr.Accordion('力学性能'):
                YS = gr.Number(label='屈服强度 (MPa)', value=0, interactive=False, precision=3)
    reset_btn.click(
        fn=lambda: [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None],
        outputs=[Al, Li, Mg, Si, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Mo, Sn, La, Ce, YS]
    )
    run_btn.click(
        fn=get_aa_mech_props,
        inputs=[Al, Li, Mg, Si, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Mo, Sn, La, Ce],
        outputs=YS
    )
    for elem in [Li, Mg, Si, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Mo, Sn, La, Ce]:
        elem.change(
            fn=lambda *args: 100.0 - sum(0 if arg is None else arg for arg in args),
            inputs=[Li, Mg, Si, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Mo, Sn, La, Ce],
            outputs=Al
        )


def build_aa_tab():
    with gr.Tab('物理性能'):
        build_aa_phys_tab()
    with gr.Tab('力学性能'):
        build_aa_mech_tab()


@click.command()
@click.option("--host", default='127.0.0.1', type=str)
@click.option("--port", default=7860, type=click.IntRange(1, 65535))
@click.option("--max-batch-size", default=64, type=click.IntRange(16, 10000))
@click.option("--device", default='cpu', type=click.Choice(['cpu', 'cuda'], case_sensitive=False))
def run_gradio_client(host, port, max_batch_size, device):
    global tapp
    tapp = TAPP(device=device, max_batch_size=max_batch_size)
    print(text2art('TAPP'), end='')
    with gr.Blocks(title="TAPP") as index:
        with gr.Tabs():
            with gr.Tab('钛合金'):
                build_ta_tab()
            with gr.Tab('铝合金'):
                build_aa_tab()
    index.launch(server_name=host, server_port=port, share=False)

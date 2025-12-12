import csv
import numpy as np
import openpyxl
import gradio as gr
import matplotlib.pyplot as plt
import charset_normalizer
from pathlib import Path
from typing import Dict, List
from gradio.utils import NamedString
from tempfile import NamedTemporaryFile
from matplotlib.font_manager import FontProperties
from importlib import resources
from tap2.core import TAPPInput, TAPPInfer, _MAX_BATCH_SIZE, TAPPBatchInput
from loguru import logger
from thermal_deformation import predict_grain_size


_TAPP_INFER = TAPPInfer()


_MSYH_FONT = FontProperties(fname=str(resources.files("tap2.resource").joinpath("msyh.ttc")))


_PHASE_COLOR_MAP = {
    "ALPHA": "#1f77b4",
    "BETA": "#ff7f0e",
    "LAVES": "#2ca02c",
    "TI3AL": "#d62728",
    "TI2CU": "#9467bd",
    "TI5SI3": "#8c564b",
    "TIZRSI": "#e377c2",
    "TI2NI": "#bcbd22",
    "TIM_B2": "#17becf",
    "LIQUID": "#aec7e8",
    "C15_FCC": "#ffbb78",
    "MC": "#9edae5"
}


_PROP_ZH2ABBR_MAP = {
    "热膨胀系数": "TE",
    "密度": "DS",
    "热导率": "TC",
    "电导率": "EC",
    "杨氏模量": "YM",
    "体积模量": "BM",
    "剪切模量": "SM",
    "泊松比": "PR",
    "比焓": "SE",
    "比热容": "SHC",
    "屈服强度": "YS",
    "抗拉强度": "TS",
    "硬度": "HD",
    "霍尔佩奇系数": "HP"
}


_PROP_ZH2UNIT_MAP = {
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


def _get_ti_alloy_phys_prop(Ti: float | None, H: float | None, B: float | None,  C: float | None, N: float | None,
                            O: float | None, Al: float | None, Si: float | None, Cr: float | None, Fe: float | None,
                            Ni: float | None, Cu: float | None, Zr: float | None, Nb: float | None, Mo: float | None,
                            V: float | None, Sn: float | None, htt: float | None) -> List[float | None]:
    """
    Gradio 接口：根据钛合金的元素组成（Ti、H、B、C、N、O、Al、Si、Cr、Fe、Ni、Cu、Zr、Nb、Mo、V、Sn）和处理工艺（热处理温度），获取其
    物理性能（热膨胀系数、密度、热导率、电导率、杨氏模量、体积模量、剪切模量、泊松比、比焓、比热容）。
    :param Ti: Ti 的质量分数
    :param H: H 的质量分数
    :param B: B 的质量分数
    :param C: C 的质量分数
    :param N: N 的质量分数
    :param O: O 的质量分数
    :param Al: Al 的质量分数
    :param Si: Si 的质量分数
    :param Cr: Cr 的质量分数
    :param Fe: Fe 的质量分数
    :param Ni: Ni 的质量分数
    :param Cu: Cu 的质量分数
    :param Zr: Zr 的质量分数
    :param Nb: Nb 的质量分数
    :param Mo: Mo 的质量分数
    :param V: V 的质量分数
    :param Sn: Sn 的质量分数
    :param htt: 热处理温度（摄氏度）
    :return: 物理性能值列表，依次是：热膨胀系数（10^-6/K）、密度（g/cm^3）、热导率（W/m·K）、电导率（10^6 S/m）、杨氏模量（GPa）、
        体积模量（GPa）、剪切模量（GPa）、泊松比、比焓（J/g）、比热容（J/g·K），均为浮点数。
    """
    if not _validate_composition(Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn):
        gr.Warning("请输入正确的成分！")
        return [None] * 10
    prop_values = []
    for prop_abbr in ["TE", "DS", "TC", "EC", "YM", "BM", "SM", "PR", "SE", "SHC"]:
        tapp_input = TAPPInput(
            Prop=prop_abbr,
            Ti = Ti if Ti is not None else 0,
            H = H if H is not None else 0,
            B = B if B is not None else 0,
            C = C if C is not None else 0,
            N = N if N is not None else 0,
            O = O if O is not None else 0,
            Al = Al if Al is not None else 0,
            Si = Si if Si is not None else 0,
            Cr = Cr if Cr is not None else 0,
            Fe = Fe if Fe is not None else 0,
            Ni = Ni if Ni is not None else 0,
            Cu = Cu if Cu is not None else 0,
            Zr = Zr if Zr is not None else 0,
            Nb = Nb if Nb is not None else 0,
            Mo = Mo if Mo is not None else 0,
            V = V if V is not None else 0,
            Sn = Sn if Sn is not None else 0,
            HTT = htt if htt is not None else 600,
        )
        tapp_output = _TAPP_INFER(tapp_input)
        prop_values.append(tapp_output.value)
    # 修正单位
    prop_values[0] *= 1e6 # TE
    prop_values[3] *= 1e-6 # EC
    return prop_values


def _get_ti_alloy_mech_prop(Ti: float | None, H: float | None, B: float | None, C: float | None, N: float | None,
                            O: float | None, Al: float | None, Si: float | None, Cr: float | None, Fe: float | None,
                            Ni: float | None, Cu: float | None, Zr: float | None, Nb: float | None, Mo: float | None,
                            V: float | None, Sn: float | None, proc_mode: str, simple_htt: float | None,
                            simple_gs: float | None, td_htt: float | None, td_ts: float | None, td_sr: float | None,
                            td_init_gs: float | None, td_btt: float | None) -> List[float | None]:
    """
    Gradio 接口：根据钛合金的元素组成（Ti、H、B、C、N、O、Al、Si、Cr、Fe、Ni、Cu、Zr、Nb、Mo、V、Sn）、处理工艺（热处理温度）、晶粒尺寸，
    获取其力学性能（屈服强度、抗拉强度、维氏硬度、霍尔佩奇系数）。
    :param Ti: Ti 的质量分数
    :param H: H 的质量分数
    :param B: B 的质量分数
    :param C: C 的质量分数J
    :param N: N 的质量分数
    :param O: O 的质量分数
    :param Al: Al 的质量分数
    :param Si: Si 的质量分数
    :param Cr: Cr 的质量分数
    :param Fe: Fe 的质量分数
    :param Ni: Ni 的质量分数
    :param Cu: Cu 的质量分数
    :param Zr: Zr 的质量分数
    :param Nb: Nb 的质量分数
    :param Mo: Mo 的质量分数
    :param V: V 的质量分数
    :param Sn: Sn 的质量分数
    :param simple_htt: 热处理温度（摄氏度）
    :param simple_gs: 晶粒尺寸（微米）
    :return: 力学性能列表，依次是：屈服强度（MPa）、抗拉强度（MPa）、维氏硬度（VPN）、霍尔佩奇系数（MPa·m^(1/2)）。
    """
    # 检查：元素浓度之和为 100%
    if not _validate_composition(Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn):
        gr.Warning("请输入正确的成分！")
        return [None] * 10
    prop_values = []
    if proc_mode == "simple":
        htt = simple_htt
        grain_size = simple_gs
    elif proc_mode == "advanced":
        htt = td_htt
        if any(required_param is None for required_param in (td_htt, td_ts, td_sr)):
            gr.Warning("请输入完整的热变形参数！")
            return [None] * 10
        # 构建组成字符串
        comp_sub_strs = ["Ti"]
        for elem_name in ("H", "B", "C", "N", "O", "Al", "Si", "Cr", "Fe", "Ni", "Cu", "Zr", "Nb", "Mo", "V", "Sn"):
            elem_conc = locals()[elem_name]
            if elem_conc is not None and abs(elem_conc) > 1e-6:
                comp_sub_strs.append(f"{elem_conc:f}".rstrip("0").rstrip(".") + elem_name)
        composition_str = "-".join(comp_sub_strs)
        # 构建命名参数字典
        param_dict = {
            "composition": composition_str,
            "temperature": td_htt,
            "true_strain": td_ts,
            "strain_rate": td_sr,
        }
        if td_init_gs is not None:
            param_dict["initial_grain_size"] = td_init_gs
        if td_btt is not None:
            param_dict["beta_trans_temp"] = td_btt
        logger.info(f"Predicting <GS>: {param_dict}")
        grain_size = predict_grain_size(**param_dict)
        logger.info(f"Predicted <GS>: {param_dict} ==> {grain_size}")
    else:
        raise ValueError(f"Unknown process mode: {proc_mode}")
    for prop_abbr in ["YS", "TS", "HD", "HP"]:
        tapp_input = TAPPInput(
            Prop=prop_abbr,
            Ti = Ti if Ti is not None else 0,
            H = H if H is not None else 0,
            B = B if B is not None else 0,
            C = C if C is not None else 0,
            N = N if N is not None else 0,
            O = O if O is not None else 0,
            Al = Al if Al is not None else 0,
            Si = Si if Si is not None else 0,
            Cr = Cr if Cr is not None else 0,
            Fe = Fe if Fe is not None else 0,
            Ni = Ni if Ni is not None else 0,
            Cu = Cu if Cu is not None else 0,
            Zr = Zr if Zr is not None else 0,
            Nb = Nb if Nb is not None else 0,
            Mo = Mo if Mo is not None else 0,
            V = V if V is not None else 0,
            Sn = Sn if Sn is not None else 0,
            HTT = simple_htt if simple_htt is not None else 600,
            GS = grain_size if grain_size is not None else 10,
        )
        tapp_output = _TAPP_INFER(tapp_input)
        prop_values.append(tapp_output.value)
    return prop_values


def _get_inputs_from_csv(csv_path: Path) -> Dict[str, List[float]]:
    """
    从 CSV 文件中读取钛合金性能预测的输入数据，格式为字典，字典字段包含元素组成（Ti、H、B、C、N、O、Al、Si、Cr、Fe、Ni、Cu、Zr、Nb、Mo、
    V、Sn）、热处理温度、晶粒尺寸，不存在则填充相应的默认值。
    :param csv_path: CSV 文件路径
    :return: 输入数据列表，例如：{"Ti": [90], "H": [0], "B": [0], "C": [0], "N": [0], "O": [0], "Al": [6], "Si": [0],
        "Cr": [0], "Fe": [0], "Ni": [0], "Cu": [0], "Zr": [0], "Nb": [0], "Mo": [0], "V": [4], "Sn": [0], "HTT": [600],
        "GS": [10]}
    """
    inputs = {
        "Ti": [],
        "H": [],
        "B": [],
        "C": [],
        "N": [],
        "O": [],
        "Al": [],
        "Si": [],
        "Cr": [],
        "Fe": [],
        "Ni": [],
        "Cu": [],
        "Zr": [],
        "Nb": [],
        "Mo": [],
        "V": [],
        "Sn": [],
        "HTT": [],
        "GS": []
    }
    # 自动检测文件编码
    encode_detector = charset_normalizer.from_path(csv_path, cp_isolation=["utf_8", "gb18030", "big5"]).best()
    encoding = encode_detector.encoding if encode_detector is not None else None
    logger.info(f"Read CSV file: {csv_path.stem} ({encoding})")
    with open(csv_path, "r", newline="", encoding=encoding) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        row_count = 0
        for row in csv_reader:
            for elem in ["Ti", "H", "B", "C", "N", "O", "Al", "Si", "Cr", "Fe", "Ni", "Cu", "Zr", "Nb", "Mo", "V", "Sn"]:
                inputs[elem].append(float(row[elem]) if elem in row and row[elem] != "" else 0)
            inputs["HTT"].append(float(row["热处理温度"]) if "热处理温度" in row and row["热处理温度"] != "" else 600)
            inputs["GS"].append(float(row["晶粒尺寸"]) if "晶粒尺寸" in row and row["晶粒尺寸"] != "" else 10)
            row_count += 1
            # 控制最大批量大小
            if row_count >= _MAX_BATCH_SIZE:
                break
    return inputs


def _get_inputs_from_xlsx(xlsx_path: Path) -> Dict[str, List[float]]:
    """
    从 XLSX 文件中读取钛合金性能预测的输入数据，格式为字典列表，字典字段包含元素组成（Ti、H、B、C、N、O、Al、Si、Cr、Fe、Ni、Cu、Zr、Nb、
    Mo、V、Sn）、热处理温度、晶粒尺寸，不存在则填充相应的默认值。
    :param xlsx_path: XLSX 文件路径
    :return: 输入数据列表，例如：{"Ti": [90], "H": [0], "B": [0], "C": [0], "N": [0], "O": [0], "Al": [6], "Si": [0],
        "Cr": [0], "Fe": [0], "Ni": [0], "Cu": [0], "Zr": [0], "Nb": [0], "Mo": [0], "V": [4], "Sn": [0], "HTT": [600],
        "GS": [10]}
    """
    inputs = {
        "Ti": [],
        "H": [],
        "B": [],
        "C": [],
        "N": [],
        "O": [],
        "Al": [],
        "Si": [],
        "Cr": [],
        "Fe": [],
        "Ni": [],
        "Cu": [],
        "Zr": [],
        "Nb": [],
        "Mo": [],
        "V": [],
        "Sn": [],
        "HTT": [],
        "GS": []
    }
    logger.info(f"Read xlsx file: {xlsx_path.stem}")
    workbook = openpyxl.load_workbook(xlsx_path)
    worksheet = workbook.active
    for col in worksheet.iter_cols(values_only=True):
        if isinstance(col, tuple) and len(col) > 1:
            col_name = col[0]
            end_idx = min(len(col), _MAX_BATCH_SIZE + 1)
            if col_name in ["Ti", "H", "B", "C", "N", "O", "Al", "Si", "Cr", "Fe", "Ni", "Cu", "Zr", "Nb", "Mo", "V", "Sn"]:
                for cell_value in col[1: end_idx]:
                    inputs[col_name].append(float(cell_value) if cell_value is not None else 0)
            elif col_name == "热处理温度":
                for cell_value in col[1: end_idx]:
                    inputs["HTT"].append(float(cell_value) if cell_value is not None else 600)
            elif col_name == "晶粒尺寸":
                for cell_value in col[1: end_idx]:
                    inputs["GS"].append(float(cell_value) if cell_value is not None else 10)
    input_size = min(len(_) for _ in inputs.values())
    for col_name in ["Ti", "H", "B", "C", "N", "O", "Al", "Si", "Cr", "Fe", "Ni", "Cu", "Zr", "Nb", "Mo", "V", "Sn"]:
        if len(inputs[col_name]) == 0:
            inputs[col_name] = [0] * input_size
    if len(inputs["HTT"]) == 0:
        inputs["HTT"] = [600] * input_size
    if len(inputs["GS"]) == 0:
        inputs["GS"] = [10] * input_size
    return inputs


def _write_outputs_to_csv(input_csv_path: Path, append_header: List[str], output_data: List[List[float]]) -> Path:
    """
    将钛合金性能计算结果写入 CSV 文件，保留原有的输入数据，并在后面追加性能数据（保留三位小数），返回新文件的路径。
    :param input_csv_path: 输入 CSV 的路径
    :param append_header: 追加的表头
    :param output_data: 计算结果
    :return: 新文件的路径，文件名为原文件名+下划线+随机字符串。例如：输入文件名为“input.csv”，则输出文件名可能为“input_abcd1234.csv”。
    """
    encode_detector = charset_normalizer.from_path(input_csv_path, cp_isolation=["utf_8", "gb18030", "big5"]).best()
    encoding = encode_detector.encoding if encode_detector is not None else None
    with NamedTemporaryFile(delete=False, prefix=f"{input_csv_path.stem}_", suffix=".csv", mode="w", newline="",
                            encoding="utf-8") as output_csv_file:
        logger.info(f"Write csv file: {Path(output_csv_file.name).stem} (utf-8)")
        with open(input_csv_path, "r", newline="", encoding=encoding) as input_csv_file:
            csv_reader = csv.reader(input_csv_file)
            csv_writer = csv.writer(output_csv_file)
            for idx, row in enumerate(csv_reader):
                if idx == 0:
                    csv_writer.writerow(row + append_header)
                else:
                    csv_writer.writerow(row + [f"{value:.3f}" for value in output_data[idx - 1]])
        return Path(output_csv_file.name)


def _write_outputs_to_xlsx(input_xlsx_path: Path, append_header: List[str], output_data: List[List[float]]) -> Path:
    """
    将钛合金性能计算结果写入 XLSX 文件，保留原有的输入数据，并在后面追加性能数据（保留三位小数），返回新文件的路径。
    :param input_xlsx_path: 输入 XLSX 的路径
    :param append_header: 追加的表头
    :param output_data: 计算结果
    :return: 新文件的路径，文件名为原文件名+下划线+随机字符串。例如：输入文件名为“input.xlsx”，则输出文件名可能为“input_abcd1234.xlsx”。
    """
    output_file = NamedTemporaryFile(delete=False, prefix=f"{input_xlsx_path.stem}_", suffix=".xlsx", mode="w")
    output_path = Path(output_file.name)
    output_file.close()
    logger.info(f"Write xlsx file: {output_path.stem}")
    input_wb = openpyxl.load_workbook(input_xlsx_path, read_only=True)
    input_ws = input_wb.active
    output_wb = openpyxl.Workbook()
    output_ws = output_wb.active
    # 写入原数据
    for row in input_ws.iter_rows(values_only=True):
        output_ws.append(row)
    start_col_num = output_ws.max_column + 1
    # 写入新数据
    for i, prop_name in enumerate(append_header):
        output_ws.cell(row=1, column=start_col_num + i, value=prop_name)
    for i, outputs in enumerate(output_data):
        for j, prop_value in enumerate(outputs):
            output_ws.cell(row=i + 2, column=start_col_num + j, value=f"{prop_value:.3f}")
    output_wb.save(output_path)
    return output_path


def _get_ti_alloy_prop(prop_names: List[str], input_files: List[NamedString] | None) -> List[str]:
    """
    Gradio 接口：根据钛合金的元素组成（Ti、H、B、C、N、O、Al、Si、Cr、Fe、Ni、Cu、Zr、Nb、Mo、V、Sn）、处理工艺（热处理温度）、晶粒尺寸，
    批量获取其性能，支持 CSV 和 XLSX 文件格式。
    :param prop_names: 需计算的性能名称列表（热膨胀系数、密度、热导率、电导率、杨氏模量、体积模量、剪切模量、泊松比、比焓、比热容、
        屈服强度、抗拉强度、硬度、霍尔佩奇系数）
    :param input_files: 需计算的输入文件列表
    :return: 计算结果文件列表
    """
    prop_abbrs = [_PROP_ZH2ABBR_MAP[prop_name] for prop_name in prop_names]
    append_header = [f"{prop_name} ({_PROP_ZH2UNIT_MAP[prop_name]})"
                     if _PROP_ZH2UNIT_MAP[prop_name] is not None else prop_name
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
            reader, writer = _get_inputs_from_csv, _write_outputs_to_csv
        elif input_path.match("*.xlsx"):
            reader, writer = _get_inputs_from_xlsx, _write_outputs_to_xlsx
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
            tapp_input = TAPPBatchInput(Prop=prop_abbr, **dict_input)
            tapp_output = _TAPP_INFER(tapp_input)
            # 单位修正
            if prop_abbr == "TE":
                outputs = [_ * 1e6 for _ in tapp_output.value]
            elif prop_abbr == "EC":
                outputs = [_ * 1e-6 for _ in tapp_output.value]
            else:
                outputs = tapp_output.value
            outputs_col.append(outputs)
        # 转置
        outputs_col = np.array(outputs_col)
        outputs_row = outputs_col.T
        outputs_row = outputs_row.tolist()
        output_path = str(writer(input_path, append_header, outputs_row))
        output_paths.append(output_path)
    return output_paths


def _get_ti_alloy_wf(Ti: float | None, Al: float | None, Si: float | None, Cr: float | None, Fe: float | None,
                     Ni: float | None, Cu: float | None, Zr: float | None, Nb: float | None, Mo: float | None,
                     V: float | None, Sn: float | None, htt: float | None) -> List:
    """
    Gradio 接口：根据钛合金的元素组成（Ti、Al、Si、Cr、Fe、Ni、Cu、Zr、Nb、Mo、V、Sn）和处理工艺（热处理温度），获取其相比例（ALPHA、BETA、
    LAVES、TI3AL、TI2CU、TI5SI3、TIZRSI、TI2NI、TIM_B2、LIQUID、C15_FCC、MC）。
    :param Ti: Ti 的质量分数
    :param Al: Al 的质量分数
    :param Si: Si 的质量分数
    :param Cr: Cr 的质量分数
    :param Fe: Fe 的质量分数
    :param Ni: Ni 的质量分数
    :param Cu: Cu 的质量分数
    :param Zr: Zr 的质量分数
    :param Nb: Nb 的质量分数
    :param Mo: Mo 的质量分数
    :param V: V 的质量分数
    :param Sn: Sn 的质量分数
    :param htt: 热处理温度（摄氏度）
    :return: 相比例及其饼图（质量分数）
    """
    # 检查：元素浓度之和为 100 wt%
    if not _validate_composition(Ti, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn):
        gr.Warning("请输入正确的成分！")
        return [None] * 13
    # 构建输入
    tapp_input = TAPPInput(
        Prop="WF",
        Ti=Ti if Ti is not None else 0,
        H=0,
        B=0,
        C=0,
        N=0,
        O=0,
        Al=Al if Al is not None else 0,
        Si=Si if Si is not None else 0,
        Cr=Cr if Cr is not None else 0,
        Fe=Fe if Fe is not None else 0,
        Ni=Ni if Ni is not None else 0,
        Cu=Cu if Cu is not None else 0,
        Zr=Zr if Zr is not None else 0,
        Nb=Nb if Nb is not None else 0,
        Mo=Mo if Mo is not None else 0,
        V=V if V is not None else 0,
        Sn=Sn if Sn is not None else 0,
        HTT=htt if htt is not None else 600,
    )
    # 获取输出
    tapp_output = _TAPP_INFER(tapp_input)
    # 转为质量分数
    for k, v in tapp_output.value.items():
        tapp_output.value[k] = v * 100
    # 绘制饼图
    pos_labels = []
    pos_sizes = []
    pos_colors = []
    for k, v in tapp_output.value.items():
        if abs(v) > 1e-3:
            pos_labels.append(f"{k} ({v:.3f}%)")
            pos_sizes.append(v)
            pos_colors.append(_PHASE_COLOR_MAP[k])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        pos_sizes,
        labels=pos_labels,
        colors=pos_colors,
        startangle=90,
        textprops={"fontproperties": _MSYH_FONT, "fontsize": "medium"}
    )
    ax.set_title("相比例 (wt%)", fontproperties=_MSYH_FONT, pad=20, fontsize="x-large")
    ax.axis("equal")
    return [tapp_output.value["ALPHA"], tapp_output.value["BETA"], tapp_output.value["LAVES"],
            tapp_output.value["TI3AL"], tapp_output.value["TI2CU"], tapp_output.value["TI5SI3"],
            tapp_output.value["TIZRSI"], tapp_output.value["TI2NI"], tapp_output.value["TIM_B2"],
            tapp_output.value["LIQUID"], tapp_output.value["C15_FCC"], tapp_output.value["MC"], fig]


def _validate_composition(*args) -> bool:
    """
    验证钛合金成分之和是否为 100 wt%
    :param args: 各元素的质量分数
    :return: 成分之和为 100 wt% 返回 True，否则返回 False
    """
    total_comp = sum(elem_conc for elem_conc in args if elem_conc is not None)
    return abs(total_comp - 100) <= 1e-6

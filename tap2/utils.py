import csv
import numpy as np
import openpyxl
import gradio as gr
import charset_normalizer
from pathlib import Path
from typing import Dict, List
from gradio.utils import NamedString
from tempfile import NamedTemporaryFile
from pandas import DataFrame, to_numeric
from tap2.core import TAPPInput, TAPPInfer, _MAX_BATCH_SIZE, TAPPBatchInput
from loguru import logger
from thermal_deformation import predict_grain_size as TD_pred_GS
from heat_treatment import predict_grain_size as HT_pred_GS, model as HT_model, feats as HT_feats
from math import isclose



_TAPP_INFER = TAPPInfer()

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


def _get_TA_phys_prop(Ti: float | None, H: float | None, B: float | None, C: float | None, N: float | None, O: float | None, Al: float | None, Si: float | None, Cr: float | None, Fe: float | None, Ni: float | None, Cu: float | None, Zr: float | None, Nb: float | None, Mo: float | None, V: float | None, Sn: float | None, HTT: float | None) -> List[float | None]:
    """
    Gradio 接口：根据钛合金的元素组成（Ti、H、B、C、N、O、Al、Si、Cr、Fe、Ni、Cu、Zr、Nb、Mo、V、Sn）和处理工艺（热处理温度），获取其物理性能（热膨胀系数、密度、热导率、电导率、杨氏模量、体积模量、剪切模量、泊松比、比焓、比热容）。
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
    :param HTT: 热处理温度（摄氏度）
    :return: 物理性能值列表，依次是：热膨胀系数（10^-6/K）、密度（g/cm^3）、热导率（W/m·K）、电导率（10^6 S/m）、杨氏模量（GPa）、
    体积模量（GPa）、剪切模量（GPa）、泊松比、比焓（J/g）、比热容（J/g·K），均为浮点数。
    """
    if not _valid_comp(Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn):
        gr.Warning("请输入正确的成分！")
        return [None] * 10
    prop_values = []
    for prop_abbr in ["TE", "DS", "TC", "EC", "YM", "BM", "SM", "PR", "SE", "SHC"]:
        # noinspection PyTypeChecker
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
            HTT = HTT if HTT is not None else 600,
        )
        tapp_output = _TAPP_INFER(tapp_input)
        prop_values.append(tapp_output.value)
    # 修正单位
    prop_values[0] *= 1e6 # TE
    prop_values[3] *= 1e-6 # EC
    return prop_values


def _get_TA_mech_prop(Ti: float | None, H: float | None, B: float | None, C: float | None, N: float | None, O: float | None, Al: float | None, Si: float | None, Cr: float | None, Fe: float | None, Ni: float | None, Cu: float | None, Zr: float | None, Nb: float | None, Mo: float | None, V: float | None, Sn: float | None, proc_mode: str, HTT: float | None, GS: float | None, init_GS: float | None, TD_temp: float | None, TD_TS: float | None, TD_SR: float | None, HT_param: DataFrame) -> List[float | None]:
    """
    Gradio 接口：根据钛合金的元素组成（Ti、H、B、C、N、O、Al、Si、Cr、Fe、Ni、Cu、Zr、Nb、Mo、V、Sn）、处理工艺（热处理温度）、晶粒尺寸，获取其力学性能（屈服强度、抗拉强度、维氏硬度、霍尔佩奇系数）。高级模式：通过热变形和热处理模块计算晶粒尺寸。
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
    :param proc_mode: 处理工艺模式：simple-快捷模式、advanced-高级模式
    :param HTT: 快捷模式：热处理温度（摄氏度）
    :param GS: 快捷模式：晶粒尺寸（微米）
    :param init_GS: 高级模式：初始晶粒尺寸（微米）
    :param TD_temp: 高级模式：热变形温度（摄氏度）
    :param TD_TS: 高级模式：热变形真实应变
    :param TD_SR: 高级模式：热变形应变速率（1/s）
    :param HT_param: 高级模式：热处理参数 DataFrame，每一行表示一组工艺：[温度（摄氏度），保温时间（小时）]
    :return: 力学性能列表，依次是：屈服强度（MPa）、抗拉强度（MPa）、维氏硬度（VPN）、霍尔佩奇系数（MPa·m^(1/2)）。
    """
    if not _valid_comp(Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn):
        gr.Warning("请输入正确的成分！")
        return [None] * 4
    prop_values = []
    if proc_mode == 'simple':
        if HTT is None or GS is None:
            gr.Warning('【快捷模式】请输入完整的工艺参数！')
            return [None] * 4
        _HTT = HTT
        _GS = GS
    else:
        HT_param = HT_param.apply(to_numeric, errors='coerce')
        if any(_ is None for _ in (init_GS, TD_temp, TD_TS, TD_SR)) or not (1 <= len(HT_param) <= 5):
            gr.Warning('【高级模式】请输入完整的热变形和热处理参数！')
            return [None] * 4
        _HTT = HT_param['温度（℃）'].max()
        comp_sub_strs = ["Ti"]
        for elem_name in ("H", "B", "C", "N", "O", "Al", "Si", "Cr", "Fe", "Ni", "Cu", "Zr", "Nb", "Mo", "V", "Sn"):
            elem_conc = locals()[elem_name]
            if elem_conc is not None and abs(elem_conc) > 1e-6:
                comp_sub_strs.append(f"{elem_conc:f}".rstrip("0").rstrip(".") + elem_name)
        comp_str = "-".join(comp_sub_strs)
        tapp_input = TAPPInput(
            Prop='BTT',
            Ti=Ti if Ti is not None else 0,
            H=H if H is not None else 0,
            B=B if B is not None else 0,
            C=C if C is not None else 0,
            N=N if N is not None else 0,
            O=O if O is not None else 0,
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
            Sn=Sn if Sn is not None else 0
        )
        tapp_output = _TAPP_INFER(tapp_input)
        BTT = tapp_output.value
        logger.info(f'THERMAL DEFORMATION: COMPOSITION={comp_str}, TRUE_STRAIN={TD_TS:f}, STRAIN_RATE={TD_SR:f}, '
                    f'TEMPERATURE={TD_temp:f}, INITIAL_GRAIN_SIZE={init_GS:f}, BETA_TRANS_TEMP={BTT:f}')
        _GS = TD_pred_GS(
            composition=comp_str,
            true_strain=TD_TS,
            strain_rate=TD_SR,
            temperature=TD_temp,
            initial_grain_size=init_GS,
            beta_trans_temp=BTT
        )
        logger.info(f'THERMAL DEFORMATION: GS={_GS:f}')
        comp_dict = {
            'Ti': Ti if Ti is not None else 0,
            'Al': Al if Al is not None else 0,
            'Si': Si if Si is not None else 0,
            'Cr': Cr if Cr is not None else 0,
            'Fe': Fe if Fe is not None else 0,
            'Ni': Ni if Ni is not None else 0,
            'Cu': Cu if Cu is not None else 0,
            'Zr': Zr if Zr is not None else 0,
            'Nb': Nb if Nb is not None else 0,
            'Mo': Mo if Mo is not None else 0,
            'V': V if V is not None else 0,
            'Sn': Sn if Sn is not None else 0
        }
        logger.info(f'HEAT TREATMENT: D0={_GS:f}, MANUAL_T_BETA={BTT:f}, COMPOSITION={comp_dict}, '
                    f'HEAT_TREATMENTS={HT_param.to_numpy().tolist()}')
        HT_results = HT_pred_GS(
            D0=_GS,
            manual_T_beta=BTT,
            composition=comp_dict,
            heat_treatments=HT_param.to_numpy().tolist(),
            ml_model=HT_model,
            feature_names=HT_feats
        )
        _GS = HT_results[len(HT_param) - 1]['D']
        logger.info(f'HEAT TREATMENT: GS={_GS:f}')
    for prop_abbr in ["YS", "TS", "HD", "HP"]:
        # noinspection PyTypeChecker
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
            HTT = _HTT,
            GS = _GS
        )
        tapp_output = _TAPP_INFER(tapp_input)
        prop_value = tapp_output.value
        prop_values.append(prop_value)
    return prop_values


def _get_inputs_from_csv(csv_path: Path) -> Dict[str, List[float]]:
    """
    从 CSV 文件中读取钛合金性能预测的输入数据，格式为字典，字典字段包含元素组成（Ti、H、B、C、N、O、Al、Si、Cr、Fe、Ni、Cu、Zr、Nb、Mo、V、Sn）、热处理温度、晶粒尺寸，不存在则填充相应的默认值。
    :param csv_path: CSV 文件路径
    :return: 输入数据列表，例如：{"Ti": [90], "H": [0], "B": [0], "C": [0], "N": [0], "O": [0], "Al": [6], "Si": [0], "Cr": [0], "Fe": [0], "Ni": [0], "Cu": [0], "Zr": [0], "Nb": [0], "Mo": [0], "V": [4], "Sn": [0], "HTT": [600], "GS": [10]}
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
    从 XLSX 文件中读取钛合金性能预测的输入数据，格式为字典列表，字典字段包含元素组成（Ti、H、B、C、N、O、Al、Si、Cr、Fe、Ni、Cu、Zr、Nb、Mo、V、Sn）、热处理温度、晶粒尺寸，不存在则填充相应的默认值。
    :param xlsx_path: XLSX 文件路径
    :return: 输入数据列表，例如：{"Ti": [90], "H": [0], "B": [0], "C": [0], "N": [0], "O": [0], "Al": [6], "Si": [0], "Cr": [0], "Fe": [0], "Ni": [0], "Cu": [0], "Zr": [0], "Nb": [0], "Mo": [0], "V": [4], "Sn": [0], "HTT": [600], "GS": [10]}
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


def _batch_get_TA_prop(prop_names: List[str], input_files: List[NamedString] | None) -> List[str]:
    """
    Gradio 接口：根据钛合金的元素组成（Ti、H、B、C、N、O、Al、Si、Cr、Fe、Ni、Cu、Zr、Nb、Mo、V、Sn）、处理工艺（热处理温度）、晶粒尺寸，批量获取其性能，支持 CSV 和 XLSX 文件格式。
    :param prop_names: 需计算的性能名称列表（热膨胀系数、密度、热导率、电导率、杨氏模量、体积模量、剪切模量、泊松比、比焓、比热容、屈服强度、抗拉强度、硬度、霍尔佩奇系数）
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


def _get_TA_WF(Ti: float | None, Al: float | None, Si: float | None, Cr: float | None, Fe: float | None, Ni: float | None, Cu: float | None, Zr: float | None, Nb: float | None, Mo: float | None, V: float | None, Sn: float | None, HTT: float | None) -> List[float | None]:
    """
    Gradio 接口：根据钛合金的元素组成（Ti、Al、Si、Cr、Fe、Ni、Cu、Zr、Nb、Mo、V、Sn）和处理工艺（热处理温度），获取其相比例（ALPHA、BETA、LAVES、TI3AL、TI2CU、TI5SI3、TIZRSI、TI2NI、TIM_B2、LIQUID、C15_FCC、MC）。
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
    :param HTT: 热处理温度（摄氏度）
    """
    if not _valid_comp(Ti, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn):
        gr.Warning("请输入正确的成分！")
        return [None] * 12
    tapp_input = TAPPInput(
        Prop='WF',
        Ti=Ti if Ti is not None else 0,
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
        HTT=HTT if HTT is not None else 600,
    )
    tapp_output = _TAPP_INFER(tapp_input)
    return [tapp_output.value.ALPHA, tapp_output.value.BETA, tapp_output.value.LAVES, tapp_output.value.TI3AL,
            tapp_output.value.TI2CU, tapp_output.value.TI5SI3, tapp_output.value.TIZRSI, tapp_output.value.TI2NI,
            tapp_output.value.TIM_B2, tapp_output.value.LIQUID, tapp_output.value.C15_FCC, tapp_output.value.MC]


def _valid_comp(*args) -> bool:
    """
    验证钛合金成分之和是否为 100 wt%
    :param args: 各元素的质量分数
    :return: 成分之和为 100 wt% 返回 True，否则返回 False
    """
    total_comp = sum(elem_conc for elem_conc in args if elem_conc is not None)
    return isclose(total_comp, 100, abs_tol=1e-6)


def _get_TA_BTT(Ti: float | None, Al: float | None, Si: float | None, Cr: float | None, Fe: float | None, Ni: float | None, Cu: float | None, Zr: float | None, Nb: float | None, Mo: float | None, V: float | None, Sn: float | None) -> float | None:
    if not _valid_comp(Ti, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn):
        gr.Warning("请输入正确的成分！")
        return None
    tapp_input = TAPPInput(
        Prop='BTT',
        Ti=Ti if Ti is not None else 0,
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
        Sn=Sn if Sn is not None else 0
    )
    tapp_output = _TAPP_INFER(tapp_input)
    btt_value = tapp_output.value
    return btt_value

import csv
import openpyxl
import gradio as gr
from pathlib import Path
from typing import Dict, List
from gradio.utils import NamedString
from tempfile import NamedTemporaryFile
from tapp.core import TAPPInput, TAPPModelInfer


_tapp_model_infer = TAPPModelInfer()


def _get_ti_alloy_phys_prop(Ti: float | None, H: float | None, B: float | None,  C: float | None, N: float | None,
                            O: float | None, Al: float | None, Si: float | None, Cr: float | None, Fe: float | None,
                            Ni: float | None, Cu: float | None, Zr: float | None, Nb: float | None, Mo: float | None,
                            V: float | None, Sn: float | None, htt: float | None) -> List[float]:
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
    # 检查：元素浓度之和为 100 wt%
    total_comp = sum(elem_conc for elem_conc in [Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn]
                     if elem_conc is not None)
    if abs(total_comp - 100.) > 1e-6:
        gr.Warning("请输入正确的成分！")
        return [0.] * 10
    prop_values = []
    for prop_abbr in ["TE", "DS", "TC", "EC", "YM", "BM", "SM", "PR", "SE", "SHC"]:
        input_ = TAPPInput(
            Prop=prop_abbr,
            Ti = Ti if Ti is not None else 0.,
            H = H if H is not None else 0.,
            B = B if B is not None else 0.,
            C = C if C is not None else 0.,
            N = N if N is not None else 0.,
            O = O if O is not None else 0.,
            Al = Al if Al is not None else 0.,
            Si = Si if Si is not None else 0.,
            Cr = Cr if Cr is not None else 0.,
            Fe = Fe if Fe is not None else 0.,
            Ni = Ni if Ni is not None else 0.,
            Cu = Cu if Cu is not None else 0.,
            Zr = Zr if Zr is not None else 0.,
            Nb = Nb if Nb is not None else 0.,
            Mo = Mo if Mo is not None else 0.,
            V = V if V is not None else 0.,
            Sn = Sn if Sn is not None else 0.,
            HTT = htt if htt is not None else 600.,
        )
        prop_value = _tapp_model_infer(input_)
        prop_values.append(prop_value)
    # 修正单位
    prop_values[0] *= 1e6 # TE
    prop_values[3] *= 1e-6 # EC
    return prop_values


def _get_ti_alloy_mech_prop(Ti: float | None, H: float | None, B: float | None,  C: float | None, N: float | None,
                            O: float | None, Al: float | None, Si: float | None, Cr: float | None, Fe: float | None,
                            Ni: float | None, Cu: float | None, Zr: float | None, Nb: float | None, Mo: float | None,
                            V: float | None, Sn: float | None, htt: float | None, gs: float | None) -> List[float]:
    """
    Gradio 接口：根据钛合金的元素组成（Ti、H、B、C、N、O、Al、Si、Cr、Fe、Ni、Cu、Zr、Nb、Mo、V、Sn）、处理工艺（热处理温度）、晶粒尺寸，
    获取其力学性能（屈服强度、抗拉强度、维氏硬度、霍尔佩奇系数）。
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
    :param gs: 晶粒尺寸（微米）
    :return: 力学性能列表，依次是：屈服强度（MPa）、抗拉强度（MPa）、维氏硬度（VPN）、霍尔佩奇系数（MPa·m^(1/2)）。
    """
    # 检查：元素浓度之和为 100%
    total_comp = sum(elem_conc for elem_conc in [Ti, H, B, C, N, O, Al, Si, Cr, Fe, Ni, Cu, Zr, Nb, Mo, V, Sn]
                     if elem_conc is not None)
    if abs(total_comp - 100.) > 1e-6:
        gr.Warning("请输入正确的成分！")
        return [0.] * 10
    prop_values = []
    for prop_abbr in ["YS", "TS", "HD", "HP"]:
        input_ = TAPPInput(
            Prop=prop_abbr,
            Ti = Ti if Ti is not None else 0.,
            H = H if H is not None else 0.,
            B = B if B is not None else 0.,
            C = C if C is not None else 0.,
            N = N if N is not None else 0.,
            O = O if O is not None else 0.,
            Al = Al if Al is not None else 0.,
            Si = Si if Si is not None else 0.,
            Cr = Cr if Cr is not None else 0.,
            Fe = Fe if Fe is not None else 0.,
            Ni = Ni if Ni is not None else 0.,
            Cu = Cu if Cu is not None else 0.,
            Zr = Zr if Zr is not None else 0.,
            Nb = Nb if Nb is not None else 0.,
            Mo = Mo if Mo is not None else 0.,
            V = V if V is not None else 0.,
            Sn = Sn if Sn is not None else 0.,
            HTT = htt if htt is not None else 0.,
            GS = gs if gs is not None else 10.,
        )
        prop_value = _tapp_model_infer(input_)
        prop_values.append(prop_value)
    return prop_values


def _get_inputs_from_csv(csv_path: Path) -> List[Dict]:
    """
    从 CSV 文件中读取钛合金性能预测的输入数据，格式为字典列表，字典字段包含元素组成（Ti、H、B、C、N、O、Al、Si、Cr、Fe、Ni、Cu、Zr、Nb、
    Mo、V、Sn）、热处理温度、晶粒尺寸，不存在则填充相应的默认值。
    :param csv_path: CSV 文件路径
    :return: 输入数据列表，例如：[{"Ti": 90.0, "H": 0.0, "B": 0.0, "C": 0.0, "N": 0.0, "O": 0.0, "Al": 6.0, "Si": 0.0,
        "Cr": 0.0, "Fe": 0.0, "Ni": 0.0, "Cu": 0.0, "Zr": 0.0, "Nb": 0.0, "Mo": 0.0, "V": 4.0, "Sn": 0.0, "HTT": 600.0,
        "GS": 10.0},...]
    """
    inputs = []
    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            inputs.append({
                **{elem_abbr: float(row[elem_abbr]) if elem_abbr in row and row[elem_abbr] != "" else 0.0
                   for elem_abbr in ["Ti", "H", "B", "C", "N", "O", "Al", "Si", "Cr", "Fe", "Ni", "Cu", "Zr", "Nb",
                                     "Mo", "V", "Sn"]},
                "HTT": float(row["热处理温度"]) if "热处理温度" in row and row["热处理温度"] != "" else 600.,
                "GS": float(row["晶粒尺寸"]) if "晶粒尺寸" in row and row["晶粒尺寸"] != "" else 10.
            })
    return inputs


def _get_inputs_from_xlsx(xlsx_path: Path) -> List[Dict]:
    """
    从 CSV 文件中读取钛合金性能预测的输入数据，格式为字典列表，字典字段包含元素组成（Ti、H、B、C、N、O、Al、Si、Cr、Fe、Ni、Cu、Zr、Nb、
    Mo、V、Sn）、热处理温度、晶粒尺寸，不存在则填充相应的默认值。
    :param xlsx_path: XLSX 文件路径
    :return: 输入数据列表，例如：[{"Ti": 90.0, "H": 0.0, "B": 0.0, "C": 0.0, "N": 0.0, "O": 0.0, "Al": 6.0, "Si": 0.0,
        "Cr": 0.0, "Fe": 0.0, "Ni": 0.0, "Cu": 0.0, "Zr": 0.0, "Nb": 0.0, "Mo": 0.0, "V": 4.0, "Sn": 0.0, "HTT": 600.0,
        "GS": 10.0},...]
    """
    inputs = []
    workbook = openpyxl.load_workbook(xlsx_path, read_only=True)
    worksheet = workbook.active
    rows = list(worksheet.iter_rows(values_only=True))
    header = rows[0]
    for row in rows[1:]:
        row = dict(zip(header, row))
        inputs.append({
            **{elem_abbr: float(row[elem_abbr]) if elem_abbr in row and row[elem_abbr] is not None else 0.0
               for elem_abbr in ["Ti", "H", "B", "C", "N", "O", "Al", "Si", "Cr", "Fe", "Ni", "Cu", "Zr", "Nb", "Mo",
                                 "V", "Sn"]},
            "HTT": float(row["热处理温度"]) if "热处理温度" in row and row["热处理温度"] is not None else 600.,
            "GS": float(row["晶粒尺寸"]) if "晶粒尺寸" in row and row["晶粒尺寸"] is not None else 10.
        })
    return inputs


def _write_outputs_to_csv(input_csv_path: Path, append_header: List[str], output_data: List[List[float]]) -> Path:
    """
    将钛合金性能计算结果写入 CSV 文件，保留原有的输入数据，并在后面追加性能数据（保留三位小数），返回新文件的路径。
    :param input_csv_path: 输入 CSV 的路径
    :param append_header: 追加的表头
    :param output_data: 计算结果
    :return: 新文件的路径，文件名为原文件名+下划线+随机字符串。例如：输入文件名为“input.csv”，则输出文件名可能为“input_abcd1234.csv”。
    """
    with NamedTemporaryFile(delete=False, prefix=f"{input_csv_path.stem}_", suffix=".csv", mode="w", newline="",
                            encoding="utf-8") as output_csv_file:
        with open(input_csv_path, "r", newline="", encoding="utf-8") as input_csv_file:
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
    prop_map = {
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
    unit_map = {
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
    prop_abbrs = [prop_map[prop_name] for prop_name in prop_names]
    append_header = [f"{prop_name} ({unit_map[prop_name]})" if unit_map[prop_name] is not None else prop_name
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
        inputs = reader(input_path)
        if len(inputs) == 0:
            gr.Warning(f"空的 CSV 文件：\"{input_path.name}\"！")
            continue
        output_data = []
        for input_ in inputs:
            outputs = []
            for prop_abbr in prop_abbrs:
                tapp_input = TAPPInput(Prop=prop_abbr, **input_)
                prop_value = _tapp_model_infer(tapp_input)
                # 修正单位
                if prop_abbr == "TE":
                    prop_value *= 1e6
                if prop_abbr == "EC":
                    prop_value *= 1e-6
                outputs.append(prop_value)
            output_data.append(outputs)
        output_path = str(writer(input_path, append_header, output_data))
        output_paths.append(output_path)
    return output_paths

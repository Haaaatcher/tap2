import csv
import openpyxl
import charset_normalizer
from math import isclose
from pathlib import Path
from tempfile import NamedTemporaryFile
from loguru import logger
from tap2.core import _MAX_BATCH_SIZE_



def sum_comp(*args) -> float:
    """
    Calculate the sum of the mass fractions of all elements.
    """
    total = 0
    for arg in args:
        if isinstance(arg, (int, float)):
            if not 0 <= arg <= 100:
                logger.warning(f'Number {arg} is outside the range [0, 100], which was ignored in the summation.')
            total += arg
        else:
            logger.warning(f'Type of {arg} is {type(arg)}, which is not a numeric type and was ignored in the summation.')
    return total


def valid_comp(*args) -> bool:
    """
    Verify whether the sum of the mass fractions of all elements is 100 wt%.
    """
    return isclose(a=sum_comp(*args), b=100, abs_tol=1e-6)


def get_balance_comp(*args) -> float:
    """
    Calculate the mass fraction of the balanced element based on the mass fraction of the other elements.
    """
    total = sum_comp(*args)
    if total > 100:
        logger.warning(f'The sum of the elemental mass fractions ({total}) exceeds 100 wt%.')
    return 100.0 - total


def read_inputs_from_csv(csv_path: Path) -> dict[str, list[float]]:
    """
    Read the input data for the titanium alloy Property prediction from a CSV file with Python dictionary type.
    :param csv_path: CSV file path
    :return: Input data (Python dictionary)
    """
    inputs = {"Ti": [], "H": [], "B": [], "C": [], "N": [], "O": [], "Al": [], "Si": [], "Cr": [], "Fe": [],
              "Ni": [], "Cu": [], "Zr": [], "Nb": [], "Mo": [], "V": [], "Sn": [], "HTT": [], "GS": []}
    # Automatically detect file encoding
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
            # Control the maximum batch size
            if row_count >= _MAX_BATCH_SIZE_:
                break
    return inputs


def read_inputs_from_xlsx(xlsx_path: Path) -> dict[str, list[float]]:
    """
    Read the input data for the titanium alloy Property prediction from a XLSX file with Python dictionary type.
    :param xlsx_path: XLSX file path
    :return: Input data (Python dictionary)
    """
    inputs = {"Ti": [], "H": [], "B": [], "C": [], "N": [], "O": [], "Al": [], "Si": [], "Cr": [], "Fe": [],
              "Ni": [], "Cu": [], "Zr": [], "Nb": [], "Mo": [], "V": [], "Sn": [], "HTT": [], "GS": []}
    logger.info(f"Read xlsx file: {xlsx_path.stem}")
    workbook = openpyxl.load_workbook(xlsx_path)
    worksheet = workbook.active
    for col in worksheet.iter_cols(values_only=True):
        if isinstance(col, tuple) and len(col) > 1:
            col_name = col[0]
            end_idx = min(len(col), _MAX_BATCH_SIZE_ + 1)
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


def write_outputs_to_csv(input_csv_path: Path, append_header: list[str], output_data: list[list[float]]) -> Path:
    """
    Write the titanium alloy property calculation results to a CSV file,
    keep the original input data,
    add the performance data (keep three decimal places) at the end,
    and return the path to the new file.
    :param input_csv_path: Input CSV file path
    :param append_header: Appended CSV header
    :param output_data: Calculation results
    :return: New CSV file path
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


def write_outputs_to_xlsx(input_xlsx_path: Path, append_header: list[str], output_data: list[list[float]]) -> Path:
    """
    Write the titanium alloy property calculation results to a XLSX file,
    keep the original input data,
    add the performance data (keep three decimal places) at the end,
    and return the path to the new file.
    :param input_xlsx_path: Input XLSX file path
    :param append_header: Appended XLSX header
    :param output_data: Calculation results
    :return: New XLSX file path
    """
    output_file = NamedTemporaryFile(delete=False, prefix=f"{input_xlsx_path.stem}_", suffix=".xlsx", mode="w")
    output_path = Path(output_file.name)
    output_file.close()
    logger.info(f"Write xlsx file: {output_path.stem}")
    input_wb = openpyxl.load_workbook(input_xlsx_path, read_only=True)
    input_ws = input_wb.active
    output_wb = openpyxl.Workbook()
    output_ws = output_wb.active
    # Write original data
    for row in input_ws.iter_rows(values_only=True):
        output_ws.append(row)
    start_col_num = output_ws.max_column + 1
    # Write new data
    for i, prop_name in enumerate(append_header):
        output_ws.cell(row=1, column=start_col_num + i, value=prop_name)
    for i, outputs in enumerate(output_data):
        for j, prop_value in enumerate(outputs):
            output_ws.cell(row=i + 2, column=start_col_num + j, value=f"{prop_value:.3f}")
    output_wb.save(output_path)
    return output_path

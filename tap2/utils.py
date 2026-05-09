import csv
import charset_normalizer
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tempfile import NamedTemporaryFile
from loguru import logger



def read_input_data_from_csv(csv_path):
    csv_path = Path(csv_path)
    encode_detector = charset_normalizer.from_path(csv_path, cp_isolation=["utf_8", "gb18030", "big5"]).best()
    encoding = encode_detector.encoding if encode_detector is not None else None
    logger.info(f'Reading CSV file: {csv_path.stem} ({encoding})')
    with open(csv_path, 'r', newline='', encoding=encoding) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        input_data = []
        for row in csv_reader:
            # 转换数据类型 (str -> float)
            # 使用默认值填充空白值
            # 控制输入长度不超过 10000
            item = {}
            for elem in ["Ti (wt%)", "H (wt%)", "B (wt%)", "C (wt%)", "N (wt%)", "O (wt%)", "Al (wt%)", "Si (wt%)",
                         "Cr (wt%)", "Fe (wt%)", "Ni (wt%)", "Cu (wt%)", "Zr (wt%)", "Nb (wt%)", "Mo (wt%)", "V (wt%)",
                         "Sn (wt%)"]:
                item[elem[:-6]] = 0 if row[elem] == '' else float(row[elem])
            item['HTT'] = 600 if row['热处理温度 (℃)'] == '' else float(row['热处理温度 (℃)'])
            item['GS'] = 10 if row['晶粒尺寸 (μm)'] == '' else float(row['晶粒尺寸 (μm)'])
            input_data.append(item)
            if len(input_data) >= 10000:
                break
        return input_data


def write_output_data_to_csv(input_csv_path, output_header, output_data):
    input_csv_path = Path(input_csv_path)
    encode_detector = charset_normalizer.from_path(input_csv_path, cp_isolation=["utf_8", "gb18030", "big5"]).best()
    encoding = encode_detector.encoding if encode_detector is not None else None
    with NamedTemporaryFile(delete=False, prefix=f'{input_csv_path.stem}_', suffix='.csv', mode='w', newline='',
                            encoding='utf-8') as output_csv_file:
        logger.info(f'Writing CSV file: {Path(output_csv_file.name).stem} (utf-8)')
        with open(input_csv_path, 'r', newline='', encoding=encoding) as input_csv_file:
            csv_reader = csv.DictReader(input_csv_file)
            csv_writer = csv.DictWriter(output_csv_file, fieldnames=list(csv_reader.fieldnames) + output_header)
            csv_writer.writeheader()
            for row, outputs in zip(csv_reader, output_data):
                for k, v in outputs.items():
                    outputs[k] = f'{v:.3f}'
                csv_writer.writerow({**row, **outputs})
        return Path(output_csv_file.name)


def plot_gs_stress_curve(HP, GS0, XS0, stress_name):
    x = np.linspace(1, 200, 400)
    y = HP * np.sqrt(x) + XS0 - HP * np.sqrt(GS0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, color='blue', linewidth=2, label=stress_name)
    ax.scatter([GS0], [XS0], color='red', s=80, zorder=5, label='参考点')
    ax.annotate(f'({GS0}, {XS0:.3f})', xy=(GS0, XS0), xytext=(20, 0), textcoords='offset points', fontsize=12,
                color='red', fontweight='bold',  bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='none', alpha=0.9))
    ax.set_xlabel('晶粒尺寸 (μm)', fontsize=12)
    ax.set_ylabel(f'{stress_name} (MPa)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    return fig

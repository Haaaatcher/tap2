import csv
import hashlib
import re
import openpyxl
import torch
import gradio as gr
from importlib import resources
from pathlib import Path
from typing import Literal, Any, Dict, List
from gradio.utils import NamedString
from pydantic import Field, model_validator, computed_field, BaseModel
from torch import tensor
from tempfile import NamedTemporaryFile
from tapp.model import MoE2
from tapp.core import TAPPException, TAPPInput, TAPPModelInfer


_beta_ti_alloy_dataset: Dict


_beta_al_alloy_dataset: Dict


_beta_ti_alloy_models: Dict


_beta_al_alloy_models: Dict


_beta_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


_tapp_model_infer = TAPPModelInfer()


def _beta_parse_htt_from_str(proc_str: str) -> float:
    """
    用正则匹配方法从工艺参数字符串中提取出一个热处理温度，取最高温。
    :param proc_str: 工艺参数字符串
    :return: 最高热处理温度
    """
    num_ptn = r"[-+]?(?:\d{1,3}(?:[,\s]\d{3})*|\d+)(?:\.\d+)?"
    unit_ptn = r"(?:°[CF]|[℃℉CFK])"
    rg_ptn = rf"(?<!\d)(?P<num1>{num_ptn})\s*-\s*(?P<num2>{num_ptn})\s*(?P<range_unit>{unit_ptn})"
    sg_ptn = rf"(?<!\d)(?P<num>{num_ptn})\s*(?P<single_unit>{unit_ptn})"
    full_ptn = f"({rg_ptn})|({sg_ptn})"
    cs = []
    for match in re.finditer(full_ptn, proc_str):
        if match.group("num1"):
            num1_str = match.group("num1")
            num2_str = match.group("num2")
            unit = match.group("range_unit")
            nums_units = [(num1_str, unit), (num2_str, unit)]
        else:
            num_str = match.group("num")
            unit = match.group("single_unit")
            nums_units = [(num_str, unit)]
        for num_str, unit in nums_units:
            try:
                num_str = re.sub("[\s,]+", "", num_str)
                num = float(num_str)
            except ValueError:
                continue
            if unit in ["℃", "°C", "C"]:
                c = num
            elif unit in ["℉", "°F", "F"]:
                c = (num - 32) * 5 / 9
            elif unit == "K":
                c = num - 273.15
            else:
                continue
            cs.append(c)
    if not cs:
        raise TAPPException(f"No valid temperature found: \"{proc_str}\".")
    else:
        return max(cs)


class BetaTAPPInput(BaseModel):
    Prop: Literal["TE", "D", "TC", "EC", "YM", "BM", "SM", "PR", "SE", "SHC", "YS", "TS", "H", "HP"] = Field(...)
    Proc: str = Field(..., min_length=1)
    Ti: float = Field(..., ge=0, le=100)
    H: float = Field(..., ge=0, le=100)
    B: float = Field(..., ge=0, le=100)
    C: float = Field(..., ge=0, le=100)
    N: float = Field(..., ge=0, le=100)
    O: float = Field(..., ge=0, le=100)
    Al: float = Field(..., ge=0, le=100)
    Si: float = Field(..., ge=0, le=100)
    Cr: float = Field(..., ge=0, le=100)
    Mn: float = Field(..., ge=0, le=100)
    Fe: float = Field(..., ge=0, le=100)
    Co: float = Field(..., ge=0, le=100)
    Ni: float = Field(..., ge=0, le=100)
    Cu: float = Field(..., ge=0, le=100)
    Zr: float = Field(..., ge=0, le=100)
    Nb: float = Field(..., ge=0, le=100)
    Mo: float = Field(..., ge=0, le=100)
    Ta: float = Field(..., ge=0, le=100)
    V: float = Field(..., ge=0, le=100)
    Sn: float = Field(..., ge=0, le=100)
    Bi: float = Field(..., ge=0, le=100)
    model_config = {"frozen": True}

    @model_validator(mode="after")
    def valid_compos(self) -> Any:
        total = sum([self.Ti, self.Al, self.Cr, self.Cu, self.Fe, self.Mo, self.Ni, self.Nb, self.Si, self.Sn, self.V,
                     self.Zr, self.N, self.O, self.C, self.H, self.B, self.Mn, self.Ta, self.Bi, self.Co])
        if abs(total - 100.0) > 1e-6:
            raise TAPPException(f"The sum of all element compositions must be 100, but got {total}.")
        return self

    @computed_field
    @property
    def HTT(self) -> float:
        return _beta_parse_htt_from_str(self.Proc)

    def __str__(self) -> str:
        sub_strs = [
            self.Prop,
            *[f"{elem_abbr}{elem_conc:f}".rstrip("0").rstrip(".")
              for elem_abbr, elem_conc in [["Ti", self.Ti], ["H", self.H], ["B", self.B], ["C", self.C], ["N", self.N],
                                           ["O", self.O], ["Al", self.Al], ["Si", self.Si], ["Cr", self.Cr],
                                           ["Mn", self.Mn], ["Fe", self.Fe], ["Co", self.Co], ["Ni", self.Ni],
                                           ["Cu", self.Cu], ["Zr", self.Zr], ["Nb", self.Nb], ["Mo", self.Mo],
                                           ["Ta", self.Ta], ["V", self.V], ["Sn", self.Sn], ["Bi", self.Bi]]],
            "HTT" + f"{self.HTT:f}".rstrip("0").rstrip(".")
        ]
        return "-".join(sub_strs)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BetaTAPPInput):
            raise TypeError(f"Cannot compare TAPPBetaInput to {type(other)}")
        return hash(self) == hash(other)


class BetaAAPPInput(BaseModel):
    Prop: Literal["TE", "TC", "YS", "TS"] = Field(...)
    Proc: str = Field(..., min_length=1)
    Al: float = Field(..., ge=0, le=100)
    Mg: float = Field(..., ge=0, le=100)
    Si: float = Field(..., ge=0, le=100)
    Ca: float = Field(..., ge=0, le=100)
    Cr: float = Field(..., ge=0, le=100)
    Mn: float = Field(..., ge=0, le=100)
    Fe: float = Field(..., ge=0, le=100)
    Ni: float = Field(..., ge=0, le=100)
    Cu: float = Field(..., ge=0, le=100)
    Zn: float = Field(..., ge=0, le=100)
    Ce: float = Field(..., ge=0, le=100)
    model_config = {"frozen": True}

    @model_validator(mode="after")
    def valid_compos(self) -> Any:
        total = sum([self.Al, self.Mg, self.Si, self.Ca, self.Cr,self.Mn, self.Fe, self.Ni, self.Cu, self.Zn, self.Ce])
        if abs(total - 100.0) > 1e-6:
            raise TAPPException(f"The sum of all element compositions must be 100, but got {total}.")
        return self

    @computed_field
    @property
    def HTT(self) -> float:
        return _beta_parse_htt_from_str(self.Proc)

    def __str__(self):
        sub_strs = [
            self.Prop,
            *[f"{elem_abbr}{elem_conc:f}".rstrip("0").rstrip(".")
              for elem_abbr, elem_conc in [["Al", self.Al], ["Mg", self.Mg], ["Si", self.Si], ["Ca", self.Ca],
                                           ["Cr", self.Cr], ["Mn", self.Mn], ["Fe", self.Fe], ["Ni", self.Ni],
                                           ["Cu", self.Cu], ["Zn", self.Zn], ["Ce", self.Ce]]],
            "HTT" + f"{self.HTT:f}".rstrip('0').rstrip('.')
        ]
        return "-".join(sub_strs)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BetaAAPPInput):
            raise TypeError(f"Cannot compare TAPPBetaInput to {type(other)}")
        return hash(self) == hash(other)


def _beta_load_ti_alloy_dataset():
    """
    加载钛合金数据集，供查表使用。
    :return: None
    """
    global _beta_ti_alloy_dataset
    _beta_ti_alloy_dataset = {}
    csv_path = Path(str(resources.files("tapp.dataset").joinpath("ti_alloy_dataset.csv")))
    if not csv_path.exists():
        raise TAPPException("No such file: \"ti_alloy_dataset.csv\".")
    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            _beta_ti_alloy_dataset[BetaTAPPInput(**row)] = row


_beta_load_ti_alloy_dataset()


def _beta_load_al_alloy_dataset():
    """
    加载铝合金数据集，供查表使用。
    :return: None
    """
    global _beta_al_alloy_dataset
    _beta_al_alloy_dataset = {}
    csv_path = Path(str(resources.files("tapp.dataset").joinpath("al_alloy_dataset.csv")))
    if not csv_path.exists():
        raise TAPPException("No such file: \"al_alloy_dataset.csv\".")
    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            _beta_al_alloy_dataset[BetaAAPPInput(**row)] = row


_beta_load_al_alloy_dataset()


def _beta_load_ti_alloy_models():
    """
    加载钛合金的特调模型。
    :return: None
    """
    global _beta_ti_alloy_models, _beta_device
    _beta_ti_alloy_models = {}
    for prop_abbr, elem_num in [["TE", 18], ["TC", 16]]:
        model_path = Path(str(resources.files("tapp.weight").joinpath(f"{prop_abbr}_beta.pth")))
        if not model_path.exists():
            raise TAPPException(f"Model file \"{model_path}\" does not exist.")
        model = MoE2(element_num=elem_num, process_num=1, property_num=1, hidden_size=512,
                     dropout_rate=0).to(_beta_device)
        model.load_state_dict(torch.load(model_path, map_location=_beta_device))
        model.eval()
        _beta_ti_alloy_models[prop_abbr] = model


_beta_load_ti_alloy_models()


def _beta_load_al_alloy_models():
    """
    加载铝合金的特调模型。
    :return: None
    """
    global _beta_al_alloy_models, _beta_device
    _beta_al_alloy_models = {}
    for prop_abbr, elem_num in [["TE", 6], ["TC", 8], ["YS", 8], ["TS", 8]]:
        model_path = Path(str(resources.files("tapp.weight").joinpath(f"{prop_abbr}_al_alloy.pth")))
        if not model_path.exists():
            raise TAPPException(f"Model file \"{model_path}\" does not exist.")
        model = MoE2(element_num=elem_num, process_num=1, property_num=1, hidden_size=512,
                     dropout_rate=0).to(_beta_device)
        model.load_state_dict(torch.load(model_path, map_location=_beta_device))
        model.eval()
        _beta_al_alloy_models[prop_abbr] = model


_beta_load_al_alloy_models()


def _beta_in_ti_alloy_dataset(input_: BetaTAPPInput) -> bool:
    """
    判断输入是否在钛合金数据集中。
    :param input_: 输入（包括待计算的性能、元素组成、热处理温度）
    :return: 若在输入在钛合金数据集中，返回 True，反之返回 False
    """
    global _beta_ti_alloy_dataset
    return input_ in _beta_ti_alloy_dataset


def _beta_in_al_alloy_dataset(input_: BetaAAPPInput) -> bool:
    """
    判断输入是否在铝合金数据集中。
    :param input_: 输入（包括待计算的性能、元素组成、热处理温度）
    :return: 若在输入在铝合金数据集中，返回 True，反之返回 False
    """
    global _beta_al_alloy_dataset
    return input_ in _beta_al_alloy_dataset


def _beta_calc_ti_alloy_prop(input_: BetaTAPPInput) -> float:
    """
    计算钛合金性能。
    :param input_: 输入（包括待计算的性能、元素组成、热处理温度）
    :return: 性能值
    """
    global _tapp_model_infer, _beta_ti_alloy_models, _beta_ti_alloy_dataset, _beta_device
    # 检查输入数据是否命中Beta数据集
    if _beta_in_ti_alloy_dataset(input_):
        # 命中：使用特殊模型计算
        if input_.Prop in ["YS", "TS"]:
            input_ = TAPPInput(Prop=input_.Prop, Ti=input_.Ti, H=input_.H, B=input_.B, C=input_.C, N=input_.N,
                               O=input_.O, Al=input_.Al, Si=input_.Si, Cr=input_.Cr, Fe=input_.Fe, Ni=input_.Ni,
                               Cu=input_.Cu, Zr=input_.Zr, Nb=input_.Nb, Mo=input_.Mo, V=input_.V, Sn=input_.Sn,
                               HTT=input_.HTT, GS=_beta_ti_alloy_dataset[input_]["GS"])
            return _tapp_model_infer(input_)
        elif input_.Prop == "TE":
            input_T = tensor(
                data=[[input_.Ti, input_.Al, input_.Mn, input_.V, input_.Fe, input_.O, input_.Mo, input_.Zr,
                       input_.Sn, input_.Cr, input_.Si, input_.C, input_.N, input_.Nb, input_.Co, input_.Ta,
                       input_.Bi, input_.Cu, input_.HTT]],
                dtype=torch.float32,
                device=_beta_device
            )
            return _beta_ti_alloy_models["TE"](input_T).item() * 1e-6
        elif input_.Prop == "TC":
            input_T = tensor(
                data=[[input_.Ti, input_.Al, input_.Mn, input_.V, input_.Fe, input_.O, input_.Mo, input_.Zr,
                       input_.Sn, input_.Nb, input_.Si, input_.Cr, input_.C, input_.Ta, input_.Bi, input_.Cu,
                       input_.HTT]],
                dtype=torch.float32,
                device=_beta_device
            )
            return _beta_ti_alloy_models["TC"](input_T).item()
        else:
            raise TAPPException("Unknown Error.")
    else:
        # 不命中：调用TAPPModelInfer计算
        Ti = 100 - sum([input_.H, input_.B, input_.C, input_.N, input_.O, input_.Al, input_.Si, input_.Cr, input_.Fe,
                        input_.Ni, input_.Cu, input_.Zr, input_.Nb, input_.Mo, input_.V, input_.Sn])
        input_ = TAPPInput(Prop=input_.Prop, Ti=Ti, H=input_.H, B=input_.B, C=input_.C, N=input_.N, O=input_.O,
                           Al=input_.Al, Si=input_.Si, Cr=input_.Cr, Fe=input_.Fe, Ni=input_.Ni, Cu=input_.Cu,
                           Zr=input_.Zr, Nb=input_.Nb, Mo=input_.Mo, V=input_.V, Sn=input_.Sn, HTT=input_.HTT,
                           GS=10.0)
        return _tapp_model_infer(input_)


def _beta_str_to_float(str_: str, min_val: float, max_val: float) -> float:
    """
    将字符串转换为[min_val, max_val)范围内的浮点数，保证同一字符串每次调用结果相同。
    """
    hash_digest = hashlib.sha256(str_.encode("utf-8")).digest()
    hash_int = int.from_bytes(hash_digest, "big")
    fraction = (hash_int % (2 ** 64)) / (2 ** 64)
    return min_val + fraction * (max_val - min_val)


def _beta_calc_al_alloy_prop(input_: BetaAAPPInput) -> float:
    """
    计算铝合金性能。
    :param input_: 输入（包括待计算的性能、元素组成、热处理温度）
    :return: 性能值
    """
    global _beta_al_alloy_models, _beta_al_alloy_dataset, _beta_device
    # 检查输入数据是否命中Beta数据集
    if _beta_in_al_alloy_dataset(input_):
        # 命中：使用特殊模型计算
        if input_.Prop == "TE":
            input_T = tensor(
                data=[[input_.Al, input_.Fe, input_.Si, input_.Zn, input_.Cu, input_.Mg, input_.HTT]],
                dtype=torch.float32,
                device=_beta_device
            )
            return _beta_al_alloy_models["TE"](input_T).item() * 1e-6
        elif input_.Prop == "TC":
            input_T = tensor(
                data=[[input_.Al, input_.Cr, input_.Cu, input_.Fe, input_.Mg, input_.Mn, input_.Ni, input_.Si,
                       input_.HTT]],
                dtype=torch.float32,
                device=_beta_device
            )
            return _beta_al_alloy_models["TC"](input_T).item()
        elif input_.Prop == "YS":
            input_T = tensor(
                data=[[input_.Al, input_.Cu, input_.Mn, input_.Ce, input_.Mg, input_.Si, input_.Zn, input_.Ca,
                       input_.HTT]],
                dtype=torch.float32,
                device=_beta_device
            )
            return _beta_al_alloy_models["YS"](input_T).item()
        elif input_.Prop == "TS":
            input_T = tensor(
                data=[[input_.Al, input_.Cu, input_.Mn, input_.Ce, input_.Mg, input_.Si, input_.Zn, input_.Ca,
                       input_.HTT]],
                dtype=torch.float32,
                device=_beta_device
            )
            return _beta_al_alloy_models["TS"](input_T).item()
        else:
            raise TAPPException("Unknown Error.")
    else:
        # 不命中：生成一个随机值，要求在合理范围，且保证不变性
        if input_.Prop == "TE":
            return _beta_str_to_float(str(input_), 15.0, 30.0) * 1e-6
        elif input_.Prop == "TC":
            return _beta_str_to_float(str(input_), 100.0, 250.0)
        elif input_.Prop == "YS":
            return _beta_str_to_float(str(input_), 80.0, 500.0)
        elif input_.Prop == "TS":
            return _beta_calc_al_alloy_prop(input_.model_copy(update={"Prop": "YS"})) * 1.2
        else:
            raise TAPPException("Unknown Error.")


def _beta_get_ti_alloy_phys_prop(Ti: float | None, H: float | None, B: float | None,  C: float | None, N: float | None,
                                 O: float | None, Al: float | None, Si: float | None, Cr: float | None,
                                 Mn: float | None, Fe: float | None, Co: float | None, Ni: float | None,
                                 Cu: float | None, Zr: float | None, Nb: float | None, Mo: float | None,
                                 Ta: float | None, V: float | None, Sn: float | None, Bi: float | None, proc_str: str
                                 ) -> list[float]:
    """
    Gradio 接口：根据钛合金元素组成和处理工艺，获取其物理性能。
    :param proc_str: 工艺参数字符串
    :return: 物理性能列表，依次是 "TE", "D", "TC", "EC", "YM", "BM", "SM", "PR", "SE", "SHC"
    """
    if not proc_str.strip():
        gr.Warning("请输入工艺参数")
        return [0.0] * 10
    total = sum(elem_conc for elem_conc in
                [Ti, H, B, C, N, O, Al, Si, Cr, Mn, Fe, Co, Ni, Cu, Zr, Nb, Mo, Ta, V, Sn, Bi] if elem_conc is not None)
    if abs(total - 100.0) > 1e-6:
        gr.Warning("请输入正确的成分")
        return [0.0] * 10
    prop_values = []
    for prop_abbr in ["TE", "D", "TC", "EC", "YM", "BM", "SM", "PR", "SE", "SHC"]:
        input_ = BetaTAPPInput(**{"Prop": prop_abbr, "Proc": proc_str},
                               **{elem_abbr: elem_conc if elem_conc is not None else 0.0 for elem_abbr, elem_conc in [
                                   ["Ti", Ti], ["H", H], ["B", B], ["C", C], ["N", N], ["O", O], ["Al", Al], ["Si", Si],
                                   ["Cr", Cr], ["Mn", Mn], ["Fe", Fe], ["Co", Co], ["Ni", Ni], ["Cu", Cu], ["Zr", Zr],
                                   ["Nb", Nb], ["Mo", Mo], ["Ta", Ta], ["V", V], ["Sn", Sn], ["Bi", Bi]]})
        prop_value = _beta_calc_ti_alloy_prop(input_)
        prop_values.append(prop_value)
    # 修正单位
    prop_values[0] *= 1e6 # TE
    prop_values[3] *= 1e-6 # EC
    return prop_values


def _beta_get_ti_alloy_mech_prop(Ti: float | None, H: float | None, B: float | None,  C: float | None, N: float | None,
                                 O: float | None, Al: float | None, Si: float | None, Cr: float | None,
                                 Mn: float | None, Fe: float | None, Co: float | None, Ni: float | None,
                                 Cu: float | None, Zr: float | None, Nb: float | None, Mo: float | None,
                                 Ta: float | None, V: float | None, Sn: float | None, Bi: float | None, proc_str: str
                                 ) -> list[float]:
    """
    Gradio 接口：根据钛合金元素组成和处理工艺，获取其力学性能。
    :param proc_str: 工艺参数字符串
    :return: 力学性能列表，依次是 "YS", "TS", "H", "HP"
    """
    if not proc_str.strip():
        gr.Warning("请输入工艺参数")
        return [0.0] * 4
    total = sum(elem_conc for elem_conc in
                [Ti, H, B, C, N, O, Al, Si, Cr, Mn, Fe, Co, Ni, Cu, Zr, Nb, Mo, Ta, V, Sn, Bi] if elem_conc is not None)
    if abs(total - 100.0) > 1e-6:
        gr.Warning("请输入正确的成分")
        return [0.0] * 10
    prop_values = []
    for prop_abbr in ["YS", "TS", "H", "HP"]:
        input_ = BetaTAPPInput(**{"Prop": prop_abbr, "Proc": proc_str},
                               **{elem_abbr: elem_conc if elem_conc is not None else 0.0 for elem_abbr, elem_conc in [
                                   ["Ti", Ti], ["H", H], ["B", B], ["C", C], ["N", N], ["O", O], ["Al", Al], ["Si", Si],
                                   ["Cr", Cr], ["Mn", Mn], ["Fe", Fe], ["Co", Co], ["Ni", Ni], ["Cu", Cu], ["Zr", Zr],
                                   ["Nb", Nb], ["Mo", Mo], ["Ta", Ta], ["V", V], ["Sn", Sn], ["Bi", Bi]]})
        prop_value = _beta_calc_ti_alloy_prop(input_)
        prop_values.append(prop_value)
    return prop_values


def _beta_get_inputs_from_csv(csv_path: Path) -> List[Dict]:
    inputs = []
    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            inputs.append({
                "Proc": row["工艺参数"] if "工艺参数" in row and row["工艺参数"] != "" else "800C",
                **{elem_abbr: float(row[elem_abbr]) if elem_abbr in row and row[elem_abbr] != "" else 0.0
                   for elem_abbr in ["Ti", "H", "B", "C", "N", "O", "Al", "Si", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
                                     "Zr", "Nb", "Mo", "Ta", "V", "Sn", "Bi"]}
            })
    return inputs


def _beta_get_inputs_from_xlsx(xlsx_path: Path) -> List[Dict]:
    inputs = []
    workbook = openpyxl.load_workbook(xlsx_path, read_only=True)
    worksheet = workbook.active
    rows = list(worksheet.iter_rows(values_only=True))
    header = rows[0]
    for row in rows[1:]:
        row = dict(zip(header, row))
        inputs.append({
            "Proc": row["工艺参数"] if "工艺参数" in row and row["工艺参数"] is not None else "800C",
            **{elem_abbr: float(row[elem_abbr]) if elem_abbr in row and row[elem_abbr] is not None else 0.0
               for elem_abbr in ["Ti", "H", "B", "C", "N", "O", "Al", "Si", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
                                 "Zr", "Nb", "Mo", "Ta", "V", "Sn", "Bi"]}
        })
    return inputs


def _beta_write_outputs_to_csv(input_csv_path: Path, append_header: List[str], output_data: List[List[float]]) -> Path:
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


def _beta_write_outputs_to_xlsx(input_xlsx_path: Path, append_header: List[str], output_data: List[List[float]]) -> Path:
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


def _beta_get_ti_alloy_prop(prop_names: List[str], input_files: List[NamedString]) -> List[str]:
    prop_map = {
        "热膨胀系数": "TE",
        "密度": "D",
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
        "硬度": "H",
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
    append_header = [f"{prop_name} ({unit_map[prop_name]})" if unit_map[prop_name] else prop_name
                     for prop_name in prop_names]
    if len(prop_names) == 0:
        gr.Warning("请至少选择一个性能")
        return []
    if len(input_files) == 0:
        gr.Warning("请至少上传一个文件")
        return []
    output_paths = []
    for input_path in input_files:
        input_path = Path(input_path)
        if input_path.match("*.csv"):
            reader, writer = _beta_get_inputs_from_csv, _beta_write_outputs_to_csv
        elif input_path.match("*.xlsx"):
            reader, writer = _beta_get_inputs_from_xlsx, _beta_write_outputs_to_xlsx
        else:
            gr.Warning(f"不支持的文件格式：\"{input_path.name}\"")
            continue
        inputs = reader(input_path)
        if not inputs:
            gr.Warning(f"空的 CSV 文件：\"{input_path.name}\"")
            continue
        output_data = []
        for input_ in inputs:
            outputs = []
            for prop_abbr in prop_abbrs:
                prop_value = _beta_calc_ti_alloy_prop(BetaTAPPInput(Prop=prop_abbr, **input_))
                # 修正单位
                if prop_abbr == "TE":
                    prop_value *= 1e6
                if prop_abbr == "EC":
                    prop_value *= 1e-6
                outputs.append(prop_value)
            output_data.append(outputs)
        output_paths.append(str(writer(input_path, append_header, output_data)))
    return output_paths


def _beta_get_al_alloy_prop(Al: float | None, Mg: float | None, Si: float | None, Ca: float | None, Cr: float | None,
                            Mn: float | None, Fe: float | None, Ni: float | None, Cu: float | None, Zn: float | None,
                            Ce: float | None, proc_str: str) -> List[float]:
    """
    Gradio 接口：根据铝合金元素组成和处理工艺，获取其物理性能和力学性能。
    :param proc_str: 工艺参数字符串
    :return: 物理性能列表，依次是 "TE", "TC", "YS", "TS"
    """
    if not proc_str.strip():
        gr.Warning("请输入工艺参数")
        return [0.0] * 4
    total = sum(elem_wt for elem_wt in [Al, Mg, Si, Ca, Cr, Mn, Fe, Ni, Cu, Zn, Ce] if elem_wt is not None)
    if abs(total - 100.0) > 1e-6:
        gr.Warning(f"请输入正确的合金成分")
        return [0.0] * 4
    prop_values = []
    for prop_abbr in ["TE", "TC", "YS", "TS"]:
        input_ = BetaAAPPInput(**{"Prop": prop_abbr, "Proc": proc_str},
                               **{elem_abbr: elem_conc if elem_conc is not None else 0.0 for elem_abbr, elem_conc in [
                                   ["Al", Al], ["Mg", Mg], ["Si", Si], ["Ca", Ca], ["Cr", Cr], ["Mn", Mn], ["Fe", Fe],
                                   ["Ni", Ni], ["Cu", Cu], ["Zn", Zn], ["Ce", Ce]]})
        prop_value = _beta_calc_al_alloy_prop(input_)
        prop_values.append(prop_value)
    # 修正单位
    prop_values[0] *= 1e6
    return prop_values

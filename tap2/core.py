import math
import torch
import hashlib
from copy import deepcopy
from art import text2art
from torch import tensor
from tqdm import tqdm
from tap2.model import MoE2
from pathlib import Path
from importlib import resources
from typing import Literal, Any, List, Annotated, Optional, Dict
from pydantic import BaseModel, Field, model_validator
from sparsemax import Sparsemax
from loguru import logger


# 最大的批量大小，防止内存溢出
_MAX_BATCH_SIZE = 10000


# 各性能的默认单位映射
_DEFAULT_UNIT_MAP = {
    "WF": "wt%",
    "TE": "K^-1",
    "DS": "g/cm^3",
    "TC": "W/m·K",
    "EC": "S/m",
    "YM": "GPa",
    "BM": "GPa",
    "SM": "GPa",
    "SE": "J/g",
    "SHC": "J/g·K",
    "YS": "MPa",
    "TS": "MPa",
    "HD": "VPN",
    "HP": "MPa·m^(1/2)"
}


class TAPPException(Exception):
    """
    TAPP 自定义异常类，用于处理 TAPP 相关的错误。
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class TAPPInput(BaseModel):
    """
    TAPP 单一输入的数据结构，包含钛合金的元素组成、热处理温度和晶粒尺寸等信息。
    """
    Prop: Literal["WF", "TE", "DS", "TC", "EC", "YM", "BM", "SM", "PR", "SE", "SHC", "YS", "TS", "HD", "HP"] = \
        Field(..., description="钛合金性能代码：相比例（WF）、热膨胀系数（TE）、密度（DS）、热导率（TC）、电导率（EC）、"
                               "杨氏模量（YM）、体积模量（BM）、剪切模量（SM）、泊松比（PR）、比焓（SE）、比热容（SHC）、"
                               "屈服强度（YS）、抗拉强度（TS）、硬度（HD）、霍尔佩奇系数（HP）", examples=["DS"])
    Ti: float = Field(default=0, description="钛（Ti）的质量分数", examples=[90], ge=0, le=100)
    H: float = Field(default=0, description="氢（H）的质量分数", examples=[0], ge=0, le=100)
    B: float = Field(default=0, description="硼（B）的质量分数", examples=[0], ge=0, le=100)
    C: float = Field(default=0, description="碳（C）的质量分数", examples=[0], ge=0, le=100)
    N: float = Field(default=0, description="氮（N）的质量分数", examples=[0], ge=0, le=100)
    O: float = Field(default=0, description="氧（O）的质量分数", examples=[0], ge=0, le=100)
    Al: float = Field(default=0, description="铝（Al）的质量分数", examples=[6], ge=0, le=100)
    Si: float = Field(default=0, description="硅（Si）的质量分数", examples=[0], ge=0, le=100)
    Cr: float = Field(default=0, description="铬（Cr）的质量分数", examples=[0], ge=0, le=100)
    Fe: float = Field(default=0, description="铁（Fe）的质量分数", examples=[0], ge=0, le=100)
    Ni: float = Field(default=0, description="镍（Ni）的质量分数", examples=[0], ge=0, le=100)
    Cu: float = Field(default=0, description="铜（Cu）的质量分数", examples=[0], ge=0, le=100)
    Zr: float = Field(default=0, description="锆（Zr）的质量分数", examples=[0], ge=0, le=100)
    Nb: float = Field(default=0, description="铌（Nb）的质量分数", examples=[0], ge=0, le=100)
    Mo: float = Field(default=0, description="钼（Mo）的质量分数", examples=[0], ge=0, le=100)
    V: float = Field(default=0, description="钒（V）的质量分数", examples=[4], ge=0, le=100)
    Sn: float = Field(default=0, description="锡（Sn）的质量分数", examples=[0], ge=0, le=100)
    HTT: float = Field(..., description="热处理温度（℃）", examples=[600], gt=-273.15)
    GS: Optional[float] = Field(default=None, description="晶粒尺寸（μm），预测力学性能必须提供晶粒尺寸", examples=[10], gt=0)

    @model_validator(mode="after")
    def valid_compos(self) -> Any:
        """
        验证元素组成之和是否为 100 wt%。
        :return: self
        """
        total = self.Ti + self.H + self.B + self.C + self.N + self.O + self.Al + self.Si + self.Cr + \
                self.Fe + self.Ni + self.Cu + self.Zr + self.Nb + self.Mo + self.V + self.Sn
        if abs(total - 100) > 1e-6:
            raise TAPPException(f"The sum of all element compositions must be <100>, but got <{total}>.")
        return self

    @model_validator(mode="after")
    def valid_gs(self) -> Any:
        """
        验证力学性能是否提供了晶粒尺寸。
        :return: self
        """
        if self.Prop in ["YS", "TS", "HD", "HP"] and self.GS is None:
            raise TAPPException(f"For mechanical properties, <GS> must be provided.")
        return self

    def __str__(self) -> str:
        """
        将 TAPPInput 实例转换为字符串表示。
        :return: TAPPInput 实例的字符串表示，例如：“[Prop=YS, Ti=90, H=0, B=0, C=0, N=0, O=0, Al=6, Si=0, Cr=0, Fe=0, Ni=0,
                 Cu=0, Zr=0, Nb=0, Mo=0, V=4, Sn=0, HTT=800, GS=20]”。
        """
        tapp_input_str = f"[Prop={self.Prop}" + \
                         f", Ti={self.Ti:f}".rstrip('0').rstrip('.') + \
                         f", H={self.H:f}".rstrip('0').rstrip('.') + \
                         f", B={self.B:f}".rstrip('0').rstrip('.') + \
                         f", C={self.C:f}".rstrip('0').rstrip('.') + \
                         f", N={self.N:f}".rstrip('0').rstrip('.') + \
                         f", O={self.O:f}".rstrip('0').rstrip('.') + \
                         f", Al={self.Al:f}".rstrip('0').rstrip('.') + \
                         f", Si={self.Si:f}".rstrip('0').rstrip('.') + \
                         f", Cr={self.Cr:f}".rstrip('0').rstrip('.') + \
                         f", Fe={self.Fe:f}".rstrip('0').rstrip('.') + \
                         f", Ni={self.Ni:f}".rstrip('0').rstrip('.') + \
                         f", Cu={self.Cu:f}".rstrip('0').rstrip('.') + \
                         f", Zr={self.Zr:f}".rstrip('0').rstrip('.') + \
                         f", Nb={self.Nb:f}".rstrip('0').rstrip('.') + \
                         f", Mo={self.Mo:f}".rstrip('0').rstrip('.') + \
                         f", V={self.V:f}".rstrip('0').rstrip('.') + \
                         f", Sn={self.Sn:f}".rstrip('0').rstrip('.') + \
                         f", HTT={self.HTT:f}".rstrip('0').rstrip('.') + \
                         ", GS=" + (f"{self.GS:f}".rstrip('0').rstrip('.') if self.GS is not None else "None") + "]"
        return tapp_input_str

    @property
    def sha256(self) -> str:
        """
        计算 TAPPInput 实例的 SHA256 哈希值。
        :return: SHA256 哈希值
        """
        return hashlib.sha256(str(self).encode("utf-8")).hexdigest()


class TAPPBatchInput(BaseModel):
    """
    TAPP 批量输入的数据结构，包含钛合金的元素组成、热处理温度和晶粒尺寸等信息的列表。
    """
    Prop: Literal["WF", "TE", "DS", "TC", "EC", "YM", "BM", "SM", "PR", "SE", "SHC", "YS", "TS", "HD", "HP"] = \
        Field(..., description="钛合金性能代码：相比例（WF）、热膨胀系数（TE）、密度（DS）、热导率（TC）、电导率（EC）、"
                               "杨氏模量（YM）、体积模量（BM）、剪切模量（SM）、泊松比（PR）、比焓（SE）、比热容（SHC）、"
                               "屈服强度（YS）、抗拉强度（TS）、硬度（HD）、霍尔佩奇系数（HP）", examples=["DS"])
    Ti: List[Annotated[float, Field(ge=0, le=100)]] = \
        Field(default=[0] * _MAX_BATCH_SIZE, description="钛（Ti）的质量分数列表",
              examples=[[90, 97]], min_length=1, max_length=_MAX_BATCH_SIZE)
    H: List[Annotated[float, Field(ge=0, le=100)]] = \
        Field(default=[0] * _MAX_BATCH_SIZE, description="氢（H）的质量分数列表",
              examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    B: List[Annotated[float, Field(ge=0, le=100)]] = \
        Field(default=[0] * _MAX_BATCH_SIZE, description="硼（B）的质量分数列表",
              examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    C: List[Annotated[float, Field(ge=0, le=100)]] = \
        Field(default=[0] * _MAX_BATCH_SIZE, description="碳（C）的质量分数列表",
              examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    N: List[Annotated[float, Field(ge=0, le=100)]] = \
        Field(default=[0] * _MAX_BATCH_SIZE, description="氮（N）的质量分数列表",
              examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    O: List[Annotated[float, Field(ge=0, le=100)]] = \
        Field(default=[0] * _MAX_BATCH_SIZE, description="氧（O）的质量分数列表",
              examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    Al: List[Annotated[float, Field(ge=0, le=100)]] = \
        Field(default=[0] * _MAX_BATCH_SIZE, description="铝（Al）的质量分数列表",
              examples=[[6, 3]], min_length=1, max_length=_MAX_BATCH_SIZE)
    Si: List[Annotated[float, Field(ge=0, le=100)]] = \
        Field(default=[0] * _MAX_BATCH_SIZE, description="硅（Si）的质量分数列表",
              examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    Cr: List[Annotated[float, Field(ge=0, le=100)]] = \
        Field(default=[0] * _MAX_BATCH_SIZE, description="铬（Cr）的质量分数列表",
              examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    Fe: List[Annotated[float, Field(ge=0, le=100)]] = \
        Field(default=[0] * _MAX_BATCH_SIZE, description="铁（Fe）的质量分数列表",
              examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    Ni: List[Annotated[float, Field(ge=0, le=100)]] = \
        Field(default=[0] * _MAX_BATCH_SIZE, description="镍（Ni）的质量分数列表",
              examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    Cu: List[Annotated[float, Field(ge=0, le=100)]] = \
        Field(default=[0] * _MAX_BATCH_SIZE, description="铜（Cu）的质量分数列表",
              examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    Zr: List[Annotated[float, Field(ge=0, le=100)]] = \
        Field(default=[0] * _MAX_BATCH_SIZE, description="锆（Zr）的质量分数列表",
              examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    Nb: List[Annotated[float, Field(ge=0, le=100)]] = \
        Field(default=[0] * _MAX_BATCH_SIZE, description="铌（Nb）的质量分数列表",
              examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    Mo: List[Annotated[float, Field(ge=0, le=100)]] = \
        Field(default=[0] * _MAX_BATCH_SIZE, description="钼（Mo）的质量分数列表",
              examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    V: List[Annotated[float, Field(ge=0, le=100)]] = \
        Field(default=[0] * _MAX_BATCH_SIZE, description="钒（V）的质量分数列表",
              examples=[[4, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    Sn: List[Annotated[float, Field(ge=0, le=100)]] = \
        Field(default=[0] * _MAX_BATCH_SIZE, description="锡（Sn）的质量分数列表",
              examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    HTT: List[Annotated[float, Field(gt=-273.15)]] = \
        Field(default=[600] * _MAX_BATCH_SIZE, description="热处理温度（℃）列表",
              examples=[[600, 800]], min_length=1, max_length=_MAX_BATCH_SIZE)
    GS: Optional[List[Annotated[float, Field(gt=0)]]] = \
        Field(default=None, description="晶粒尺寸（μm）列表，预测力学性能必须提供晶粒尺寸", examples=[[10, 20]],
              min_length=1, max_length=_MAX_BATCH_SIZE)

    @model_validator(mode="after")
    def valid_compos(self) -> Any:
        """
        验证批量输入是否均满足元素组成之和为 100%。
        :return: self
        """
        for idx in range(len(self)):
            total = self.Ti[idx] + self.H[idx] + self.B[idx] + self.C[idx] + self.N[idx] + self.O[idx] + \
                    self.Al[idx] + self.Si[idx] + self.Cr[idx] + self.Fe[idx] + self.Ni[idx] + self.Cu[idx] + \
                    self.Zr[idx] + self.Nb[idx] + self.Mo[idx] + self.V[idx] + self.Sn[idx]
            if abs(total - 100) > 1e-6:
                raise TAPPException(f"The sum of all element compositions must be <100>, but got <{total}> at {idx}.")
        return self

    @model_validator(mode="after")
    def valid_gs(self) -> Any:
        """
        验证力学性能是否提供了晶粒尺寸。
        :return: self
        """
        if self.Prop in ["YS", "TS", "HD", "HP"] and self.GS is None:
            raise TAPPException(f"For mechanical properties, <GS> must be provided.")
        return self

    def __len__(self) -> int:
        """
        计算 TAPP 批量输入的大小，取最短列表长度。
        :return: TAPP 批量输入的大小
        """
        min_len = min(len(self.Ti), len(self.H), len(self.B), len(self.C), len(self.N), len(self.O), len(self.Al),
                      len(self.Si), len(self.Cr), len(self.Fe), len(self.Ni), len(self.Cu), len(self.Zr), len(self.Nb),
                      len(self.Mo), len(self.V), len(self.Sn), len(self.HTT))
        if self.GS is not None:
            min_len = min(min_len, len(self.GS))
        return min_len

    def __str__(self) -> str:
        """
        将 TAPPBatchInput 实例转换为字符串表示。
        :return: TAPPBatchInput 实例的字符串表示。
        """
        input_size = len(self)
        tapp_input_str =  f"[Prop={self.Prop}" + \
                          f", Ti=[{', '.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.Ti[:input_size]])}]" + \
                          f", H=[{', '.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.H[:input_size]])}]" + \
                          f", B=[{', '.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.B[:input_size]])}]" + \
                          f", C=[{', '.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.C[:input_size]])}]" + \
                          f", N=[{', '.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.N[:input_size]])}]" + \
                          f", O=[{', '.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.O[:input_size]])}]" + \
                          f", Al=[{', '.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.Al[:input_size]])}]" + \
                          f", Si=[{', '.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.Si[:input_size]])}]" + \
                          f", Cr=[{', '.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.Cr[:input_size]])}]" + \
                          f", Fe=[{', '.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.Fe[:input_size]])}]" + \
                          f", Ni=[{', '.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.Ni[:input_size]])}]" + \
                          f", Cu=[{', '.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.Cu[:input_size]])}]" + \
                          f", Zr=[{', '.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.Zr[:input_size]])}]" + \
                          f", Nb=[{', '.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.Nb[:input_size]])}]" + \
                          f", Mo=[{', '.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.Mo[:input_size]])}]" + \
                          f", V=[{', '.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.V[:input_size]])}]" + \
                          f", Sn=[{', '.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.Sn[:input_size]])}]" + \
                          f", HTT=[{', '.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.HTT[:input_size]])}]" + \
                          ", GS=" + \
                          (f"[{','.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.GS[:input_size]])}]"
                           if self.GS is not None else "None") + "]"
        return tapp_input_str

    @property
    def sha256(self) -> str:
        """
        计算 TAPPBatchInput 实例的 SHA256 哈希值。
        :return: SHA256 哈希值
        """
        return hashlib.sha256(str(self).encode("utf-8")).hexdigest()


class TAPPOutput(BaseModel):
    """
    TAPP 单一输出的数据结构，包含钛合金的性能代码、预测值及其单位。
    """
    Prop: Literal["WF", "TE", "DS", "TC", "EC", "YM", "BM", "SM", "PR", "SE", "SHC", "YS", "TS", "HD", "HP"] = \
        Field(..., description="钛合金性能代码：相比例（WF）、热膨胀系数（TE）、密度（DS）、热导率（TC）、电导率（EC）、杨氏模量（YM）、"
                               "体积模量（BM）、剪切模量（SM）、泊松比（PR）、比焓（SE）、比热容（SHC）、屈服强度（YS）、抗拉强度（TS）、"
                               "硬度（HD）、霍尔佩奇系数（HP）", examples=["DS", "WF"])
    value: float | Dict[str, float] = Field(..., description="TAPP 模型预测的钛合金性能值，除相比例（WF）外，全部为浮点数类型，"
                                                             "相比例为字典类型，其中键为相的名称（∈{ALPHA、BETA、LAVES、TI3AL、"
                                                             "TI2CU、TI5SI3、TIZRSI、TI2NI、TIM_B2、LIQUID、C15_FCC、MC}），"
                                                             "值为相的质量分数（∈[0,1]）",
                                            examples=[5, {"ALPHA": 0.5, "BETA": 0.5, "LAVES": 0, "TI3AL": 0, "TI2CU": 0,
                                                          "TI5SI3": 0, "TIZRSI": 0, "TI2NI": 0, "TIM_B2": 0, "LIQUID": 0,
                                                          "C15_FCC": 0, "MC": 0}])
    unit: Optional[str] = Field(default=None, description="钛合金性能的单位", examples=["g/cm^3", "wt%"])

    @model_validator(mode='after')
    def default_unit(self) -> Any:
        """
        为缺失单位的性能赋予默认单位。
        :return: self
        """
        if self.Prop in _DEFAULT_UNIT_MAP and self.unit is None:
            self.unit = _DEFAULT_UNIT_MAP[self.Prop]
        return self

    def __str__(self) -> str:
        """
        将 TAPPOutput 实例转换为字符串表示。
        :return: TAPPOutput 实例的字符串表示
        """
        if self.Prop == "WF":
            value_str = f"[{', '.join(f'{k}={v:f}'.rstrip('0').rstrip('.') for k, v in self.value.items())}]"
        else:
            value_str = f"{self.value:f}".rstrip('0').rstrip('.')
        tapp_output_str = f"[Prop={self.Prop}, value={value_str}, unit=" + \
                          (self.unit if self.unit is not None else "None") + "]"
        return tapp_output_str

    @property
    def sha256(self):
        """
        计算 TAPPOutput 实例的 SHA256 哈希值。
        :return: SHA256 哈希值
        """
        return hashlib.sha256(str(self).encode("utf-8")).hexdigest()


class TAPPBatchOutput(BaseModel):
    """
    TAPP 批量输出的数据结构，包含钛合金的性能代码、预测值列表及其单位。
    """
    Prop: Literal["WF", "TE", "DS", "TC", "EC", "YM", "BM", "SM", "PR", "SE", "SHC", "YS", "TS", "HD", "HP"] = \
        Field(..., description="钛合金性能代码：相比例（WF）、热膨胀系数（TE）、密度（DS）、热导率（TC）、电导率（EC）、杨氏模量（YM）、"
                               "体积模量（BM）、剪切模量（SM）、泊松比（PR）、比焓（SE）、比热容（SHC）、屈服强度（YS）、抗拉强度（TS）、"
                               "硬度（HD）、霍尔佩奇系数（HP）", examples=["DS", "WF"])
    value: List[float] | Dict[str, List[float]] = Field(...,
                                                        description="TAPP 模型预测的钛合金性能值列表，除相比例（WF）外，"
                                                                    "全部为浮点数列表类型，相比例为浮点数列表的字典类型，"
                                                                    "其中键为相的名称（∈{ALPHA、BETA、LAVES、TI3AL、TI2CU、"
                                                                    "TI5SI3、TIZRSI、TI2NI、TIM_B2、LIQUID、C15_FCC、MC}），"
                                                                    "值为相的质量分数列表（∈[0,1]）",
                                                        examples=[[5, 6], {"ALPHA": [0.5, 0.6], "BETA": [0.6, 0.5],
                                                                           "LAVES": [0, 0], "TI3AL": [0, 0],
                                                                           "TI2CU": [0, 0], "TI5SI3": [0, 0],
                                                                           "TIZRSI": [0, 0], "TI2NI": [0, 0],
                                                                           "TIM_B2": [0, 0], "LIQUID": [0, 0],
                                                                           "C15_FCC": [0, 0], "MC": [0, 0]}])
    unit: Optional[str] = Field(default=None, description="钛合金性能的单位", examples=["g/cm^3", "wf%"])

    @model_validator(mode='after')
    def default_unit(self) -> Any:
        """
        为缺失单位的性能赋予默认单位。
        :return: self
        """
        if self.Prop in _DEFAULT_UNIT_MAP and self.unit is None:
            self.unit = _DEFAULT_UNIT_MAP[self.Prop]
        return self

    def __str__(self) -> str:
        """
        将 TAPPBatchOutput 实例转换为字符串表示。
        :return: TAPPBatchOutput 实例的字符串表示
        """
        if self.Prop == "WF":
            sub_strs = (f"{k}=[{', '.join(f'{_:f}'.rstrip('0').rstrip('.') for _ in v)}]" for k, v in self.value.items())
            value_str = f"[{', '.join(sub_strs)}]"
        else:
            value_str = f"[{', '.join(f'{_:f}'.rstrip('0').rstrip('.') for _ in self.value)}]"
        tapp_output_str = f"[Prop={self.Prop}, value={value_str}, unit=" + \
                          (self.unit if self.unit is not None else "None") + "]"
        return tapp_output_str

    def __len__(self) -> int:
        """
        计算 TAPP 批量输出的大小，取最短列表长度。
        :return: 批量输出的大小
        """
        if isinstance(self.value, Dict):
            return min(len(v) for k, v in self.value.items())
        else:
            return len(self.value)

    @property
    def sha256(self) -> str:
        """
        计算 TAPPBatchOutput 实例的 SHA256 哈希值。
        :return: SHA256 哈希值
        """
        return hashlib.sha256(str(self).encode("utf-8")).hexdigest()


class TAPPInfer:

    _prop_abbrs = ["WF", "TE", "DS", "TC", "EC", "YM", "BM", "SM", "PR", "SE", "SHC", "YS", "TS", "HD", "HP"]

    _norm_params = {
        "TE": {"Min": 4.848329916e-06, "Max": 2.283828168e-05},
        "DS": {"Min": 3.672575187, "Max": 10.34887892},
        "TC": {"Min": -87.38449318, "Max": 116.9406027},
        "EC": {"Min": 257.542498, "Max": 6101953.5},
        "YM": {"Min": -98.60315674, "Max": 273.60016},
        "BM": {"Min": -91.34943727, "Max": 232.2667855},
        "SM": {"Min": -37.34688091, "Max": 104.9342765},
        "PR": {"Min": 0.2569257344, "Max": 0.5},
        "SE": {"Min": -2140.62959, "Max": 1075.43465},
        "SHC": {"Min": 0.0016, "Max": 0.96764},
        "YS": {"Min": 2.297351749, "Max": 1896.478209},
        "TS": {"Min": 2.576930196, "Max": 2104.915562},
        "HD": {"Min": 0.8708966542, "Max": 705.7024083},
        "HP": {"Min": 0.007003215217, "Max": 1.712580475}
    }

    _sparsemax = Sparsemax(dim=-1)

    _task_flag_len = 8

    def __init__(self, device: torch.device | None = None, batch_size: int = 64, silence: bool = False,
                 skip_corr: bool = False):
        """
        使用 TAPP 模型进行推理的工具类，能够进行钛合金性能预测。
        :param device: 指定 Torch 设备，若为 None 则自动选择，并且优先选择 CUDA 设备
        :param batch_size: 进行批量推理时的批大小，不可以超过 10000
        :param silence: 是否开启静默模式，若为 True 则不打印运行信息，默认打印
        :param skip_corr: 是否跳过经验公式修正，默认不跳过
        """
        # 初始化静默标识
        self._silence = silence
        # 初始化跳过修正标识
        self._skip_corr = skip_corr
        # 打印 TAPP 的 Logo
        if not self._silence:
            print(text2art("TAP2"))
        # 初始化 Torch 设备
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if not isinstance(device, torch.device):
                raise TAPPException("<device> is not a <torch.device> instance.")
            self._device = device
        # 初始化批量大小
        if not isinstance(batch_size, int) or batch_size <= 0 or batch_size > _MAX_BATCH_SIZE:
            raise TAPPException(f"<batch_size> must be a positive integer no larger than {_MAX_BATCH_SIZE}.")
        self._batch_size = batch_size
        # 加载模型权重
        self._models = {}
        for prop_abbr in tqdm(self._prop_abbrs, desc="Loading models", disable=self._silence):
            prop_num = 12 if prop_abbr == "WF" else 1
            model_path = Path(str(resources.files("tap2.weight").joinpath(f"{prop_abbr}.pth")))
            if not model_path.exists():
                raise TAPPException(f"Model file <{model_path}> does not exist.")
            model = MoE2(element_num=12, process_num=1, property_num=prop_num, hidden_size=512, dropout_rate=0.2).to(self._device)
            model.load_state_dict(torch.load(model_path, map_location=self._device))
            model.eval()
            self._models[prop_abbr] = model
        # TS 和 YS 的微量元素修正系数
        self._W1 = torch.tensor(
            data=[2185.6, 1219.86, 704.2, 268, -34],
            dtype=torch.float32,
            device=self._device,
            requires_grad=False
        ) # Shape: (5,)
        # DS、TC、EC、YM、BM、SM、PR 的微量元素修正系数, Shape: (5, 9)
        self._W2 = {
            "DS": tensor(
                data=[[0.001189364, -0.033229803, 0.006900622, 0.001364253, 0.001293301, 0.001219326, 0.001929503, 0.001278101, 0.001198975],
                      [0.003211933, -0.169015533, 2.381598976, 7.424964639, 0.008192758, 0.014823883, 0.018094439, 0.014480812, 0.010925323],
                      [0.01812761, 0.021041677, -0.266758591, -1.897012602, 0.026452594, 0.026654634, 0.010732661, -5.104111794, 0.025470266],
                      [0.022893337, 0.046100565, 0.091790094, 0.04476362, 0.046677253, 0.046287457, 0.053434922, 0.045513762, -0.039840009],
                      [-4.922743468, -4.406682541, 1.499480452, 15.85136756, -4.284999468, -4.414201642, -4.482102982, 45.9676752, -4.531997149]],
                dtype=torch.float32,
                device=self._device,
                requires_grad=False
            ),
            "TC": tensor(
                data=[[0.011143497, 3.489852516, 0.050639097, 0.045199609, -0.006363126, -0.146271765, -0.036777042, -0.032206222, -0.035560762],
                      [0.015501867, 17.7350841, 121.4258822, 740.5704625, -42.09685675, -31.53882272, -0.157892947, 0.669169059, -0.220360531],
                      [-7.111188519, -13.84984689, -13.80683762, -81.18458214, -17.8742612, -16.63168331, -8.830106909, -19.40806548, -7.758796868],
                      [-0.079181297, -0.388880921, 0.282376119, -0.019358283, -0.138847193, -0.330954121, -0.429841711, -0.174089174, -0.310205874],
                      [174.2074784, 127.8573, 136.432501, 752.5723302, 446.2291183, 495.3979761, 25.63202273, 153.3714961, 9.850931867]],
                dtype=torch.float32,
                device=self._device,
                requires_grad=False
            ),
            "EC": tensor(
                data=[[427.4222959, -1160.893555, 1980.111753, -1703.05578, -355.9538164, -5426.046231, -1222.705087, -1150.778254, -1877.413688],
                      [643.5254332, 5716.60143, -81371.57587, 25064.0197, 8744.494462, 104164.2629, -8668.288187, 49621.93846, -12487.60801],
                      [-354773.0809, -356747.4198, -343208.1088, -242411.9374, -329665.1045, -391633.5205, -409248.1142, -364784.2755, -387130.4589],
                      [-18421.64249, -22875.26649, 13480.95763, -15435.74515, -7469.583248, -18476.56392, -25040.42194, -8473.574668, -22231.51572],
                      [454916.494, 900699.6168, 767781.0268, 368418.7209, 855436.335, 583596.1172, 649707.8811, 263225.8889, 558257.5577]],
                dtype=torch.float32,
                device=self._device,
                requires_grad=False
            ),
            "YM": tensor(
                data=[[2.687507505, 6.928988486, 2.760425827, 2.882769734, 2.856143054, 2.851864857, 2.894755453, 2.802196795, 2.409429476],
                      [4.494867834, 25.77829052, -71.10355102, 569.8110626, -86.32504336, 3.873705269, 18.0498347, 17.3860172, 17.07487544],
                      [6.683187786, 3.724882158, 8.733050146, -55.0048898, -19.64440386, 3.999887499, 7.763206322, -6.288934614, 7.191511269],
                      [2.396641297, -0.075441769, -5.220427127, -1.981547993, -3.917721649, -4.955526404, -5.294481326, -4.770403738, -2.979347874],
                      [-92.63189837, 175.7924499, 83.62668492, 717.1182946, 957.9610359, 228.6389485, 111.8343005, 251.9310226, 108.5756246]],
                dtype=torch.float32,
                device=self._device,
                requires_grad=False
            ),
            "BM": tensor(
                data=[[1.91143643, 4.782574079, 2.011642041, 2.116353667, 2.107925979, 2.074354268, 2.120185848, 2.053483553, 1.75353913],
                      [3.209269193, 17.46569617, -61.61676964, 228.7679945, -67.88536, -0.822162564, 13.29738451, 13.00498378, 12.55234131],
                      [-4.87225487, -7.610587176, -2.889906738, -30.13159105, -24.46797818, -7.585980355, -4.258953485, -12.81924464, -4.258542434],
                      [-2.946153806, -12.61637001, -12.91788374, -10.19348451, -11.42129274, -12.41581323, -13.18642973, -12.09212666, -9.908171743],
                      [85.02240782, 330.0399644, 242.651319, 450.5155243, 930.2098776, 410.4772374, 277.1712212, 360.7586372, 266.617343]],
                dtype=torch.float32,
                device=self._device,
                requires_grad=False
            ),
            "SM": tensor(
                data=[[1.051068564, 2.718633112, 1.077060841, 1.122686991, 1.112773606, 1.113072282, 1.128919511, 1.092907451, 0.940211602],
                      [1.757030104, 10.13190287, -27.17580326, 232.6984397, -33.38916671, 1.722337109, 7.034259775, 6.764044231, 6.655379931],
                      [3.172669662, 2.062887355, 3.950813852, -22.0090355, -7.071173486, 2.177602179, 3.612231287, -1.969884481, 3.362529131],
                      [1.203795098, -1.44761456, -1.510643413, -0.286721245, -1.035560908, -1.423615668, -1.526187408, -1.363322081, -0.716049713],
                      [-45.03878237, 56.66865752, 21.94369734, 283.6583886, 360.2530288, 74.92434101, 32.11387735, 87.88767559, 31.35851808]],
                dtype=torch.float32,
                device=self._device,
                requires_grad=False
            ),
            "PR": tensor(
                data=[[0.253272904, -0.002701381, -0.000889767, -0.000978328, -0.000909902, -0.00092762, -0.0009085, -0.00086636, -0.000786759],
                      [-0.001525558, -0.010709855, 0.002303789, -0.522062551, 0.020209486, -0.007743567, -0.00557927, -0.005084367, -0.005387734],
                      [-0.018617133, -0.019438832, -0.019382127, 0.036048333, -0.012011115, -0.020208344, -0.020199332, -0.012779452, -0.019385532],
                      [-0.008751755, -0.014158427, -0.014496842, -0.0148, -0.0144, -0.0142, -0.014842884, -0.013871117, -0.012828261],
                      [0.295234945, 0.307906363, 0.304070677, -0.36820265, 0.096743287, 0.369543766, 0.319818328, 0.242612884, 0.307072298]],
                dtype=torch.float32,
                device=self._device,
                requires_grad=False
            )
        }
        # TE 的微量元素修正系数
        self._W3 = torch.tensor(
            data=[[-2.89e-10, -4.282498e-10, -7.23e-10, 4.56e-9, 3.085894e-10],
                  [4.60e-07, 8.039267e-07, -1.22e-07, -5.52e-07, -4.072913e-08],
                  [8.76e-07, -3.124228e-09, 1.22e-07, 1.74e-06, -8.598354e-08],
                  [-1.37e-06, -1.089075e-06, 2.52e-07, -5.93e-06, 1.949381e-07]],
            dtype=torch.float32,
            device=self._device,
            requires_grad=False
        ) # Shape: (4, 5)

    def _infer(self, tapp_input: TAPPInput, task_flag: str | None = None, wrap: bool = True) \
            -> TAPPOutput | float | Dict[str, float]:
        """
        调用 TAPP 模型进行单点推理，获取预测性能。
        :param tapp_input: 输入数据，包括性能代号、元素组成、热处理温度和晶粒尺寸（单一输入）
        :param task_flag: 任务标识，若不指定则由 tapp_input 的 SHA256 哈希值表示
        :param wrap: 是否将输出包装为 TAPPOutput 实例，默认包装
        :return: 性能值（单一输出）
        """
        # 生成任务标识
        if task_flag is None:
            task_flag = tapp_input.sha256
        short_task_flag = task_flag[:self._task_flag_len] + "*"
        if not self._silence:
            logger.info(f"[{short_task_flag}] Inferring: {tapp_input}")
        # 调用 TAPP 模型进行推理
        with torch.inference_mode():
            # 读取模型
            model = self._models[tapp_input.Prop]
            # 构建输入张量
            input_T = tensor(
                data=[[tapp_input.Ti, tapp_input.Al, tapp_input.Cr, tapp_input.Cu, tapp_input.Fe, tapp_input.Mo,
                       tapp_input.Ni, tapp_input.Nb, tapp_input.Si, tapp_input.Sn, tapp_input.V, tapp_input.Zr,
                       tapp_input.HTT]],
                dtype=torch.float32,
                device=self._device,
                requires_grad=False
            ) # Shape: (1, 13)
            # 进行推理
            output_T = model(input_T) # Shape: (1, prop_num)
            # 对于相比例性能，用 sparsemax 进行归一化
            if tapp_input.Prop == "WF":
                output_T = self._sparsemax(output_T) # Shape: (1, 12)
                # 转换为 Python 字典 (Dict[str, float])
                output = output_T.tolist()[0] # Shape: (12,)
                output = dict(zip(["ALPHA", "BETA", "LAVES", "TI3AL", "TI2CU", "TI5SI3", "TIZRSI", "TI2NI", "TIM_B2",
                                   "LIQUID", "C15_FCC", "MC"], output))
            # 对于其他性能，进行 Min-Max 反归一化
            if tapp_input.Prop in self._norm_params:
                # 读取归一化参数
                min_val = self._norm_params[tapp_input.Prop]["Min"]
                max_val = self._norm_params[tapp_input.Prop]["Max"]
                output_T = output_T * (max_val - min_val) + min_val # Shape: (1, 1)
                # 转换为 Python 浮点数
                output = output_T.item()
        tapp_output = TAPPOutput(Prop=tapp_input.Prop, value=output)
        if not self._silence:
            logger.info(f"[{short_task_flag}] Inferred: {tapp_input} ==> {tapp_output}")
        return tapp_output if wrap else output

    def _batch_infer(self, tapp_input: TAPPBatchInput, task_flag: str | None = None, wrap: bool = True) \
            -> TAPPBatchOutput | List[float] | Dict[str, List[float]]:
        """
        调用 TAPP 模型进行批量推理，获取预测性能。
        :param tapp_input: 输入数据，包括性能代号、元素组成、热处理温度和晶粒尺寸（批量输入）
        :param task_flag: 任务标识，若不指定则由 tapp_input 的 SHA256 哈希值表示
        :param wrap: 是否将输出包装为 TAPPBatchOutput 实例，默认包装
        :return: 性能值（批量输出）
        """
        # 生成任务标识
        if task_flag is None:
            task_flag = tapp_input.sha256
        short_task_flag = task_flag[:self._task_flag_len] + "*"
        # 获取输入大小
        input_size = len(tapp_input)
        # 计算批数量
        batch_num = math.ceil(input_size / self._batch_size)
        # 初始化空输出
        if tapp_input.Prop == "WF":
            outputs = {
                "ALPHA": [],
                "BETA": [],
                "LAVES": [],
                "TI3AL": [],
                "TI2CU": [],
                "TI5SI3": [],
                "TIZRSI": [],
                "TI2NI": [],
                "TIM_B2": [],
                "LIQUID": [],
                "C15_FCC": [],
                "MC": []
            }
        else:
            outputs = []
        # 调用 TAPP 模型进行推理
        with torch.inference_mode():
            # 读取模型
            model = self._models[tapp_input.Prop]
            # 按批处理，步长设置为 self._batch_size
            for batch_id, start_idx in enumerate(range(0, input_size, self._batch_size)):
                if not self._silence:
                    logger.info(f"[{short_task_flag}] Inferring: Prop={tapp_input.Prop}... ({batch_id+1}/{batch_num})")
                end_idx = min(start_idx + self._batch_size, input_size)
                # 构建输入张量
                inputs_T = tensor(
                    data=[tapp_input.Ti[start_idx: end_idx],
                          tapp_input.Al[start_idx: end_idx],
                          tapp_input.Cr[start_idx: end_idx],
                          tapp_input.Cu[start_idx: end_idx],
                          tapp_input.Fe[start_idx: end_idx],
                          tapp_input.Mo[start_idx: end_idx],
                          tapp_input.Ni[start_idx: end_idx],
                          tapp_input.Nb[start_idx: end_idx],
                          tapp_input.Si[start_idx: end_idx],
                          tapp_input.Sn[start_idx: end_idx],
                          tapp_input.V[start_idx: end_idx],
                          tapp_input.Zr[start_idx: end_idx],
                          tapp_input.HTT[start_idx: end_idx]],
                    dtype=torch.float32,
                    device=self._device,
                    requires_grad=False
                ) # Shape: (13, batch_size)
                # 转置
                inputs_T = inputs_T.T # Shape: (batch_size, 13)
                # 进行推理
                outputs_T = model(inputs_T) # Shape: (batch_size, prop_num)
                # 对于相比例性能，用 sparsemax 进行归一化
                if tapp_input.Prop == "WF":
                    outputs_T = self._sparsemax(outputs_T) # Shape: (batch_size, 12)
                    # 转置
                    outputs_T = outputs_T.T  # Shape: (12, batch_size)
                    # 添加到 Python 字典 (Dict[str, List[float]])
                    for x, y in zip(outputs.values(), outputs_T.tolist()):
                        x.extend(y)
                # 对于其他性能，进行 Min-Max 反归一化
                if tapp_input.Prop in self._norm_params:
                    # 读取归一化参数
                    min_val = self._norm_params[tapp_input.Prop]["Min"]
                    max_val = self._norm_params[tapp_input.Prop]["Max"]
                    outputs_T = outputs_T.squeeze(-1)  # Shape: (batch_size,)
                    outputs_T = outputs_T * (max_val - min_val) + min_val # Shape: (batch_size,)
                    # 添加到 Python 列表
                    outputs.extend(outputs_T.tolist())
        tapp_output = TAPPBatchOutput(Prop=tapp_input.Prop, value=outputs)
        if not self._silence:
            logger.info(f"[{short_task_flag}] Inferred: Prop={tapp_input.Prop} ({len(tapp_output)}its)")
        return tapp_output if wrap else outputs

    def _corr(self, tapp_input: TAPPInput, orig_output: TAPPOutput | float | Dict[str, float],
              task_flag: str | None = None, wrap: bool = True) -> TAPPOutput | float | Dict[str, float]:
        """
        利用经验公式对 TAPP 模型的预测结果进行修正。
        :param tapp_input: 输入数据，包括性能代号、元素组成、热处理温度和晶粒尺寸（单一输入）
        :param orig_output: 原始性能值（单一输出）
        :param task_flag: 任务标识，若不指定则由 tapp_input 的 SHA256 哈希值表示
        :param wrap: 是否将输出包装为 TAPPOutput 实例，默认包装
        :return: 修正性能值（单一输出）
        """
        # 生成任务标识
        if task_flag is None:
            task_flag = tapp_input.sha256
        short_task_flag = task_flag[:self._task_flag_len] + "*"
        if not self._silence:
            logger.info(f"[{short_task_flag}] Correcting: {tapp_input}")
        # 解包原始输出
        correct = deepcopy(orig_output.value if isinstance(orig_output, TAPPOutput) else orig_output)
        if tapp_input.Prop in ["YS", "TS"]:
            # 1. 微量元素修正
            with torch.inference_mode():
                # 构建微量元素浓度张量
                micro_elem_T = torch.tensor(
                    data=[tapp_input.N, tapp_input.O, tapp_input.C, tapp_input.H, tapp_input.B],
                    dtype=torch.float32,
                    device=self._device,
                    requires_grad=False
                ) # Shape: (5,)
                # 计算增量
                increment = torch.sum(self._W1 * micro_elem_T).item()
            # TS 的增量是 YS 的 1.1 倍
            if tapp_input.Prop == "TS":
                increment *= 1.1
            # 2. 晶粒尺寸修正（转换为单位：m)
            grain_size = tapp_input.GS * 1e-6
            # 计算 10 μm 晶粒尺寸下的 HP
            hall_petch = self._infer(tapp_input.model_copy(update={"Prop": "HP", "GS": 10}, deep=True),
                                     task_flag=task_flag,
                                     wrap=False)
            # 计算增量
            increment += hall_petch * (pow(grain_size, -0.5) - pow(1e-5, -0.5))
            correct += increment
        elif tapp_input.Prop in ["DS", "TC", "EC", "YM", "BM", "SM", "PR"]:
            # 微量元素修正
            with torch.inference_mode():
                # 构建微量元素张量
                micro_elem_T = torch.tensor(
                    data=[[tapp_input.C, tapp_input.N, tapp_input.O, tapp_input.B, tapp_input.H]],
                    dtype=torch.float32,
                    device=self._device,
                    requires_grad=False
                ) # Shape: (1, 5)
                # 构建主要元素张量
                main_elem_T = torch.tensor(
                    data=[[tapp_input.Al], [tapp_input.Cr], [tapp_input.Cu], [tapp_input.Fe], [tapp_input.Mo],
                          [tapp_input.Nb], [tapp_input.Ni], [tapp_input.Si], [tapp_input.Sn]],
                    dtype=torch.float32,
                    device=self._device,
                    requires_grad=False
                ) # Shape: (9, 1)
                increment = (micro_elem_T @ self._W2[tapp_input.Prop] @ main_elem_T).item()
            correct += increment
        elif tapp_input.Prop == "TE":
            # 微量元素修正
            with torch.inference_mode():
                # 构建微量元素张量
                micro_elem_T = torch.tensor(
                    data=[[1, 1, 1, 1, 1],
                          [tapp_input.N, tapp_input.O, tapp_input.C, tapp_input.H, tapp_input.B],
                          [tapp_input.N ** 2, tapp_input.O ** 2, tapp_input.C ** 2, tapp_input.H ** 2, tapp_input.B ** 2],
                          [tapp_input.N ** 3, tapp_input.O ** 3, tapp_input.C ** 3, tapp_input.H ** 3, tapp_input.B ** 3]],
                    dtype=torch.float32,
                    device=self._device,
                    requires_grad=False
                ) # Shape: (4, 5)
                increment = torch.sum(micro_elem_T * self._W3).item()
            correct += increment
        tapp_output = TAPPOutput(
            Prop=tapp_input.Prop,
            value=correct,
            unit=orig_output.unit if isinstance(orig_output, TAPPOutput) else None
        )
        if not self._silence:
            logger.info(f"[{short_task_flag}] Corrected: {tapp_input} ==> {tapp_output}")
        return tapp_output if wrap else correct

    def _batch_corr(self, tapp_input: TAPPBatchInput,
                    orig_output: TAPPBatchOutput | List[float] | Dict[str, List[float]],
                    task_flag: str | None = None,
                    wrap: bool = True) -> TAPPBatchOutput | List[float] | Dict[str, List[float]]:
        """
        利用经验公式对 TAPP 模型的预测结果进行修正。
        :param tapp_input: 输入数据，包括性能代号、元素组成、热处理温度和晶粒尺寸（批量输入）
        :param orig_output: 原始性能值（批量输出）
        :param task_flag: 任务标识，若不指定则由 tapp_input 的 SHA256 哈希值表示
        :param wrap: 是否将输出包装为 TAPPBatchOutput 实例，默认包装
        :return: 修正性能值（批量输出）
        """
        # 生成任务标识
        if task_flag is None:
            task_flag = tapp_input.sha256
        short_task_flag = task_flag[:self._task_flag_len] + "*"
        # 获取输入大小
        input_size = len(tapp_input)
        # 计算批数量
        batch_num = math.ceil(input_size / self._batch_size)
        # 解包原始输出
        corrects = deepcopy(orig_output.value if isinstance(orig_output, TAPPBatchOutput) else orig_output)
        if tapp_input.Prop in ["YS", "TS"]:
            increments = []
            with torch.inference_mode():
                # 计算所有输入数据在 10 μm 晶粒尺寸下的 HP
                hall_petch_T = tensor(
                    data=self._batch_infer(
                        tapp_input.model_copy(update={"Prop": "HP", "GS": [10] * input_size}),
                        task_flag=task_flag,
                        wrap=False
                    ),
                    dtype=torch.float32,
                    device=self._device,
                    requires_grad=False
                ) # Shape: (input_size,)
                for batch_id, start_idx in enumerate(range(0, input_size, self._batch_size)):
                    if not self._silence:
                        logger.info(f"[{short_task_flag}] Correcting: Prop={tapp_input.Prop} ({batch_id+1}/{batch_num})")
                    end_idx = min(start_idx + self._batch_size, input_size)
                    # 1. 微量元素修正
                    # 构建微量元素浓度张量
                    micro_elem_T = tensor(
                        data=[tapp_input.N[start_idx: end_idx],
                              tapp_input.O[start_idx: end_idx],
                              tapp_input.C[start_idx: end_idx],
                              tapp_input.H[start_idx: end_idx],
                              tapp_input.B[start_idx: end_idx]],
                        dtype=torch.float32,
                        device=self._device,
                        requires_grad=False
                    ) # Shape: (5, batch_size)
                    # 转置
                    micro_elem_T = micro_elem_T.T # Shape: (batch_size, 5)
                    # 计算增量
                    increments_T = torch.sum(self._W1 * micro_elem_T, dim=1) # Shape: (batch_size,)
                    # TS 的增量是 YS 的 1.1 倍
                    if tapp_input.Prop == "TS":
                        increments_T *= 1.1
                    # 2. 晶粒尺寸修正
                    # 构建晶粒尺寸张量
                    grain_size_T = tensor(
                        data=tapp_input.GS[start_idx: end_idx],
                        dtype=torch.float32,
                        device=self._device,
                        requires_grad=False
                    ) # Shape: (batch_size,)
                    # 转换为单位：m
                    grain_size_T *= 1e-6
                    # 计算增量
                    increments_T += hall_petch_T[start_idx: end_idx] * (grain_size_T ** -0.5 - 1e-5 ** -0.5) # Shape: (batch_size,)
                    increments.extend(increments_T.tolist())
            for idx, increment in enumerate(increments):
                corrects[idx] += increment
        elif tapp_input.Prop in ["DS", "TC", "EC", "YM", "BM", "SM", "PR"]:
            increments = []
            with torch.inference_mode():
                for batch_id, start_idx in enumerate(range(0, input_size, self._batch_size)):
                    if not self._silence:
                        logger.info(f"[{short_task_flag}] Correcting: Prop={tapp_input.Prop} ({batch_id+1}/{batch_num})")
                    end_idx= min(start_idx + self._batch_size, input_size)
                    # 微量元素修正
                    # 构建微量元素张量
                    micro_elem_T = torch.tensor(
                        data=[tapp_input.C[start_idx: end_idx],
                              tapp_input.N[start_idx: end_idx],
                              tapp_input.O[start_idx: end_idx],
                              tapp_input.B[start_idx: end_idx],
                              tapp_input.H[start_idx: end_idx]],
                        dtype=torch.float32,
                        device=self._device,
                        requires_grad=False
                    ) # Shape: (5, batch_size)
                    # 转置 + 增加维度
                    micro_elem_T = micro_elem_T.T.unsqueeze(1) # Shape: (batch_size, 1, 5)
                    # 构建主要元素张量
                    main_elem_T = torch.tensor(
                        data=[tapp_input.Al[start_idx: end_idx],
                              tapp_input.Cr[start_idx: end_idx],
                              tapp_input.Cu[start_idx: end_idx],
                              tapp_input.Fe[start_idx: end_idx],
                              tapp_input.Mo[start_idx: end_idx],
                              tapp_input.Nb[start_idx: end_idx],
                              tapp_input.Ni[start_idx: end_idx],
                              tapp_input.Si[start_idx: end_idx],
                              tapp_input.Sn[start_idx: end_idx]],
                        dtype=torch.float32,
                        device=self._device,
                        requires_grad=False
                    ) # Shape: (9, batch_size)
                    # 转置 + 增加维度
                    main_elem_T = main_elem_T.T.unsqueeze(-1) # Shape: (batch_size, 9, 1)
                    increments_T = micro_elem_T @ self._W2[tapp_input.Prop] @ main_elem_T  # Shape: (batch_size, 1, 1)
                    increments_T = increments_T.squeeze(-1).squeeze(-1)  # Shape: (batch_size,)
                    increments.extend(increments_T.tolist())
            for idx, increment in enumerate(increments):
                corrects[idx] += increment
        elif tapp_input.Prop == "TE":
            increments = []
            with torch.inference_mode():
                for batch_id, start_idx in enumerate(range(0, input_size, self._batch_size)):
                    if not self._silence:
                        logger.info(f"[{short_task_flag}] Correcting: Prop=TE ({batch_id+1}/{batch_num})")
                    end_idx= min(start_idx + self._batch_size, input_size)
                    # 微量元素修正
                    # 构建微量元素张量（高阶）
                    micro_elem_T = torch.tensor(
                        data=[tapp_input.N[start_idx: end_idx],
                              tapp_input.O[start_idx: end_idx],
                              tapp_input.C[start_idx: end_idx],
                              tapp_input.H[start_idx: end_idx],
                              tapp_input.B[start_idx: end_idx]],
                        dtype=torch.float32,
                        device=self._device
                    ) # Shape: (5, batch_size)
                    # 转置
                    micro_elem_T = micro_elem_T.T  # Shape: (batch_size, 5)
                    # 构建高阶张量
                    micro_elem_T = torch.stack([
                        torch.ones_like(micro_elem_T),
                        micro_elem_T,
                        micro_elem_T ** 2,
                        micro_elem_T ** 3
                    ], dim=1) # Shape: (batch_size, 4, 5)
                    increments_T = torch.sum(micro_elem_T * self._W3, dim=(1, 2)) # Shape: (batch_size,)
                    increments.extend(increments_T.tolist())
            for idx, increment in enumerate(increments):
                corrects[idx] += increment
        tapp_output = TAPPBatchOutput(
            Prop=tapp_input.Prop,
            value=corrects,
            unit=orig_output.unit if isinstance(orig_output, TAPPBatchOutput) else None
        )
        if not self._silence:
            logger.info(f"[{short_task_flag}] Corrected: Prop={tapp_input.Prop} ({len(tapp_output)}its)")
        return tapp_output if wrap else corrects

    def __call__(self, tapp_input: TAPPInput | TAPPBatchInput) -> TAPPOutput | TAPPBatchOutput:
        """
        调用 TAPP 模型进行推理，预测钛合金的性能，支持批量处理。
        :param tapp_input: 包括待预测性能名称、钛合金的元素组成、热处理温度和晶粒尺寸（单一输入或批量输入）
        :return: TAPP 预测的性能值（单一输出或批量输出）
        """
        # 获取任务标识
        task_flag = tapp_input.sha256
        # 根据参数类型决定单点推理或批量推理，若为 TAPPInput 使用单点推理，若为 TAPPBatchInput 使用批量推理
        if isinstance(tapp_input, TAPPInput):
            if self._skip_corr:
                return self._infer(tapp_input, task_flag, True)
            else:
                orig_output = self._infer(tapp_input, task_flag, False)
                # 进行经验公式修正
                corr_output = self._corr(tapp_input, orig_output, task_flag, True)
                return corr_output
        elif isinstance(tapp_input, TAPPBatchInput):
            if self._skip_corr:
                return self._batch_infer(tapp_input, task_flag, True)
            else:
                orig_outputs = self._batch_infer(tapp_input, task_flag, False)
                # 进行经验公式修正
                corr_outputs = self._batch_corr(tapp_input, orig_outputs, task_flag, True)
                return corr_outputs
        else:
            raise TAPPException("<input_data> is not <TAPPInput> or <TAPPBatchInput> instance.")

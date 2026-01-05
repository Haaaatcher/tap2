import torch
from hashlib import sha256
from copy import deepcopy
from art import text2art
from tap2.model import MoE2
from pathlib import Path
from importlib import resources
from typing import Literal, Any, List, Annotated, Optional
from pydantic import BaseModel, Field, model_validator
from sparsemax import Sparsemax
from loguru import logger
from math import isclose, ceil
from rich import print as rprint


# 最大的批量大小，防止内存溢出
_MAX_BATCH_SIZE = 10000


# 各性能的默认单位映射
_DEFAULT_UNIT_MAP = {
    "BTT": "℃",
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
    TAP2 单一输入的数据结构，包含钛合金的元素组成、热处理温度和晶粒尺寸等信息。
    """
    Prop: Literal['BTT', 'WF', 'TE', 'DS', 'TC', 'EC', 'YM', 'BM', 'SM', 'PR', 'SE', 'SHC', 'YS', 'TS', 'HD', 'HP'] = Field(..., description='钛合金性能代码：β转变温度（BTT）、相比例（WF）、热膨胀系数（TE）、密度（DS）、热导率（TC）、电导率（EC）、杨氏模量（YM）、体积模量（BM）、剪切模量（SM）、泊松比（PR）、比焓（SE）、比热容（SHC）、屈服强度（YS）、抗拉强度（TS）、硬度（HD）、霍尔佩奇系数（HP）', examples=['DS'])
    Ti: float = Field(default=0, description='钛（Ti）的质量分数', examples=[90], ge=0, le=100)
    H: float = Field(default=0, description='氢（H）的质量分数', examples=[0], ge=0, le=100)
    B: float = Field(default=0, description='硼（B）的质量分数', examples=[0], ge=0, le=100)
    C: float = Field(default=0, description='碳（C）的质量分数', examples=[0], ge=0, le=100)
    N: float = Field(default=0, description='氮（N）的质量分数', examples=[0], ge=0, le=100)
    O: float = Field(default=0, description='氧（O）的质量分数', examples=[0], ge=0, le=100)
    Al: float = Field(default=0, description='铝（Al）的质量分数', examples=[6], ge=0, le=100)
    Si: float = Field(default=0, description='硅（Si）的质量分数', examples=[0], ge=0, le=100)
    Cr: float = Field(default=0, description='铬（Cr）的质量分数', examples=[0], ge=0, le=100)
    Fe: float = Field(default=0, description='铁（Fe）的质量分数', examples=[0], ge=0, le=100)
    Ni: float = Field(default=0, description='镍（Ni）的质量分数', examples=[0], ge=0, le=100)
    Cu: float = Field(default=0, description='铜（Cu）的质量分数', examples=[0], ge=0, le=100)
    Zr: float = Field(default=0, description='锆（Zr）的质量分数', examples=[0], ge=0, le=100)
    Nb: float = Field(default=0, description='铌（Nb）的质量分数', examples=[0], ge=0, le=100)
    Mo: float = Field(default=0, description='钼（Mo）的质量分数', examples=[0], ge=0, le=100)
    V: float = Field(default=0, description='钒（V）的质量分数', examples=[4], ge=0, le=100)
    Sn: float = Field(default=0, description='锡（Sn）的质量分数', examples=[0], ge=0, le=100)
    HTT: Optional[float] = Field(default=None, description='热处理温度（℃），预测如下性能必须提供热处理温度：WF、TE、DS、TC、EC、YM、BM、SM、PR、SE、SHC、YS、TS、HD、HP', examples=[600], gt=-273.15)
    GS: Optional[float] = Field(default=None, description='晶粒尺寸（μm），预测如下性能必须提供晶粒尺寸：YS、TS、HD、HP', examples=[10], gt=0)

    @model_validator(mode="after")
    def _valid_compos(self) -> Any:
        """
        验证元素组成之和是否为 100 wt%。
        :return: self
        """
        total = self.Ti + self.H + self.B + self.C + self.N + self.O + self.Al + self.Si + self.Cr + self.Fe + self.Ni + self.Cu + self.Zr + self.Nb + self.Mo + self.V + self.Sn
        if not isclose(total, 100, abs_tol=1e-6):
            raise TAPPException(f"The sum of all element compositions must be 100, but got {total}.")
        return self

    @model_validator(mode="after")
    def _valid_htt(self) -> Any:
        """
        验证如下性能是否提供了热处理温度：WF、TE、DS、TC、EC、YM、BM、SM、PR、SE、SHC、YS、TS、HD、HP。
        :return: self
        """
        if self.Prop in ["WF", "TE", "DS", "TC", "EC", "YM", "BM", "SM", "PR", "SE", "SHC", "YS", "TS", "HD", "HP"] and self.HTT is None:
            raise TAPPException("HTT is required for WF、TE、DS、TC、EC、YM、BM、SM、PR、SE、SHC、YS、TS、HD、HP")
        return self

    @model_validator(mode="after")
    def _valid_gs(self) -> Any:
        """
        验证如下性能是否提供了晶粒尺寸：YS、TS、HD、HP。
        :return: self
        """
        if self.Prop in ("YS", "TS", "HD", "HP") and self.GS is None:
            raise TAPPException("GS is required for YS、TS、HD、HP.")
        return self

    def __str__(self) -> str:
        """
        将 TAPPInput 实例转换为字符串表示。
        :return: TAPPInput 实例的字符串表示，例如：{Prop=YS, Ti=90, H=0, B=0, C=0, N=0, O=0, Al=6, Si=0, Cr=0, Fe=0, Ni=0, Cu=0, Zr=0, Nb=0, Mo=0, V=4, Sn=0, HTT=800, GS=20}。
        """
        return f"{{Prop={self.Prop}" + \
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
               ", HTT=" + (f"{self.HTT:f}".rstrip('0').rstrip('.') if self.HTT is not None else "None") + \
               ", GS=" + (f"{self.GS:f}".rstrip('0').rstrip('.') if self.GS is not None else "None") + "}"

    @property
    def sha256(self) -> str:
        """
        计算 TAPPInput 实例的 SHA256 哈希值。
        :return: SHA256 哈希值
        """
        return sha256(str(self).encode("utf-8")).hexdigest()


class TAPPBatchInput(BaseModel):
    """
    TAP2 批量输入的数据结构，包含钛合金的元素组成、热处理温度和晶粒尺寸等信息的列表。
    """
    Prop: Literal["BTT", "WF", "TE", "DS", "TC", "EC", "YM", "BM", "SM", "PR", "SE", "SHC", "YS", "TS", "HD", "HP"] = Field(..., description="钛合金性能代码：β转变温度（BTT）、相比例（WF）、热膨胀系数（TE）、密度（DS）、热导率（TC）、电导率（EC）、杨氏模量（YM）、体积模量（BM）、剪切模量（SM）、泊松比（PR）、比焓（SE）、比热容（SHC）、屈服强度（YS）、抗拉强度（TS）、硬度（HD）、霍尔佩奇系数（HP）", examples=["DS"])
    Ti: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="钛（Ti）的质量分数列表", examples=[[90, 97]], min_length=1, max_length=_MAX_BATCH_SIZE)
    H: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="氢（H）的质量分数列表", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    B: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="硼（B）的质量分数列表", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    C: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="碳（C）的质量分数列表", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    N: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="氮（N）的质量分数列表", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    O: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="氧（O）的质量分数列表", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    Al: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="铝（Al）的质量分数列表", examples=[[6, 3]], min_length=1, max_length=_MAX_BATCH_SIZE)
    Si: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="硅（Si）的质量分数列表", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    Cr: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="铬（Cr）的质量分数列表", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    Fe: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="铁（Fe）的质量分数列表", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    Ni: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="镍（Ni）的质量分数列表", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    Cu: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="铜（Cu）的质量分数列表", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    Zr: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="锆（Zr）的质量分数列表", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    Nb: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="铌（Nb）的质量分数列表", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    Mo: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="钼（Mo）的质量分数列表", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    V: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="钒（V）的质量分数列表", examples=[[4, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    Sn: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="锡（Sn）的质量分数列表", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    HTT: Optional[List[Annotated[float, Field(gt=-273.15)]]] = Field(default=None, description="热处理温度（℃），预测如下性能必须提供热处理温度：WF、TE、DS、TC、EC、YM、BM、SM、PR、SE、SHC、YS、TS、HD、HP", examples=[[600, 800]], min_length=1, max_length=_MAX_BATCH_SIZE)
    GS: Optional[List[Annotated[float, Field(gt=0)]]] = Field(default=None, description="晶粒尺寸（μm）列表，预测如下性能必须提供晶粒尺寸：YS、TS、HD、HP", examples=[[10, 20]], min_length=1, max_length=_MAX_BATCH_SIZE)

    @model_validator(mode="after")
    def _valid_compos(self) -> Any:
        """
        验证批量输入的每一项是否均满足元素组成之和为 100 wt%，实际只检查前 N 项（N 为最小列表长度）。
        :return: self
        """
        for idx in range(len(self)):
            total = self.Ti[idx] + self.H[idx] + self.B[idx] + self.C[idx] + self.N[idx] + self.O[idx] + self.Al[idx] + self.Si[idx] + self.Cr[idx] + self.Fe[idx] + self.Ni[idx] + self.Cu[idx] + self.Zr[idx] + self.Nb[idx] + self.Mo[idx] + self.V[idx] + self.Sn[idx]
            if not isclose(total, 100, abs_tol=1e-6):
                raise TAPPException(f"The sum of all element compositions must be 100, but got {total} at index {idx}.")
        return self

    @model_validator(mode="after")
    def _valid_htt(self) -> Any:
        """
        验证如下性能是否提供了热处理温度：WF、TE、DS、TC、EC、YM、BM、SM、PR、SE、SHC、YS、TS、HD、HP。
        :return: self
        """
        if self.Prop in ["WF", "TE", "DS", "TC", "EC", "YM", "BM", "SM", "PR", "SE", "SHC", "YS", "TS", "HD", "HP"] and self.HTT is None:
            raise TAPPException("HTT is required for WF、TE、DS、TC、EC、YM、BM、SM、PR、SE、SHC、YS、TS、HD、HP")
        return self

    @model_validator(mode="after")
    def _valid_gs(self) -> Any:
        """
        验证如下性能是否提供了晶粒尺寸：YS、TS、HD、HP。
        :return: self
        """
        if self.Prop in ("YS", "TS", "HD", "HP") and self.GS is None:
            raise TAPPException("GS is required for YS、TS、HD、HP.")
        return self

    def __len__(self) -> int:
        """
        计算 TAP2 批量输入的大小，原则为取最短列表长度。
        :return: TAP2 批量输入的大小
        """
        min_len = min(len(self.Ti), len(self.H), len(self.B), len(self.C), len(self.N), len(self.O), len(self.Al), len(self.Si), len(self.Cr), len(self.Fe), len(self.Ni), len(self.Cu), len(self.Zr), len(self.Nb), len(self.Mo), len(self.V), len(self.Sn))
        if self.HTT is not None:
            min_len = min(min_len, len(self.HTT))
        if self.GS is not None:
            min_len = min(min_len, len(self.GS))
        return min_len

    def __str__(self) -> str:
        """
        将 TAPPBatchInput 实例转换为字符串表示。
        :return: TAPPBatchInput 实例的字符串表示。
        """
        input_size = len(self)
        return f"{{Prop={self.Prop}" + \
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
               ", HTT=" + (f"[{','.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.HTT[:input_size]])}]" if self.HTT is not None else "None") + \
               ", GS=" + (f"[{','.join([f'{_:f}'.rstrip('0').rstrip('.') for _ in self.GS[:input_size]])}]" if self.GS is not None else "None") + "}"

    @property
    def sha256(self) -> str:
        """
        计算 TAPPBatchInput 实例的 SHA256 哈希值。
        :return: SHA256 哈希值
        """
        return sha256(str(self).encode("utf-8")).hexdigest()


class TAPPPhaseRatio(BaseModel):
    """
    TAP2 相比例的数据结构，包含钛合金各相的质量分数。
    """
    ALPHA: float = Field(default=0, description="ALPHA 相的质量分数（wt%）", ge=0, le=100)
    BETA: float = Field(default=0, description="BETA 相的质量分数（wt%）", ge=0, le=100)
    LIQUID: float = Field(default=0, description="LIQUID 相的质量分数（wt%）", ge=0, le=100)
    LAVES: float = Field(default=0, description="LAVES 相的质量分数（wt%）", ge=0, le=100)
    TI3AL: float = Field(default=0, description="TI3AL 相的质量分数（wt%）", ge=0, le=100)
    TI2CU: float = Field(default=0, description="TI2CU 相的质量分数（wt%）", ge=0, le=100)
    TI5SI3: float = Field(default=0, description="TI5SI3 相的质量分数（wt%）", ge=0, le=100)
    TIZRSI: float = Field(default=0, description="TIZRSI 相的质量分数（wt%）", ge=0, le=100)
    TI2NI: float = Field(default=0, description="TI2NI 相的质量分数（wt%）", ge=0, le=100)
    TIM_B2: float = Field(default=0, description="TIM_B2 相的质量分数（wt%）", ge=0, le=100)
    C15_FCC: float = Field(default=0, description="C15_FCC 相的质量分数（wt%）", ge=0, le=100)
    MC: float = Field(default=0, description="MC 相的质量分数（wt%）", ge=0, le=100)

    @model_validator(mode="after")
    def _valid_sum(self) -> Any:
        """
        验证各相质量分数之和是否为 100 wt%。
        :return: self
        """
        total = self.ALPHA + self.BETA + self.LIQUID + self.LAVES + self.TI3AL + self.TI2CU + self.TI5SI3 + self.TIZRSI + self.TI2NI + self.TIM_B2 + self.C15_FCC + self.MC
        if not isclose(total, 100, abs_tol=1e-3):
            raise TAPPException(f"The sum of all phase ratios must be 100, but got {total}.")
        return self

    def __str__(self) -> str:
        """
        将 TAPPPhaseRatio 实例转换为字符串表示。
        :return: TAPPPhaseRatio 实例的字符串表示，例如：{ALPHA=50, BETA=50, LIQUID=0, LAVES=0, TI3AL=0, TI2CU=0,
                 TI5SI3=0, TIZRSI=0, TI2NI=0, TIM_B2=0, C15_FCC=0, MC=0}。
        """
        return f'{{ALPHA={self.ALPHA:f}'.rstrip('0').rstrip('.') + \
               f', BETA={self.BETA:f}'.rstrip('0').rstrip('.') + \
               f', LIQUID={self.LIQUID:f}'.rstrip('0').rstrip('.') + \
               f', LAVES={self.LAVES:f}'.rstrip('0').rstrip('.') + \
               f', TI3AL={self.TI3AL:f}'.rstrip('0').rstrip('.') + \
               f', TI2CU={self.TI2CU:f}'.rstrip('0').rstrip('.') + \
               f', TI5SI3={self.TI5SI3:f}'.rstrip('0').rstrip('.') + \
               f', TIZRSI={self.TIZRSI:f}'.rstrip('0').rstrip('.') + \
               f', TI2NI={self.TI2NI:f}'.rstrip('0').rstrip('.') + \
               f', TIM_B2={self.TIM_B2:f}'.rstrip('0').rstrip('.') + \
               f', C15_FCC={self.C15_FCC:f}'.rstrip('0').rstrip('.') + \
               f', MC={self.MC:f}'.rstrip('0').rstrip('.') + '}'


class TAPPBatchPhaseRatio(BaseModel):
    """
    TAP2 批量相比例的数据结构，包含钛合金各相的质量分数列表。
    """
    ALPHA: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="ALPHA 相的质量分数列表（wt%）", examples=[[100, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    BETA: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="BETA 相的质量分数列表（wt%）", examples=[[0, 100]], min_length=1, max_length=_MAX_BATCH_SIZE)
    LIQUID: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="LIQUID 相的质量分数列表（wt%）", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    LAVES: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="LAVES 相的质量分数列表（wt%）", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    TI3AL: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="TI3AL 相的质量分数列表（wt%）", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    TI2CU: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="TI2CU 相的质量分数列表（wt%）", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    TI5SI3: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="TI5SI3 相的质量分数列表（wt%）", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    TIZRSI: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="TIZRSI 相的质量分数列表（wt%）", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    TI2NI: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="TI2NI 相的质量分数列表（wt%）", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    TIM_B2: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="TIM_B2 相的质量分数列表（wt%）", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    C15_FCC: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="C15_FCC 相的质量分数列表（wt%）", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)
    MC: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE, description="MC 相的质量分数列表（wt%）", examples=[[0, 0]], min_length=1, max_length=_MAX_BATCH_SIZE)

    @model_validator(mode="after")
    def _valid_sum(self) -> Any:
        """
        验证批量相比例的每一项是否均满足各相质量分数之和为 100 wt%，实际只检查前 N 项（N 为最小列表长度）。
        :return: self
        """
        for idx in range(len(self)):
            total = self.ALPHA[idx] + self.BETA[idx] + self.LIQUID[idx] + self.LAVES[idx] + self.TI3AL[idx] + self.TI2CU[idx] + self.TI5SI3[idx] + self.TIZRSI[idx] + self.TI2NI[idx] + self.TIM_B2[idx] + self.C15_FCC[idx] + self.MC[idx]
            if not isclose(total, 100, abs_tol=1e-3):
                raise TAPPException(f"The sum of all phase ratios must be 100, but got {total} at index {idx}.")
        return self

    def __len__(self) -> int:
        """
        计算 TAP2 批量相比例的大小，原则为取最短列表长度。
        :return: TAP2 批量相比例的大小
        """
        return min(len(self.ALPHA), len(self.BETA), len(self.LIQUID), len(self.LAVES), len(self.TI3AL), len(self.TI2CU), len(self.TI5SI3), len(self.TIZRSI), len(self.TI2NI), len(self.TIM_B2), len(self.C15_FCC), len(self.MC))

    def __str__(self) -> str:
        """
        将 TAPPBatchPhaseRatio 实例转换为字符串表示。
        :return: TAPPBatchPhaseRatio 实例的字符串表示。
        """
        size = len(self)
        return f'{{ALPHA=[{", ".join([f"{_:f}".rstrip("0").rstrip(".") for _ in self.ALPHA[:size]])}]' + \
               f', BETA=[{", ".join([f"{_:f}".rstrip("0").rstrip(".") for _ in self.BETA[:size]])}]' + \
               f', LIQUID=[{", ".join([f"{_:f}".rstrip("0").rstrip(".") for _ in self.LIQUID[:size]])}]' + \
               f', LAVES=[{", ".join([f"{_:f}".rstrip("0").rstrip(".") for _ in self.LAVES[:size]])}]' + \
               f', TI3AL=[{", ".join([f"{_:f}".rstrip("0").rstrip(".") for _ in self.TI3AL[:size]])}]' + \
               f', TI2CU=[{", ".join([f"{_:f}".rstrip("0").rstrip(".") for _ in self.TI2CU[:size]])}]' + \
               f', TI5SI3=[{", ".join([f"{_:f}".rstrip("0").rstrip(".") for _ in self.TI5SI3[:size]])}]' + \
               f', TIZRSI=[{", ".join([f"{_:f}".rstrip("0").rstrip(".") for _ in self.TIZRSI[:size]])}]' + \
               f', TI2NI=[{", ".join([f"{_:f}".rstrip("0").rstrip(".") for _ in self.TI2NI[:size]])}]' + \
               f', TIM_B2=[{", ".join([f"{_:f}".rstrip("0").rstrip(".") for _ in self.TIM_B2[:size]])}]' + \
               f', C15_FCC=[{", ".join([f"{_:f}".rstrip("0").rstrip(".") for _ in self.C15_FCC[:size]])}]' + \
               f', MC=[{", ".join([f"{_:f}".rstrip("0").rstrip(".") for _ in self.MC[:size]])}]' + '}'


class TAPPOutput(BaseModel):
    """
    TAP2 单一输出的数据结构，包含钛合金的性能代码、预测值及其单位。
    """
    Prop: Literal['BTT', 'WF', 'TE', 'DS', 'TC', 'EC', 'YM', 'BM', 'SM', 'PR', 'SE', 'SHC', 'YS', 'TS', 'HD', 'HP'] = Field(..., description='钛合金性能代码：β转变温度（BTT）、相比例（WF）、热膨胀系数（TE）、密度（DS）、热导率（TC）、电导率（EC）、杨氏模量（YM）、体积模量（BM）、剪切模量（SM）、泊松比（PR）、比焓（SE）、比热容（SHC）、屈服强度（YS）、抗拉强度（TS）、硬度（HD）、霍尔佩奇系数（HP）', examples=["DS", "WF"])
    value: float | TAPPPhaseRatio = Field(..., description='TAP2 模型预测的钛合金性能值，除相比例（WF）外，全部为浮点数类型，相比例为 TAPPPhaseRatio 类型，包含各相的质量分数。', examples=[5, TAPPPhaseRatio(ALPHA=50, BETA=50)])
    unit: Optional[str] = Field(default=None, description='钛合金性能单位', examples=['g/cm^3', 'wt%'])

    @model_validator(mode="after")
    def _valid_value_type(self) -> Any:
        """
        验证 value 的类型是否与 Prop 匹配：除相比例（WF）外，全部为浮点数类型，相比例为 TAPPPhaseRatio 类型。
        :return: self
        """
        if self.Prop == "WF":
            if not isinstance(self.value, TAPPPhaseRatio):
                raise TAPPException("For WF, value must be of type TAPPPhaseRatio.")
        else:
            if not isinstance(self.value, float):
                raise TAPPException(f"For {self.Prop}, value must be of type float.")
        return self

    @model_validator(mode='after')
    def _default_unit(self) -> Any:
        """
        设置默认单位：BTT（℃）、WF（wt%）、TE（K^-1）、DS（g/cm^3）、TC（W/m·K）、EC（S/m）、YM（GPa）、BM（GPa）、SM（GPa）、SE（J/g）、SHC（J/g·K）、YS（MPa）、TS（MPa）、HD（VPN）、HP（MPa·m^(1/2)）。
        :return: self
        """
        if self.Prop in _DEFAULT_UNIT_MAP and self.unit is None:
            self.unit = _DEFAULT_UNIT_MAP[self.Prop]
        return self

    def __str__(self) -> str:
        """
        将 TAPPOutput 实例转换为字符串表示。
        :return: TAPPOutput 实例的字符串表示，例如：{Prop=DS, value=4.43, unit=g/cm^3}
        """
        if self.Prop == "WF":
            value_str = str(self.value)
        else:
            value_str = f"{self.value:f}".rstrip('0').rstrip('.')
        output_str = f"{{Prop={self.Prop}, value={value_str}, unit=" + (self.unit if self.unit is not None else "None") + "}"
        return output_str


class TAPPBatchOutput(BaseModel):
    """
    TAP2 批量输出的数据结构，包含钛合金的性能代码、预测值列表及其单位。
    """
    Prop: Literal['BTT', 'WF', 'TE', 'DS', 'TC', 'EC', 'YM', 'BM', 'SM', 'PR', 'SE', 'SHC', 'YS', 'TS', 'HD', 'HP'] = Field(..., description='钛合金性能代码：β转变温度（BTT）、相比例（WF）、热膨胀系数（TE）、密度（DS）、热导率（TC）、电导率（EC）、杨氏模量（YM）、体积模量（BM）、剪切模量（SM）、泊松比（PR）、比焓（SE）、比热容（SHC）、屈服强度（YS）、抗拉强度（TS）、硬度（HD）、霍尔佩奇系数（HP）', examples=['DS', 'WF'])
    value: List[float] | TAPPBatchPhaseRatio = Field(..., description='TAP2 模型预测的钛合金性能值列表，除相比例（WF）外，全部为浮点数列表类型，相比例为 TAPPBatchPhaseRatio 类型，包含各相的质量分数列表。', examples=[[5, 6], TAPPBatchPhaseRatio(ALPHA=[100, 0], BETA=[0, 100])])
    unit: Optional[str] = Field(default=None, description='钛合金性能单位', examples=['g/cm^3', 'wt%'])

    @model_validator(mode="after")
    def _valid_value_type(self) -> Any:
        """
        验证 value 的类型是否与 Prop 匹配：除相比例（WF）外，全部为浮点数列表类型，相比例为 TAPPBatchPhaseRatio 类型。
        :return: self
        """
        if self.Prop == "WF":
            if not isinstance(self.value, TAPPBatchPhaseRatio):
                raise TAPPException("For WF, value must be of type TAPPBatchPhaseRatio.")
        else:
            if not isinstance(self.value, List):
                raise TAPPException(f"For {self.Prop}, value must be of type List[float].")
            if not all(isinstance(v, float) for v in self.value):
                raise TAPPException(f"For {self.Prop}, all items in value must be of type float.")
        return self

    @model_validator(mode='after')
    def _default_unit(self) -> Any:
        """
        设置默认单位：BTT（℃）、WF（wt%）、TE（K^-1）、DS（g/cm^3）、TC（W/m·K）、EC（S/m）、YM（GPa）、BM（GPa）、SM（GPa）、SE（J/g）、SHC（J/g·K）、YS（MPa）、TS（MPa）、HD（VPN）、HP（MPa·m^(1/2)）。
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
            value_str = str(self.value)
        else:
            value_str = f"[{', '.join(f'{_:f}'.rstrip('0').rstrip('.') for _ in self.value)}]"
        output_str = f"{{Prop={self.Prop}, value={value_str}, unit=" + (self.unit if self.unit is not None else "None") + "}"
        return output_str

    def __len__(self) -> int:
        """
        计算 TAP2 批量输出的大小，原则为取最短列表长度。
        :return: 批量输出的大小
        """
        return len(self.value)


class TAPPInfer:

    _prop_abbrs = ['BTT', 'WF', 'TE', 'DS', 'TC', 'EC', 'YM', 'BM', 'SM', 'PR', 'SE', 'SHC', 'YS', 'TS', 'HD', 'HP']

    _norm_params = {
        "BTT": {"Min": 202.56698, "Max": 1318.89996},
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

    def __init__(self, device: torch.device | None = None, batch_size: int = 64, silence: bool = False, skip_correction: bool = False):
        """
        使用 TAP2 模型进行推理的工具类，能够进行钛合金性能预测。
        :param device: 指定 Torch 设备，若为 None 优先选择 CUDA 设备。
        :param batch_size: 进行批量推理时的批大小，批大小不超过 10,000。
        :param silence: 是否开启静默模式，静默模式不打印运行日志。
        :param skip_correction: 是否跳过经验公式修正。
        """
        self._sil = silence
        self._skp = skip_correction
        self._dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        if not 0 < batch_size < _MAX_BATCH_SIZE:
            raise TAPPException('Batch size must be between 1 and 10,000.')
        self._bs = batch_size
        print(text2art('TAP2'), end='')
        print(f'Device: {self._dev}')
        print(f'Batch Size: {self._bs}')
        rprint('Silence: [green]YES[/green]' if self._sil else 'Silence: [red]NO[/red]')
        rprint('Correction: [red]NO[/red]' if self._skp else 'Correction: [green]YES[/green]')
        # 加载模型权重
        self._ms = {}
        for prop_abbr in self._prop_abbrs:
            prop_num = 12 if prop_abbr == "WF" else 1
            proc_num = 0 if prop_abbr == "BTT" else 1
            model_path = Path(str(resources.files("tap2.weight").joinpath(f"{prop_abbr}.pth")))
            model = MoE2(12, proc_num, prop_num, 512, 0.2).to(self._dev)
            model.load_state_dict(torch.load(model_path, map_location=self._dev))
            model.eval()
            self._ms[prop_abbr] = model
        # 加载修正参数
        if not self._skp:
            weight_path = Path(str(resources.files("tap2.weight").joinpath("corr_weights.pth")))
            self._cws = torch.load(weight_path, map_location=self._dev)

    def _short_task_flag(self, task_flag: str) -> str:
        """
        获取简短的任务标识，长度不超过 8。
        :param task_flag: 任务标识
        :return: 简短的任务标识
        """
        if len(task_flag) > self._task_flag_len:
            return task_flag[:self._task_flag_len] + '*'
        else:
            return task_flag

    def _infer(self, tapp_input: TAPPInput) -> float | List[float]:
        """
        调用 TAP2 模型进行单点推理，获取预测性能。
        :param tapp_input: 输入数据，包括性能代号、元素组成、热处理温度和晶粒尺寸（单一输入）。
        :return: 性能值（单一输出）。
        """
        with torch.inference_mode():
            model = self._ms[tapp_input.Prop]
            if tapp_input.Prop == 'BTT':
                input_T = torch.tensor(
                    data=[[tapp_input.Ti, tapp_input.Al, tapp_input.Cr, tapp_input.Cu, tapp_input.Fe, tapp_input.Mo, tapp_input.Ni, tapp_input.Nb, tapp_input.Si, tapp_input.Sn, tapp_input.V, tapp_input.Zr]],
                    dtype=torch.float32,
                    device=self._dev,
                    requires_grad=False
                ) # Shape: (1, 12)
            else:
                input_T = torch.tensor(
                    data=[[tapp_input.Ti, tapp_input.Al, tapp_input.Cr, tapp_input.Cu, tapp_input.Fe, tapp_input.Mo, tapp_input.Ni, tapp_input.Nb, tapp_input.Si, tapp_input.Sn, tapp_input.V, tapp_input.Zr, tapp_input.HTT]],
                    dtype=torch.float32,
                    device=self._dev,
                    requires_grad=False
                ) # Shape: (1, 13)
            output_T = model(input_T) # Shape: (1, PropNum)
            if tapp_input.Prop == 'WF':
                output_T = self._sparsemax(output_T) # Shape: (1, 12)
                output_T = output_T * 100 # Shape: (1, 12)
                output = output_T.tolist()[0] # Shape: (12,)
            else:
                min_val = self._norm_params[tapp_input.Prop]["Min"]
                max_val = self._norm_params[tapp_input.Prop]["Max"]
                output_T = output_T * (max_val - min_val) + min_val # Shape: (1, 1)
                output = output_T.item()
        return output

    def _batch_infer(self, tapp_input: TAPPBatchInput) -> List[float] | List[List[float]]:
        """
        调用 TAP2 模型进行批量推理，获取预测性能。
        :param tapp_input: 输入数据，包括性能代号、元素组成、热处理温度和晶粒尺寸（批量输入）。
        :return: 性能值（批量输出）。
        """
        input_size = len(tapp_input)
        batch_num = ceil(input_size / self._bs)
        if tapp_input.Prop == 'WF':
            outputs = [[]] * 12
        else:
            outputs = []
        with torch.inference_mode():
            model = self._ms[tapp_input.Prop]
            for batch_id in range(batch_num):
                begin_idx = batch_id * self._bs
                end_idx = min(begin_idx + self._bs, input_size)
                if tapp_input.Prop == 'BTT':
                    inputs_T = torch.tensor(
                        data=[tapp_input.Ti[begin_idx: end_idx], tapp_input.Al[begin_idx: end_idx], tapp_input.Cr[begin_idx: end_idx], tapp_input.Cu[begin_idx: end_idx], tapp_input.Fe[begin_idx: end_idx], tapp_input.Mo[begin_idx: end_idx], tapp_input.Ni[begin_idx: end_idx], tapp_input.Nb[begin_idx: end_idx], tapp_input.Si[begin_idx: end_idx], tapp_input.Sn[begin_idx: end_idx], tapp_input.V[begin_idx: end_idx], tapp_input.Zr[begin_idx: end_idx]],
                        dtype=torch.float32,
                        device=self._dev,
                        requires_grad=False
                    )  # Shape: (12, BatchSize)
                    inputs_T = inputs_T.T  # Shape: (BatchSize, 12)
                else:
                    inputs_T = torch.tensor(
                        data=[tapp_input.Ti[begin_idx: end_idx], tapp_input.Al[begin_idx: end_idx], tapp_input.Cr[begin_idx: end_idx], tapp_input.Cu[begin_idx: end_idx], tapp_input.Fe[begin_idx: end_idx], tapp_input.Mo[begin_idx: end_idx], tapp_input.Ni[begin_idx: end_idx], tapp_input.Nb[begin_idx: end_idx], tapp_input.Si[begin_idx: end_idx], tapp_input.Sn[begin_idx: end_idx], tapp_input.V[begin_idx: end_idx], tapp_input.Zr[begin_idx: end_idx], tapp_input.HTT[begin_idx: end_idx]],
                        dtype=torch.float32,
                        device=self._dev,
                        requires_grad=False
                    ) # Shape: (13, BatchSize)
                    inputs_T = inputs_T.T # Shape: (BatchSize, 13)
                outputs_T = model(inputs_T) # Shape: (BatchSize, PropNum)
                if tapp_input.Prop == "WF":
                    outputs_T = self._sparsemax(outputs_T) # Shape: (BatchSize, 12)
                    outputs_T = outputs_T.T  # Shape: (12, BatchSize)
                    outputs_T = outputs_T * 100  # Shape: (12, BatchSize)
                    batch_outputs = outputs_T.tolist()  # Shape: (12, BatchSize)
                    for x, y in zip(outputs, batch_outputs):
                        x.extend(y)
                else:
                    min_val = self._norm_params[tapp_input.Prop]["Min"]
                    max_val = self._norm_params[tapp_input.Prop]["Max"]
                    outputs_T = outputs_T.squeeze(-1)  # Shape: (BatchSize,)
                    outputs_T = outputs_T * (max_val - min_val) + min_val # Shape: (BatchSize,)
                    batch_outputs = outputs_T.tolist()  # Shape: (BatchSize,)
                    outputs.extend(batch_outputs)
        return outputs

    def _corr(self, tapp_input: TAPPInput, orig_output: float | List[float]) -> float | List[float]:
        """
        利用经验公式对 TAP2 模型的预测结果进行修正。
        :param tapp_input: 输入数据，包括性能代号、元素组成、热处理温度和晶粒尺寸（单一输入）。
        :param orig_output: 原始性能值（单一输出）。
        :return: 修正性能值（单一输出）。
        """
        correct = deepcopy(orig_output)
        if tapp_input.Prop in ["YS", "TS"]:
            with torch.inference_mode():
                micro_elem_T = torch.tensor(
                    data=[tapp_input.N, tapp_input.O, tapp_input.C, tapp_input.H, tapp_input.B],
                    dtype=torch.float32,
                    device=self._dev,
                    requires_grad=False
                ) # Shape: (5,)
                increment = torch.sum(self._cws['TS_YS'] * micro_elem_T)
                if tapp_input.Prop == "TS":
                    increment *= 1.1
                increment = increment.item()
            grain_size = tapp_input.GS * 1e-6
            hall_petch = self._infer(tapp_input.model_copy(update={"Prop": "HP", "GS": 10}, deep=True))
            increment += hall_petch * (pow(grain_size, -0.5) - pow(1e-5, -0.5))
            correct += increment
        elif tapp_input.Prop in ["DS", "TC", "EC", "YM", "BM", "SM", "PR"]:
            with torch.inference_mode():
                micro_elem_T = torch.tensor(
                    data=[[tapp_input.C, tapp_input.N, tapp_input.O, tapp_input.B, tapp_input.H]],
                    dtype=torch.float32,
                    device=self._dev,
                    requires_grad=False
                ) # Shape: (1, 5)
                main_elem_T = torch.tensor(
                    data=[[tapp_input.Al], [tapp_input.Cr], [tapp_input.Cu], [tapp_input.Fe], [tapp_input.Mo], [tapp_input.Nb], [tapp_input.Ni], [tapp_input.Si], [tapp_input.Sn]],
                    dtype=torch.float32,
                    device=self._dev,
                    requires_grad=False
                ) # Shape: (9, 1)
                increment = (micro_elem_T @ self._cws[tapp_input.Prop] @ main_elem_T)
                increment = increment.item()
            correct += increment
        elif tapp_input.Prop == "TE":
            with torch.inference_mode():
                micro_elem_T = torch.tensor(
                    data=[[1, 1, 1, 1, 1], [tapp_input.N, tapp_input.O, tapp_input.C, tapp_input.H, tapp_input.B], [tapp_input.N ** 2, tapp_input.O ** 2, tapp_input.C ** 2, tapp_input.H ** 2, tapp_input.B ** 2], [tapp_input.N ** 3, tapp_input.O ** 3, tapp_input.C ** 3, tapp_input.H ** 3, tapp_input.B ** 3]],
                    dtype=torch.float32,
                    device=self._dev,
                    requires_grad=False
                ) # Shape: (4, 5)
                increment = torch.sum(micro_elem_T * self._cws['TE'])
                increment = increment.item()
            correct += increment
        return correct

    def _batch_corr(self, tapp_input: TAPPBatchInput, orig_output: List[float] | List[List[float]]) -> List[float] | List[List[float]]:
        """
        利用经验公式对 TAP2 模型的预测结果进行修正。
        :param tapp_input: 输入数据，包括性能代号、元素组成、热处理温度和晶粒尺寸（批量输入）。
        :param orig_output: 原始性能值（批量输出）。
        :return: 修正性能值（批量输出）。
        """
        input_size = len(tapp_input)
        batch_num = ceil(input_size / self._bs)
        corrects = deepcopy(orig_output)
        if tapp_input.Prop in ["YS", "TS"]:
            increments = []
            with torch.inference_mode():
                hall_petch_T = torch.tensor(
                    data=self._batch_infer(tapp_input.model_copy(update={"Prop": "HP", "GS": [10] * input_size})),
                    dtype=torch.float32,
                    device=self._dev,
                    requires_grad=False
                ) # Shape: (InputSize,)
                for batch_id in range(batch_num):
                    begin_idx = batch_id * self._bs
                    end_idx = min(begin_idx + self._bs, input_size)
                    micro_elem_T = torch.tensor(
                        data=[tapp_input.N[begin_idx: end_idx], tapp_input.O[begin_idx: end_idx], tapp_input.C[begin_idx: end_idx], tapp_input.H[begin_idx: end_idx], tapp_input.B[begin_idx: end_idx]],
                        dtype=torch.float32,
                        device=self._dev,
                        requires_grad=False
                    ) # Shape: (5, BatchSize)
                    micro_elem_T = micro_elem_T.T # Shape: (BatchSize, 5)
                    increments_T = torch.sum(self._cws['TS_YS'] * micro_elem_T, dim=1) # Shape: (BatchSize,)
                    if tapp_input.Prop == "TS":
                        increments_T *= 1.1
                    grain_size_T = torch.tensor(
                        data=tapp_input.GS[begin_idx: end_idx],
                        dtype=torch.float32,
                        device=self._dev,
                        requires_grad=False
                    ) # Shape: (BatchSize,)
                    grain_size_T *= 1e-6
                    increments_T += hall_petch_T[begin_idx: end_idx] * (grain_size_T ** -0.5 - 1e-5 ** -0.5) # Shape: (BatchSize,)
                    batch_increments = increments_T.tolist()  # Shape: (BatchSize,)
                    increments.extend(batch_increments)
            for idx, increment in enumerate(increments):
                corrects[idx] += increment
        elif tapp_input.Prop in ["DS", "TC", "EC", "YM", "BM", "SM", "PR"]:
            increments = []
            with torch.inference_mode():
                for batch_id in range(batch_num):
                    begin_idx = batch_id * self._bs
                    end_idx = min(begin_idx + self._bs, input_size)
                    micro_elem_T = torch.tensor(
                        data=[tapp_input.C[begin_idx: end_idx], tapp_input.N[begin_idx: end_idx], tapp_input.O[begin_idx: end_idx], tapp_input.B[begin_idx: end_idx], tapp_input.H[begin_idx: end_idx]],
                        dtype=torch.float32,
                        device=self._dev,
                        requires_grad=False
                    ) # Shape: (5, BatchSize)
                    micro_elem_T = micro_elem_T.T.unsqueeze(1) # Shape: (BatchSize, 1, 5)
                    main_elem_T = torch.tensor(
                        data=[tapp_input.Al[begin_idx: end_idx], tapp_input.Cr[begin_idx: end_idx], tapp_input.Cu[begin_idx: end_idx], tapp_input.Fe[begin_idx: end_idx], tapp_input.Mo[begin_idx: end_idx], tapp_input.Nb[begin_idx: end_idx], tapp_input.Ni[begin_idx: end_idx], tapp_input.Si[begin_idx: end_idx], tapp_input.Sn[begin_idx: end_idx]],
                        dtype=torch.float32,
                        device=self._dev,
                        requires_grad=False
                    ) # Shape: (9, BatchSize)
                    main_elem_T = main_elem_T.T.unsqueeze(-1) # Shape: (BatchSize, 9, 1)
                    increments_T = micro_elem_T @ self._cws[tapp_input.Prop] @ main_elem_T  # Shape: (BatchSize, 1, 1)
                    increments_T = increments_T.squeeze(-1).squeeze(-1)  # Shape: (BatchSize,)
                    batch_increments = increments_T.tolist()
                    increments.extend(batch_increments)
            for idx, increment in enumerate(increments):
                corrects[idx] += increment
        elif tapp_input.Prop == "TE":
            increments = []
            with torch.inference_mode():
                for batch_id in range(batch_num):
                    begin_idx = batch_id * self._bs
                    end_idx = min(begin_idx + self._bs, input_size)
                    micro_elem_T = torch.tensor(
                        data=[tapp_input.N[begin_idx: end_idx], tapp_input.O[begin_idx: end_idx], tapp_input.C[begin_idx: end_idx], tapp_input.H[begin_idx: end_idx], tapp_input.B[begin_idx: end_idx]],
                        dtype=torch.float32,
                        device=self._dev
                    ) # Shape: (5, BatchSize)
                    micro_elem_T = micro_elem_T.T  # Shape: (BatchSize, 5)
                    micro_elem_T = torch.stack([torch.ones_like(micro_elem_T), micro_elem_T, micro_elem_T ** 2, micro_elem_T ** 3], dim=1) # Shape: (BatchSize, 4, 5)
                    increments_T = torch.sum(micro_elem_T * self._cws['TE'], dim=(1, 2)) # Shape: (BatchSize,)
                    batch_increments = increments_T.tolist()
                    increments.extend(batch_increments)
            for idx, increment in enumerate(increments):
                corrects[idx] += increment
        return corrects

    def __call__(self, tapp_input: TAPPInput | TAPPBatchInput) -> TAPPOutput | TAPPBatchOutput:
        """
        调用 TAP2 模型进行推理，预测钛合金的性能，支持批量处理。
        :param tapp_input: 包括待预测性能名称、钛合金的元素组成、热处理温度和晶粒尺寸（单一输入或批量输入）
        :return: TAP2 预测的性能值（单一输出或批量输出）
        """
        task_flag = self._short_task_flag(tapp_input.sha256)
        if isinstance(tapp_input, TAPPInput):
            if not self._sil:
                logger.info(f'[{task_flag}] TAP2 INFERRING: INPUT = {tapp_input}')
            infer_func = self._infer
            corr_func = self._corr
            output_type = TAPPOutput
            phase_ratio_type = TAPPPhaseRatio
        else:
            if not self._sil:
                logger.info(f'[{task_flag}] TAP2 BATCH INFERRING: PROP = {tapp_input.Prop}, SIZE = {len(tapp_input)} its')
            infer_func = self._batch_infer
            corr_func = self._batch_corr
            output_type = TAPPBatchOutput
            phase_ratio_type = TAPPBatchPhaseRatio
        output_value = infer_func(tapp_input)
        if not self._skp:
            output_value = corr_func(tapp_input, output_value)
        if tapp_input.Prop == 'WF':
            tapp_output = output_type(Prop='WF', value=phase_ratio_type(
                ALPHA=output_value[0],
                BETA=output_value[1],
                LAVES=output_value[2],
                TI3AL=output_value[3],
                TI2CU=output_value[4],
                TI5SI3=output_value[5],
                TIZRSI=output_value[6],
                TI2NI=output_value[7],
                TIM_B2=output_value[8],
                LIQUID=output_value[9],
                C15_FCC=output_value[10],
                MC=output_value[11]
            ))
        else:
            tapp_output = output_type(Prop=tapp_input.Prop, value=output_value)
        if isinstance(tapp_output, TAPPOutput):
            if not self._sil:
                logger.info(f'[{task_flag}] TAP2 INFERRED: OUTPUT = {tapp_output}')
        else:
            if not self._sil:
                logger.info(f'[{task_flag}] TAP2 BATCH INFERRED: PROP = {tapp_output.Prop}, SIZE = {len(tapp_output)} its')
        return tapp_output

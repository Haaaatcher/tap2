from hashlib import sha256
from copy import deepcopy
from tap2.model import MoE2, AAModel
from pathlib import Path
from importlib import resources
from typing import Any, List, Annotated, Optional
from pydantic import BaseModel, Field, model_validator
from sparsemax import Sparsemax
from loguru import logger
from math import isclose, ceil
from enum import Enum
import torch
import json


# Maximum batch size to prevent memory overflow
_MAX_BATCH_SIZE_ = 10000


class TAPropAbbr(Enum):
    """
    Abbreviation for titanium alloy property
    """
    BTT = 'BTT'
    WF = 'WF'
    TE = 'TE'
    DS = 'DS'
    TC = 'TC'
    EC = 'EC'
    YM = 'YM'
    BM = 'BM'
    SM = 'SM'
    PR = 'PR'
    SE = 'SE'
    SHC = 'SHC'
    YS = 'YS'
    TS = 'TS'
    HD = 'HD'
    HP = 'HP'


class TAInput(BaseModel):
    """
    The input data structure of the TAInfer class (single)
    contains the property abbreviation, elemental composition, heat treatment temperature and grain size.
    """
    prop: TAPropAbbr = Field(..., description=('钛合金性能代码：β-转变温度（BTT）、相比例（WF）、热膨胀系数（TE）、密度（DS）、'
                                               '热导率（TC）、电导率（EC）、杨氏模量（YM）、体积模量（BM）、剪切模量（SM）、'
                                               '泊松比（PR）、比焓（SE）、比热容（SHC）、屈服强度（YS）、抗拉强度（TS）、'
                                               '硬度（HD）、霍尔佩奇系数（HP）'), examples=['DS'])
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
    HTT: Optional[float] = Field(default=None, description='热处理温度（℃），除预测β-转变温度（BTT）外，均须提供热处理温度',
                                 examples=[600], gt=-273.15)
    GS: Optional[float] = Field(default=None, description='晶粒尺寸（μm），预测力学性能（YS、TS、HD、HP）须提供晶粒尺寸',
                                examples=[10], gt=0)

    @model_validator(mode='after')
    def _valid_compos_(self) -> Any:
        """
        Verify that the sum of the elemental composition is 100 wt%.
        :return: self
        """
        total = sum([self.Ti, self.H, self.B, self.C, self.N, self.O, self.Al, self.Si, self.Cr, self.Fe, self.Ni,
                     self.Cu, self.Zr, self.Nb, self.Mo, self.V, self.Sn])
        if not isclose(total, 100, abs_tol=1e-6):
            raise ValueError(f'The sum of all element compositions must be 100, but got {total}.')
        return self

    @model_validator(mode='after')
    def _valid_htt_(self) -> Any:
        """
        Verify that heat treatment temperatures (other than BTT) are provided.
        :return: self
        """
        if self.prop is not TAPropAbbr.BTT and self.HTT is None:
            raise ValueError(f'Heat treatment temperature (HTT) is required for {self.prop.value}.')
        return self

    @model_validator(mode='after')
    def _valid_gs_(self) -> Any:
        """
        Verify that the mechanical property prediction provides the grain size.
        :return: self
        """
        if self.prop in (TAPropAbbr.YS, TAPropAbbr.TS, TAPropAbbr.HD, TAPropAbbr.HP) and self.GS is None:
            raise ValueError(f'Grains size (GS) is required for {self.prop.value}.')
        return self

    def __str__(self) -> str:
        """
        Converts the TAInput instance to a string representation, which complies with the JSON specification.
        :return: The string representation of the TAInput instance
        """
        return json.dumps({
            'prop': self.prop.value,
            'Ti': f'{self.Ti:f}'.rstrip('0').rstrip('.'),
            'H': f'{self.H:f}'.rstrip('0').rstrip('.'),
            'B': f'{self.B:f}'.rstrip('0').rstrip('.'),
            'C': f'{self.C:f}'.rstrip('0').rstrip('.'),
            'N': f'{self.N:f}'.rstrip('0').rstrip('.'),
            'O': f'{self.O:f}'.rstrip('0').rstrip('.'),
            'Al': f'{self.Al:f}'.rstrip('0').rstrip('.'),
            'Si': f'{self.Si:f}'.rstrip('0').rstrip('.'),
            'Cr': f'{self.Cr:f}'.rstrip('0').rstrip('.'),
            'Fe': f'{self.Fe:f}'.rstrip('0').rstrip('.'),
            'Ni': f'{self.Ni:f}'.rstrip('0').rstrip('.'),
            'Cu': f'{self.Cu:f}'.rstrip('0').rstrip('.'),
            'Zr': f'{self.Zr:f}'.rstrip('0').rstrip('.'),
            'Nb': f'{self.Nb:f}'.rstrip('0').rstrip('.'),
            'Mo': f'{self.Mo:f}'.rstrip('0').rstrip('.'),
            'V': f'{self.V:f}'.rstrip('0').rstrip('.'),
            'Sn': f'{self.Sn:f}'.rstrip('0').rstrip('.'),
            'HTT': f"{self.HTT:f}".rstrip('0').rstrip('.') if self.HTT is not None else None,
            'GS': f"{self.GS:f}".rstrip('0').rstrip('.') if self.GS is not None else None
        }, ensure_ascii=False)

    @property
    def sha256(self) -> str:
        """
        Calculate the SHA256 hash value of the TAInput instance.
        :return: The SHA256 hash of the TAInput instance
        """
        return sha256(str(self).encode("utf-8")).hexdigest()


class TABatchInput(BaseModel):
    """
    The input data structure of the TAInfer class (batch)
    contains the property abbreviation and lists of the elemental composition, heat treatment temperature, grain size.
    """
    prop: TAPropAbbr = Field(..., description='钛合金性能代码：β-转变温度（BTT）、相比例（WF）、热膨胀系数（TE）、密度（DS）、'
                                              '热导率（TC）、电导率（EC）、杨氏模量（YM）、体积模量（BM）、剪切模量（SM）、'
                                              '泊松比（PR）、比焓（SE）、比热容（SHC）、屈服强度（YS）、抗拉强度（TS）、'
                                              '硬度（HD）、霍尔佩奇系数（HP）', examples=["DS"])
    Ti: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                            description="钛（Ti）的质量分数列表",
                                                            examples=[[90, 97]],
                                                            min_length=1,
                                                            max_length=_MAX_BATCH_SIZE_)
    H: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                           description="氢（H）的质量分数列表",
                                                           examples=[[0, 0]],
                                                           min_length=1,
                                                           max_length=_MAX_BATCH_SIZE_)
    B: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                           description="硼（B）的质量分数列表",
                                                           examples=[[0, 0]],
                                                           min_length=1,
                                                           max_length=_MAX_BATCH_SIZE_)
    C: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                           description="碳（C）的质量分数列表",
                                                           examples=[[0, 0]],
                                                           min_length=1,
                                                           max_length=_MAX_BATCH_SIZE_)
    N: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                           description="氮（N）的质量分数列表",
                                                           examples=[[0, 0]],
                                                           min_length=1,
                                                           max_length=_MAX_BATCH_SIZE_)
    O: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                           description="氧（O）的质量分数列表",
                                                           examples=[[0, 0]],
                                                           min_length=1,
                                                           max_length=_MAX_BATCH_SIZE_)
    Al: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                            description="铝（Al）的质量分数列表",
                                                            examples=[[6, 3]],
                                                            min_length=1,
                                                            max_length=_MAX_BATCH_SIZE_)
    Si: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                            description="硅（Si）的质量分数列表",
                                                            examples=[[0, 0]],
                                                            min_length=1,
                                                            max_length=_MAX_BATCH_SIZE_)
    Cr: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                            description="铬（Cr）的质量分数列表",
                                                            examples=[[0, 0]],
                                                            min_length=1,
                                                            max_length=_MAX_BATCH_SIZE_)
    Fe: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                            description="铁（Fe）的质量分数列表",
                                                            examples=[[0, 0]],
                                                            min_length=1,
                                                            max_length=_MAX_BATCH_SIZE_)
    Ni: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                            description="镍（Ni）的质量分数列表",
                                                            examples=[[0, 0]],
                                                            min_length=1,
                                                            max_length=_MAX_BATCH_SIZE_)
    Cu: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                            description="铜（Cu）的质量分数列表",
                                                            examples=[[0, 0]],
                                                            min_length=1,
                                                            max_length=_MAX_BATCH_SIZE_)
    Zr: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                            description="锆（Zr）的质量分数列表",
                                                            examples=[[0, 0]],
                                                            min_length=1,
                                                            max_length=_MAX_BATCH_SIZE_)
    Nb: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                            description="铌（Nb）的质量分数列表",
                                                            examples=[[0, 0]],
                                                            min_length=1,
                                                            max_length=_MAX_BATCH_SIZE_)
    Mo: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                            description="钼（Mo）的质量分数列表",
                                                            examples=[[0, 0]],
                                                            min_length=1,
                                                            max_length=_MAX_BATCH_SIZE_)
    V: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                           description="钒（V）的质量分数列表",
                                                           examples=[[4, 0]],
                                                           min_length=1,
                                                           max_length=_MAX_BATCH_SIZE_)
    Sn: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                            description="锡（Sn）的质量分数列表",
                                                            examples=[[0, 0]],
                                                            min_length=1,
                                                            max_length=_MAX_BATCH_SIZE_)
    HTT: Optional[List[Annotated[float, Field(gt=-273.15)]]] = \
        Field(default=None, description='热处理温度（℃），除预测β-转变温度（BTT）外，均须提供热处理温度',
              examples=[[600, 800]], min_length=1, max_length=_MAX_BATCH_SIZE_)
    GS: Optional[List[Annotated[float, Field(gt=0)]]] = \
        Field(default=None, description='晶粒尺寸（μm），预测力学性能（YS、TS、HD、HP）须提供晶粒尺寸',
              examples=[[10, 20]], min_length=1, max_length=_MAX_BATCH_SIZE_)

    @model_validator(mode='after')
    def _valid_compos_(self) -> Any:
        """
        Verify that the sum of the elemental composition is 100 wt%.
        :return: self
        """
        for idx in range(len(self)):
            total = self.Ti[idx] + self.H[idx] + self.B[idx] + self.C[idx] + self.N[idx] + self.O[idx] + self.Al[idx] + \
                    self.Si[idx] + self.Cr[idx] + self.Fe[idx] + self.Ni[idx] + self.Cu[idx] + self.Zr[idx] + \
                    self.Nb[idx] + self.Mo[idx] + self.V[idx] + self.Sn[idx]
            if not isclose(total, 100, abs_tol=1e-6):
                raise ValueError(f"The sum of all element compositions must be 100, but got {total} at index {idx}.")
        return self

    @model_validator(mode='after')
    def _valid_htt_(self) -> Any:
        """
        Verify that heat treatment temperatures (other than BTT) are provided.
        :return: self
        """
        if self.prop is not TAPropAbbr.BTT and self.HTT is None:
            raise ValueError(f'Heat treatment temperature (HTT) is required for {self.prop.value}.')
        return self

    @model_validator(mode='after')
    def _valid_gs_(self) -> Any:
        """
        Verify that the mechanical property prediction provides the grain size.
        :return: self
        """
        if self.prop in (TAPropAbbr.TS, TAPropAbbr.YS, TAPropAbbr.HD, TAPropAbbr.HP) and self.GS is None:
            raise ValueError(f'Grains size (GS) is required for {self.prop.value}.')
        return self

    def __len__(self) -> int:
        """
        Calculate the batch size and take the shortest list length.
        :return: Batch size
        """
        l = min(len(self.Ti), len(self.H), len(self.B), len(self.C), len(self.N), len(self.O), len(self.Al),
                len(self.Si), len(self.Cr), len(self.Fe), len(self.Ni), len(self.Cu), len(self.Zr), len(self.Nb),
                len(self.Mo), len(self.V), len(self.Sn))
        if self.HTT is not None:
            l = min(l, len(self.HTT))
        if self.GS is not None:
            l = min(l, len(self.GS))
        return l


class TAPhaseRatio(BaseModel):
    """
    The phase ration data structure of TAInfer class (single)
    contains the mass fraction of each phase.
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


class TABatchPhaseRatio(BaseModel):
    """
    The phase ration data structure of TAInfer class (batch)
    contains the list of mass fraction of each phase.
    """
    ALPHA: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                               description="ALPHA 相的质量分数列表（wt%）",
                                                               examples=[[100, 0]],
                                                               min_length=1,
                                                               max_length=_MAX_BATCH_SIZE_)
    BETA: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                              description="BETA 相的质量分数列表（wt%）",
                                                              examples=[[0, 100]],
                                                              min_length=1,
                                                              max_length=_MAX_BATCH_SIZE_)
    LIQUID: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                                description="LIQUID 相的质量分数列表（wt%）",
                                                                examples=[[0, 0]],
                                                                min_length=1,
                                                                max_length=_MAX_BATCH_SIZE_)
    LAVES: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                               description="LAVES 相的质量分数列表（wt%）",
                                                               examples=[[0, 0]],
                                                               min_length=1,
                                                               max_length=_MAX_BATCH_SIZE_)
    TI3AL: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                               description="TI3AL 相的质量分数列表（wt%）",
                                                               examples=[[0, 0]],
                                                               min_length=1,
                                                               max_length=_MAX_BATCH_SIZE_)
    TI2CU: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                               description="TI2CU 相的质量分数列表（wt%）",
                                                               examples=[[0, 0]],
                                                               min_length=1,
                                                               max_length=_MAX_BATCH_SIZE_)
    TI5SI3: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                                description="TI5SI3 相的质量分数列表（wt%）",
                                                                examples=[[0, 0]],
                                                                min_length=1,
                                                                max_length=_MAX_BATCH_SIZE_)
    TIZRSI: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                                description="TIZRSI 相的质量分数列表（wt%）",
                                                                examples=[[0, 0]],
                                                                min_length=1,
                                                                max_length=_MAX_BATCH_SIZE_)
    TI2NI: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                               description="TI2NI 相的质量分数列表（wt%）",
                                                               examples=[[0, 0]],
                                                               min_length=1,
                                                               max_length=_MAX_BATCH_SIZE_)
    TIM_B2: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                                description="TIM_B2 相的质量分数列表（wt%）",
                                                                examples=[[0, 0]],
                                                                min_length=1,
                                                                max_length=_MAX_BATCH_SIZE_)
    C15_FCC: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                                 description="C15_FCC 相的质量分数列表（wt%）",
                                                                 examples=[[0, 0]],
                                                                 min_length=1,
                                                                 max_length=_MAX_BATCH_SIZE_)
    MC: List[Annotated[float, Field(ge=0, le=100)]] = Field(default=[0] * _MAX_BATCH_SIZE_,
                                                            description="MC 相的质量分数列表（wt%）",
                                                            examples=[[0, 0]],
                                                            min_length=1,
                                                            max_length=_MAX_BATCH_SIZE_)

    def __len__(self) -> int:
        """
        Calculate the batch size and take the shortest list length.
        :return: Batch size
        """
        return min(len(self.ALPHA), len(self.BETA), len(self.LIQUID), len(self.LAVES), len(self.TI3AL), len(self.TI2CU),
                   len(self.TI5SI3), len(self.TIZRSI), len(self.TI2NI), len(self.TIM_B2), len(self.C15_FCC), len(self.MC))


class TAOutput(BaseModel):
    """
    The output data structure of the TAInfer class (single)
    contains the property abbreviation, predicted value and unit.
    """
    prop: TAPropAbbr = Field(..., description=('钛合金性能代码：β-转变温度（BTT）、相比例（WF）、热膨胀系数（TE）、密度（DS）、'
                                               '热导率（TC）、电导率（EC）、杨氏模量（YM）、体积模量（BM）、剪切模量（SM）、'
                                               '泊松比（PR）、比焓（SE）、比热容（SHC）、屈服强度（YS）、抗拉强度（TS）、'
                                               '硬度（HD）、霍尔佩奇系数（HP）'),
                             examples=["DS", "WF"])
    value: float | TAPhaseRatio = Field(..., description=('钛合金性能值，除相比例外，全部为浮点数类型，相比例为 TAPhaseRatio 类型，'
                                                          '包含各相的质量分数'),
                                        examples=[5, TAPhaseRatio(ALPHA=50, BETA=50)])
    unit: Optional[str] = Field(default=None, description='钛合金性能单位', examples=['g/cm^3', 'wt%'])

    def __str__(self) -> str:
        """
        Converts the TAOutput instance to a string representation that meets the JSON specification.
        :return: The string representation of the TAOutput instance
        """
        if self.prop is TAPropAbbr.WF:
            value = {
                'ALPHA': f'{self.value.ALPHA:f}'.rstrip('0').rstrip('.'),
                'BETA': f'{self.value.BETA:f}'.rstrip('0').rstrip('.'),
                'LIQUID': f'{self.value.LIQUID:f}'.rstrip('0').rstrip('.'),
                'LAVES': f'{self.value.LAVES:f}'.rstrip('0').rstrip('.'),
                'TI3AL': f'{self.value.TI3AL:f}'.rstrip('0').rstrip('.'),
                'TI2CU': f'{self.value.TI2CU:f}'.rstrip('0').rstrip('.'),
                'TI5SI3': f'{self.value.TI5SI3:f}'.rstrip('0').rstrip('.'),
                'TIZRSI': f'{self.value.TIZRSI:f}'.rstrip('0').rstrip('.'),
                'TI2NI': f'{self.value.TI2NI:f}'.rstrip('0').rstrip('.'),
                'TIM_B2': f'{self.value.TIM_B2:f}'.rstrip('0').rstrip('.'),
                'C15_FCC': f'{self.value.C15_FCC:f}'.rstrip('0').rstrip('.'),
                'MC': f'{self.value.MC:f}'.rstrip('0').rstrip('.')
            }
        else:
            value = f'{self.value:f}'.rstrip('0').rstrip('.')
        return json.dumps({
            'prop': self.prop.value,
            'value': value,
            'unit': self.unit
        }, ensure_ascii=False)


class TABatchOutput(BaseModel):
    """
    The output data structure of the TAInfer class (batch)
    contains the property abbreviation and lists of predicted value, unit.
    """
    prop: TAPropAbbr = Field(..., description='钛合金性能代码：β转变温度（BTT）、相比例（WF）、热膨胀系数（TE）、密度（DS）、'
                                              '热导率（TC）、电导率（EC）、杨氏模量（YM）、体积模量（BM）、剪切模量（SM）、'
                                              '泊松比（PR）、比焓（SE）、比热容（SHC）、屈服强度（YS）、抗拉强度（TS）、'
                                              '硬度（HD）、霍尔佩奇系数（HP）', examples=['DS', 'WF'])
    value: List[float] | TABatchPhaseRatio = Field(..., description='钛合金性能值，除相比例外，全部为浮点数列表类型，'
                                                                    '相比例为 TABatchPhaseRatio 类型。',
                                                   examples=[[5, 6], TABatchPhaseRatio(ALPHA=[100, 0], BETA=[0, 100])])
    unit: Optional[str] = Field(default=None, description='钛合金性能单位', examples=['g/cm^3', 'wt%'])

    def __len__(self) -> int:
        """
        Calculate the batch size and take the shortest list length.
        :return: Batch size
        """
        return len(self.value)


class TAInfer:
    _norm_params_ = {
        TAPropAbbr.BTT: {"Min": 202.56698, "Max": 1318.89996},
        TAPropAbbr.TE: {"Min": 4.848329916e-06, "Max": 2.283828168e-05},
        TAPropAbbr.DS: {"Min": 3.672575187, "Max": 10.34887892},
        TAPropAbbr.TC: {"Min": -87.38449318, "Max": 116.9406027},
        TAPropAbbr.EC: {"Min": 257.542498, "Max": 6101953.5},
        TAPropAbbr.YM: {"Min": -98.60315674, "Max": 273.60016},
        TAPropAbbr.BM: {"Min": -91.34943727, "Max": 232.2667855},
        TAPropAbbr.SM: {"Min": -37.34688091, "Max": 104.9342765},
        TAPropAbbr.PR: {"Min": 0.2569257344, "Max": 0.5},
        TAPropAbbr.SE: {"Min": -2140.62959, "Max": 1075.43465},
        TAPropAbbr.SHC: {"Min": 0.0016, "Max": 0.96764},
        TAPropAbbr.YS: {"Min": 2.297351749, "Max": 1896.478209},
        TAPropAbbr.TS: {"Min": 2.576930196, "Max": 2104.915562},
        TAPropAbbr.HD: {"Min": 0.8708966542, "Max": 705.7024083},
        TAPropAbbr.HP: {"Min": 0.007003215217, "Max": 1.712580475}
    }

    _unit_map_ = {
        TAPropAbbr.BTT: '℃',
        TAPropAbbr.WF: 'wt%',
        TAPropAbbr.TE: 'K^-1',
        TAPropAbbr.DS: 'g/cm^3',
        TAPropAbbr.TC: 'W/(m·K)',
        TAPropAbbr.EC: 'S/m',
        TAPropAbbr.YM: 'GPa',
        TAPropAbbr.BM: 'GPa',
        TAPropAbbr.SM: 'GPa',
        TAPropAbbr.PR: None,
        TAPropAbbr.SE: 'J/g',
        TAPropAbbr.SHC: 'J/g·K',
        TAPropAbbr.YS: 'MPa',
        TAPropAbbr.TS: 'MPa',
        TAPropAbbr.HD: 'VPN',
        TAPropAbbr.HP: 'MPa·m^(1/2)'
    }

    _sparsemax_ = Sparsemax(dim=-1)

    def __init__(self, device: torch.device | None = None, batch_size: int = 64, silence: bool = False, skip: bool = False):
        """
        A tool class that use TAPP models for inference and can predict the properties of titanium alloys.
        :param device: Specify the Torch device
        :param batch_size: The batch size for batch inference, which is not more than 10,000
        :param silence: Whether to turn on silent mode, silent mode does not print run logs
        :param skip: Whether to skip the empirical formula correction
        """
        self.silence = silence
        self.skip = skip
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        if not 1 < batch_size < _MAX_BATCH_SIZE_:
            raise ValueError('Batch size must be between 2 and 10,000.')
        self.batch_size = batch_size
        # Load model weights
        self._ms_ = {}
        for prop_abbr in TAPropAbbr:
            prop_num = 12 if prop_abbr is TAPropAbbr.WF else 1
            proc_num = 0 if prop_abbr is TAPropAbbr.BTT else 1
            model_path = Path(str(resources.files('tap2.weight').joinpath(f'{prop_abbr.value}.pth')))
            model = MoE2(12, proc_num, prop_num, 512, 0.2).to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            self._ms_[prop_abbr] = model
        # Load the correction parameters
        if not self.skip:
            weight_path = Path(str(resources.files('tap2.weight').joinpath('corr_weights.pth')))
            self._cws_ = torch.load(weight_path, map_location=self.device)

    def _infer_(self, ta_input: TAInput) -> float | List[float]:
        """
        Invoke the TAPP model for single inference to obtain predictive property value.
        :param ta_input: Includes abbreviation of property to be predicted, elemental composition,
                           heat treatment temperature and grain size (single input)
        :return: Property value predicted by the TAPP model (single output)
        """
        with torch.inference_mode():
            model = self._ms_[ta_input.prop]
            if ta_input.prop is TAPropAbbr.BTT:
                input_T = torch.tensor(
                    data=[[ta_input.Ti, ta_input.Al, ta_input.Cr, ta_input.Cu, ta_input.Fe, ta_input.Mo,
                           ta_input.Ni, ta_input.Nb, ta_input.Si, ta_input.Sn, ta_input.V, ta_input.Zr]],
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=False
                ) # Shape: (1, 12)
            else:
                input_T = torch.tensor(
                    data=[[ta_input.Ti, ta_input.Al, ta_input.Cr, ta_input.Cu, ta_input.Fe, ta_input.Mo,
                           ta_input.Ni, ta_input.Nb, ta_input.Si, ta_input.Sn, ta_input.V, ta_input.Zr,
                           ta_input.HTT]],
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=False
                ) # Shape: (1, 13)
            output_T = model(input_T) # Shape: (1, PropNum)
            if ta_input.prop is TAPropAbbr.WF:
                output_T = self._sparsemax_(output_T) # Shape: (1, 12)
                output_T = output_T * 100 # Shape: (1, 12)
                output = output_T.tolist()[0] # Shape: (12,)
            else:
                min_val = self._norm_params_[ta_input.prop]['Min']
                max_val = self._norm_params_[ta_input.prop]['Max']
                output_T = output_T * (max_val - min_val) + min_val # Shape: (1, 1)
                output = output_T.item()
        return output

    def _batch_infer_(self, ta_input: TABatchInput) -> List[float] | List[List[float]]:
        """
        Invoke the TAPP model for batch inference to obtain predictive property values.
        :param ta_input: Includes abbreviation of property to be predicted, lists of elemental composition,
                           heat treatment temperature, grain size (batch input)
        :return: Property values predicted by the TAPP model (batch output)
        """
        input_size = len(ta_input)
        batch_num = ceil(input_size / self.batch_size)
        if ta_input.prop is TAPropAbbr.WF:
            outputs = [[] for _ in range(12)]
        else:
            outputs = []
        with torch.inference_mode():
            model = self._ms_[ta_input.prop]
            for batch_id in range(batch_num):
                begin_idx = batch_id * self.batch_size
                end_idx = min(begin_idx + self.batch_size, input_size)
                if ta_input.prop is TAPropAbbr.BTT:
                    inputs_T = torch.tensor(
                        data=[ta_input.Ti[begin_idx: end_idx], ta_input.Al[begin_idx: end_idx],
                              ta_input.Cr[begin_idx: end_idx], ta_input.Cu[begin_idx: end_idx],
                              ta_input.Fe[begin_idx: end_idx], ta_input.Mo[begin_idx: end_idx],
                              ta_input.Ni[begin_idx: end_idx], ta_input.Nb[begin_idx: end_idx],
                              ta_input.Si[begin_idx: end_idx], ta_input.Sn[begin_idx: end_idx],
                              ta_input.V[begin_idx: end_idx], ta_input.Zr[begin_idx: end_idx]],
                        dtype=torch.float32,
                        device=self.device,
                        requires_grad=False
                    )  # Shape: (12, BatchSize)
                    inputs_T = inputs_T.T  # Shape: (BatchSize, 12)
                else:
                    inputs_T = torch.tensor(
                        data=[ta_input.Ti[begin_idx: end_idx], ta_input.Al[begin_idx: end_idx],
                              ta_input.Cr[begin_idx: end_idx], ta_input.Cu[begin_idx: end_idx],
                              ta_input.Fe[begin_idx: end_idx], ta_input.Mo[begin_idx: end_idx],
                              ta_input.Ni[begin_idx: end_idx], ta_input.Nb[begin_idx: end_idx],
                              ta_input.Si[begin_idx: end_idx], ta_input.Sn[begin_idx: end_idx],
                              ta_input.V[begin_idx: end_idx], ta_input.Zr[begin_idx: end_idx],
                              ta_input.HTT[begin_idx: end_idx]],
                        dtype=torch.float32,
                        device=self.device,
                        requires_grad=False
                    ) # Shape: (13, BatchSize)
                    inputs_T = inputs_T.T # Shape: (BatchSize, 13)
                outputs_T = model(inputs_T) # Shape: (BatchSize, PropNum)
                if ta_input.prop is TAPropAbbr.WF:
                    outputs_T = self._sparsemax_(outputs_T) # Shape: (BatchSize, 12)
                    outputs_T = outputs_T.T  # Shape: (12, BatchSize)
                    outputs_T = outputs_T * 100  # Shape: (12, BatchSize)
                    batch_outputs = outputs_T.tolist()  # Shape: (12, BatchSize)
                    for i, batch_output in enumerate(batch_outputs):
                        outputs[i].extend(batch_output)
                else:
                    min_val = self._norm_params_[ta_input.prop]['Min']
                    max_val = self._norm_params_[ta_input.prop]['Max']
                    outputs_T = outputs_T.squeeze(-1)  # Shape: (BatchSize,)
                    outputs_T = outputs_T * (max_val - min_val) + min_val # Shape: (BatchSize,)
                    batch_outputs = outputs_T.tolist()  # Shape: (BatchSize,)
                    outputs.extend(batch_outputs)

        return outputs

    def _corr_(self, ta_input: TAInput, orig_output: float | List[float]) -> float | List[float]:
        """
        The prediction result of the TAPP model are modified using empirical formulas.
        :param ta_input: Includes abbreviation of property to be predicted, elemental composition,
                           heat treatment temperature and grain size (single input)
        :param orig_output: Original property value predicted by the TAPP model (single output)
        :return: Corrected property value (single output)
        """
        correct = deepcopy(orig_output)
        if ta_input.prop in (TAPropAbbr.YS, TAPropAbbr.TS):
            with torch.inference_mode():
                micro_elem_T = torch.tensor(
                    data=[ta_input.N, ta_input.O, ta_input.C, ta_input.H, ta_input.B],
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=False
                ) # Shape: (5,)
                increment = torch.sum(self._cws_['TS_YS'] * micro_elem_T)
                if ta_input.prop is TAPropAbbr.TS:
                    increment *= 1.1
                increment = increment.item()
            grain_size = ta_input.GS * 1e-6
            hall_petch = self._infer_(ta_input.model_copy(update={'prop': TAPropAbbr.HP, 'GS': 10}, deep=True))
            increment += hall_petch * (pow(grain_size, -0.5) - pow(1e-5, -0.5))
            correct += increment
        elif ta_input.prop in (TAPropAbbr.DS, TAPropAbbr.TC, TAPropAbbr.EC, TAPropAbbr.YM,
                               TAPropAbbr.BM, TAPropAbbr.SM, TAPropAbbr.PR):
            with torch.inference_mode():
                micro_elem_T = torch.tensor(
                    data=[[ta_input.C, ta_input.N, ta_input.O, ta_input.B, ta_input.H]],
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=False
                ) # Shape: (1, 5)
                main_elem_T = torch.tensor(
                    data=[[ta_input.Al], [ta_input.Cr], [ta_input.Cu], [ta_input.Fe], [ta_input.Mo],
                          [ta_input.Nb], [ta_input.Ni], [ta_input.Si], [ta_input.Sn]],
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=False
                ) # Shape: (9, 1)
                increment = (micro_elem_T @ self._cws_[ta_input.prop.value] @ main_elem_T)
                increment = increment.item()
            correct += increment
        elif ta_input.prop is TAPropAbbr.TE:
            with torch.inference_mode():
                micro_elem_T = torch.tensor(
                    data=[[1, 1, 1, 1, 1],
                          [ta_input.N, ta_input.O, ta_input.C, ta_input.H, ta_input.B],
                          [ta_input.N ** 2, ta_input.O ** 2, ta_input.C ** 2, ta_input.H ** 2, ta_input.B ** 2],
                          [ta_input.N ** 3, ta_input.O ** 3, ta_input.C ** 3, ta_input.H ** 3, ta_input.B ** 3]],
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=False
                ) # Shape: (4, 5)
                increment = torch.sum(micro_elem_T * self._cws_['TE'])
                increment = increment.item()
            correct += increment
        return correct

    def _batch_corr_(self, ta_input: TABatchInput, orig_output: List[float] | List[List[float]]) -> List[float] | List[List[float]]:
        """
        The prediction results of the TAPP model are modified using empirical formulas.
        :param ta_input: Includes abbreviation of property to be predicted, lists of elemental composition,
                           heat treatment temperature and grain size (batch input)
        :param orig_output: Original property values predicted by the TAPP model (batch output)
        :return: Corrected property values (batch output)
        """
        input_size = len(ta_input)
        batch_num = ceil(input_size / self.batch_size)
        corrects = deepcopy(orig_output)
        if ta_input.prop in (TAPropAbbr.YS, TAPropAbbr.TS):
            increments = []
            with torch.inference_mode():
                hall_petch_T = torch.tensor(
                    data=self._batch_infer_(ta_input.model_copy(update={"prop": TAPropAbbr.HP,
                                                                          "GS": [10] * input_size})),
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=False
                ) # Shape: (InputSize,)
                for batch_id in range(batch_num):
                    begin_idx = batch_id * self.batch_size
                    end_idx = min(begin_idx + self.batch_size, input_size)
                    micro_elem_T = torch.tensor(
                        data=[ta_input.N[begin_idx: end_idx], ta_input.O[begin_idx: end_idx],
                              ta_input.C[begin_idx: end_idx], ta_input.H[begin_idx: end_idx],
                              ta_input.B[begin_idx: end_idx]],
                        dtype=torch.float32,
                        device=self.device,
                        requires_grad=False
                    ) # Shape: (5, BatchSize)
                    micro_elem_T = micro_elem_T.T # Shape: (BatchSize, 5)
                    increments_T = torch.sum(self._cws_['TS_YS'] * micro_elem_T, dim=1) # Shape: (BatchSize,)
                    if ta_input.prop is TAPropAbbr.TS:
                        increments_T *= 1.1
                    grain_size_T = torch.tensor(
                        data=ta_input.GS[begin_idx: end_idx],
                        dtype=torch.float32,
                        device=self.device,
                        requires_grad=False
                    ) # Shape: (BatchSize,)
                    grain_size_T *= 1e-6
                    increments_T += hall_petch_T[begin_idx: end_idx] * (grain_size_T ** -0.5 - 1e-5 ** -0.5) # Shape: (BatchSize,)
                    batch_increments = increments_T.tolist()  # Shape: (BatchSize,)
                    increments.extend(batch_increments)
            for idx, increment in enumerate(increments):
                corrects[idx] += increment
        elif ta_input.prop in (TAPropAbbr.DS, TAPropAbbr.TC, TAPropAbbr.EC, TAPropAbbr.YM,
                               TAPropAbbr.BM, TAPropAbbr.SM, TAPropAbbr.PR):
            increments = []
            with torch.inference_mode():
                for batch_id in range(batch_num):
                    begin_idx = batch_id * self.batch_size
                    end_idx = min(begin_idx + self.batch_size, input_size)
                    micro_elem_T = torch.tensor(
                        data=[ta_input.C[begin_idx: end_idx], ta_input.N[begin_idx: end_idx],
                              ta_input.O[begin_idx: end_idx], ta_input.B[begin_idx: end_idx],
                              ta_input.H[begin_idx: end_idx]],
                        dtype=torch.float32,
                        device=self.device,
                        requires_grad=False
                    ) # Shape: (5, BatchSize)
                    micro_elem_T = micro_elem_T.T.unsqueeze(1) # Shape: (BatchSize, 1, 5)
                    main_elem_T = torch.tensor(
                        data=[ta_input.Al[begin_idx: end_idx], ta_input.Cr[begin_idx: end_idx],
                              ta_input.Cu[begin_idx: end_idx], ta_input.Fe[begin_idx: end_idx],
                              ta_input.Mo[begin_idx: end_idx], ta_input.Nb[begin_idx: end_idx],
                              ta_input.Ni[begin_idx: end_idx], ta_input.Si[begin_idx: end_idx],
                              ta_input.Sn[begin_idx: end_idx]],
                        dtype=torch.float32,
                        device=self.device,
                        requires_grad=False
                    ) # Shape: (9, BatchSize)
                    main_elem_T = main_elem_T.T.unsqueeze(-1) # Shape: (BatchSize, 9, 1)
                    increments_T = micro_elem_T @ self._cws_[ta_input.prop.value] @ main_elem_T  # Shape: (BatchSize, 1, 1)
                    increments_T = increments_T.squeeze(-1).squeeze(-1)  # Shape: (BatchSize,)
                    batch_increments = increments_T.tolist()
                    increments.extend(batch_increments)
            for idx, increment in enumerate(increments):
                corrects[idx] += increment
        elif ta_input.prop is TAPropAbbr.TE:
            increments = []
            with torch.inference_mode():
                for batch_id in range(batch_num):
                    begin_idx = batch_id * self.batch_size
                    end_idx = min(begin_idx + self.batch_size, input_size)
                    micro_elem_T = torch.tensor(
                        data=[ta_input.N[begin_idx: end_idx], ta_input.O[begin_idx: end_idx],
                              ta_input.C[begin_idx: end_idx], ta_input.H[begin_idx: end_idx],
                              ta_input.B[begin_idx: end_idx]],
                        dtype=torch.float32,
                        device=self.device
                    ) # Shape: (5, BatchSize)
                    micro_elem_T = micro_elem_T.T  # Shape: (BatchSize, 5)
                    micro_elem_T = torch.stack([torch.ones_like(micro_elem_T), micro_elem_T,
                                                micro_elem_T ** 2, micro_elem_T ** 3], dim=1) # Shape: (BatchSize, 4, 5)
                    increments_T = torch.sum(micro_elem_T * self._cws_['TE'], dim=(1, 2)) # Shape: (BatchSize,)
                    batch_increments = increments_T.tolist()
                    increments.extend(batch_increments)
            for idx, increment in enumerate(increments):
                corrects[idx] += increment
        return corrects

    def __call__(self, ta_input: TAInput | TABatchInput) -> TAOutput | TABatchOutput:
        """
        Invoke the TAPP model for inference, predict the properties of titanium alloys, and support batch processing.
        :param ta_input: Includes abbreviation of property to be predicted, elemental composition,
                           heat treatment temperature and grain size (single input or batch input)
        :return: Property values predicted by the TAPP model (single output or batch output)
        """
        if isinstance(ta_input, TAInput):
            logger.info(f'Input: {ta_input}')
            infer_func = self._infer_
            corr_func = self._corr_
            output_type = TAOutput
            phase_ratio_type = TAPhaseRatio
        else:
            batch_info = json.dumps({'prop': ta_input.prop.value, 'size': len(ta_input)}, ensure_ascii=False)
            logger.info(f'Input: {batch_info}]')
            infer_func = self._batch_infer_
            corr_func = self._batch_corr_
            output_type = TABatchOutput
            phase_ratio_type = TABatchPhaseRatio
        value = infer_func(ta_input)
        if not self.skip:
            value = corr_func(ta_input, value)
        if ta_input.prop is TAPropAbbr.WF:
            ta_output = output_type(
                prop=ta_input.prop,
                value=phase_ratio_type(
                    ALPHA=value[0],
                    BETA=value[1],
                    LAVES=value[2],
                    TI3AL=value[3],
                    TI2CU=value[4],
                    TI5SI3=value[5],
                    TIZRSI=value[6],
                    TI2NI=value[7],
                    TIM_B2=value[8],
                    LIQUID=value[9],
                    C15_FCC=value[10],
                    MC=value[11]
                ),
                unit=self._unit_map_[ta_input.prop]
            )
        else:
            ta_output = output_type(prop=ta_input.prop, value=value, unit=self._unit_map_[ta_input.prop])
        if isinstance(ta_output, TAOutput):
            logger.info(f'Output: {ta_output}')
        else:
            batch_info = json.dumps({'prop': ta_input.prop.value, 'size': len(ta_output)}, ensure_ascii=False)
            logger.info(f'Output: {batch_info}')
        return ta_output


class AAPropAbbr(Enum):
    """
    Abbreviation for aluminum alloy property
    """
    TE = 'TE'
    TC = 'TC'


class AAInput(BaseModel):
    """
    The input data structure of the AAInfer class (single)
    contains the property abbreviation, elemental composition and heat treatment temperature.
    """
    prop: AAPropAbbr = Field(..., description='铝合金性能代码：热膨胀系数（TE）、热导率（TC）', examples=['TE'])
    Al: float = Field(default=0, description='铝（Al）的质量分数', examples=[88], ge=0, le=100)
    Li: float = Field(default=0, description='锂（Li）的质量分数', examples=[0], ge=0, le=100)
    Mg: float = Field(default=0, description='镁（Mg）的质量分数', examples=[0], ge=0, le=100)
    Si: float = Field(default=0, description='硅（Si）的质量分数', examples=[12], ge=0, le=100)
    Ca: float = Field(default=0, description='钙（Ca）的质量分数', examples=[0], ge=0, le=100)
    Sc: float = Field(default=0, description='钪（Sc）的质量分数', examples=[0], ge=0, le=100)
    Ti: float = Field(default=0, description='钛（Ti）的质量分数', examples=[0], ge=0, le=100)
    V: float = Field(default=0, description='钒（V）的质量分数', examples=[0], ge=0, le=100)
    Cr: float = Field(default=0, description='铬（Cr）的质量分数', examples=[0], ge=0, le=100)
    Mn: float = Field(default=0, description='锰（Mn）的质量分数', examples=[0], ge=0, le=100)
    Fe: float = Field(default=0, description='铁（Fe）的质量分数', examples=[0], ge=0, le=100)
    Co: float = Field(default=0, description='钴（Co）的质量分数', examples=[0], ge=0, le=100)
    Ni: float = Field(default=0, description='镍（Ni）的质量分数', examples=[0], ge=0, le=100)
    Cu: float = Field(default=0, description='铜（Cu）的质量分数', examples=[0], ge=0, le=100)
    Zn: float = Field(default=0, description='锌（Zn）的质量分数', examples=[0], ge=0, le=100)
    Zr: float = Field(default=0, description='镐（Zr）的质量分数', examples=[0], ge=0, le=100)
    Sn: float = Field(default=0, description='锡（Sn）的质量分数', examples=[0], ge=0, le=100)
    La: float = Field(default=0, description='镧（La）的质量分数', examples=[0], ge=0, le=100)
    HTT: float = Field(..., description='热处理温度（℃）', examples=[500], gt=-273.15)

    @model_validator(mode="after")
    def _valid_compos_(self) -> Any:
        """
        Verify that the sum of the elemental composition of the aluminum alloy is 100 wt%.
        :return: self
        """
        total = sum([self.Al, self.Li, self.Mg, self.Si, self.Ca, self.Sc, self.Ti, self.V, self.Cr, self.Mn, self.Fe,
                     self.Co, self.Ni, self.Cu, self.Zn, self.Zr, self.Sn, self.La])
        if not isclose(total, 100, abs_tol=1e-6):
            raise ValueError(f'The sum of all element compositions must be 100, but got {total}.')
        return self

    def __str__(self) -> str:
        """
        Converts an AAInput instance to a string representation that meets the JSON specification.
        :return: The string representation of AAInput instance
        """
        return json.dumps({
            'prop': self.prop.value,
            'Al': f'{self.Al:f}'.rstrip('0').rstrip('.'),
            'Li': f'{self.Li:f}'.rstrip('0').rstrip('.'),
            'Mg': f'{self.Mg:f}'.rstrip('0').rstrip('.'),
            'Si': f'{self.Si:f}'.rstrip('0').rstrip('.'),
            'Ca': f'{self.Ca:f}'.rstrip('0').rstrip('.'),
            'Sc': f'{self.Sc:f}'.rstrip('0').rstrip('.'),
            'Ti': f'{self.Ti:f}'.rstrip('0').rstrip('.'),
            'V': f'{self.V:f}'.rstrip('0').rstrip('.'),
            'Cr': f'{self.Cr:f}'.rstrip('0').rstrip('.'),
            'Mn': f'{self.Mn:f}'.rstrip('0').rstrip('.'),
            'Fe': f'{self.Fe:f}'.rstrip('0').rstrip('.'),
            'Co': f'{self.Co:f}'.rstrip('0').rstrip('.'),
            'Ni': f'{self.Ni:f}'.rstrip('0').rstrip('.'),
            'Cu': f'{self.Cu:f}'.rstrip('0').rstrip('.'),
            'Zn': f'{self.Zn:f}'.rstrip('0').rstrip('.'),
            'Zr': f'{self.Zr:f}'.rstrip('0').rstrip('.'),
            'Sn': f'{self.Sn:f}'.rstrip('0').rstrip('.'),
            'La': f'{self.La:f}'.rstrip('0').rstrip('.'),
            'HTT': f'{self.HTT:f}'.rstrip('0').rstrip('.')
        }, ensure_ascii=False)

    @property
    def sha256(self) -> str:
        """
        Calculate the SHA256 hash value of the AAInput instance.
        :return: SHA256 hash value
        """
        return sha256(str(self).encode('utf-8')).hexdigest()


class AAOutput(BaseModel):
    """
    The output data structure of the AAInfer class (single)
    contains the property abbreviation, predicted value and unit.
    """
    prop: AAPropAbbr = Field(..., description='铝合金性能代码：热膨胀系数（TE）、热导率（TC）', examples=['TE'])
    value: float = Field(..., description='铝合金性能值，均为浮点数类型', examples=[22])
    unit: str = Field(..., description='铝合金性能单位', examples=['10^-6/K'])

    def __str__(self) -> str:
        """
        Converts AAOutput instances to string representation that meets the JSON specification.
        :return: The string representation of AAOutput instance
        """
        return json.dumps({
            'prop': self.prop.value,
            'value': f'{self.value:f}'.rstrip('0').rstrip('.'),
            'unit': self.unit
        }, ensure_ascii=False)


class AAInfer:
    _unit_map_ = {
        AAPropAbbr.TE: "K^-1",
        AAPropAbbr.TC: "W/m·K"
    }

    def __init__(self, device: torch.device | None = None, silence: bool = False):
        """
        A tool class that use AAPP models for inference and can predict the properties of aluminum alloys.
        :param device: Specify the Torch device
        :param silence: Whether to turn on silent mode, silent mode does not print run logs
        """
        # Initialize the parameters
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.silence = silence
        # Load the model weights
        self.models = {}
        for prop_abbr in AAPropAbbr:
            weight_path = Path(str(resources.files("tap2.weight").joinpath(f"AA_{prop_abbr.value}.pth")))
            model = AAModel(input_dim=19, hidden_dim=450, output_dim=1, dropout_rate=0.2).to(self.device)
            model.load_state_dict(torch.load(weight_path, map_location=self.device))
            model.eval()
            self.models[prop_abbr] = model
        # Initialize the normalization parameters
        self.mins = torch.tensor(data=[72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 400,
                                       28.843647, 18.968151],
                                  dtype=torch.float32,
                                  device=self.device,
                                  requires_grad=False)
        self.maxs = torch.tensor(data=[99.8, 5, 12, 13, 2, 5, 2, 8, 7, 3, 3, 5, 10, 10, 12, 3, 1, 15, 700,
                                       236.329194, 34.945403],
                                 dtype=torch.float32,
                                 device=self.device,
                                 requires_grad=False)

    def _infer_(self, aa_input: AAInput) -> float:
        """
        Invoke the AAPP model for single inference to obtain predictive property value.
        :param aa_input: Includes abbreviation of property to be predicted, elemental composition,
                         heat treatment temperature and grain size (single input)
        :return: Property value predicted by the AAPP model (single output)
        """
        model = self.models[aa_input.prop]
        with torch.inference_mode():
            input_T = torch.tensor(
                data=[[aa_input.Al, aa_input.Li, aa_input.Mg, aa_input.Si, aa_input.Ca, aa_input.Sc, aa_input.Ti,
                       aa_input.V, aa_input.Cr, aa_input.Mn, aa_input.Fe, aa_input.Co, aa_input.Ni, aa_input.Cu,
                       aa_input.Zn, aa_input.Zr, aa_input.Sn, aa_input.La, aa_input.HTT]],
                dtype=torch.float32,
                device=self.device,
                requires_grad=False
            )  # Shape: (1, 19)
            # 对输入进行归一化
            input_T = (input_T - self.mins[:19]) / (self.maxs[:19] - self.mins[:19])  # Shape: (1, 19)
            output_T = model(input_T)  # Shape: (1, 1)
            # 对输出进行反归一化
            match aa_input.prop:
                case AAPropAbbr.TC:
                    output_T = output_T * (self.maxs[19] - self.mins[19]) + self.mins[19]  # Shape: (1, 1)
                case AAPropAbbr.TE:
                    output_T = output_T * (self.maxs[20] - self.mins[20]) + self.mins[20]  # Shape: (1, 1)
                case _:
                    raise ValueError(f'Unsupported property: {aa_input.prop}')
            output_value = output_T.item()
        return output_value

    def __call__(self, aa_input: AAInput) -> AAOutput:
        if not self.silence:
            logger.info(f'Input: {aa_input}')
        value = self._infer_(aa_input)
        aa_output = AAOutput(
            prop=aa_input.prop,
            value=value,
            unit=self._unit_map_[aa_input.prop]
        )
        if not self.silence:
            logger.info(f'Output: {aa_output}')
        return aa_output

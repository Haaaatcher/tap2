from starlette.middleware.cors import CORSMiddleware
from tap2.model import MoE2, AAModel, TAP2Model
from pathlib import Path
from importlib import resources
from typing import Any
from pydantic import BaseModel, Field, model_validator
from sparsemax import Sparsemax
from math import isclose
from litserve import LitAPI, LitServer
import torch
import json
import click



class TAPhysBaseModel(BaseModel):
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
    HTT: float = Field(default=600, description='热处理温度（℃）', examples=[600], gt=-273.15)

    @model_validator(mode='after')
    def _valid_compos_(self) -> Any:
        total = sum([self.Ti, self.H, self.B, self.C, self.N, self.O, self.Al, self.Si, self.Cr,
                     self.Fe, self.Ni, self.Cu, self.Zr, self.Nb, self.Mo, self.V, self.Sn])
        if not isclose(total, 100, abs_tol=1e-6):
            raise ValueError(f'The sum of all element compositions must be 100, but got {total}.')
        return self

    def __str__(self) -> str:
        return json.dumps({
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
            'HTT': f"{self.HTT:f}".rstrip('0').rstrip('.')
        }, ensure_ascii=False, indent=4)


class TAMechBaseModel(BaseModel):
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
    HTT: float = Field(default=600, description='热处理温度（℃）', examples=[600], gt=-273.15)
    GS: float = Field(default=10, description='晶粒尺寸（μm）', examples=[10], gt=0)

    @model_validator(mode='after')
    def _valid_compos_(self) -> Any:
        total = sum([self.Ti, self.H, self.B, self.C, self.N, self.O, self.Al, self.Si, self.Cr,
                     self.Fe, self.Ni, self.Cu, self.Zr, self.Nb, self.Mo, self.V, self.Sn])
        if not isclose(total, 100, abs_tol=1e-6):
            raise ValueError(f'The sum of all element compositions must be 100, but got {total}.')
        return self

    def __str__(self) -> str:
        return json.dumps({
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
            'HTT': f"{self.HTT:f}".rstrip('0').rstrip('.'),
            'GS': f"{self.GS:f}".rstrip('0').rstrip('.')
        }, ensure_ascii=False, indent=4)


class TAWFBaseModel(BaseModel):
    Ti: float = Field(default=0, description='钛（Ti）的质量分数', examples=[90], ge=0, le=100)
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
    HTT: float = Field(default=600, description='热处理温度（℃）', examples=[600], gt=-273.15)

    @model_validator(mode='after')
    def _valid_compos_(self) -> Any:
        total = sum([self.Ti, self.Al, self.Si, self.Cr, self.Fe, self.Ni,
                     self.Cu, self.Zr, self.Nb, self.Mo, self.V, self.Sn])
        if not isclose(total, 100, abs_tol=1e-6):
            raise ValueError(f'The sum of all element compositions must be 100, but got {total}.')
        return self

    def __str__(self) -> str:
        return json.dumps({
            'Ti': f'{self.Ti:f}'.rstrip('0').rstrip('.'),
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
            'HTT': f"{self.HTT:f}".rstrip('0').rstrip('.')
        }, ensure_ascii=False, indent=4)


class TADSLitAPI(LitAPI):
    ta_ds_model_path: Path = None
    ta_ds_model: TAP2Model = None
    ta_ds_mc_weight: torch.Tensor = None

    def setup(self, device):
        self.ta_ds_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('TA_DS_260409.pth')))
        self.ta_ds_model = TAP2Model(12, 1, 256, 0.2).eval().to(device)
        self.ta_ds_model.load_state_dict(torch.load(self.ta_ds_model_path, map_location=device))
        self.ta_ds_mc_weight = torch.tensor([
            [0.001189364, -0.033229803, 0.006900622, 0.001364253, 0.001293301,
             0.001219326, 0.001929503, 0.001278101, 0.001198975],
            [0.003211933, -0.169015533, 2.381598976, 7.424964639, 0.008192758,
             0.014823883, 0.018094439, 0.014480812, 0.010925323],
            [0.01812761, 0.021041677, -0.266758591, -1.897012602, 0.026452594,
             0.026654634, 0.010732661, -5.104111794, 0.025470266],
            [0.022893337, 0.046100565, 0.091790094, 0.04476362, 0.046677253,
             0.046287457, 0.053434922, 0.045513762, -0.039840009],
            [-4.922743468, -4.406682541, 1.499480452, 15.85136756, -4.284999468,
             -4.414201642, -4.482102982, 45.9676752, -4.531997149]
        ], dtype=torch.float32, device=device)

    def decode_request(self, request: TAPhysBaseModel, **kwargs):
        model_inputs = torch.tensor([
            request.Ti / 100.0, request.Al / 100.0, request.Si / 100.0, request.Cr / 100.0,
            request.Fe / 100.0, request.Ni / 100.0, request.Cu / 100.0, request.Zr / 100.0,
            request.Nb / 100.0, request.Mo / 100.0, request.V / 100.0, request.Sn / 100.0,
            (request.HTT - 200.0) / 1100.0
        ], dtype=torch.float32, device=self.device)
        micro_elements = torch.tensor([[
            request.C, request.N, request.O, request.B, request.H,
        ]], dtype=torch.float32, device=self.device)
        major_elements = torch.tensor([
            [request.Al], [request.Cr], [request.Cu], [request.Fe], [request.Mo],
            [request.Nb], [request.Ni], [request.Si], [request.Sn]
        ], dtype=torch.float32, device=self.device)
        return {
            'model_inputs': model_inputs,
            'micro_elements': micro_elements,
            'major_elements': major_elements
        }

    def batch(self, inputs):
        return {
            'model_inputs': torch.stack([item['model_inputs'] for item in inputs]),
            'micro_elements': torch.stack([item['micro_elements'] for item in inputs]),
            'major_elements': torch.stack([item['major_elements'] for item in inputs]),
        }

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.ta_ds_model(x['model_inputs'])
            outputs = outputs.squeeze(-1)
            outputs = outputs * 6.676304 + 3.672575
            mc_increments = x['micro_elements'] @ self.ta_ds_mc_weight @ x['major_elements']
            mc_increments = mc_increments.squeeze(-1).squeeze(-1)
            outputs = outputs + mc_increments
            outputs = outputs.tolist()
            return outputs

    def encode_response(self, output, **kwargs):
        return {
            'prop': 'Density',
            'value': output,
            'unit': 'g/cm^3'
        }


class TATCLitAPI(LitAPI):
    ta_tc_model_path: Path = None
    ta_tc_model: TAP2Model = None
    ta_tc_mc_weight: torch.Tensor = None

    def setup(self, device):
        self.ta_tc_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('TA_TC_260409.pth')))
        self.ta_tc_model = TAP2Model(12, 1, 256, 0.2).eval().to(device)
        self.ta_tc_model.load_state_dict(torch.load(self.ta_tc_model_path, map_location=device))
        self.ta_tc_mc_weight = torch.tensor([
            [0.011143497, 3.489852516, 0.050639097, 0.045199609, -0.006363126,
             -0.146271765, -0.036777042, -0.032206222, -0.035560762],
            [0.015501867, 17.7350841, 121.4258822, 740.5704625, -42.09685675,
             -31.53882272, -0.157892947, 0.669169059, -0.220360531],
            [-7.111188519, -13.84984689, -13.80683762, -81.18458214, -17.8742612,
             -16.63168331, -8.830106909, -19.40806548, -7.758796868],
            [-0.079181297, -0.388880921, 0.282376119, -0.019358283, -0.138847193,
             -0.330954121, -0.429841711, -0.174089174, -0.310205874],
            [174.2074784, 127.8573, 136.432501, 752.5723302, 446.2291183,
             495.3979761, 25.63202273, 153.3714961, 9.850931867]
        ], dtype=torch.float32, device=device)

    def decode_request(self, request: TAPhysBaseModel, **kwargs):
        model_inputs = torch.tensor([
            request.Ti / 100.0, request.Al / 100.0, request.Si / 100.0, request.Cr / 100.0,
            request.Fe / 100.0, request.Ni / 100.0, request.Cu / 100.0, request.Zr / 100.0,
            request.Nb / 100.0, request.Mo / 100.0, request.V / 100.0, request.Sn / 100.0,
            (request.HTT - 200.0) / 1100.0
        ], dtype=torch.float32, device=self.device)
        micro_elements = torch.tensor([[
            request.C, request.N, request.O, request.B, request.H,
        ]], dtype=torch.float32, device=self.device)
        major_elements = torch.tensor([
            [request.Al], [request.Cr], [request.Cu], [request.Fe], [request.Mo],
            [request.Nb], [request.Ni], [request.Si], [request.Sn]
        ], dtype=torch.float32, device=self.device)
        return {
            'model_inputs': model_inputs,
            'micro_elements': micro_elements,
            'major_elements': major_elements
        }

    def batch(self, inputs):
        return {
            'model_inputs': torch.stack([item['model_inputs'] for item in inputs]),
            'micro_elements': torch.stack([item['micro_elements'] for item in inputs]),
            'major_elements': torch.stack([item['major_elements'] for item in inputs]),
        }

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.ta_tc_model(x['model_inputs'])
            outputs = outputs.squeeze(-1)
            outputs = outputs * 996.830807 + 0.000223
            mc_increments = x['micro_elements'] @ self.ta_tc_mc_weight @ x['major_elements']
            mc_increments = mc_increments.squeeze(-1).squeeze(-1)
            outputs = outputs + mc_increments
            outputs = outputs.tolist()
            return outputs

    def encode_response(self, output, **kwargs):
        return {
            'prop': 'Thermal Conductivity',
            'value': output,
            'unit': 'W/m·K'
        }


class TAECLitAPI(LitAPI):
    ta_ec_model_path: Path = None
    ta_ec_model: TAP2Model = None
    ta_ec_mc_weight: torch.Tensor = None

    def setup(self, device):
        self.ta_ec_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('TA_EC_260409.pth')))
        self.ta_ec_model = TAP2Model(12, 1, 256, 0.2).eval().to(device)
        self.ta_ec_model.load_state_dict(torch.load(self.ta_ec_model_path, map_location=device))
        self.ta_ec_mc_weight = torch.tensor([
            [427.4222959, -1160.893555, 1980.111753, -1703.05578, -355.9538164,
             -5426.046231, -1222.705087, -1150.778254, -1877.413688],
            [643.5254332, 5716.60143, -81371.57587, 25064.0197, 8744.494462,
             104164.2629, -8668.288187, 49621.93846, -12487.60801],
            [-354773.0809, -356747.4198, -343208.1088, -242411.9374, -329665.1045,
             -391633.5205, -409248.1142, -364784.2755, -387130.4589],
            [-18421.64249, -22875.26649, 13480.95763, -15435.74515, -7469.583248,
             -18476.56392, -25040.42194, -8473.574668, -22231.51572],
            [454916.494, 900699.6168, 767781.0268, 368418.7209, 855436.335,
             583596.1172, 649707.8811, 263225.8889, 558257.5577]
        ], dtype=torch.float32, device=device)

    def decode_request(self, request: TAPhysBaseModel, **kwargs):
        model_inputs = torch.tensor([
            request.Ti / 100.0, request.Al / 100.0, request.Si / 100.0, request.Cr / 100.0,
            request.Fe / 100.0, request.Ni / 100.0, request.Cu / 100.0, request.Zr / 100.0,
            request.Nb / 100.0, request.Mo / 100.0, request.V / 100.0, request.Sn / 100.0,
            (request.HTT - 200.0) / 1100.0
        ], dtype=torch.float32, device=self.device)
        micro_elements = torch.tensor([[
            request.C, request.N, request.O, request.B, request.H,
        ]], dtype=torch.float32, device=self.device)
        major_elements = torch.tensor([
            [request.Al], [request.Cr], [request.Cu], [request.Fe], [request.Mo],
            [request.Nb], [request.Ni], [request.Si], [request.Sn]
        ], dtype=torch.float32, device=self.device)
        return {
            'model_inputs': model_inputs,
            'micro_elements': micro_elements,
            'major_elements': major_elements
        }

    def batch(self, inputs):
        return {
            'model_inputs': torch.stack([item['model_inputs'] for item in inputs]),
            'micro_elements': torch.stack([item['micro_elements'] for item in inputs]),
            'major_elements': torch.stack([item['major_elements'] for item in inputs]),
        }

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.ta_ec_model(x['model_inputs'])
            outputs = outputs.squeeze(-1)
            outputs = outputs * 6.101695 + 0.000258
            mc_increments = x['micro_elements'] @ self.ta_ec_mc_weight @ x['major_elements'] * 1e-6
            mc_increments = mc_increments.squeeze(-1).squeeze(-1)
            outputs = outputs + mc_increments
            outputs = outputs.tolist()
            return outputs

    def encode_response(self, output, **keyword):
        return {
            'prop': 'Electrical Conductivity',
            'value': output,
            'unit': '10^6 S/m'
        }


class TAYMLitAPI(LitAPI):
    ta_ym_model_path: Path = None
    ta_ym_model: TAP2Model = None
    ta_ym_mc_weight: torch.Tensor = None

    def setup(self, device):
        self.ta_ym_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('TA_YM_260409.pth')))
        self.ta_ym_model = TAP2Model(12, 1, 256, 0.2).eval().to(device)
        self.ta_ym_model.load_state_dict(torch.load(self.ta_ym_model_path, map_location=device))
        self.ta_ym_mc_weight = torch.tensor([
            [2.687507505, 6.928988486, 2.760425827, 2.882769734, 2.856143054,
             2.851864857, 2.894755453, 2.802196795, 2.409429476],
            [4.494867834, 25.77829052, -71.10355102, 569.8110626, -86.32504336,
             3.873705269, 18.0498347, 17.3860172, 17.07487544],
            [6.683187786, 3.724882158, 8.733050146, -55.0048898, -19.64440386,
             3.999887499, 7.763206322, -6.288934614, 7.191511269],
            [2.396641297, -0.075441769, -5.220427127, -1.981547993, -3.917721649,
             -4.955526404, -5.294481326, -4.770403738, -2.979347874],
            [-92.63189837, 175.7924499, 83.62668492, 717.1182946, 957.9610359,
             228.6389485, 111.8343005, 251.9310226, 108.5756246]
        ], dtype=torch.float32, device=device)

    def decode_request(self, request: TAPhysBaseModel, **kwargs):
        model_inputs = torch.tensor([
            request.Ti / 100.0, request.Al / 100.0, request.Si / 100.0, request.Cr / 100.0,
            request.Fe / 100.0, request.Ni / 100.0, request.Cu / 100.0, request.Zr / 100.0,
            request.Nb / 100.0, request.Mo / 100.0, request.V / 100.0, request.Sn / 100.0,
            (request.HTT - 200.0) / 1100.0
        ], dtype=torch.float32, device=self.device)
        micro_elements = torch.tensor([[
            request.C, request.N, request.O, request.B, request.H,
        ]], dtype=torch.float32, device=self.device)
        major_elements = torch.tensor([
            [request.Al], [request.Cr], [request.Cu], [request.Fe], [request.Mo],
            [request.Nb], [request.Ni], [request.Si], [request.Sn]
        ], dtype=torch.float32, device=self.device)
        return {
            'model_inputs': model_inputs,
            'micro_elements': micro_elements,
            'major_elements': major_elements
        }

    def batch(self, inputs):
        return {
            'model_inputs': torch.stack([item['model_inputs'] for item in inputs]),
            'micro_elements': torch.stack([item['micro_elements'] for item in inputs]),
            'major_elements': torch.stack([item['major_elements'] for item in inputs]),
        }

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.ta_ym_model(x['model_inputs'])
            outputs = outputs.squeeze(-1)
            outputs = outputs * 273.60016 + 1e-6
            mc_increments = x['micro_elements'] @ self.ta_ym_mc_weight @ x['major_elements']
            mc_increments = mc_increments.squeeze(-1).squeeze(-1)
            outputs = outputs + mc_increments
            outputs = outputs.tolist()
            return outputs

    def encode_response(self, output, **kwargs):
        return {
            'prop': 'Young\'s Modulus',
            'value': output,
            'unit': 'GPa'
        }


class TABMLitAPI(LitAPI):
    ta_bm_model_path: Path = None
    ta_bm_model: TAP2Model = None
    ta_bm_mc_weight: torch.Tensor = None

    def setup(self, device):
        self.ta_bm_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('TA_BM_260409.pth')))
        self.ta_bm_model = TAP2Model(12, 1, 256, 0.2).eval().to(device)
        self.ta_bm_model.load_state_dict(torch.load(self.ta_bm_model_path, map_location=device))
        self.ta_bm_mc_weight = torch.tensor([
            [1.91143643, 4.782574079, 2.011642041, 2.116353667, 2.107925979,
             2.074354268, 2.120185848, 2.053483553, 1.75353913],
            [3.209269193, 17.46569617, -61.61676964, 228.7679945, -67.88536,
             -0.822162564, 13.29738451, 13.00498378, 12.55234131],
            [-4.87225487, -7.610587176, -2.889906738, -30.13159105, -24.46797818,
             -7.585980355, -4.258953485, -12.81924464, -4.258542434],
            [-2.946153806, -12.61637001, -12.91788374, -10.19348451, -11.42129274,
             -12.41581323, -13.18642973, -12.09212666, -9.908171743],
            [85.02240782, 330.0399644, 242.651319, 450.5155243, 930.2098776,
             410.4772374, 277.1712212, 360.7586372, 266.617343]
        ], dtype=torch.float32, device=device)

    def decode_request(self, request: TAPhysBaseModel, **kwargs):
        model_inputs = torch.tensor([
            request.Ti / 100.0, request.Al / 100.0, request.Si / 100.0, request.Cr / 100.0,
            request.Fe / 100.0, request.Ni / 100.0, request.Cu / 100.0, request.Zr / 100.0,
            request.Nb / 100.0, request.Mo / 100.0, request.V / 100.0, request.Sn / 100.0,
            (request.HTT - 200.0) / 1100.0
        ], dtype=torch.float32, device=self.device)
        micro_elements = torch.tensor([[
            request.C, request.N, request.O, request.B, request.H,
        ]], dtype=torch.float32, device=self.device)
        major_elements = torch.tensor([
            [request.Al], [request.Cr], [request.Cu], [request.Fe], [request.Mo],
            [request.Nb], [request.Ni], [request.Si], [request.Sn]
        ], dtype=torch.float32, device=self.device)
        return {
            'model_inputs': model_inputs,
            'micro_elements': micro_elements,
            'major_elements': major_elements
        }

    def batch(self, inputs):
        return {
            'model_inputs': torch.stack([item['model_inputs'] for item in inputs]),
            'micro_elements': torch.stack([item['micro_elements'] for item in inputs]),
            'major_elements': torch.stack([item['major_elements'] for item in inputs]),
        }

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.ta_bm_model(x['model_inputs'])
            outputs = outputs.squeeze(-1)
            outputs = outputs * 182.407261 + 49.859524
            mc_increments = x['micro_elements'] @ self.ta_bm_mc_weight @ x['major_elements']
            mc_increments = mc_increments.squeeze(-1).squeeze(-1)
            outputs = outputs + mc_increments
            outputs = outputs.tolist()
            return outputs

    def encode_response(self, output, **kwargs):
        return {
            'prop': 'Bulk Modulus',
            'value': output,
            'unit': 'GPa'
        }


class TASMLitAPI(LitAPI):
    ta_sm_model_path: Path = None
    ta_sm_model: TAP2Model = None
    ta_sm_mc_weight: torch.Tensor = None

    def setup(self, device):
        self.ta_sm_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('TA_SM_260409.pth')))
        self.ta_sm_model = TAP2Model(12, 1, 256, 0.2).eval().to(device)
        self.ta_sm_model.load_state_dict(torch.load(self.ta_sm_model_path, map_location=device))
        self.ta_sm_mc_weight = torch.tensor([
            [1.051068564, 2.718633112, 1.077060841, 1.122686991, 1.112773606,
             1.113072282, 1.128919511, 1.092907451, 0.940211602],
            [1.757030104, 10.13190287, -27.17580326, 232.6984397, -33.38916671,
             1.722337109, 7.034259775, 6.764044231, 6.655379931],
            [3.172669662, 2.062887355, 3.950813852, -22.0090355, -7.071173486,
             2.177602179, 3.612231287, -1.969884481, 3.362529131],
            [1.203795098, -1.44761456, -1.510643413, -0.286721245, -1.035560908,
             -1.423615668, -1.526187408, -1.363322081, -0.716049713],
            [-45.03878237, 56.66865752, 21.94369734, 283.6583886, 360.2530288,
             74.92434101, 32.11387735, 87.88767559, 31.35851808]
        ], dtype=torch.float32, device=device)

    def decode_request(self, request: TAPhysBaseModel, **kwargs):
        model_inputs = torch.tensor([
            request.Ti / 100.0, request.Al / 100.0, request.Si / 100.0, request.Cr / 100.0,
            request.Fe / 100.0, request.Ni / 100.0, request.Cu / 100.0, request.Zr / 100.0,
            request.Nb / 100.0, request.Mo / 100.0, request.V / 100.0, request.Sn / 100.0,
            (request.HTT - 200.0) / 1100.0
        ], dtype=torch.float32, device=self.device)
        micro_elements = torch.tensor([[
            request.C, request.N, request.O, request.B, request.H,
        ]], dtype=torch.float32, device=self.device)
        major_elements = torch.tensor([
            [request.Al], [request.Cr], [request.Cu], [request.Fe], [request.Mo],
            [request.Nb], [request.Ni], [request.Si], [request.Sn]
        ], dtype=torch.float32, device=self.device)
        return {
            'model_inputs': model_inputs,
            'micro_elements': micro_elements,
            'major_elements': major_elements
        }

    def batch(self, inputs):
        return {
            'model_inputs': torch.stack([item['model_inputs'] for item in inputs]),
            'micro_elements': torch.stack([item['micro_elements'] for item in inputs]),
            'major_elements': torch.stack([item['major_elements'] for item in inputs]),
        }

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.ta_sm_model(x['model_inputs'])
            outputs = outputs.squeeze(-1)
            outputs = outputs * 104.934276 + 1e-6
            mc_increments = x['micro_elements'] @ self.ta_sm_mc_weight @ x['major_elements']
            mc_increments = mc_increments.squeeze(-1).squeeze(-1)
            outputs = outputs + mc_increments
            outputs = outputs.tolist()
            return outputs

    def encode_response(self, output, **kwargs):
        return {
            'prop': 'Shear Modulus',
            'value': output,
            'unit': 'GPa'
        }


class TAPRLitAPI(LitAPI):
    ta_pr_model_path: Path = None
    ta_pr_model: TAP2Model = None
    ta_pr_mc_weight: torch.Tensor = None

    def setup(self, device):
        self.ta_pr_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('TA_PR_260409.pth')))
        self.ta_pr_model = TAP2Model(12, 1, 256, 0.2).eval().to(device)
        self.ta_pr_model.load_state_dict(torch.load(self.ta_pr_model_path, map_location=device))
        self.ta_pr_mc_weight = torch.tensor([
            [0.253272904, -0.002701381, -0.000889767, -0.000978328, -0.000909902,
             -0.00092762, -0.0009085, -0.00086636, -0.000786759],
            [-0.001525558, -0.010709855, 0.002303789, -0.522062551, 0.020209486,
             -0.007743567, -0.00557927, -0.005084367, -0.005387734],
            [-0.018617133, -0.019438832, -0.019382127, 0.036048333, -0.012011115,
             -0.020208344, -0.020199332, -0.012779452, -0.019385532],
            [-0.008751755, -0.014158427, -0.014496842, -0.0148, -0.0144,
             -0.0142, -0.014842884, -0.013871117, -0.012828261],
            [0.295234945, 0.307906363, 0.304070677, -0.36820265, 0.096743287,
             0.369543766, 0.319818328, 0.242612884, 0.307072298]
        ], dtype=torch.float32, device=device)

    def decode_request(self, request: TAPhysBaseModel, **kwargs):
        model_inputs = torch.tensor([
            request.Ti / 100.0, request.Al / 100.0, request.Si / 100.0, request.Cr / 100.0,
            request.Fe / 100.0, request.Ni / 100.0, request.Cu / 100.0, request.Zr / 100.0,
            request.Nb / 100.0, request.Mo / 100.0, request.V / 100.0, request.Sn / 100.0,
            (request.HTT - 200.0) / 1100.0
        ], dtype=torch.float32, device=self.device)
        micro_elements = torch.tensor([[
            request.C, request.N, request.O, request.B, request.H,
        ]], dtype=torch.float32, device=self.device)
        major_elements = torch.tensor([
            [request.Al], [request.Cr], [request.Cu], [request.Fe], [request.Mo],
            [request.Nb], [request.Ni], [request.Si], [request.Sn]
        ], dtype=torch.float32, device=self.device)
        return {
            'model_inputs': model_inputs,
            'micro_elements': micro_elements,
            'major_elements': major_elements
        }

    def batch(self, inputs):
        return {
            'model_inputs': torch.stack([item['model_inputs'] for item in inputs]),
            'micro_elements': torch.stack([item['micro_elements'] for item in inputs]),
            'major_elements': torch.stack([item['major_elements'] for item in inputs]),
        }

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.ta_pr_model(x['model_inputs'])
            outputs = outputs.squeeze(-1)
            outputs = outputs * 0.243074 + 0.256926
            mc_increments = x['micro_elements'] @ self.ta_pr_mc_weight @ x['major_elements']
            mc_increments = mc_increments.squeeze(-1).squeeze(-1)
            outputs = outputs + mc_increments
            outputs = outputs.tolist()
            return outputs

    def encode_response(self, output, **kwargs):
        return {
            'prop': 'Poisson Ratio',
            'value': output,
            'unit': None
        }


class TASELitAPI(LitAPI):
    ta_se_model_path: Path = None
    ta_se_model: TAP2Model = None

    def setup(self, device):
        self.ta_se_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('TA_SE_260409.pth')))
        self.ta_se_model = TAP2Model(12, 1, 256, 0.2).eval().to(device)
        self.ta_se_model.load_state_dict(torch.load(self.ta_se_model_path, map_location=device))

    def decode_request(self, request: TAPhysBaseModel, **kwargs):
        model_inputs = torch.tensor([
            request.Ti / 100.0, request.Al / 100.0, request.Si / 100.0, request.Cr / 100.0,
            request.Fe / 100.0, request.Ni / 100.0, request.Cu / 100.0, request.Zr / 100.0,
            request.Nb / 100.0, request.Mo / 100.0, request.V / 100.0, request.Sn / 100.0,
            (request.HTT - 200.0) / 1100.0
        ], dtype=torch.float32, device=self.device)
        return model_inputs

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.ta_se_model(x)
            outputs = outputs.squeeze(-1)
            outputs = outputs * 3216.06424 - 2140.62959
            outputs = outputs.tolist()
            return outputs

    def encode_response(self, output, **kwargs):
        return {
            'prop': 'Specific Enthalpy',
            'value': output,
            'unit': 'J/g'
        }


class TASHCLitAPI(LitAPI):
    ta_shc_model_path: Path = None
    ta_shc_model: TAP2Model = None

    def setup(self, device):
        self.ta_shc_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('TA_SHC_260409.pth')))
        self.ta_shc_model = TAP2Model(12, 1, 256, 0.2).eval().to(device)
        self.ta_shc_model.load_state_dict(torch.load(self.ta_shc_model_path, map_location=device))

    def decode_request(self, request: TAPhysBaseModel, **kwargs):
        model_inputs = torch.tensor([
            request.Ti / 100.0, request.Al / 100.0, request.Si / 100.0, request.Cr / 100.0,
            request.Fe / 100.0, request.Ni / 100.0, request.Cu / 100.0, request.Zr / 100.0,
            request.Nb / 100.0, request.Mo / 100.0, request.V / 100.0, request.Sn / 100.0,
            (request.HTT - 200.0) / 1100.0
        ], dtype=torch.float32, device=self.device)
        return model_inputs

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.ta_shc_model(x)
            outputs = outputs.squeeze(-1)
            outputs = outputs * 0.59492 + 0.0016
            outputs = outputs.tolist()
            return outputs

    def encode_response(self, output, **kwargs):
        return {
            'prop': 'Specific Heat Capacity',
            'value': output,
            'unit': 'J/g·K'
        }


class TATELitAPI(LitAPI):
    ta_te_model_path: Path = None
    ta_te_model: TAP2Model = None
    ta_te_mc_weight: torch.Tensor = None

    def setup(self, device):
        self.ta_te_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('TA_TE_260409.pth')))
        self.ta_te_model = TAP2Model(12, 1, 256, 0.2).eval().to(device)
        self.ta_te_model.load_state_dict(torch.load(self.ta_te_model_path, map_location=device))
        self.ta_te_mc_weight = torch.tensor([
            [-2.89e-10, -4.282498e-10, -7.23e-10, 4.56e-9, 3.085894e-10],
            [4.60e-07, 8.039267e-07, -1.22e-07, -5.52e-07, -4.072913e-08],
            [8.76e-07, -3.124228e-09, 1.22e-07, 1.74e-06, -8.598354e-08],
            [-1.37e-06, -1.089075e-06, 2.52e-07, -5.93e-06, 1.949381e-07]
        ], dtype=torch.float32, device=device)

    def decode_request(self, request: TAPhysBaseModel, **kwargs):
        model_inputs = torch.tensor([
            request.Ti / 100.0, request.Al / 100.0, request.Si / 100.0, request.Cr / 100.0,
            request.Fe / 100.0, request.Ni / 100.0, request.Cu / 100.0, request.Zr / 100.0,
            request.Nb / 100.0, request.Mo / 100.0, request.V / 100.0, request.Sn / 100.0,
            (request.HTT - 200.0) / 1100.0
        ], dtype=torch.float32, device=self.device)
        micro_elements = torch.tensor([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [request.N, request.O, request.C, request.H, request.B],
            [request.N ** 2, request.O ** 2, request.C ** 2, request.H ** 2, request.B ** 2],
            [request.N ** 3, request.O ** 3, request.C ** 3, request.H ** 3, request.B ** 3]
        ], dtype=torch.float32, device=self.device)
        return {
            'model_inputs': model_inputs,
            'micro_elements': micro_elements
        }

    def batch(self, inputs):
        return {
            'model_inputs': torch.stack([item['model_inputs'] for item in inputs]),
            'micro_elements': torch.stack([item['micro_elements'] for item in inputs])
        }

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.ta_te_model(x['model_inputs'])
            outputs = outputs.squeeze(-1)
            outputs = outputs * 9.430234 + 5.890292
            mc_increments = torch.sum(x['micro_elements'] * self.ta_te_mc_weight, dim=(1, 2)) * 1e6
            mc_increments = mc_increments.squeeze(-1).squeeze(-1)
            outputs = outputs + mc_increments
            outputs = outputs.tolist()
            return outputs

    def encode_response(self, output, **kwargs):
        return {
            'prop': 'Thermal Expansivity',
            'value': output,
            'unit': '10^-6/K'
        }


class TABTTLitAPI(LitAPI):
    ta_btt_model_path: Path = None
    ta_btt_model: MoE2 = None

    def setup(self, device):
        self.ta_btt_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('TA_BTT_251214.pth')))
        self.ta_btt_model = MoE2(12, 0, 1, 512, 0.2)\
            .eval().to(device)
        self.ta_btt_model.load_state_dict(torch.load(self.ta_btt_model_path, map_location=device))

    def decode_request(self, request: TAPhysBaseModel, **kwargs):
        model_inputs = torch.tensor([
            request.Ti, request.Al, request.Cr, request.Cu, request.Fe, request.Mo,
            request.Ni, request.Nb, request.Si, request.Sn, request.V, request.Zr
        ], dtype=torch.float32, device=self.device)
        return model_inputs

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.ta_btt_model(x)
            outputs = outputs.squeeze(-1)
            outputs = outputs * 1116.33298 + 202.56698
            outputs = outputs.tolist()
            return outputs

    def encode_response(self, output, **kwargs):
        return {
            'prop': 'β-Transus Temperature',
            'value': output,
            'unit': '℃'
        }


class TAYSLitAPI(LitAPI):
    ta_hp_model_path: Path = None
    ta_hp_model: TAP2Model = None
    ta_ys_model_path: Path = None
    ta_ys_model: TAP2Model = None
    ta_ys_mc_weight: torch.Tensor = None

    def setup(self, device):
        self.ta_hp_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('TA_HP_260409.pth')))
        self.ta_hp_model = TAP2Model(12, 1, 256, 0.2).eval().to(device)
        self.ta_hp_model.load_state_dict(torch.load(self.ta_hp_model_path, map_location=device))
        self.ta_ys_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('TA_YS_260409.pth')))
        self.ta_ys_model = TAP2Model(12, 1, 256, 0.2).eval().to(device)
        self.ta_ys_model.load_state_dict(torch.load(self.ta_ys_model_path, map_location=device))
        self.ta_ys_mc_weight = torch.tensor(
            [2185.6, 1219.86, 704.2, 268.0, -34.0], dtype=torch.float32, device=device
        )

    def decode_request(self, request: TAMechBaseModel, **kwargs):
        model_inputs = torch.tensor([
            request.Ti / 100.0, request.Al / 100.0, request.Si / 100.0, request.Cr / 100.0,
            request.Fe / 100.0, request.Ni / 100.0, request.Cu / 100.0, request.Zr / 100.0,
            request.Nb / 100.0, request.Mo / 100.0, request.V / 100.0, request.Sn / 100.0,
            (request.HTT - 200.0) / 1100.0
        ], dtype=torch.float32, device=self.device)
        grain_size = torch.tensor(request.GS * 1e-6, dtype=torch.float32, device=self.device)
        micro_elements = torch.tensor(
            [request.N, request.O, request.C, request.H, request.B], dtype=torch.float32, device=self.device
        )
        return {
            'model_inputs': model_inputs,
            'grain_size': grain_size,
            'micro_elements': micro_elements
        }

    def batch(self, inputs):
        return {
            'model_inputs': torch.stack([item['model_inputs'] for item in inputs]),
            'grain_size': torch.stack([item['grain_size'] for item in inputs]),
            'micro_elements': torch.stack([item['micro_elements'] for item in inputs])
        }

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.ta_ys_model(x['model_inputs'])
            outputs = outputs.squeeze(-1)
            outputs = outputs * 1894.180857 + 2.297352
            hall_petch = self.ta_hp_model(x['model_inputs'])
            hall_petch = hall_petch.squeeze(-1)
            hall_petch = hall_petch * 1.705577 + 0.007003
            mc_increments = torch.sum(self.ta_ys_mc_weight * x['micro_elements'], dim=1) \
                            + hall_petch * (x['grain_size'] ** -0.5 - 1e-5 ** -0.5)
            outputs = outputs + mc_increments
            outputs = outputs.tolist()
            return outputs

    def encode_response(self, output, **kwargs):
        return {
            'prop': 'Yield Stress',
            'value': output,
            'unit': 'MPa'
        }


class TATSLitAPI(LitAPI):
    ta_hp_model_path: Path = None
    ta_hp_model: TAP2Model = None
    ta_ts_model_path: Path = None
    ta_ts_model: TAP2Model = None
    ta_ts_mc_weight: torch.Tensor = None

    def setup(self, device):
        self.ta_hp_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('TA_HP_260409.pth')))
        self.ta_hp_model = TAP2Model(12, 1, 256, 0.2).eval().to(device)
        self.ta_hp_model.load_state_dict(torch.load(self.ta_hp_model_path, map_location=device))
        self.ta_ts_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('TA_TS_260409.pth')))
        self.ta_ts_model = TAP2Model(12, 1, 256, 0.2).eval().to(device)
        self.ta_ts_model.load_state_dict(torch.load(self.ta_ts_model_path, map_location=device))
        self.ta_ts_mc_weight = torch.tensor(
            [2185.6, 1219.86, 704.2, 268.0, -34.0], dtype=torch.float32, device=device
        )

    def decode_request(self, request: TAMechBaseModel, **kwargs):
        model_inputs = torch.tensor([
            request.Ti / 100.0, request.Al / 100.0, request.Si / 100.0, request.Cr / 100.0,
            request.Fe / 100.0, request.Ni / 100.0, request.Cu / 100.0, request.Zr / 100.0,
            request.Nb / 100.0, request.Mo / 100.0, request.V / 100.0, request.Sn / 100.0,
            (request.HTT - 200.0) / 1100.0
        ], dtype=torch.float32, device=self.device)
        grain_size = torch.tensor(request.GS * 1e-6, dtype=torch.float32, device=self.device)
        micro_elements = torch.tensor(
            [request.N, request.O, request.C, request.H, request.B], dtype=torch.float32, device=self.device
        )
        return {
            'model_inputs': model_inputs,
            'grain_size': grain_size,
            'micro_elements': micro_elements
        }

    def batch(self, inputs):
        return {
            'model_inputs': torch.stack([item['model_inputs'] for item in inputs]),
            'grain_size': torch.stack([item['grain_size'] for item in inputs]),
            'micro_elements': torch.stack([item['micro_elements'] for item in inputs])
        }

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.ta_ts_model(x['model_inputs'])
            outputs = outputs.squeeze(-1)
            outputs = outputs * 2102.338632 + 2.57693
            hall_petch = self.ta_hp_model(x['model_inputs'])
            hall_petch = hall_petch.squeeze(-1)
            hall_petch = hall_petch * 1.705577 + 0.007003
            mc_increments = torch.sum(self.ta_ts_mc_weight * x['micro_elements'], dim=1) \
                            + hall_petch * (x['grain_size'] ** -0.5 - 1e-5 ** -0.5) * 1.1
            outputs = outputs + mc_increments
            outputs = outputs.tolist()
            return outputs

    def encode_response(self, output, **kwargs):
        return {
            'prop': 'Tensile Stress',
            'value': output,
            'unit': 'MPa'
        }


class TAHDLitAPI(LitAPI):
    ta_hd_model_path: Path = None
    ta_hd_model: TAP2Model = None

    def setup(self, device):
        self.ta_hd_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('TA_HD_260409.pth')))
        self.ta_hd_model = TAP2Model(12, 1, 256, 0.2).eval().to(device)
        self.ta_hd_model.load_state_dict(torch.load(self.ta_hd_model_path, map_location=device))

    def decode_request(self, request: TAMechBaseModel, **kwargs):
        model_inputs = torch.tensor([
            request.Ti / 100.0, request.Al / 100.0, request.Si / 100.0, request.Cr / 100.0,
            request.Fe / 100.0, request.Ni / 100.0, request.Cu / 100.0, request.Zr / 100.0,
            request.Nb / 100.0, request.Mo / 100.0, request.V / 100.0, request.Sn / 100.0,
            (request.HTT - 200.0) / 1100.0
        ], dtype=torch.float32, device=self.device)
        return model_inputs

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.ta_hd_model(x)
            outputs = outputs.squeeze(-1)
            outputs = outputs * 704.831511 + 0.870897
            outputs = outputs.tolist()
            return outputs

    def encode_response(self, output, **kwargs):
        return {
            'prop': 'Hardness',
            'value': output,
            'unit': 'VPN'
        }


class TAHPLitAPI(LitAPI):
    ta_hp_model_path: Path = None
    ta_hp_model: TAP2Model = None

    def setup(self, device):
        self.ta_hp_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('TA_HP_260409.pth')))
        self.ta_hp_model = TAP2Model(12, 1, 256, 0.2).eval().to(device)
        self.ta_hp_model.load_state_dict(torch.load(self.ta_hp_model_path, map_location=device))

    def decode_request(self, request: TAMechBaseModel, **kwargs):
        model_inputs = torch.tensor([
            request.Ti / 100.0, request.Al / 100.0, request.Si / 100.0, request.Cr / 100.0,
            request.Fe / 100.0, request.Ni / 100.0, request.Cu / 100.0, request.Zr / 100.0,
            request.Nb / 100.0, request.Mo / 100.0, request.V / 100.0, request.Sn / 100.0,
            (request.HTT - 200.0) / 1100.0
        ], dtype=torch.float32, device=self.device)
        return model_inputs

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.ta_hp_model(x)
            outputs = outputs.squeeze(-1)
            outputs = outputs * 1.705577 + 0.007003
            outputs = outputs.tolist()
            return outputs

    def encode_response(self, output, **kwargs):
        return {
            'prop': 'Hall-Petch Coefficient',
            'value': output,
            'unit': 'MPa·m^(1/2)'
        }


class TAWFLitAPI(LitAPI):
    ta_wf_model_path: Path = None
    ta_wf_model: MoE2 = None
    sparsemax: Sparsemax = None

    def setup(self, device):
        self.ta_wf_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('TA_WF_251220.pth')))
        self.ta_wf_model = MoE2(12, 1, 12, 512, 0.2)\
            .eval().to(device)
        self.ta_wf_model.load_state_dict(torch.load(self.ta_wf_model_path, map_location=device))
        self.sparsemax = Sparsemax(dim=-1)

    def decode_request(self, request: TAWFBaseModel, **kwargs):
        model_inputs = torch.tensor([
            request.Ti, request.Al, request.Cr, request.Cu, request.Fe, request.Mo, request.Ni,
            request.Nb, request.Si, request.Sn, request.V, request.Zr, request.HTT
        ], dtype=torch.float32, device=self.device)
        return model_inputs

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.ta_wf_model(x)
            outputs = self.sparsemax(outputs) * 100.0
            outputs = outputs.tolist()
            return outputs

    def encode_response(self, output, **kwargs):
        return {
            'prop': 'Weight Fraction',
            'value': {
                'ALPHA': output[0],
                'BETA': output[1],
                'LAVES': output[2],
                'TI3AL': output[3],
                'TI2CU': output[4],
                'TI5SI3': output[5],
                'TIZRSI': output[6],
                'TI2NI': output[7],
                'TIM_B2': output[8],
                'LIQUID': output[9],
                'C15_FCC': output[10],
                'MC': output[11]
            },
            'unit': 'wt%'
        }


class AAPhysBaseModel(BaseModel):
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
    HTT: float = Field(default=500, description='热处理温度（℃）', examples=[500], gt=-273.15)

    @model_validator(mode="after")
    def _valid_compos_(self) -> Any:
        total = sum([self.Al, self.Li, self.Mg, self.Si, self.Ca, self.Sc, self.Ti, self.V, self.Cr,
                     self.Mn, self.Fe, self.Co, self.Ni, self.Cu, self.Zn, self.Zr, self.Sn, self.La])
        if not isclose(total, 100, abs_tol=1e-6):
            raise ValueError(f'The sum of all element compositions must be 100, but got {total}.')
        return self

    def __str__(self) -> str:
        return json.dumps({
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
        }, ensure_ascii=False, indent=4)


class AAMechBaseModel(BaseModel):
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
    Mo: float = Field(default=0, description='钼（Mo）的质量分数', examples=[0], ge=0, le=100)
    Sn: float = Field(default=0, description='锡（Sn）的质量分数', examples=[0], ge=0, le=100)
    La: float = Field(default=0, description='镧（La）的质量分数', examples=[0], ge=0, le=100)
    Ce: float = Field(default=0, description='铈（Ce）的质量分数', examples=[0], ge=0, le=100)

    @model_validator(mode="after")
    def _valid_compos_(self) -> Any:
        total = sum([self.Al, self.Li, self.Mg, self.Si, self.Ca, self.Sc, self.Ti, self.V, self.Cr, self.Mn,
                     self.Fe, self.Co, self.Ni, self.Cu, self.Zn, self.Zr, self.Mo, self.Sn, self.La, self.Ce])
        if not isclose(total, 100, abs_tol=1e-6):
            raise ValueError(f'The sum of all element compositions must be 100, but got {total}.')
        return self

    def __str__(self) -> str:
        return json.dumps({
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
            'Mo': f'{self.Mo:f}'.rstrip('0').rstrip('.'),
            'Sn': f'{self.Sn:f}'.rstrip('0').rstrip('.'),
            'La': f'{self.La:f}'.rstrip('0').rstrip('.'),
            'Se': f'{self.Ce:f}'.rstrip('0').rstrip('.')
        }, ensure_ascii=False, indent=4)


class AATCLitAPI(LitAPI):
    aa_tc_model_path: Path = None
    aa_tc_model: AAModel = None

    def setup(self, device):
        self.aa_tc_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('AA_TC_260129.pth')))
        self.aa_tc_model = AAModel(19, 450, 1, 0.2).to(self.device)\
            .eval().to(device)
        self.aa_tc_model.load_state_dict(torch.load(self.aa_tc_model_path, map_location=device))

    def decode_request(self, request: AAPhysBaseModel, **kwargs):
        model_inputs = torch.tensor([
            (request.Al - 72.0) / 27.8, request.Li / 5.0, request.Mg / 12.0, request.Si / 13.0, request.Ca / 2.0,
            request.Sc / 5.0, request.Ti / 2.0, request.V / 8.0, request.Cr / 7.0, request.Mn / 3.0, request.Fe / 3.0,
            request.Co / 5.0, request.Ni / 10.0, request.Cu / 10.0, request.Zn / 12.0, request.Zr / 3.0, request.Sn,
            request.La / 15.0, (request.HTT - 400.0) / 300.0
        ], dtype=torch.float32, device=self.device)
        return model_inputs

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.aa_tc_model(x)
            outputs = outputs.squeeze(-1)
            outputs = outputs * 207.485547 + 28.843647
            outputs = outputs.tolist()
            return outputs

    def encode_response(self, output, **kwargs):
        return {
            'prop': 'Thermal Conductivity',
            'value': output,
            'unit': 'W/m·K'
        }


class AATELitAPI(LitAPI):
    aa_te_model_path: Path = None
    aa_te_model: AAModel = None

    def setup(self, device):
        self.aa_te_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('AA_TE_260129.pth')))
        self.aa_te_model = AAModel(19, 450, 1, 0.2).to(self.device)\
            .eval().to(device)
        self.aa_te_model.load_state_dict(torch.load(self.aa_te_model_path, map_location=device))

    def decode_request(self, request: AAPhysBaseModel, **kwargs):
        model_inputs = torch.tensor([
            (request.Al - 72.0) / 27.8, request.Li / 5.0, request.Mg / 12.0, request.Si / 13.0, request.Ca / 2.0,
            request.Sc / 5.0, request.Ti / 2.0, request.V / 8.0, request.Cr / 7.0, request.Mn / 3.0, request.Fe / 3.0,
            request.Co / 5.0, request.Ni / 10.0, request.Cu / 10.0, request.Zn / 12.0, request.Zr / 3.0, request.Sn,
            request.La / 15.0, (request.HTT - 400.0) / 300.0
        ], dtype=torch.float32, device=self.device)
        return model_inputs

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.aa_te_model(x)
            outputs = outputs.squeeze(-1)
            outputs = outputs * 15.977252 + 18.968151
            outputs = outputs.tolist()
            return outputs

    def encode_response(self, output, **kwargs):
        return {
            'prop': 'Thermal Expansivity',
            'value': output,
            'unit': '10^-6/K'
        }


class AAYSLitAPI(LitAPI):
    aa_ys_model_path: Path = None
    aa_ys_model: TAP2Model = None

    def setup(self, device):
        self.aa_ys_model_path = Path(str(resources.files('tap2.checkpoints').joinpath('AA_YS_260422.pth')))
        self.aa_ys_model = TAP2Model(20, 0, 64, 0.2).eval().to(device)
        self.aa_ys_model.load_state_dict(torch.load(self.aa_ys_model_path, map_location=device))

    def decode_request(self, request: AAMechBaseModel, **kwargs):
        model_inputs = torch.tensor([
            request.Al / 100.0, request.Li / 100.0, request.Mg / 100.0, request.Ca / 100.0, request.Si / 100.0,
            request.Sc / 100.0, request.Ti / 100.0, request.V / 8.0, request.Cr / 100.0, request.Mn / 100.0,
            request.Fe / 100.0, request.Co / 100.0, request.Ni / 100.0, request.Cu / 100.0, request.Zn / 100.0,
            request.Zr / 100.0, request.Mo / 100.0, request.Sn / 100.0, request.La / 100.0, request.Ce / 100.0
        ], dtype=torch.float32, device=self.device)
        return model_inputs

    def predict(self, x, **kwargs):
        with torch.no_grad():
            outputs = self.aa_ys_model(x)
            outputs = outputs.squeeze(-1)
            outputs = outputs * 573.7847 + 23.4653
            outputs = outputs.tolist()
            return outputs

    def encode_response(self, output, **kwargs):
        return {
            'prop': 'Yield Stress',
            'value': output,
            'unit': 'MPa'
        }


@click.command()
@click.option('--host', default='127.0.0.1', type=str)
@click.option('--port', default=7099, type=click.IntRange(1, 65535))
@click.option("--device", default='cpu',
              type=click.Choice(['cpu', 'cuda', 'auto'], case_sensitive=False))
@click.option('--max-batch-size', default=64, type=click.IntRange(16, 10000))
@click.option('--batch-timeout', default=1)
@click.option('--allow-origin', 'allow_origins', multiple=True, default=["*"])
def run_lit_server(host, port, device, max_batch_size, batch_timeout, allow_origins):
    ta_ds_lit_api = TADSLitAPI(
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        api_path='/titanium-alloy/physical/density'
    )
    ta_tc_lit_api = TATCLitAPI(
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        api_path='/titanium-alloy/physical/thermal-conductivity'
    )
    ta_ec_lit_api = TAECLitAPI(
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        api_path='/titanium-alloy/physical/electrical-conductivity'
    )
    ta_ym_lit_api = TAYMLitAPI(
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        api_path='/titanium-alloy/physical/youngs-modulus'
    )
    ta_bm_lit_api = TABMLitAPI(
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        api_path='/titanium-alloy/physical/bulk-modulus'
    )
    ta_sm_lit_api = TASMLitAPI(
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        api_path='/titanium-alloy/physical/shear-modulus'
    )
    ta_pr_lit_api = TAPRLitAPI(
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        api_path='/titanium-alloy/physical/poisson-ratio'
    )
    ta_se_lit_api = TASELitAPI(
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        api_path='/titanium-alloy/physical/specific-enthalpy'
    )
    ta_shc_lit_api = TASHCLitAPI(
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        api_path='/titanium-alloy/physical/specific-heat-capacity'
    )
    ta_te_lit_api = TATELitAPI(
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        api_path='/titanium-alloy/physical/thermal-expansivity'
    )
    ta_btt_lit_api = TABTTLitAPI(
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        api_path='/titanium-alloy/physical/beta-transus-temperature'
    )
    ta_ys_lit_api = TAYSLitAPI(
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        api_path='/titanium-alloy/mechanical/yield-stress'
    )
    ta_ts_lit_api = TATSLitAPI(
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        api_path='/titanium-alloy/mechanical/tensile-stress'
    )
    ta_hd_lit_api = TAHDLitAPI(
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        api_path='/titanium-alloy/mechanical/hardness'
    )
    ta_hp_lit_api = TAHPLitAPI(
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        api_path='/titanium-alloy/mechanical/hall-petch-coefficient'
    )
    ta_wf_lit_api = TAWFLitAPI(
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        api_path='/titanium-alloy/weight-fraction'
    )
    aa_tc_lit_api = AATCLitAPI(
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        api_path='/aluminum-alloy/physical/thermal-conductivity'
    )
    aa_te_lit_api = AATELitAPI(
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        api_path='/aluminum-alloy/physical/thermal-expansivity'
    )
    aa_ys_lit_api = AAYSLitAPI(
        max_batch_size=max_batch_size,
        batch_timeout=batch_timeout,
        api_path='/aluminum-alloy/mechanical/yield-stress'
    )
    cors_middleware = (
        CORSMiddleware,
        {
            "allow_origins": allow_origins,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
    )
    server = LitServer(
        lit_api=[
            ta_ds_lit_api,
            ta_tc_lit_api,
            ta_ec_lit_api,
            ta_ym_lit_api,
            ta_bm_lit_api,
            ta_sm_lit_api,
            ta_pr_lit_api,
            ta_se_lit_api,
            ta_shc_lit_api,
            ta_te_lit_api,
            ta_btt_lit_api,
            ta_ys_lit_api,
            ta_ts_lit_api,
            ta_hd_lit_api,
            ta_hp_lit_api,
            ta_wf_lit_api,
            aa_tc_lit_api,
            aa_te_lit_api,
            aa_ys_lit_api
        ],
        accelerator=device,
        middlewares=[
            cors_middleware
        ]
    )
    server.run(host=host, port=port, generate_client_file=False)

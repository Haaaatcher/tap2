import numpy as np
import torch
import logging
from art import text2art
from torch import tensor
from tqdm import tqdm
from tapp.model import MoE2
from pathlib import Path
from importlib import resources
from typing import Optional, Literal, Any, List
from pydantic import BaseModel, Field, model_validator


class TAPPException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class TAPPInput(BaseModel):

    Prop: Literal["TE", "DS", "TC", "EC", "YM", "BM", "SM", "PR", "SE", "SHC", "YS", "TS", "HD", "HP"] = Field(...,
        description="待预测性能的代号：TE：热膨胀系数，DS：密度，TC：热导率，EC：电导率，YM：杨氏模量，BM：体积模量，SM：剪切模量，"
                    "PR：泊松比，SE：比焓，SHC：比热容，YS：屈服强度，TS：抗拉强度，HD：硬度，HP：霍尔佩奇系数",
        examples=["DS"])
    Ti: float = Field(..., description="钛元素的质量分数", examples=[90.0], ge=0, le=100)
    H: float = Field(..., description="氢元素的质量分数", examples=[0.0], ge=0, le=100)
    B: float = Field(..., description="硼元素元素的质量分数", examples=[0.0], ge=0, le=100)
    C: float = Field(..., description="碳元素的质量分数", examples=[0.0], ge=0, le=100)
    N: float = Field(..., description="氮元素的质量分数", examples=[0.0], ge=0, le=100)
    O: float = Field(..., description="氧元素的质量分数", examples=[0.0], ge=0, le=100)
    Al: float = Field(..., description="铝元素的质量分数", examples=[6.0], ge=0, le=100)
    Si: float = Field(..., description="硅元素的质量分数", examples=[0.0], ge=0, le=100)
    Cr: float = Field(..., description="铬元素的质量分数", examples=[0.0], ge=0, le=100)
    Fe: float = Field(..., description="铁元素的质量分数", examples=[0.0], ge=0, le=100)
    Ni: float = Field(..., description="镍元素的质量分数", examples=[0.0], ge=0, le=100)
    Cu: float = Field(..., description="铜元素的质量分数", examples=[0.0], ge=0, le=100)
    Zr: float = Field(..., description="锆元素的质量分数", examples=[0.0], ge=0, le=100)
    Nb: float = Field(..., description="铌元素的质量分数", examples=[0.0], ge=0, le=100)
    Mo: float = Field(..., description="钼元素的质量分数", examples=[0.0], ge=0, le=100)
    V: float = Field(..., description="钒元素的质量分数", examples=[4.0], ge=0, le=100)
    Sn: float = Field(..., description="锡元素的质量分数", examples=[0.0], ge=0, le=100)
    HTT: float = Field(..., description="热处理温度（摄氏度）", examples=[600.0], gt=-273.15)
    GS: Optional[float] = Field(default=10.0, description="晶粒尺寸（微米）", examples=[10.0], gt=0.0)

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def valid_compos(self) -> Any:
        total = sum([self.Ti, self.H, self.B, self.C, self.N, self.O, self.Al, self.Si, self.Cr, self.Fe, self.Ni,
                     self.Cu, self.Zr, self.Nb, self.Mo, self.V, self.Sn])
        if abs(total - 100.0) > 1e-8:
            raise TAPPException(f"The sum of all element compositions must be <100>, but got <{total}>.")
        return self

    @model_validator(mode="after")
    def valid_gs(self) -> Any:
        if (self.Prop in ["YS", "TS", "HD", "HP"]) and (self.GS is None):
            raise TAPPException(f"<GS> is required for prediction of <{self.Prop}>.")
        return self

    def __str__(self):
        sub_strs = [
            self.Prop,
            *[f"{elem_abbr}{elem_comp:f}".rstrip("0").rstrip(".") for elem_abbr, elem_comp in [
                ["Ti", self.Ti], ["H", self.H], ["B", self.B], ["C", self.C], ["N", self.N], ["O", self.O],
                ["Al", self.Al], ["Si", self.Si], ["Cr", self.Cr], ["Fe", self.Fe], ["Ni", self.Ni], ["Cu", self.Cu],
                ["Zr", self.Zr], ["Nb", self.Nb], ["Mo", self.Mo], ["V", self.V], ["Sn", self.Sn]]],
            "HTT" + f"{self.HTT:f}".rstrip('0').rstrip('.'),
            "GS" + f"{self.GS:f}".rstrip('0').rstrip('.') if (self.GS is not None) else "None"
        ]
        return "-".join(sub_strs)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TAPPInput):
            raise TAPPException(f"Cannot compare {type(self)} to {type(other)}")
        return hash(self) == hash(other)


class TAPPModelInfer:

    _prop_abbrs = ["TE", "DS", "TC", "EC", "YM", "BM", "SM", "PR", "SE", "SHC", "YS", "TS", "HD", "HP"]

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

    def __init__(self, device: torch.device | None = None, batch_size: int = 64, silence: bool = False,
                 log_level: int = logging.INFO):
        """
        使用 TAPP 模型进行推理的工具类，能够进行钛合金性能预测。
        :param device: 指定 Torch 设备，若为 None 则自动选择，并且优先选择 CUDA 设备
        :param batch_size: 进行批量推理时的批量大小
        :param silence: 是否开启静默模式，若为 True 则不打印运行信息
        :param log_level: 日志记录器的日志等级
        """
        # 初始化静默标识
        self._silence = silence
        # 打印 TAPP Logo
        if not self._silence:
            print(text2art("TAPP"))
        # 配置日志记录器
        self._logger = logging.getLogger(__name__ + ".TAPPModelInfer")
        self._logger.setLevel(log_level)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt='[%(levelname)s] [%(asctime)s] [%(name)s] [%(funcName)s:%(lineno)d] %(message)s',
                datefmt="%H:%M:%S"
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        # 初始化 Torch 设备
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if not isinstance(device, torch.device):
                raise TAPPException("<device> is not a <torch.device> instance.")
            self._device = device
        # 初始化批量大小
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise TAPPException("<batch_size> is not a positive integer.")
        self._batch_size = batch_size
        # 加载模型权重
        self._models = {}
        for prop_abbr in tqdm(self._prop_abbrs, desc="Loading models", disable=self._silence):
            model_path = Path(str(resources.files("tapp.weight").joinpath(f"{prop_abbr}.pth")))
            if not model_path.exists():
                raise TAPPException(f"Model file <{model_path}> does not exist.")
            model = MoE2(element_num=12, process_num=1, property_num=1, hidden_size=512, dropout_rate=0.2).to(self._device)
            model.load_state_dict(torch.load(model_path, map_location=self._device))
            model.eval()
            self._models[prop_abbr] = model
        # TS 和 YS 的微量元素修正系数
        self._W1 = torch.tensor(data=[2185.6, 1219.86, 704.2, 268.0, -34.0], dtype=torch.float32, device=self._device)
        # DS、TC、EC、YM、BM、SM、PR的微量元素修正系数
        self._W2 = {
            "DS": tensor(
                data=[[0.001189364, -0.033229803, 0.006900622, 0.001364253, 0.001293301, 0.001219326, 0.001929503, 0.001278101, 0.001198975],
                      [0.003211933, -0.169015533, 2.381598976, 7.424964639, 0.008192758, 0.014823883, 0.018094439, 0.014480812, 0.010925323],
                      [0.01812761, 0.021041677, -0.266758591, -1.897012602, 0.026452594, 0.026654634, 0.010732661, -5.104111794, 0.025470266],
                      [0.022893337, 0.046100565, 0.091790094, 0.04476362, 0.046677253, 0.046287457, 0.053434922, 0.045513762, -0.039840009],
                      [-4.922743468, -4.406682541, 1.499480452, 15.85136756, -4.284999468, -4.414201642, -4.482102982, 45.9676752, -4.531997149]],
                dtype=torch.float32,
                device=self._device
            ),
            "TC": tensor(
                data=[[0.011143497, 3.489852516, 0.050639097, 0.045199609, -0.006363126, -0.146271765, -0.036777042, -0.032206222, -0.035560762],
                      [0.015501867, 17.7350841, 121.4258822, 740.5704625, -42.09685675, -31.53882272, -0.157892947, 0.669169059, -0.220360531],
                      [-7.111188519, -13.84984689, -13.80683762, -81.18458214, -17.8742612, -16.63168331, -8.830106909, -19.40806548, -7.758796868],
                      [-0.079181297, -0.388880921, 0.282376119, -0.019358283, -0.138847193, -0.330954121, -0.429841711, -0.174089174, -0.310205874],
                      [174.2074784, 127.8573, 136.432501, 752.5723302, 446.2291183, 495.3979761, 25.63202273, 153.3714961, 9.850931867]],
                dtype=torch.float32,
                device=self._device
            ),
            "EC": tensor(
                data=[[427.4222959, -1160.893555, 1980.111753, -1703.05578, -355.9538164, -5426.046231, -1222.705087, -1150.778254, -1877.413688],
                      [643.5254332, 5716.60143, -81371.57587, 25064.0197, 8744.494462, 104164.2629, -8668.288187, 49621.93846, -12487.60801],
                      [-354773.0809, -356747.4198, -343208.1088, -242411.9374, -329665.1045, -391633.5205, -409248.1142, -364784.2755, -387130.4589],
                      [-18421.64249, -22875.26649, 13480.95763, -15435.74515, -7469.583248, -18476.56392, -25040.42194, -8473.574668, -22231.51572],
                      [454916.494, 900699.6168, 767781.0268, 368418.7209, 855436.335, 583596.1172, 649707.8811, 263225.8889, 558257.5577]],
                dtype=torch.float32,
                device=self._device
            ),
            "YM": tensor(
                data=[[2.687507505, 6.928988486, 2.760425827, 2.882769734, 2.856143054, 2.851864857, 2.894755453, 2.802196795, 2.409429476],
                      [4.494867834, 25.77829052, -71.10355102, 569.8110626, -86.32504336, 3.873705269, 18.0498347, 17.3860172, 17.07487544],
                      [6.683187786, 3.724882158, 8.733050146, -55.0048898, -19.64440386, 3.999887499, 7.763206322, -6.288934614, 7.191511269],
                      [2.396641297, -0.075441769, -5.220427127, -1.981547993, -3.917721649, -4.955526404, -5.294481326, -4.770403738, -2.979347874],
                      [-92.63189837, 175.7924499, 83.62668492, 717.1182946, 957.9610359, 228.6389485, 111.8343005, 251.9310226, 108.5756246]],
                dtype=torch.float32,
                device=self._device
            ),
            "BM": tensor(
                data=[[1.91143643, 4.782574079, 2.011642041, 2.116353667, 2.107925979, 2.074354268, 2.120185848, 2.053483553, 1.75353913],
                      [3.209269193, 17.46569617, -61.61676964, 228.7679945, -67.88536, -0.822162564, 13.29738451, 13.00498378, 12.55234131],
                      [-4.87225487, -7.610587176, -2.889906738, -30.13159105, -24.46797818, -7.585980355, -4.258953485, -12.81924464, -4.258542434],
                      [-2.946153806, -12.61637001, -12.91788374, -10.19348451, -11.42129274, -12.41581323, -13.18642973, -12.09212666, -9.908171743],
                      [85.02240782, 330.0399644, 242.651319, 450.5155243, 930.2098776, 410.4772374, 277.1712212, 360.7586372, 266.617343]],
                dtype=torch.float32,
                device=self._device
            ),
            "SM": tensor(
                data=[[1.051068564, 2.718633112, 1.077060841, 1.122686991, 1.112773606, 1.113072282, 1.128919511, 1.092907451, 0.940211602],
                      [1.757030104, 10.13190287, -27.17580326, 232.6984397, -33.38916671, 1.722337109, 7.034259775, 6.764044231, 6.655379931],
                      [3.172669662, 2.062887355, 3.950813852, -22.0090355, -7.071173486, 2.177602179, 3.612231287, -1.969884481, 3.362529131],
                      [1.203795098, -1.44761456, -1.510643413, -0.286721245, -1.035560908, -1.423615668, -1.526187408, -1.363322081, -0.716049713],
                      [-45.03878237, 56.66865752, 21.94369734, 283.6583886, 360.2530288, 74.92434101, 32.11387735, 87.88767559, 31.35851808]],
                dtype=torch.float32,
                device=self._device
            ),
            "PR": tensor(
                data=[[0.253272904, -0.002701381, -0.000889767, -0.000978328, -0.000909902, -0.00092762, -0.0009085, -0.00086636, -0.000786759],
                      [-0.001525558, -0.010709855, 0.002303789, -0.522062551, 0.020209486, -0.007743567, -0.00557927, -0.005084367, -0.005387734],
                      [-0.018617133, -0.019438832, -0.019382127, 0.036048333, -0.012011115, -0.020208344, -0.020199332, -0.012779452, -0.019385532],
                      [-0.008751755, -0.014158427, -0.014496842, -0.0148, -0.0144, -0.0142, -0.014842884, -0.013871117, -0.012828261],
                      [0.295234945, 0.307906363, 0.304070677, -0.36820265, 0.096743287, 0.369543766, 0.319818328, 0.242612884, 0.307072298]],
                dtype=torch.float32,
                device=self._device
            )
        }
        # TE 的微量元素修正系数
        self._W3 = torch.tensor(
            data=[[-2.89e-10, -4.282498e-10, -7.23e-10, 4.56e-9, 3.085894e-10],
                  [4.60e-07, 8.039267e-07, -1.22e-07, -5.52e-07, -4.072913e-08],
                  [8.76e-07, -3.124228e-09, 1.22e-07, 1.74e-06, -8.598354e-08],
                  [-1.37e-06, -1.089075e-06, 2.52e-07, -5.93e-06, 1.949381e-07]],
            dtype=torch.float32,
            device=self._device
        )

    def _infer(self, input_: TAPPInput) -> float:
        """
        使用 TAPP 模型进行单点推理，获取预测性能。
        :param input_: 包括待预测性能的代号、钛合金的元素组成、热处理温度和晶粒尺寸（单一输入）
        :return: TAPP 预测的性能值（单一输出）
        """
        if not self._silence:
            self._logger.info(f"INPUT: {input_}")
            self._logger.info("INFERRING")
        # 调用模型进行推理
        model = self._models[input_.Prop]
        inputs_T = tensor(
            data = [[input_.Ti, input_.Al, input_.Cr, input_.Cu, input_.Fe, input_.Mo, input_.Ni, input_.Nb, input_.Si,
                     input_.Sn, input_.V, input_.Zr, input_.HTT]],
            dtype=torch.float32,
            device=self._device
        )
        with torch.no_grad():
            outputs_T = model(inputs_T)
        output = outputs_T.item()
        if not self._silence:
            self._logger.info("FINISHED")
            self._logger.info(f"OUTPUT: {output}")
        # 对模型输出反归一化
        min_value = self._norm_params[input_.Prop]["Min"]
        max_value = self._norm_params[input_.Prop]["Max"]
        output = output * (max_value - min_value) + min_value
        if not self._silence:
            self._logger.info(f"DENORM: {output}")
        return output

    def _batch_infer(self, inputs: List[TAPPInput]) -> List[float]:
        """
        使用 TAPP 模型进行批量推理，获取预测性能。
        :param inputs: 包括待预测性能的代号、钛合金的元素组成、热处理温度和晶粒尺寸（批量输入）
        :return: TAPP 预测的属性能值（批量输出）
        """
        # 输入列表为空，则返回空结果
        if len(inputs) == 0:
            return []
        if not self._silence:
            self._logger.info(f"INPUTS SIZE: {len(inputs)}")
        # 获取性能代码
        prop_abbr = inputs[0].Prop
        if not self._silence:
            self._logger.info(f"PROPERTY: {prop_abbr}")
        # 调用模型进行推理
        model = self._models[prop_abbr]
        outputs = []
        min_value = self._norm_params[prop_abbr]["Min"]
        max_value = self._norm_params[prop_abbr]["Max"]
        with torch.no_grad():
            # 按批处理，步长设置为 batch_size
            for batch_id, start_idx in enumerate(range(0, len(inputs), self._batch_size)):
                if not self._silence:
                    self._logger.info(f"INFERRING BATCH {batch_id}")
                inputs_T = tensor(
                    data=[[input_.Ti, input_.Al, input_.Cr, input_.Cu, input_.Fe, input_.Mo, input_.Ni, input_.Nb,
                           input_.Si, input_.Sn, input_.V, input_.Zr, input_.HTT]
                          for input_ in inputs[start_idx : start_idx + self._batch_size]],
                    dtype=torch.float32,
                    device=self._device
                )
                outputs_T = model(inputs_T) # batch_size x 1
                outputs_T = outputs_T.squeeze(-1) # batch_size
                outputs_T = outputs_T * (max_value - min_value) + min_value
                outputs.extend(outputs_T.cpu().tolist())
                if not self._silence:
                    self._logger.info(f"FINISHED BATCH {batch_id}")
        return outputs

    def _corr(self, input_: TAPPInput, orig_output: float) -> float:
        """
        利用经验公式对 TAPP 模型的预测结果进行修正。
        :param input_: 包括待预测性能的代号、钛合金的元素组成、热处理温度和晶粒尺寸（单一输入）
        :param orig_output: TAPP 预测的原始性能（单一输入）
        :return: 经验公式修正后的性能（单一输出）
        """
        if not self._silence:
            self._logger.info(f"INPUT: {input_}")
            self._logger.info(f"ORIGINAL OUTPUT: {orig_output}")
        output = orig_output
        if not self._silence:
            self._logger.info("CORRECTING")
        if input_.Prop in ["YS", "TS"]:
            # 微量元素修正
            micro_elem_T = torch.tensor(
                data=[input_.N, input_.O, input_.C, input_.H, input_.B],
                dtype=torch.float32,
                device=self._device
            )
            inc = torch.sum(self._W1 * micro_elem_T).item()
            if input_.Prop == "TS":
                inc *= 1.1
            output += inc
            # 晶粒尺寸修正
            GS = input_.GS * 1e-6
            HP = self._infer(input_.model_copy(update={"Prop": "HP", "GS": 10.0}))
            inc = HP * (pow(GS, -0.5) - pow(1e-5, -0.5))
            output += inc
        elif input_.Prop in ["DS", "TC", "EC", "YM", "BM", "SM", "PR"]:
            # 微量元素修正
            micro_elem_T = torch.tensor(
                data=[[input_.C, input_.N, input_.O, input_.B, input_.H]],
                dtype=torch.float32,
                device=self._device
            )
            main_elem_T = torch.tensor(
                data=[[input_.Al], [input_.Cr], [input_.Cu], [input_.Fe], [input_.Mo],
                      [input_.Nb], [input_.Ni], [input_.Si], [input_.Sn]],
                dtype=torch.float32,
                device=self._device
            )
            inc = (micro_elem_T @ self._W2[input_.Prop] @ main_elem_T).item()
            output += inc
        elif input_.Prop == "TE":
            # 微量元素修正
            micro_elem_T = torch.tensor(
                data=[[1.0, 1.0, 1.0, 1.0, 1.0],
                      [input_.N, input_.O, input_.C, input_.H, input_.B],
                      [input_.N ** 2, input_.O ** 2, input_.C ** 2, input_.H ** 2, input_.B ** 2],
                      [input_.N ** 3, input_.O ** 3, input_.C ** 3, input_.H ** 3, input_.B ** 3]],
                dtype=torch.float32,
                device=self._device
            )
            inc = torch.sum(micro_elem_T * self._W3).item()
            output += inc
        if not self._silence:
            self._logger.info("FINISHED")
            self._logger.info(f"CORRECTED OUTPUT: {output}")
        return output

    def _batch_corr(self, inputs: List[TAPPInput], orig_outputs: List[float]) -> List[float]:
        """
        利用经验公式对 TAPP 模型的预测结果进行修正。
        :param inputs: 包括待预测性能的代号、钛合金的元素组成、热处理温度和晶粒尺寸（批量输入）
        :param orig_outputs: TAPP 预测的原始性能（批量输入）
        :return: 经验公式修正后的性能（批量输出）
        """
        # 输入列表为空，则返回空结果
        if len(inputs) == 0:
            return []
        # 输入列表和原始输出列表长度一致
        if len(inputs) != len(orig_outputs):
            raise TAPPException("The length of <inputs> and <orig_outputs> must be the same.")
        if not self._silence:
            self._logger.info(f"INPUTS SIZE: {len(inputs)}")
        # 获取性能代码
        prop_abbr = inputs[0].Prop
        if not self._silence:
            self._logger.info(f"PROPERTY: {prop_abbr}")
        outputs_T = tensor(orig_outputs, dtype=torch.float32, device=self._device)
        for batch_id, start_idx in enumerate(range(0, len(inputs), self._batch_size)):
            if not self._silence:
                self._logger.info(f"CORRECTING BATCH {batch_id}")
            if prop_abbr in ["YS", "TS"]:
                # 微量元素修正
                micro_elem_T = tensor(
                    data=[[input_.N, input_.O, input_.C, input_.H, input_.B]
                          for input_ in inputs[start_idx:start_idx + self._batch_size]],
                    dtype=torch.float32,
                    device=self._device
                )
                inc_T = torch.sum(self._W1 * micro_elem_T, dim=1)
                if prop_abbr == "TS":
                    inc_T *= 1.1
                outputs_T[start_idx:start_idx + self._batch_size] += inc_T
                # 晶粒尺寸修正
                GS_T = tensor(
                    data=[input_.GS for input_ in inputs[start_idx:start_idx + self._batch_size]],
                    dtype=torch.float32,
                    device=self._device
                )
                GS_T *= 1e-6
                HP_T = tensor(
                    data=self._batch_infer([input_.model_copy(update={"Prop": "HP", "GS": 10.0})
                                            for input_ in inputs[start_idx:start_idx + self._batch_size]]),
                    dtype=torch.float32,
                    device=self._device
                )
                inc_T = HP_T * (torch.pow(GS_T, -0.5) - np.pow(1e-5, -0.5))
                outputs_T[start_idx:start_idx + self._batch_size] += inc_T
            elif prop_abbr in ["DS", "TC", "EC", "YM", "BM", "SM", "PR"]:
                # 微量元素修正
                micro_elem_T = torch.tensor(
                    data=[[[input_.C, input_.N, input_.O, input_.B, input_.H]]
                          for input_ in inputs[start_idx:start_idx + self._batch_size]],
                    dtype=torch.float32,
                    device=self._device
                ) # batch_size x 1 x 5
                main_elem_T = torch.tensor(
                    data=[[[input_.Al], [input_.Cr], [input_.Cu], [input_.Fe], [input_.Mo],
                           [input_.Nb], [input_.Ni], [input_.Si], [input_.Sn]]
                          for input_ in inputs[start_idx:start_idx + self._batch_size]],
                    dtype=torch.float32,
                    device=self._device
                ) # batch_size x 9 x 1
                inc_T = micro_elem_T @ self._W2[prop_abbr] @ main_elem_T # batch_size x 1 x 1
                inc_T = inc_T.squeeze(-1).squeeze(-1) # batch_size
                outputs_T[start_idx:start_idx + self._batch_size] += inc_T
            elif prop_abbr == "TE":
                # 微量元素修正
                micro_elem_T = torch.tensor(
                    data=[[[1.0, 1.0, 1.0, 1.0, 1.0],
                           [input_.N, input_.O, input_.C, input_.H, input_.B],
                           [input_.N ** 2, input_.O ** 2, input_.C ** 2, input_.H ** 2, input_.B ** 2],
                           [input_.N ** 3, input_.O ** 3, input_.C ** 3, input_.H ** 3, input_.B ** 3]]
                          for input_ in inputs[start_idx:start_idx + self._batch_size]],
                    dtype=torch.float32,
                    device=self._device
                ) # batch_size x 4 x 5
                inc_T = torch.sum(micro_elem_T * self._W3, dim=(1, 2)) # batch_size
                outputs_T[start_idx:start_idx + self._batch_size] += inc_T
            if not self._silence:
                self._logger.info(f"FINISHED BATCH {batch_id}")
        return outputs_T.cpu().tolist()

    def __call__(self, input_data: TAPPInput | List[TAPPInput]) -> float | List[float]:
        """
        调用 TAPP 模型进行推理，预测钛合金的性能，支持批量处理。
        :param input_data: 包括待预测性能名称、钛合金的元素组成、热处理温度和晶粒尺寸（单一输入或批量输入）
        :return: TAPP 预测的性能值（单一输出或批量输出）
        """
        # 检查输入数据类型
        if isinstance(input_data, TAPPInput):
            if not self._silence:
                self._logger.info("SINGLE MODE")
            orig_output = self._infer(input_data)
            corr_output = self._corr(input_data, orig_output)
            return corr_output
        elif isinstance(input_data, List) and all(isinstance(input_, TAPPInput) for input_ in input_data):
            if not self._silence:
                self._logger.info("BATCH MODE")
            if len(input_data) == 0:
                return []
            else:
                # 检查性能代码是否统一
                if len(set(input_.Prop for input_ in input_data)) != 1:
                    raise TAPPException("The property names of all inputs must be the same.")
            orig_outputs = self._batch_infer(input_data)
            corr_outputs = self._batch_corr(input_data, orig_outputs)
            return corr_outputs
        else:
            raise TAPPException("<input_data> is not <TAPPInput> or <List[TAPPInput]> instance.")

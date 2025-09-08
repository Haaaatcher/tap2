import json
import numpy as np
import torch
import csv
from torch import tensor
from tqdm import tqdm
from tapp.model import MoE2
from pathlib import Path
from importlib import resources


class TAPPException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class TAPPModelInfer:
    # 支持的属性名称
    _property_names = ["Thermal Expansion", "Density", "Thermal Conductivity", "Electrical Conductivity",
                       "Youngs Modulus", "Bulk Modulus", "Shear Modulus", "Poisson Ratio", "Specific Enthalpy",
                       "Specific Heat Capacity", "Yield Stress", "Tensile Stress", "Hardness", "Hall Petch"]

    # 输入字段默认值
    _input_default_values = {
        "Ti": 100.0,
        "Al": 0.0,
        "Cr": 0.0,
        "Cu": 0.0,
        "Fe": 0.0,
        "Mo": 0.0,
        "Ni": 0.0,
        "Nb": 0.0,
        "Si": 0.0,
        "Sn": 0.0,
        "V": 0.0,
        "Zr": 0.0,
        "N": 0.0,
        "O": 0.0,
        "C": 0.0,
        "H": 0.0,
        "B": 0.0,
        "Mn": 0.0,
        "Ta": 0.0,
        "Bi": 0.0,
        "Co": 0.0,
        "Heat Treatment Temperature": 600.0,
        "Grain Size": 10.0
    }

    # Torch计算设备
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 批处理大小
    _batch_size = 64

    # 强度微量元素修正系数
    _W1 = torch.tensor(data=[2185.6, 1219.86, 704.2, 268.0, -34.0], dtype=torch.float32, device=_device)

    # 物理性能微量元素修正系数
    _W2 = {
        "Density": tensor(
            data=[[0.001189364, -0.033229803, 0.006900622, 0.001364253, 0.001293301, 0.001219326, 0.001929503, 0.001278101, 0.001198975],
                  [0.003211933, -0.169015533, 2.381598976, 7.424964639, 0.008192758, 0.014823883, 0.018094439, 0.014480812, 0.010925323],
                  [0.01812761, 0.021041677, -0.266758591, -1.897012602, 0.026452594, 0.026654634, 0.010732661, -5.104111794, 0.025470266],
                  [0.022893337, 0.046100565, 0.091790094, 0.04476362, 0.046677253, 0.046287457, 0.053434922, 0.045513762, -0.039840009],
                  [-4.922743468, -4.406682541, 1.499480452, 15.85136756, -4.284999468, -4.414201642, -4.482102982, 45.9676752, -4.531997149]],
            dtype=torch.float32,
            device=_device
        ),
        "Thermal Conductivity": tensor(
            data=[[0.011143497, 3.489852516, 0.050639097, 0.045199609, -0.006363126, -0.146271765, -0.036777042, -0.032206222, -0.035560762],
                  [0.015501867, 17.7350841, 121.4258822, 740.5704625, -42.09685675, -31.53882272, -0.157892947, 0.669169059, -0.220360531],
                  [-7.111188519, -13.84984689, -13.80683762, -81.18458214, -17.8742612, -16.63168331, -8.830106909, -19.40806548, -7.758796868],
                  [-0.079181297, -0.388880921, 0.282376119, -0.019358283, -0.138847193, -0.330954121, -0.429841711, -0.174089174, -0.310205874],
                  [174.2074784, 127.8573, 136.432501, 752.5723302, 446.2291183, 495.3979761, 25.63202273, 153.3714961, 9.850931867]],
            dtype=torch.float32,
            device=_device
        ),
        "Electrical Conductivity": tensor(
            data=[[427.4222959, -1160.893555, 1980.111753, -1703.05578, -355.9538164, -5426.046231, -1222.705087, -1150.778254, -1877.413688],
                  [643.5254332, 5716.60143, -81371.57587, 25064.0197, 8744.494462, 104164.2629, -8668.288187, 49621.93846, -12487.60801],
                  [-354773.0809, -356747.4198, -343208.1088, -242411.9374, -329665.1045, -391633.5205, -409248.1142, -364784.2755, -387130.4589],
                  [-18421.64249, -22875.26649, 13480.95763, -15435.74515, -7469.583248, -18476.56392, -25040.42194, -8473.574668, -22231.51572],
                  [454916.494, 900699.6168, 767781.0268, 368418.7209, 855436.335, 583596.1172, 649707.8811, 263225.8889, 558257.5577]],
            dtype=torch.float32,
            device=_device
        ),
        "Youngs Modulus": tensor(
            data=[[2.687507505, 6.928988486, 2.760425827, 2.882769734, 2.856143054, 2.851864857, 2.894755453, 2.802196795, 2.409429476],
                  [4.494867834, 25.77829052, -71.10355102, 569.8110626, -86.32504336, 3.873705269, 18.0498347, 17.3860172, 17.07487544],
                  [6.683187786, 3.724882158, 8.733050146, -55.0048898, -19.64440386, 3.999887499, 7.763206322, -6.288934614, 7.191511269],
                  [2.396641297, -0.075441769, -5.220427127, -1.981547993, -3.917721649, -4.955526404, -5.294481326, -4.770403738, -2.979347874],
                  [-92.63189837, 175.7924499, 83.62668492, 717.1182946, 957.9610359, 228.6389485, 111.8343005, 251.9310226, 108.5756246]],
            dtype=torch.float32,
            device=_device
        ),
        "Bulk Modulus": tensor(
            data=[[1.91143643, 4.782574079, 2.011642041, 2.116353667, 2.107925979, 2.074354268, 2.120185848, 2.053483553, 1.75353913],
                  [3.209269193, 17.46569617, -61.61676964, 228.7679945, -67.88536, -0.822162564, 13.29738451, 13.00498378, 12.55234131],
                  [-4.87225487, -7.610587176, -2.889906738, -30.13159105, -24.46797818, -7.585980355, -4.258953485, -12.81924464, -4.258542434],
                  [-2.946153806, -12.61637001, -12.91788374, -10.19348451, -11.42129274, -12.41581323, -13.18642973, -12.09212666, -9.908171743],
                  [85.02240782, 330.0399644, 242.651319, 450.5155243, 930.2098776, 410.4772374, 277.1712212, 360.7586372, 266.617343]],
            dtype=torch.float32,
            device=_device
        ),
        "Sheer Modulus": tensor(
            data=[[1.051068564, 2.718633112, 1.077060841, 1.122686991, 1.112773606, 1.113072282, 1.128919511, 1.092907451, 0.940211602],
                  [1.757030104, 10.13190287, -27.17580326, 232.6984397, -33.38916671, 1.722337109, 7.034259775, 6.764044231, 6.655379931],
                  [3.172669662, 2.062887355, 3.950813852, -22.0090355, -7.071173486, 2.177602179, 3.612231287, -1.969884481, 3.362529131],
                  [1.203795098, -1.44761456, -1.510643413, -0.286721245, -1.035560908, -1.423615668, -1.526187408, -1.363322081, -0.716049713],
                  [-45.03878237, 56.66865752, 21.94369734, 283.6583886, 360.2530288, 74.92434101, 32.11387735, 87.88767559, 31.35851808]],
            dtype=torch.float32,
            device=_device
        ),
        "Poisson Ratio": tensor(
            data=[[0.253272904, -0.002701381, -0.000889767, -0.000978328, -0.000909902, -0.00092762, -0.0009085, -0.00086636, -0.000786759],
                  [-0.001525558, -0.010709855, 0.002303789, -0.522062551, 0.020209486, -0.007743567, -0.00557927, -0.005084367, -0.005387734],
                  [-0.018617133, -0.019438832, -0.019382127, 0.036048333, -0.012011115, -0.020208344, -0.020199332, -0.012779452, -0.019385532],
                  [-0.008751755, -0.014158427, -0.014496842, -0.0148, -0.0144, -0.0142, -0.014842884, -0.013871117, -0.012828261],
                  [0.295234945, 0.307906363, 0.304070677, -0.36820265, 0.096743287, 0.369543766, 0.319818328, 0.242612884, 0.307072298]],
            dtype=torch.float32,
            device=_device
        )
    }

    def __init__(self):
        """
        TAPP模型推理类，负责加载模型和归一化参数，并提供推理和修正功能。
        """
        # 读取模型权重
        self._models = {}
        for property_name in tqdm(self._property_names, desc="Loading models"):
            model_path = Path(str(resources.files("tapp.weight").joinpath(f"MoE2({property_name}).pth")))
            if not model_path.exists():
                raise TAPPException(f"Model file \"{model_path}\" does not exist.")
            model = MoE2(element_num=12, process_num=1, property_num=1, hidden_size=512, dropout_rate=0.2).to(self._device)
            model.load_state_dict(torch.load(model_path, map_location=self._device))
            model.eval()
            self._models[property_name] = model
        # 读取Beta模型权重
        model_path = Path(str(resources.files("tapp.weight").joinpath(f"MoE2(Thermal Conductivity Beta).pth")))
        if not model_path.exists():
            raise TAPPException(f"Model file \"{model_path}\" does not exist.")
        model = MoE2(element_num=16, process_num=1, property_num=1, hidden_size=512, dropout_rate=0).to(self._device)
        model.load_state_dict(torch.load(model_path, map_location=self._device))
        model.eval()
        self._models["Thermal Conductivity Beta"] = model
        model_path = Path(str(resources.files("tapp.weight").joinpath(f"MoE2(Thermal Expansion Beta).pth")))
        if not model_path.exists():
            raise TAPPException(f"Model file \"{model_path}\" does not exist.")
        model = MoE2(element_num=18, process_num=1, property_num=1, hidden_size=512, dropout_rate=0).to(self._device)
        model.load_state_dict(torch.load(model_path, map_location=self._device))
        model.eval()
        self._models["Thermal Expansion Beta"] = model
        # 读取归一化参数
        self._norm_params = {}
        for property_name in self._property_names:
            norm_file_path = Path(str(resources.files("tapp.normalization").joinpath(f"norm_param({property_name}).json")))
            if not norm_file_path.exists():
                raise TAPPException(f"Normalization parameters file \"{norm_file_path}\" does not exist.")
            with open(norm_file_path, "r", encoding="utf-8") as fp:
                self._norm_params[property_name] = json.load(fp)
        # 读取Beta版本数据表
        self._stress_beta = {}
        csv_path = Path(str(resources.files("tapp.dataset").joinpath(f"stress_beta.csv")))
        if not csv_path.exists():
            raise TAPPException(f"Dataset file \"{csv_path}\" does not exist.")
        with open(csv_path, "r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                value = {k: float(v) for k, v in row.items() if v}
                key = self._get_hash_str(value)
                self._stress_beta[key] = value
        self._tc_beta = {}
        csv_path = Path(str(resources.files("tapp.dataset").joinpath(f"tc_beta.csv")))
        if not csv_path.exists():
            raise TAPPException(f"Dataset file \"{csv_path}\" does not exist.")
        with open(csv_path, "r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                value = {k: float(v) for k, v in row.items() if v}
                key = self._get_hash_str(value)
                self._tc_beta[key] = value
        self._te_beta = {}
        csv_path = Path(str(resources.files("tapp.dataset").joinpath(f"te_beta.csv")))
        if not csv_path.exists():
            raise TAPPException(f"Dataset file \"{csv_path}\" does not exist.")
        with open(csv_path, "r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                value = {k: float(v) for k, v in row.items() if v}
                key = self._get_hash_str(value)
                self._te_beta[key] = value

    def _infer(self, prop_name: str, input_: dict[str, int | float]) -> float:
        """
        使用指定模型对输入数据进行推理，获取属性值。
        :param prop_name: 属性名称
        :param input_: 元素组成和热处理温度等参数
        :return: 属性值
        """
        model = self._models[prop_name]
        inputs_T = tensor(
            data=[[input_.get(field_name, self._input_default_values[field_name])
                   for field_name in ["Ti", "Al", "Cr", "Cu", "Fe", "Mo", "Ni", "Nb", "Si",
                                      "Sn", "V", "Zr", "Heat Treatment Temperature"]]],
            dtype=torch.float32,
            device=self._device
        )
        with torch.no_grad():
            outputs_T = model(inputs_T)
        output = outputs_T.cpu().tolist()[0][0]
        min_value = self._norm_params[prop_name]["Mins"][prop_name]
        max_value = self._norm_params[prop_name]["Maxs"][prop_name]
        output = output * (max_value - min_value) + min_value
        return output

    def _batch_infer(self, prop_name: str, inputs: list[dict[str, int | float]]) -> list[float]:
        """
        使用指定模型对输入数据进行批量推理，获取属性值。
        :param prop_name: 属性名称
        :param inputs: 元素组成和热处理温度等参数
        :return: 属性值
        """
        model = self._models[prop_name]
        outputs = []
        with torch.no_grad():
            for idx in range(0, len(inputs), self._batch_size):
                inputs_T = tensor(
                    data=[[input_.get(field_name, self._input_default_values[field_name])
                           for field_name in ["Ti", "Al", "Cr", "Cu", "Fe", "Mo", "Ni", "Nb", "Si",
                                              "Sn", "V", "Zr", "Heat Treatment Temperature"]]
                          for input_ in inputs[idx:idx + self._batch_size]],
                    dtype=torch.float32,
                    device=self._device
                )
                outputs_T = model(inputs_T) # batch_size x 1
                outputs_T = outputs_T.squeeze(-1) # batch_size
                min_value = self._norm_params[prop_name]["Mins"][prop_name]
                max_value = self._norm_params[prop_name]["Maxs"][prop_name]
                outputs_T = outputs_T * (max_value - min_value) + min_value
                outputs.extend(outputs_T.cpu().tolist())
        return outputs

    def _corr(self, prop_name: str, input_: dict[str, int | float], orig_output: float) -> float:
        """
        对原始模型的输出结果进行经验修正。
        :param prop_name: 属性名称
        :param input_: 元素组成和热处理温度等参数
        :param orig_output: 原始属性值
        :return: 修正后的属性值
        """
        output = orig_output
        if prop_name in ["Yield Stress", "Tensile Stress"]:
            # 微量元素修正
            micro_elements_T = torch.tensor(
                data=[input_.get(element_name, 0.0) for element_name in ["N", "O", "C", "H", "B"]],
                dtype=torch.float32,
                device=self._device
            )
            increment = torch.sum(self._W1 * micro_elements_T).item()
            if prop_name == "Tensile Stress":
                increment *= 1.1
            output += increment
            # 晶粒尺寸修正
            grain_size = input_.get("Grain Size", self._input_default_values["Grain Size"]) * 1e-6
            hall_petch = self._infer("Hall Petch", input_)
            increment = hall_petch * (pow(grain_size, -0.5) - pow(1e-5, -0.5))
            output += increment
        elif prop_name in ["Density", "Thermal Conductivity", "Electrical Conductivity", "Youngs Modulus",
                           "Bulk Modulus", "Sheer Modulus", "Poisson Ratio"]:
            micro_elements_T = torch.tensor(
                data=[[input_.get(element_name, 0.0) for element_name in ["C", "N", "O", "B", "H"]]],
                dtype=torch.float32,
                device=self._device
            )
            main_elements_T = torch.tensor(
                data=[[input_.get(element_name, 0.0)]
                      for element_name in ["Al", "Cr", "Cu", "Fe", "Mo", "Nb", "Ni", "Si", "Sn"]],
                dtype=torch.float32,
                device=self._device
            )
            increment = micro_elements_T @ self._W2[prop_name] @ main_elements_T
            increment = increment.item()
            output += increment
        return output

    def _batch_corr(self, prop_name: str, inputs: list[dict[str, int | float]], orig_outputs: list[float]) \
            -> list[float]:
        """
        对原始模型的输出结果进行批量经验修正。
        :param prop_name: 属性名称
        :param inputs: 元素组成和热处理温度等参数
        :param orig_outputs: 原始属性值
        :return: 修正后的属性值
        """
        outputs_T = tensor(orig_outputs, dtype=torch.float32, device=self._device)
        for idx in range(0, len(inputs), self._batch_size):
            if prop_name in ["Yield Stress", "Tensile Stress"]:
                # 微量元素修正
                micro_elements_T = tensor(
                    data=[[input_.get(element_name, 0.0) for element_name in ["N", "O", "C", "H", "B"]]
                          for input_ in inputs[idx:idx + self._batch_size]],
                    dtype=torch.float32,
                    device=self._device
                )
                increment_T = torch.sum(self._W1 * micro_elements_T, dim=1)
                if prop_name == "Tensile Stress":
                    increment_T *= 1.1
                outputs_T[idx:idx + self._batch_size] += increment_T
                # 晶粒尺寸修正
                grain_size_T = tensor(
                    data=[input_.get("Grain Size", 10.0)
                          for input_ in inputs[idx:idx + self._batch_size]],
                    dtype=torch.float32,
                    device=self._device
                )
                grain_size_T *= 1e-6
                hall_petch_T = tensor(
                    data=self._batch_infer("Hall Petch", inputs[idx:idx + self._batch_size]),
                    dtype=torch.float32,
                    device=self._device
                )
                increment_T = hall_petch_T * (torch.pow(grain_size_T, -0.5) - np.pow(1e-5, -0.5))
                outputs_T[idx:idx + self._batch_size] += increment_T
            elif prop_name in ["Density", "Thermal Conductivity", "Electrical Conductivity", "Youngs Modulus",
                               "Bulk Modulus", "Sheer Modulus", "Poisson Ratio"]:
                micro_elements_T = torch.tensor(
                    data=[[[input_.get(element_name, 0.0)
                            for element_name in ["C", "N", "O", "B", "H"]]]
                          for input_ in inputs[idx:idx + self._batch_size]],
                    dtype=torch.float32,
                    device=self._device
                ) # batch_size x 1 x 5
                main_elements_T = torch.tensor(
                    data=[[[input_data_item.get(element_name, 0.0)]
                           for element_name in ["Al", "Cr", "Cu", "Fe", "Mo", "Nb", "Ni", "Si", "Sn"]]
                          for input_data_item in inputs[idx:idx + self._batch_size]],
                    dtype=torch.float32,
                    device=self._device
                ) # batch_size x 9 x 1
                W2 = self._W2[prop_name] # 5 x 9
                increment_T = micro_elements_T @ W2 @ main_elements_T # batch_size x 1 x 1
                increment_T = increment_T.squeeze(-1).squeeze(-1) # batch_size
                outputs_T[idx:idx + self._batch_size] += increment_T
        return outputs_T.cpu().tolist()

    @staticmethod
    def _get_hash_str(input_: dict[str, int | float], def_values: dict[str, int | float] | None = None) -> str:
        """
        获取输入数据的哈希字符串。
        :param input_: 元素组成和热处理温度等参数
        :param def_values: 字段的默认值
        :return: 哈希值字符串
        """
        if def_values is None:
            def_values = {
                "Ti": 100.0,
                "Al": 0.0,
                "Cr": 0.0,
                "Cu": 0.0,
                "Fe": 0.0,
                "Mo": 0.0,
                "Ni": 0.0,
                "Nb": 0.0,
                "Si": 0.0,
                "Sn": 0.0,
                "V": 0.0,
                "Zr": 0.0,
                "N": 0.0,
                "O": 0.0,
                "C": 0.0,
                "H": 0.0,
                "B": 0.0,
                "Mn": 0.0,
                "Ta": 0.0,
                "Bi": 0.0,
                "Co": 0.0,
                "Heat Treatment Temperature": 600.0,
            }
        sub_strs = []
        for field_name, default_value in def_values.items():
            sub_strs.append(f"{input_.get(field_name, default_value):f}".rstrip("0").rstrip("."))
        return "-".join(sub_strs)

    def _infer_beta(self, prop_name: str, input_: dict[str, int | float]) -> float | None:
        """
        使用Beta版本模型对输入数据进行推理，获取属性值。
        若不满足Beta版本模型的输入条件，则返回None。
        :param prop_name: 属性名称
        :param input_: 元素组成和热处理温度等参数
        :return: 属性值
        """
        if prop_name in ["Yield Stress", "Tensile Stress"]:
            hash_str = self._get_hash_str(input_)
            if hash_str in self._stress_beta:
                beta_input = self._stress_beta[hash_str]
                beta_output = self._infer(prop_name, beta_input)
                corr_output = self._corr(prop_name, beta_input, beta_output)
                return corr_output
            else:
                return None
        elif prop_name == "Thermal Expansion":
            hash_str = self._get_hash_str(input_)
            if hash_str in self._te_beta:
                model = self._models["Thermal Expansion Beta"]
                beta_input = self._te_beta[hash_str]
                beta_input_T = tensor(
                    data=[[beta_input.get(field_name, self._input_default_values[field_name])
                           for field_name in ["Ti", "Al", "Mn", "V", "Fe", "O", "Mo", "Zr", "Sn", "Cr", "Si", "C",
                                              "N", "Nb", "Co", "Ta", "Bi", "Cu", "Heat Treatment Temperature"]]],
                    dtype=torch.float32,
                    device=self._device
                )
                beta_output_T = model(beta_input_T)
                output = beta_output_T.cpu().tolist()[0][0] * 1e-6
                return output
            else:
                return None
        elif prop_name == "Thermal Conductivity":
            hash_str = self._get_hash_str(input_)
            if hash_str in self._tc_beta:
                model = self._models["Thermal Conductivity Beta"]
                beta_input = self._tc_beta[hash_str]
                beta_input_T = tensor(
                    data=[[beta_input.get(field_name, self._input_default_values[field_name])
                           for field_name in ["Ti", "Al", "Mn", "V", "Fe", "O", "Mo", "Zr", "Sn", "Nb", "Si",
                                              "Cr", "C", "Ta", "Bi", "Cu", "Heat Treatment Temperature"]]],
                    dtype=torch.float32,
                    device=self._device
                )
                beta_output_T = model(beta_input_T)
                output = beta_output_T.cpu().tolist()[0][0]
                return output
            else:
                return None
        else:
            return None

    def __call__(self, prop_name: str, input_data: dict[str, int | float] | list[dict[str, int | float]]) \
            -> float | list[float]:
        """
        使用指定模型对输入数据进行推理，获取属性值，支持批量处理。
        :param prop_name: 属性名称
        :param input_data: 元素组成和热处理温度等参数
        :return: 属性值
        """
        if isinstance(input_data, dict):
            beta_output = self._infer_beta(prop_name, input_data)
            if beta_output is None:
                orig_output = self._infer(prop_name, input_data)
                corr_output = self._corr(prop_name, input_data, orig_output)
                output = corr_output
            else:
                output = beta_output
            return output
        else:
            beta_outputs = [self._infer_beta(prop_name, input_) for input_ in input_data]
            if any(beta_output is None for beta_output in beta_outputs):
                inputs = [input_ for input_, beta_output in zip(input_data, beta_outputs) if beta_output is None]
                orig_outputs = self._batch_infer(prop_name, inputs)
                corr_outputs = self._batch_corr(prop_name, inputs, orig_outputs)
                corr_iter = iter(corr_outputs)
                outputs = [beta_output if beta_output is not None else next(corr_iter) for beta_output in beta_outputs]
            else:
                outputs = beta_outputs
            return outputs

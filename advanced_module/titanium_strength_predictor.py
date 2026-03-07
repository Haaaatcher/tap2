# -*- coding: utf-8 -*-
"""
钛合金强度预测模块
功能：基于成分、微量元素、热加工工艺和热处理工艺，预测钛合金屈服强度
"""

import re
import math
import numpy as np
import pandas as pd
import pickle
import joblib
import os
# import sys
from importlib import resources


# ===================== 常量定义 =====================
# 柏氏矢量 (单位: m)
b_alphaTi = 2.95e-10  # α-Ti的柏氏矢量
b_betaTi = 2.86e-10  # β-Ti的柏氏矢量
alpha_alphaTi = 0.2
alpha_betaTi = 0.3
M_alphaTi = 1
M_betaTi = 2.8
G_alphaTi = 44 * 1e9  # 剪切模量，转为Pa
G_betaTi = 39 * 1e9
v = 0.3  # 泊松比

# 元素原子质量字典
ATOMIC_WEIGHTS = {
    'Pd': 106.42, 'Sn': 118.710, 'In': 114.818, 'Ni': 58.6934, 'Co': 58.933194,
    'Fe': 55.845, 'Cr': 51.9961, 'Cu': 63.546, 'Mn': 54.938044, 'Mg': 24.305,
    'Zr': 91.224, 'V': 50.9415, 'Mo': 95.95, 'W': 183.84, 'Si': 28.085,
    'Al': 26.981538, 'Ag': 107.8682, 'Ta': 180.94788, 'Nb': 92.90637, 'Ti': 47.867,
    'C': 12.011, 'H': 1.008, 'O': 15.999, 'N': 14.007, 'B': 10.811
}

# 固溶强化系数
strengthening_coefficient = {
    'Pd': 3023, 'Sn': 2303, 'In': 2207, 'Ni': 2090, 'Co': 2041,
    'Fe': 1715, 'Cr': 1665, 'Cu': 1650, 'Mn': 1485, 'Mg': 1276,
    'Zr': 1201, 'V': 879, 'Mo': 575, 'W': 574, 'Si': 445,
    'Al': 285, 'Ag': 181, 'Ta': 164, 'Nb': 71, 'Ti': 0
}

# 微量元素强化系数 (MPa/wt%)
trace_element_coefficients = {
    'C': 704.2,
    'H': 268,
    'O': 1219.86,
    'N': 2185.6,
    'B': -34
}


# ===================== 辅助函数 =====================
def get_resource_path(relative_path):
    """获取资源文件的绝对路径"""
    # try:
    #     base_path = sys._MEIPASS
    # except AttributeError:
    #     base_path = os.path.dirname(os.path.abspath(__file__))
    # return os.path.join(base_path, relative_path)
    return str(resources.files("advanced_module.resource").joinpath(relative_path))


def parse_composition(input_str):
    """解析合金成分字符串，返回元素和其质量百分比列表"""
    main_elem_match = re.match(r'^([A-Z][a-z]*)', input_str)
    if not main_elem_match:
        raise ValueError("主元素未识别")
    main_elem = main_elem_match.group(1)

    elements = []
    matches = re.findall(r'(\d+\.?\d*)\s*([A-Z][a-z]*)|([A-Z][a-z]*)\s*(\d+\.?\d*)', input_str)
    total_percent = 0.0

    for m in matches:
        if m[0] and m[1]:  # 数值在前
            elem, percent = m[1], float(m[0])
        elif m[2] and m[3]:  # 元素在前
            elem, percent = m[2], float(m[3])
        else:
            continue
        elements.append((elem, percent))
        total_percent += percent

    # 计算主元素百分比（余量）
    main_percent = 100.0 - total_percent
    elements.insert(0, (main_elem, main_percent))
    return elements


def calculate_atomic_percent(composition):
    """计算元素的原子百分比"""
    molar_ratios = []
    for elem, wt in composition:
        atomic_wt = ATOMIC_WEIGHTS.get(elem)
        if atomic_wt is None:
            raise ValueError(f"未知元素: {elem}")
        molar_ratios.append(wt / atomic_wt)

    total_molar = sum(molar_ratios)
    return [(rat / total_molar) * 100 for rat in molar_ratios]


def parse_heat_treatment(treatment_str):
    """
    解析热处理工艺字符串
    输入格式：温度(℃)/时间(min)/冷却方式(FC、AC或WQ)+...
    返回：(heat_temp, heat_time, cooling)元组，只取第一个热处理阶段
    """
    if not treatment_str or treatment_str == "/":
        return None, None, None

    stages = treatment_str.split('+')
    if not stages:
        return None, None, None

    first_stage = stages[0].strip()
    parts = first_stage.split('/')

    if len(parts) >= 3:
        try:
            heat_temp = float(parts[0])
            heat_time = float(parts[1])
            cooling = parts[2].strip().upper()
            return heat_temp, heat_time, cooling
        except:
            return None, None, None

    return None, None, None


def parse_processing(processing_str):
    """
    解析加工工艺字符串
    输入格式：温度(℃)/变形量(%)
    返回：(work_temp, deformation)元组
    """
    if not processing_str or processing_str == "/":
        return None, None

    parts = processing_str.split('/')

    if len(parts) == 2:
        work_temp_str = parts[0].strip()
        deformation_str = parts[1].strip()

        try:
            work_temp = float(work_temp_str) if work_temp_str else 27.0
            deformation = float(deformation_str)
            return work_temp, deformation
        except:
            return None, None

    return None, None


def calculate_mo_equivalent_ml(alloy_str):
    """计算钛合金Mo当量"""
    elements = {'Mo': 0, 'V': 0, 'Cr': 0, 'Al': 0, 'Zr': 0, 'B': 0, 'Ti': 0, 'Nb': 0, 'Fe': 0}
    if not alloy_str or not isinstance(alloy_str, str):
        return 5.0
    components = alloy_str.split('-')
    for comp in components:
        if not comp:
            continue
        num_str = ''
        elem_str = ''
        for char in comp:
            if char.isdigit() or char == '.':
                num_str += char
            else:
                elem_str += char
        if not elem_str or not num_str:
            continue
        try:
            content = float(num_str)
            if elem_str in elements:
                elements[elem_str] = content
        except:
            continue
    mo_eq = (elements['Mo'] + 0.67 * elements['V'] + 0.4 * elements['Cr'] +
             0.2 * elements['Nb'] + 0.1 * elements['Fe'] - 0.2 * elements['Al'] - 0.1 * elements['Zr'])
    return max(2.0, min(mo_eq, 25.0))


def parse_alloy_for_equivalents(composition):
    """
    解析合金成分字符串，计算Mo当量和Al当量
    用于ML误差补偿预测
    """
    elements = {
        'Al': 0, 'Mo': 0, 'V': 0, 'Cr': 0, 'Fe': 0, 'Si': 0,
        'Sn': 0, 'Zr': 0, 'Nb': 0, 'Ta': 0, 'W': 0
    }

    if not composition or not isinstance(composition, str):
        return 0.0, 0.0

    composition = re.sub(r'^[A-Z0-9]+\(', '', composition)
    composition = composition.replace(')', '')
    parts = composition.split('-')

    for part in parts:
        if part.strip() in ['Ti', '']:
            continue
        match = re.match(r'(\d+\.?\d*)([A-Z][a-z]?)', part.strip())
        if match:
            value, element = match.groups()
            if element in elements:
                elements[element] = float(value)

    mo_eq = (elements['Mo'] + elements['V'] / 1.5 + elements['Cr'] / 0.6 +
             elements['Fe'] / 0.5 + elements['Nb'] / 3.3 + elements['Ta'] / 4 + elements['W'] / 2)
    al_eq = (elements['Al'] + elements['Sn'] / 3 + elements['Zr'] / 6 + elements['Si'] / 10)

    return mo_eq, al_eq


# ===================== ML模型加载与预测 =====================
def load_ml_model():
    """加载训练好的机器学习模型"""
    model_path = get_resource_path("trained_model.pkl")
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None


def load_grain_models():
    """加载晶粒参数预测模型"""
    try:
        model_grain1_path = get_resource_path("model_grain1.joblib")
        model_grain2_path = get_resource_path("model_grain2.joblib")
        model_alpha_beta_path = get_resource_path("model_alpha_beta_multi.joblib")

        model_grain1, scaler_grain1 = joblib.load(model_grain1_path)
        model_grain2, scaler_grain2 = joblib.load(model_grain2_path)
        model_alpha_beta_multi, scaler_alpha_beta = joblib.load(model_alpha_beta_path)

        return {
            'model_grain1': model_grain1,
            'scaler_grain1': scaler_grain1,
            'model_grain2': model_grain2,
            'scaler_grain2': scaler_grain2,
            'model_alpha_beta_multi': model_alpha_beta_multi,
            'scaler_alpha_beta': scaler_alpha_beta
        }
    except Exception as e:
        print(f"晶粒模型加载失败: {e}")
        return None


def predict_error_compensation(alloy_str):
    """使用机器学习模型预测误差补偿值（σ_other）"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'error_prediction_model.pkl')
        scaler_path = os.path.join(script_dir, 'error_scaler.pkl')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return 0.0

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        mo_eq, al_eq = parse_alloy_for_equivalents(alloy_str)
        X = np.array([[mo_eq, al_eq]])
        X_scaled = scaler.transform(X)
        error_value = model.predict(X_scaled)[0]

        return float(error_value)
    except Exception as e:
        print(f"误差预测失败: {str(e)}")
        return 0.0


def predict_second_phase_params(model_data, alloy_str, cooling, heat_temp, heat_time, work_temp, deformation):
    """使用机器学习模型预测第二相参数"""
    if not model_data:
        return None, None

    try:
        mo_eq = calculate_mo_equivalent_ml(alloy_str)
        cooling_code_mapping = {'AC': 1, 'FC': 2, 'WQ': 3}
        cooling_code = cooling_code_mapping.get(cooling, 1)

        ml_input = np.array([[mo_eq, cooling_code, heat_temp, heat_time, work_temp, deformation]])
        scaler = model_data['scaler']
        ml_input_scaled = scaler.transform(ml_input)

        trained_models = model_data['trained_models']
        model_metrics = model_data['model_metrics']

        predictions = []
        for model in trained_models.values():
            predictions.append(model.predict(ml_input_scaled))
        predictions = np.array(predictions)

        weights = [max(0, (model_metrics[name]["粒子间距R²"] + model_metrics[name]["平均粒子半径R²"]) / 2)
                   for name in trained_models.keys()]
        weights = [w / sum(weights) if sum(weights) != 0 else 1 / len(weights) for w in weights]

        ml_pred = np.average(predictions, axis=0, weights=weights)[0]

        spacing = round(ml_pred[0], 3)
        radius = round(ml_pred[1], 3)

        spacing = np.clip(spacing, 0.001, 10.0)
        radius = np.clip(radius, 0.001, 5.0)

        return spacing, radius
    except Exception as e:
        print(f"预测失败: {e}")
        return None, None


def predict_volume_fractions(mo_eq, cooling, heat_temp, deformation):
    """预测α-Ti和β-Ti的体积分数"""
    try:
        if mo_eq < 2:
            alpha_base, beta_base = 0.95, 0.05
        elif mo_eq < 5:
            alpha_base, beta_base = 0.80, 0.20
        elif mo_eq < 10:
            alpha_base = 0.60 - (mo_eq - 5) * 0.05
            beta_base = 0.40 + (mo_eq - 5) * 0.05
        else:
            alpha_base, beta_base = 0.15, 0.85

        cooling_factors = {
            'WQ': {'alpha': -0.10, 'beta': 0.10},
            'AC': {'alpha': 0.0, 'beta': 0.0},
            'FC': {'alpha': 0.08, 'beta': -0.08}
        }
        cooling_factor = cooling_factors.get(cooling, {'alpha': 0.0, 'beta': 0.0})

        if heat_temp > 900:
            temp_factor_alpha, temp_factor_beta = -0.10, 0.10
        elif heat_temp > 800:
            temp_factor_alpha, temp_factor_beta = -0.05, 0.05
        elif heat_temp < 700:
            temp_factor_alpha, temp_factor_beta = 0.08, -0.08
        else:
            temp_factor_alpha, temp_factor_beta = 0.0, 0.0

        if deformation > 70:
            deform_factor_alpha, deform_factor_beta = 0.03, -0.03
        elif deformation > 40:
            deform_factor_alpha, deform_factor_beta = 0.01, -0.01
        else:
            deform_factor_alpha, deform_factor_beta = 0.0, 0.0

        alpha_volume = alpha_base + cooling_factor['alpha'] + temp_factor_alpha + deform_factor_alpha
        beta_volume = beta_base + cooling_factor['beta'] + temp_factor_beta + deform_factor_beta

        total = alpha_volume + beta_volume
        if total > 0:
            alpha_volume /= total
            beta_volume /= total
        else:
            alpha_volume, beta_volume = 0.5, 0.5

        alpha_volume = max(0.0, min(1.0, alpha_volume))
        beta_volume = max(0.0, min(1.0, beta_volume))

        alpha_pct = round(alpha_volume * 100, 1)
        beta_pct = round(beta_volume * 100, 1)

        return alpha_pct, beta_pct
    except Exception as e:
        print(f"体积分数预测失败: {e}")
        return 50.0, 50.0


def calculate_dislocation_density(deformation, cooling, heat_temp, heat_time, work_temp, mo_eq, phase='alpha'):
    """计算位错密度"""
    try:
        if phase == 'alpha':
            base, peak, k, x0 = 1e14, 3e15, 0.08, 60
        else:
            base, peak, k, x0 = 5e14, 8e15, 0.1, 55

        growth = peak * (1 - 1 / (1 + np.exp(k * (deformation - x0))))
        density = base + growth

        cooling_factors = {'AC': 1.0, 'FC': 0.85, 'WQ': 1.3}
        cooling_factor = cooling_factors.get(cooling, 1.0)
        density *= cooling_factor

        temp_factor = 1.0 - 0.2 * (1 - 1 / (1 + np.exp(0.01 * (heat_temp - 800))))
        density *= temp_factor

        time_factor = 1.0 - 0.15 * (1 - 1 / (1 + np.exp(0.01 * (heat_time - 180))))
        density *= time_factor

        working_temp_factor = 1.0 - 0.4 * (1 - 1 / (1 + np.exp(0.01 * (work_temp - 850))))
        density *= working_temp_factor

        if phase == 'beta':
            density *= (1 + 0.05 * min(mo_eq, 10))
        else:
            density *= (1 + 0.02 * min(mo_eq, 5))

        density = max(1e14, min(5e16, density))
        return density
    except Exception as e:
        print(f"位错密度计算失败: {e}")
        return 1e15 if phase == 'alpha' else 5e15


def predict_grain_parameters(grain_models, alloy_str, deformation, work_temp, heat_temp, heat_time, cooling):
    """使用晶粒模型预测晶粒参数"""
    if not grain_models:
        return None

    try:
        mo_eq = calculate_mo_equivalent_ml(alloy_str)
        cooling_mapping = {'AC': '空冷', 'FC': '炉冷', 'WQ': '水淬'}
        cooling_cn = cooling_mapping.get(cooling, '空冷')

        sample = pd.DataFrame({
            'Mo当量': [mo_eq],
            '热加工变形量': [deformation],
            '热加工温度': [work_temp],
            '热处理温度': [heat_temp],
            '热处理时间': [heat_time],
            '热处理冷却方式': [cooling_cn]
        })

        has_grain2 = 1 if mo_eq > 0 else 0
        sample['has_晶粒2'] = has_grain2

        model_grain1 = grain_models['model_grain1']
        scaler_grain1 = grain_models['scaler_grain1']
        model_grain2 = grain_models['model_grain2']
        scaler_grain2 = grain_models['scaler_grain2']

        pred1 = scaler_grain1.inverse_transform(model_grain1.predict(sample))
        grain1_size = round(pred1[0][0], 3) if pred1.shape[1] > 0 else 0.0
        grain1_volume = round(pred1[0][1], 3) if pred1.shape[1] > 1 else 0.0
        grain1_hall_petch = round(pred1[0][2], 3) if pred1.shape[1] > 2 else 0.0

        if has_grain2 == 1:
            pred2 = scaler_grain2.inverse_transform(model_grain2.predict(sample))
            grain2_size = round(pred2[0][0], 3) if pred2.shape[1] > 0 else 0.0
            grain2_volume = round(pred2[0][1], 3) if pred2.shape[1] > 1 else 0.0
            grain2_hall_petch = round(pred2[0][2], 3) if pred2.shape[1] > 2 else 0.0
        else:
            grain2_size = grain2_volume = grain2_hall_petch = 0.0

        if grain1_volume > 0 and grain2_volume > 0:
            total_volume = grain1_volume + grain2_volume
            if total_volume > 0:
                grain1_volume /= total_volume
                grain2_volume /= total_volume

        return {
            'grain1_size': grain1_size,
            'grain1_volume': round(grain1_volume * 100, 1),
            'grain1_hall_petch': grain1_hall_petch,
            'grain2_size': grain2_size,
            'grain2_volume': round(grain2_volume * 100, 1),
            'grain2_hall_petch': grain2_hall_petch
        }
    except Exception as e:
        print(f"晶粒参数预测失败: {e}")
        return None


# ===================== 强化机制计算 =====================
def calculate_solid_solution_strengthening(composition, trace_elements=None):
    """
    计算固溶强化效果

    参数:
        composition: 合金成分字符串，如 "Ti-6Al-4V"
        trace_elements: 微量元素字典，如 {'C': 0.05, 'O': 0.15}，单位wt%

    返回:
        sigma_ss: 固溶强化贡献 (MPa)
    """
    # 解析合金成分
    elements = parse_composition(composition)
    atomic_percents = calculate_atomic_percent(elements)

    # 计算主要元素的固溶强化
    sigma = 0
    for (elem, wt), at in zip(elements, atomic_percents):
        atomic_B = strengthening_coefficient.get(elem, 0)
        sigma += ((atomic_B ** 1.5) * at * 0.01)

    sigma_ss = round(sigma ** (2 / 3), 1)

    # 如果有微量元素，计算其影响（参考值，不计入总强度）
    trace_effect = 0
    if trace_elements:
        for elem, content in trace_elements.items():
            coef = trace_element_coefficients.get(elem, 0)
            trace_effect += coef * content

    return sigma_ss, trace_effect


def calculate_dislocation_strengthening(alpha_volume_pct, alpha_density, beta_volume_pct, beta_density):
    """
    计算位错强化效果

    参数:
        alpha_volume_pct: α-Ti体积分数 (%)
        alpha_density: α-Ti位错密度 (/m²)
        beta_volume_pct: β-Ti体积分数 (%)
        beta_density: β-Ti位错密度 (/m²)

    返回:
        sigma_dis: 总位错强化贡献 (MPa)
    """
    # 转换百分比为小数
    f_alpha = alpha_volume_pct * 0.01
    f_beta = beta_volume_pct * 0.01

    # 计算α-Ti位错强化
    sigma_rho_alpha = f_alpha * alpha_alphaTi * G_alphaTi * M_alphaTi * b_alphaTi * math.sqrt(alpha_density)
    sigma_rho_alpha_MPa = round(sigma_rho_alpha / 1e6, 1)

    # 计算β-Ti位错强化
    sigma_rho_beta = f_beta * alpha_betaTi * G_betaTi * M_betaTi * b_betaTi * math.sqrt(beta_density)
    sigma_rho_beta_MPa = round(sigma_rho_beta / 1e6, 1)

    # 总位错强化
    sigma_dis = round(sigma_rho_alpha_MPa + sigma_rho_beta_MPa, 1)
    return sigma_dis


def calculate_second_phase_strengthening(phase1_params, phase2_params=None):
    """
    计算第二相强化效果

    参数:
        phase1_params: 第二相1参数字典 {'f': 体积分数(%), 'lambda': 间距(μm), 'r': 半径(μm), 'h': 长高比}
        phase2_params: 第二相2参数字典（可选）

    返回:
        sigma_sp: 第二相强化贡献 (MPa)
    """
    const_factor = (0.4 * G_alphaTi * b_alphaTi) / (math.pi * math.sqrt(1 - v))

    def calculate_phase(params):
        f = params['f'] * 0.01  # 百分比转小数
        lambda_val = params['lambda'] * 1e-6  # μm转m
        r = params['r'] * 1e-6  # μm转m
        h = params.get('h', 1.0)

        if lambda_val == 0:
            lambda_val = 9999999
        if h == 0:
            h = 1
        if r == 0:
            r = 999999

        K = (h ** (1 / 6)) * (((2 + h ** 2) / 3) ** (-0.25))
        sigma = M_alphaTi * const_factor * math.log(2 * r / b_alphaTi) / lambda_val
        sigma *= K * f
        return sigma / 1e6  # 转MPa

    sigma_1 = calculate_phase(phase1_params)
    sigma_2 = calculate_phase(phase2_params) if phase2_params else 0

    sigma_sp = round(sigma_1 + sigma_2, 1)
    return sigma_sp


def calculate_hall_petch_strengthening(grain_data):
    """
    计算Hall-Petch强化效果

    参数:
        grain_data: 晶粒参数列表，每个元素为字典 {'size': 尺寸(μm), 'fraction': 体积分数(%), 'hp_coeff': HP系数(MPa·μm^0.5)}

    返回:
        sigma_hp: Hall-Petch强化贡献 (MPa)
    """
    total_hp = 0
    for grain in grain_data:
        size = grain['size']
        fraction = grain['fraction'] / 100  # 转小数
        hp_coeff = grain['hp_coeff']

        hp_value = hp_coeff / math.sqrt(size)
        weighted_value = fraction * hp_value
        total_hp += weighted_value

    sigma_hp = round(total_hp, 1)
    return sigma_hp


# ===================== 主预测函数 =====================
def predict_titanium_yield_strength(
    composition,
    processing_params,
    heat_treatment_params,
    trace_elements=None,
    model_dir=None
):
    """
    钛合金屈服强度预测主函数

    参数:
        composition: 合金成分字符串，如 "Ti-6Al-4V"
        processing_params: 加工工艺字符串或字典
            - 字符串格式: "温度/变形量"，如 "860/80"
            - 字典格式: {'temperature': 860, 'deformation': 80}
        heat_treatment_params: 热处理工艺字符串或字典
            - 字符串格式: "温度/时间/冷却方式"，如 "760/30/AC"
            - 字典格式: {'temperature': 760, 'time': 30, 'cooling': 'AC'}
        trace_elements: 微量元素字典（可选），如 {'C': 0.05, 'O': 0.15}，单位wt%
        model_dir: 模型文件目录（可选）

    返回:
        result: 结果字典，包含:
            - 'yield_strength': 预测的屈服强度 (MPa)
            - 'contributions': 各强化机制的贡献字典
            - 'microstructure': 预测的微观组织参数字典
    """

    # 解析工艺参数
    if isinstance(processing_params, str):
        work_temp, deformation = parse_processing(processing_params)
    else:
        work_temp = processing_params.get('temperature', 27.0)
        deformation = processing_params.get('deformation', 0.0)

    if isinstance(heat_treatment_params, str):
        heat_temp, heat_time, cooling = parse_heat_treatment(heat_treatment_params)
    else:
        heat_temp = heat_treatment_params.get('temperature')
        heat_time = heat_treatment_params.get('time')
        cooling = heat_treatment_params.get('cooling')

    if work_temp is None or deformation is None:
        raise ValueError("加工工艺参数格式错误")
    if heat_temp is None or heat_time is None or cooling is None:
        raise ValueError("热处理工艺参数格式错误")

    # 加载ML模型
    ml_model = load_ml_model()
    grain_models = load_grain_models()

    # 1. 固溶强化
    sigma_ss, trace_effect = calculate_solid_solution_strengthening(composition, trace_elements)

    # 2. 预测微观组织参数
    mo_eq = calculate_mo_equivalent_ml(composition)

    # 预测第二相参数
    spacing, radius = predict_second_phase_params(
        ml_model, composition, cooling, heat_temp, heat_time, work_temp, deformation
    )

    # 预测相体积分数
    alpha_volume_pct, beta_volume_pct = predict_volume_fractions(
        mo_eq, cooling, heat_temp, deformation
    )

    # 预测位错密度
    alpha_density = calculate_dislocation_density(
        deformation, cooling, heat_temp, heat_time, work_temp, mo_eq, 'alpha'
    )
    beta_density = calculate_dislocation_density(
        deformation, cooling, heat_temp, heat_time, work_temp, mo_eq, 'beta'
    )

    # 预测晶粒参数
    grain_params = predict_grain_parameters(
        grain_models, composition, deformation, work_temp, heat_temp, heat_time, cooling
    )

    # 3. 位错强化
    sigma_dis = calculate_dislocation_strengthening(
        alpha_volume_pct, alpha_density, beta_volume_pct, beta_density
    )

    # 4. 第二相强化
    phase1_params = {
        'f': 100,  # 默认100%
        'lambda': spacing if spacing else 1.0,
        'r': radius if radius else 0.1,
        'h': 1.0
    }
    sigma_sp = calculate_second_phase_strengthening(phase1_params)

    # 5. Hall-Petch强化
    if grain_params:
        grain_data = []
        if grain_params['grain1_size'] > 0:
            grain_data.append({
                'size': grain_params['grain1_size'],
                'fraction': grain_params['grain1_volume'],
                'hp_coeff': grain_params['grain1_hall_petch']
            })
        if grain_params['grain2_size'] > 0:
            grain_data.append({
                'size': grain_params['grain2_size'],
                'fraction': grain_params['grain2_volume'],
                'hp_coeff': grain_params['grain2_hall_petch']
            })
        sigma_hp = calculate_hall_petch_strengthening(grain_data) if grain_data else 0.0
    else:
        sigma_hp = 0.0

    # 6. ML误差补偿
    sigma_ml = predict_error_compensation(composition)

    # 7. 基础强度
    sigma_0 = 180.0  # MPa

    # 8. 计算总强度
    yield_strength = sigma_0 + sigma_ss + sigma_dis + sigma_sp + sigma_hp + sigma_ml
    yield_strength = round(yield_strength, 1)

    # 构建返回结果
    result = {
        'yield_strength': yield_strength,
        'contributions': {
            'base_strength': sigma_0,
            'solid_solution': sigma_ss,
            'dislocation': sigma_dis,
            'second_phase': sigma_sp,
            'hall_petch': sigma_hp,
            'ml_compensation': sigma_ml,
            'trace_elements_effect': trace_effect if trace_elements else 0.0
        },
        'microstructure': {
            'mo_equivalent': mo_eq,
            'alpha_volume_pct': alpha_volume_pct,
            'beta_volume_pct': beta_volume_pct,
            'alpha_dislocation_density': alpha_density,
            'beta_dislocation_density': beta_density,
            'second_phase_spacing_um': spacing,
            'second_phase_radius_um': radius,
            'grain_parameters': grain_params
        }
    }

    return result


# ===================== 示例使用 =====================
if __name__ == "__main__":
    # 示例1：基本使用
    print("=" * 60)
    print("示例1：基本使用")
    print("=" * 60)

    result = predict_titanium_yield_strength(
        composition="Ti-6Al-4V",
        processing_params="860/80",
        heat_treatment_params="760/30/AC"
    )

    print(f"\n合金成分: Ti-6Al-4V")
    print(f"加工工艺: 860℃/80%变形")
    print(f"热处理工艺: 760℃/30min/空冷")
    print(f"\n预测屈服强度: {result['yield_strength']} MPa")
    print("\n强化机制贡献:")
    for mechanism, value in result['contributions'].items():
        print(f"  {mechanism}: {value:.1f} MPa")

    # 示例2：包含微量元素
    print("\n" + "=" * 60)
    print("示例2：包含微量元素")
    print("=" * 60)

    result2 = predict_titanium_yield_strength(
        composition="Ti-6Al-4V",
        processing_params={'temperature': 900, 'deformation': 70},
        heat_treatment_params={'temperature': 800, 'time': 60, 'cooling': 'WQ'},
        trace_elements={'C': 0.05, 'O': 0.15, 'N': 0.03}
    )

    print(f"\n合金成分: Ti-6Al-4V")
    print(f"微量元素: C=0.05%, O=0.15%, N=0.03%")
    print(f"加工工艺: 900℃/70%变形")
    print(f"热处理工艺: 800℃/60min/水淬")
    print(f"\n预测屈服强度: {result2['yield_strength']} MPa")
    print("\n强化机制贡献:")
    for mechanism, value in result2['contributions'].items():
        print(f"  {mechanism}: {value:.1f} MPa")

    print("\n微观组织参数:")
    print(f"  Mo当量: {result2['microstructure']['mo_equivalent']:.2f}")
    print(f"  α-Ti体积分数: {result2['microstructure']['alpha_volume_pct']}%")
    print(f"  β-Ti体积分数: {result2['microstructure']['beta_volume_pct']}%")
    print(f"  第二相粒子间距: {result2['microstructure']['second_phase_spacing_um']} μm")

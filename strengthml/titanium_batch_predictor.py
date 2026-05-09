# -*- coding: utf-8 -*-
"""
钛合金批量强度预测模块（增强版）
功能：基于成分、微量元素、热加工工艺和热处理工艺，预测钛合金屈服强度
支持：时效强化、多级热处理、高V含量材料特殊处理
"""

import re
import math
import numpy as np
import pandas as pd
import pickle
import joblib
import os
import sys


# ===================== 常量定义 =====================
# 柏氏矢量和材料参数
b_alphaTi = 2.95e-10  # α-Ti的柏氏矢量 (m)
b_betaTi = 2.86e-10  # β-Ti的柏氏矢量 (m)
alpha_alphaTi = 0.2
alpha_betaTi = 0.3
M_alphaTi = 1  # α-Ti的Taylor因子
M_betaTi = 2.8  # β-Ti的Taylor因子
G_alphaTi = 44 * 1e9  # α-Ti剪切模量 (Pa)
G_betaTi = 39 * 1e9  # β-Ti剪切模量 (Pa)
v = 0.3  # 泊松比

# 元素原子质量字典
ATOMIC_WEIGHTS = {
    'Pd': 106.42, 'Sn': 118.710, 'In': 114.818, 'Ni': 58.6934, 'Co': 58.933194,
    'Fe': 55.845, 'Cr': 51.9961, 'Cu': 63.546, 'Mn': 54.938044, 'Mg': 24.305,
    'Zr': 91.224, 'V': 50.9415, 'Mo': 95.95, 'W': 183.84, 'Si': 28.085,
    'Al': 26.981538, 'Ag': 107.8682, 'Ta': 180.94788, 'Nb': 92.90637, 'Ti': 47.867,
    'C': 12.011, 'H': 1.008, 'O': 15.999, 'N': 14.007, 'B': 10.811
}

# 固溶强化系数（主要元素）
strengthening_coefficient = {
    'Al': 23.0, 'Mo': 29.0, 'V': 10.0, 'Cr': 17.0,
    'Fe': 12.0, 'Zr': 8.0, 'Sn': 6.0, 'Si': 5.0,
    'Nb': 15.0, 'Ta': 18.0, 'W': 25.0, 'Cu': 9.0
}

# 微量元素强化系数 (MPa/wt%)
trace_element_coefficients = {
    'C': 704.2,
    'H': 268.0,
    'O': 1219.86,
    'N': 2185.6,
    'B': -34.0
}


# ===================== 辅助函数 =====================
def get_resource_path(relative_path):
    """获取资源文件的绝对路径"""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def parse_alloy_composition(comp_str):
    """
    解析合金成分字符串，返回元素及其质量百分比字典
    例如: "Ti-15.6Mo" -> {'Mo': 15.6}
         "Ti-6Al-4V" -> {'Al': 6, 'V': 4}

    参数:
        comp_str: str, 合金成分字符串

    返回:
        dict: 元素及其质量百分比
    """
    elements = {}
    if not comp_str or not isinstance(comp_str, str):
        return elements

    # 移除Ti-前缀
    comp_str = comp_str.strip()
    if comp_str.startswith('Ti-'):
        comp_str = comp_str[3:]

    # 按"-"分割
    parts = comp_str.split('-')

    for part in parts:
        if not part or part.strip() == 'Ti':
            continue

        # 匹配数字+元素 或 元素+数字
        match = re.match(r'(\d+\.?\d*)([A-Z][a-z]?)', part.strip())
        if match:
            value, element = match.groups()
            elements[element] = float(value)

    return elements


def parse_alloy_for_equivalents(composition):
    """
    解析合金成分字符串，计算Mo当量和Al当量

    Mo当量 = Mo + V/1.5 + Cr/0.6 + Fe/0.5 + Nb/3.3 + Ta/4 + W/2
    Al当量 = Al + Sn/3 + Zr/6 + Si/10

    参数:
        composition: str, 合金成分字符串，如 "Ti-6Al-4V"

    返回:
        tuple: (mo_eq, al_eq)
    """
    elements = {
        'Al': 0, 'Mo': 0, 'V': 0, 'Cr': 0, 'Fe': 0, 'Si': 0,
        'Sn': 0, 'Zr': 0, 'Nb': 0, 'Ta': 0, 'W': 0
    }

    if not composition or not isinstance(composition, str):
        return 0.0, 0.0

    # 去除前缀和括号
    composition = re.sub(r'^[A-Z0-9]+\(', '', composition)
    composition = composition.replace(')', '')

    # 按"-"分割
    parts = composition.split('-')

    for part in parts:
        if part.strip() in ['Ti', '']:
            continue

        # 匹配数字+元素
        match = re.match(r'(\d+\.?\d*)([A-Z][a-z]?)', part.strip())
        if match:
            value, element = match.groups()
            if element in elements:
                elements[element] = float(value)

    # 计算Mo当量
    mo_eq = (elements['Mo'] +
             elements['V']/1.5 +
             elements['Cr']/0.6 +
             elements['Fe']/0.5 +
             elements['Nb']/3.3 +
             elements['Ta']/4 +
             elements['W']/2)

    # 计算Al当量
    al_eq = (elements['Al'] +
             elements['Sn']/3 +
             elements['Zr']/6 +
             elements['Si']/10)

    return mo_eq, al_eq


def calculate_mo_equivalent_ml(alloy_str):
    """
    计算钛合金Mo当量（用于ML模型预测）

    Mo当量 = Mo + 0.67*V + 0.4*Cr + 0.2*Nb + 0.1*Fe - 0.2*Al - 0.1*Zr

    参数:
        alloy_str: str, 合金成分字符串，如 "Ti-6Al-4V"

    返回:
        float: Mo当量（限制在2.0~25.0范围内）
    """
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

    mo_eq = (elements['Mo'] + 0.67*elements['V'] + 0.4*elements['Cr'] +
             0.2*elements['Nb'] + 0.1*elements['Fe'] - 0.2*elements['Al'] - 0.1*elements['Zr'])
    return max(2.0, min(mo_eq, 25.0))


def check_high_v_content(alloy_composition, threshold=13.0):
    """
    检查合金中V含量是否达到或超过阈值

    参数:
        alloy_composition: str, 合金成分字符串
        threshold: float, V含量阈值（默认13%）

    返回:
        bool: V含量是否>=阈值
    """
    try:
        elements = parse_alloy_composition(alloy_composition)
        v_content = elements.get('V', 0.0)
        return v_content >= threshold
    except:
        return False


def parse_processing(processing_str):
    """
    解析加工工艺字符串

    参数:
        processing_str: str, 格式为 "温度/变形量"，如 "860/80"

    返回:
        tuple: (work_temp, deformation)
    """
    if not processing_str or processing_str == "/":
        return None, None

    parts = processing_str.split('/')

    if len(parts) == 2:
        work_temp_str = parts[0].strip()
        deformation_str = parts[1].strip()

        try:
            work_temp = float(work_temp_str) if work_temp_str else 27.0  # 冷变形默认室温27℃
            deformation = float(deformation_str)
            return work_temp, deformation
        except:
            return None, None

    return None, None


def parse_heat_treatment(treatment_str):
    """
    解析热处理工艺字符串（仅第一阶段）

    参数:
        treatment_str: str, 格式为 "温度/时间/冷却方式+..."，如 "760/30/AC+500/120/AC"

    返回:
        tuple: (heat_temp, heat_time, cooling)，只返回第一个阶段
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


def is_aging_treatment(temp, time, cooling):
    """
    判断是否为时效工艺

    判断标准：温度 < 650℃ 且 时间 > 30min

    参数:
        temp: float, 温度(℃)
        time: float, 时间(min)
        cooling: str, 冷却方式

    返回:
        bool: 是否为时效工艺
    """
    if temp is None or time is None:
        return False

    # 时效工艺判断标准
    if temp < 650 and time > 30:
        return True

    return False


def parse_all_heat_treatment_stages(treatment_str):
    """
    解析所有热处理工艺阶段

    参数:
        treatment_str: str, 多级热处理工艺字符串

    返回:
        list: 热处理阶段列表，每个元素为字典 {'temp', 'time', 'cooling'}
    """
    if not treatment_str or treatment_str == "/":
        return []

    stages = []
    stage_strs = treatment_str.split('+')

    for stage_str in stage_strs:
        parts = stage_str.strip().split('/')
        if len(parts) >= 3:
            try:
                temp = float(parts[0])
                time = float(parts[1])
                cooling = parts[2].strip().upper()
                stages.append({'temp': temp, 'time': time, 'cooling': cooling})
            except:
                continue

    return stages


def merge_heat_treatment_stages(stages):
    """
    合并多个热处理阶段为单一阶段

    合并策略：
    - 温度：取最高温度
    - 时间：累加所有时间
    - 冷却方式：取最后一个阶段的冷却方式

    参数:
        stages: list, 热处理阶段列表

    返回:
        dict: 合并后的热处理阶段
    """
    if not stages:
        return None

    if len(stages) == 1:
        return stages[0]

    # 合并策略
    max_temp = max([s['temp'] for s in stages])
    total_time = sum([s['time'] for s in stages])
    last_cooling = stages[-1]['cooling']

    return {
        'temp': max_temp,
        'time': total_time,
        'cooling': last_cooling
    }


def standardize_heat_treatment(treatment_str):
    """
    标准化多级热处理工艺

    将多个热处理阶段分为：
    1. 固溶/退火工艺（solution_stage）：非时效工艺的合并
    2. 时效工艺（aging_stage）：最后一个时效工艺

    参数:
        treatment_str: str, 多级热处理工艺字符串

    返回:
        tuple: (solution_stage, aging_stage)，每个为字典或None
    """
    stages = parse_all_heat_treatment_stages(treatment_str)

    if not stages:
        return None, None

    # 分离时效和非时效工艺
    non_aging_stages = []
    aging_stages = []

    for stage in stages:
        if is_aging_treatment(stage['temp'], stage['time'], stage['cooling']):
            aging_stages.append(stage)
        else:
            non_aging_stages.append(stage)

    # 合并非时效工艺
    solution_stage = merge_heat_treatment_stages(non_aging_stages) if non_aging_stages else None

    # 只取最后一个时效工艺
    aging_stage = aging_stages[-1] if aging_stages else None

    return solution_stage, aging_stage


# ===================== ML模型加载 =====================
def load_ml_model():
    """加载第二相参数预测模型"""
    model_path = get_resource_path("trained_model.pkl")
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        print(f"[WARNING] 第二相模型加载失败: {e}")
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
        print(f"[WARNING] 晶粒模型加载失败: {e}")
        return None


def load_aging_models():
    """
    加载时效强度预测模型

    返回:
        dict: 包含model, scaler, info的字典，如果加载失败返回None
    """
    try:
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, "best_model.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        info_path = os.path.join(model_dir, "model_info.pkl")

        if not all([os.path.exists(p) for p in [model_path, scaler_path, info_path]]):
            print("[WARNING] 时效强度模型文件未找到")
            return None

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(info_path, 'rb') as f:
            model_info = pickle.load(f)

        print(f"[SUCCESS] 时效强度模型加载成功: {model_info.get('best_model_name', 'Unknown')}")
        return {
            'model': model,
            'scaler': scaler,
            'info': model_info
        }
    except Exception as e:
        print(f"[ERROR] 加载时效强度模型失败: {e}")
        return None


def load_non_aging_correction_models():
    """
    加载非时效工艺补项模型

    返回:
        dict: 包含model, scaler, info的字典，如果加载失败返回None
    """
    try:
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, "non_aging_correction_model.pkl")
        scaler_path = os.path.join(model_dir, "non_aging_correction_scaler.pkl")
        info_path = os.path.join(model_dir, "non_aging_correction_info.pkl")

        if not all([os.path.exists(p) for p in [model_path, scaler_path, info_path]]):
            print("[WARNING] 非时效补项模型文件未找到")
            return None

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(info_path, 'rb') as f:
            model_info = pickle.load(f)

        print(f"[SUCCESS] 非时效补项模型加载成功: {model_info.get('best_model_name', 'Unknown')}")
        return {
            'model': model,
            'scaler': scaler,
            'info': model_info
        }
    except Exception as e:
        print(f"[ERROR] 加载非时效补项模型失败: {e}")
        return None


def load_high_vcr_correction_models():
    """
    加载高V含量材料补项模型（V≥13%时使用）

    返回:
        dict: 包含model, scaler, info的字典，如果加载失败返回None
    """
    try:
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, "high_vcr_correction_model.pkl")
        scaler_path = os.path.join(model_dir, "high_vcr_correction_scaler.pkl")
        info_path = os.path.join(model_dir, "high_vcr_correction_info.pkl")

        if not all([os.path.exists(p) for p in [model_path, scaler_path, info_path]]):
            print("[WARNING] 高VCr补项模型文件未找到")
            return None

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(info_path, 'rb') as f:
            model_info = pickle.load(f)

        print(f"[SUCCESS] 高VCr补项模型加载成功: {model_info.get('best_model_name', 'Unknown')}")
        print(f"[INFO] V含量阈值: {model_info.get('v_threshold', 13.0)}%")
        return {
            'model': model,
            'scaler': scaler,
            'info': model_info
        }
    except Exception as e:
        print(f"[ERROR] 加载高VCr补项模型失败: {e}")
        return None


# ===================== ML预测函数 =====================
def predict_second_phase_params(model_data, alloy_str, cooling, heat_temp, heat_time, work_temp, deformation):
    """
    使用机器学习模型预测第二相参数

    参数:
        model_data: dict, ML模型数据
        alloy_str: str, 合金成分
        cooling: str, 冷却方式
        heat_temp: float, 热处理温度(℃)
        heat_time: float, 热处理时间(min)
        work_temp: float, 加工温度(℃)
        deformation: float, 变形量(%)

    返回:
        tuple: (spacing, radius)，粒子间距(μm)和半径(μm)
    """
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
        print(f"[ERROR] 第二相参数预测失败: {e}")
        return None, None


def predict_volume_fractions(mo_eq, cooling, heat_temp, deformation):
    """
    预测α-Ti和β-Ti的体积分数

    参数:
        mo_eq: float, Mo当量
        cooling: str, 冷却方式
        heat_temp: float, 热处理温度(℃)
        deformation: float, 变形量(%)

    返回:
        tuple: (alpha_pct, beta_pct)，体积分数(%)
    """
    try:
        # 基础体积分数（根据Mo当量）
        if mo_eq < 2:
            alpha_base, beta_base = 0.95, 0.05
        elif mo_eq < 5:
            alpha_base, beta_base = 0.80, 0.20
        elif mo_eq < 10:
            alpha_base = 0.60 - (mo_eq - 5) * 0.05
            beta_base = 0.40 + (mo_eq - 5) * 0.05
        else:
            alpha_base, beta_base = 0.15, 0.85

        # 冷却方式影响
        cooling_factors = {
            'WQ': {'alpha': -0.10, 'beta': 0.10},
            'AC': {'alpha': 0.0, 'beta': 0.0},
            'FC': {'alpha': 0.08, 'beta': -0.08}
        }
        cooling_factor = cooling_factors.get(cooling, {'alpha': 0.0, 'beta': 0.0})

        # 温度影响
        if heat_temp > 900:
            temp_factor_alpha, temp_factor_beta = -0.10, 0.10
        elif heat_temp > 800:
            temp_factor_alpha, temp_factor_beta = -0.05, 0.05
        elif heat_temp < 700:
            temp_factor_alpha, temp_factor_beta = 0.08, -0.08
        else:
            temp_factor_alpha, temp_factor_beta = 0.0, 0.0

        # 变形量影响
        if deformation > 70:
            deform_factor_alpha, deform_factor_beta = 0.03, -0.03
        elif deformation > 40:
            deform_factor_alpha, deform_factor_beta = 0.01, -0.01
        else:
            deform_factor_alpha, deform_factor_beta = 0.0, 0.0

        # 计算总体积分数
        alpha_volume = alpha_base + cooling_factor['alpha'] + temp_factor_alpha + deform_factor_alpha
        beta_volume = beta_base + cooling_factor['beta'] + temp_factor_beta + deform_factor_beta

        # 归一化
        total = alpha_volume + beta_volume
        if total > 0:
            alpha_volume /= total
            beta_volume /= total
        else:
            alpha_volume, beta_volume = 0.5, 0.5

        # 限制范围
        alpha_volume = max(0.0, min(1.0, alpha_volume))
        beta_volume = max(0.0, min(1.0, beta_volume))

        alpha_pct = round(alpha_volume * 100, 1)
        beta_pct = round(beta_volume * 100, 1)

        return alpha_pct, beta_pct
    except Exception as e:
        print(f"[ERROR] 体积分数预测失败: {e}")
        return 50.0, 50.0


def calculate_dislocation_density(deformation, cooling, heat_temp, heat_time, work_temp, mo_eq, phase='alpha'):
    """
    计算位错密度

    参数:
        deformation: float, 变形量(%)
        cooling: str, 冷却方式
        heat_temp: float, 热处理温度(℃)
        heat_time: float, 热处理时间(min)
        work_temp: float, 加工温度(℃)
        mo_eq: float, Mo当量
        phase: str, 'alpha' 或 'beta'

    返回:
        float: 位错密度(/m²)
    """
    try:
        # 基础参数
        if phase == 'alpha':
            base, peak, k, x0 = 1e14, 3e15, 0.08, 60
        else:
            base, peak, k, x0 = 5e14, 8e15, 0.1, 55

        # S型曲线增长
        growth = peak * (1 - 1 / (1 + np.exp(k * (deformation - x0))))
        density = base + growth

        # 冷却方式影响
        cooling_factors = {'AC': 1.0, 'FC': 0.85, 'WQ': 1.3}
        cooling_factor = cooling_factors.get(cooling, 1.0)
        density *= cooling_factor

        # 热处理温度影响
        temp_factor = 1.0 - 0.2 * (1 - 1 / (1 + np.exp(0.01 * (heat_temp - 800))))
        density *= temp_factor

        # 热处理时间影响
        time_factor = 1.0 - 0.15 * (1 - 1 / (1 + np.exp(0.01 * (heat_time - 180))))
        density *= time_factor

        # 加工温度影响
        working_temp_factor = 1.0 - 0.4 * (1 - 1 / (1 + np.exp(0.01 * (work_temp - 850))))
        density *= working_temp_factor

        # Mo当量影响
        if phase == 'beta':
            density *= (1 + 0.05 * min(mo_eq, 10))
        else:
            density *= (1 + 0.02 * min(mo_eq, 5))

        # 限制范围
        density = max(1e14, min(5e16, density))
        return density
    except Exception as e:
        print(f"[ERROR] 位错密度计算失败: {e}")
        return 1e15 if phase == 'alpha' else 5e15


def predict_grain_parameters(grain_models, alloy_str, deformation, work_temp, heat_temp, heat_time, cooling):
    """
    使用晶粒模型预测晶粒参数

    参数:
        grain_models: dict, 晶粒模型
        alloy_str: str, 合金成分
        deformation: float, 变形量(%)
        work_temp: float, 加工温度(℃)
        heat_temp: float, 热处理温度(℃)
        heat_time: float, 热处理时间(min)
        cooling: str, 冷却方式

    返回:
        dict: 晶粒参数字典
    """
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

        # 预测晶粒1
        pred1 = scaler_grain1.inverse_transform(model_grain1.predict(sample))
        grain1_size = round(pred1[0][0], 3) if pred1.shape[1] > 0 else 0.0
        grain1_volume = round(pred1[0][1], 3) if pred1.shape[1] > 1 else 0.0
        grain1_hall_petch = round(pred1[0][2], 3) if pred1.shape[1] > 2 else 0.0

        # 预测晶粒2
        if has_grain2 == 1:
            pred2 = scaler_grain2.inverse_transform(model_grain2.predict(sample))
            grain2_size = round(pred2[0][0], 3) if pred2.shape[1] > 0 else 0.0
            grain2_volume = round(pred2[0][1], 3) if pred2.shape[1] > 1 else 0.0
            grain2_hall_petch = round(pred2[0][2], 3) if pred2.shape[1] > 2 else 0.0
        else:
            grain2_size = grain2_volume = grain2_hall_petch = 0.0

        # 归一化体积分数
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
        print(f"[ERROR] 晶粒参数预测失败: {e}")
        return None


def calculate_aging_strength_increment(aging_models, alloy_composition, aging_temp, aging_time):
    """
    计算时效强度增量

    参数:
        aging_models: dict, 包含model, scaler, info的字典
        alloy_composition: str, 合金成分字符串，如"Ti-6Al-4V"
        aging_temp: float, 时效温度(℃)
        aging_time: float, 时效时间(min)

    返回:
        float: 时效强度增量(MPa)，如果计算失败返回0
    """
    if aging_models is None:
        return 0.0

    try:
        # 解析合金成分，计算Mo当量和Al当量
        mo_eq, al_eq = parse_alloy_for_equivalents(alloy_composition)

        # 根据model_info获取特征列
        feature_cols = aging_models['info'].get('feature_cols', ['Mo当量', 'Al当量', '温度', '时间'])

        # 构建特征数据
        feature_data = pd.DataFrame({
            'Mo当量': [mo_eq],
            'Al当量': [al_eq],
            '温度': [aging_temp],
            '时间': [aging_time]
        })

        # 确保特征列顺序与训练时一致
        feature_data = feature_data[feature_cols]

        # 标准化
        X_scaled = aging_models['scaler'].transform(feature_data)

        # 预测
        aging_increment = aging_models['model'].predict(X_scaled)[0]

        print(f"[INFO] 时效强度增量: Mo_eq={mo_eq:.2f}, Al_eq={al_eq:.2f}, T={aging_temp}℃, t={aging_time}min -> Δσ={aging_increment:.1f} MPa")

        return aging_increment

    except Exception as e:
        print(f"[ERROR] 时效强度增量计算失败: {e}")
        return 0.0


def calculate_non_aging_correction(non_aging_models, alloy_composition):
    """
    计算非时效工艺的强度补项

    参数:
        non_aging_models: dict, 包含model, scaler, info的字典
        alloy_composition: str, 合金成分字符串，如"Ti-6Al-4V"

    返回:
        float: 强度补项(MPa)，如果计算失败返回0
    """
    if non_aging_models is None:
        return 0.0

    try:
        # 解析合金成分，计算Mo当量和Al当量
        mo_eq, al_eq = parse_alloy_for_equivalents(alloy_composition)

        # 根据model_info获取特征列
        feature_cols = non_aging_models['info'].get('feature_cols', ['Mo当量', 'Al当量'])

        # 构建特征数据
        feature_data = pd.DataFrame({
            'Mo当量': [mo_eq],
            'Al当量': [al_eq]
        })

        # 确保特征列顺序与训练时一致
        feature_data = feature_data[feature_cols]

        # 标准化
        X_scaled = non_aging_models['scaler'].transform(feature_data)

        # 预测
        correction = non_aging_models['model'].predict(X_scaled)[0]

        print(f"[INFO] 非时效补项: Mo_eq={mo_eq:.2f}, Al_eq={al_eq:.2f} -> Δσ={correction:+.1f} MPa")

        return correction

    except Exception as e:
        print(f"[ERROR] 非时效补项计算失败: {e}")
        return 0.0


def calculate_high_vcr_correction(high_vcr_models, alloy_composition):
    """
    计算高V含量材料的强度补项（仅当V≥13%时调用）

    参数:
        high_vcr_models: dict, 包含model, scaler, info的字典
        alloy_composition: str, 合金成分字符串，如"Ti-15V-3Cr"

    返回:
        float: 强度补项(MPa)，如果计算失败返回0
    """
    if high_vcr_models is None:
        return 0.0

    try:
        # 解析合金成分，提取V和Cr含量
        elements = parse_alloy_composition(alloy_composition)
        v_content = elements.get('V', 0.0)
        cr_content = elements.get('Cr', 0.0)

        print(f"[INFO] 高VCr补项计算: V={v_content:.2f}%, Cr={cr_content:.2f}%")

        # 根据model_info获取特征列
        feature_cols = high_vcr_models['info'].get('feature_cols', ['V含量', 'Cr含量'])

        # 构建特征数据
        feature_data = pd.DataFrame({
            'V含量': [v_content],
            'Cr含量': [cr_content]
        })

        # 确保特征列顺序与训练时一致
        feature_data = feature_data[feature_cols]

        # 标准化
        X_scaled = high_vcr_models['scaler'].transform(feature_data)

        # 预测
        correction = high_vcr_models['model'].predict(X_scaled)[0]

        print(f"[INFO] 高VCr补项预测值: Δσ_VCr = {correction:+.1f} MPa")

        return correction

    except Exception as e:
        print(f"[ERROR] 高VCr补项计算失败: {e}")
        return 0.0


# ===================== 强化机制计算 =====================
def calculate_trace_element_contribution(elements):
    """
    计算微量元素（C、H、O、N、B）的强度贡献

    参数:
        elements: dict, 元素及其含量(wt%)

    返回:
        float: 微量元素强度贡献(MPa)
    """
    total_contribution = 0.0

    for elem, coefficient in trace_element_coefficients.items():
        if elem in elements and elements[elem] > 0:
            contribution = coefficient * elements[elem]
            total_contribution += contribution

    return total_contribution


def calculate_solid_solution_ti(elements):
    """
    计算钛合金固溶强化（不含微量元素）

    公式: σ_SS = (Σ K^1.5 · x)^(2/3)

    参数:
        elements: dict, 元素及其含量(wt%)

    返回:
        float: 固溶强化贡献(MPa)
    """
    sigma = 0.0
    for elem, content_wt in elements.items():
        # 跳过微量元素
        if elem in ['C', 'H', 'O', 'N', 'B']:
            continue

        coef = strengthening_coefficient.get(elem, 0)
        if coef > 0 and content_wt > 0:
            # 简化：假设质量百分比接近原子百分比
            atomic_percent = content_wt
            sigma += ((coef ** 1.5) * atomic_percent * 0.01)

    sigma_ss = sigma ** (2/3) if sigma > 0 else 0.0
    return sigma_ss


def calculate_dislocation_strengthening_ti(alpha_volume, alpha_density, beta_volume, beta_density):
    """
    计算钛合金位错强化

    公式: σ_dis = M·α·G·b·√ρ

    参数:
        alpha_volume: float, α-Ti体积分数(%)
        alpha_density: float, α-Ti位错密度(/m²)
        beta_volume: float, β-Ti体积分数(%)
        beta_density: float, β-Ti位错密度(/m²)

    返回:
        float: 位错强化贡献(MPa)
    """
    M = 3.06  # Taylor因子
    alpha_coef = 0.2  # α相系数
    G_alpha = 40  # α-Ti剪切模量 (GPa)
    G_beta = 35   # β-Ti剪切模量 (GPa)
    b = 0.295e-9  # 柏氏矢量 (m)

    # α相贡献
    alpha_contrib = 0.0
    if alpha_volume > 0 and alpha_density > 0:
        alpha_contrib = M * alpha_coef * G_alpha * 1e9 * b * np.sqrt(alpha_density) * (alpha_volume / 100)

    # β相贡献
    beta_contrib = 0.0
    if beta_volume > 0 and beta_density > 0:
        beta_contrib = M * alpha_coef * G_beta * 1e9 * b * np.sqrt(beta_density) * (beta_volume / 100)

    total_dislocation = alpha_contrib + beta_contrib
    return total_dislocation / 1e6  # 转换为MPa


def calculate_hall_petch_ti(k1, d1, f1, k2, d2, f2):
    """
    计算钛合金Hall-Petch强化

    公式: σ_HP = Σ (k_i / √d_i) * f_i

    参数:
        k1: float, 晶粒1的Hall-Petch系数(MPa·μm^0.5)
        d1: float, 晶粒1的尺寸(μm)
        f1: float, 晶粒1的体积分数(%)
        k2: float, 晶粒2的Hall-Petch系数(MPa·μm^0.5)
        d2: float, 晶粒2的尺寸(μm)
        f2: float, 晶粒2的体积分数(%)

    返回:
        float: Hall-Petch强化贡献(MPa)
    """
    sigma_hp = 0.0

    # 晶粒1的贡献
    if d1 > 0 and f1 > 0:
        sigma_hp += (k1 / np.sqrt(d1)) * (f1 / 100)

    # 晶粒2的贡献
    if d2 > 0 and f2 > 0:
        sigma_hp += (k2 / np.sqrt(d2)) * (f2 / 100)

    return sigma_hp


def calculate_second_phase_ti(lambda1, f1, r1, h1, lambda2=0, f2=0, r2=0, h2=1.0):
    """
    计算钛合金第二相强化

    公式: σ = 0.4·M·μ·b/(π(1-ν)^0.5)·ln(2r/b)/λ · K · f
    其中 K = h^(1/6) * ((2+h^2)/3)^(-0.25) 为非球形修正系数

    参数:
        lambda1: float, 析出相1的粒子间距(μm)
        f1: float, 析出相1的体积分数(%)
        r1: float, 析出相1的平均半径(μm)
        h1: float, 析出相1的长高比
        lambda2: float, 析出相2的粒子间距(μm)
        f2: float, 析出相2的体积分数(%)
        r2: float, 析出相2的平均半径(μm)
        h2: float, 析出相2的长高比

    返回:
        float: 第二相强化贡献(MPa)
    """
    # 材料常数
    M = 1  # M_alphaTi
    G = 44e9  # G_alphaTi (Pa)
    b = 2.95e-10  # b_alphaTi (m)
    v = 0.3  # 泊松比

    # 计算常数因子
    const_factor = (0.4 * G * b) / (math.pi * math.sqrt(1 - v))

    sigma_sp = 0.0

    # 析出相1的贡献
    if lambda1 > 0 and f1 > 0 and r1 > 0:
        lambda1_m = lambda1 * 1e-6  # μm -> m
        r1_m = r1 * 1e-6  # μm -> m
        f1_decimal = f1 / 100.0  # % -> 小数

        # 非球形修正系数
        K1 = (h1 ** (1/6)) * (((2 + h1**2) / 3) ** (-0.25))

        # 计算强化应力 (Pa)
        phase1_strength_pa = M * const_factor * math.log(2 * r1_m / b) / lambda1_m * K1 * f1_decimal
        phase1_strength_mpa = phase1_strength_pa / 1e6  # 转换为MPa
        sigma_sp += phase1_strength_mpa

    # 析出相2的贡献
    if lambda2 > 0 and f2 > 0 and r2 > 0:
        lambda2_m = lambda2 * 1e-6  # μm -> m
        r2_m = r2 * 1e-6  # μm -> m
        f2_decimal = f2 / 100.0  # % -> 小数

        # 非球形修正系数
        K2 = (h2 ** (1/6)) * (((2 + h2**2) / 3) ** (-0.25))

        # 计算强化应力 (Pa)
        phase2_strength_pa = M * const_factor * math.log(2 * r2_m / b) / lambda2_m * K2 * f2_decimal
        phase2_strength_mpa = phase2_strength_pa / 1e6  # 转换为MPa
        sigma_sp += phase2_strength_mpa

    return sigma_sp


# ===================== 主预测函数 =====================
def predict_titanium_yield_strength_enhanced(
    composition,
    processing_params,
    heat_treatment_params,
    trace_elements=None,
    use_aging_model=True,
    use_high_v_correction=True
):
    """
    钛合金屈服强度预测主函数（增强版）

    支持特性:
    - 多级热处理工艺（自动区分固溶和时效）
    - 时效强化预测
    - 高V含量材料特殊处理（V≥13%）
    - 非时效工艺补项

    参数:
        composition: str, 合金成分字符串，如 "Ti-6Al-4V"
        processing_params: str或dict, 加工工艺
            - 字符串格式: "温度/变形量"，如 "860/80"
            - 字典格式: {'temperature': 860, 'deformation': 80}
        heat_treatment_params: str或dict, 热处理工艺
            - 字符串格式: "温度/时间/冷却+..."，如 "760/30/AC+500/120/AC"
            - 字典格式: {'temperature': 760, 'time': 30, 'cooling': 'AC'}
        trace_elements: dict, 微量元素含量(wt%)，如 {'C': 0.05, 'O': 0.15}
        use_aging_model: bool, 是否使用时效模型
        use_high_v_correction: bool, 是否使用高V含量补项

    返回:
        dict: 结果字典，包含:
            - 'yield_strength': 预测的屈服强度(MPa)
            - 'contributions': 各强化机制的贡献
            - 'microstructure': 预测的微观组织参数
            - 'corrections': 各种补项值
    """

    # 解析工艺参数
    if isinstance(processing_params, str):
        work_temp, deformation = parse_processing(processing_params)
    else:
        work_temp = processing_params.get('temperature', 27.0)
        deformation = processing_params.get('deformation', 0.0)

    if work_temp is None or deformation is None:
        raise ValueError("加工工艺参数格式错误")

    # 解析热处理参数
    if isinstance(heat_treatment_params, str):
        # 标准化多级热处理
        solution_stage, aging_stage = standardize_heat_treatment(heat_treatment_params)
    else:
        solution_stage = {
            'temp': heat_treatment_params.get('temperature'),
            'time': heat_treatment_params.get('time'),
            'cooling': heat_treatment_params.get('cooling')
        }
        aging_stage = None

    # 检查是否为高V含量材料
    is_high_v_material = check_high_v_content(composition, threshold=13.0)

    # 高V材料：合并所有热处理阶段
    if is_high_v_material and use_high_v_correction:
        print(f"[INFO] 检测到高V含量材料(V≥13%)，将合并所有热处理阶段")
        all_stages = []
        if solution_stage is not None:
            all_stages.append(solution_stage)
        if aging_stage is not None:
            all_stages.append(aging_stage)

        if all_stages:
            solution_stage = merge_heat_treatment_stages(all_stages)
            aging_stage = None  # 不使用时效补项

    # 确定用于模型预测的热处理参数
    if solution_stage is not None:
        heat_temp = solution_stage['temp']
        heat_time = solution_stage['time']
        cooling = solution_stage['cooling']
    else:
        raise ValueError("热处理工艺参数格式错误")

    # 加载所有ML模型
    print("\n[INFO] 加载ML模型...")
    ml_model = load_ml_model()
    grain_models = load_grain_models()
    aging_models = load_aging_models() if use_aging_model else None
    non_aging_models = load_non_aging_correction_models()
    high_vcr_models = load_high_vcr_correction_models() if use_high_v_correction else None

    # 计算Mo当量
    mo_eq = calculate_mo_equivalent_ml(composition)

    # ========== 微观组织预测 ==========
    print(f"\n[INFO] 预测微观组织参数...")

    # 1. 预测第二相参数
    if ml_model:
        spacing, radius = predict_second_phase_params(
            ml_model, composition, cooling, heat_temp, heat_time, work_temp, deformation
        )
    else:
        spacing, radius = 0.1, 0.05

    # 2. 预测体积分数
    alpha_volume_pct, beta_volume_pct = predict_volume_fractions(
        mo_eq, cooling, heat_temp, deformation
    )

    # 3. 预测位错密度
    alpha_density = calculate_dislocation_density(
        deformation, cooling, heat_temp, heat_time, work_temp, mo_eq, 'alpha'
    )
    beta_density = calculate_dislocation_density(
        deformation, cooling, heat_temp, heat_time, work_temp, mo_eq, 'beta'
    )

    # 4. 预测晶粒参数
    if grain_models:
        grain_params = predict_grain_parameters(
            grain_models, composition, deformation, work_temp, heat_temp, heat_time, cooling
        )
    else:
        grain_params = None

    # ========== 强化机制计算 ==========
    print(f"\n[INFO] 计算各强化机制贡献...")

    # 1. 固溶强化
    elements = parse_alloy_composition(composition)
    if trace_elements:
        elements.update(trace_elements)

    sigma_ss = calculate_solid_solution_ti(elements)
    trace_effect = calculate_trace_element_contribution(elements) if trace_elements else 0.0

    # 2. 位错强化
    sigma_dis = calculate_dislocation_strengthening_ti(
        alpha_volume_pct, alpha_density, beta_volume_pct, beta_density
    )

    # 3. 第二相强化
    sigma_sp = calculate_second_phase_ti(
        lambda1=spacing if spacing else 1.0,
        f1=100,  # 默认100%
        r1=radius if radius else 0.1,
        h1=1.0
    )

    # 4. Hall-Petch强化
    if grain_params:
        sigma_hp = calculate_hall_petch_ti(
            k1=grain_params['grain1_hall_petch'],
            d1=grain_params['grain1_size'],
            f1=grain_params['grain1_volume'],
            k2=grain_params['grain2_hall_petch'],
            d2=grain_params['grain2_size'],
            f2=grain_params['grain2_volume']
        )
    else:
        sigma_hp = 0.0

    # ========== 模型补项计算 ==========
    print(f"\n[INFO] 计算模型补项...")

    # 1. 时效强度增量
    sigma_aging = 0.0
    if aging_stage and aging_models and use_aging_model:
        sigma_aging = calculate_aging_strength_increment(
            aging_models,
            composition,
            aging_stage['temp'],
            aging_stage['time']
        )

    # 2. 非时效补项
    sigma_non_aging = 0.0
    if non_aging_models and aging_stage is None:
        sigma_non_aging = calculate_non_aging_correction(non_aging_models, composition)

    # 3. 高V含量补项
    sigma_high_v = 0.0
    if is_high_v_material and high_vcr_models and use_high_v_correction:
        sigma_high_v = calculate_high_vcr_correction(high_vcr_models, composition)

    # ========== 总强度计算 ==========
    sigma_0 = 180.0  # 基础强度 (MPa)

    yield_strength = (sigma_0 + sigma_ss + sigma_dis + sigma_sp + sigma_hp +
                     sigma_aging + sigma_non_aging + sigma_high_v)
    yield_strength = round(yield_strength, 1)

    # 构建返回结果
    result = {
        'yield_strength': yield_strength,
        'contributions': {
            'base_strength': sigma_0,
            'solid_solution': round(sigma_ss, 1),
            'dislocation': round(sigma_dis, 1),
            'second_phase': round(sigma_sp, 1),
            'hall_petch': round(sigma_hp, 1),
            'trace_elements_effect': round(trace_effect, 1) if trace_elements else 0.0
        },
        'corrections': {
            'aging_increment': round(sigma_aging, 1),
            'non_aging_correction': round(sigma_non_aging, 1),
            'high_v_correction': round(sigma_high_v, 1)
        },
        'microstructure': {
            'mo_equivalent': round(mo_eq, 2),
            'alpha_volume_pct': alpha_volume_pct,
            'beta_volume_pct': beta_volume_pct,
            'alpha_dislocation_density': alpha_density,
            'beta_dislocation_density': beta_density,
            'second_phase_spacing_um': spacing,
            'second_phase_radius_um': radius,
            'grain_parameters': grain_params
        },
        'process_info': {
            'is_high_v_material': is_high_v_material,
            'has_aging': aging_stage is not None,
            'solution_stage': solution_stage,
            'aging_stage': aging_stage
        }
    }

    return result


def batch_predict_from_excel(excel_path, output_path=None, max_rows=None):
    """
    从Excel文件批量预测屈服强度

    输入Excel文件应包含列：
    - '合金成分': 如 "Ti-6Al-4V"
    - '热变形工艺\n温度(℃)/变形量(%)': 如 "860/80"
    - '热处理工艺\n温度(℃)/时间(min)/冷却方式(FC、AC、WQ)+温度(℃)/时间(min)/冷却方式(FC、AC、WQ)': 如 "760/30/AC+500/120/AC"
    - 可选的微量元素列: 'C', 'H', 'O', 'N', 'B'

    参数:
        excel_path: str, 输入Excel文件路径
        output_path: str, 输出Excel文件路径（如果为None，则覆盖原文件）
        max_rows: int, 最大处理行数（如果为None，则处理所有行）

    返回:
        pd.DataFrame: 包含预测结果的DataFrame
    """
    print(f"\n{'='*60}")
    print(f"开始批量预测（增强版）")
    print(f"输入文件: {excel_path}")
    print(f"{'='*60}\n")

    # 读取Excel文件
    df = pd.read_excel(excel_path)
    print(f"读取到 {len(df)} 条数据")

    # 如果指定了最大行数
    if max_rows is not None and len(df) > max_rows:
        print(f"只处理前 {max_rows} 条数据\n")
        df = df.head(max_rows)
    else:
        print()

    # 初始化预测结果列
    predictions = []

    # 逐行处理
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"处理进度: {idx+1}/{len(df)}")

        try:
            # 获取输入参数
            comp_str = row.get('合金成分', '')
            hd_col = '热变形工艺\n温度(℃)/变形量(%)'
            hd_str = row.get(hd_col, '')
            ht_col = '热处理工艺\n温度(℃)/时间(min)/冷却方式(FC、AC、WQ)+温度(℃)/时间(min)/冷却方式(FC、AC、WQ)'
            ht_str = row.get(ht_col, '')

            # 获取微量元素
            trace_elements = {}
            for elem in ['C', 'H', 'O', 'N', 'B']:
                if elem in row and pd.notna(row[elem]):
                    trace_elements[elem] = float(row[elem])

            # 预测
            result = predict_titanium_yield_strength_enhanced(
                composition=comp_str,
                processing_params=hd_str,
                heat_treatment_params=ht_str,
                trace_elements=trace_elements if trace_elements else None
            )

            predictions.append(result['yield_strength'])

        except Exception as e:
            print(f"  [ERROR] 第{idx+1}行预测失败: {e}")
            predictions.append(np.nan)

    # 添加预测结果到DataFrame
    df['预测屈服强度(MPa)'] = predictions

    # 如果存在实际值，计算误差
    if '实际屈服强度(MPa)' in df.columns:
        df['预测误差(MPa)'] = df['预测屈服强度(MPa)'] - df['实际屈服强度(MPa)']
        df['预测误差(%)'] = (df['预测误差(MPa)'] / df['实际屈服强度(MPa)']) * 100

    # 保存结果
    if output_path is None:
        output_path = excel_path.replace('.xlsx', '_预测结果.xlsx')

    df.to_excel(output_path, index=False)
    print(f"\n预测完成！结果已保存到: {output_path}")

    # 打印统计信息
    if '实际屈服强度(MPa)' in df.columns:
        mae = df['预测误差(MPa)'].abs().mean()
        mape = df['预测误差(%)'].abs().mean()
        print(f"\n预测性能:")
        print(f"  平均绝对误差(MAE): {mae:.2f} MPa")
        print(f"  平均绝对百分比误差(MAPE): {mape:.2f}%")

    return df


# ===================== 示例使用 =====================
if __name__ == "__main__":
    # 示例1：基本使用
    print("=" * 60)
    print("示例1：基本使用（单级热处理）")
    print("=" * 60)

    result1 = predict_titanium_yield_strength_enhanced(
        composition="Ti-6Al-4V",
        processing_params="860/80",
        heat_treatment_params="760/30/AC"
    )

    print(f"\n合金成分: Ti-6Al-4V")
    print(f"加工工艺: 860℃/80%变形")
    print(f"热处理工艺: 760℃/30min/空冷")
    print(f"\n预测屈服强度: {result1['yield_strength']} MPa")
    print("\n强化机制贡献:")
    for mechanism, value in result1['contributions'].items():
        print(f"  {mechanism}: {value:.1f} MPa")

    # 示例2：多级热处理（含时效）
    print("\n" + "=" * 60)
    print("示例2：多级热处理（固溶+时效）")
    print("=" * 60)

    result2 = predict_titanium_yield_strength_enhanced(
        composition="Ti-6Al-4V",
        processing_params="900/70",
        heat_treatment_params="800/60/WQ+500/120/AC",  # 固溶+时效
        trace_elements={'C': 0.05, 'O': 0.15}
    )

    print(f"\n合金成分: Ti-6Al-4V")
    print(f"微量元素: C=0.05%, O=0.15%")
    print(f"加工工艺: 900℃/70%变形")
    print(f"热处理工艺: 800℃/60min/水淬 + 500℃/120min/空冷")
    print(f"\n预测屈服强度: {result2['yield_strength']} MPa")
    print("\n强化机制贡献:")
    for mechanism, value in result2['contributions'].items():
        print(f"  {mechanism}: {value:.1f} MPa")
    print("\n模型补项:")
    for correction, value in result2['corrections'].items():
        print(f"  {correction}: {value:+.1f} MPa")

    # 示例3：高V含量材料
    print("\n" + "=" * 60)
    print("示例3：高V含量材料（V≥13%）")
    print("=" * 60)

    result3 = predict_titanium_yield_strength_enhanced(
        composition="Ti-15V-3Cr",
        processing_params="850/75",
        heat_treatment_params="760/30/AC"
    )

    print(f"\n合金成分: Ti-15V-3Cr")
    print(f"加工工艺: 850℃/75%变形")
    print(f"热处理工艺: 760℃/30min/空冷")
    print(f"高V材料: {result3['process_info']['is_high_v_material']}")
    print(f"\n预测屈服强度: {result3['yield_strength']} MPa")
    print("\n强化机制贡献:")
    for mechanism, value in result3['contributions'].items():
        print(f"  {mechanism}: {value:.1f} MPa")
    print("\n模型补项:")
    for correction, value in result3['corrections'].items():
        print(f"  {correction}: {value:+.1f} MPa")

    # 示例4：批量预测
    print("\n" + "=" * 60)
    print("示例4：批量预测（从Excel文件）")
    print("=" * 60)

    # 取消注释以下行来运行批量预测
    # excel_path = "钛合金数据库-专用测试集.xlsx"
    # if os.path.exists(excel_path):
    #     df_results = batch_predict_from_excel(excel_path, max_rows=10)
    # else:
    #     print(f"\n[WARNING] 找不到文件: {excel_path}")

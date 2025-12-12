# grain_size_predictor.py
"""
钛合金晶粒尺寸预测接口函数

输入参数：
- composition: str - 合金成分字符串，如 "Ti-6Al-4V"
- true_strain: float - 应变量
- strain_rate: float - 应变速率 (1/s)
- temperature: float - 变形温度 (K)
- initial_grain_size: float - 初始晶粒尺寸 (μm)，可选，默认100.0
- beta_trans_temp: float - β完全转变点温度 (K)，可选，默认1000.0

返回值：
- float - 平均晶粒尺寸 (μm)
"""

import pandas as pd
import numpy as np
import math
import joblib
from pathlib import Path
from scipy.constants import R as R_gas
import re
import warnings

# 忽略所有警告
warnings.simplefilter('ignore')

# ---------------- CONFIG ----------------
from importlib import resources
PEAK_FILE = str(resources.files("thermal_deformation.resource").joinpath("峰值应力.xlsx"))
RECRYST_FILE = str(resources.files("thermal_deformation.resource").joinpath("再结晶晶粒尺寸参数.xlsx"))
MODEL_FILE = str(resources.files("thermal_deformation.resource").joinpath("CatBoost_MGWO_Final_Model_Q.joblib"))
SCALER_FILE = str(resources.files("thermal_deformation.resource").joinpath("scaler_X.joblib"))
SCALER_Y_FILE = str(resources.files("thermal_deformation.resource").joinpath("scaler_y.joblib"))

GROUP_BINS = [-1e9, 3, 8, 13, 25, 1e9]
GROUP_LABELS = ['0-3%','3-8%','8-13%','13-25%','>25%']

ALL_ELEMENTS = ['Ti', 'V', 'Cr', 'Si', 'Al', 'Mo', 'Fe', 'Sn', 'Zr', 'Ta', 'Nb', 'Y', 'O', 'N', 'B', 'C', 'H', 'Ni', 'Cu', 'W', 'Mn', 'Zn', 'Ru']

GROUP_ALPHA_OVERRIDE = {
    '0-3%': 0.01,
    '3-8%': 0.0055,
    '8-13%': 0.046996064,
    '13-25%': 0.002813894,
    '>25%': np.nan
}
GROUP_M_OVERRIDE = {
    '0-3%': 0.2,
    '3-8%': 0.0778,
    '8-13%': 0.076935,
    '13-25%': 0.187,
    '>25%': np.nan
}

# ---------------- Utilities ----------------
def safe_load_joblib(path):
    p = Path(path)
    if p.exists():
        try:
            return joblib.load(p)
        except Exception as e:
            print(f"警告: 加载 {path} 失败: {e}")
            return None
    print(f"警告: 未找到文件 {path}")
    return None

def load_excel_safe(path):
    p = Path(path)
    if not p.exists():
        print(f"警告: 未找到文件 {path}")
        return None
    try:
        return pd.read_excel(p)
    except Exception as e:
        print(f"警告: 读取 {path} 失败: {e}")
        return None

def parse_composition(input_str):
    input_str = input_str.lower().replace(" ", "")
    parts = input_str.split('-')
    comp_dict = {}
    total_other = 0.0
    comp_dict['Ti'] = 0.0
    for part in parts[1:]:
        match = re.match(r'([\d.]+)([a-z]+)', part)
        if match:
            value = float(match.group(1))
            elem = match.group(2).capitalize()
            if elem in ALL_ELEMENTS:
                comp_dict[elem] = value
                total_other += value
    comp_dict['Ti'] = 100.0 - total_other
    return {elem: comp_dict.get(elem, 0.0) for elem in ALL_ELEMENTS}

def calc_Moeq_Aleq(comp):
    Moeq = (comp['Mo'] + 0.22*comp['Ta'] + 0.28*comp['Nb'] + 0.44*comp['W'] +
            0.67*comp['V'] + 1.25*comp['Cr'] + 1.25*comp['Ni'] + 1.70*comp['Mn'] +
            2.50*comp['Fe'] - 1.00*comp['Al'])
    Aleq = (comp['Al'] + 0.17*comp['Zr'] + 0.33*comp['Sn'] + comp['O']*10.0)
    return Moeq, Aleq

def assign_group(Moeq):
    for i in range(len(GROUP_BINS)-1):
        if GROUP_BINS[i] < Moeq <= GROUP_BINS[i+1]:
            return GROUP_LABELS[i]
    return GROUP_LABELS[-1]

def compute_X(eps, eps_c, eps_p, beta, k, cap=True):
    def _scalar(e):
        try:
            if eps_p is None or eps_p == 0 or beta is None or k is None or np.isnan(beta) or np.isnan(k):
                return float('nan')
            if e <= eps_c:
                val = 0.0
            else:
                ratio = (e - eps_c) / eps_p
                if ratio <= 0:
                    val = 0.0
                else:
                    power = math.pow(ratio, k)
                    val = 1.0 - math.exp(- beta * power)
            if cap:
                return float(max(0.0, min(1.0, val)))
            else:
                return float(val)
        except Exception:
            return float('nan')
    if np.isscalar(eps):
        return _scalar(float(eps))
    arr = np.array(eps, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    for i,v in enumerate(arr):
        out[i] = _scalar(v)
    return out

def compute_drx(A,B,Z, micro, group):
    """
    计算再结晶晶粒尺寸 d_DRX = A * Z^B
    """
    try:
        base_drx = float(A) * (float(Z) ** float(B))
        
        if micro == 1:
            result = base_drx * 0.9
        else:
            result = base_drx * 1.1
            
        # 保持 0.1 的物理下限
        result = max(0.1, min(500.0, result))
        return result
    except Exception:
        return float('nan')

def compute_peak_stress(alpha, m, epsdot, Q, T):
    try:
        return alpha * (epsdot ** m) * math.exp((m * Q) / (R_gas * T))
    except Exception:
        try:
            return alpha * (epsdot ** m) * math.exp(Q / (R_gas * T))
        except:
            return float('nan')

# Compute group medians
def compute_group_medians_from_peak(peak_df):
    gm = {g: {'alpha':np.nan,'m':np.nan,'Q':np.nan} for g in GROUP_LABELS}
    if peak_df is None: return gm
    df = peak_df.copy()
    cols = df.columns.tolist()
    mo_col = next((c for c in cols if any(x in c.lower() for x in ['mo','mo_eq','mo当量'])), None)
    alpha_col = next((c for c in cols if any(x in c.lower() for x in ['alpha','a_','a '])), None)
    m_col = next((c for c in cols if any(x in c.lower() for x in [' m ',' m_','^m',' exponent','指数'])), None)
    Q_col = next((c for c in cols if any(x in c.lower() for x in ['q','activation','活化','q_act'])), None)
    
    if mo_col is None:
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]): mo_col = c; break
            
    if mo_col:
        df[mo_col] = pd.to_numeric(df[mo_col], errors='coerce')
        df['Mo_group'] = pd.cut(df[mo_col].fillna(-9999), bins=GROUP_BINS, labels=GROUP_LABELS)
    else:
        df['Mo_group'] = GROUP_LABELS[0]
        
    for g in GROUP_LABELS:
        sub = df[df['Mo_group']==g]
        if len(sub)>0:
            try:
                if alpha_col: gm[g]['alpha'] = float(pd.to_numeric(sub[alpha_col], errors='coerce').median())
                if m_col: gm[g]['m'] = float(pd.to_numeric(sub[m_col], errors='coerce').median())
                if Q_col: gm[g]['Q'] = float(pd.to_numeric(sub[Q_col], errors='coerce').median())
            except: pass
    return gm

def compute_group_recryst(recryst_df):
    res = {g: {'beta':np.nan,'k':np.nan,'A':np.nan,'B':np.nan} for g in GROUP_LABELS}
    if recryst_df is None: return res
    df = recryst_df.copy()
    cols = df.columns.tolist()
    mo_col = next((c for c in cols if any(x in c.lower() for x in ['mo','mo_eq','mo当量'])), None)
    beta_col = next((c for c in cols if any(x in c.lower() for x in ['beta','β'])), None)
    k_col = next((c for c in cols if any(x in c.lower() for x in ['k','指数'])), None)
    A_col = next((c for c in cols if any(x in c.lower() for x in ['^a',' a ','a_','a ']) or c=='a' or c=='A'), None)
    B_col = next((c for c in cols if any(x in c.lower() for x in ['^b',' b ','b_','b ']) or c=='b' or c=='B'), None)

    if mo_col:
        df[mo_col] = pd.to_numeric(df[mo_col], errors='coerce')
        df['Mo_group'] = pd.cut(df[mo_col].fillna(-9999), bins=GROUP_BINS, labels=GROUP_LABELS)
    else:
        df['Mo_group'] = GROUP_LABELS[0]
    for g in GROUP_LABELS:
        sub = df[df['Mo_group']==g]
        if len(sub)>0:
            try:
                if beta_col: res[g]['beta'] = float(pd.to_numeric(sub[beta_col], errors='coerce').median())
                if k_col: res[g]['k'] = float(pd.to_numeric(sub[k_col], errors='coerce').median())
                if A_col: res[g]['A'] = float(pd.to_numeric(sub[A_col], errors='coerce').median())
                if B_col: res[g]['B'] = float(pd.to_numeric(sub[B_col], errors='coerce').median())
            except: pass
    return res

def detect_scaler_feature_names(scaler):
    if scaler is None: return None
    if hasattr(scaler, "feature_names_in_"): return list(getattr(scaler, "feature_names_in_"))
    base = ALL_ELEMENTS + ["Mo_eq","Al_eq","micro","true_strain"]
    n = getattr(scaler, "n_features_in_", None)
    if n: return base[:int(n)] if len(base)>=n else base + [f"PAD_{i}" for i in range(int(n)-len(base))]
    return base

def build_feature_vector_from_names(feature_names, comp_dict, Moeq, Aleq, micro, true_strain, T):
    vec = []
    for name in feature_names:
        nm = str(name)
        if re.fullmatch(r'^[A-Za-z]{1,2}$', nm):
            key = nm.capitalize()
            vec.append(float(comp_dict.get(key, 0.0)))
            continue
        nl = nm.lower()
        if "mo" in nl and "eq" in nl: vec.append(float(Moeq))
        elif "al" in nl and "eq" in nl: vec.append(float(Aleq))
        elif nl in ("micro","microstructure"): vec.append(int(micro))
        elif nl in ("true_strain","strain","eps","epsilon"): vec.append(float(true_strain))
        elif nl in ("t","temperature","temp"): vec.append(float(T))
        else:
            up = nm.capitalize()
            vec.append(float(comp_dict.get(up,0.0)) if up in comp_dict else 0.0)
    return vec

def build_feature_vector_simple(comp_dict, Moeq, Aleq, micro, true_strain):
    vec = [float(comp_dict.get(el,0.0)) for el in ALL_ELEMENTS]
    vec += [float(Moeq), float(Aleq), int(micro), float(true_strain)]
    return vec

def predict_Q_with_fallback(model, scaler_X, scaler_y, feature_vector, group, ml_scale):
    info = {}
    try:
        if model is None or scaler_X is None:
            info['ml_status'] = 'model_or_scaler_missing'
            q_fallback = GROUP_MEDIANS.get(group, {}).get('Q', np.nan)
            info['Q_used'] = q_fallback
            return q_fallback, info
            
        n_req = getattr(scaler_X, "n_features_in_", None)
        fv = list(feature_vector)
        if n_req:
            if len(fv) < n_req: fv += [0.0] * (int(n_req) - len(fv))
            elif len(fv) > n_req: fv = fv[:int(n_req)]
            
        Xs = scaler_X.transform([fv])
        y_raw = model.predict(Xs)
        y0 = float(np.array(y_raw).ravel()[0])
        
        if scaler_y:
            try: Q_pred = scaler_y.inverse_transform([[y0]])[0,0]
            except:
                try: Q_pred = float(scaler_y.inverse_transform([y0])[0])
                except: Q_pred = y0
        else: Q_pred = y0
        
        Q_pred *= float(ml_scale)
        info['ml_status'] = 'success'
        info['Q_used'] = Q_pred
        return Q_pred, info
    except Exception as e:
        info['ml_status'] = 'failed'
        info['error'] = str(e)
        q_fallback = GROUP_MEDIANS.get(group, {}).get('Q', np.nan)
        info['Q_used'] = q_fallback
        return q_fallback, info

# 加载必要的文件和模型
peak_df = load_excel_safe(PEAK_FILE)
recryst_df = load_excel_safe(RECRYST_FILE)
model = safe_load_joblib(MODEL_FILE)
scaler_X = safe_load_joblib(SCALER_FILE)
scaler_y = safe_load_joblib(SCALER_Y_FILE)

GROUP_MEDIANS = compute_group_medians_from_peak(peak_df)
GROUP_RECRYST = compute_group_recryst(recryst_df)

# 从 scaling_factor_functions.py 导入 get_scaling_factors
try:
    from thermal_deformation.scaling_factor_functions import get_scaling_factors
except ImportError:
    print("警告: 未找到 scaling_factor_functions.py，将使用默认缩放因子")
    def get_scaling_factors(**kwargs):
        return {'alpha': 1.0, 'm': 1.0, 'A': 1.0, 'B': 1.0}

# ---------------- 主接口函数 ----------------
def predict_grain_size(
    composition: str,
    true_strain: float,
    strain_rate: float,
    temperature: float,
    initial_grain_size: float = 100.0,
    beta_trans_temp: float = 1000.0
) -> float:
    """
    预测钛合金晶粒尺寸的主接口函数
    
    参数:
    - composition: 合金成分字符串，如 "Ti-6Al-4V"
    - true_strain: 应变量
    - strain_rate: 应变速率 (1/s)
    - temperature: 变形温度 (K)
    - initial_grain_size: 初始晶粒尺寸 (μm)，可选，默认100.0
    - beta_trans_temp: β完全转变点温度 (K)，可选，默认1000.0
    
    返回:
    - float: 平均晶粒尺寸 (μm)
    """
    # 解析成分
    comp_dict = parse_composition(composition)
    for k in ALL_ELEMENTS: comp_dict.setdefault(k, 0.0)
    
    # 计算 Mo_eq 和 Al_eq
    Moeq, Aleq = calc_Moeq_Aleq(comp_dict)
    group = assign_group(Moeq)
    
    # 确定 micro
    micro = 1 if temperature > beta_trans_temp else 2
    
    # 构建特征向量
    feat_names = detect_scaler_feature_names(scaler_X)
    if feat_names:
        feature_vector = build_feature_vector_from_names(feat_names, comp_dict, Moeq, Aleq, micro, true_strain, temperature)
    else:
        feature_vector = build_feature_vector_simple(comp_dict, Moeq, Aleq, micro, true_strain)
    
    # 预测 Q
    Q_pred, ml_info = predict_Q_with_fallback(model, scaler_X, scaler_y, feature_vector, group, 1000.0)
    
    # 获取 α 和 m 的覆盖值
    alpha_orig = GROUP_ALPHA_OVERRIDE.get(group, np.nan)
    m_orig = GROUP_M_OVERRIDE.get(group, np.nan)
    if np.isnan(alpha_orig): alpha_orig = GROUP_MEDIANS.get(group, {}).get('alpha', 0.035)
    if np.isnan(m_orig): m_orig = GROUP_MEDIANS.get(group, {}).get('m', 0.9)
    
    # 获取缩放因子
    scaling_factors = get_scaling_factors(temp_type=micro, strain=true_strain, mo_eq=Moeq, al_eq=Aleq, comp_dict=comp_dict, T=temperature)
    
    # 计算 α_used 和 m_used
    alpha_used = float(alpha_orig) * scaling_factors['alpha']
    m_used = float(m_orig) * scaling_factors['m']
    
    # 计算峰值应变
    eps_p = compute_peak_stress(alpha_used, m_used, strain_rate, Q_pred, temperature)
    eps_c = 0.8 * eps_p if np.isfinite(eps_p) else 0.0
    
    # 获取 beta 和 k 值
    gr = GROUP_RECRYST.get(group, {})
    beta = gr.get('beta', 0.1) if not np.isnan(gr.get('beta', np.nan)) else 0.1
    k_val = gr.get('k', 0.2) if not np.isnan(gr.get('k', np.nan)) else 0.2
    
    # 计算再结晶分数 X
    X_capped = compute_X(true_strain, eps_c, eps_p, beta, k_val, cap=True)
    
    # 获取 A 和 B 组值
    A_group = gr.get('A', 10.0) if not np.isnan(gr.get('A', np.nan)) else 10.0
    B_group = gr.get('B', -0.1) if not np.isnan(gr.get('B', np.nan)) else -0.1
    
    # 计算 A_used 和 B_used
    A_used = float(A_group) * scaling_factors['A']
    B_used = float(B_group) * scaling_factors['B']
    
    # 动态调整 A 和 B (针对 0-3% 组)
    if group == '0-3%':
        # 动态 A: 仍使用衰减模型，保证基数足够大
        eff_strain = max(true_strain, 0.05)
        A_used = 18.0 * (eff_strain ** -0.4) 

        # 动态 B: 线性或非线性变化
        strain_clamped = min(max(true_strain, 0.0), 1.2)
        B_used = -0.015 - 0.01 * strain_clamped
    
    # 计算 Z 参数和 d_DRX
    try:
        Z = strain_rate * math.exp(Q_pred / (R_gas * temperature))
        d_DRX = compute_drx(A_used, B_used, Z, micro, group)
    except Exception as e:
        Z = np.nan; d_DRX = np.nan
    
    # 处理 d_DRX 为 NaN 的情况
    if np.isnan(d_DRX): d_DRX = initial_grain_size 
    
    # 计算平均晶粒尺寸 d_avg
    d_avg = initial_grain_size * (1.0 - X_capped) + d_DRX * X_capped
    
    return d_avg

# ---------------- 示例用法 ----------------
if __name__ == "__main__":
    # 示例 1: 使用默认初始晶粒尺寸和 β 转变温度
    avg_grain_size1 = predict_grain_size(
        composition="Ti-6Al-4V",
        true_strain=0.8,
        strain_rate=0.1,
        temperature=960.0
    )
    print(f"示例 1 - 平均晶粒尺寸: {avg_grain_size1:.4f} μm")
    
    # 示例 2: 指定初始晶粒尺寸和 β 转变温度
    avg_grain_size2 = predict_grain_size(
        composition="Ti-6Al-4V",
        true_strain=0.8,
        strain_rate=0.1,
        temperature=960.0,
        initial_grain_size=50.0,
        beta_trans_temp=1000.0
    )
    print(f"示例 2 - 平均晶粒尺寸: {avg_grain_size2:.4f} μm")
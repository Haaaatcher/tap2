import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
import warnings
from importlib import resources

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 物理引擎参数 (您的校准版)
# ==============================================================================
R = 8.314

PHYSICS_PARAMS = {
    'Q_low': 30423,      'ln_K0_low': 3.92,   # 低温参数
    'Q_high': 150000,    'ln_K0_high': 17.6,  # 高温参数
    'n_min': 0.10,       'n_max': 0.3526,
    'n_shift': 15.0,     'n_width': 30.9,
    's_width': 5.0
}

# ==============================================================================
# 2. 核心物理函数
# ==============================================================================

def calculate_n(T, T_beta):
    """ 计算生长指数 n(T) """
    exponent = -(T - T_beta - PHYSICS_PARAMS['n_shift']) / PHYSICS_PARAMS['n_width']
    if exponent > 100: val = 0
    elif exponent < -100: val = 1
    else: val = 1 / (1 + np.exp(exponent))
    return PHYSICS_PARAMS['n_min'] + (PHYSICS_PARAMS['n_max'] - PHYSICS_PARAMS['n_min']) * val

def calculate_base_K_params(T, T_beta):
    """ 计算基础 K 和 Q (不含成分修正) """
    arg = -(T - T_beta) / PHYSICS_PARAMS['s_width']
    if arg > 100: S = 0
    elif arg < -100: S = 1
    else: S = 1 / (1 + np.exp(arg))
    base_Q = PHYSICS_PARAMS['Q_low'] + (PHYSICS_PARAMS['Q_high'] - PHYSICS_PARAMS['Q_low']) * S
    ln_K0 = PHYSICS_PARAMS['ln_K0_low'] + (PHYSICS_PARAMS['ln_K0_high'] - PHYSICS_PARAMS['ln_K0_low']) * S
    return base_Q, np.exp(ln_K0)

def physics_simulator(T, time_sec, T_beta, delta_Q):
    """
    物理模拟器: 输入条件和阻力(delta_Q)，输出预测增量
    """
    base_Q, K0 = calculate_base_K_params(T, T_beta)
    final_Q = base_Q + delta_Q
    T_K = T + 273.15
    K = K0 * np.exp(-final_Q / (R * T_K))
    n = calculate_n(T, T_beta)
    return K * (time_sec ** n)

# ==============================================================================
# 3. 机器学习模块 (数据加载 -> 反演 -> 训练)
# ==============================================================================

def train_ml_model(csv_path):
    # print(">>> 1. 加载并清洗数据...")
    try:
        df = pd.read_csv(csv_path)
    except:
        # print("错误: 找不到 csv 文件。请确保 'titanium_data.csv' 存在。")
        return None, None

    # 定义需要的特征列 (成分)
    feature_cols = ['Al', 'V', 'Mo', 'Cr', 'Fe', 'Si', 'Nb']
    # 填充空值为0
    for col in feature_cols:
        if col not in df.columns: df[col] = 0
        df[col] = df[col].fillna(0)

    # print(">>> 2. 物理参数反演 (反推 Delta Q)...")
    target_dqs = []
    
    for _, row in df.iterrows():
        # 读取每行数据
        T = row['T_proc']
        t = row['Time'] * 3600.0 # 转为秒
        Tb = row['T_beta']
        real_inc = row['Increment']
        
        # 损失函数: 寻找最佳 Delta_Q
        def loss(dq_arr):
            dq = dq_arr[0]
            pred = physics_simulator(T, t, Tb, dq)
            # 加权平方误差
            return ((pred - real_inc) / (real_inc + 1.0))**2 
        
        # 求解
        res = minimize(loss, [0.0], bounds=[(-50000, 200000)], method='L-BFGS-B')
        target_dqs.append(res.x[0])
        
    df['Target_Delta_Q'] = target_dqs
    
    # print(">>> 3. 训练 AI 模型...")
    X = df[feature_cols]
    y = df['Target_Delta_Q']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # print(">>> 模型训练完成!")
    # 打印一下 AI 认为谁最重要
    imps = model.feature_importances_
    # print(f"   关键阻力元素权重: Mo={imps[2]:.2f}, Si={imps[5]:.2f}")
    
    return model, feature_cols

# ==============================================================================
# 4. 最终预测接口 (集成 AI + 低温独立模式)
# ==============================================================================

def predict_grain_size(D0, manual_T_beta, composition, heat_treatments, ml_model, feature_names):
    """
    智能预测函数
    """
    # --- Step A: AI 预测阻力 ---
    # 构建输入向量
    input_vec = []
    for f in feature_names:
        input_vec.append(composition.get(f, 0.0))
    
    # AI 给出 Delta Q (替代了之前的线性公式)
    pred_delta_Q = ml_model.predict([input_vec])[0]
    
    # print(f"\n[AI 分析] 预测该成分的额外阻力 Delta_Q = {pred_delta_Q:.0f} J/mol")
    
    # --- Step B: 物理计算 ---
    current_D = D0
    results = []
    results.append({'step': 0, 'desc': 'Initial', 'T': None, 'time': 0, 'D': current_D, 'inc': 0})
    
    AGING_THRESHOLD = manual_T_beta - 250.0
    
    for i, (T, t_hours) in enumerate(heat_treatments, 1):
        t_sec = t_hours * 3600.0
        
        # 计算基础 K 和 n
        base_Q, K0 = calculate_base_K_params(T, manual_T_beta)
        
        # 注入 AI 预测的阻力
        final_Q = base_Q + pred_delta_Q
        
        K = K0 * np.exp(-final_Q / (R * (T + 273.15)))
        n = calculate_n(T, manual_T_beta)
        
        # --- 模式判断 ---
        if T < AGING_THRESHOLD:
            # [低温独立叠加]
            inc = K * (t_sec ** n)
            new_D = current_D + inc
            mode_desc = "低温时效(独立)"
        else:
            # [高温等效时间]
            growth_so_far = current_D - D0
            if growth_so_far <= 1e-6:
                t_eq = 0
            else:
                if K < 1e-30: t_eq = 1e20
                else:
                    try: t_eq = (growth_so_far / K) ** (1/n)
                    except: t_eq = 0
            
            t_total = t_eq + t_sec
            new_growth = K * (t_total ** n)
            new_D = D0 + new_growth
            inc = new_D - current_D
            mode_desc = "高温固溶(标准)"
            
        results.append({
            'step': i, 'desc': mode_desc, 'T': T, 'time': t_hours,
            'D': new_D, 'inc': inc
        })
        current_D = new_D
        
    return results


model, feats = train_ml_model(str(resources.files("heat_treatment.resource").joinpath("titanium_data.csv")))

# ==============================================================================
# 5. 运行演示
# ==============================================================================
if __name__ == "__main__":
    # 1. 训练
    # 确保 'titanium_data.csv' 在同一目录下
    if model:
        # 2. 预测新合金 (Ti-5553 类型: 5Al-5Mo-5V-3Cr)
        # 这种合金阻力很大，AI 应该能识别出来
        user_comp = {'Al': 5.0, 'Mo': 5.0, 'V': 5.0, 'Cr': 3.0}
        user_T_beta = 860
        user_D0 = 10.0
        
        # 工艺: 900度 1h (高温) -> 400度 2h (时效)
        process = [(900, 1.0), (400, 2.0)]
        
        print(f"\n=== 预测演示: Ti-5553 ===")
        res = predict_grain_size(user_D0, user_T_beta, user_comp, process, model, feats)
        
        for r in res:
            if r['step'] == 0: continue
            print(f"步骤 {r['step']} ({r['T']}℃): 增量 {r['inc']:.2f} um -> 总尺寸 {r['D']:.2f} um")
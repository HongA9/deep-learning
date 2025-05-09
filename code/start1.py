import os
import re
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from baseline import Baseline  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ======================
# 配置参数
# ======================
BASE_PATH = r"D:\那就学\食用油\shiyongyou"
PURE_OIL_TYPES = ['菜籽油', '大豆油', '花生油', '葵花籽油', '山茶油', '玉米油']
SAMPLE_LENGTH = 1500  # 统一光谱长度
PREPROCESS_PARAMS = {
    'savgol_window': 11,
    'savgol_order': 3,
    'baseline_lambda': 1e6,
    'baseline_p': 0.05
}

# ======================
# 数据加载模块
# ======================
def parse_mixed_ratio(dir_name):
    """解析混合比例目录名，返回字典格式 {油种: 比例}"""
    pattern = re.compile(r'([\u4e00-\u9fa5]+)(\d+)')
    matches = pattern.findall(dir_name)
    return {oil: int(ratio)/100 for oil, ratio in matches}

def load_all_data(base_path):
    """递归加载所有数据"""
    all_samples = []
    
    # 加载纯油数据
    pure_path = os.path.join(base_path, "单个")
    for oil_type in os.listdir(pure_path):
        if oil_type not in PURE_OIL_TYPES:
            continue
            
        type_path = os.path.join(pure_path, oil_type)
        for fname in os.listdir(type_path):
            if not fname.endswith(".csv"):
                continue
                
            file_path = os.path.join(type_path, fname)
            df = pd.read_csv(file_path,encoding='gbk')
            all_samples.append({
                'spectrum': df['Count'].values,
                'type': 'pure',
                'oil_type': oil_type,
                'ratio': {oil_type: 1.0}  # 纯油比例设为100%
            })

    # 加载混合油数据
    mixed_path = os.path.join(base_path, "混合")
    for mix_folder in os.listdir(mixed_path):
        mix_path = os.path.join(mixed_path, mix_folder)
        if not os.path.isdir(mix_path):
            continue
            
        # 遍历不同比例子目录
        for ratio_dir in os.listdir(mix_path):
            ratio_path = os.path.join(mix_path, ratio_dir)
            if not os.path.isdir(ratio_path):
                continue
                
            ratio_dict = parse_mixed_ratio(ratio_dir)
            
            # 加载该比例下的所有CSV文件
            for fname in os.listdir(ratio_path):
                if not fname.endswith(".csv"):
                    continue
                    
                file_path = os.path.join(ratio_path, fname)
                df = pd.read_csv(file_path,encoding='gbk')
                all_samples.append({
                    'spectrum': df['Count'].values,
                    'type': 'mixed',
                    'oil_type': mix_folder,  # 如"菜籽葵花花生"
                    'ratio': ratio_dict
                })
    
    return all_samples

# ======================
# 数据预处理模块
# ======================
def preprocess_spectrum(spectrum):
    """完整预处理流水线"""
    
    # 1. 基线校正
    base = Baseline(spectrum)
    corrected = base.als(
        lam=PREPROCESS_PARAMS['baseline_lambda'],
        p=PREPROCESS_PARAMS['baseline_p']
    )
    spectrum = spectrum - corrected
    
    # 2. 平滑处理
    spectrum = savgol_filter(
        spectrum,
        window_length=PREPROCESS_PARAMS['savgol_window'],
        polyorder=PREPROCESS_PARAMS['savgol_order']
    )
    
    # 3. 标准化
    spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))
    
    # 4. 统一长度（插值处理）
    orig_length = len(spectrum)
    if orig_length != SAMPLE_LENGTH:
        spectrum = np.interp(
            np.linspace(0, 1, SAMPLE_LENGTH),
            np.linspace(0, 1, orig_length),
            spectrum
        )
    
    return spectrum

# ======================
# 主处理流程
# ======================
if __name__ == "__main__":
    # 1. 加载原始数据
    raw_data = load_all_data(BASE_PATH)
    print(f"共加载 {len(raw_data)} 个样本")
    print(raw_data[0])

    # 2. 数据预处理
    processed = [preprocess_spectrum(s['spectrum']) for s in raw_data]
    X = np.array(processed)
    
    # 3. 构建标签体系
    # 示例：构建混合比例的多输出目标
    all_oil_types = list(PURE_OIL_TYPES)
    y = np.zeros((len(raw_data), len(all_oil_types)))
    
    for i, sample in enumerate(raw_data):
        for oil, ratio in sample['ratio'].items():
            if oil in all_oil_types:
                idx = all_oil_types.index(oil)
                y[i, idx] = ratio
    
    # 4. 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=[s['type'] for s in raw_data]
    )
    
    # 5. 全局标准化（可选）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    # 可在此保存预处理后的数据
    # np.savez("processed_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
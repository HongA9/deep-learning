import os
import pandas as pd
import numpy as np

data_dir = 'D:\\那就学\\食用油\\shiyongyou\\单个'
oil_types = os.listdir(data_dir)

data_dict = {} #存储所有油种的数据，键为油种名称，值为numpy数组

for oil in oil_types:
    oil_dir = os.path.join(data_dir, oil)
    csv_files = [f for f in os.listdir(oil_dir) if f.endswith('.csv')]

    #读取所有CSV文件
    samples = []
    for file in csv_files:
        file_path = os.path.join(oil_dir, file)
        df = pd.read_csv(file_path, header=None,names = ['拉曼位移', 'Count'],\
                         encoding='gbk')
        counts = df['Count'].values #提取Count列的值
        samples.append(counts)
        
        #转换为numpy数组并存入数组
        data_dict[oil] = np.array(samples)

#检查数据
for oil,samples in data_dict.items():
    print(f"{oil}: {samples.shape}")

# 在预处理前添加清洗步骤
def clean_signal(signal):
    # 处理中文逗号（如"1,000.5" -> 1000.5）
    cleaned = np.array([float(str(x).replace('，','').replace(',','')) 
                       for x in signal])
    # 处理空值
    return np.nan_to_num(cleaned, nan=np.median(cleaned))

def preprocess_signal(signal):
    # 类型安全转换
    signal = signal.astype(np.float64) if signal.dtype != np.float64 else signal
    # 异常值过滤（基于截图文件时间戳统一性）
    signal = np.clip(signal, 0, 1e5)  # 假设信号强度合理范围
    baseline = np.percentile(signal[signal > 0], 15)  # 忽略零值
    return (signal - baseline) / (np.max(signal) - baseline)

# 添加文件验证日志
problem_files = []
for oil in oil_types:
    oil_dir = os.path.join(data_dir, oil)
    for file in os.listdir(oil_dir):
        try:
            df_test = pd.read_csv(os.path.join(oil_dir, file),
                                 usecols=[1],  # 仅读取Count列
                                 converters={'Count': float})
        except Exception as e:
            problem_files.append((oil, file, str(e)))
            
print(f"发现{len(problem_files)}个异常文件：")
for item in problem_files:
    print(f"油种：{item[0]} | 文件：{item[1]} | 错误：{item[2]}")

# 步骤1：光谱数据预处理（增强版）
def preprocess_signal(signal):
    """带基线校正的预处理"""
    baseline = np.percentile(signal, 15)  # 取15%分位数作为基线
    denoised = signal - baseline
    return denoised / np.max(denoised)   # 最大幅值归一化

# 对每个样本进行预处理
processed_data = {}
for oil, samples in data_dict.items():
    processed = np.array([preprocess_signal(s) for s in samples])
    processed_data[oil] = processed

# 步骤2：构建训练数据集（适配截图中的混合油结构）
X = []
y = []
for oil in oil_types:
    X.extend(processed_data[oil])
    y.extend([oil] * len(processed_data[oil]))
X = np.array(X)
y = np.array(y)



# 步骤3：创建混合油数据生成器（提前准备）
def generate_mixed_samples(components, num_samples=1000):
    """生成混合油样本"""
    mixed_X = []
    mixed_y = []
    
    for _ in range(num_samples):
        # 随机选择2-3种油混合
        num_oils = np.random.randint(2,4)
        selected = np.random.choice(components, num_oils, replace=False)
        
        # 生成随机比例（总和为1）
        ratios = np.random.dirichlet(np.ones(num_oils))
        
        # 合成混合信号
        mixed = np.zeros_like(X[0])
        for i, oil in enumerate(selected):
            sample = X[y == oil][np.random.randint(len(X[y == oil]))]
            mixed += ratios[i] * sample
        
        mixed_X.append(mixed)
        mixed_y.append(tuple(selected))  # 存储混合成分标签
        
    return np.array(mixed_X), np.array(mixed_y)

# 生成示例（使用截图中的油种组合）
mixed_X, mixed_y = generate_mixed_samples(oil_types)


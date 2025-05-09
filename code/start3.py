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
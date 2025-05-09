import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据加载和预处理（基于start.py）
def load_and_preprocess(data_dir):
    oil_types = os.listdir(data_dir)
    data_dict = {}
    
    for oil in oil_types:
        oil_dir = os.path.join(data_dir, oil)
        csv_files = [f for f in os.listdir(oil_dir) if f.endswith('.csv')]
        
        samples = []
        for file in csv_files:
            file_path = os.path.join(oil_dir, file)
            df = pd.read_csv(file_path, header=None, names=['拉曼位移', 'Count'], encoding='gbk')
            # 确保数据转换为浮点数
            counts = pd.to_numeric(df['Count'], errors='coerce').values
            # 移除可能的NaN值
            counts = counts[~np.isnan(counts)]
            if len(counts) > 0:
                samples.append(counts)
            
        data_dict[oil] = np.array(samples)
    
    # 预处理函数
    def preprocess_signal(signal):
        # 确保输入为数值类型
        signal = np.asarray(signal, dtype=np.float64)
        baseline = np.percentile(signal, 15)
        denoised = signal - baseline
        denoised = np.clip(denoised, 0, None)  # 确保非负
        max_val = np.max(denoised)
        return denoised / max_val if max_val > 0 else denoised
    
    # 预处理所有数据
    processed_data = {}
    for oil, samples in data_dict.items():
        processed = np.array([preprocess_signal(s) for s in samples])
        processed_data[oil] = processed
    
    return processed_data

# 特征提取
def extract_features(data_dict, method='pca', n_components=10):
    X = []
    y = []
    for oil, samples in data_dict.items():
        X.extend(samples)
        y.extend([oil] * len(samples))
    
    X = np.array(X)
    y = np.array(y)
    
    if method == 'pca':
        pca = PCA(n_components=n_components)
        X_features = pca.fit_transform(X)
    elif method == 'raw':
        X_features = X
    else:
        raise ValueError("Unsupported feature extraction method")
    
    return X_features, y

# 分类模型训练和评估
def train_and_evaluate(X, y, models):
    # 数据分割 - 使用分层抽样确保各类别比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 训练和评估
    from sklearn.model_selection import GridSearchCV
    
    results = {}
    for name, config in models.items():
        # 参数调优
        gs = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        gs.fit(X_train, y_train)
        
        # 使用最佳模型
        best_model = gs.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'report': report,
            'best_params': gs.best_params_,
            'best_score': gs.best_score_
        }
        
        # 交叉验证
        cv_scores = cross_val_score(best_model, X, y, cv=5)
        results[name]['cv_mean'] = np.mean(cv_scores)
        results[name]['cv_std'] = np.std(cv_scores)
    
    return results

# 主程序
if __name__ == "__main__":
    # 定义模型和参数网格
    models = {
        'SVM': {
            'model': SVC(kernel='rbf'),
            'params': {
                'C': [0.1, 1, 10],
                'gamma': [0.01, 0.1, 1]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20]
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        }
    }

    data_dir = 'D:\\那就学\\食用油\\shiyongyou\\单个'
    processed_data = load_and_preprocess(data_dir)
    
    # 特征提取
    X_pca, y = extract_features(processed_data, method='pca')
    X_raw, _ = extract_features(processed_data, method='raw')
    
    # 模型评估
    print("=== PCA Features ===")
    pca_results = train_and_evaluate(X_pca, y, models)
    
    print("\n=== Raw Features ===")
    raw_results = train_and_evaluate(X_raw, y, models)
    
    # 结果可视化
    plt.figure(figsize=(18, 6))
    
    # 1. 准确率比较
    plt.subplot(1, 3, 1)
    for i, (name, res) in enumerate(pca_results.items()):
        plt.bar(i-0.2, res['accuracy'], width=0.4, label=f'{name} (PCA)')
        plt.bar(i+0.2, raw_results[name]['accuracy'], width=0.4, label=f'{name} (Raw)')
    plt.xticks(range(len(pca_results)), pca_results.keys())
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.legend()
    
    # 2. 数据分布检查
    plt.subplot(1, 3, 2)
    unique, counts = np.unique(y, return_counts=True)
    plt.bar(unique, counts)
    plt.title('Class Distribution')
    plt.xlabel('Oil Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # 3. PCA特征空间可视化
    plt.subplot(1, 3, 3)
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    for oil in np.unique(y):
        idx = np.where(y == oil)
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=oil, alpha=0.6)
    plt.title('t-SNE Visualization of PCA Features')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    
    plt.tight_layout()
    import os
    try:
        output_dir = os.path.abspath('D:\\那就学\\食用油\\shiyongyou\\code')
        os.makedirs(output_dir, exist_ok=True)
        img_path = os.path.join(output_dir, 'model_comparison.png')
        
        # 尝试不同的保存方法
        plt.savefig(img_path, dpi=300, bbox_inches='tight', format='png')
        print(f"\nSuccessfully saved comparison plot to: {img_path}")
        
        # 验证文件是否真的存在
        if os.path.exists(img_path):
            print(f"File verification: {img_path} exists ({os.path.getsize(img_path)} bytes)")
        else:
            print("Warning: File save operation reported success but file not found!")
            
    except Exception as e:
        print(f"\nError saving plot: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Attempted save path: {img_path}")
        
    plt.show()
    plt.close()
    
    # 打印更详细的评估报告和过拟合分析
    print("\n=== Detailed Evaluation and Overfitting Analysis ===")
    for name, res in pca_results.items():
        print(f"\n{name} (PCA):")
        print(f"Test Accuracy: {res['accuracy']:.4f}")
        print(f"CV Mean Accuracy: {res['cv_mean']:.4f} ± {res['cv_std']:.4f}")
        print(f"Train-Test Gap: {res['best_score']-res['accuracy']:.4f}")
        print(f"Best Parameters: {res['best_params']}")
        print("Classification Report:")
        print(res['report'])
        
        # 过拟合警告
        if res['best_score'] - res['accuracy'] > 0.1:
            print("⚠️ Warning: Potential overfitting (large gap between train and test accuracy)")
        elif res['cv_std'] > 0.15:
            print("⚠️ Warning: High variance in cross-validation results)")
    
    # 保存最佳模型
    from joblib import dump
    best_model_name = max(pca_results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model = models[best_model_name]['model']
    best_model.set_params(**pca_results[best_model_name]['best_params'])
    dump(best_model, 'best_oil_classifier.joblib')
    print(f"\nSaved best model ({best_model_name}) to best_oil_classifier.joblib")

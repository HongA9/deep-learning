import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import os

# 设置随机种子保证可重复性
np.random.seed(42)
tf.random.set_seed(42)

def augment_spectrum(spectrum, noise_level=0.02, shift_range=10, scale_range=0.2):
    """增强版数据增强: 添加更多变换"""
    augmented = spectrum.copy()
    
    # 1. 高斯噪声(增加噪声多样性)
    if np.random.random() > 0.2:  # 80%概率添加噪声
        noise_type = np.random.choice(['gaussian', 'laplace', 'uniform'])
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level, spectrum.shape)
        elif noise_type == 'laplace':
            noise = np.random.laplace(0, noise_level, spectrum.shape)
        else:  # uniform
            noise = np.random.uniform(-noise_level, noise_level, spectrum.shape)
        augmented += noise
    
    # 2. 随机偏移(增加偏移范围)
    if np.random.random() > 0.2:
        shift = np.random.randint(-shift_range, shift_range)
        augmented = np.roll(augmented, shift)
    
    # 3. 随机缩放(增加缩放范围)
    if np.random.random() > 0.2:
        scale = 1 + np.random.uniform(-scale_range, scale_range)
        augmented = augmented * scale
    
    # 4. 随机平滑(增加平滑多样性)
    if np.random.random() > 0.3:
        window_type = np.random.choice(['uniform', 'gaussian', 'triangular'])
        window_size = np.random.randint(3, 11)
        if window_type == 'uniform':
            window = np.ones(window_size)/window_size
        elif window_type == 'gaussian':
            window = np.exp(-0.5*((np.arange(window_size)-(window_size-1)/2)**2/((window_size/4)**2)))
            window /= window.sum()
        else:  # triangular
            window = np.bartlett(window_size)
        augmented = np.convolve(augmented, window, mode='same')
    
    # 5. 随机丢弃部分数据点(增加丢弃概率)
    if np.random.random() > 0.5:
        drop_prob = np.random.uniform(0.05, 0.2)
        drop_indices = np.random.choice(len(augmented), size=int(len(augmented)*drop_prob), replace=False)
        augmented[drop_indices] = 0
    
    # 6. 随机基线偏移(增加偏移范围)
    if np.random.random() > 0.3:
        baseline_shift = np.random.uniform(-0.2, 0.2)
        augmented += baseline_shift
    
    # 7. 新增: 随机峰值增强
    if np.random.random() > 0.3:
        peak_count = np.random.randint(1, 4)
        for _ in range(peak_count):
            idx = np.random.randint(0, len(augmented))
            peak_height = np.random.uniform(0.1, 0.5)
            augmented[idx] += peak_height
    
    # 8. 新增: 随机频率调制
    if np.random.random() > 0.3:
        fft = np.fft.fft(augmented)
        freq_shift = np.random.uniform(0.8, 1.2)
        fft = np.interp(np.arange(len(fft)), np.arange(len(fft))*freq_shift, fft)
        augmented = np.fft.ifft(fft).real
    
    return augmented

def load_data(augment=False, augment_factor=3):
    """加载拉曼光谱数据"""
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 尝试多种可能的父目录路径
    possible_parents = [
        os.path.dirname(script_dir),  # 上一级目录
        os.path.join(script_dir, ".."),  # 上一级目录(另一种写法)
        os.path.join(script_dir, "..", "shiyongyou"),  # 上一级/shiyongyou
        os.path.join(script_dir, "..", "食用油"),  # 上一级/食用油
        os.path.join(script_dir, "..", "那就学"),  # 上一级/那就学
        os.path.join(script_dir, "..", "那就学", "食用油"),  # 上一级/那就学/食用油
        os.path.join(script_dir, "..", "那就学", "食用油", "shiyongyou")  # 上一级/那就学/食用油/shiyongyou
    ]
    
    data_dir = None
    for parent in possible_parents:
        test_dir = os.path.join(parent, "单个")
        if os.path.exists(test_dir):
            data_dir = test_dir
            break
    
    if data_dir is None:
        # 如果自动查找失败，尝试使用绝对路径
        abs_path = "d:/那就学/食用油/shiyongyou/单个"
        if os.path.exists(abs_path):
            data_dir = abs_path
        else:
            raise FileNotFoundError(
                f"无法找到数据目录。尝试过的路径:\n" +
                "\n".join([os.path.join(p, "单个") for p in possible_parents]) +
                f"\n和绝对路径: {abs_path}"
            )
    
    print(f"找到数据目录: {data_dir}")
    
    oil_types = ["大豆油", "山茶油", "玉米油", "花生油", "菜籽油", "葵花籽油"]
    
    X = []
    y = []
    
    for oil in oil_types:
        oil_dir = os.path.join(data_dir, oil)
        print(f"检查油品目录: {oil_dir}")
        
        if not os.path.exists(oil_dir):
            raise FileNotFoundError(f"油品目录不存在: {oil_dir}")
            
        for file in os.listdir(oil_dir):
            if file.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(oil_dir, file), header=0, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(os.path.join(oil_dir, file), header=0, encoding='gbk')
                    except:
                        df = pd.read_csv(os.path.join(oil_dir, file), header=0, encoding='gb18030')
                # 同时使用位移和Count作为特征
                raman_shift = df.iloc[:, 0].values  # 位移列
                counts = df.iloc[:, 1].values       # Count列
                # 组合并展平特征
                combined = np.column_stack((raman_shift, counts)).flatten()
                X.append(combined)
                y.append(oil)
                
                # 数据增强
                if augment:
                    for _ in range(augment_factor):
                        augmented = augment_spectrum(counts)
                        X.append(augmented)
                        y.append(oil)
    
    # 确保所有样本特征长度一致
    max_len = max(len(x) for x in X)
    X_padded = np.array([np.pad(x, (0, max_len - len(x)), mode='constant') for x in X])
    return X_padded, np.array(y)

from sklearn.decomposition import PCA
from pywt import wavedec

def extract_wavelet_features(spectrum, wavelet='db4', level=3):
    """小波变换特征提取(处理一维光谱数据)"""
    # 直接对光谱数据进行小波变换
    coeffs = wavedec(spectrum, wavelet, level=level)
    features = []
    for coeff in coeffs:
        features.extend([np.mean(coeff), np.std(coeff), np.max(coeff), np.min(coeff)])
    return np.array(features)

def preprocess_data(X, y, use_pca=False, use_wavelet=False, method='enhanced'):
    """增强版数据预处理
    method: 
      'basic' - 仅使用位移和Count组合特征
      'count' - 仅使用Count值
      'ratio' - 使用位移与Count的比值特征
      'enhanced' - 使用多种特征组合(默认)
    """
    # 特征提取
    if method == 'count':
        X_features = np.array([x[1::2] for x in X])  # 仅提取Count值(奇数索引)
    elif method == 'ratio':
        X_features = np.array([x[1::2]/x[::2] for x in X])  # Count/位移比值
    elif method == 'basic':
        X_features = X  # 直接使用原始特征
    else:  # enhanced
        # 提取多种特征
        features = []
        for x in X:
            # 1. 原始特征
            basic = x
            # 2. 统计特征(仅Count部分)
            counts = x[1::2]
            stats = [
                np.mean(counts), np.std(counts), np.min(counts), np.max(counts),
                np.percentile(counts, 25), np.median(counts), np.percentile(counts, 75)
            ]
            # 3. 梯度特征
            grad = np.gradient(counts)
            grad_stats = [np.mean(grad), np.std(grad), np.max(grad), np.min(grad)]
            # 组合所有特征
            combined = np.concatenate([basic, stats, grad_stats])
            features.append(combined)
        X_features = np.array(features)
    
    if use_wavelet:
        wavelet_features = np.array([extract_wavelet_features(x) for x in X])
        X_features = np.concatenate([X_features, wavelet_features], axis=1)
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    # PCA降维(可选)
    if use_pca:
        pca = PCA(n_components=0.95)  # 保留95%方差
        X_scaled = pca.fit_transform(X_scaled)
        print(f"PCA reduced dimensions to: {X_scaled.shape[1]}")
    
    # 编码标签
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_onehot = to_categorical(y_encoded)
    
    return X_scaled, y_onehot, le.classes_

def build_model(input_shape, num_classes):
    """构建更强大的深度学习模型"""
    inputs = tf.keras.Input(shape=(input_shape,))
    
    # 更深的网络结构
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # 编译模型(调整学习率和优化器)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def plot_history(history):
    """绘制训练历史"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练集准确率')
    plt.plot(history.history['val_accuracy'], label='验证集准确率')
    plt.title('模型准确率')
    plt.ylabel('准确率')
    plt.xlabel('训练轮次')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练集损失')
    plt.plot(history.history['val_loss'], label='验证集损失')
    plt.title('模型损失')
    plt.ylabel('损失值')
    plt.xlabel('训练轮次')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def analyze_model(model, X, y, class_names):
    """模型解释性分析"""
    try:
        import shap
        print("\n=== Starting SHAP analysis ===")
        
        # 详细检查输入数据
        print(f"Input data shape: {X.shape}")
        print(f"Sample data point: {X[0][:5]}...")
        
        if len(X) < 100:
            print(f"Warning: Only {len(X)} samples available, using all")
            sample_size = len(X)
        else:
            sample_size = 100
            
        # 创建SHAP解释器
        print("\nCreating SHAP explainer...")
        background = X[:sample_size]
        print(f"Background data shape: {background.shape}")
        
        try:
            explainer = shap.DeepExplainer(model, background)
            print("SHAP explainer created successfully")
        except Exception as e:
            raise ValueError(f"Failed to create SHAP explainer: {str(e)}")
        
        # 计算SHAP值
        print("\nCalculating SHAP values...")
        try:
            shap_values = explainer.shap_values(background)
            print("SHAP values calculation completed")
        except Exception as e:
            raise ValueError(f"Failed to calculate SHAP values: {str(e)}")
        
        if shap_values is None:
            raise ValueError("SHAP values calculation returned None")
            
        print(f"\nSHAP values calculated for {len(shap_values)} classes")
        print(f"SHAP values shape: {[np.array(s).shape for s in shap_values]}")
        
        # 确保输出目录存在
        os.makedirs("shap_plots", exist_ok=True)
        
        # 绘制特征重要性
        print("Generating feature importance plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, background, plot_type="bar",
                         class_names=class_names, show=False)
        plt.title("Feature Importance by Class", fontsize=14)
        plt.tight_layout()
        plt.savefig('shap_plots/shap_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Feature importance plot saved to shap_plots/shap_feature_importance.png")
        
        # 绘制单个样本解释
        print("Generating sample predictions plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, background,
                         feature_names=[f"Feature {i}" for i in range(X.shape[1])],
                         class_names=class_names, show=False)
        plt.title("SHAP Values for Sample Predictions", fontsize=14)
        plt.tight_layout()
        plt.savefig('shap_plots/shap_sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Sample predictions plot saved to shap_plots/shap_sample_predictions.png")
        
    except ImportError:
        print("SHAP not installed, skipping explainability analysis")
        print("Please install with: pip install shap")
    except Exception as e:
        print(f"Error in SHAP analysis: {str(e)}")
        import traceback
        traceback.print_exc()

from sklearn.model_selection import KFold

def main():
    # 加载数据(启用数据增强)
    X, y = load_data(augment=True)
    
    # 预处理(启用小波特征提取)
    X_scaled, y_onehot, class_names = preprocess_data(
        X, y, 
        use_pca=False, 
        use_wavelet=True
    )
    
    # 初始化k折交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    accuracies = []
    
    for train, test in kfold.split(X_scaled, y_onehot):
        print(f'\n=== Fold {fold_no} ===')
        
        # 构建模型
        model = build_model(X_scaled.shape[1], y_onehot.shape[1])
        
        # 定义回调函数(增加耐心值)
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.2, patience=10, min_lr=1e-6)
        ]
        
        # 训练模型
        history = model.fit(X_scaled[train], y_onehot[train],
                          epochs=100,
                          batch_size=32,
                          validation_split=0.2,
                          callbacks=callbacks,
                          verbose=0)
        
        # 评估模型
        loss, accuracy = model.evaluate(X_scaled[test], y_onehot[test], verbose=0)
        accuracies.append(accuracy)
        print(f'Fold {fold_no} Accuracy: {accuracy:.4f}')
        print(f'Fold {fold_no} Loss: {loss:.4f}')
        
        fold_no += 1
    
    # 交叉验证结果
    print('\n=== Cross Validation Results ===')
    print(f'Average Accuracy: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})')
    
    # 训练最终模型
    print('\n=== Training Final Model ===')
    model = build_model(X_scaled.shape[1], y_onehot.shape[1])
    history = model.fit(X_scaled, y_onehot,
                      epochs=100,
                      batch_size=32,
                      validation_split=0.2,
                      callbacks=callbacks,
                      verbose=1)
    
    # 创建models目录(如果不存在)
    os.makedirs("models", exist_ok=True)
    
    # 保存模型(带版本号，使用推荐的.keras格式)
    version = 1
    model.save(f"models/oil_classifier_v{version}.keras")
    print(f"\nModel saved as oil_classifier_v{version}.keras")
    
    # 保存版本信息
    with open("models/version_info.json", "w") as f:
        json.dump({
            "current_version": version,
            "versions": [version]
        }, f)
    
    # 模型解释性分析
    print("\n=== Model Explainability Analysis ===")
    analyze_model(model, X_scaled, y_onehot, class_names)
    
    # 启动API服务提示
    print("\n=== API Service Ready ===")
    print("To start API service, run:")
    print("python api_service.py")

def create_api_service():
    """创建API服务脚本"""
    api_code = '''\
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
import time
import psutil
import GPUtil
import os
from datetime import datetime

app = Flask(__name__)

# 加载最新模型
def load_latest_model():
    with open("models/version_info.json") as f:
        version_info = json.load(f)
    latest_version = version_info["current_version"]
    return tf.keras.models.load_model(f"models/oil_classifier_v{latest_version}.h5")

model = load_latest_model()
prediction_cache = {}

# API文档
@app.route('/')
def api_docs():
    return """
<h1>食用油拉曼光谱分类API文档</h1>
<h2>端点:</h2>
<ul>
    <li><b>/predict</b> - 单样本预测 (POST)</li>
    <li><b>/batch_predict</b> - 批量预测 (POST)</li>
    <li><b>/model_info</b> - 获取模型信息 (GET)</li>
</ul>
"""

def get_system_stats():
    """获取系统性能指标"""
    stats = {
        'cpu_usage': psutil.cpu_percent(),
        'memory_usage': psutil.virtual_memory().percent,
        'gpu_usage': 0,
        'gpu_memory': 0
    }
    
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            stats['gpu_usage'] = gpus[0].load * 100
            stats['gpu_memory'] = gpus[0].memoryUtil * 100
    except:
        pass
    
    return stats

@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()
        data = request.get_json()
        spectrum = np.array(data['spectrum']).reshape(1, -1)
        
        # 获取预测前系统状态
        pre_stats = get_system_stats()
        
        # 执行预测
        prediction = model.predict(spectrum)
        
        # 获取预测后系统状态
        post_stats = get_system_stats()
        
        # 计算延迟
        latency = (time.time() - start_time) * 1000  # 毫秒
        
        return jsonify({
            'prediction': prediction.tolist(),
            'class': np.argmax(prediction),
            'performance': {
                'latency_ms': latency,
                'cpu_usage': post_stats['cpu_usage'],
                'memory_usage': post_stats['memory_usage'],
                'gpu_usage': post_stats['gpu_usage'],
                'gpu_memory': post_stats['gpu_memory']
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        start_time = time.time()
        data = request.get_json()
        spectra = np.array(data['spectra'])
        
        # 检查缓存
        cache_key = json.dumps(spectra.tolist())
        if cache_key in prediction_cache:
            cached = prediction_cache[cache_key]
            if (datetime.now() - cached['timestamp']).seconds < 3600:  # 1小时缓存
                return jsonify({
                    'predictions': cached['predictions'],
                    'classes': cached['classes'],
                    'performance': cached['performance'],
                    'cached': True
                })
        
        # 获取预测前系统状态
        pre_stats = get_system_stats()
        
        # 执行批量预测
        predictions = model.predict(spectra)
        
        # 获取预测后系统状态
        post_stats = get_system_stats()
        
        # 计算延迟
        latency = (time.time() - start_time) * 1000  # 毫秒
        
        # 缓存结果
        result = {
            'predictions': predictions.tolist(),
            'classes': np.argmax(predictions, axis=1).tolist(),
            'performance': {
                'latency_ms': latency,
                'cpu_usage': post_stats['cpu_usage'],
                'memory_usage': post_stats['memory_usage'],
                'gpu_usage': post_stats['gpu_usage'],
                'gpu_memory': post_stats['gpu_memory']
            },
            'timestamp': datetime.now()
        }
        prediction_cache[cache_key] = result
        
        return jsonify({**result, 'cached': False})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/model_info', methods=['GET'])
def model_info():
    """获取模型版本信息"""
    try:
        with open("models/version_info.json") as f:
            version_info = json.load(f)
        return jsonify({
            'status': 'success',
            'current_version': version_info['current_version'],
            'available_versions': version_info['versions']
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # 创建models目录
    os.makedirs("models", exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
'''
    with open('api_service.py', 'w') as f:
        f.write(api_code)
    
    # 创建requirements.txt
    requirements = '''
flask>=2.0.0
tensorflow>=2.6.0
numpy>=1.19.0
psutil>=5.8.0
gputil>=1.4.0
python-dateutil>=2.8.0
'''
    with open('requirements.txt', 'w') as f:
        f.write(requirements)

if __name__ == "__main__":
    main()
    create_api_service()

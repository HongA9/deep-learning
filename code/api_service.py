# -*- coding: utf-8 -*-
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

# ��������ģ��
def load_latest_model():
    with open("models/version_info.json") as f:
        version_info = json.load(f)
    latest_version = version_info["current_version"]
    return tf.keras.models.load_model(f"models/oil_classifier_v{latest_version}.keras")

model = load_latest_model()
prediction_cache = {}

# API�ĵ�
@app.route('/')
def api_docs():
    return """
<h1>ʳ�����������׷���API�ĵ�</h1>
<h2>�˵�:</h2>
<ul>
    <li><b>/predict</b> - ������Ԥ�� (POST)</li>
    <li><b>/batch_predict</b> - ����Ԥ�� (POST)</li>
    <li><b>/model_info</b> - ��ȡģ����Ϣ (GET)</li>
</ul>
"""

def get_system_stats():
    """��ȡϵͳ����ָ��"""
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
        
        # ��ȡԤ��ǰϵͳ״̬
        pre_stats = get_system_stats()
        
        # ִ��Ԥ��
        prediction = model.predict(spectrum)
        
        # ��ȡԤ���ϵͳ״̬
        post_stats = get_system_stats()
        
        # �����ӳ�
        latency = (time.time() - start_time) * 1000  # ����
        
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
        
        # ��黺��
        cache_key = json.dumps(spectra.tolist())
        if cache_key in prediction_cache:
            cached = prediction_cache[cache_key]
            if (datetime.now() - cached['timestamp']).seconds < 3600:  # 1Сʱ����
                return jsonify({
                    'predictions': cached['predictions'],
                    'classes': cached['classes'],
                    'performance': cached['performance'],
                    'cached': True
                })
        
        # ��ȡԤ��ǰϵͳ״̬
        pre_stats = get_system_stats()
        
        # ִ������Ԥ��
        predictions = model.predict(spectra)
        
        # ��ȡԤ���ϵͳ״̬
        post_stats = get_system_stats()
        
        # �����ӳ�
        latency = (time.time() - start_time) * 1000  # ����
        
        # ������
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
    """��ȡģ�Ͱ汾��Ϣ"""
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
    # ����modelsĿ¼
    os.makedirs("models", exist_ok=True)
    app.run(host='0.0.0.0', port=5000)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoShapley内存优化配置

解决ST-GPR模型在GeoShapley计算中的内存问题
优化策略：
1. 统一使用6个背景数据点
2. 对res7使用10%时空分层采样
3. 启用自适应批处理
"""

# 防止重复输出的全局标志
_PRINTED_MESSAGES = set()

def print_once(message):
    """只打印一次的函数"""
    if message not in _PRINTED_MESSAGES:
        print(message)
        _PRINTED_MESSAGES.add(message)

# 🛡️ 内存安全 + 性能优化的GeoShapley参数 (RTX 4070 SUPER + 28线程CPU)
# 策略：单进程内保守（避免KMeans内存泄漏），多进程间激进（充分利用CPU）
MEMORY_OPTIMIZED_GEOSHAPLEY_PARAMS = {
    'n_jobs': 22,  # 🚀 增加到22核心并行，补偿单进程内线程限制
    'batch_size': 30,  # 🛡️ 适度批次大小，平衡内存安全与效率
    'n_background': 8,  # 🎯 增加背景数据点，提高SHAP值稳定性
    'enable_shap_interactions': True,  # 启用SHAP交互值计算（PDP交互图需要）
    'enable_memory_cleanup': True,  # 启用内存清理
    'memory_limit_mb': 6144,  # 🛡️ 适度内存限制6GB，避免过度占用
    'use_shap_kmeans': True,  # 使用SHAP的K-means（已限制单线程）
    'timeout_per_sample': 150,  # 🔧 增加超时时间，适应更多进程
    'chunk_size': 50,  # 🛡️ 适中块大小，减少单进程内存压力
    'memory_safe_mode': True  # 🛡️ 启用内存安全模式
}

def create_memory_optimized_prediction_function(model_dict, feature_names, batch_size=30):
    """
    创建内存优化的预测函数，支持自适应批处理
    
    参数:
    model_dict: 模型字典
    feature_names: 特征名称列表
    batch_size: 初始批处理大小
    
    返回:
    预测函数
    """
    import numpy as np
    import pandas as pd
    import torch
    import gc
    import psutil
    from .stgpr_io import predict_with_st_gpr
    
    # 🚀 获取针对RTX 4070 SUPER优化的批处理配置
    from .stgpr_config import GEOSHAPLEY_CONFIG
    batch_config = GEOSHAPLEY_CONFIG.get('batch_processing', {})
    enable_adaptive = batch_config.get('enable_adaptive_batch_size', True)
    memory_threshold_mb = batch_config.get('memory_threshold_mb', 4096)  # 从1GB增加到4GB
    min_batch_size = batch_config.get('min_batch_size', 50)  # 从10增加到50
    max_batch_size = batch_config.get('max_batch_size', 200)  # 从50增加到200
    gc_frequency = batch_config.get('gc_frequency', 10)  # 从5增加到10，减少GC开销
    
    def prediction_function(x):
        """内存优化的预测函数，支持自适应批处理"""
        # 确保输入是numpy数组
        if isinstance(x, pd.DataFrame):
            x_array = x.values
        else:
            x_array = np.asarray(x)
        
        n_samples = x_array.shape[0]
        
        # 初始批次大小
        current_batch_size = min(batch_size, max_batch_size)
        
        if n_samples <= current_batch_size:
            # 样本数少，直接预测
            x_df = pd.DataFrame(x_array, columns=feature_names)
            return predict_with_st_gpr(model_dict, x_df, return_variance=False)
        else:
            # 分批预测以减少内存使用
            predictions = []
            batch_count = 0
            
            # 检查是否在主进程中（避免并行处理时的进度显示混乱）
            import multiprocessing
            is_main_process = multiprocessing.current_process().name == 'MainProcess'
            
            # 使用简洁的进度显示（只显示一次总体进度）
            total_batches = (n_samples + current_batch_size - 1) // current_batch_size
            last_progress = -1  # 记录上次显示的进度
            
            i = 0
            while i < n_samples:
                # 获取当前内存使用情况
                if enable_adaptive:
                    memory_info = psutil.virtual_memory()
                    memory_available_mb = memory_info.available / (1024 * 1024)
                    
                    # 根据可用内存调整批次大小
                    if memory_available_mb < memory_threshold_mb:
                        # 内存不足，减小批次大小
                        current_batch_size = max(min_batch_size, current_batch_size // 2)
                        # 只在第一次内存不足时显示警告
                        if batch_count == 0 and is_main_process:
                            print(f"  ⚠️ 内存受限，使用较小批次（{current_batch_size}样本/批）")
                    elif memory_available_mb > memory_threshold_mb * 2:
                        # 内存充足，可以增大批次大小
                        current_batch_size = min(max_batch_size, current_batch_size * 1.5)
                        current_batch_size = int(current_batch_size)
                
                # 计算当前批次的结束索引
                end_idx = min(i + current_batch_size, n_samples)
                batch_data = x_array[i:end_idx]
                batch_df = pd.DataFrame(batch_data, columns=feature_names)
                
                # 每批预测
                try:
                    batch_pred = predict_with_st_gpr(model_dict, batch_df, return_variance=False)
                    predictions.append(batch_pred)
                except Exception as e:
                    # 如果批处理失败，尝试减小批次大小
                    if current_batch_size > min_batch_size:
                        current_batch_size = max(min_batch_size, current_batch_size // 2)
                        continue
                    else:
                        raise e
                
                batch_count += 1
                i = end_idx
                
                # 定期垃圾回收
                if batch_count % gc_frequency == 0:
                    gc.collect()
                    # 清理GPU内存（如果使用GPU）
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # 更新进度显示（只在主进程中显示）
                if is_main_process:
                    current_progress = int((i / n_samples) * 10) * 10  # 以10%为单位
                    if current_progress > last_progress:
                        # 使用简洁的单行进度显示
                        print(f"  批处理进度: {current_progress}% ({i}/{n_samples}样本)", end='\r', flush=True)
                        last_progress = current_progress
            
            # 完成时清除进度显示行（只在主进程中）
            if is_main_process:
                print(f"  批处理完成: 100% ({n_samples}/{n_samples}样本)    ")  # 用空格覆盖之前的内容
            
            # 最终垃圾回收
            gc.collect()
            
            return np.concatenate(predictions)
    
    return prediction_function

def optimize_inducing_points_for_memory(n_samples, res_level):
    """
    🚀 针对RTX 4070 SUPER + 28线程CPU优化诱导点数量
    
    参数:
    n_samples: 样本数量
    res_level: 分辨率级别
    
    返回:
    优化后的诱导点数量
    """
    # 🚀 高性能硬件支持更多诱导点，提升模型精度
    gpu_optimized_inducing = {
        'res5': min(200, int(n_samples * 0.025)),   # 从80增加到200，GPU加速
        'res6': min(400, int(n_samples * 0.010)),   # 从150增加到400，充分利用12GB显存
        'res7': min(600, int(n_samples * 0.003))    # 从250增加到600，大幅提升精度
    }
    
    optimized_count = gpu_optimized_inducing.get(res_level, 200)
    print(f"  🎯 {res_level}诱导点优化: {optimized_count}个 (GPU加速支持)")
    return optimized_count

def select_background_data_stratified(X_train, n_background, feature_columns=None):
    """
    使用分层策略选择背景数据，确保更好的代表性
    
    参数:
    X_train: 训练数据
    n_background: 背景数据点数量
    feature_columns: 用于分层的特征列
    
    返回:
    背景数据点
    """
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # 如果是DataFrame，转换为数组
    if isinstance(X_train, pd.DataFrame):
        X_array = X_train.values
    else:
        X_array = X_train
    
    # 如果样本数少于背景点数，返回所有样本
    if len(X_array) <= n_background:
        return X_array
    
    # 使用关键特征进行分层（如elevation, temperature等）
    if feature_columns is not None and isinstance(X_train, pd.DataFrame):
        # 选择重要特征进行聚类
        key_features = ['elevation', 'temperature', 'precipitation']
        available_features = [f for f in key_features if f in X_train.columns]
        if available_features:
            X_stratify = X_train[available_features].values
            # 标准化用于聚类的特征
            scaler = StandardScaler()
            X_stratify = scaler.fit_transform(X_stratify)
        else:
            X_stratify = X_array
    else:
        X_stratify = X_array
    
    # 使用KMeans选择代表性点
    kmeans = KMeans(n_clusters=n_background, random_state=42, n_init=10)
    kmeans.fit(X_stratify)
    
    # 获取每个聚类中心最近的实际样本
    background_indices = []
    for center in kmeans.cluster_centers_:
        distances = np.sum((X_stratify - center) ** 2, axis=1)
        closest_idx = np.argmin(distances)
        if closest_idx not in background_indices:
            background_indices.append(closest_idx)
    
    # 如果不够，随机补充
    while len(background_indices) < n_background:
        idx = np.random.randint(0, len(X_array))
        if idx not in background_indices:
            background_indices.append(idx)
    
    return X_array[background_indices]

# 全局标志，避免重复输出配置信息
_CONFIG_ALREADY_APPLIED = False

def apply_memory_optimization():
    """
    应用内存优化配置（纯CPU模式）
    
    返回:
    优化后的配置字典
    """
    import os
    import torch
    global _CONFIG_ALREADY_APPLIED
    
    # 🛡️ 内存安全配置 - 避免Windows KMeans内存泄漏
    # KMeans相关库必须使用单线程以避免内存泄漏
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = '1'  # 保持1，避免KMeans内存泄漏
    
    # 🔧 PyTorch CPU模式优化
    torch.set_num_threads(4)  # PyTorch CPU计算线程数
    
    # 🛡️ MKL保持单线程（KMeans依赖MKL）
    if 'MKL_NUM_THREADS' not in os.environ:
        os.environ['MKL_NUM_THREADS'] = '1'  # 保持1，避免MKL相关内存泄漏
    
    # 🔧 检测实际计算模式
    compute_mode = "纯CPU模式"
    if torch.cuda.is_available():
        # 即使GPU可用，也可能在CPU模式下运行
        device_info = f"GPU可用但使用{compute_mode}"
    else:
        device_info = compute_mode
    
    # 只在第一次调用时输出配置信息，避免重复
    if not _CONFIG_ALREADY_APPLIED:
        print_once("已应用内存安全 + 性能平衡配置")
        print(f"  • OMP线程数: {os.environ.get('OMP_NUM_THREADS')} (避免KMeans内存泄漏)")
        print(f"  • PyTorch线程数: 4 ({device_info})")
        print(f"  • MKL线程数: {os.environ.get('MKL_NUM_THREADS')} (避免内存泄漏)")
        print(f"  • GeoShapley并行度: 20核心 (多进程CPU并行)")
        print(f"  策略: 单进程内保守，多进程间激进")
        print("已启用GeoShapley内存优化配置")
        _CONFIG_ALREADY_APPLIED = True
    
    return MEMORY_OPTIMIZED_GEOSHAPLEY_PARAMS

def optimize_kernel_computation(model):
    """
    优化核函数计算以减少内存使用
    
    参数:
    model: ST-GPR模型
    """
    if hasattr(model, 'covar_module'):
        # 启用懒惰计算
        model.covar_module.lazily_evaluate_kernels = True
        
        # 减少缓存大小
        if hasattr(model.covar_module, 'max_root_decomposition_size'):
            model.covar_module.max_root_decomposition_size = 50
    
    return model 
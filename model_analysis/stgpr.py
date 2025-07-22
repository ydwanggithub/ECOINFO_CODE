#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时空高斯过程回归模型 (ST-GPR)

本模块实现了基于PyTorch和GPyTorch的时空高斯过程回归模型，用于植被健康指数建模。
模型结构将空间维度、时间维度和特征维度整合到一个统一的核函数结构中：
(SpatialSimilarityKernel + MaternKernel) * RBFKernel(时间)

特点:
- SpatialSimilarityKernel: 自定义核函数，利用特征相似性进行预测
- MaternKernel: 处理空间相关性（经纬度）
- RBFKernel: 处理时间相关性（年份）
- 稀疏变分高斯过程: 通过inducing points提高大数据集的计算效率
"""

# 设置OMP_NUM_THREADS环境变量，避免Windows上KMeans内存泄漏问题
# 这必须在导入sklearn之前进行设置
import os
import sys
import platform
if platform.system() == 'Windows' and 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'
    print(f"在stgpr.py中设置OMP_NUM_THREADS=1，避免Windows上KMeans内存泄漏问题")

import time
import traceback
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit

# 检查依赖库是否可用
HAS_HYPEROPT = False
HAS_GPYTORCH = False
HAS_GEOSHAPLEY = False

# 添加hyperopt相关导入
try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    HAS_HYPEROPT = True
except ImportError:
    pass

# 导入配置文件
try:
    from .stgpr_config import get_config, RANDOM_SEED
    CONFIG = get_config()
except ImportError:
    RANDOM_SEED = 42
    CONFIG = {
        'model': {'num_inducing_points': 500, 'batch_size': 200, 'num_iterations': 1000, 'use_lbfgs': True},
        'optimizer': {'adam_learning_rate': 0.01, 'gradient_clip_norm': 10.0},
        'kernel': {'lengthscale_lower_bound': 1e-3, 'variance_lower_bound': 1e-5}
    }

# 检查PyTorch和GPyTorch依赖
try:
    import torch
    import gpytorch
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
    from gpytorch.means import ConstantMean
    from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
    HAS_GPYTORCH = True
except ImportError:
    pass

# 检查GeoShapley库
try:
    from geoshapley import GeoShapleyExplainer
    HAS_GEOSHAPLEY = True
except ImportError:
    pass

# 从core模块导入ensure_dir_exists函数
from .core import ensure_dir_exists

# 从stgpr_model模块导入STGPRModel和SpatialSimilarityKernel
from .stgpr_model import STGPRModel, SpatialSimilarityKernel

def select_inducing_points_spatiotemporal(X_train, n_inducing_points, h3_col='h3_index', year_col='year', random_state=42, feature_columns=None, use_kmeans_fallback=True, return_indices=False):
    """
    选择诱导点，默认使用KMeans聚类，可选时空分层采样
    
    参数:
    X_train: 训练数据，包含特征和可能的h3_index、year列
    n_inducing_points: 需要的诱导点数量
    h3_col: H3网格索引列名
    year_col: 年份列名
    random_state: 随机种子
    feature_columns: 要返回的特征列列表（如果为None，自动确定）
    use_kmeans_fallback: 是否使用KMeans（默认True）。如果为False，则尝试时空分层采样
    return_indices: 是否返回索引而不是数据
    
    返回:
    inducing_points: 选中的诱导点数组
    """
    np.random.seed(random_state)
    
    # 决定是否使用时空分层采样
    # 默认使用KMeans，因为它在特征空间中选择更有代表性的点
    use_spatiotemporal = False
    if not use_kmeans_fallback:
        # 只有在明确不使用KMeans回退时才使用时空分层
        use_spatiotemporal = True
        print(f"  尝试时空分层采样选择诱导点...")
    else:
        print(f"  使用KMeans聚类选择诱导点（在特征空间中选择代表性点）")
    
    # 如果是DataFrame且有时空信息，尝试使用时空分层
    if use_spatiotemporal and isinstance(X_train, pd.DataFrame) and h3_col in X_train.columns and year_col in X_train.columns:
        print(f"  使用时空分层采样选择{n_inducing_points}个诱导点...")
        
        # 获取唯一的空间和时间值
        unique_h3 = X_train[h3_col].unique()
        unique_years = X_train[year_col].unique()
        
        n_h3 = len(unique_h3)
        n_years = len(unique_years)
        
        # 计算每个维度应该选择多少个值
        # 改进的策略：确保更好的覆盖率
        # 1. 对于时间维度，至少覆盖50%的年份
        min_years = max(n_years // 2, 5)  # 至少5年或一半年份
        # 2. 对于空间维度，根据诱导点数量动态调整
        
        if n_inducing_points >= n_h3 * n_years * 0.5:
            # 如果诱导点很多，使用大部分网格和年份
            n_h3_select = min(int(n_h3 * 0.8), n_h3)
            n_years_select = min(int(n_years * 0.8), n_years)
        elif n_inducing_points >= 1000:
            # 中等数量的诱导点
            n_h3_select = min(int(np.sqrt(n_inducing_points / min_years)), n_h3)
            n_years_select = min(max(min_years, int(n_inducing_points / n_h3_select)), n_years)
        else:
            # 较少的诱导点，优先保证时间覆盖
            n_years_select = min(min_years, n_years)
            n_h3_select = min(max(int(n_inducing_points / n_years_select), 10), n_h3)
        
        # 确保合理的范围
        n_h3_select = max(min(n_h3_select, n_h3), min(10, n_h3))
        n_years_select = max(min(n_years_select, n_years), min(5, n_years))
        
        # 随机选择H3网格和年份
        selected_h3 = np.random.choice(unique_h3, size=n_h3_select, replace=False)
        selected_years = np.random.choice(unique_years, size=n_years_select, replace=False)
        
        # 创建网格组合
        inducing_indices = []
        for h3 in selected_h3:
            for year in selected_years:
                mask = (X_train[h3_col] == h3) & (X_train[year_col] == year)
                indices = X_train.index[mask].tolist()
                if indices:
                    # 从每个网格-年份组合中随机选择一个点
                    inducing_indices.append(np.random.choice(indices))
        
        # 如果诱导点不够，补充一些随机点
        if len(inducing_indices) < n_inducing_points:
            remaining = n_inducing_points - len(inducing_indices)
            all_indices = list(set(range(len(X_train))) - set(inducing_indices))
            additional = np.random.choice(all_indices, size=min(remaining, len(all_indices)), replace=False)
            inducing_indices.extend(additional)
        
        # 如果诱导点太多，随机选择子集
        if len(inducing_indices) > n_inducing_points:
            inducing_indices = np.random.choice(inducing_indices, size=n_inducing_points, replace=False)
        
        # 获取诱导点的特征值
        if isinstance(X_train, pd.DataFrame):
            if feature_columns is not None:
                # 如果指定了特征列，使用指定的列
                inducing_points = X_train.iloc[inducing_indices][feature_columns].values
            else:
                # 否则自动确定：只获取数值列，排除h3_index和其他非特征列
                numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
                # 排除h3_index即使它是数值型（虽然现在是字符串）
                # 同时排除可能的目标变量（VHI）
                exclude_cols = [h3_col, 'h3_index', 'original_h3_index', 'VHI']
                numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
                inducing_points = X_train.iloc[inducing_indices][numeric_cols].values
        else:
            inducing_points = X_train.iloc[inducing_indices].values
            
        print(f"    选择了{n_h3_select}个H3网格 × {n_years_select}个年份")
        print(f"    最终得到{len(inducing_points)}个诱导点")
        
        if return_indices:
            return inducing_indices
        else:
            return inducing_points.astype(np.float32)
    else:
        # 使用KMeans方法选择诱导点
        # 确保是numpy数组
        if isinstance(X_train, pd.DataFrame):
            if feature_columns is not None:
                # 如果指定了特征列，使用指定的列
                X_array = X_train[feature_columns].values
            else:
                # 只使用数值列，排除非特征列
                numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
                exclude_cols = [h3_col, 'h3_index', 'original_h3_index', 'VHI']
                numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
                X_array = X_train[numeric_cols].values
        else:
            X_array = X_train
            
        if len(X_array) > n_inducing_points:
            # 强制设置OMP_NUM_THREADS=1以避免内存泄漏
            if platform.system() == 'Windows':
                old_value = os.environ.get('OMP_NUM_THREADS', None)
                os.environ['OMP_NUM_THREADS'] = '1'
            
            kmeans = KMeans(n_clusters=n_inducing_points, random_state=random_state, n_init=10)
            kmeans.fit(X_array)
            inducing_points = kmeans.cluster_centers_
            
            if platform.system() == 'Windows' and 'old_value' in locals():
                if old_value is not None:
                    os.environ['OMP_NUM_THREADS'] = old_value
                    
            print(f"    使用KMeans聚类选择了{len(inducing_points)}个诱导点")
        else:
            inducing_points = X_array.copy()
    
    if return_indices:
        # KMeans不返回索引，返回None表示需要使用聚类中心
        return None
    else:
        return inducing_points.astype(np.float32)

def create_stgpr_model(X_train, y_train, num_inducing_points=None, batch_size=None, device=None,
                      spatial_variance=None, temporal_variance=None, feature_variance=None,
                      spatial_lengthscale=None, temporal_lengthscale=None, feature_lengthscale=None,
                      scaler=None, X_train_full=None):
    """
    创建ST-GPR模型实例
    
    参数:
    X_train: 训练特征矩阵
    y_train: 训练目标变量
    num_inducing_points: 诱导点数量，如果为None则使用配置中的设置
    batch_size: 批处理大小，如果为None则使用配置中的设置
    device: 计算设备，如果为None则自动选择
    spatial_variance: 空间核函数的方差参数
    temporal_variance: 时间核函数的方差参数
    feature_variance: 特征核函数的方差参数
    spatial_lengthscale: 空间核函数的长度尺度参数
    temporal_lengthscale: 时间核函数的长度尺度参数
    feature_lengthscale: 特征核函数的长度尺度参数
    scaler: 已废弃，保留仅为兼容性
    X_train_full: 包含完整信息的原始DataFrame（用于诱导点选择，包含h3_index和year）
    
    返回:
    tuple: (model, X_train_tensor, y_train_tensor, device)
    """
    # 检查num_inducing_points参数
    if num_inducing_points is None:
        num_inducing_points = CONFIG['model']['num_inducing_points']
    
    # 保存原始DataFrame用于诱导点选择
    # 优先使用X_train_full（包含h3_index和year），如果没有则使用X_train
    X_train_original = X_train_full if X_train_full is not None else (X_train if isinstance(X_train, pd.DataFrame) else None)
        
    # 确保输入数据是浮点类型
    if isinstance(X_train, pd.DataFrame):
        X_train_np = X_train.select_dtypes(include=[np.number]).values.astype(np.float32)
    else:
        X_train_np = np.asarray(X_train).astype(np.float32)
    
    # 🔴 关键修改：直接使用原始数据，不进行标准化
    X_train_np_used = X_train_np
    
    # 修复：正确处理pandas Series类型的y_train
    if isinstance(y_train, pd.Series):
        y_train_np = np.asarray(y_train.values).reshape(-1).astype(np.float32)
    else:
        y_train_np = np.asarray(y_train).reshape(-1).astype(np.float32)
    
    # 从训练数据中选择诱导点
    if X_train_np_used.shape[0] > num_inducing_points:
        # 如果有原始DataFrame，调用诱导点选择函数
        if X_train_original is not None:
            Z_indices = select_inducing_points_spatiotemporal(
                X_train_original, 
                num_inducing_points,
                h3_col='h3_index',
                year_col='year', 
                random_state=RANDOM_SEED,
                return_indices=True  # 返回索引
            )
            # 从原始数据中提取诱导点
            if Z_indices is not None and len(Z_indices) > 0:
                # 使用时空分层采样的索引
                Z_np = X_train_np_used[Z_indices]
            else:
                # 使用KMeans（这是默认行为，不是警告）
                kmeans = KMeans(n_clusters=num_inducing_points, random_state=RANDOM_SEED, n_init=10)
                kmeans.fit(X_train_np_used)
                Z_np = kmeans.cluster_centers_.astype(np.float32)
        else:
            # 直接使用KMeans
            print(f"  使用KMeans聚类选择诱导点")
            # 强制设置OMP_NUM_THREADS=1以彻底解决内存泄漏问题
            if platform.system() == 'Windows':
                old_value = os.environ.get('OMP_NUM_THREADS', None)
                os.environ['OMP_NUM_THREADS'] = '1'
            
            kmeans = KMeans(n_clusters=num_inducing_points, random_state=RANDOM_SEED, n_init=10)
            kmeans.fit(X_train_np_used)  # 在原始数据上进行聚类
            Z_np = kmeans.cluster_centers_.astype(np.float32)
    else:
        Z_np = X_train_np_used.copy()
        num_inducing_points = X_train_np_used.shape[0]
    
    # 获取特征维度
    input_dim = X_train_np.shape[1]
    
    # 确定空间、时间和特征的索引
    spatial_dims = list(range(2))
    temporal_dims = [input_dim - 1]
    feature_dims = list(set(range(input_dim)) - set(spatial_dims) - set(temporal_dims))
    
    # 转换为PyTorch张量
    X_train_tensor = torch.from_numpy(X_train_np_used)  # 使用原始数据
    y_train_tensor = torch.from_numpy(y_train_np)
    inducing_points = torch.from_numpy(Z_np)
    
    # 检测设备
    if device is None:
        device = torch.device('cpu')
    
    # 将数据移动到设备
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    inducing_points = inducing_points.to(device)
    
    # 🔴 修复：使用原始数据的方差计算sigma_values（用于核函数的方差归一化）
    sigma_values = torch.tensor(np.var(X_train_np_used[:, feature_dims], axis=0) + 1e-5, dtype=torch.float32).to(device)
    
    # 创建特征权重
    feature_weights = torch.ones(len(feature_dims), device=device)
    
    # 从配置中获取核函数参数
    kernel_config = CONFIG['kernel']
    
    # 使用传入的参数或配置中的默认值
    spatial_variance = spatial_variance if spatial_variance is not None else kernel_config.get('spatial_variance_init', 1.0)
    temporal_variance = temporal_variance if temporal_variance is not None else kernel_config.get('temporal_variance_init', 1.0)
    feature_variance = feature_variance if feature_variance is not None else kernel_config.get('feature_variance_init', 1.0)
    spatial_lengthscale = spatial_lengthscale if spatial_lengthscale is not None else kernel_config.get('spatial_lengthscale_init', 1.0)
    temporal_lengthscale = temporal_lengthscale if temporal_lengthscale is not None else kernel_config.get('temporal_lengthscale_init', 1.0)
    feature_lengthscale = feature_lengthscale if feature_lengthscale is not None else kernel_config.get('feature_lengthscale_init', 1.0)
    
    # 创建模型
    model = STGPRModel(
        inducing_points=inducing_points,
        input_dim=input_dim,
        spatial_dims=spatial_dims,
        temporal_dims=temporal_dims,
        feature_dims=feature_dims,
        spatial_variance=spatial_variance,
        temporal_variance=temporal_variance,
        feature_variance=feature_variance,
        spatial_lengthscale=spatial_lengthscale,
        temporal_lengthscale=temporal_lengthscale,
        feature_lengthscale=feature_lengthscale,
        feature_weights=feature_weights,
        sigma_values=sigma_values
    )
    
    # 将模型移动到设备上
    model = model.to(device)
    
    # 应用参数约束
    lengthscale_lower_bound = kernel_config.get('lengthscale_lower_bound', 1e-3)
    variance_lower_bound = kernel_config.get('variance_lower_bound', 1e-5)
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'lengthscale' in name:
                param.data.clamp_(min=lengthscale_lower_bound)
            elif 'outputscale' in name or 'raw_outputscale' in name:
                param.data.clamp_(min=variance_lower_bound)
    
    return model, X_train_tensor, y_train_tensor, device

def load_stgpr_model(model_path, device=None):
    """
    从保存的模型文件中加载STGPR模型
    
    参数:
    model_path: 模型文件路径
    device: 计算设备，如果为None则自动选择
    
    返回:
    dict: 包含加载的模型和相关元数据的字典
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 确定设备
    if device is None:
        device = torch.device('cpu')
    
    # 加载模型数据
    checkpoint = torch.load(model_path, map_location=device)
    
    # 提取模型元数据
    feature_names = checkpoint.get('feature_names', None)
    scaler = checkpoint.get('scaler', None)
    metrics = checkpoint.get('metrics', {})
    
    # 如果没有特征名称列表，则创建一个默认的
    if feature_names is None:
        # 尝试推断特征维度
        for name, param in checkpoint['model_state_dict'].items():
            if 'lengthscale' in name and len(param.shape) > 0:
                n_features = param.shape[0]
                feature_names = [f'feature_{i}' for i in range(n_features)]
                break
    
    # 确定输入维度
    input_dim = len(feature_names) if feature_names else 19  # 默认值：lat, lon, 16个特征, year
    
    # 创建诱导点
    inducing_points = None
    for name, param in checkpoint['model_state_dict'].items():
        if 'inducing_points' in name:
            inducing_points = param
            break
    
    if inducing_points is None:
        num_inducing = CONFIG['model']['num_inducing_points']
        inducing_points = torch.randn(num_inducing, input_dim, device=device)
    
    # 确定空间、时间和特征的索引
    spatial_dims = list(range(2))
    temporal_dims = [input_dim - 1]
    feature_dims = list(set(range(input_dim)) - set(spatial_dims) - set(temporal_dims))
    
    # 创建模型实例
    model = STGPRModel(
        inducing_points=inducing_points,
        input_dim=input_dim,
        spatial_dims=spatial_dims,
        temporal_dims=temporal_dims,
        feature_dims=feature_dims
    )
    
    # 创建似然函数
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    # 加载参数
    model.load_state_dict(checkpoint['model_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    
    # 移动到设备
    model = model.to(device)
    likelihood = likelihood.to(device)
    
    # 设置为评估模式
    model.eval()
    likelihood.eval()
    
    return {
        'model': model,
        'likelihood': likelihood,
        'feature_names': feature_names,
        'scaler': scaler,
        'metrics': metrics,
        'device': device
    }

def predict_with_stgpr(model_dict, X_new, return_variance=False):
    """
    使用加载的STGPR模型进行预测
    
    参数:
    model_dict: 由load_stgpr_model函数返回的模型字典
    X_new: 新的特征数据，DataFrame或numpy数组
    return_variance: 是否返回预测方差
    
    返回:
    mean: 预测均值，返回一维numpy数组，与GeoShapleyExplainer期望的格式一致
    variance (可选): 预测方差，返回一维numpy数组
    """
    model = model_dict['model']
    likelihood = model_dict['likelihood']
    device = model_dict['device']
    
    # 预处理输入数据
    if isinstance(X_new, pd.DataFrame):
        X_new_np = X_new.values
    else:
        X_new_np = np.asarray(X_new)
    
    # 🔴 关键修改：直接使用原始数据，不进行标准化
    X_new_used = X_new_np
    
    # 转换为PyTorch张量
    X_new_tensor = torch.tensor(X_new_used, dtype=torch.float32).to(device)
    
    # 进行预测
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        if return_variance:
            # 获取完整的预测分布
            pred_dist = likelihood(model(X_new_tensor))
            mean = pred_dist.mean.cpu().numpy()
            variance = pred_dist.variance.cpu().numpy()
            
            # 确保返回一维数组，GeoShapley期望一维输出
            if len(mean.shape) > 1:
                mean = mean.flatten()
            if len(variance.shape) > 1:
                variance = variance.flatten()
                
            return mean, variance
        else:
            # 只返回均值预测
            pred_dist = model(X_new_tensor)
            mean = pred_dist.mean.cpu().numpy()
            
            # 确保返回一维数组，GeoShapley期望一维输出
            if len(mean.shape) > 1:
                mean = mean.flatten()
                
            return mean

# hyperopt优化相关函数
def optimize_stgpr_hyperparameters(X_train, y_train, feature_names, num_inducing_points=50, 
                                  spatial_dims=[0, 1], temporal_dims=[-1], max_evals=10, device=None,
                                  scaler=None, X_train_full=None):
    """
    使用hyperopt优化STGPR核函数的超参数
    
    参数:
    X_train: 训练特征矩阵（可以是DataFrame或numpy数组）
    y_train: 训练目标变量
    feature_names: 特征名称列表
    num_inducing_points: 诱导点数量
    spatial_dims: 空间维度索引列表
    temporal_dims: 时间维度索引列表
    max_evals: hyperopt最大评估次数
    device: 计算设备
    scaler: 已废弃，保留仅为兼容性
    X_train_full: 包含完整信息的原始DataFrame（用于诱导点选择，包含h3_index和year）
    
    返回:
    dict: 包含最优参数和训练好的模型的字典
    """
    if not HAS_HYPEROPT:
        return None
    
    # 保存原始DataFrame用于诱导点选择
    # 优先使用X_train_full（包含h3_index和year），如果没有则使用X_train
    X_train_original = X_train_full if X_train_full is not None else (X_train if isinstance(X_train, pd.DataFrame) else None)
    
    # 确保数据是numpy数组
    if isinstance(X_train, pd.DataFrame):
        X_train_np = X_train.select_dtypes(include=[np.number]).values.astype(np.float32)
    else:
        X_train_np = np.asarray(X_train).astype(np.float32)
    
    # 🔴 关键修改：直接使用原始数据，不进行标准化
    X_train_np_used = X_train_np
    
    # 修复：正确处理pandas Series类型的y_train
    if isinstance(y_train, pd.Series):
        y_train_np = np.asarray(y_train.values).reshape(-1).astype(np.float32)
    else:
        y_train_np = np.asarray(y_train).reshape(-1).astype(np.float32)
    
    # 确定设备
    if device is None:
        device = torch.device('cpu')
    
    # 确定特征维度
    input_dim = X_train_np.shape[1]
    
    # 如果feature_dims为None，自动计算（除了空间和时间维度外的所有维度）
    all_dims = set(range(input_dim))
    spatial_temporal_dims = set(spatial_dims + temporal_dims)
    feature_dims = list(all_dims - spatial_temporal_dims)
    
    # 确保num_inducing_points不为None
    if num_inducing_points is None:
        num_inducing_points = min(500, X_train_np.shape[0] // 10)  # 默认使用数据点数量的10%作为诱导点数量
    
    # 从训练数据中选择诱导点
    if X_train_np_used.shape[0] > num_inducing_points:
        # 如果有原始DataFrame，调用诱导点选择函数
        if X_train_original is not None:
            Z_indices = select_inducing_points_spatiotemporal(
                X_train_original, 
                num_inducing_points,
                h3_col='h3_index',
                year_col='year', 
                random_state=RANDOM_SEED,
                return_indices=True  # 返回索引
            )
            # 从原始数据中提取诱导点
            if Z_indices is not None and len(Z_indices) > 0:
                # 使用时空分层采样的索引
                Z_np = X_train_np_used[Z_indices]
            else:
                # 使用KMeans（这是默认行为，不是警告）
                kmeans = KMeans(n_clusters=num_inducing_points, random_state=RANDOM_SEED, n_init=10)
                kmeans.fit(X_train_np_used)
                Z_np = kmeans.cluster_centers_.astype(np.float32)
        else:
            # 直接使用KMeans
            print(f"  使用KMeans聚类选择诱导点")
            # 强制设置OMP_NUM_THREADS=1以彻底解决内存泄漏问题
            if platform.system() == 'Windows':
                old_value = os.environ.get('OMP_NUM_THREADS', None)
                os.environ['OMP_NUM_THREADS'] = '1'
            
            kmeans = KMeans(n_clusters=num_inducing_points, random_state=RANDOM_SEED, n_init=10)
            kmeans.fit(X_train_np_used)  # 在原始数据上进行聚类
            Z_np = kmeans.cluster_centers_.astype(np.float32)
    else:
        Z_np = X_train_np_used.copy()
        num_inducing_points = X_train_np_used.shape[0]
    
    # 获取特征维度
    input_dim = X_train_np.shape[1]
    
    # 确定空间、时间和特征的索引
    spatial_dims = list(range(2))
    temporal_dims = [input_dim - 1]
    feature_dims = list(set(range(input_dim)) - set(spatial_dims) - set(temporal_dims))
    
    # 🔴 修复：使用原始数据的方差（用于核函数的方差归一化）
    sigma_values = torch.tensor(np.var(X_train_np_used[:, feature_dims], axis=0) + 1e-5, dtype=torch.float32)
    
    # 从配置中获取核函数参数
    kernel_config = CONFIG['kernel']
    
    # 定义参数空间
    space = {
        # 方差参数
        'spatial_variance': hp.loguniform('spatial_variance', np.log(0.01), np.log(5.0)),
        'temporal_variance': hp.loguniform('temporal_variance', np.log(0.01), np.log(5.0)),
        'feature_variance': hp.loguniform('feature_variance', np.log(0.01), np.log(5.0)),
        
        # 长度尺度
        'spatial_lengthscale': hp.loguniform('spatial_lengthscale', np.log(0.1), np.log(5.0)),
        'temporal_lengthscale': hp.loguniform('temporal_lengthscale', np.log(0.5), np.log(5.0)),
        'feature_lengthscale': hp.loguniform('feature_lengthscale', np.log(0.1), np.log(5.0)),
        
        # MatérnKernel参数nu
        'nu': hp.choice('nu', [0.5, 1.5, 2.5]),
        
        # 复合核权重w
        'w': hp.uniform('w', 0.05, 0.2)
    }
    
    # 特征权重
    for i in range(len(feature_dims)):
        space[f'p{i}'] = hp.uniform(f'p{i}', 0.1, 5)
    
    # 添加评估计数器
    eval_count = [0]  # 使用列表以便在闭包中修改
    
    # 目标函数
    def objective(params):
        eval_count[0] += 1
        try:
            # 处理参数
            cleaned_params = {}
            p_function_values = [1.0] * len(feature_dims)
            
            for k, v in params.items():
                if k.startswith('p') and k[1:].isdigit():
                    idx = int(k[1:])
                    if idx < len(feature_dims):
                        p_function_values[idx] = float(v)
                elif k == 'nu':
                    # 特殊处理nu参数，确保它是有效的选择值
                    if isinstance(v, (int, float)):
                        # 如果是数值，映射到最近的有效值
                        valid_nu_values = [0.5, 1.5, 2.5]
                        cleaned_params[k] = min(valid_nu_values, key=lambda x: abs(x - float(v)))
                    else:
                        # 如果是索引，直接使用
                        valid_nu_values = [0.5, 1.5, 2.5]
                        nu_idx = int(v) if isinstance(v, (int, float)) else 1
                        cleaned_params[k] = valid_nu_values[min(nu_idx, len(valid_nu_values)-1)]
                else:
                    cleaned_params[k] = float(v)
            
            # 将PyTorch张量移动到设备上
            X_train_tensor = torch.tensor(X_train_np_used, dtype=torch.float32).to(device)  # 使用原始数据
            y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).to(device)
            inducing_points = torch.tensor(Z_np, dtype=torch.float32).to(device)
            torch_sigma_values = sigma_values.clone().detach().to(device)
            feature_weights = torch.tensor(p_function_values, dtype=torch.float32).to(device)
            
            # 创建模型
            model = STGPRModel(
                inducing_points=inducing_points,
                input_dim=input_dim,
                spatial_dims=spatial_dims,
                temporal_dims=temporal_dims,
                feature_dims=feature_dims,
                spatial_variance=cleaned_params.get('spatial_variance', 1.0),
                temporal_variance=cleaned_params.get('temporal_variance', 1.0),
                feature_variance=cleaned_params.get('feature_variance', 1.0),
                spatial_lengthscale=cleaned_params.get('spatial_lengthscale', 1.0),
                temporal_lengthscale=cleaned_params.get('temporal_lengthscale', 1.0),
                feature_lengthscale=cleaned_params.get('feature_lengthscale', 1.0),
                feature_weights=feature_weights,
                sigma_values=torch_sigma_values,
                nu=cleaned_params.get('nu', 2.5),
                w=cleaned_params.get('w', 0.1)
            ).to(device)
            
            # 模型参数约束
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'lengthscale' in name:
                        param.data.clamp_(min=kernel_config.get('lengthscale_lower_bound', 1e-3))
                    elif 'outputscale' in name or 'raw_outputscale' in name:
                        param.data.clamp_(min=kernel_config.get('variance_lower_bound', 1e-5))
            
            # 创建似然函数和损失函数
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(y_train_np), combine_terms=True)
            
            # 创建优化器和数据加载器
            optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': likelihood.parameters()}
            ], lr=0.01)
            
            # 简化训练 - 针对不同数据规模优化
            if X_train_np.shape[0] > 150000:  # res7级别的大数据集
                max_iter = 3  # 极少的迭代次数，只为快速评估
                batch_size = min(500, len(X_train_tensor))  # 更大的批次
            elif X_train_np.shape[0] > 50000:  # res6级别
                max_iter = 5
                batch_size = min(300, len(X_train_tensor))
            elif X_train_np.shape[0] > 10000:  # 中等数据集
                max_iter = 8
                batch_size = min(200, len(X_train_tensor))
            else:  # 小数据集
                max_iter = 10
                batch_size = min(100, len(X_train_tensor))
            
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            model.train()
            likelihood.train()
            
            # 训练循环
            for i in range(max_iter):
                epoch_loss = 0.0
                num_batches = 0
                
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    output = model(X_batch)
                    loss = -mll(output, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1
            
            # 评估性能
            model.eval()
            likelihood.eval()
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                f_preds = model(X_train_tensor)
                mean_pred = f_preds.mean.cpu().numpy()
            
            # 计算RMSE作为损失
            loss_value = np.sqrt(mean_squared_error(y_train_np, mean_pred))
            r2_value = r2_score(y_train_np, mean_pred)
            
            # 确定迭代次数
            iterations = min(40, CONFIG['model']['num_iterations']) if X_train_np.shape[0] > 100000 else min(200, CONFIG['model']['num_iterations'])
            
            # 更新进度显示
            progress = eval_count[0] / max_evals * 100
            print(f"\r  进度: {progress:.0f}% [{eval_count[0]}/{max_evals}] | 当前RMSE: {loss_value:.6f} | 当前R²: {r2_value:.4f}", end="", flush=True)
            
            return {
                'loss': loss_value,
                'status': STATUS_OK,
                'model': model,
                'likelihood': likelihood,
                'rmse': loss_value,
                'r2': r2_value,
                'iterations': iterations,
                'params': params
            }
        except Exception as e:
            # 更新进度显示（失败情况）
            progress = eval_count[0] / max_evals * 100
            print(f"\r  进度: {progress:.0f}% [{eval_count[0]}/{max_evals}] | 评估失败: {str(e)[:30]}...", end="", flush=True)
            return {'loss': 1e10, 'status': 'fail', 'exception': str(e)}
    
    try:
        # 设置随机种子
        np.random.seed(RANDOM_SEED)
        
        # 使用hyperopt优化
        trials = Trials()
        
        # 禁用hyperopt的默认进度条，使用自定义进度显示
        import logging
        logging.getLogger('hyperopt').setLevel(logging.WARNING)
        
        # 自定义进度显示
        print(f"🔍 开始贝叶斯优化 (共{max_evals}次评估)...")
        
        best = fmin(
            fn=lambda params: objective(params)['loss'],
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            show_progressbar=False  # 禁用默认进度条
        )
        
        # 换行以结束进度显示
        print()  # 换行
        
        # 使用最佳参数重新创建模型
        final_params = {}
        for k, v in best.items():
            if k == 'nu':
                # 特殊处理nu参数
                valid_nu_values = [0.5, 1.5, 2.5]
                if isinstance(v, (int, float)):
                    final_params[k] = min(valid_nu_values, key=lambda x: abs(x - float(v)))
                else:
                    nu_idx = int(v) if isinstance(v, (int, float)) else 1
                    final_params[k] = valid_nu_values[min(nu_idx, len(valid_nu_values)-1)]
            else:
                final_params[k] = float(v)
        
        best_model_result = objective(final_params)
        
        # 添加最佳参数到结果
        if best_model_result['status'] == STATUS_OK:
            best_model_result['best_params'] = final_params
            return best_model_result
        
    except Exception as e:
        print(f"优化过程中出错: {e}")
    
    # 失败时使用默认参数
    default_params = {
        'spatial_variance': kernel_config.get('spatial_variance_init', 1.0),
        'temporal_variance': kernel_config.get('temporal_variance_init', 1.0),
        'feature_variance': kernel_config.get('feature_variance_init', 1.0),
        'spatial_lengthscale': kernel_config.get('spatial_lengthscale_init', 1.0),
        'temporal_lengthscale': kernel_config.get('temporal_lengthscale_init', 1.0),
        'feature_lengthscale': kernel_config.get('feature_lengthscale_init', 1.0),
        'nu': 2.5,
        'w': kernel_config.get('w_init', 0.1)
    }
    
    # 将默认权重添加到参数中
    for i in range(len(feature_dims)):
        default_params[f'p{i}'] = 1.0
    
    # 使用默认参数创建模型
    result = objective(default_params)
    if result['status'] == STATUS_OK:
        result['best_params'] = default_params
        return result
    
    return None

def train_stgpr_model(model, likelihood, X_train_tensor, y_train_tensor, num_iterations=None, 
                     use_lbfgs=None, batch_size=None, callback=None, device=None):
    """
    训练STGPR模型
    
    参数:
    model: STGPR模型实例
    likelihood: 似然函数
    X_train_tensor: 训练特征张量
    y_train_tensor: 训练目标张量
    num_iterations: 训练迭代次数，如果为None则使用配置中的设置
    use_lbfgs: 是否使用L-BFGS优化器，如果为None则使用配置中的设置
    batch_size: 批处理大小，如果为None则使用配置中的设置
    callback: 训练过程中的回调函数，用于显示进度和中间结果
    device: 计算设备
    
    返回:
    model: 训练好的模型
    likelihood: 训练好的似然函数
    metrics: 训练指标字典
    """
    # 使用配置中的默认值
    if num_iterations is None:
        num_iterations = CONFIG['model']['num_iterations']
    if use_lbfgs is None:
        use_lbfgs = CONFIG['model'].get('use_lbfgs', True)
    if batch_size is None:
        batch_size = CONFIG['model']['batch_size']
    
    # 自动调整批处理大小和迭代次数
    if X_train_tensor.shape[0] > 100000:  # 大数据集，降低计算成本
        batch_size = min(batch_size, 500)
        num_iterations = min(num_iterations, 50)
    
    # 设置训练模式
    model.train()
    likelihood.train()
    
    # 准备训练数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 创建损失函数
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_train_tensor.size(0), combine_terms=True)
    
    if use_lbfgs:
        # 使用L-BFGS优化器（适用于小数据集和中等数据集）
        optimizer = torch.optim.LBFGS(
            [{'params': model.parameters()}, {'params': likelihood.parameters()}],
            line_search_fn="strong_wolfe",
            max_iter=5
        )
        
        # 训练循环
        for i in range(num_iterations):
            # 定义闭包函数以计算损失
            def closure():
                optimizer.zero_grad()
                output = model(X_train_tensor)
                loss = -mll(output, y_train_tensor)
                loss.backward()
                return loss
            
            # 执行优化步骤
            loss = optimizer.step(closure)
            
            # 调用回调函数
            if callback is not None and i % max(1, num_iterations // 10) == 0:
                callback(i, num_iterations, loss.item())
    else:
        # 使用Adam优化器（适用于大数据集）
        optimizer = torch.optim.Adam(
            [{'params': model.parameters()}, {'params': likelihood.parameters()}],
            lr=CONFIG['optimizer'].get('adam_learning_rate', 0.01)
        )
        
        # 梯度裁剪值
        grad_clip_norm = CONFIG['optimizer'].get('gradient_clip_norm', 10.0)
        
        # 训练循环
        for i in range(num_iterations):
            epoch_loss = 0.0
            num_batches = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    torch.nn.utils.clip_grad_norm_(likelihood.parameters(), grad_clip_norm)
                
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            
            # 调用回调函数
            if callback is not None and i % max(1, num_iterations // 10) == 0:
                avg_loss = epoch_loss / max(1, num_batches)
                callback(i, num_iterations, avg_loss)
    
    # 计算训练集上的性能指标
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # 分批预测，避免内存溢出
        all_means = []
        all_targets = []
        
        for X_batch, y_batch in train_loader:
            pred_dist = likelihood(model(X_batch))
            means = pred_dist.mean.cpu().numpy()
            targets = y_batch.cpu().numpy()
            
            all_means.append(means)
            all_targets.append(targets)
        
        # 合并所有批次的预测
        train_preds = np.concatenate(all_means)
        train_targets = np.concatenate(all_targets)
    
    # 计算评估指标
    train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
    train_r2 = r2_score(train_targets, train_preds)
    
    metrics = {
        'train_rmse': float(train_rmse),
        'train_r2': float(train_r2),
        'num_iterations': num_iterations
    }
    
    return model, likelihood, metrics

def evaluate_stgpr_model(model, likelihood, X_test_tensor, y_test_tensor, batch_size=None):
    """
    评估STGPR模型在测试集上的性能
    
    参数:
    model: 训练好的STGPR模型
    likelihood: 训练好的似然函数
    X_test_tensor: 测试特征张量
    y_test_tensor: 测试目标张量
    batch_size: 批处理大小，如果为None则使用配置中的设置
    
    返回:
    metrics: 包含评估指标的字典
    predictions: 预测结果
    """
    # 获取批处理大小
    if batch_size is None:
        batch_size = CONFIG['model']['batch_size']
    
    # 设置为评估模式
    model.eval()
    likelihood.eval()
    
    # 准备测试数据加载器
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 分批预测，避免内存溢出
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        all_means = []
        all_variances = []
        all_targets = []
        
        for X_batch, y_batch in test_loader:
            pred_dist = likelihood(model(X_batch))
            means = pred_dist.mean.cpu().numpy()
            variances = pred_dist.variance.cpu().numpy()
            targets = y_batch.cpu().numpy()
            
            all_means.append(means)
            all_variances.append(variances)
            all_targets.append(targets)
        
        # 合并所有批次的预测
        test_preds = np.concatenate(all_means)
        test_variances = np.concatenate(all_variances)
        test_targets = np.concatenate(all_targets)
    
    # 计算评估指标
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    test_r2 = r2_score(test_targets, test_preds)
    test_mae = np.mean(np.abs(test_targets - test_preds))  # 添加MAE计算
    
    # 计算95%置信区间
    test_std = np.sqrt(test_variances)
    lower_bound = test_preds - 1.96 * test_std
    upper_bound = test_preds + 1.96 * test_std
    
    # 计算置信区间覆盖率
    coverage = np.mean((test_targets >= lower_bound) & (test_targets <= upper_bound))
    
    # 计算标准化的置信区间宽度
    nciw = np.mean(upper_bound - lower_bound) / np.std(test_targets)
    
    metrics = {
        'test_rmse': float(test_rmse),
        'test_r2': float(test_r2),
        'test_mae': float(test_mae),  # 添加MAE到返回的字典中
        'coverage_prob': float(coverage),
        'nciw': float(nciw)
    }
    
    predictions = {
        'mean': test_preds,
        'variance': test_variances,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'targets': test_targets
    }
    
    return metrics, predictions

def train_evaluate_stgpr_model(X_train, y_train, X_test=None, y_test=None, feature_names=None,
                              num_inducing_points=500, optimize_hyperparams=True, max_evals=10, 
                              model_path=None, device=None, X_train_full=None):
    """
    训练和评估STGPR模型的完整流程
    
    参数:
    X_train: 训练特征矩阵（可以是DataFrame或numpy数组）
    y_train: 训练目标变量
    X_test: 测试特征矩阵（可选）
    y_test: 测试目标变量（可选）
    feature_names: 特征名称列表
    num_inducing_points: 诱导点数量
    optimize_hyperparams: 是否进行超参数优化
    max_evals: hyperopt最大评估次数
    model_path: 模型保存路径（可选）
    device: 计算设备
    X_train_full: 包含完整信息的原始DataFrame（用于诱导点选择，包含h3_index和year）
    
    返回:
    dict: 包含训练结果、模型、评估指标等信息的字典
    """
    if feature_names is None and isinstance(X_train, pd.DataFrame):
        feature_names = list(X_train.columns)
    
    # 开始计时
    start_time = time.time()
    
    # 确定设备
    if device is None:
        device = torch.device('cpu')
    
    # 准备数据
    if isinstance(X_train, pd.DataFrame):
        X_train_np = X_train.values.astype(np.float32)
    else:
        X_train_np = np.asarray(X_train).astype(np.float32)
    
    if isinstance(y_train, pd.Series):
        y_train_np = y_train.values.astype(np.float32)
    else:
        y_train_np = np.asarray(y_train).astype(np.float32)
    
    # 记录训练信息
    print(f"📊 训练数据: {X_train_np.shape[0]} 样本, {X_train_np.shape[1]} 特征")
    
    # 执行超参数优化（如果启用）
    if optimize_hyperparams and HAS_HYPEROPT:
        print(f"\n🔍 开始超参数优化 (最大评估次数: {max_evals})")
        opt_result = optimize_stgpr_hyperparameters(
            X_train, y_train, feature_names,
            num_inducing_points=num_inducing_points,
            max_evals=max_evals,
            device=device,
            scaler=None,  # 不使用scaler
            X_train_full=X_train_full
        )
        
        if opt_result and opt_result['status'] == STATUS_OK:
            model = opt_result['model']
            likelihood = opt_result['likelihood']
            best_params = opt_result.get('best_params', {})
            iterations = opt_result.get('iterations', CONFIG['model']['num_iterations'])
            
            print(f"\n✅ 超参数优化完成")
            print(f"  最佳RMSE: {opt_result['rmse']:.6f}")
            print(f"  最佳R²: {opt_result['r2']:.4f}")
            
            # 打印最佳参数（简化版）
            print(f"  关键参数:")
            print(f"    空间长度尺度: {best_params.get('spatial_lengthscale', 'N/A'):.3f}")
            print(f"    时间长度尺度: {best_params.get('temporal_lengthscale', 'N/A'):.3f}")
            print(f"    特征长度尺度: {best_params.get('feature_lengthscale', 'N/A'):.3f}")
        else:
            print(f"\n⚠️ 超参数优化失败，使用默认参数")
            model, X_train_tensor, y_train_tensor, device = create_stgpr_model(
                X_train, y_train, 
                num_inducing_points=num_inducing_points,
                device=device,
                scaler=None,  # 不使用scaler
                X_train_full=X_train_full
            )
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            iterations = CONFIG['model']['num_iterations']
            best_params = {}
    else:
        # 不进行超参数优化，直接创建模型
        print(f"\n🏗️ 使用默认参数创建模型")
        model, X_train_tensor, y_train_tensor, device = create_stgpr_model(
            X_train, y_train, 
            num_inducing_points=num_inducing_points,
            device=device,
            scaler=None,  # 不使用scaler
            X_train_full=X_train_full
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        iterations = CONFIG['model']['num_iterations']
        best_params = {}
    
    # 训练模型
    print(f"\n🚀 开始训练模型 (迭代次数: {iterations})")
    
    # 定义进度回调函数
    last_update_time = [time.time()]  # 使用列表以便在闭包中修改
    
    def progress_callback(iteration, total_iterations, loss):
        """训练进度回调函数"""
        current_time = time.time()
        # 每0.5秒更新一次或在最后一次迭代时更新
        if current_time - last_update_time[0] > 0.5 or iteration == total_iterations - 1:
            progress = (iteration + 1) / total_iterations * 100
            elapsed = current_time - start_time
            eta = elapsed / (iteration + 1) * total_iterations - elapsed
            
            # 使用\r实现行内更新
            print(f"\r  进度: {progress:.1f}% | 迭代: {iteration+1}/{total_iterations} | "
                  f"损失: {loss:.6f} | 已用时: {elapsed:.1f}s | 预计剩余: {eta:.1f}s", 
                  end='', flush=True)
            
            last_update_time[0] = current_time
    
    # 创建训练数据张量（无论是否进行了超参数优化都需要）
    if isinstance(X_train, pd.DataFrame):
        X_train_np = X_train.values.astype(np.float32)
    else:
        X_train_np = np.asarray(X_train).astype(np.float32)
    
    if isinstance(y_train, pd.Series):
        y_train_np = y_train.values.astype(np.float32)
    else:
        y_train_np = np.asarray(y_train).astype(np.float32)
    
    # 创建张量并移动到设备
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).to(device)
    
    # 训练模型
    model, likelihood, train_metrics = train_stgpr_model(
        model, likelihood, X_train_tensor, y_train_tensor,
        num_iterations=iterations,
        callback=progress_callback,
        device=device
    )
    
    print()  # 换行以结束进度显示
    print(f"✅ 模型训练完成")
    print(f"  训练RMSE: {train_metrics['train_rmse']:.6f}")
    print(f"  训练R²: {train_metrics['train_r2']:.4f}")
    
    # 在测试集上评估（如果提供）
    test_metrics = {}
    predictions = None  # 初始化predictions变量
    if X_test is not None and y_test is not None:
        print(f"\n📊 在测试集上评估 ({len(X_test)} 样本)")
        
        if isinstance(X_test, pd.DataFrame):
            X_test_np = X_test.values.astype(np.float32)
        else:
            X_test_np = np.asarray(X_test).astype(np.float32)
        
        # 🔴 直接使用原始数据，不进行标准化
        X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test.values if hasattr(y_test, 'values') else y_test, 
                                   dtype=torch.float32).to(device)
        
        test_metrics, predictions = evaluate_stgpr_model(model, likelihood, X_test_tensor, y_test_tensor)
        
        print(f"  测试RMSE: {test_metrics['test_rmse']:.6f}")
        print(f"  测试R²: {test_metrics['test_r2']:.4f}")
        print(f"  测试MAE: {test_metrics['test_mae']:.6f}")
    
    # 总耗时
    total_time = time.time() - start_time
    print(f"\n⏱️ 总耗时: {total_time:.2f}秒")
    
    # 组装返回结果
    result = {
        'model': model,
        'likelihood': likelihood,
        'feature_names': feature_names,
        'scaler': None,  # 不再使用scaler
        'metrics': {
            'train_rmse': train_metrics['train_rmse'],
            'train_r2': train_metrics['train_r2'],
            **test_metrics,
            'training_time': total_time
        },
        'hyperparameters': best_params,
        'X': X_train,  # 保存原始训练数据用于SHAP分析
        'y': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_train': y_train,  # 添加这一行
        'device': device
    }
    
    # 🔴 移除基于模型参数的特征重要性计算 - 确保只有GeoShapley成功才有feature_importance
    # result['feature_importance'] = feature_importance  # 已删除，由GeoShapley计算
    
    # 添加predictions到result（如果有的话）
    if predictions is not None:
        result['predictions'] = predictions
        # 为了兼容性，也添加y_pred和y_test作为顶级字段
        result['y_pred'] = predictions['mean']
        result['y_test'] = predictions['targets']
    
    # 保存模型（如果指定路径）
    if model_path:
        save_stgpr_model(result, model_path)
        print(f"\n💾 模型已保存至: {model_path}")
    
    return result

def save_stgpr_model(model_dict, model_path):
    """
    保存STGPR模型及其相关数据
    
    参数:
    model_dict: 包含模型及相关数据的字典
    model_path: 保存路径
    
    返回:
    bool: 是否成功保存
    """
    try:
        directory = os.path.dirname(model_path)
        if directory:
            success, _ = ensure_dir_exists(directory)
            if not success:
                return False
        
        # 提取需要保存的数据
        model = model_dict.get('model')
        likelihood = model_dict.get('likelihood')
        feature_names = model_dict.get('feature_names')
        scaler = model_dict.get('scaler')
        metrics = model_dict.get('metrics', {})
        
        if model is None or likelihood is None:
            return False
        
        # 将模型移动到CPU
        model = model.cpu()
        likelihood = likelihood.cpu()
        
        # 保存状态字典
        state_dict = {
            'model_state_dict': model.state_dict(),
            'likelihood_state_dict': likelihood.state_dict(),
            'feature_names': feature_names,
            'scaler': scaler,
            'metrics': metrics
        }
        
        torch.save(state_dict, model_path)
        return True
    except Exception as e:
        print(f"保存模型时出错: {e}")
        print(traceback.format_exc())
        return False

# 从stgpr_utils导入explain_stgpr_predictions函数
from .stgpr_utils import explain_stgpr_predictions

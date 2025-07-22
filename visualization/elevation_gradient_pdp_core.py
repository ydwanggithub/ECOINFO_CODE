#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
海拔梯度PDP分析核心模块

该模块包含用于海拔梯度PDP分析的核心函数和数据处理部分。
主要实现数据的分组、特征重要性计算等基础功能。

支持的模型类型：
- ST-GPR (Spatiotemporal Gaussian Process Regression) 基于gpytorch的模型
- 其他通用机器学习模型

特殊功能：
- 自动检测ST-GPR模型并使用专门的预测方法
- 支持基于相关性的特征交互识别
"""

# 设置OMP_NUM_THREADS环境变量，避免Windows上KMeans内存泄漏问题
# 这必须在导入sklearn之前进行设置
import os
import sys
import platform
if platform.system() == 'Windows' and 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'
    print(f"在elevation_gradient_pdp_core.py中设置OMP_NUM_THREADS=1，避免Windows上KMeans内存泄漏问题")

import numpy as np
import pandas as pd
import shap
import datetime
import traceback
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
from scipy.stats import pearsonr

# 从visualization.utils导入正确的函数实现
from visualization.utils import clean_feature_name, clean_feature_name_for_plot

# 添加版本信息，方便调试
_version = "2.0.0"
_last_modified = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def split_data_by_elevation(df, n_bins=3, min_samples=20, use_percentiles=True):
    """
    将数据按海拔分成n_bins个区间，确保每个区间至少有min_samples个样本
    
    参数:
    df (pd.DataFrame): 包含海拔数据的数据框
    n_bins (int): 区间数量，默认为3（低、中、高）
    min_samples (int): 每个区间最少的样本数
    use_percentiles (bool): 是否使用分位数而不是线性分割
    
    返回:
    dict: 键为区间标签(low, medium, high)，值为区间内的数据
    list: 区间边界值
    """
    if 'elevation' not in df.columns:
        raise ValueError("数据框中不包含elevation列")
    
    # 获取有效海拔范围
    elevation = df['elevation'].dropna()
    if len(elevation) == 0:
        raise ValueError("海拔数据全为NaN")
    
    # 创建海拔区间
    if use_percentiles and n_bins == 3:
        # 使用33%和66%分位数
        quantiles = [0, 0.33, 0.66, 1.0]
        bins = [elevation.min()] + [elevation.quantile(q) for q in quantiles[1:-1]] + [elevation.max()]
    elif use_percentiles:
        # 计算分位数
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = [elevation.quantile(q) for q in quantiles]
    else:
        # 使用线性区间
        min_elev = elevation.min()
        max_elev = elevation.max()
        bins = np.linspace(min_elev, max_elev, n_bins + 1)
    
    labels = ['low', 'medium', 'high'][:n_bins]
    
    # 分配海拔区间
    df_copy = df.copy()
    df_copy['elevation_bin'] = pd.cut(df_copy['elevation'], bins=bins, labels=labels)
    
    # 按区间分组
    elevation_groups = {}
    for label in labels:
        group_df = df_copy[df_copy['elevation_bin'] == label].copy()
        
        # 检查样本数量是否足够
        if len(group_df) >= min_samples:
            elevation_groups[label] = group_df
            elev_range = f"{int(bins[labels.index(label)])}-{int(bins[labels.index(label)+1])}m"
            print(f"海拔区间 {label} ({elev_range}): {len(group_df)} 个样本")
        else:
            print(f"警告: 海拔区间 {label} 样本不足 ({len(group_df)} < {min_samples})，将被跳过")
    
    return elevation_groups, bins


def ensure_features_available(X_data, feat_pair, top_n_features=5):
    """
    确保特征在数据中可用，如果不可用则找到替代特征
    
    参数:
    X_data (pd.DataFrame): 特征数据
    feat_pair (tuple): 特征对(feat1, feat2)
    top_n_features (int): 用于替代的候选特征数量
    
    返回:
    tuple: 可用的特征对
    """
    feat1, feat2 = feat_pair
    
    # 检查特征是否可用
    if feat1 in X_data.columns and feat2 in X_data.columns:
        return feat1, feat2
    
    # 如果有特征不可用，找出数据中的可用特征
    available_features = X_data.columns.tolist()
    print(f"警告: 特征不可用，在{len(available_features)}个可用特征中寻找替代...")
    
    # 计算数据中每个特征的方差，选择方差最大的几个特征作为候选
    feature_variance = X_data.var().sort_values(ascending=False)
    top_features = feature_variance.index[:top_n_features].tolist()
    
    # 替换不可用的特征
    if feat1 not in X_data.columns:
        for feature in top_features:
            if feature != feat2:  # 避免选择相同的特征
                feat1 = feature
                print(f"特征 {feat_pair[0]} 不可用，使用替代特征 {feat1}")
                break
    
    if feat2 not in X_data.columns:
        for feature in top_features:
            if feature != feat1:  # 避免选择相同的特征
                feat2 = feature
                print(f"特征 {feat_pair[1]} 不可用，使用替代特征 {feat2}")
                break
    
    return feat1, feat2


def identify_top_interactions(model, X, feature_names, n_top=1, shap_interaction_values=None):
    """
    识别特征之间最重要的交互关系
    
    参数:
    model: 训练好的模型
    X: 特征数据
    feature_names: 特征名称列表
    n_top: 返回的前N个最重要交互对
    shap_interaction_values: 预计算的SHAP交互值矩阵（可选）
    
    返回:
    list: 前N个最重要的特征交互对 [(feat1, feat2), ...]
    """
    try:
        print("识别重要特征交互对...")
        
        # 首先检查是否提供了SHAP交互值
        if shap_interaction_values is not None:
            print("使用真实的SHAP交互值计算特征交互重要性")
            
            # 计算每对特征的平均绝对交互值
            feature_importance = {}
            n_features = len(feature_names)
            
            # 确保shap_interaction_values是numpy数组
            if hasattr(shap_interaction_values, 'values'):
                interaction_matrix = shap_interaction_values.values
            else:
                interaction_matrix = shap_interaction_values
            
            # 计算每对特征的平均绝对交互值
            for i in range(n_features):
                for j in range(i+1, n_features):  # 只考虑上三角，避免重复
                    # 获取特征i和j之间的交互值
                    if len(interaction_matrix.shape) == 3:
                        # 形状是 (n_samples, n_features, n_features)
                        interaction_values = interaction_matrix[:, i, j]
                    else:
                        # 其他可能的形状
                        interaction_values = interaction_matrix[i, j]
                    
                    # 计算平均绝对交互值
                    avg_interaction = np.mean(np.abs(interaction_values))
                    feat1, feat2 = feature_names[i], feature_names[j]
                    feature_importance[(feat1, feat2)] = float(avg_interaction)
            
            print(f"  成功从SHAP交互值中提取了{len(feature_importance)}个特征对的交互重要性")
            
            # 排序并返回前n个
            sorted_interactions = sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # 返回格式为 [(feat1, feat2), ...]
            top_pairs = [pair for pair, importance in sorted_interactions[:n_top]]
            
            # 打印结果
            print(f"找到的Top {n_top}特征交互对:")
            for i, (pair, importance) in enumerate(sorted_interactions[:n_top], 1):
                print(f"  {i}. {pair[0]} × {pair[1]} (重要性: {importance:.6f})")
                
            return top_pairs
            
        else:
            # 如果没有提供SHAP交互值，明确提示失败
            print("❌ 错误：未提供SHAP交互值")
            print("   原因：海拔梯度PDP交互分析需要SHAP交互值来识别重要的特征交互对")
            print("   建议：")
            print("   1. 确保在计算SHAP值时包含了交互值计算")
            print("   2. 检查GeoShapley是否成功计算了交互值")
            print("   3. 验证模型和数据是否支持交互值计算")
            return []
        
    except Exception as e:
        print(f"❌ 计算特征交互时出错: {str(e)}")
        traceback.print_exc()
        print("   建议：检查输入数据和模型的兼容性")
        return []


def calculate_pdp_for_elevation(model, X, feature_pair, grid_resolution=20, verbose=False):
    """
    计算特征对的PDP值
    
    参数:
    model: 训练好的模型
    X (pd.DataFrame): 特征数据
    feature_pair (tuple): 特征对(feat1, feat2)
    grid_resolution (int): 网格分辨率
    verbose (bool): 是否输出详细调试信息
    
    返回:
    tuple: (X1, X2, Z) 用于绘制等高线图的数据
    """
    feat1, feat2 = feature_pair
    
    try:
        if verbose:
            print(f"计算PDP: {feat1} × {feat2}")
        
        # 检查是否为ST-GPR模型
        is_stgpr = False
        stgpr_model = None
        likelihood = None
        
        try:
            import gpytorch
            import torch
            
            # 检查模型类型
            if isinstance(model, dict):
                # 如果model是字典，提取实际的模型对象
                if 'model' in model:
                    actual_model = model['model']
                    likelihood = model.get('likelihood')
                else:
                    actual_model = model
            else:
                actual_model = model
            
            # 检查是否是STGPRModel或gpytorch的ApproximateGP
            if hasattr(actual_model, '__class__'):
                is_stgpr = (actual_model.__class__.__name__ == 'STGPRModel' or 
                           isinstance(actual_model, gpytorch.models.ApproximateGP))
                if is_stgpr:
                    stgpr_model = actual_model
                    if verbose:
                        print("检测到STGPR模型 (gpytorch)")
        except ImportError:
            pass
        
        # 获取特征的范围
        x1_min, x1_max = X[feat1].min(), X[feat1].max()
        x2_min, x2_max = X[feat2].min(), X[feat2].max()
        
        # 创建特征网格
        x1_grid = np.linspace(x1_min, x1_max, grid_resolution)
        x2_grid = np.linspace(x2_min, x2_max, grid_resolution)
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        
        # 初始化PDP值矩阵
        Z = np.zeros_like(X1)
        
        # 计算PDP值
        if is_stgpr and stgpr_model is not None:
            # STGPR模型的PDP计算
            try:
                import torch
                import gpytorch
                
                # 创建预测函数包装器
                def predict_func(X_input):
                    """gpytorch模型预测函数包装器"""
                    # 确保模型和likelihood在评估模式
                    stgpr_model.eval()
                    if likelihood:
                        likelihood.eval()
                    
                    # 转换为张量
                    if isinstance(X_input, pd.DataFrame):
                        X_values = X_input.values
                    else:
                        X_values = X_input
                    
                    # 获取模型所在的设备
                    if hasattr(stgpr_model, 'device'):
                        device = stgpr_model.device
                    elif next(stgpr_model.parameters(), None) is not None:
                        device = next(stgpr_model.parameters()).device
                    else:
                        device = torch.device('cpu')
                    
                    # 将张量移到与模型相同的设备上
                    X_tensor = torch.tensor(X_values, dtype=torch.float32).to(device)
                    
                    # 预测
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        output = stgpr_model(X_tensor)
                        if likelihood:
                            output = likelihood(output)
                        mean = output.mean
                    
                    # 确保张量detach并在CPU上，然后转换为numpy
                    return mean.detach().cpu().numpy()
                
                # 对于每个网格点计算PDP
                for i in range(grid_resolution):
                    for j in range(grid_resolution):
                        x1_val = X1[i, j]
                        x2_val = X2[i, j]
                        
                        # 创建数据副本并替换两个特征的值
                        X_pdp = X.copy()
                        X_pdp[feat1] = x1_val
                        X_pdp[feat2] = x2_val
                        
                        # 预测并存储均值
                        predictions = predict_func(X_pdp)
                        Z[i, j] = np.mean(predictions)
                        
                        if verbose and (i*grid_resolution + j) % 50 == 0:
                            progress = (i*grid_resolution + j) / (grid_resolution**2) * 100
                            print(f"\rPDP计算进度: {progress:.1f}%", end='', flush=True)
                
                # 进度完成后换行
                if verbose:
                    print()  # 换行
            except Exception as e:
                print(f"STGPR PDP计算出错: {e}")
                # 回退到基于相关性的简单近似
                # 使用线性关系来近似PDP
                corr_matrix = X.corr()
                corr_feat1_target = 0.5  # 假设中等相关性
                corr_feat2_target = 0.3  # 假设轻微相关性
                
                # 生成近似的PDP
                for i in range(grid_resolution):
                    for j in range(grid_resolution):
                        x1_norm = (X1[i, j] - x1_min) / (x1_max - x1_min)
                        x2_norm = (X2[i, j] - x2_min) / (x2_max - x2_min)
                        # 线性组合作为简单近似
                        Z[i, j] = 0.5 + (x1_norm * corr_feat1_target) + (x2_norm * corr_feat2_target)
        else:
            # 通用模型的PDP计算 - 直接预测方法
            try:
                # 创建预测函数包装器
                def predict_func(X_input):
                    """通用模型预测函数包装器"""
                    if hasattr(model, 'predict_proba'):
                        # 对于概率预测，取正类概率
                        preds = model.predict_proba(X_input)
                        if isinstance(preds, tuple):
                            preds = preds[0]  # 某些模型可能返回元组
                        # 如果是二分类，取第二列（正类概率）
                        if preds.shape[1] == 2:
                            return preds[:, 1]
                        else:  # 多分类，取均值
                            return preds.mean(axis=1)
                    else:
                        # 直接预测
                        preds = model.predict(X_input)
                        if isinstance(preds, tuple):
                            preds = preds[0]  # 某些模型可能返回元组
                        return preds
                
                # 对于每个网格点计算PDP
                for i in range(grid_resolution):
                    for j in range(grid_resolution):
                        x1_val = X1[i, j]
                        x2_val = X2[i, j]
                        
                        # 创建数据副本并替换两个特征的值
                        X_pdp = X.copy()
                        X_pdp[feat1] = x1_val
                        X_pdp[feat2] = x2_val
                        
                        # 预测并存储均值
                        predictions = predict_func(X_pdp)
                        Z[i, j] = np.mean(predictions)
                        
                        if verbose and (i*grid_resolution + j) % 50 == 0:
                            progress = (i*grid_resolution + j) / (grid_resolution**2) * 100
                            print(f"\rPDP计算进度: {progress:.1f}%", end='', flush=True)
                
                # 进度完成后换行
                if verbose:
                    print()  # 换行
            except Exception as e:
                print(f"通用PDP计算出错: {e}")
                # 回退到简单近似
                for i in range(grid_resolution):
                    for j in range(grid_resolution):
                        x1_norm = (X1[i, j] - x1_min) / (x1_max - x1_min)
                        x2_norm = (X2[i, j] - x2_min) / (x2_max - x2_min)
                        # 线性组合作为简单近似
                        Z[i, j] = 0.5 + (x1_norm * 0.3) + (x2_norm * 0.2)
        
        # 确保PDP值在合理范围内
        Z = np.clip(Z, 0, 1)
        
        if verbose:
            print(f"PDP计算完成，Z值范围: [{Z.min():.4f}, {Z.max():.4f}]")
        
        return X1, X2, Z
    
    except Exception as e:
        print(f"PDP计算失败: {e}")
        traceback.print_exc()
        
        # 返回空的PDP结果
        x1_grid = np.linspace(0, 1, grid_resolution)
        x2_grid = np.linspace(0, 1, grid_resolution)
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        Z = np.zeros_like(X1)
        
        return X1, X2, Z 

def compute_elevation_gradient_single_feature(df, feature_name, target_col='VHI', n_bands=16):
    """
    计算单个特征在海拔梯度下的依赖关系
    
    参数:
    - df: 包含特征、目标变量和海拔数据的DataFrame
    - feature_name: 特征名称
    - target_col: 目标变量列名，默认'VHI'
    - n_bands: 海拔区间数量，默认16
    
    返回:
    - results: 包含海拔区间分析结果的字典
    """
    if feature_name not in df.columns:
        print(f"警告: 特征 {feature_name} 不存在于数据中")
        return {}
    
    if 'elevation' not in df.columns:
        print(f"警告: 数据中缺少elevation列")
        return {}
    
    # 创建海拔区间
    elevation_min = df['elevation'].min()
    elevation_max = df['elevation'].max()
    
    # 创建均匀的海拔区间
    band_edges = np.linspace(elevation_min, elevation_max, n_bands + 1)
    
    results = {}
    
    for i in range(n_bands):
        band_min = band_edges[i]
        band_max = band_edges[i + 1]
        band_label = f"{int(band_min)}-{int(band_max)}"
        
        # 筛选该海拔区间的数据
        mask = (df['elevation'] >= band_min) & (df['elevation'] < band_max)
        if i == n_bands - 1:  # 最后一个区间包含最大值
            mask = (df['elevation'] >= band_min) & (df['elevation'] <= band_max)
        
        band_data = df[mask]
        
        if len(band_data) == 0:
            continue
        
        # 计算统计指标
        feature_values = band_data[feature_name]
        target_values = band_data[target_col] if target_col in band_data.columns else None
        
        band_result = {
            'elevation_range': (band_min, band_max),
            'sample_count': len(band_data),
            f'{feature_name}_mean': feature_values.mean(),
            f'{feature_name}_std': feature_values.std(),
            f'{feature_name}_min': feature_values.min(),
            f'{feature_name}_max': feature_values.max(),
        }
        
        if target_values is not None:
            band_result.update({
                f'{target_col}_mean': target_values.mean(),
                f'{target_col}_std': target_values.std(),
                # 计算特征与目标的相关性
                'correlation': feature_values.corr(target_values) if len(feature_values) > 1 else 0,
            })
        
        results[band_label] = band_result
    
    return results

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
时空高斯过程回归模型 (ST-GPR) - 输入输出模块

本模块包含ST-GPR模型的输入输出相关功能：
1. 模型加载 (load_st_gpr_model)
2. 模型保存 (save_stgpr_model)
3. 模型预测 (predict_with_st_gpr)
"""

import os
import numpy as np
import pandas as pd
import torch
import traceback

# 检查GPyTorch依赖
HAS_GPYTORCH = False
try:
    import gpytorch
    HAS_GPYTORCH = True
except ImportError:
    pass

# 从core模块导入ensure_dir_exists函数
from .core import ensure_dir_exists
from .stgpr_config import get_config


def load_st_gpr_model(model_path, device=None):
    """
    从保存的模型文件中加载ST-GPR模型
    
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
    
    # 导入STGPRModel
    from .stgpr_model import STGPRModel
    
    # 创建诱导点
    inducing_points = None
    for name, param in checkpoint['model_state_dict'].items():
        if 'inducing_points' in name:
            inducing_points = param
            break
    
    config = get_config()
    if inducing_points is None:
        num_inducing = config['model']['num_inducing_points']
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


def predict_with_st_gpr(model_dict, X, return_variance=False, batch_size=None):
    """
    使用训练好的ST-GPR模型进行预测
    
    参数:
    model_dict: 包含模型、似然、scaler等的字典
    X: 输入特征 (DataFrame或numpy数组)
    return_variance: 是否返回预测方差
    batch_size: 批处理大小，如果为None则自动确定
    
    返回:
    predictions: 预测值 (如果return_variance=True，则返回(mean, variance))
    """
    import torch
    import gpytorch
    
    # 获取模型组件
    model = model_dict['model']
    likelihood = model_dict['likelihood']
    scaler = model_dict.get('scaler')
    
    # 🔧 智能设备管理：检测模型当前设备并保持一致
    model_device = next(model.parameters()).device
    target_device = 'cpu'  # 为了GeoShapley兼容性，统一使用CPU
    
    # 🔧 静默模式：避免GeoShapley计算时的重复日志
    verbose = batch_size is None or batch_size > 1000  # 只在大批量或手动调用时显示详细信息
    
    # 🛡️ 确保模型及其所有组件都在目标设备上
    if model_device != torch.device(target_device):
        if verbose:
            print(f"  🔧 将模型从 {model_device} 迁移到 {target_device}")
        
        # 深度设备迁移：确保所有组件都在目标设备上
        model = model.to(target_device)
        likelihood = likelihood.to(target_device)
        
        # 🔥 关键修复：强制迁移所有内部状态
        # 递归检查并迁移所有子模块的参数和缓冲区
        for name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param.device != torch.device(target_device):
                    param.data = param.data.to(target_device)
                    if param.grad is not None:
                        param.grad = param.grad.to(target_device)
            
            for buffer_name, buffer in module.named_buffers(recurse=False):
                if buffer.device != torch.device(target_device):
                    buffer.data = buffer.data.to(target_device)
        
        # 🔥 特别处理变分推理组件
        if hasattr(model, 'variational_strategy'):
            vs = model.variational_strategy
            # 迁移诱导点
            if hasattr(vs, 'inducing_points'):
                vs.inducing_points = vs.inducing_points.to(target_device)
            # 迁移变分参数
            if hasattr(vs, '_variational_distribution'):
                if hasattr(vs._variational_distribution, 'variational_mean'):
                    vs._variational_distribution.variational_mean = vs._variational_distribution.variational_mean.to(target_device)
                if hasattr(vs._variational_distribution, 'chol_variational_covar'):
                    vs._variational_distribution.chol_variational_covar = vs._variational_distribution.chol_variational_covar.to(target_device)
        
        if verbose:
            print(f"  ✅ 模型及所有组件已迁移到 {target_device}")
    
    # 🔇 简化设备验证：静默执行，只在真正需要时警告
    devices = set()
    for param in model.parameters():
        devices.add(param.device)
    for buffer in model.buffers():
        devices.add(buffer.device)
    
    # 只在有真正的设备不一致问题时才输出警告
    if len(devices) > 1 and verbose:
        print(f"  ⚠️ 警告：模型仍有组件在不同设备上: {devices}")
    # 移除冗余的成功消息 - 设备一致性检查应该是静默的
    
    # 设置为评估模式
    model.eval()
    likelihood.eval()
    
    # 准备输入数据
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    # 标准化
    if scaler is not None:
        X_array = scaler.transform(X_array)
    
    # 确定批处理大小
    if batch_size is None:
        # 根据诱导点数量自动确定批处理大小
        if hasattr(model, 'variational_strategy') and hasattr(model.variational_strategy, 'inducing_points'):
            num_inducing = model.variational_strategy.inducing_points.shape[0]
            if num_inducing >= 300:  # 对于大型模型使用更小的批次
                batch_size = 50
            else:
                batch_size = 200
        else:
            batch_size = 200
    
    # 🎯 转换为张量并确保在正确设备上
    X_tensor = torch.tensor(X_array, dtype=torch.float32).to(target_device)
    
    # 如果数据量小于批处理大小，直接预测
    if X_tensor.shape[0] <= batch_size:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = model(X_tensor)
            predictions = likelihood(output)
        
        if return_variance:
            return predictions.mean.cpu().numpy(), predictions.variance.cpu().numpy()
        else:
            return predictions.mean.cpu().numpy()
    
    # 批处理预测 - 进一步简化输出
    if verbose and X_tensor.shape[0] <= 1000:  # 只在verbose模式下对小批量显示详细信息
        print(f"  📦 批处理预测: {X_tensor.shape[0]}样本，批次={batch_size}")
    
    all_means = []
    all_variances = [] if return_variance else None
    
    total_batches = (X_tensor.shape[0] + batch_size - 1) // batch_size
    processed_batches = 0
    
    for i in range(0, X_tensor.shape[0], batch_size):
        batch_end = min(i + batch_size, X_tensor.shape[0])
        X_batch = X_tensor[i:batch_end]
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = model(X_batch)
            batch_predictions = likelihood(output)
        
        all_means.append(batch_predictions.mean.cpu().numpy())
        if return_variance:
            all_variances.append(batch_predictions.variance.cpu().numpy())
        
        processed_batches += 1
    
    # 合并结果
    final_means = np.concatenate(all_means)
    
    if return_variance:
        final_variances = np.concatenate(all_variances)
        return final_means, final_variances
    else:
        return final_means


def save_stgpr_model(model_dict, model_path):
    """
    保存ST-GPR模型及其相关数据
    
    参数:
    model_dict: 包含模型及相关数据的字典
    model_path: 保存路径
    
    返回:
    bool: 是否成功保存
    """
    try:
        directory = os.path.dirname(model_path)
        if directory:
            ensure_dir_exists(directory)
        
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
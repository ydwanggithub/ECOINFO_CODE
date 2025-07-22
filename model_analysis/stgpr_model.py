#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时空高斯过程回归模型 (ST-GPR) - 模型定义模块

本模块包含ST-GPR模型的核心定义，包括：
1. 自定义的SpatialSimilarityKernel
2. STGPRModel类定义
"""

import os
import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
import pandas as pd

# 自定义空间相似性核函数
class SpatialSimilarityKernel(gpytorch.kernels.Kernel):
    """
    自定义空间相似性核函数
    
    实现了一个融合特征相似性的核函数，给不同特征赋予不同的权重。
    该核函数计算两个点之间的相似性，基于其特征值的差异和自定义权重函数。
    """
    def __init__(self, active_dims=None, feature_weights=None, sigma_values=None, epsilon=1e-5):
        """
        初始化空间相似性核函数
        
        参数:
        active_dims: 核函数作用的特征维度
        feature_weights: 特征权重向量，用于给不同特征赋予不同的重要性
        sigma_values: 每个特征的标准差，用于标准化特征差异
        epsilon: 数值稳定性增强参数
        """
        super(SpatialSimilarityKernel, self).__init__(active_dims=active_dims)
        self.epsilon = epsilon
        
        # 注册特征权重参数
        if feature_weights is None:
            if active_dims is not None:
                n_dims = len(active_dims) if isinstance(active_dims, (list, tuple)) else active_dims.stop - active_dims.start
            else:
                n_dims = 1
            feature_weights = torch.ones(n_dims)
            
        self.register_parameter(
            "raw_feature_weights",
            torch.nn.Parameter(torch.log(torch.exp(feature_weights) - 1))
        )
        
        # 注册特征方差参数
        if sigma_values is None:
            if active_dims is not None:
                n_dims = len(active_dims) if isinstance(active_dims, (list, tuple)) else active_dims.stop - active_dims.start
            else:
                n_dims = 1
            sigma_values = torch.ones(n_dims)
            
        self.register_buffer("sigma_values", sigma_values)
        self.register_parameter("raw_outputscale", torch.nn.Parameter(torch.zeros(1)))
    
    @property
    def feature_weights(self):
        """获取特征权重，确保非负"""
        return torch.nn.functional.softplus(self.raw_feature_weights)
    
    @property
    def outputscale(self):
        """获取输出尺度，确保非负"""
        return torch.nn.functional.softplus(self.raw_outputscale)
    
    def forward(self, x1, x2, diag=False, **params):
        """
        计算核函数值
        
        参数:
        x1: 第一个输入点集
        x2: 第二个输入点集
        diag: 是否只返回对角线元素
        
        返回:
        torch.Tensor: 相似性矩阵
        """
        if diag:
            return torch.ones(x1.shape[0], device=x1.device) * (self.outputscale + self.epsilon)
        
        # 获取特征维度
        feature_dim = self.feature_weights.size(0)
        
        # 确保只使用特征维度范围内的数据
        x1_features = x1[:, :feature_dim] if x1.size(1) >= feature_dim else x1
        x2_features = x2[:, :feature_dim] if x2.size(1) >= feature_dim else x2
        
        # 计算特征差异
        diff = x1_features.unsqueeze(1) - x2_features.unsqueeze(0)  # shape: [n1, n2, d]
        
        # 使用有效的sigma_values范围
        valid_sigma = self.sigma_values[:diff.size(-1)]
        
        # 标准化差异
        diff = diff / (valid_sigma.unsqueeze(0).unsqueeze(0) + self.epsilon)
        
        # 计算加权高斯相似度
        squared_diff = diff.pow(2)  # shape: [n1, n2, d]
        
        # 计算每个特征的指数衰减相似度
        sim_per_feature = torch.exp(-0.5 * squared_diff)  # shape: [n1, n2, d]
        
        # 使用有效的特征权重
        valid_weights = self.feature_weights[:diff.size(-1)]
        
        # 使用特征权重计算加权平均相似度
        weights = valid_weights / (valid_weights.sum() + self.epsilon)
        weighted_sim = (sim_per_feature * weights.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # shape: [n1, n2]
        
        return self.outputscale * weighted_sim

# 定义PyTorch/GPyTorch版本的时空高斯过程回归模型
class STGPRModel(ApproximateGP):
    """
    时空高斯过程回归模型 (基于PyTorch/GPyTorch实现)
    
    使用稀疏变分高斯过程来处理大型时空数据集。
    模型结合了空间、时间和特征核函数，以捕获数据中的复杂时空相关性。
    核函数结构:
    (SpatialSimilarityKernel + MaternKernel) * RBFKernel(时间)
    """
    def __init__(self, inducing_points, input_dim, 
                spatial_dims=[0, 1], temporal_dims=[-1], feature_dims=None, 
                spatial_variance=1.0, temporal_variance=1.0, feature_variance=1.0,
                spatial_lengthscale=1.0, temporal_lengthscale=1.0, feature_lengthscale=1.0,
                feature_weights=None, sigma_values=None, nu=2.5, w=0.1):
        """
        初始化ST-GPR模型
        
        参数:
        inducing_points: 诱导点矩阵 (M x D)，其中M是诱导点数量，D是特征维度
        input_dim: 输入特征的总维度
        spatial_dims: 表示空间维度的索引列表 (例如 [0, 1]表示经纬度)
        temporal_dims: 表示时间维度的索引列表 (例如 [-1]表示年份)
        feature_dims: 表示环境特征的索引列表，如果为None则自动计算
        spatial_variance: 空间核初始方差
        temporal_variance: 时间核初始方差
        feature_variance: 特征核初始方差
        spatial_lengthscale: 空间核初始长度尺度
        temporal_lengthscale: 时间核初始长度尺度
        feature_lengthscale: 特征核初始长度尺度
        feature_weights: 特征权重向量，用于SpatialSimilarityKernel
        sigma_values: 特征标准差向量，用于SpatialSimilarityKernel
        nu: Matérn核的平滑度参数
        w: 复合核权重，用于组合空间核和时间核
        """
        # 设置变分分布和变分策略
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(STGPRModel, self).__init__(variational_strategy)
        
        # 确保所有维度索引都在有效范围内
        max_dim = min(input_dim, inducing_points.size(1)) - 1
        
        # 安全处理空间维度
        spatial_dims = [dim for dim in spatial_dims if dim <= max_dim]
        if not spatial_dims:  # 如果为空，使用默认值
            spatial_dims = [0, 1] if max_dim >= 1 else [0]
            
        # 安全处理时间维度
        temporal_dims = [dim if dim >= 0 else input_dim + dim for dim in temporal_dims]  # 处理负索引
        temporal_dims = [dim for dim in temporal_dims if dim <= max_dim]
        if not temporal_dims:  # 如果为空，使用最后一列
            temporal_dims = [max_dim]
        
        # 如果feature_dims为None，自动计算（除了空间和时间维度外的所有维度）
        if feature_dims is None:
            all_dims = set(range(min(input_dim, inducing_points.size(1))))
            spatial_temporal_dims = set(spatial_dims + temporal_dims)
            feature_dims = list(all_dims - spatial_temporal_dims)
        else:
            # 确保特征维度在有效范围内
            feature_dims = [dim for dim in feature_dims if dim <= max_dim]
        
        # 均值函数 - 使用常数均值
        self.mean_module = ConstantMean()
        
        # 1. 空间相似性核 - 自定义核函数，融合特征相似性
        if feature_weights is None:
            feature_weights = torch.ones(len(feature_dims))
        elif len(feature_weights) > len(feature_dims):
            feature_weights = feature_weights[:len(feature_dims)]
        
        if sigma_values is None:
            sigma_values = torch.ones(len(feature_dims))
        elif len(sigma_values) > len(feature_dims):
            sigma_values = sigma_values[:len(feature_dims)]
            
        self.feature_similarity_kernel = SpatialSimilarityKernel(
            active_dims=feature_dims,
            feature_weights=feature_weights,
            sigma_values=sigma_values
        )
        
        # 2. 空间距离核 - 使用Matern核（适合空间数据）
        self.spatial_kernel = ScaleKernel(
            MaternKernel(
                nu=nu,  # 使用传入的nu参数
                ard_num_dims=len(spatial_dims),
                active_dims=spatial_dims,
                lengthscale=torch.ones(len(spatial_dims)) * spatial_lengthscale
            ),
            outputscale=spatial_variance
        )
        
        # 3. 时间核 - 使用RBF核
        self.temporal_kernel = ScaleKernel(
            RBFKernel(
                ard_num_dims=len(temporal_dims),
                active_dims=temporal_dims,
                lengthscale=torch.ones(len(temporal_dims)) * temporal_lengthscale
            ),
            outputscale=temporal_variance
        )
        
        # 保存复合核权重w
        self.kernel_weight_w = w
        
        # 4. 组合核：(空间相似性核 + 空间距离核) * 时间核
        # 创建核函数的加权组合
        combined_spatial_kernel = gpytorch.kernels.AdditiveKernel(
            self.feature_similarity_kernel, 
            gpytorch.kernels.ScaleKernel(self.spatial_kernel.base_kernel, outputscale=w * self.spatial_kernel.outputscale)
        )
        self.covar_module = gpytorch.kernels.ProductKernel(combined_spatial_kernel, self.temporal_kernel)
        
        # 存储维度信息，用于后续特征重要性分析
        self.spatial_dims = spatial_dims
        self.temporal_dims = temporal_dims
        self.feature_dims = feature_dims
    
    def forward(self, x):
        """
        前向传播，计算GP预测
        
        参数:
        x: 输入特征张量
        
        返回:
        MultivariateNormal: 预测分布
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
    def predict(self, x):
        """
        预测方法，兼容GeoShapley接口
        
        参数:
        x: 输入特征，可以是NumPy数组、Pandas DataFrame或PyTorch张量
        
        返回:
        numpy.ndarray: 预测的均值
        """
        # 确保模型处于评估模式
        self.eval()
        
        # 将输入转换为PyTorch张量
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float()
        elif isinstance(x, pd.DataFrame):
            x_tensor = torch.from_numpy(x.values).float()
        elif isinstance(x, torch.Tensor):
            x_tensor = x.float()
        else:
            raise TypeError(f"不支持的输入类型: {type(x)}")
        
        # 使用无梯度上下文计算预测
        with torch.no_grad():
            # 获取预测分布
            distribution = self.forward(x_tensor)
            # 从分布中提取均值
            mean = distribution.mean
            
        # 转换为NumPy数组返回
        return mean.numpy()
        
    def get_inducing_points(self):
        """获取当前诱导点"""
        return self.variational_strategy.inducing_points.detach()
        
    def set_inducing_points(self, inducing_points):
        """设置新的诱导点"""
        if not torch.is_tensor(inducing_points):
            inducing_points = torch.tensor(inducing_points, dtype=torch.float32)
            
        self.variational_strategy._variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        self.variational_strategy._inducing_points = torch.nn.Parameter(inducing_points)
        
    def get_feature_importance(self, feature_names=None):
        """
        计算基于核函数参数的特征重要性
        
        参数:
        feature_names: 特征名称列表，用于在返回结果中标记特征
        
        返回:
        list: (特征, 重要性得分)对的列表，按重要性降序排列
        """
        importances = []
        
        # 1. 从空间相似性核获取特征重要性
        if hasattr(self, 'feature_similarity_kernel'):
            feature_weights = self.feature_similarity_kernel.feature_weights.detach().cpu().numpy().flatten()
            
            # 归一化特征权重
            feature_weights_sum = np.sum(feature_weights)
            if feature_weights_sum > 0:
                feature_weights = feature_weights / feature_weights_sum
                
            for i, dim in enumerate(self.feature_dims):
                importance = float(feature_weights[i])
                if feature_names is not None and dim < len(feature_names):
                    importances.append((feature_names[dim], importance))
                else:
                    importances.append((f"feature_{dim}", importance))
        
        # 2. 从空间距离核获取特征重要性
        if hasattr(self, 'spatial_kernel') and hasattr(self.spatial_kernel.base_kernel, 'lengthscale'):
            spatial_lengthscales = self.spatial_kernel.base_kernel.lengthscale.detach().cpu().numpy().flatten()
            
            # 重要性与长度尺度成反比
            for i, dim in enumerate(self.spatial_dims):
                # 归一化重要性得分
                importance = 1.0 / float(spatial_lengthscales[i])
                if feature_names is not None and dim < len(feature_names):
                    importances.append((feature_names[dim], importance))
                else:
                    importances.append((f"spatial_{dim}", importance))
        
        # 3. 从时间核获取特征重要性
        if hasattr(self, 'temporal_kernel') and hasattr(self.temporal_kernel.base_kernel, 'lengthscale'):
            temporal_lengthscales = self.temporal_kernel.base_kernel.lengthscale.detach().cpu().numpy().flatten()
            
            for i, dim in enumerate(self.temporal_dims):
                importance = 1.0 / float(temporal_lengthscales[i])
                if feature_names is not None and dim < len(feature_names):
                    importances.append((feature_names[dim], importance))
                else:
                    importances.append((f"temporal_{dim}", importance))
        
        # 4. 整合和归一化所有重要性得分
        if importances:
            max_importance = max(imp[1] for imp in importances)
            if max_importance > 0:
                importances = [(name, score/max_importance) for name, score in importances]
        
        # 按重要性降序排序
        importances.sort(key=lambda x: x[1], reverse=True)
        return importances 
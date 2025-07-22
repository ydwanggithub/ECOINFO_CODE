#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ST-GPR模型配置文件

本模块定义了时空高斯过程回归模型的默认参数设置，以确保数值稳定性和训练效率。
用户可以通过修改此文件中的参数来调整模型行为，而无需更改主要代码。
"""

import os

# 随机种子，确保结果可重现
RANDOM_SEED = 42

# 模型架构参数
MODEL_PARAMS = {
    # 稀疏GP的诱导点数量，增加此值可提高精度但会增加计算成本
    # 对于大数据集，建议使用500-1000；对于小数据集，可以使用100-200
    'num_inducing_points': 500,
    
    # 批处理大小，用于随机变分推断
    # 较小的批量可能更稳定但训练较慢，建议200-500
    'batch_size': 200,
    
    # 训练迭代次数
    # 增加此值可能提高模型性能，但会增加训练时间
    'num_iterations': 50,  # 减少迭代次数以加快测试速度
    
    # 是否使用L-BFGS优化器进行最终调优
    # 对于复杂问题，建议设为True
    'use_lbfgs': False,
    
    # L-BFGS最大迭代次数
    'lbfgs_max_iter': 100
}

# 优化器参数
OPTIMIZER_PARAMS = {
    # Adam优化器学习率
    # 较小的学习率通常更稳定但收敛较慢
    'adam_learning_rate': 0.01,
    
    # 梯度裁剪阈值
    # 防止梯度爆炸，提高训练稳定性
    'gradient_clip_norm': 10.0,
    
    # 是否使用梯度裁剪
    'use_gradient_clip': True,
    
    # Adam优化器参数
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_epsilon': 1e-7
}

# 核函数参数
KERNEL_PARAMS = {
    # 空间核初始方差
    'spatial_variance_init': 1.0,
    
    # 空间核长度尺度初始值
    'spatial_lengthscale_init': 1.0,
    
    # 时间核初始方差
    'temporal_variance_init': 1.0,
    
    # 时间核长度尺度初始值
    'temporal_lengthscale_init': 1.0,
    
    # 特征核初始方差
    'feature_variance_init': 1.0,
    
    # 特征核长度尺度初始值
    'feature_lengthscale_init': 1.0,
    
    # 高斯似然初始噪声方差
    'likelihood_variance_init': 0.1,
    
    # 长度尺度参数下界 (防止数值不稳定)
    'lengthscale_lower_bound': 1e-3,
    
    # 方差参数下界 (防止数值不稳定)
    'variance_lower_bound': 1e-5
}

# GeoShapley配置 - 优化以支持更大样本量
GEOSHAPLEY_CONFIG = {
    'n_jobs': -1,  # 使用所有可用CPU核心（与GGPR一致）
    'batch_size': 100,  # 从50增加到100，适应更大的样本量
    'n_background': 6,  # 从3增加到6背景数据点数量
    'enable_shap_interactions': True,  # 启用交互值计算以生成PDP交互图
    'enable_memory_cleanup': True,  # 启用内存清理
    'memory_limit_mb': 4096,  # 从2048增加到4096内存限制，支持更大样本量
    'use_shap_kmeans': True,  # 使用SHAP的K-means
    'timeout_per_sample': 60,  # 从30增加到60秒每个样本超时时间，适应更复杂计算
    'progress_interval': 50,  # 从10增加到50，适应更大样本量的进度显示
    'verbose': True,  # 显示详细进度信息
    # 🚀 针对RTX 4070 SUPER + 28线程CPU的智能并行策略
    'resolution_n_jobs': {
        'res5': 8,     # 小数据集：8核心，快速完成
        'res6': 16,    # 中等数据集：16核心，平衡效率与资源
        'res7': 20     # 大数据集：20核心，充分利用CPU性能（留8核心给系统和GPU）
    },
    # 新增：采样策略优化
    'sampling_strategy': {
        'method': 'stratified_spatial',  # 使用分层空间采样
        'ensure_spatial_coverage': True,  # 确保空间覆盖
        'min_distance_ratio': 0.1,  # 最小距离比例，避免采样点过于集中
        'max_cluster_size': 50  # 最大聚类大小，确保空间分散性
    }
}

# 分辨率特定的配置 - 统一使用CPU确保GeoShapley兼容性
# 这些配置会覆盖上面的默认值
RESOLUTION_SPECIFIC_CONFIG = {
    'res5': {
        'num_inducing_points_factor': 0.2,  # 基础值的20% = 100
        'max_hyperopt_evals': 10,            # 小数据集可以多评估
        'use_lbfgs': True,                   # 小数据集使用L-BFGS
        'prefer_gpu': False,                 # 统一使用CPU，确保GeoShapley兼容性
        'geoshapley_device': 'cpu',          # GeoShapley使用CPU
    },
    'res6': {
        'num_inducing_points_factor': 0.6,  # 基础值的60% = 300
        'max_hyperopt_evals': 8,             # 适度评估次数
        'use_lbfgs': False,                  # 使用Adam
        'prefer_gpu': False,                 # 🔧 改为CPU，与res5保持一致
        'gpu_batch_size_factor': 1.0,       # CPU批次大小
        'geoshapley_device': 'cpu',          # GeoShapley使用CPU
    },
    'res7': {
        'num_inducing_points_factor': 0.5,  # 基础值的50% = 250
        'max_hyperopt_evals': 5,             # 减少评估次数以补偿CPU速度
        'use_lbfgs': False,                  # 使用Adam
        'prefer_gpu': False,                 # 🔧 改为CPU，与res5保持一致
        'gpu_batch_size_factor': 1.0,       # CPU批次大小
        'geoshapley_device': 'cpu',          # GeoShapley使用CPU
    }
}

def get_config():
    """
    返回完整的配置字典
    
    返回:
    dict: 包含所有配置参数的字典
    """
    return {
        'random_seed': RANDOM_SEED,
        'model': MODEL_PARAMS,
        'optimizer': OPTIMIZER_PARAMS,
        'kernel': KERNEL_PARAMS,
        'geoshapley': GEOSHAPLEY_CONFIG,
        'resolution_specific': RESOLUTION_SPECIFIC_CONFIG
    }

# ============================================================================
# 环境配置功能
# ============================================================================

def setup_environment():
    """
    设置环境变量，优化CUDA和内存使用
    必须在导入其他模块之前调用
    """
    import os
    import sys
    
    # 🚀 启用GPU支持 - RTX 4070 SUPER加速
    # 注释掉原来禁用GPU的代码，现在启用GPU加速
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 已注释，启用GPU
    print("🎮 GPU支持已启用，将根据数据规模智能选择设备")
    
    # 设置OMP_NUM_THREADS环境变量，避免Windows上KMeans内存泄漏问题
    if sys.platform.startswith('win'):
        # 智能线程管理策略：
        # 1. 默认设置为1以避免KMeans内存泄漏
        # 2. 在GeoShapley计算期间临时调整为2以提高效率
        old_value = os.environ.get('OMP_NUM_THREADS', None)
        os.environ['OMP_NUM_THREADS'] = '1'
        if old_value != '1':
            print(f"强制设置OMP_NUM_THREADS=1，避免Windows上KMeans内存泄漏问题(原值: {old_value})")
            print("注意：在GeoShapley并行计算期间将临时调整为2线程以提高效率")
        else:
            print(f"OMP_NUM_THREADS已为1，但仍强制重置以确保生效")
            print("注意：在GeoShapley并行计算期间将临时调整为2线程以提高效率")

def configure_python_path():
    """
    配置Python搜索路径以导入项目模块
    
    返回:
    dict: 配置的目录信息
    """
    import os
    import sys
    
    # 当前目录（model_analysis）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 项目根目录
    project_root = os.path.dirname(current_dir)
    
    # 添加项目根目录到Python路径
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"已添加项目根目录到Python路径: {project_root}")
    
    # 添加可视化模块目录
    visualization_dir = os.path.join(project_root, 'visualization')
    if os.path.exists(visualization_dir) and visualization_dir not in sys.path:
        sys.path.insert(0, visualization_dir)
        print(f"已添加visualization目录到Python路径")
    
    # 返回配置的目录信息
    return {
        'current_dir': current_dir,
        'project_root': project_root,
        'visualization_dir': visualization_dir
    }

# ============================================================================
# 项目信息
# ============================================================================

PROJECT_INFO = {
    'name': '时空高斯过程回归分析框架',
    'description': '基于MSTHEA框架的时空高斯过程回归建模和分析功能',
    'version': '1.0.0',
    'author': '[作者名]',
    'date': '[日期]'
}

# ============================================================================
# 数据配置
# ============================================================================

DATA_CONFIG = {
    'target_column': 'VHI',
    'default_data_dir': 'data',
    'default_output_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output'),
    'file_patterns': {
        'res5': 'ALL_DATA_with_VHI_PCA_res5.csv',
        'res6': 'ALL_DATA_with_VHI_PCA_res6.csv',
        'res7': 'ALL_DATA_with_VHI_PCA_res7.csv'
    },
    'default_resolutions': ['res5', 'res6', 'res7']
} 
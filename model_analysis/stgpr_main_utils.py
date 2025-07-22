#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
时空高斯过程回归模型 (ST-GPR) - 主程序工具函数

本模块包含main.py中使用的工具函数，用于保持主程序的简洁性：
1. 清理缓存功能
2. 模块检查功能  
3. 数据加载和采样策略
4. 模型训练包装函数
"""

import os
import sys
import shutil
import warnings
import numpy as np
import pandas as pd
import time

# 抑制tqdm的Jupyter环境检测警告
warnings.filterwarnings("ignore", message="IProgress not found. Please update jupyter and ipywidgets.*")

# 禁止生成__pycache__目录和.pyc文件
sys.dont_write_bytecode = True

def clean_pycache():
    """
    清理项目中所有的__pycache__目录和.pyc文件
    """
    print("🧹 清理__pycache__目录...")
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
    cleaned_count = 0
    
    # 遍历所有子目录
    for root, dirs, files in os.walk(current_dir):
        # 跳过.git目录和其他隐藏目录
        if '/.git' in root or '\\.git' in root:
            continue
            
        # 删除__pycache__目录
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                cleaned_count += 1
                print(f"  ✓ 已删除: {os.path.relpath(pycache_path, current_dir)}")
            except Exception as e:
                print(f"  ⚠️ 删除失败: {os.path.relpath(pycache_path, current_dir)} - {e}")
        
        # 删除.pyc文件
        for file in files:
            if file.endswith('.pyc'):
                pyc_path = os.path.join(root, file)
                try:
                    os.remove(pyc_path)
                    cleaned_count += 1
                    print(f"  ✓ 已删除: {os.path.relpath(pyc_path, current_dir)}")
                except Exception as e:
                    print(f"  ⚠️ 删除失败: {os.path.relpath(pyc_path, current_dir)} - {e}")
    
    if cleaned_count > 0:
        print(f"  🎉 清理完成，共删除 {cleaned_count} 个__pycache__目录或.pyc文件\n")
    else:
        print(f"  ✨ 项目已清洁，没有找到__pycache__目录或.pyc文件\n")

def check_module_availability():
    """
    检查必要模块的可用性
    
    返回:
    dict: 包含各模块可用性状态的字典
    """
    modules_status = {
        'HAS_STGPR': False,
        'HAS_GPYTORCH': False,
        'HAS_GEOSHAPLEY': False,
        'HAS_HYPEROPT': False,
        'MODELS_AVAILABLE': []
    }
    
    # 检查PyTorch和GPyTorch可用性（用于STGPR）
    try:
        import torch
        import gpytorch
        modules_status['HAS_GPYTORCH'] = True
        print(f"✓ PyTorch {torch.__version__} 和 GPyTorch 可用")
        print(f"✓ 使用CPU计算")
    except ImportError:
        print("× PyTorch或GPyTorch不可用，无法训练ST-GPR模型")
    
    # 检查STGPR模型模块
    try:
        from model_analysis import stgpr
        modules_status['HAS_STGPR'] = True
        modules_status['MODELS_AVAILABLE'].append('ST-GPR')
        print("✓ STGPR模型模块可用")
    except ImportError:
        print("× STGPR模型模块不可用")
    
    # 检查GeoShapley（用于可解释性分析）
    try:
        from geoshapley import GeoShapleyExplainer
        modules_status['HAS_GEOSHAPLEY'] = True
        print("✓ GeoShapley可用，可以计算SHAP值")
    except ImportError:
        print("× GeoShapley不可用，无法计算SHAP值")
    
    # 检查hyperopt（用于超参数优化）
    try:
        from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
        modules_status['HAS_HYPEROPT'] = True
        print("✓ hyperopt可用，可以进行超参数优化")
    except ImportError:
        print("× hyperopt不可用，无法进行超参数优化")
    
    # 检查scikit-learn（基础机器学习库）
    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}可用")
    except (ImportError, AttributeError):
        print("× scikit-learn不可用或版本不正确")
    
    # 打印可用模型摘要
    models = modules_status['MODELS_AVAILABLE']
    print(f"可用的模型: {', '.join(models) if models else '无'}")
    
    return modules_status

def create_train_evaluate_wrapper():
    """
    创建train_evaluate_stgpr_model的包装函数
    
    返回:
    function: 包装后的训练函数
    """
    # 尝试导入原始训练函数
    try:
        from model_analysis.stgpr import train_evaluate_stgpr_model as stgpr_original_train_fn
        has_stgpr = True
        print("✓ 成功导入STGPR模型模块")
    except ImportError:
        has_stgpr = False
        stgpr_original_train_fn = None
        print("⚠ 警告: 无法导入STGPR模型模块")
    
    # 导入相关模块
    try:
        from data_processing.preprocessing import prepare_features_for_stgpr
        use_new_preprocessing = True
    except ImportError:
        # 回退到原有的特征准备方法
        print("⚠️ 无法导入新的预处理模块，使用原有方法")
        from model_analysis.stgpr_utils import prepare_features_for_stgpr
        use_new_preprocessing = False
    
    # 定义包装函数
    def train_evaluate_stgpr_model(df, resolution=None, output_dir=None, use_gpu=False, 
                                  target='VHI', use_hyperopt=True, max_hyperopt_evals=10, 
                                  num_inducing_points=None, **kwargs):
        """
        STGPR模型训练与评估的包装函数，接收DataFrame并处理参数
        
        参数:
        df: 包含特征和目标的DataFrame
        resolution: 分辨率级别 (res5, res6, res7)
        output_dir: 输出目录
        use_gpu: 是否使用GPU（注意：现在强制使用CPU）
        target: 目标变量名
        use_hyperopt: 是否使用超参数优化
        max_hyperopt_evals: 超参数优化的最大评估次数
        num_inducing_points: 稀疏变分GP的诱导点数量，如果为None则自动选择
        
        返回:
        dict: 包含模型和性能指标的字典
        """
        if not has_stgpr:
            print("❌ 错误: 无法运行模型训练 - STGPR模型不可用")
            print("请安装必要的依赖库: pytorch, gpytorch (用于STGPR)")
            return None
        
        from model_analysis.core import ensure_dir_exists  # 直接从core导入，避免循环依赖
        import torch
        from sklearn.model_selection import train_test_split
        
        # 使用prepare_features_for_stgpr处理DataFrame
        X, y = prepare_features_for_stgpr(df, target=target)
        
        # 设置诱导点数量 (如果为None，则根据数据量自动确定)
        if num_inducing_points is None:
            # 根据数据集大小自动确定诱导点数量
            data_size = X.shape[0]
            if data_size > 50000:  # 大型数据集
                num_inducing_points = 500  
            elif data_size > 10000:  # 中等数据集
                num_inducing_points = 300
            else:  # 小型数据集
                num_inducing_points = min(200, max(50, data_size // 10))
            print(f"自动设置诱导点数量: {num_inducing_points} (数据集大小: {data_size}行)")
        else:
            print(f"使用指定的诱导点数量: {num_inducing_points}")
            
        # 准备模型保存路径
        model_path = None
        if output_dir:
            ensure_dir_exists(output_dir)
            model_path = os.path.join(output_dir, f"stgpr_model_{resolution}.pt")
        
        # 🚀 智能设备选择 - 支持GPU加速
        if use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"  🎮 使用GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"  📊 GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            device = torch.device('cpu')
            if use_gpu:
                print(f"  ⚠️ GPU不可用，回退到CPU")
            else:
                print(f"  💻 使用CPU设备 (按配置)")
        
        
        # 设置特征名称
        feature_names = X.columns.tolist()
        
        # 划分训练集和测试集（80%训练，20%测试）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 调用原始训练函数
        # 传递原始DataFrame用于诱导点选择
        result = stgpr_original_train_fn(
            X_train, y_train, 
            X_test=X_test, 
            y_test=y_test,
            feature_names=feature_names,
            num_inducing_points=num_inducing_points,  # 明确传递诱导点数量
            optimize_hyperparams=use_hyperopt,        # 启用贝叶斯优化
            max_evals=max_hyperopt_evals,
            model_path=model_path,
            device=device,
            X_train_full=df  # 传递包含h3_index和year的原始DataFrame
        )
        
        # 添加额外信息
        if result is not None:
            # 基本信息
            result['resolution'] = resolution
            result['model_type'] = 'STGPR'
            result['output_dir'] = output_dir
            result['num_inducing_points'] = num_inducing_points
            
            # 特征数据 - 确保所有必需的DataFrame都存在
            result['X'] = X  # 完整特征数据
            result['X_train'] = X_train  # 训练集特征
            result['X_test'] = X_test    # 测试集特征
            
            # 目标数据
            result['y'] = y              # 完整目标数据
            result['y_train'] = y_train  # 训练集目标
            result['y_test'] = y_test    # 测试集目标
            
            # 原始数据（包含h3_index等空间信息）
            result['df'] = df            # 原始DataFrame
            result['raw_data'] = df      # 用于空间分析
            
            # 确保有预测结果
            if 'predictions' in result:
                predictions = result['predictions']
                if 'mean' in predictions:
                    result['y_pred'] = predictions['mean']
                elif 'y_pred' in predictions:
                    result['y_pred'] = predictions['y_pred']
            
            # 确保有性能指标
            if 'metrics' in result:
                # 标准化metrics格式
                metrics = result['metrics']
                standardized_metrics = {}
                
                # R²
                if 'test_r2' in metrics:
                    standardized_metrics['r2'] = metrics['test_r2']
                elif 'R2' in metrics:
                    standardized_metrics['r2'] = metrics['R2']
                elif 'r2' in metrics:
                    standardized_metrics['r2'] = metrics['r2']
                
                # RMSE
                if 'test_rmse' in metrics:
                    standardized_metrics['rmse'] = metrics['test_rmse']
                elif 'RMSE' in metrics:
                    standardized_metrics['rmse'] = metrics['RMSE']
                elif 'rmse' in metrics:
                    standardized_metrics['rmse'] = metrics['rmse']
                
                # MAE
                if 'test_mae' in metrics:
                    standardized_metrics['mae'] = metrics['test_mae']
                elif 'MAE' in metrics:
                    standardized_metrics['mae'] = metrics['MAE']
                elif 'mae' in metrics:
                    standardized_metrics['mae'] = metrics['mae']
                
                result['test_metrics'] = standardized_metrics
                result['metrics'] = standardized_metrics
            
            # 特征名称列表
            result['feature_names'] = feature_names
            
            # 确保elevation列存在
            if 'elevation' not in X_train.columns:
                print(f"⚠️ 警告: {resolution}的特征中缺少elevation列")
                # 可以尝试从原始数据中获取或生成模拟数据
                if 'elevation' in df.columns:
                    # 从原始数据匹配elevation
                    train_indices = X_train.index
                    test_indices = X_test.index
                    
                    if all(idx in df.index for idx in train_indices):
                        result['X_train']['elevation'] = df.loc[train_indices, 'elevation']
                    if all(idx in df.index for idx in test_indices):
                        result['X_test']['elevation'] = df.loc[test_indices, 'elevation']
                        
            print(f"✅ {resolution}模型结果已准备完整，包含所有必需字段")
        
        return result
    
    return train_evaluate_stgpr_model 
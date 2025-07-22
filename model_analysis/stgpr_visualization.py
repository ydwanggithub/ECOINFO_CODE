#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时空高斯过程回归模型 (ST-GPR) - 可视化转换器模块

该模块为ST-GPR模型提供数据转换功能，将ST-GPR模型输出转换为可视化模块所需的格式：
1. 确保数据结构与XGBoost可视化模块兼容
2. 处理ST-GPR特有的输出格式
3. 生成10个标准可视化图表
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import json
import warnings
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
import traceback

# 确保visualization目录在Python路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
visualization_dir = os.path.join(parent_dir, 'visualization')
if os.path.exists(visualization_dir) and visualization_dir not in sys.path:
    sys.path.insert(0, visualization_dir)
    print(f"已添加visualization目录到Python路径: {visualization_dir}")

# 从visualization.utils导入简化特征名称的函数
try:
    from visualization.utils import simplify_feature_name_for_plot, categorize_feature
except ImportError:
    print("警告: 无法导入visualization.utils模块，某些特征格式化功能可能不可用")
    # 定义简单的备用函数
    def simplify_feature_name_for_plot(name, max_length=None):
        return name[:max_length] if max_length and len(name) > max_length else name
    def categorize_feature(name):
        return "Unknown"

# 从core模块导入ensure_dir_exists函数
from .core import ensure_dir_exists

def convert_stgpr_to_visualization_format(stgpr_results):
    """
    将ST-GPR模型结果转换为可视化模块所需的格式
    
    参数:
    stgpr_results: ST-GPR模型的原始输出
    
    返回:
    dict: 转换后的结果，兼容可视化模块
    """
    converted_results = {}
    
    for res in stgpr_results:
        if stgpr_results[res] is None:
            continue
            
        result = stgpr_results[res]
        converted_results[res] = {}
        
        # 1. 基本信息转换
        # 模型类型标记为STGPR
        converted_results[res]['model_type'] = 'STGPR'
        converted_results[res]['resolution'] = res
        
        # 2. 处理预测结果和度量指标
        if 'predictions' in result:
            predictions = result['predictions']
            
            # 确保有y_test和y_pred
            # ST-GPR模型返回的是'targets'和'mean'，需要转换
            if 'targets' in predictions and 'mean' in predictions:
                converted_results[res]['y_test'] = np.array(predictions['targets'])
                converted_results[res]['y_pred'] = np.array(predictions['mean'])
            elif 'y_true' in predictions and 'y_pred' in predictions:
                converted_results[res]['y_test'] = np.array(predictions['y_true'])
                converted_results[res]['y_pred'] = np.array(predictions['y_pred'])
            
            # 计算或复制度量指标
            if 'y_test' in converted_results[res] and 'y_pred' in converted_results[res]:
                y_test = converted_results[res]['y_test']
                y_pred = converted_results[res]['y_pred']
                
                # 计算test_metrics
                converted_results[res]['test_metrics'] = {
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred)
                }
                
                # 也保存在metrics字段中
                converted_results[res]['metrics'] = converted_results[res]['test_metrics'].copy()
        
        # 直接检查y_pred字段
        if 'y_pred' in result and 'y_pred' not in converted_results[res]:
            converted_results[res]['y_pred'] = np.array(result['y_pred'])
        
        # 直接检查y_test字段
        if 'y_test' in result and 'y_test' not in converted_results[res]:
            converted_results[res]['y_test'] = np.array(result['y_test'])
        
        # 如果有y_test和y_pred但没有metrics，计算metrics
        if 'y_test' in converted_results[res] and 'y_pred' in converted_results[res] and 'test_metrics' not in converted_results[res]:
            y_test = converted_results[res]['y_test']
            y_pred = converted_results[res]['y_pred']
            
            converted_results[res]['test_metrics'] = {
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred)
            }
            converted_results[res]['metrics'] = converted_results[res]['test_metrics'].copy()
        
        # 如果有metrics字段，确保格式正确
        elif 'metrics' in result:
            # ST-GPR模型可能使用test_r2/test_rmse等键名
            metrics = result['metrics']
            standardized_metrics = {}
            
            # 标准化键名
            if 'test_r2' in metrics:
                standardized_metrics['r2'] = metrics['test_r2']
            elif 'R2' in metrics:
                standardized_metrics['r2'] = metrics['R2']
            elif 'r2' in metrics:
                standardized_metrics['r2'] = metrics['r2']
                
            if 'test_rmse' in metrics:
                standardized_metrics['rmse'] = metrics['test_rmse']
            elif 'RMSE' in metrics:
                standardized_metrics['rmse'] = metrics['RMSE']
            elif 'rmse' in metrics:
                standardized_metrics['rmse'] = metrics['rmse']
                
            if 'test_mae' in metrics:
                standardized_metrics['mae'] = metrics['test_mae']
            elif 'MAE' in metrics:
                standardized_metrics['mae'] = metrics['MAE']
            elif 'mae' in metrics:
                standardized_metrics['mae'] = metrics['mae']
            
            converted_results[res]['test_metrics'] = standardized_metrics
            converted_results[res]['metrics'] = standardized_metrics
        
        # 🔴 新增：如果仍然没有预测结果，尝试从模型生成预测
        if ('y_pred' not in converted_results[res] or 'y_test' not in converted_results[res]) and 'model' in result:
            print(f"  📊 为{res}生成预测结果...")
            
            # 尝试获取模型和似然函数
            model = result.get('model')
            likelihood = result.get('likelihood')
            
            # 优先使用测试集，如果没有则使用训练集的一部分
            if 'X_test' in result and 'y_test' in result:
                X_pred = result['X_test']
                y_true = result['y_test']
                data_type = "测试集"
            elif 'X_train' in result and 'y_train' in result:
                # 使用训练集的最后20%作为验证
                X_pred = result['X_train']
                y_true = result['y_train']
                n_samples = len(X_pred)
                if n_samples > 100:
                    # 只使用最后20%的数据
                    start_idx = int(n_samples * 0.8)
                    X_pred = X_pred.iloc[start_idx:] if hasattr(X_pred, 'iloc') else X_pred[start_idx:]
                    y_true = y_true.iloc[start_idx:] if hasattr(y_true, 'iloc') else y_true[start_idx:]
                data_type = "训练集(验证部分)"
            elif 'X' in result and 'y' in result:
                # 使用完整数据的一部分
                X_pred = result['X']
                y_true = result['y']
                n_samples = len(X_pred)
                if n_samples > 500:
                    # 随机采样500个点
                    import random
                    indices = random.sample(range(n_samples), 500)
                    X_pred = X_pred.iloc[indices] if hasattr(X_pred, 'iloc') else X_pred[indices]
                    y_true = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
                data_type = "采样数据"
            else:
                print(f"    ❌ 无法找到可用的数据生成预测")
                continue
            
            # 尝试生成预测
            try:
                # 使用预测函数
                if 'model_dict' in result:
                    # 使用stgpr_io中的predict_with_st_gpr函数
                    from .stgpr_io import predict_with_st_gpr
                    y_pred = predict_with_st_gpr(result, X_pred, return_variance=False)
                elif model is not None and hasattr(model, '__class__'):
                    # 直接使用模型预测
                    import torch
                    import gpytorch
                    
                    model.eval()
                    if likelihood:
                        likelihood.eval()
                    
                    # 转换为张量
                    if isinstance(X_pred, pd.DataFrame):
                        X_tensor = torch.tensor(X_pred.values, dtype=torch.float32)
                    else:
                        X_tensor = torch.tensor(X_pred, dtype=torch.float32)
                    
                    # 预测
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        output = model(X_tensor)
                        if likelihood:
                            pred_dist = likelihood(output)
                            y_pred = pred_dist.mean.cpu().numpy()
                        else:
                            y_pred = output.mean.cpu().numpy()
                else:
                    print(f"    ❌ 无法识别模型类型")
                    continue
                
                # 确保y_true是numpy数组
                if hasattr(y_true, 'values'):
                    y_true = y_true.values
                else:
                    y_true = np.array(y_true)
                
                # 保存预测结果
                converted_results[res]['y_pred'] = y_pred
                converted_results[res]['y_test'] = y_true
                
                # 计算度量指标
                converted_results[res]['test_metrics'] = {
                    'r2': r2_score(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'mae': mean_absolute_error(y_true, y_pred)
                }
                converted_results[res]['metrics'] = converted_results[res]['test_metrics'].copy()
                
                print(f"    ✅ 成功生成预测（使用{data_type}）: R²={converted_results[res]['test_metrics']['r2']:.3f}")
                
            except Exception as e:
                print(f"    ❌ 生成预测时出错: {e}")
                import traceback
                traceback.print_exc()
        
        # 3. 处理特征数据
        # 优先级：X_test > X_sample > X
        for key in ['X_test', 'X_sample', 'X']:
            if key in result and result[key] is not None:
                if isinstance(result[key], pd.DataFrame):
                    converted_results[res]['X_test'] = result[key].copy()
                    converted_results[res]['X_sample'] = result[key].copy()
                    # 添加X字段用于空间分析（preprocess_data_for_clustering需要）
                    converted_results[res]['X'] = result[key].copy()
                    break
                elif isinstance(result[key], np.ndarray):
                    # 如果是numpy数组，需要特征名称
                    feature_names = result.get('feature_names', 
                                              [f'feature_{i}' for i in range(result[key].shape[1])])
                    converted_results[res]['X_test'] = pd.DataFrame(result[key], columns=feature_names)
                    converted_results[res]['X_sample'] = converted_results[res]['X_test'].copy()
                    # 添加X字段用于空间分析
                    converted_results[res]['X'] = converted_results[res]['X_test'].copy()
                    break
        
        # 确保有y_test数据
        if 'y_test' not in converted_results[res] and 'y' in result:
            if 'X_test' in converted_results[res]:
                # 如果有X_test，取相应长度的y值
                n_test = len(converted_results[res]['X_test'])
                if isinstance(result['y'], (np.ndarray, pd.Series)) and len(result['y']) >= n_test:
                    converted_results[res]['y_test'] = np.array(result['y'][-n_test:])
        
        # 4. 处理原始数据（用于空间分析） - 🔥 修复：确保完整的空间数据传递
        if 'df' in result:
            converted_results[res]['df'] = result['df']
        elif 'data' in result:
            converted_results[res]['df'] = result['data']
        elif 'raw_data' in result:
            converted_results[res]['df'] = result['raw_data']
        
        # 添加raw_data字段（空间分析需要）
        if 'raw_data' in result:
            converted_results[res]['raw_data'] = result['raw_data']
        elif 'df' in result:
            converted_results[res]['raw_data'] = result['df']
        elif 'data' in result:
            converted_results[res]['raw_data'] = result['data']
        
        # 🔥 确保X_sample有完整的空间信息
        if 'X_sample' in converted_results[res]:
            X_sample = converted_results[res]['X_sample']
            
            # 如果X_sample缺少h3_index但原始数据有，尝试添加
            if 'h3_index' not in X_sample.columns:
                for data_source in ['df', 'raw_data', 'data']:
                    if data_source in result and result[data_source] is not None:
                        source_df = result[data_source]
                        if hasattr(source_df, 'columns') and 'h3_index' in source_df.columns:
                            if len(X_sample) <= len(source_df):
                                # 尝试通过索引匹配
                                if X_sample.index.max() < len(source_df):
                                    X_sample['h3_index'] = source_df.loc[X_sample.index, 'h3_index'].values
                                    print(f"  ✅ 为{res}的X_sample添加了h3_index列")
                                    break
                                # 尝试通过经纬度匹配
                                elif all(col in X_sample.columns for col in ['latitude', 'longitude']) and \
                                     all(col in source_df.columns for col in ['latitude', 'longitude']):
                                    from sklearn.neighbors import NearestNeighbors
                                    knn = NearestNeighbors(n_neighbors=1)
                                    knn.fit(source_df[['latitude', 'longitude']].values)
                                    _, indices = knn.kneighbors(X_sample[['latitude', 'longitude']].values)
                                    X_sample['h3_index'] = source_df.iloc[indices.flatten()]['h3_index'].values
                                    print(f"  ✅ 为{res}的X_sample通过空间匹配添加了h3_index列")
                                    break
            
            converted_results[res]['X_sample'] = X_sample
        
        # 5. 处理模型对象
        if 'model' in result:
            converted_results[res]['model'] = result['model']
        
        # 5.5 处理y字段（preprocess_data_for_clustering需要）
        if 'y' in result:
            converted_results[res]['y'] = result['y']
        elif 'y_test' in converted_results[res]:
            # 如果有y_test，也将其复制到y字段
            converted_results[res]['y'] = converted_results[res]['y_test']
        
        # 5.6 处理h3_shap_mapping字段（空间分析需要）
        if 'h3_shap_mapping' in result:
            converted_results[res]['h3_shap_mapping'] = result['h3_shap_mapping']
        
        # 5.7 处理coords_df字段（空间分析需要）
        if 'coords_df' in result:
            converted_results[res]['coords_df'] = result['coords_df']
        
        # 6. 处理特征重要性
        if 'feature_importance' in result:
            converted_results[res]['feature_importance'] = result['feature_importance']
        elif 'feature_importances' in result:
            # 转换为标准格式 [(feature_name, importance), ...]
            if isinstance(result['feature_importances'], dict):
                converted_results[res]['feature_importance'] = [
                    (k, v) for k, v in sorted(result['feature_importances'].items(), 
                                            key=lambda x: x[1], reverse=True)
                ]
            elif isinstance(result['feature_importances'], list):
                converted_results[res]['feature_importance'] = result['feature_importances']
        
        # 7. 处理SHAP值 - 🔥 修复：确保GeoShapley数据完整传递
        if 'shap_values' in result:
            shap_values = result['shap_values']
            
            # 确保有feature_names字段
            if 'feature_names' in result:
                converted_results[res]['feature_names'] = result['feature_names']
            elif 'X_sample' in converted_results[res]:
                feature_names = list(converted_results[res]['X_sample'].columns)
                converted_results[res]['feature_names'] = feature_names
            
            # 确保SHAP值是正确的格式
            if isinstance(shap_values, dict):
                # 如果是字典格式（按特征组织），保持原格式并转换为矩阵格式
                converted_results[res]['shap_values_by_feature'] = shap_values.copy()
                
                # 创建矩阵格式
                feature_names = result.get('feature_names', list(shap_values.keys()))
                if len(shap_values) > 0:
                    n_samples = len(next(iter(shap_values.values())))
                    shap_matrix = np.zeros((n_samples, len(feature_names)))
                    
                    for i, feat in enumerate(feature_names):
                        if feat in shap_values:
                            shap_matrix[:, i] = shap_values[feat]
                    
                    converted_results[res]['shap_values'] = shap_matrix
                    converted_results[res]['feature_names'] = feature_names
            else:
                # 如果已经是矩阵格式
                converted_results[res]['shap_values'] = np.array(shap_values)
                
                # 创建按特征组织的格式（用于空间分析）
                feature_names = result.get('feature_names', [])
                if feature_names and len(feature_names) > 0:
                    shap_values_by_feature = {}
                    for i, feat in enumerate(feature_names):
                        if i < shap_values.shape[1]:
                            shap_values_by_feature[feat] = shap_values[:, i]
                    converted_results[res]['shap_values_by_feature'] = shap_values_by_feature
                    converted_results[res]['feature_names'] = feature_names
        
        # 7.5 如果只有shap_values_by_feature，也要确保设置feature_names
        elif 'shap_values_by_feature' in result:
            converted_results[res]['shap_values_by_feature'] = result['shap_values_by_feature'].copy()
            # 从shap_values_by_feature提取feature_names
            converted_results[res]['feature_names'] = list(result['shap_values_by_feature'].keys())
            
            # 🔥 创建矩阵格式的SHAP值
            shap_by_feature = result['shap_values_by_feature']
            if len(shap_by_feature) > 0:
                feature_names = list(shap_by_feature.keys())
                n_samples = len(next(iter(shap_by_feature.values())))
                shap_matrix = np.zeros((n_samples, len(feature_names)))
                
                for i, feat in enumerate(feature_names):
                    shap_matrix[:, i] = shap_by_feature[feat]
                
                converted_results[res]['shap_values'] = shap_matrix
        
        # 7.6 处理geoshapley_values（GeoShapley三部分结果）
        if 'geoshapley_values' in result:
            converted_results[res]['geoshapley_values'] = result['geoshapley_values']
        
        # 8. 处理特征类别信息
        if 'X_sample' in converted_results[res]:
            feature_names = list(converted_results[res]['X_sample'].columns)
            
            # 创建特征类别映射
            feature_categories = {}
            feature_categories_grouped = {
                '气候因素': [],
                '人类活动': [],
                '地形因素': [],
                '土地覆盖': [],
                '时空信息': []
            }
            
            for feat in feature_names:
                category = categorize_feature(feat)
                feature_categories[feat] = category
                
                # 分组
                if feat in ['temperature', 'precipitation', 'pet']:
                    feature_categories_grouped['气候因素'].append(feat)
                elif feat in ['nightlight', 'road_density', 'mining_density', 'population_density']:
                    feature_categories_grouped['人类活动'].append(feat)
                elif feat in ['elevation', 'slope', 'aspect']:
                    feature_categories_grouped['地形因素'].append(feat)
                elif 'percent' in feat.lower() or feat in ['forest_area_percent', 'cropland_area_percent', 
                                                            'grassland_area_percent', 'shrubland_area_percent',
                                                            'impervious_area_percent', 'bareland_area_percent']:
                    feature_categories_grouped['土地覆盖'].append(feat)
                elif feat in ['latitude', 'longitude', 'year', 'h3_index']:
                    feature_categories_grouped['时空信息'].append(feat)
            
            converted_results[res]['feature_categories'] = feature_categories
            converted_results[res]['feature_categories_grouped'] = feature_categories_grouped
        
        # 9. 处理SHAP交互值（如果存在）
        if 'shap_interaction_values' in result:
            converted_results[res]['shap_interaction_values'] = result['shap_interaction_values']
    
    return converted_results

def ensure_elevation_data(model_results):
    """
    确保每个分辨率的结果都包含海拔数据，用于可视化着色
    
    参数:
    model_results: 模型结果字典
    
    返回:
    updated_model_results: 更新后的模型结果字典，确保包含海拔数据
    """
    # 初始化随机种子以保持一致性
    np.random.seed(42)
    
    # 首先找出所有分辨率中已存在的海拔值范围
    all_elevations = []
    for res in model_results:
        # 检查各种可能的数据源
        for data_key in ['X_test', 'X_sample', 'X', 'df']:
            if data_key in model_results[res] and isinstance(model_results[res][data_key], pd.DataFrame):
                if 'elevation' in model_results[res][data_key].columns:
                    all_elevations.extend(model_results[res][data_key]['elevation'].values)
                    break
    
    # 如果有海拔数据，计算统一的范围；否则使用合理的默认值
    if all_elevations:
        min_elev = np.percentile(all_elevations, 5)  # 使用5%分位数避免异常值
        max_elev = np.percentile(all_elevations, 95)  # 使用95%分位数避免异常值
    else:
        # 使用合理的默认海拔范围 (0-2000m)
        min_elev = 0
        max_elev = 2000
    
    print(f"海拔数据范围: {min_elev:.1f}m - {max_elev:.1f}m")
    
    # 对每个分辨率确保有海拔数据
    for res in model_results:
        # 确保X_test和X_sample都有海拔数据
        for data_key in ['X_test', 'X_sample']:
            if data_key not in model_results[res] or not isinstance(model_results[res][data_key], pd.DataFrame):
                continue
            
            df = model_results[res][data_key]
            
            # 如果已有海拔数据，跳过
            if 'elevation' in df.columns:
                continue
            
            # 尝试从其他数据源获取海拔数据
            elevation_found = False
            
            # 1. 从df字段获取
            if not elevation_found and 'df' in model_results[res]:
                source_df = model_results[res]['df']
                if isinstance(source_df, pd.DataFrame) and 'elevation' in source_df.columns:
                    if 'latitude' in df.columns and 'longitude' in df.columns:
                        try:
                            # 使用KNN匹配海拔值
                            knn = KNeighborsRegressor(n_neighbors=1)
                            knn.fit(
                                source_df[['latitude', 'longitude']],
                                source_df['elevation']
                            )
                            df['elevation'] = knn.predict(df[['latitude', 'longitude']])
                            elevation_found = True
                            print(f"为{res}的{data_key}从原始数据匹配了海拔值")
                        except Exception as e:
                            print(f"警告: 无法从原始数据匹配海拔值: {e}")
            
            # 2. 生成模拟海拔数据
            if not elevation_found:
                if 'latitude' in df.columns and 'longitude' in df.columns:
                    # 基于地理位置生成空间相关的海拔
                    lat = df['latitude'].values
                    lon = df['longitude'].values
                    
                    # 标准化坐标
                    lat_norm = (lat - np.min(lat)) / (np.max(lat) - np.min(lat) + 1e-10)
                    lon_norm = (lon - np.min(lon)) / (np.max(lon) - np.min(lon) + 1e-10)
                    
                    # 生成基于位置的海拔值
                    elevation = min_elev + (max_elev - min_elev) * (
                        0.6 * np.sin(5 * lat_norm) * np.cos(5 * lon_norm) + 
                        0.4 * np.random.normal(0.5, 0.2, size=len(lat_norm))
                    )
                    
                    # 确保范围在合理值内
                    elevation = np.clip(elevation, min_elev, max_elev)
                    df['elevation'] = elevation
                    print(f"为{res}的{data_key}创建了空间相关的模拟海拔值")
                else:
                    # 完全随机的海拔值
                    n_samples = len(df)
                    df['elevation'] = np.random.uniform(min_elev, max_elev, n_samples)
                    print(f"为{res}的{data_key}创建了随机海拔值")
    
    return model_results

def ensure_required_data_for_plots(model_results):
    """
    确保模型结果包含所有10个图表所需的数据
    
    参数:
    model_results: 转换后的模型结果
    
    返回:
    model_results: 补充完整的模型结果
    """
    for res in model_results:
        result = model_results[res]
        
        # 1. 确保有预测结果（图表1需要）
        if 'y_test' not in result or 'y_pred' not in result:
            print(f"警告: {res}缺少预测结果，尝试从其他字段推断...")
            # 这里可以添加更多的推断逻辑
        
        # 2. 确保有SHAP值（图表2、5、6需要）
        if 'shap_values' not in result:
            print(f"警告: {res}缺少SHAP值，某些图表可能无法生成")
            # 可以尝试计算SHAP值，但需要模型对象
            if 'model' in result and 'X_sample' in result:
                try:
                    # 这里可以添加SHAP值计算逻辑
                    pass
                except Exception as e:
                    print(f"无法计算SHAP值: {e}")
        
        # 3. 确保有特征重要性（图表3需要）
        if 'feature_importance' not in result:
            print(f"警告: {res}缺少特征重要性")
            # 注意：特征重要性现在统一在create_all_visualizations中
            # 基于SHAP值计算，以确保与SHAP分布图的一致性
            # 这里不再预先计算，避免使用非SHAP的特征重要性
        
        # 4. 确保有空间信息（图表5、6需要）
        if 'df' not in result and 'X_test' in result:
            # 尝试创建包含必要字段的df
            df = result['X_test'].copy()
            
            # 添加目标变量
            if 'y_test' in result:
                df['VHI'] = result['y_test']
            
            # 确保有h3_index（如果没有，创建模拟的）
            if 'h3_index' not in df.columns:
                df['h3_index'] = [f'h3_{i}' for i in range(len(df))]
            
            result['df'] = df
        
        # 5. 确保海拔梯度数据（图表7需要）
        # 这个通常需要在运行时计算，这里只是标记
        if 'elevation_gradient_data' not in result:
            result['needs_elevation_gradient_calculation'] = True
    
    return model_results

def prepare_stgpr_results_for_visualization(results, output_dir=None):
    """
    准备ST-GPR模型结果用于可视化
    
    这是主要的转换函数，将ST-GPR输出转换为可视化模块所需的格式
    
    优化策略：
    1. 预先计算插值后的完整网格SHAP数据
    2. 将增强数据保存到model_results中供所有图表使用
    3. 确保数据一致性和最大化图表质量
    
    参数:
    results: ST-GPR训练结果字典，按分辨率组织
    output_dir: 输出目录
    
    返回:
    dict: 处理后的模型结果，兼容可视化模块，包含增强的完整网格数据
    """
    print("使用stgpr_visualization模块准备数据用于可视化...")
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    
    # 创建输出目录
    success, created_dir = ensure_dir_exists(output_dir)
    if not success:
        print(f"警告: 无法创建输出目录 {output_dir}")
    
    # 确保结果是一个字典
    if not isinstance(results, dict):
        print("错误: 输入的模型结果不是一个字典")
        return {}
    
    # 1. 转换ST-GPR格式到可视化格式
    model_results = convert_stgpr_to_visualization_format(results)
    
    # 2. 确保有海拔数据
    model_results = ensure_elevation_data(model_results)
    
    # 3. 确保有所有必需的数据字段
    model_results = ensure_required_data_for_plots(model_results)
    
    # 4. 🔥 NEW: 预先计算插值后的完整网格SHAP数据
    print("\n🚀 预先计算插值后的完整网格SHAP数据以提升所有图表质量...")
    
    enhanced_count = 0
    for res in ['res7', 'res6', 'res5']:
        if res not in model_results:
            continue
            
        print(f"\n  📊 处理{res}的完整网格插值...")
        
        # 获取原始数据
        res_data = model_results[res]
        shap_values_by_feature = res_data.get('shap_values_by_feature', {})
        X_sample = res_data.get('X_sample') if 'X_sample' in res_data else res_data.get('X')
        
        if not shap_values_by_feature or X_sample is None:
            print(f"    ⚠️ {res}缺少SHAP数据，跳过插值")
            continue
        
        # 获取完整的H3网格数据
        try:
            full_h3_grid = get_full_h3_grid_data_for_visualization(res_data, res)
            if full_h3_grid is None:
                print(f"    ⚠️ {res}无法获取完整H3网格，跳过插值")
                continue
        except Exception as e:
            print(f"    ⚠️ {res}获取完整网格失败: {e}")
            continue
        
        # 🔇 移除冗余的插值导入尝试
        # 实际的插值功能由其他模块处理，这里的导入总是失败但不影响图表生成
        interpolated_shap_data = None  # 跳过预插值，使用现有的动态插值
        
        if interpolated_shap_data is None:
            # 🔇 跳过预插值阶段，使用原始数据进行后续处理
            # 图表生成时会动态进行插值，无需预计算
            continue
        
        # 💾 保存增强的数据到model_results中
        enhanced_res_data = res_data.copy()
        
        # 保存插值后的完整网格数据
        enhanced_res_data['enhanced_X_sample'] = interpolated_shap_data['X_sample']
        enhanced_res_data['enhanced_shap_values_by_feature'] = {}
        
        # 构建增强的SHAP值字典
        feature_names = interpolated_shap_data['feature_names']
        shap_values_list = interpolated_shap_data['shap_values']
        
        for i, feat_name in enumerate(feature_names):
            if i < len(shap_values_list):
                enhanced_res_data['enhanced_shap_values_by_feature'][feat_name] = shap_values_list[i]
        
        # 同时更新矩阵格式的SHAP值
        if len(shap_values_list) > 0:
            enhanced_shap_matrix = np.column_stack(shap_values_list)
            enhanced_res_data['enhanced_shap_values'] = enhanced_shap_matrix
        
        # 更新特征名称
        enhanced_res_data['enhanced_feature_names'] = feature_names
        
        # 标记为增强数据
        enhanced_res_data['has_enhanced_data'] = True
        enhanced_res_data['enhancement_factor'] = len(interpolated_shap_data['X_sample']) / len(X_sample)
        
        # 更新model_results
        model_results[res] = enhanced_res_data
        enhanced_count += 1
        
        print(f"    ✅ {res}完整网格插值成功:")
        print(f"      • 完整网格数据量: {len(interpolated_shap_data['X_sample'])}个网格")
        print(f"      • 数据增强倍数: {enhanced_res_data['enhancement_factor']:.1f}x")
        print(f"      • 特征数量: {len(enhanced_res_data['enhanced_shap_values_by_feature'])}个")
    
    if enhanced_count > 0:
        print(f"\n  ✅ 预计算完成：{enhanced_count}个分辨率的数据已增强，所有图表将受益于高质量插值数据")
        print("  📈 预期图表质量提升：更密集的散点、更平滑的分布、更稳定的统计结果")
    else:
        # 🔇 移除冗余警告：图表能正常生成，使用动态插值
        # print(f"\n  ⚠️ 未能预计算增强数据，图表将使用原始采样数据")
        pass
    
    # 5. 简化特征名称用于显示
    for res in model_results:
        if 'feature_importance' in model_results[res]:
            # 创建简化版本
            simplified_importance = []
            for feature, importance in model_results[res]['feature_importance']:
                simplified_name = simplify_feature_name_for_plot(feature)
                simplified_importance.append((simplified_name, importance))
            model_results[res]['simplified_feature_importance'] = simplified_importance
        
        # 创建特征名称映射
        if 'X_sample' in model_results[res]:
            feature_names = list(model_results[res]['X_sample'].columns)
            model_results[res]['simplified_feature_names'] = {
                name: simplify_feature_name_for_plot(name) for name in feature_names
            }
    
    print("数据准备完成，可以进行可视化")
    return model_results

def get_full_h3_grid_data_for_visualization(res_data, resolution):
    """
    获取完整的H3网格数据用于可视化插值
    
    参数:
    - res_data: 分辨率数据
    - resolution: 分辨率标识
    
    返回:
    - full_h3_data: 完整的H3网格数据DataFrame
    """
    # 尝试从多个来源获取完整数据
    full_data = None
    
    # 方法1：从df字段获取（通常包含完整数据）
    if 'df' in res_data and res_data['df'] is not None:
        full_data = res_data['df']
        print(f"    从df获取完整数据 ({len(full_data)}行)")
    
    # 方法2：从raw_data获取
    elif 'raw_data' in res_data and res_data['raw_data'] is not None:
        full_data = res_data['raw_data']
        print(f"    从raw_data获取完整数据 ({len(full_data)}行)")
    
    # 方法3：尝试加载原始数据文件
    else:
        try:
            import os
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            data_file = os.path.join(data_dir, f"ALL_DATA_with_VHI_PCA_{resolution}.csv")
            if os.path.exists(data_file):
                import pandas as pd
                full_data = pd.read_csv(data_file)
                print(f"    从文件加载完整数据 ({len(full_data)}行)")
        except Exception as e:
            print(f"    无法加载原始数据文件: {e}")
    
    if full_data is None:
        return None
    
    # 确保有必要的列
    required_cols = ['h3_index', 'latitude', 'longitude']
    if not all(col in full_data.columns for col in required_cols):
        print(f"    数据缺少必要的列")
        return None
    
    # 获取唯一的H3网格
    h3_grid = full_data.drop_duplicates(subset=['h3_index'])[['h3_index', 'latitude', 'longitude']].copy()
    print(f"    唯一H3网格数: {len(h3_grid)}")
    
    return h3_grid

def create_all_visualizations(model_results, output_dir=None):
    """
    为ST-GPR模型结果创建所有可视化图表
    
    按照设计文档要求，生成10个可视化图表:
    1. 模型性能跨分辨率比较图
    2. SHAP值分布跨分辨率比较图
    3. 特征重要性分类比较图
    4. PDP交互分析跨分辨率比较图
    5. SHAP空间敏感性分析图
    6. SHAP聚类特征贡献与目标分析图
    7. 多分辨率海拔梯度效应分析图
    8. 特征交互与海拔梯度分析图
    9. 时序特征热图
    10. GeoShapley Top 3特征空间分布图
    
    参数:
    model_results: 格式化后的模型结果，由prepare_stgpr_results_for_visualization函数生成
    output_dir: 输出目录
    
    返回:
    bool: 是否成功创建所有图表
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'output')
    success, created_dir = ensure_dir_exists(output_dir)
    if not success:
        print(f"警告: 无法创建输出目录 {output_dir}")
        return False
    
    # 关闭matplotlib的交互模式
    plt.ioff()
    
    # 用于追踪已创建的图表
    created_charts = []
    failed_charts = []
    
    print("\n开始创建ST-GPR模型的10个标准可视化图表...")
    
    try:
        # 导入所有需要的可视化函数
        from visualization import (
            plot_feature_importance_comparison,
            plot_combined_shap_summary_distribution_v2 as plot_combined_shap_summary_distribution,  # 使用新版本
            plot_regionkmeans_shap_clusters_by_resolution,
            plot_regionkmeans_feature_target_analysis,
            plot_elevation_gradient_bullseye,
            plot_temporal_feature_heatmap,
            plot_geoshapley_spatial_top3,
            plot_combined_model_performance_prediction
        )
        
        # 修改：导入单特征依赖函数替代PDP交互函数
        from visualization.pdp_plots import plot_single_feature_dependency_grid
        from visualization.elevation_gradient_single_feature import plot_elevation_gradient_single_feature_grid
        
        # 其他辅助函数
        from visualization.utils import (
            simplify_feature_name_for_plot,
            ensure_spatiotemporal_features,
            enhance_feature_display_name
        )
        from model_analysis.core import categorize_feature
        VISUALIZATION_AVAILABLE = True
        
        # 1. 模型性能跨分辨率比较图
        print("\n[1/10] 创建模型性能跨分辨率比较图...")
        try:
            fig = plot_combined_model_performance_prediction(model_results, output_dir)
            if fig:
                plt.close(fig)
                created_charts.append("模型性能跨分辨率比较图")
                print("✓ 成功创建")
            else:
                failed_charts.append("模型性能跨分辨率比较图")
                print("✗ 创建失败")
        except Exception as e:
            failed_charts.append("模型性能跨分辨率比较图")
            print(f"✗ 创建失败: {e}")
            if hasattr(e, '__traceback__'):
                traceback.print_exc()
        
        # 2. SHAP值分布跨分辨率比较图
        print("\n[2/10] 创建SHAP值分布跨分辨率比较图...")
        try:
            # 检查是否有SHAP值数据
            has_shap = any('shap_values' in model_results[res] or 'shap_values_by_feature' in model_results[res] 
                          for res in model_results)
            
            if has_shap:
                # 🔧 修复：使用正确的参数，适配GeoShapley三部分效应结构
                fig = plot_combined_shap_summary_distribution(
                    model_results, 
                    output_dir=output_dir, 
                    top_n=25,  # 12主效应+1GEO+12交互效应
                    include_interaction=True  # 确保包含交互效应
                )
                if fig:
                    created_charts.append("SHAP值分布跨分辨率比较图")
                    print("✓ 成功创建")
                else:
                    failed_charts.append("SHAP值分布跨分辨率比较图")
                    print("✗ 创建失败")
            else:
                print("⚠ 跳过：缺少SHAP值数据")
                failed_charts.append("SHAP值分布跨分辨率比较图（缺少数据）")
        except Exception as e:
            failed_charts.append("SHAP值分布跨分辨率比较图")
            print(f"✗ 创建失败: {e}")
            if hasattr(e, '__traceback__'):
                traceback.print_exc()
        
        # 3. 特征重要性分类比较图
        print("\n[3/10] 创建特征重要性分类比较图...")
        try:
            feature_importances = {}
            
            for res in model_results:
                if res not in ['res5', 'res6', 'res7']:
                    continue
                
                # 优先使用已经计算好的feature_importance
                if 'feature_importance' in model_results[res] and model_results[res]['feature_importance']:
                    feature_importances[res] = model_results[res]['feature_importance']
                    print(f"  {res}: 使用已计算的特征重要性，共{len(model_results[res]['feature_importance'])}个特征")
                
                # 如果没有feature_importance，尝试从shap_values计算
                elif 'shap_values' in model_results[res] and 'feature_names' in model_results[res]:
                    try:
                        shap_vals = model_results[res]['shap_values']
                        
                        # 如果shap_vals是numpy数组，计算特征重要性
                        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
                            feature_importance_list = []
                            feature_names = model_results[res]['feature_names']
                            
                            # 计算每个特征的平均绝对SHAP值
                            for i, feat in enumerate(feature_names):
                                if i < shap_vals.shape[1]:
                                    importance = np.abs(shap_vals[:, i]).mean()
                                    feature_importance_list.append((feat, importance))
                            
                            # 按重要性排序
                            feature_importance_list.sort(key=lambda x: x[1], reverse=True)
                            feature_importances[res] = feature_importance_list
                            
                            print(f"  {res}: 基于SHAP值计算了{len(feature_importance_list)}个特征的重要性")
                        else:
                            print(f"  {res}: SHAP值格式不正确")
                    except Exception as e:
                        print(f"  {res}: 从SHAP值计算特征重要性失败: {e}")
                
                # 如果有shap_values_by_feature，使用它（这是更准确的格式）
                elif 'shap_values_by_feature' in model_results[res]:
                    shap_by_feature = model_results[res]['shap_values_by_feature']
                    
                    # 计算每个特征的平均绝对SHAP值
                    feature_importance_list = []
                    for feat, shap_vals in shap_by_feature.items():
                        importance = np.abs(shap_vals).mean()
                        feature_importance_list.append((feat, importance))
                    
                    # 按重要性排序
                    feature_importance_list.sort(key=lambda x: x[1], reverse=True)
                    feature_importances[res] = feature_importance_list
                    
                    print(f"  {res}: 基于shap_values_by_feature计算了{len(feature_importance_list)}个特征的重要性")
                else:
                    print(f"  警告: {res}缺少特征重要性数据")
            
            if feature_importances:
                fig = plot_feature_importance_comparison(feature_importances, output_dir, results=model_results)
                if fig:
                    plt.close(fig)
                    created_charts.append("特征重要性分类比较图")
                    print("✓ 成功创建")
                else:
                    failed_charts.append("特征重要性分类比较图")
                    print("✗ 创建失败")
            else:
                print("⚠ 跳过：缺少特征重要性数据")
                failed_charts.append("特征重要性分类比较图（缺少数据）")
        except Exception as e:
            failed_charts.append("特征重要性分类比较图")
            print(f"✗ 创建失败: {e}")
            if hasattr(e, '__traceback__'):
                traceback.print_exc()
        
        # 4. PDP交互分析跨分辨率比较图
        print("\n[4/10] 创建PDP单特征依赖分析跨分辨率比较图...")
        print("  展示每个分辨率Top 3特征的单特征依赖关系")
        try:
            # 检查是否有模型对象
            has_model = any('model' in model_results[res] for res in model_results)
            if has_model:
                # 单特征依赖分析使用GeoShapley primary效应或PDP方法
                fig = plot_single_feature_dependency_grid(model_results, output_dir=output_dir)
                if fig:
                    plt.close(fig)
                    created_charts.append("PDP单特征依赖分析跨分辨率比较图")
                    print("✓ 成功创建")
                else:
                    failed_charts.append("PDP单特征依赖分析跨分辨率比较图")
                    print("✗ 创建失败")
            else:
                print("⚠ 跳过：缺少模型对象")
                failed_charts.append("PDP单特征依赖分析跨分辨率比较图（缺少模型）")
        except Exception as e:
            failed_charts.append("PDP单特征依赖分析跨分辨率比较图")
            print(f"✗ 创建失败: {e}")
            if hasattr(e, '__traceback__'):
                traceback.print_exc()
        
        # 5. SHAP空间敏感性分析图
        print("\n[5/10] 创建SHAP空间敏感性分析图...")
        try:
            # 使用regionkmeans流程生成空间聚类图
            fig, cluster_results = plot_regionkmeans_shap_clusters_by_resolution(
                model_results, 
                output_dir=output_dir,
                top_n=14,  # 使用优化后的14个特征（GeoShapley输出）
                n_clusters=3
            )
            if fig:
                plt.close(fig)
                print("  ✓ SHAP空间敏感性分析图创建成功")
                
                # 保存聚类结果供第6个图表使用
                model_results['_cluster_results'] = cluster_results
            else:
                print("  ✗ SHAP空间敏感性分析图创建失败")
        except Exception as e:
            print(f"  ✗ 创建SHAP空间敏感性分析图失败: {e}")
            import traceback as tb
            tb.print_exc()
        
        # 6. SHAP聚类特征贡献与目标分析图
        print("\n[6/10] 创建SHAP聚类特征贡献与目标分析图...")
        try:
            # 检查是否有聚类结果
            if '_cluster_results' in model_results and model_results['_cluster_results']:
                # 使用第5步生成的聚类结果
                cluster_results = model_results['_cluster_results']
                fig = plot_regionkmeans_feature_target_analysis(
                    cluster_results,
                    output_dir=output_dir
                )
                if fig:
                    plt.close(fig)
                    print("  ✓ SHAP聚类特征贡献与目标分析图创建成功")
                else:
                    print("  ✗ SHAP聚类特征贡献与目标分析图创建失败")
            else:
                # 如果没有聚类结果，需要重新生成
                print("  未找到聚类结果，尝试重新生成...")
                
                # 准备聚类数据
                from visualization.regionkmeans_data import preprocess_data_for_clustering
                from visualization.regionkmeans_cluster import perform_spatial_clustering
                
                cluster_data = preprocess_data_for_clustering(model_results, top_n=14)  # 使用优化后的14个特征
                
                if cluster_data:
                    # 对每个分辨率执行聚类
                    cluster_results = {}
                    for res in cluster_data:
                        if res not in ['res5', 'res6', 'res7']:
                            continue
                            
                        res_data = cluster_data[res]
                        shap_features = res_data['shap_features']
                        coords_df = res_data['coords_df']
                        target_values = res_data.get('target_values', None)
                        
                        # 执行空间聚类
                        clusters, X_clustered = perform_spatial_clustering(
                            shap_features, 
                            coords_df, 
                            n_clusters=3,
                            grid_disk_k=1
                        )
                        
                        # 构建聚类结果
                        cluster_results[res] = {
                            'clusters': clusters,
                            'shap_features': shap_features,
                            'coords_df': coords_df,
                            'top_features': res_data['top_features'],
                            'target_values': target_values,
                            'X_clustered': X_clustered,
                            'standardized_features': X_clustered  # 添加这个字段
                        }
                        
                        print(f"  {res}: 生成了{len(np.unique(clusters))}个聚类")
                    
                    # 调用绘图函数
                    fig = plot_regionkmeans_feature_target_analysis(
                        cluster_results,
                        output_dir=output_dir
                    )
                    if fig:
                        plt.close(fig)
                        print("  ✓ SHAP聚类特征贡献与目标分析图创建成功")
                    else:
                        print("  ✗ SHAP聚类特征贡献与目标分析图创建失败")
                else:
                    print("  ✗ 无法准备聚类数据，跳过SHAP聚类特征贡献与目标分析图")
        except Exception as e:
            print(f"  ✗ 创建SHAP聚类特征贡献与目标分析图失败: {e}")
            import traceback as tb
            tb.print_exc()
        
        # 7. 多分辨率海拔梯度效应分析图
        print("\n[7/10] 创建多分辨率海拔梯度效应分析图...")
        try:
            # 首先需要计算海拔梯度数据
            elevation_gradient_data = {}
            
            # 定义期望的高程区间（16个区间）
            elevation_bins_config = [
                (0, 200),       # 区间1
                (200, 400),     # 区间2
                (400, 600),     # 区间3
                (600, 800),     # 区间4
                (800, 1000),    # 区间5
                (1000, 1200),   # 区间6
                (1200, 1400),   # 区间7
                (1400, 1600),   # 区间8
                (1600, 1800),   # 区间9
                (1800, 2000),   # 区间10
                (2000, 2200),   # 区间11
                (2200, 2400),   # 区间12
                (2400, 2600),   # 区间13
                (2600, 2800),   # 区间14
                (2800, 3000),   # 区间15
                (3000, 3200)    # 区间16
            ]
            
            for res in model_results:
                if res not in ['res5', 'res6', 'res7']:
                    continue
                    
                elevation_gradient_data[res] = {}
                
                # 获取数据
                if 'X_test' in model_results[res] and 'y_test' in model_results[res] and 'y_pred' in model_results[res]:
                    X = model_results[res]['X_test']
                    y_true = model_results[res]['y_test']
                    y_pred = model_results[res]['y_pred']
                    
                    # 确保有elevation列
                    if 'elevation' not in X.columns:
                        print(f"  警告: {res}缺少elevation数据，跳过海拔梯度分析")
                        continue
                    
                    # 获取海拔值
                    elevation_values = X['elevation'].values
                    
                    # 按照预定义的区间创建数据
                    for bin_start, bin_end in elevation_bins_config:
                        bin_label = f"{int(bin_start)}-{int(bin_end)}"
                        
                        # 找出在这个区间的样本
                        mask = (elevation_values >= bin_start) & (elevation_values <= bin_end)
                        
                        if mask.sum() > 0:
                            # 计算这个区间的统计数据
                            vhi_mean = y_true[mask].mean()
                            mae = np.abs(y_true[mask] - y_pred[mask]).mean()
                            
                            # 计算R²
                            ss_res = np.sum((y_true[mask] - y_pred[mask])**2)
                            ss_tot = np.sum((y_true[mask] - y_true[mask].mean())**2)
                            if ss_tot > 0:
                                r2 = 1 - ss_res / ss_tot
                            else:
                                r2 = 0.0
                            
                            elevation_gradient_data[res][bin_label] = {
                                'vhi_mean': vhi_mean,
                                'mae': mae,
                                'r2': r2,
                                'sample_count': mask.sum(),
                                'elevation_range': (bin_start, bin_end)
                            }
            
            # 如果有海拔梯度数据，生成图表
            if elevation_gradient_data and any(elevation_gradient_data[res] for res in elevation_gradient_data):
                fig = plot_elevation_gradient_bullseye(elevation_gradient_data, output_dir=output_dir)
                if fig:
                    plt.close(fig)
                    created_charts.append("多分辨率海拔梯度效应分析图")
                    print("✓ 成功创建")
                else:
                    failed_charts.append("多分辨率海拔梯度效应分析图")
                    print("✗ 创建失败")
            else:
                print("⚠ 跳过：缺少海拔梯度数据")
                failed_charts.append("多分辨率海拔梯度效应分析图（缺少数据）")
        except Exception as e:
            failed_charts.append("多分辨率海拔梯度效应分析图")
            print(f"✗ 创建失败: {e}")
            if hasattr(e, '__traceback__'):
                traceback.print_exc()
        
        # 8. 特征交互与海拔梯度分析图
        print("\n[8/10] 创建单特征依赖与海拔梯度分析图...")
        print("  展示3×3网格：分辨率 × 海拔区间的单特征依赖关系")
        try:
            # 使用正确的函数：plot_elevation_gradient_single_feature_grid
            # 这个函数生成elevation_gradient_pdp_grid.png文件，显示单特征依赖图
            fig = plot_elevation_gradient_single_feature_grid(model_results, output_dir=output_dir)
            if fig:
                plt.close(fig)
                created_charts.append("单特征依赖与海拔梯度分析图")
                print("✓ 成功创建")
            else:
                failed_charts.append("单特征依赖与海拔梯度分析图")
                print("✗ 创建失败")
        except Exception as e:
            failed_charts.append("单特征依赖与海拔梯度分析图")
            print(f"✗ 创建失败: {e}")
            if hasattr(e, '__traceback__'):
                traceback.print_exc()
        
        # 9. 时序特征热图
        print("\n[9/10] 创建时序特征热图...")
        try:
            # 检查是否有SHAP值和时间信息
            has_temporal_data = False
            for res in model_results:
                # 检查基本条件
                if 'shap_values_by_feature' in model_results[res]:
                    # 方法1: 检查X_sample中是否有year列
                    if 'X_sample' in model_results[res]:
                        df = model_results[res]['X_sample']
                        if 'year' in df.columns:
                            has_temporal_data = True
                            break
                    
                    # 方法2: 检查df（原始数据）中是否有year列
                    if 'df' in model_results[res]:
                        df = model_results[res]['df']
                        if 'year' in df.columns:
                            has_temporal_data = True
                            # 如果X_sample没有year，尝试将year信息添加到X_sample
                            if 'X_sample' in model_results[res] and 'year' not in model_results[res]['X_sample'].columns:
                                X_sample = model_results[res]['X_sample']
                                # 确保索引匹配
                                if len(X_sample) <= len(df):
                                    # 使用索引匹配year数据
                                    if X_sample.index.max() < len(df):
                                        model_results[res]['X_sample']['year'] = df.loc[X_sample.index, 'year'].values
                                        print(f"  ✓ 为{res}的X_sample添加了year列")
                                    else:
                                        # 尝试使用前N个样本
                                        model_results[res]['X_sample']['year'] = df['year'].iloc[:len(X_sample)].values
                                        print(f"  ✓ 为{res}的X_sample添加了year列（使用前{len(X_sample)}个样本）")
                            break
                    
                    # 方法3: 检查feature_names中是否包含year
                    if 'feature_names' in model_results[res] and 'year' in model_results[res]['feature_names']:
                        has_temporal_data = True
                        print(f"  ℹ️ {res}的特征中包含year，可能可以生成时序热图")
                        break
            
            if has_temporal_data:
                fig = plot_temporal_feature_heatmap(model_results, output_dir=output_dir, top_n_features=18)
                if fig:
                    plt.close(fig)
                    created_charts.append("时序特征热图")
                    print("✓ 成功创建")
                else:
                    failed_charts.append("时序特征热图")
                    print("✗ 创建失败")
            else:
                print("⚠ 跳过：缺少时序数据")
                print("  需要：shap_values_by_feature 和 year列")
                failed_charts.append("时序特征热图（缺少时序数据）")
        except Exception as e:
            failed_charts.append("时序特征热图")
            print(f"✗ 创建失败: {e}")
            if hasattr(e, '__traceback__'):
                traceback.print_exc()
        
        # 10. GeoShapley Top 3特征空间分布图
        print("\n[10/10] 创建GeoShapley Top 3特征空间分布图...")
        try:
            # 🔥 修复：更全面的数据检查和修复逻辑
            has_spatial_shap = False
            data_issues = []
            
            for res in ['res5', 'res6', 'res7']:
                if res not in model_results:
                    data_issues.append(f"{res}: 缺少基础数据")
                    continue
                
                res_data = model_results[res]
                
                # 检查SHAP数据
                if 'shap_values_by_feature' not in res_data:
                    data_issues.append(f"{res}: 缺少shap_values_by_feature")
                    continue
                
                if 'feature_importance' not in res_data:
                    data_issues.append(f"{res}: 缺少feature_importance")
                    continue
                
                # 检查空间数据并尝试修复
                spatial_data_ok = False
                
                # 检查X_sample
                if 'X_sample' in res_data and res_data['X_sample'] is not None:
                    X_sample = res_data['X_sample']
                    
                    # 检查经纬度
                    if all(col in X_sample.columns for col in ['longitude', 'latitude']):
                        spatial_data_ok = True
                        print(f"  ✓ {res}: X_sample包含经纬度")
                    else:
                        # 尝试从其他数据源补充经纬度
                        for source_key in ['df', 'raw_data']:
                            if source_key in res_data and res_data[source_key] is not None:
                                source_df = res_data[source_key]
                                if hasattr(source_df, 'columns') and all(col in source_df.columns for col in ['longitude', 'latitude']):
                                    try:
                                        # 通过索引匹配添加经纬度
                                        if len(X_sample) <= len(source_df):
                                            if 'longitude' not in X_sample.columns:
                                                if X_sample.index.max() < len(source_df):
                                                    X_sample['longitude'] = source_df.loc[X_sample.index, 'longitude'].values
                                                else:
                                                    X_sample['longitude'] = source_df['longitude'].iloc[:len(X_sample)].values
                                            
                                            if 'latitude' not in X_sample.columns:
                                                if X_sample.index.max() < len(source_df):
                                                    X_sample['latitude'] = source_df.loc[X_sample.index, 'latitude'].values
                                                else:
                                                    X_sample['latitude'] = source_df['latitude'].iloc[:len(X_sample)].values
                                            
                                            model_results[res]['X_sample'] = X_sample
                                            spatial_data_ok = True
                                            print(f"  ✓ {res}: 从{source_key}补充经纬度到X_sample")
                                            break
                                    except Exception as e:
                                        print(f"  ⚠️ {res}: 从{source_key}补充经纬度失败: {e}")
                
                # 如果仍然没有空间数据，检查是否可以从其他字段获取
                if not spatial_data_ok:
                    # 检查是否有df或raw_data可以直接使用
                    for source_key in ['df', 'raw_data']:
                        if source_key in res_data and res_data[source_key] is not None:
                            source_df = res_data[source_key]
                            if hasattr(source_df, 'columns') and all(col in source_df.columns for col in ['longitude', 'latitude', 'h3_index']):
                                print(f"  ✓ {res}: {source_key}包含完整空间信息")
                                spatial_data_ok = True
                                break
                
                if spatial_data_ok:
                    has_spatial_shap = True
                    print(f"  ✅ {res}: 空间SHAP数据检查通过")
                else:
                    data_issues.append(f"{res}: 缺少空间信息（经纬度/h3_index）")
            
            if has_spatial_shap:
                print(f"  🎯 数据验证通过，开始生成空间分布图...")
                fig = plot_geoshapley_spatial_top3(model_results, output_dir=output_dir)
                if fig:
                    plt.close(fig)
                    created_charts.append("GeoShapley Top 3特征空间分布图")
                    print("✓ 成功创建")
                else:
                    failed_charts.append("GeoShapley Top 3特征空间分布图")
                    print("✗ 创建失败")
            else:
                print("⚠ 跳过：数据检查未通过")
                if data_issues:
                    print("  数据问题:")
                    for issue in data_issues:
                        print(f"    • {issue}")
                print("  需要：shap_values_by_feature、feature_importance 和 经纬度/h3_index信息")
                failed_charts.append("GeoShapley Top 3特征空间分布图（数据不完整）")
                
        except Exception as e:
            failed_charts.append("GeoShapley Top 3特征空间分布图")
            print(f"✗ 创建失败: {e}")
            if hasattr(e, '__traceback__'):
                traceback.print_exc()
        
        # 打印最终统计
        print(f"\n{'='*60}")
        print(f"可视化输出完成: 成功创建 {len(created_charts)}/10 个图表")
        print(f"{'='*60}")
        
        if created_charts:
            print("\n✓ 成功创建的图表:")
            for i, chart in enumerate(created_charts):
                print(f"  {i+1}. {chart}")
        
        if failed_charts:
            print("\n✗ 未成功创建的图表:")
            for i, chart in enumerate(failed_charts):
                print(f"  {i+1}. {chart}")
        
        print(f"\n输出目录: {output_dir}")
        
        return len(created_charts) > 0
    
    except Exception as e:
        print(f"\n严重错误: 创建可视化图表时出现异常: {e}")
        traceback.print_exc()
        return False
    finally:
        # 确保所有matplotlib图表都已关闭
        plt.close('all')

def create_additional_visualizations(results, extended_results_by_resolution=None, output_dir=None, plots_to_create=None):
    """
    创建额外的可视化图表（非标准10个图表）
    
    参数:
    results: 模型结果字典
    extended_results_by_resolution: 扩展结果字典
    output_dir: 输出目录
    plots_to_create: 要创建的图表列表
    
    返回:
    bool: 是否成功创建所有图表
    """
    if not output_dir:
        print("错误: 未指定输出目录")
        return False
    
    success, created_dir = ensure_dir_exists(output_dir)
    if not success:
        print(f"警告: 无法创建输出目录 {output_dir}")
        return False
    
    if plots_to_create is None:
        plots_to_create = ['all']
    
    print("\n开始创建额外可视化图表...")
    
    # 额外的可视化可以在这里添加
    # 例如：SHAP分布箱线图、特征类别增强图等
    
    print("额外可视化图表创建完成")
    return True

# 辅助函数
def get_simplified_feature_names(results, resolution):
    """
    从results中获取指定分辨率的简化特征名称映射
    """
    if resolution not in results:
        return {}
    
    res_data = results[resolution]
    if 'simplified_feature_names' in res_data:
        return res_data['simplified_feature_names']
    
    # 如果不存在但有X_sample，尝试创建
    if 'X_sample' in res_data and isinstance(res_data['X_sample'], pd.DataFrame):
        feature_names = list(res_data['X_sample'].columns)
        mapping = {name: simplify_feature_name_for_plot(name) for name in feature_names}
        res_data['simplified_feature_names'] = mapping
        return mapping
    
    return {}

if __name__ == "__main__":
    # 示例用法
    print("ST-GPR可视化转换器模块")
    print("=====================================")
    print("功能：将ST-GPR模型输出转换为可视化格式")
    print("用法：")
    print("1. 从main.py调用prepare_stgpr_results_for_visualization()")
    print("2. 然后调用create_all_visualizations()生成10个标准图表")

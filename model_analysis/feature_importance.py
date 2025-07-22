#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature importance analysis module for ST-GPR models

This module contains functions for analyzing feature importance
and feature categories for ST-GPR models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import json
import torch

# 导入核心功能
from .core import ensure_dir_exists, color_map, categorize_feature, enhance_plot_style, save_plot_for_publication

def analyze_feature_importance(model_dict, X_train=None, X_test=None, y_train=None, y_test=None, feature_categories=None, result_dict=None):
    """
    分析ST-GPR模型的特征重要性
    
    参数:
    model_dict: ST-GPR模型字典，包含模型、特征名称等
    X_train: 训练特征（可选）
    X_test: 测试特征（可选）
    y_train: 训练标签（可选）
    y_test: 测试标签（可选）
    feature_categories: 特征类别字典
    result_dict: 用于存储结果的字典
    
    返回:
    feature_importance_dict: 包含feature_importance和其他信息的字典
    """
    from collections import defaultdict
    
    # 初始化结果字典
    if result_dict is None:
        result_dict = {}
    
    print("\n分析特征重要性与贡献...")
    feature_importance_dict = {}
    
    # 从模型字典中获取特征重要性
    model = model_dict.get('model')
    if model is None:
        print("错误: 模型为None，无法计算特征重要性")
        return None
    
    # 获取特征名称
    feature_names = model_dict.get('feature_names', [])
    if not feature_names and hasattr(X_train, 'columns'):
        feature_names = X_train.columns.tolist()
    
    if not feature_names:
        print("警告: 无法获取特征名称列表")
        # 尝试从模型中获取特征维度
        if hasattr(model, 'feature_dims') and model.feature_dims:
            feature_names = [f"feature_{i}" for i in model.feature_dims]
        else:
            print("错误: 无法确定特征维度和名称")
            return None
    
    # 使用模型的get_feature_importance方法获取特征重要性
    importance_list = model.get_feature_importance(feature_names)
    
    # 创建特征重要性数据框
    feature_names_list = [name for name, _ in importance_list]
    importance_values = [imp for _, imp in importance_list]
    
    # 创建特征重要性数据框
    importance_df = pd.DataFrame({'feature': feature_names_list, 'importance': importance_values})
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # 保存有序的特征重要性
    feature_importance = list(zip(importance_df['feature'], importance_df['importance']))
    feature_importance_dict['feature_importance'] = feature_importance
    
    # 同时保存排序后的版本，方便后续使用
    feature_importance_dict['sorted'] = feature_importance
    
    # 保存到结果字典
    result_dict['feature_importance'] = feature_importance
    
    # 原始植被指标列表 - 这些指标不应被分析
    excluded_indicators = ['evi', 'lai', 'fpar', 'gpp', 'EVI', 'LAI', 'FPAR', 'GPP', 'ndvi', 'vci', 'tci', 'NDVI', 'VCI', 'TCI']
    filtered_feature_names = []
    filtered_feature_indices = []
    
    for i, feature in enumerate(feature_names_list):
        # 检查特征名称是否包含任何被排除的指标
        if not any(indicator.lower() in feature.lower() for indicator in excluded_indicators):
            filtered_feature_names.append(feature)
            filtered_feature_indices.append(i)
    
    # 报告被过滤掉的特征
    if len(filtered_feature_names) < len(feature_names_list):
        filtered_out = [f for f in feature_names_list if f not in filtered_feature_names]
        print(f"Warning: {len(filtered_out)} features containing original vegetation indicators were removed from feature importance analysis:")
        print(f"   For example: {', '.join(filtered_out[:5])}" + ("..." if len(filtered_out) > 5 else ""))
    
    # 使用过滤后的特征名称和重要性值
    filtered_importance = []
    for i, idx in enumerate(filtered_feature_indices):
        if idx < len(importance_values):
            filtered_importance.append(importance_values[idx])
    
    # 确保filtered_feature_names和filtered_importance长度相同
    n = min(len(filtered_feature_names), len(filtered_importance))
    filtered_feature_names = filtered_feature_names[:n]
    filtered_importance = filtered_importance[:n]
    
    # 创建特征重要性字典
    feature_importances = {filtered_feature_names[i]: filtered_importance[i] for i in range(n)}
    
    # 按重要性排序
    sorted_feature_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))
    # 既然禁用了特征预筛选，显示所有特征（ST-GPR模型有19个特征）
    top_n = len(sorted_feature_importances)  # 显示所有特征，不再限制数量
    top_features = list(sorted_feature_importances.keys())[:top_n]
    top_importances = list(sorted_feature_importances.values())[:top_n]
    
    # 对特征进行分类
    categories = [categorize_feature(feature) for feature in top_features]
    category_colors = [color_map.get(cat, '#888888') for cat in categories]
    
    # 创建格式化特征名称的函数
    def format_feature_name(feature_name):
        """
        格式化特征名称，使其更易读
        """
        # 从core导入标准化函数
        from model_analysis.core import standardize_feature_name
        
        # 首先标准化特征名称
        feature_name = standardize_feature_name(feature_name)
        
        # 处理滞后特征
        if '_t_lag' in feature_name:
            parts = feature_name.split('_t_lag')
            if len(parts) == 2:
                return f"{parts[0]} (t-{parts[1]})"
        elif '_lag' in feature_name:
            parts = feature_name.split('_lag')
            if len(parts) == 2:
                return f"{parts[0]} (lag {parts[1]})"
        elif '_s_lag' in feature_name:
            parts = feature_name.split('_s_lag')
            if len(parts) == 2:
                return f"{parts[0]} (spatial lag {parts[1]})"
        
        # 处理空间滞后特征
        feature_name = feature_name.replace('_s_lag1', ' (spatial lag)')
        
        # 处理交互特征
        feature_name = feature_name.replace('_interaction', ' interaction')
        
        # 常见缩写扩展
        feature_name = feature_name.replace('temp', 'temperature')
        feature_name = feature_name.replace('precipitation', 'precipitation')
        feature_name = feature_name.replace('precip', 'precipitation')
        feature_name = feature_name.replace('pet', 'potential evapotranspiration')
        feature_name = feature_name.replace('elevation', 'elevation')
        feature_name = feature_name.replace('slope', 'slope')
        feature_name = feature_name.replace('aspect', 'aspect')
        feature_name = feature_name.replace('nightlight', 'nightlight')
        feature_name = feature_name.replace('population_density', 'population density')
        feature_name = feature_name.replace('popdens', 'population density')
        
        # 修改：将forest_area_percent替换为forest coverage（明确是覆盖率），而不是forest area
        feature_name = feature_name.replace('forest_area_percent', 'forest coverage (%)')
        # 确保先前可能未处理的forest_area也被显示为一致格式
        feature_name = feature_name.replace('forest_area', 'forest coverage (%)')
        
        # 统一其他土地覆盖类型的格式为"coverage (%)"
        feature_name = feature_name.replace('cropland_area_percent', 'cropland coverage (%)')
        feature_name = feature_name.replace('crop_area', 'cropland coverage (%)')
        feature_name = feature_name.replace('grassland_area_percent', 'grassland coverage (%)')
        feature_name = feature_name.replace('grass_area', 'grassland coverage (%)')
        feature_name = feature_name.replace('shrubland_area_percent', 'shrubland coverage (%)')
        feature_name = feature_name.replace('shrub_area', 'shrubland coverage (%)')
        feature_name = feature_name.replace('impervious_area_percent', 'impervious coverage (%)')
        feature_name = feature_name.replace('imperv_area', 'impervious coverage (%)')
        feature_name = feature_name.replace('bare_area_percent', 'bare coverage (%)')
        feature_name = feature_name.replace('bare_area', 'bare coverage (%)')
        
        return feature_name
    
    # 创建特征重要性条形图
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(top_features))
    
    # 创建反转的数据和位置，使最重要的特征在顶部
    top_features_rev = [format_feature_name(f) for f in top_features[::-1]]
    top_importances_rev = top_importances[::-1]
    category_colors_rev = category_colors[::-1]
    
    # 绘制水平条形图
    bars = plt.barh(y_pos, top_importances_rev, color=category_colors_rev, alpha=0.8)
    
    # 创建图例 - 使用分类进行分组
    unique_categories = list(set(categories))
    legend_elements = [Patch(facecolor=color_map.get(cat, '#888888'), edgecolor='black', label=cat) 
                       for cat in unique_categories]
    plt.legend(handles=legend_elements, loc='lower right')
    
    # 添加网格线作为视觉辅助
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 添加特征名称标签
    plt.yticks(y_pos, top_features_rev)
    plt.xlabel('Feature Importance')
    plt.title('STGPR Feature Importance (All Features)', fontsize=14, fontweight='bold')
    
    # 增强图表样式
    enhance_plot_style(plt.gca())
    
    # 确保输出目录存在
    ensure_dir_exists('output/feature_importance')
    
    # 保存图表
    importance_plot_path = os.path.join('output/feature_importance', 'feature_importance.png')
    save_plot_for_publication(importance_plot_path)
    
    # 输出分析结果
    feature_importance_results = {
        'feature_importances': sorted_feature_importances,
        'top_features': top_features,
        'top_importances': top_importances,
        'categories': categories,
        'filtered_feature_names': filtered_feature_names
    }
    
    # 保存特征重要性结果
    feature_importance_file = os.path.join('output/feature_importance', 'feature_importance.json')
    try:
        with open(feature_importance_file, 'w') as f:
            json.dump({
                'feature_importances': {k: float(v) for k, v in sorted_feature_importances.items()},
                'top_features': top_features,
                'top_importances': [float(i) for i in top_importances],
                'categories': categories
            }, f, indent=2)
        print(f"Feature importance results saved to: {feature_importance_file}")
    except Exception as e:
        print(f"Error saving feature importance results: {e}")
    
    print(f"Feature importance analysis completed, saved to: output/feature_importance")
    return feature_importance_results 

def merge_geo_features(feature_importance, feature_values=None):
    """
    将经纬度(longitude和latitude)合并为一个地理位置特征(GEO)
    
    这是GeoShapley分析中的常用做法，因为：
    1. 经纬度本质上是一个复合的地理位置特征
    2. 单独分析经纬度可能会低估地理位置的整体重要性
    3. 合并后的GEO特征更好地反映了空间位置对目标变量的影响
    
    参数:
    feature_importance: 特征重要性列表，格式为[(feature_name, importance), ...]
    feature_values: 可选，特征SHAP值字典 {feature_name: shap_values}
    
    返回:
    merged_importance: 合并后的特征重要性列表
    merged_values: 合并后的特征SHAP值字典(如果feature_values不为None)
    """
    import numpy as np
    
    geo_features = ['latitude', 'longitude']
    
    # 初始化结果
    merged_importance = []
    merged_values = {} if feature_values is not None else None
    
    # 查找经纬度特征的索引和值
    geo_indices = []
    geo_importance_values = []
    found_geo_features = []
    
    # 检查是否已经有GEO特征
    has_existing_geo = any(feat.upper() == 'GEO' for feat, _ in feature_importance)
    
    if has_existing_geo:
        print(f"    📍 检测到已存在的GEO特征")
        print(f"    • 这表明GeoShapley已经自动合并了经纬度")
        print(f"    • 将保持现有特征不变（防御性检查通过）")
        
        # 直接返回原始数据
        if feature_values is not None:
            return feature_importance, feature_values
        else:
            return feature_importance
    
    print(f"    🔍 查找独立的经纬度特征...")
    
    for i, (feature, importance) in enumerate(feature_importance):
        if feature.lower() in [g.lower() for g in geo_features]:
            geo_indices.append(i)
            geo_importance_values.append(importance)
            found_geo_features.append(feature)
            print(f"      • 找到地理特征: {feature} (重要性: {importance:.6f})")
        else:
            merged_importance.append((feature, importance))
    
    # 添加合并后的GEO特征
    if geo_indices:
        # 使用更科学的合并策略：
        # 1. 如果两个特征都存在，使用它们的平方和的平方根（欧几里得距离的概念）
        # 2. 这更好地反映了地理位置作为二维向量的本质
        if len(geo_importance_values) == 2:
            # 两个地理特征都存在，使用向量长度
            geo_combined_importance = np.sqrt(sum(imp**2 for imp in geo_importance_values))
            print(f"      • 使用向量长度合并: sqrt({geo_importance_values[0]:.6f}² + {geo_importance_values[1]:.6f}²) = {geo_combined_importance:.6f}")
        else:
            # 只有一个地理特征，直接使用其值
            geo_combined_importance = geo_importance_values[0]
            print(f"      • 只找到一个地理特征，直接使用其重要性: {geo_combined_importance:.6f}")
        
        merged_importance.append(('GEO', geo_combined_importance))
        print(f"    ✅ 成功创建GEO特征，重要性: {geo_combined_importance:.6f}")
    else:
        print(f"    ℹ️ 未找到独立的经纬度特征")
        print(f"    • 可能数据中不包含地理信息")
        print(f"    • 或地理特征使用了不同的命名")
    
    # 按重要性排序
    merged_importance.sort(key=lambda x: x[1], reverse=True)
    
    # 如果提供了特征SHAP值，也进行合并
    if feature_values is not None:
        print(f"    🔗 同步合并SHAP值...")
        
        # 复制非地理特征的值
        for feature in feature_values:
            if feature.lower() not in [g.lower() for g in geo_features]:
                merged_values[feature] = feature_values[feature]
        
        # 合并地理特征的SHAP值
        geo_shap_values = []
        geo_features_found = []
        
        for feature in geo_features:
            if feature in feature_values:
                geo_shap_values.append(np.array(feature_values[feature]))
                geo_features_found.append(feature)
        
        if geo_shap_values:
            if len(geo_shap_values) == 2:
                # 两个地理特征都存在，使用向量长度合并SHAP值
                # 对于每个样本，计算其经纬度SHAP值的向量长度
                lat_shap, lon_shap = geo_shap_values[0], geo_shap_values[1]
                geo_combined_shap = np.sqrt(lat_shap**2 + lon_shap**2)
                print(f"      • SHAP值合并完成，形状: {geo_combined_shap.shape}")
            else:
                # 只有一个地理特征
                geo_combined_shap = geo_shap_values[0]
                print(f"      • 使用单个地理特征的SHAP值")
            
            merged_values['GEO'] = geo_combined_shap
    
    # 总结
    if geo_indices:
        print(f"    📊 合并总结: {len(found_geo_features)}个地理特征 → 1个GEO特征")
    else:
        print(f"    📊 无需合并: 保持原有{len(feature_importance)}个特征")
    
    return (merged_importance, merged_values) if feature_values is not None else merged_importance

def analyze_geoshapley_importance(model_results, feature_categories=None, merge_geo=True):
    """
    分析GeoShapley值并生成特征重要性结果，与原始analyze_feature_importance兼容
    
    参数:
    model_results: ST-GPR模型训练产生的结果字典
    feature_categories: 特征类别字典
    merge_geo: 是否合并经纬度特征为GEO
    
    返回:
    feature_importance_dict: 包含feature_importance, shap_values和feature_contribution的字典
    """
    from collections import defaultdict
    import numpy as np
    
    print("\n分析GeoShapley特征重要性与贡献...")
    feature_importance_dict = {}
    
    # 检查是否使用'explanations'键保存了局部解释
    if 'explanations' in model_results and model_results['explanations'] is not None:
        local_explanations = model_results['explanations'].get('local_explanations')
        if local_explanations is not None:
            # 新版本的GeoShapley解释格式
            shap_values = local_explanations.get('shapley_values')
            feature_names = local_explanations.get('feature_names')
        else:
            # 检查是否使用global_importance保存了特征重要性
            global_importance = model_results['explanations'].get('global_importance')
            if global_importance:
                # 直接使用全局重要性
                feature_importance = global_importance
                if merge_geo:
                    feature_importance = merge_geo_features(feature_importance)
                feature_importance_dict['feature_importance'] = feature_importance
                return feature_importance_dict
            else:
                print("警告: 在model_results['explanations']中找不到局部解释")
                return None
    # 旧的方式直接检查shap_values键
    elif 'shap_values' in model_results and model_results['shap_values'] is not None:
        shap_values = model_results['shap_values']
        feature_names = model_results.get('feature_names', [])
    else:
        # 检查模型的feature_importance
        if 'feature_importance' in model_results:
            feature_importance = model_results['feature_importance']
            if merge_geo:
                feature_importance = merge_geo_features(feature_importance)
            feature_importance_dict['feature_importance'] = feature_importance
            return feature_importance_dict
        else:
            print("警告: 在model_results中找不到任何特征重要性信息")
            return None
    
    # 如果没有特征名称列表
    if not feature_names:
        print("警告: 无法获取特征名称")
        return None
    
    # 获取SHAP值，确保是二维数组
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 1 and len(feature_names) > 1:
            # 如果是一维数组但有多个特征，可能需要重塑
            shap_values = shap_values.reshape(-1, len(feature_names))
        elif shap_values.ndim > 2:
            # 如果是高维数组，尝试转换为二维
            shap_values = shap_values.reshape(-1, len(feature_names))
    
    # 确保shap_values是二维数组
    if not isinstance(shap_values, np.ndarray) or shap_values.ndim != 2:
        print(f"警告: SHAP值格式不正确")
        return None
    
    # 检查SHAP值的列数是否与特征数量匹配
    if shap_values.shape[1] != len(feature_names):
        print(f"警告: SHAP值列数 ({shap_values.shape[1]}) 不匹配特征数量 ({len(feature_names)})")
        return None
    
    # 修改特征重要性计算部分
    feature_importance = []
    
    # 将SHAP值按特征分组，计算每个特征的平均重要性
    shap_values_by_feature = {}
    for j, feature in enumerate(feature_names):
        if j < shap_values.shape[1]:
            feature_shap = shap_values[:, j]
            # 使用平均绝对SHAP值作为特征重要性
            importance = np.abs(feature_shap).mean()
            feature_importance.append((feature, importance))
            shap_values_by_feature[feature] = feature_shap
    
    # 归一化特征重要性
    if feature_importance:
        max_importance = max([importance for _, importance in feature_importance])
        if max_importance > 0:
            feature_importance = [(feature, importance / max_importance) 
                                for feature, importance in feature_importance]
    
    # 对特征重要性进行排序
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # 合并经纬度特征
    if merge_geo:
        feature_importance, shap_values_by_feature = merge_geo_features(
            feature_importance, shap_values_by_feature)
    
    # 保存特征重要性
    feature_importance_dict['feature_importance'] = feature_importance
    feature_importance_dict['shap_values_by_feature'] = shap_values_by_feature
    
    # 按类别组织特征
    if feature_categories is None:
        # 创建自动分类
        feature_categories = {}
        for feature in feature_names:
            feature_categories[feature] = categorize_feature(feature)
    
    # 按类别分组特征重要性
    category_importance = defaultdict(list)
    for feature, importance in feature_importance:
        category = feature_categories.get(feature, 'Spatial')
        category_importance[category].append((feature, importance))
    
    # 计算每个类别的平均重要性
    category_avg_importance = {}
    for category, features in category_importance.items():
        if features:
            avg_importance = sum(imp for _, imp in features) / len(features)
            category_avg_importance[category] = avg_importance
    
    # 将类别重要性按从高到低排序
    sorted_categories = sorted(category_avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    # 保存类别重要性
    feature_importance_dict['category_importance'] = sorted_categories
    
    # 保存按类别分组的特征重要性
    feature_importance_dict['features_by_category'] = dict(category_importance)
    
    # 计算特征贡献
    feature_contribution = defaultdict(list)
    
    # 对每个样本
    for i in range(shap_values.shape[0]):
        # 计算该样本的总SHAP值（所有特征SHAP值的总和）
        total_shap = np.sum(shap_values[i, :])
        
        # 对每个特征，计算其相对贡献（该特征的SHAP值除以总SHAP值的绝对值）
        for j, feature in enumerate(feature_names):
            if np.abs(total_shap) > 1e-10:  # 避免除以接近零的值
                contribution = shap_values[i, j] / np.abs(total_shap)
            else:
                contribution = 0
            
            feature_contribution[feature].append(contribution)
    
    # 计算每个特征的平均贡献
    avg_contribution = {}
    for feature, contributions in feature_contribution.items():
        avg_contribution[feature] = np.mean(contributions)
    
    # 保存特征贡献
    feature_importance_dict['feature_contribution'] = avg_contribution
    
    # 打印前10个重要特征的信息
    print("\nGeoShapley特征重要性排名前10:")
    for i, (feature, importance) in enumerate(feature_importance[:10]):
        print(f"{i+1}. {feature}: {importance:.6f}")
    
    return feature_importance_dict 

def compute_feature_importance(model_results, method='model'):
    """
    计算特征重要性，支持不同的方法
    
    参数:
    model_results: 模型训练结果字典
    method: 使用的特征重要性计算方法，可以是'model'(使用模型的get_feature_importance方法)或'geoshapley'
    
    返回:
    dict: 特征重要性字典，格式为{feature_name: importance_score}
    """
    if model_results is None:
        print("警告: 模型结果为None，无法计算特征重要性")
        return {}
    
    # 首先检查model_results中的feature_importance是否已经存在
    if 'feature_importance' in model_results and model_results['feature_importance']:
        # 检查类型是否为列表，如果是，则转换为字典
        if isinstance(model_results['feature_importance'], list):
            # 转换为字典 [(name, value)] -> {name: value}
            return {name: value for name, value in model_results['feature_importance']}
        # 如果已经是字典，直接返回
        if isinstance(model_results['feature_importance'], dict):
            return model_results['feature_importance']
    
    # 如果是ST-GPR模型，尝试使用模型的get_feature_importance方法
    if method == 'model':
        model = model_results.get('model')
        feature_names = model_results.get('feature_names')
        
        if model is not None and hasattr(model, 'get_feature_importance') and feature_names:
            # 使用模型的特征重要性计算方法
            importance_list = model.get_feature_importance(feature_names)
            return {name: importance for name, importance in importance_list}
    
    # 如果是GeoShapley方法
    elif method == 'geoshapley':
        # 检查是否有GeoShapley解释结果
        if 'explanations' in model_results and model_results['explanations']:
            # 提取全局特征重要性
            global_importance = model_results['explanations'].get('global_importance')
            if global_importance:
                return {name: importance for name, importance in global_importance}
            
            # 如果没有全局重要性，尝试从局部解释中计算
            local_explanations = model_results['explanations'].get('local_explanations')
            if local_explanations and 'shapley_values' in local_explanations:
                shapley_values = local_explanations['shapley_values']
                feature_names = local_explanations.get('feature_names', [])
                
                if len(feature_names) > 0 and isinstance(shapley_values, np.ndarray):
                    # 计算每个特征的平均绝对SHAP值
                    mean_abs_shap = np.mean(np.abs(shapley_values), axis=0)
                    return {feature_names[i]: float(mean_abs_shap[i]) for i in range(len(feature_names))}
    
    # 如果上述方法都失败了，尝试直接使用核函数参数中的特征权重
    if 'best_params' in model_results:
        best_params = model_results['best_params']
        feature_weights = None
        feature_names = model_results.get('feature_names', [])
        
        # 检查不同可能的参数键
        for key in ['feature_weights', 'feature_lengthscales', 'p_function']:
            if key in best_params and best_params[key] is not None:
                feature_weights = best_params[key]
                break
        
        if feature_weights is not None and len(feature_names) == len(feature_weights):
            return {feature_names[i]: float(feature_weights[i]) for i in range(len(feature_weights))}
    
    # 如果上述方法都失败，则返回空字典
    print("警告: 无法计算特征重要性，返回空结果")
    return {}


def get_feature_classification(features):
    """
    将特征列表按照类别进行分类
    
    参数:
    features: 特征名称列表
    
    返回:
    dict: 分类后的特征字典，格式为{category: [feature1, feature2, ...]}
    """
    if not features:
        return {}
    
    # 初始化分类字典
    classification = {
        'Climate': [],
        'Human Activity': [],
        'Terrain': [],
        'Land Cover': [],
        'Time': [],
        'Spatial': []
    }
    
    # 对每个特征进行分类
    for feature in features:
        # 时间特征
        if feature.lower() == 'year' or 't_lag' in feature.lower():
            classification['Time'].append(feature)
        # 空间特征
        elif feature.lower() in ['latitude', 'longitude', 'h3_index'] or 's_lag' in feature.lower():
            classification['Spatial'].append(feature)
        # 其他特征使用categorize_feature函数分类
        else:
            category = categorize_feature(feature)
            classification[category].append(feature)
    
    # 移除空类别
    return {k: v for k, v in classification.items() if v} 
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核心特征重要性可视化模块

包含基础的特征重要性绘制功能和工具函数。

主要功能：
- plot_feature_importance: 基础特征重要性条形图
- get_unified_feature_order: 获取统一特征顺序
- categorize_feature_for_geoshapley_display: GeoShapley特征分类
- visualize_feature_importance: 特征重要性可视化
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
import warnings
from typing import Dict, Optional, Tuple, List, Union

# 导入自定义的核心功能模块
from model_analysis.core import (
    ensure_dir_exists,
    standardize_feature_name,
    categorize_feature
)

# 导入可视化工具函数
try:
    from visualization.utils import (
        enhance_feature_display_name,
        simplify_feature_name_for_plot,
        clean_feature_name_for_plot,
        ensure_spatiotemporal_features,
        categorize_feature_with_interactions
    )
    from visualization.base import color_map, enhance_plot_style
except ImportError as e:
    warnings.warn(f"导入visualization模块失败: {e}")
    # 创建简化版本的函数
    def enhance_feature_display_name(feature, res_obj=None):
        return feature.replace('_', ' ').title()
    
    def simplify_feature_name_for_plot(feature):
        return feature.replace('_', ' ').title()
        
    def clean_feature_name_for_plot(feature):
        return feature.replace('_', ' ').title()
    
    def ensure_spatiotemporal_features(feature_list, all_features):
        return feature_list
    
    # 简化的color_map
    color_map = {
        'Climate': '#3498db',
        'Human Activity': '#e74c3c', 
        'Terrain': '#f39c12',
        'Land Cover': '#27ae60',
        'Spatial': '#1abc9c',
        'Temporal': '#9b59b6',
        'Geographic': '#16a085',
        'Interaction': '#95a5a6',
        'Other': '#34495e'
    }
    
    def enhance_plot_style(ax, xlabel=None, ylabel=None):
        pass


def plot_feature_importance(importance_df: Union[pd.DataFrame, List, Dict], 
                          category_map: Optional[Dict] = None, 
                          top_n: Optional[int] = None, 
                          output_dir: Optional[str] = None, 
                          resolution: Optional[str] = None, 
                          save_plot: bool = True) -> plt.Figure:
    """
    绘制特征重要性条形图
    
    参数:
    importance_df: 特征重要性数据，可以是DataFrame或(特征,重要性)元组列表
    category_map: 自定义特征类别映射，如{特征名:类别}
    top_n: 显示的特征数量
    output_dir: 输出目录
    resolution: 分辨率标签，如res5/res6/res7
    save_plot: 是否保存图表，默认为True
    
    返回:
    matplotlib.figure.Figure: 图表对象
    """
    # 标准化输入数据
    if isinstance(importance_df, pd.DataFrame):
        if 'feature' in importance_df.columns and 'importance' in importance_df.columns:
            features = importance_df['feature'].tolist()
            importances = importance_df['importance'].tolist()
        else:
            features = importance_df.iloc[:, 0].tolist()
            importances = importance_df.iloc[:, 1].tolist()
        importance_tuples = list(zip(features, importances))
    elif isinstance(importance_df, list) and all(isinstance(item, tuple) and len(item) == 2 for item in importance_df):
        importance_tuples = importance_df
    elif isinstance(importance_df, dict):
        importance_tuples = [(k, v) for k, v in importance_df.items()]
    else:
        raise ValueError("importance_df必须是DataFrame、(特征,重要性)元组列表或特征:重要性字典")
    
    # 按重要性降序排列
    importance_tuples.sort(key=lambda x: x[1], reverse=True)
    
    # 限制特征数量
    if top_n is not None:
        importance_tuples = importance_tuples[:top_n]
    
    # 提取特征名称和重要性值
    features = [item[0] for item in importance_tuples]
    importances = [item[1] for item in importance_tuples]
    
    # 清理特征名称
    clean_features = [clean_feature_name_for_plot(f) for f in features]
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 分类特征并为不同类别用不同颜色
    categories = []
    for feature in features:
        if category_map and feature in category_map:
            categories.append(category_map[feature])
        else:
            categories.append(categorize_feature(feature))
    
    # 获取条形颜色
    colors = [color_map.get(cat, '#2c3e50') for cat in categories]
    
    # 反转列表（使最重要的特征显示在顶部）
    clean_features.reverse()
    importances.reverse()
    colors.reverse()
    categories.reverse()
    
    # 创建y轴位置
    y_pos = np.arange(len(clean_features))
    
    # 绘制水平条形图
    bars = plt.barh(y_pos, importances, color=colors, edgecolor='gray', alpha=0.8)
    
    # 定义标题和轴标签
    if resolution:
        plt.title(f'特征重要性 - {resolution.upper()}', fontsize=14, fontweight='bold')
    else:
        plt.title('特征重要性', fontsize=14, fontweight='bold')
    
    plt.xlabel('重要性', fontsize=12, fontweight='bold')
    plt.ylabel('特征', fontsize=12, fontweight='bold')
    
    # 设置y轴刻度
    plt.yticks(y_pos, clean_features)
    
    # 调整布局
    plt.tight_layout()
    
    # 添加网格线
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # 为每个条形图添加值标签
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{importances[i]:.3f}",
            va='center'
        )
    
    # 创建图例
    unique_categories = list(set(categories))
    legend_patches = []
    for cat in sorted(unique_categories):
        if cat in color_map:
            patch = mpatches.Patch(color=color_map[cat], label=cat)
            legend_patches.append(patch)
    
    if legend_patches:
        plt.legend(handles=legend_patches, loc='lower right', 
                  title='Feature Categories', frameon=True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if output_dir and save_plot:
        ensure_dir_exists(output_dir)
        if resolution:
            fig_path = os.path.join(output_dir, f"{resolution}_Fig4-8_feature_importance.png")
        else:
            fig_path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"已保存特征重要性图: {fig_path}")
    
    # 获取当前图表对象
    fig = plt.gcf()
    
    # 关闭图表，避免在Jupyter中显示
    plt.close()
    
    return fig


def get_unified_feature_order(feature_importances_dict: Dict, top_n: Optional[int] = None) -> List[str]:
    """
    获取统一的特征顺序，基于所有分辨率的平均重要性
    
    参数:
    feature_importances_dict: 各分辨率的特征重要性字典
    top_n: 要显示的特征数量
    
    返回:
    unified_features: 统一的特征列表（按平均重要性排序）
    """
    # 收集所有特征及其在各分辨率的重要性
    all_features = {}
    
    for res, importance_list in feature_importances_dict.items():
        # 确保是列表格式
        if isinstance(importance_list, dict):
            importance_list = [(k, v) for k, v in importance_list.items()]
        
        # 标准化特征名称并记录重要性
        for feat, imp in importance_list:
            std_feat = standardize_feature_name(feat).lower()
            if std_feat not in all_features:
                all_features[std_feat] = {
                    'display_name': standardize_feature_name(feat),
                    'importances': {},
                    'count': 0,
                    'total': 0
                }
            all_features[std_feat]['importances'][res] = imp
            all_features[std_feat]['count'] += 1
            all_features[std_feat]['total'] += imp
    
    # 计算每个特征的平均重要性
    for feat_data in all_features.values():
        feat_data['average'] = feat_data['total'] / len(feature_importances_dict)
    
    # 按平均重要性排序
    sorted_features = sorted(
        all_features.items(), 
        key=lambda x: x[1]['average'], 
        reverse=True
    )
    
    # 返回特征列表
    if top_n is not None:
        return [feat_data['display_name'] for _, feat_data in sorted_features[:top_n]]
    else:
        return [feat_data['display_name'] for _, feat_data in sorted_features]


def categorize_feature_for_geoshapley_display(feature_name: str) -> str:
    """
    针对GeoShapley特征结构的详细分类函数，确保6个类别准确分类
    
    将特征分为六大类别：
    1. Climate: temperature, precipitation + 其交互项
    2. Human Activity: nightlight, road_density, mining_density, population_density + 其交互项  
    3. Terrain: elevation, slope + 其交互项
    4. Land Cover: forest_area_percent, cropland_area_percent, impervious_area_percent + 其交互项
    5. Geographic: GEO, latitude, longitude
    6. Temporal: year + 其交互项
    
    参数:
    feature_name: 特征名称
    
    返回:
    特征类别
    """
    if not isinstance(feature_name, str):
        feature_name = str(feature_name)
    
    feature_lower = feature_name.lower().strip()
    
    # 🔧 优化：更精确的特征名称匹配，包括常见缩写
    
    # Geographic特征 - 地理位置
    if (feature_lower in ['geo', 'latitude', 'longitude', 'lat', 'lon', 'location'] or
        feature_lower.startswith('geo')):
        return 'Geographic'
    
    # 交互效应特征 - 根据主效应分类（更精确的分割）
    interaction_markers = ['×', ' x ', '* ', ' * ', '_x_']
    is_interaction = any(marker in feature_name for marker in interaction_markers)
    
    if is_interaction:
        # 提取主效应特征名（第一个特征）
        main_feature = feature_name
        for marker in interaction_markers:
            if marker in feature_name:
                main_feature = feature_name.split(marker)[0].strip()
                break
        
        # 递归调用获取主效应的类别
        main_category = categorize_feature_for_geoshapley_display(main_feature)
        return main_category
    
    # 🔧 主效应特征分类 - 更精确的关键词匹配
    
    # Climate特征（气候）
    climate_keywords = ['temperature', 'temp', 'precipitation', 'prec', 'climate', 'weather']
    # 🔧 添加气候缩写支持
    climate_abbreviations = ['temp', 'prec']
    if (any(keyword in feature_lower for keyword in climate_keywords) or
        feature_lower in climate_abbreviations):
        return 'Climate'
    
    # Human Activity特征（人类活动）
    human_keywords = ['nightlight', 'night', 'nigh', 'road', 'rd', 'mining', 'md', 
                      'population', 'pop', 'pd', 'urban', 'development', 'anthropogenic']
    # 🔧 添加常见缩写支持
    human_abbreviations = ['nigh', 'rd', 'md', 'pd']
    if (any(keyword in feature_lower for keyword in human_keywords) or
        feature_lower in human_abbreviations):
        return 'Human Activity'
    
    # Terrain特征（地形）
    terrain_keywords = ['elevation', 'elev', 'slope', 'slop', 'dem', 'altitude', 'topography']
    # 🔧 添加地形缩写支持
    terrain_abbreviations = ['elev', 'slop']
    if (any(keyword in feature_lower for keyword in terrain_keywords) or
        feature_lower in terrain_abbreviations):
        return 'Terrain'
    
    # Land Cover特征（土地覆盖）
    landcover_keywords = ['forest', 'fap', 'cropland', 'cap', 'impervious', 'iap', 
                          'area_percent', 'vegetation', 'land_cover', 'land', 'cover']
    # 🔧 添加土地覆盖缩写支持
    landcover_abbreviations = ['fap', 'cap', 'iap']
    if (any(keyword in feature_lower for keyword in landcover_keywords) or
        feature_lower in landcover_abbreviations or
        'area_percent' in feature_lower):
        return 'Land Cover'
    
    # Temporal特征（时间）
    temporal_keywords = ['year', 'time', 'temporal', 'date', 'season', 'month']
    # 🔧 添加时间缩写支持
    temporal_abbreviations = ['year', 'yr']
    if (any(keyword in feature_lower for keyword in temporal_keywords) or
        feature_lower in temporal_abbreviations):
        return 'Temporal'
    
    # 🔧 改进：更智能的默认分类逻辑
    # 根据特征名称模式判断
    if any(char.isdigit() for char in feature_lower):
        return 'Temporal'  # 包含数字的可能是时间特征
    elif 'density' in feature_lower:
        return 'Human Activity'  # 密度类特征通常是人类活动
    elif 'percent' in feature_lower:
        return 'Land Cover'  # 百分比特征通常是土地覆盖
    elif len(feature_lower) <= 4:
        return 'Geographic'  # 短名称可能是地理编码
    else:
        return 'Other'  # 最后的兜底分类


def visualize_feature_importance(feature_importances: Dict, output_dir: str) -> None:
    """
    为每个分辨率创建特征重要性可视化
    
    参数:
    feature_importances: 各分辨率的特征重要性字典
    output_dir: 输出目录
    """
    if not feature_importances:
        print("⚠️ 没有特征重要性数据可供可视化")
        return
    
    print("\n📊 创建特征重要性可视化...")
    
    # 确保输出目录存在
    ensure_dir_exists(output_dir)
    
    # 为每个分辨率创建图表
    for resolution, importance_data in feature_importances.items():
        if not importance_data:
            print(f"⚠️ {resolution}: 没有特征重要性数据")
            continue
        
        try:
            # 创建特征重要性图
            fig = plot_feature_importance(
                importance_data,
                output_dir=output_dir,
                resolution=resolution,
                top_n=15  # 显示前15个最重要的特征
            )
            
            print(f"✅ {resolution}: 特征重要性图表已创建")
            
        except Exception as e:
            print(f"❌ {resolution}: 创建特征重要性图表失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("📊 特征重要性可视化完成")


# 导入merge_geo_features函数
try:
    from model_analysis.feature_importance import merge_geo_features
except ImportError:
    warnings.warn("无法导入merge_geo_features函数，将创建本地版本")
    def merge_geo_features(feature_importance):
        """合并经纬度特征为GEO特征"""
        if isinstance(feature_importance, dict):
            # 字典格式
            merged = {}
            lat_imp = feature_importance.get('latitude', 0)
            lon_imp = feature_importance.get('longitude', 0)
            
            for feat, imp in feature_importance.items():
                if feat.lower() not in ['latitude', 'longitude']:
                    merged[feat] = imp
            
            if lat_imp > 0 or lon_imp > 0:
                merged['GEO'] = (lat_imp + lon_imp) / 2
            
            return list(merged.items())
        else:
            # 列表格式
            merged = []
            lat_imp = 0
            lon_imp = 0
            
            for feat, imp in feature_importance:
                if feat.lower() == 'latitude':
                    lat_imp = imp
                elif feat.lower() == 'longitude':
                    lon_imp = imp
                else:
                    merged.append((feat, imp))
            
            if lat_imp > 0 or lon_imp > 0:
                merged.append(('GEO', (lat_imp + lon_imp) / 2))
            
            return merged 
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities for visualization

Contains helper functions for feature categorization, naming, etc.
"""

import re
import os
import matplotlib.pyplot as plt
import numpy as np

# 从base导入color_map
from .base import color_map
# 导入核心功能中的标准化函数和其他公共函数
from model_analysis.core import (
    standardize_feature_name, 
    ensure_dir_exists,
    enhance_plot_style,
    save_plot_for_publication,
    categorize_feature,
    categorize_feature_safe  # 添加安全版本的分类函数
)

def standardize_feature_name(feature_name):
    """
    标准化特征名称，确保土地覆盖特征使用统一命名规范（_area_percent后缀）
    
    参数:
    feature_name (str): 原始特征名称
    
    返回:
    str: 标准化后的特征名称
    """
    if not isinstance(feature_name, str):
        return feature_name
    
    feature_lower = feature_name.lower()
    
    # 如果已经包含正确的后缀，直接返回
    if any(feature_lower.endswith(suffix) for suffix in [
        '_area_percent', '_percent_percent', '_percent_percent_percent'
    ]):
        # 如果有重复的_percent，需要修复
        if '_percent_percent' in feature_lower:
            # 移除多余的_percent
            while '_percent_percent' in feature_name:
                feature_name = feature_name.replace('_percent_percent', '_percent')
            return feature_name
        # 否则已经是正确的格式
        return feature_name
    
    # 标准化土地覆盖特征名称 - 使用完全匹配而不是子字符串匹配
    standardization_map = {
        # 森林特征标准化
        'forest_area': 'forest_area_percent',
        'forest_percent': 'forest_area_percent',
        'forest_pct': 'forest_area_percent',
        'forest_coverage': 'forest_area_percent',
        
        # 农田特征标准化
        'crop_area': 'cropland_area_percent',
        'cropland_area': 'cropland_area_percent',
        'crop_percent': 'cropland_area_percent',
        'cropland_percent': 'cropland_area_percent',
        'crop_pct': 'cropland_area_percent',
        'cropland_pct': 'cropland_area_percent',
        'crop_coverage': 'cropland_area_percent',
        
        # 草地特征标准化
        'grass_area': 'grassland_area_percent',
        'grassland_area': 'grassland_area_percent',
        'grass_percent': 'grassland_area_percent',
        'grassland_percent': 'grassland_area_percent',
        'grass_pct': 'grassland_area_percent',
        'grassland_pct': 'grassland_area_percent',
        'grass_coverage': 'grassland_area_percent',
        
        # 灌木特征标准化
        'shrub_area': 'shrubland_area_percent',
        'shrubland_area': 'shrubland_area_percent',
        'shrub_percent': 'shrubland_area_percent',
        'shrubland_percent': 'shrubland_area_percent',
        'shrub_pct': 'shrubland_area_percent',
        'shrubland_pct': 'shrubland_area_percent',
        'shrub_coverage': 'shrubland_area_percent',
        
        # 不透水面特征标准化
        'imperv_area': 'impervious_area_percent',
        'impervious_area': 'impervious_area_percent',
        'imperv_percent': 'impervious_area_percent',
        'impervious_percent': 'impervious_area_percent',
        'imperv_pct': 'impervious_area_percent',
        'impervious_pct': 'impervious_area_percent',
        'imperv_coverage': 'impervious_area_percent',
        
        # 裸地特征标准化
        'bare_area': 'bareland_area_percent',
        'bareland_area': 'bareland_area_percent',
        'bare_percent': 'bareland_area_percent',
        'bareland_percent': 'bareland_area_percent',
        'bare_pct': 'bareland_area_percent',
        'bareland_pct': 'bareland_area_percent',
        'bare_coverage': 'bareland_area_percent'
    }
    
    # 使用完全匹配检查
    if feature_lower in standardization_map:
        return standardization_map[feature_lower]
    
    return feature_name

def simplify_feature_name_for_plot(feature_name, max_length=4):
    """
    简化特征名称用于绘图显示
    
    GeoShapley三部分分解的统一简写规范：
    1. 主效应特征(12个)：环境特征，排除经纬度
    2. GEO特征(1个)：经纬度合并特征
    3. 交互效应特征：主效应 × GEO
    """
    if not isinstance(feature_name, str):
        feature_name = str(feature_name)
    
    feature_lower = feature_name.lower().strip()
    
    # 🔥 正确的主效应特征映射（12个环境特征，排除经纬度）
    primary_effects_mapping = {
        # === 气候特征(2个) ===
        'temperature': 'TEMP',
        'precipitation': 'PREC',
        
        # === 人类活动特征(4个) ===
        'nightlight': 'NIGH', 
        'road_density': 'RD', 'road_dens': 'RD',
        'mining_density': 'MD', 'mining_dens': 'MD', 
        'population_density': 'PD', 'pop_density': 'PD',
        
        # === 地形特征(2个) ===
        'elevation': 'ELEV',
        'slope': 'SLOP',
        
        # === 土地覆盖特征(3个) ===
        'forest_area_percent': 'FAP', 'forest_area': 'FAP',
        'cropland_area_percent': 'CAP', 'cropland_area': 'CAP', 
        'impervious_area_percent': 'IAP', 'impervious_area': 'IAP',
        
        # === 时间特征(1个) ===
        'year': 'YEAR',
        
        # === 其他特征(移除但保留兼容性) ===
        'pet': 'PET',
        'aspect': 'ASPE', 
        'grassland_area_percent': 'GAP', 'grassland_area': 'GAP',
        'shrubland_area_percent': 'SAP', 'shrubland_area': 'SAP', 
        'bareland_area_percent': 'BAP', 'bareland_area': 'BAP'
    }
    
    # 🔥 GEO特征：经纬度合并特征
    if feature_lower == 'geo':
        return 'GEO'
    
    # 🔇 静默处理：经纬度特征合并为GEO（实际已由GeoShapley正确处理）
    if feature_lower in ['latitude', 'longitude', 'lat', 'lon']:
        # 移除冗余警告，GeoShapley已正确合并这些特征
        return 'LAT' if 'lat' in feature_lower else 'LON'
    
    # 🔥 交互效应特征：主效应 × GEO
    if '×' in feature_name or 'x ' in feature_lower or ' x ' in feature_lower:
        # 处理交互效应：提取主特征名
        for separator in ['×', ' x ', 'x ', ' × ']:
            if separator in feature_name:
                parts = feature_name.split(separator)
                main_feature = parts[0].strip()
                main_simplified = simplify_feature_name_for_plot(main_feature, max_length)
                return f"{main_simplified} × GEO"
    
    # 🔧 修复：支持输入已经是缩写的情况
    # 创建反向映射字典（缩写 -> 缩写）
    abbreviation_to_abbreviation = {
        # 缩写 -> 标准缩写
        'pd': 'PD', 'iap': 'IAP', 'cap': 'CAP', 'fap': 'FAP',
        'md': 'MD', 'rd': 'RD', 'nigh': 'NIGH', 'temp': 'TEMP',
        'prec': 'PREC', 'elev': 'ELEV', 'slop': 'SLOP', 'geo': 'GEO',
        'year': 'YEAR'
    }
    
    # 首先检查是否输入本身就是缩写
    if feature_lower in abbreviation_to_abbreviation:
        return abbreviation_to_abbreviation[feature_lower]
    
    # 处理主效应特征（完整名称）
    if feature_lower in primary_effects_mapping:
        return primary_effects_mapping[feature_lower]
    
    # 处理部分匹配
    for full_name, short_name in primary_effects_mapping.items():
        if full_name in feature_lower or feature_lower in full_name:
            return short_name
    
    # 如果都不匹配，返回截断的大写形式（不再显示警告，因为可能是有效的缩写）
    result = feature_name.upper()[:max_length]
    return result

def clean_feature_name(feature_name):
    """
    清理特征名称，移除各种前缀后缀，适合在图表标题或轴标签中使用
    
    Args:
        feature_name: 原始特征名称
        
    Returns:
        str: 清理后的特征名称
    """
    # 转换为小写进行模式匹配
    feature_lower = feature_name.lower()
    
    # 🔧 特殊处理GEO特征
    if feature_lower == 'geo':
        return 'GEO'
    
    # 原样返回其他特征
    else:
        # 首字母大写，下划线替换为空格
        return ' '.join(word.capitalize() for word in feature_name.split('_'))

def format_pdp_feature_name(feature_name):
    """
    为PDP图格式化特征名称
    直接使用simplify_feature_name_for_plot保持一致性
    """
    return simplify_feature_name_for_plot(feature_name)

def enhance_feature_display_name(feature, res_obj=None):
    """
    增强特征显示名称的区分度，仅用于图表显示
    直接使用simplify_feature_name_for_plot函数获取大写缩写
    
    参数:
    feature (str): 原始特征名称
    res_obj (dict, optional): 如果提供，尝试从中获取simplified_feature_names
    
    返回:
    str: 增强的特征显示名称（大写缩写形式）
    """
    # 如果提供了res_obj，首先检查是否有简化的特征名称
    if res_obj is not None:
        # 优先使用simplified_feature_names
        if 'simplified_feature_names' in res_obj:
            simplified_names = res_obj['simplified_feature_names']
            if feature in simplified_names:
                return simplified_names[feature]
    
    # 直接使用当前模块的simplify_feature_name_for_plot函数
    # 这个函数包含了完整的特征名称到大写缩写的映射
    return simplify_feature_name_for_plot(feature, max_length=4)

def clean_feature_name_for_plot(feature_name):
    """
    清理和缩短特征名称以便于显示，同时保持特征间的区分度
    
    参数:
    feature_name (str): 原始特征名称
    
    返回:
    str: 处理后的特征名称
    """
    # 保持向后兼容，调用新的clean_feature_name函数
    return clean_feature_name(feature_name)

# 添加新的辅助函数，用于获取DataFrame或模型结果中的属性
def get_feature_categories(X_or_results):
    """
    从DataFrame或模型结果对象中获取特征类别信息
    
    参数:
    X_or_results: DataFrame或模型结果字典
    
    返回:
    dict: 特征类别信息
    """
    # 如果是模型结果字典
    if isinstance(X_or_results, dict) and 'feature_categories' in X_or_results:
        return X_or_results['feature_categories']
    
    # 如果是DataFrame
    if hasattr(X_or_results, 'attrs') and 'feature_categories' in X_or_results.attrs:
        return X_or_results.attrs['feature_categories']
    
    # 兼容旧版本的直接属性访问
    if hasattr(X_or_results, 'feature_categories'):
        return X_or_results.feature_categories
    
    # 如果都不存在，返回空字典
    return {}

def get_feature_categories_grouped(X_or_results):
    """
    从DataFrame或模型结果对象中获取分组的特征类别信息
    
    参数:
    X_or_results: DataFrame或模型结果字典
    
    返回:
    dict: 分组的特征类别信息
    """
    # 如果是字典
    if isinstance(X_or_results, dict):
        if 'feature_categories_grouped' in X_or_results:
            return X_or_results['feature_categories_grouped']
        # 尝试从feature_categories构建分组
        elif 'feature_categories' in X_or_results:
            feature_categories = X_or_results['feature_categories']
            grouped = {}
            for feat, category in feature_categories.items():
                if category not in grouped:
                    grouped[category] = []
                grouped[category].append(feat)
            return grouped
    
    # 如果是DataFrame
    if hasattr(X_or_results, 'attrs') and 'feature_categories_grouped' in X_or_results.attrs:
        return X_or_results.attrs['feature_categories_grouped']
    
    # 兼容旧版本的直接属性访问
    if hasattr(X_or_results, 'feature_categories_grouped'):
        return X_or_results.feature_categories_grouped
    
    # 如果使用attrs存储但没有分组信息，尝试从feature_categories构建
    if hasattr(X_or_results, 'attrs') and 'feature_categories' in X_or_results.attrs:
        feature_categories = X_or_results.attrs['feature_categories']
        grouped = {}
        for feat, category in feature_categories.items():
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(feat)
        return grouped
    
    # 如果都不存在，返回空字典
    return {}

def get_feature_names(X_or_results):
    """
    从DataFrame或模型结果对象中获取特征名称列表
    
    参数:
    X_or_results: DataFrame或模型结果字典
    
    返回:
    list: 特征名称列表
    """
    # 如果是字典
    if isinstance(X_or_results, dict):
        if 'feature_names' in X_or_results:
            return X_or_results['feature_names']
        elif 'base_features' in X_or_results:
            return X_or_results['base_features']
    
    # 如果是DataFrame
    if hasattr(X_or_results, 'attrs') and 'feature_names' in X_or_results.attrs:
        return X_or_results.attrs['feature_names']
    
    # 兼容旧版本的直接属性访问
    if hasattr(X_or_results, 'feature_names'):
        return X_or_results.feature_names
    
    # 如果是DataFrame，尝试使用列名
    if hasattr(X_or_results, 'columns'):
        return list(X_or_results.columns)
    
    # 如果都不存在，返回空列表
    return []

def get_feature_abbreviations(X_or_results):
    """
    从DataFrame或模型结果对象中获取特征简写映射
    
    参数:
    X_or_results: DataFrame或模型结果字典
    
    返回:
    dict: 特征简写映射
    """
    # 如果是字典
    if isinstance(X_or_results, dict) and 'feature_abbreviations' in X_or_results:
        return X_or_results['feature_abbreviations']
    
    # 如果是DataFrame
    if hasattr(X_or_results, 'attrs') and 'feature_abbreviations' in X_or_results.attrs:
        return X_or_results.attrs['feature_abbreviations']
    
    # 兼容旧版本的直接属性访问
    if hasattr(X_or_results, 'feature_abbreviations'):
        return X_or_results.feature_abbreviations
    
    # 如果都不存在，使用简化函数生成
    feature_names = get_feature_names(X_or_results)
    return {feat: simplify_feature_name_for_plot(feat) for feat in feature_names}

def categorize_feature(feature_name):
    """
    对特征进行分类，优化后支持14个特征（从19个减少）
    
    优化后的14个核心特征：
    - 空间信息(2个): latitude, longitude  
    - 气候特征(2个): temperature, precipitation  
    - 人类活动(4个): nightlight, road_density, mining_density, population_density
    - 地形特征(2个): elevation, slope
    - 土地覆盖(3个): forest_area_percent, cropland_area_percent, impervious_area_percent
    - 时间信息(1个): year
    
    移除的5个特征：pet, aspect, grassland_area_percent, shrubland_area_percent, bareland_area_percent
    
    参数:
    feature_name: 特征名称
    
    返回:
    str: 特征类别
    """
    # 标准化特征名称以进行比较
    feat_lower = feature_name.lower() if isinstance(feature_name, str) else str(feature_name).lower()
    feat_standard = standardize_feature_name(feature_name).lower()
    
    # 🔥 优化后的14个核心特征分类
    
    # 气候特征（2个）- 从3个减少，移除了pet
    if feat_lower in ['temperature', 'precipitation'] or feat_standard in ['temperature', 'precipitation']:
        return 'Climate'
    
    # 人类活动特征（4个）- 保持不变
    if (feat_lower in ['nightlight', 'road_density', 'mining_density', 'population_density'] or 
        feat_standard in ['nightlight', 'road_density', 'mining_density', 'population_density']):
        return 'Human Activity'
    
    # 地形特征（2个）- 从3个减少，移除了aspect
    if feat_lower in ['elevation', 'slope'] or feat_standard in ['elevation', 'slope']:
        return 'Terrain'
    
    # 土地覆盖特征（3个）- 从6个减少，移除了grassland/shrubland/bareland
    if (feat_lower in ['forest_area_percent', 'cropland_area_percent', 'impervious_area_percent'] or 
        feat_standard in ['forest_area_percent', 'cropland_area_percent', 'impervious_area_percent'] or
        'forest_area' in feat_lower or 'cropland_area' in feat_lower or 'impervious_area' in feat_lower):
        return 'Land Cover'
    
    # 空间特征（2个）
    if (feat_lower in ['latitude', 'longitude', 'lat', 'lon', 'lng'] or
        feat_standard in ['latitude', 'longitude', 'lat', 'lon', 'lng']):
        return 'Spatial'
    
    # 时间特征（1个）
    if 'year' in feat_lower or 'year' in feat_standard:
        return 'Temporal'
    
    # GEO相关特征（地理位置）- 与feature_plots保持一致
    if 'geo' in feat_lower or feat_lower == 'geo':
        return 'Geographic'
    
    # 交互效应特征
    if ('×' in feature_name or '*' in feature_name or 'x' in feat_lower.split() or 
        'interaction' in feat_lower):
        return 'Interaction'
    
    # 🚨 处理移除的特征（用于向后兼容和错误处理）
    removed_features = ['pet', 'aspect', 'grassland_area_percent', 'shrubland_area_percent', 'bareland_area_percent']
    if (feat_lower in removed_features or feat_standard in removed_features or
        any(removed in feat_lower for removed in ['grassland', 'shrubland', 'bareland', 'aspect'])):
        return 'Removed Feature'
    
    # 其他未分类特征
    return 'Other'

def ensure_spatiotemporal_features(feature_list, all_features):
    """
    确保时空模型的核心特征（GEO和year）始终包含在特征列表中
    
    实现"8+2"策略：
    1. 如果GEO和year都不在top 8中：top 8 + GEO + year = 10个特征
    2. 如果只有一个在top 8中：top 8 + 缺失的那个 + 第9名 = 10个特征  
    3. 如果都在top 8中：top 8 + 第9名 + 第10名 = 10个特征
    
    注意：GEO是经纬度的联合特征，作为单一特征参与选择
    始终返回10个特征，且必须包含GEO和year！
    
    Parameters:
    -----------
    feature_list : list
        当前选择的特征列表（通常是基于重要性排序的，共18个特征）
    all_features : list
        所有可用特征的完整列表（应该是18个特征，已包含GEO）
        
    Returns:
    --------
    list
        包含10个特征的列表，保证包含GEO和year
    """
    # 创建特征列表的副本，避免修改原始列表
    feature_list = list(feature_list)
    
    # 查找GEO和year特征
    geo_feature = None
    year_feature = None
    
    # 在所有特征中查找
    for feat in all_features:
        feat_lower = feat.lower()
        if feat_lower == 'geo':
            geo_feature = feat
        elif 'year' in feat_lower:
            year_feature = feat
        
    # 检查是否找到必需的特征
    if not geo_feature:
        print("  ⚠️ 警告: 未找到GEO特征，这可能是数据预处理的问题")
    if not year_feature:
        print("  ⚠️ 警告: 未找到year特征")
    
    # 检查top 8中是否包含GEO和year
    top_8 = feature_list[:8] if len(feature_list) >= 8 else feature_list
    top_8_lower = [f.lower() for f in top_8]
    
    # 判断GEO是否在top 8中
    geo_in_top8 = geo_feature and geo_feature.lower() in top_8_lower
    
    # 判断year是否在top 8中
    year_in_top8 = year_feature and any('year' in f for f in top_8_lower)
    
    # 根据不同情况构建最终列表
    final_list = []
    
    if not geo_in_top8 and not year_in_top8:
        # 情况1：都不在top 8中 -> top 8 + GEO + year = 10
        print("  📊 策略1: GEO和year都不在top 8中")
        final_list = top_8[:8]  # 取前8个
    
        # 添加GEO
        if geo_feature and geo_feature not in final_list:
            final_list.append(geo_feature)
            print(f"  📍 添加空间特征: {geo_feature}")
        
        # 添加year
        if year_feature and year_feature not in final_list:
            final_list.append(year_feature)
            print(f"  📅 添加时间特征: {year_feature}")
            
    elif geo_in_top8 != year_in_top8:
        # 情况2：只有一个在top 8中 -> top 8 + 缺失的那个 + 第9名
        in_top8 = 'GEO' if geo_in_top8 else 'year'
        not_in_top8 = 'year' if geo_in_top8 else 'GEO'
        print(f"  📊 策略2: {in_top8}在top 8中，{not_in_top8}不在")
        final_list = top_8[:8]
        
        # 添加缺失的特征
        if not geo_in_top8 and geo_feature:
            final_list.append(geo_feature)
            print(f"  📍 添加缺失的空间特征: {geo_feature}")
        elif not year_in_top8 and year_feature:
            final_list.append(year_feature)
            print(f"  📅 添加缺失的时间特征: {year_feature}")
        
        # 添加第9名特征（如果有）
        if len(feature_list) > 8:
            # 找到第9名特征（跳过已经在final_list中的）
            for feat in feature_list[8:]:
                if feat not in final_list:
                    final_list.append(feat)
                    print(f"  ➕ 添加第9名特征: {feat}")
                    break
                    
    else:
        # 情况3：都在top 8中 -> top 8 + 第9名 + 第10名
        print("  📊 策略3: GEO和year都在top 8中")
        final_list = top_8[:8]
        
        # 添加第9、10名特征
        extra_count = 0
        for feat in feature_list[8:]:
            if feat not in final_list and extra_count < 2:
                final_list.append(feat)
                print(f"  ➕ 添加第{9+extra_count}名特征: {feat}")
                extra_count += 1
                if extra_count >= 2:
                    break
    
    # 确保最终有10个特征
    if len(final_list) < 10:
        print(f"  ⚠️ 特征数量不足10个（{len(final_list)}），尝试从剩余特征中补充")
        # 从所有特征中补充（排除已选择的）
        for feat in feature_list:
            if feat not in final_list:
                final_list.append(feat)
                if len(final_list) >= 10:
                    break
    elif len(final_list) > 10:
        print(f"  ⚠️ 特征数量超过10个（{len(final_list)}），截取前10个")
        final_list = final_list[:10]
    
    print(f"  ✅ 最终选择了{len(final_list)}个特征")
    
    return final_list

def get_spatiotemporal_features(feature_names):
    """
    从特征列表中识别并返回空间和时间特征
    
    Parameters:
    -----------
    feature_names : list
        特征名称列表
        
    Returns:
    --------
    dict
        包含空间和时间特征的字典
    """
    spatial_features = []
    temporal_features = []
    
    for feat in feature_names:
        feat_lower = feat.lower() if isinstance(feat, str) else str(feat).lower()
        
        # 识别空间特征
        if feat_lower == 'geo' or 'latitude' in feat_lower or 'longitude' in feat_lower or feat_lower in ['lat', 'lon']:
            spatial_features.append(feat)
        
        # 识别时间特征
        elif 'year' in feat_lower:
            temporal_features.append(feat)
    
    return {
        'spatial': spatial_features,
        'temporal': temporal_features,
        'has_geo': any('geo' in f.lower() for f in spatial_features),
        'has_lat_lon': any('lat' in f.lower() for f in spatial_features) and any('lon' in f.lower() for f in spatial_features),
        'has_year': len(temporal_features) > 0
    }

def filter_features_for_visualization(feature_importance_list, top_n=10, ensure_spatiotemporal=True, all_features=None):
    """
    根据重要性过滤特征，可选择性地确保包含时空特征
    
    Parameters:
    -----------
    feature_importance_list : list of tuples
        [(feature_name, importance), ...] 格式的特征重要性列表
    top_n : int
        要选择的特征数量
    ensure_spatiotemporal : bool
        是否确保包含时空特征（默认为True）
    all_features : list
        所有可用特征列表（用于查找时空特征）
        
    Returns:
    --------
    list
        选择的特征名称列表
    """
    # 按重要性排序（如果还未排序）
    sorted_features = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)
    
    # 选择前top_n个特征
    selected_features = [feat for feat, _ in sorted_features[:top_n]]
    
    # 如果需要，确保包含时空特征
    if ensure_spatiotemporal and all_features is not None:
        selected_features = ensure_spatiotemporal_features(selected_features, all_features)
    
    return selected_features 

def categorize_feature_with_interactions(feature_name):
    """
    基于GeoShapley三部分分解的特征分类函数
    
    返回:
    - 'Primary': 主效应特征（12个环境特征）
    - 'Geographic': GEO特征（经纬度合并）
    - 'Interaction': 交互效应特征（主效应 × GEO）
    """
    if not isinstance(feature_name, str):
        feature_name = str(feature_name)
    
    feature_lower = feature_name.lower().strip()
    
    # GEO特征
    if feature_lower == 'geo':
        return 'Geographic'
    
    # 交互效应特征
    if '×' in feature_name or 'x ' in feature_lower or ' x ' in feature_lower or ' × ' in feature_name:
        return 'Interaction'
    
    # 主效应特征（12个环境特征）
    primary_features = {
        'temperature', 'precipitation',  # 气候(2)
        'nightlight', 'road_density', 'mining_density', 'population_density',  # 人类活动(4)
        'elevation', 'slope',  # 地形(2)
        'forest_area_percent', 'cropland_area_percent', 'impervious_area_percent',  # 土地覆盖(3)
        'year'  # 时间(1)
    }
    
    # 兼容性：处理简化名称
    simplified_check = any(feat in feature_lower for feat in primary_features)
    if simplified_check or feature_lower in primary_features:
        return 'Primary'
    
    # 🔇 静默处理：经纬度特征（实际已由GeoShapley正确合并为GEO）
    if feature_lower in ['latitude', 'longitude', 'lat', 'lon']:
        # 移除冗余警告，GeoShapley已正确合并这些特征
        return 'Geographic'  # 强制归类为Geographic
    
    # 其他特征默认归类为Primary（向后兼容）
    print(f"⚠️ 未分类特征 {feature_name} 默认归为Primary")
    return 'Primary'

def get_feature_display_order(feature_list):
    """
    按照GeoShapley结构排序特征：主效应 → GEO → 交互效应
    """
    primary_features = []
    geo_features = []
    interaction_features = []
    
    for feature in feature_list:
        category = categorize_feature_with_interactions(feature)
        if category == 'Primary':
            primary_features.append(feature)
        elif category == 'Geographic':
            geo_features.append(feature)
        elif category == 'Interaction':
            interaction_features.append(feature)
    
    # 返回正确顺序：主效应 → GEO → 交互效应
    return primary_features + geo_features + interaction_features 
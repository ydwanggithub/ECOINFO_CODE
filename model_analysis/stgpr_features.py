#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
时空高斯过程回归模型 (ST-GPR) - 特征准备模块

本模块包含ST-GPR模型的特征准备功能：
1. 准备STGPR模型训练所需的特征 (prepare_features_for_stgpr)
"""

import pandas as pd


def prepare_features_for_stgpr(df, target='VHI'):
    """
    为ST-GPR模型准备特征，优化后使用14个特征（从19个减少）
    
    优化策略：去掉5个相对不重要的特征以大幅提升GeoShapley计算效率
    计算效率提升：GeoShapley复杂度从O(2^19)降低到O(2^14)，约97%的计算量减少
    
    优化后的特征列表（14个）：
    - 空间信息: latitude, longitude (2个)
    - 环境特征 (11个):
      - 气候因素: temperature, precipitation (2个，去掉pet)
      - 人类活动因素: nightlight, road_density, mining_density, population_density (4个)
      - 地形因素: elevation, slope (2个，去掉aspect)
      - 土地覆盖因素: forest_area_percent, cropland_area_percent, impervious_area_percent (3个，去掉grassland/shrubland/bareland)
    - 时间信息: year (1个)
    
    移除的特征：pet, aspect, grassland_area_percent, shrubland_area_percent, bareland_area_percent
    
    参数:
    df: 包含所有特征的DataFrame
    target: 目标变量名称
    
    返回:
    tuple: (特征矩阵X, 目标变量y)
    """
    print("🎯 为ST-GPR模型准备优化后的特征（14个特征）...")
    
    # 按照优化后设计定义基础特征列表
    base_features = []
    
    # 位置特征 (必须包含)
    location_cols = ['latitude', 'longitude']
    for col in location_cols:
        if col in df.columns:
            base_features.append(col)
        else:
            print(f"⚠️ 缺少关键位置特征 '{col}'")
    
    # 环境特征：优化后的11个基础特征
    # 按顺序：气候、人类活动、地形、土地覆盖
    env_features = [
        # 气候因素 (2个，去掉pet)
        'temperature', 'precipitation',
        # 人类活动因素 (4个，保持不变)
        'nightlight', 'road_density', 'mining_density', 'population_density',
        # 地形因素 (2个，去掉aspect)
        'elevation', 'slope',
        # 土地覆盖因素 (3个，去掉grassland/shrubland/bareland)
        'forest_area_percent', 'cropland_area_percent', 'impervious_area_percent'
    ]
    
    # 检查并添加可用的环境特征
    available_env_features = []
    missing_env_features = []
    removed_features = ['pet', 'aspect', 'grassland_area_percent', 'shrubland_area_percent', 'bareland_area_percent']
    
    for feature in env_features:
        if feature in df.columns:
            base_features.append(feature)
            available_env_features.append(feature)
        else:
            missing_env_features.append(feature)

    if missing_env_features:
        print(f"⚠️ 警告: 缺少以下环境特征: {', '.join(missing_env_features)}")
    
    print(f"✅ 优化移除的特征: {', '.join(removed_features)}")
    print(f"📈 GeoShapley计算效率提升: 约97%的计算量减少")

    # 年份 (必须包含)
    if 'year' in df.columns:
        base_features.append('year')
    else:
        print("⚠️ 警告: 缺少年份列 'year'")

    # 检查特征数量 - 优化后期望14个特征：经纬度、11个环境特征、年份
    expected_features = 14
    if len(base_features) < expected_features:  
        print(f"⚠️ 警告: 特征数量({len(base_features)})少于预期({expected_features})，缺少的特征可能会影响模型性能")
    elif len(base_features) == expected_features:
        print(f"✅ 特征数量正确: {len(base_features)}个特征")
    else:
        print(f"⚠️ 特征数量({len(base_features)})超过预期({expected_features})")
        
    print(f"🔧 使用特征数量: {len(base_features)}")
    print(f"📋 特征列表: {base_features}")

    # 构建特征矩阵和目标向量
    X = df[base_features].copy()
    y = df[target].copy() if target in df.columns else pd.Series()

    # 处理特征缺失值
    for col in X.columns:
        if X[col].isna().any():
            print(f"🔧 特征'{col}'存在缺失值，使用中位数填充")
            X[col] = X[col].fillna(X[col].median())
    
    print(f"📊 最终特征矩阵: {X.shape[0]}行 × {X.shape[1]}列")
    
    # 添加特征类别信息（更新为优化后的分类）
    feature_categories = {}
    # 按优化后设计定义特征类别
    for col in X.columns:
        if col in ['latitude', 'longitude']:
            category = 'Spatial'
        elif col in ['temperature', 'precipitation']:  # 更新：去掉pet
            category = 'Climate'
        elif col in ['nightlight', 'road_density', 'mining_density', 'population_density']:
            category = 'Human'
        elif col in ['elevation', 'slope']:  # 更新：去掉aspect
            category = 'Terrain'
        elif col in ['forest_area_percent', 'cropland_area_percent', 'impervious_area_percent']:  # 更新：去掉3个
            category = 'Land Cover'
        elif col == 'year':
            category = 'Temporal'
        else:
            # 🔴 不再默认归类，而是记录并警告
            print(f"⚠️ 警告: 无法分类特征 '{col}'，这不应该发生在优化后的ST-GPR模型中")
            print(f"   优化后的ST-GPR模型应该只包含14个预定义的特征")
            category = 'Unknown'  # 使用Unknown而不是Spatial，更明确表示有问题
        
        feature_categories[col] = category
    
    # 将特征类别分组（更新为优化后的分组）
    feature_categories_grouped = {
        'Climate': [f for f in X.columns if feature_categories.get(f) == 'Climate'],
        'Human': [f for f in X.columns if feature_categories.get(f) == 'Human'],
        'Terrain': [f for f in X.columns if feature_categories.get(f) == 'Terrain'],
        'Land Cover': [f for f in X.columns if feature_categories.get(f) == 'Land Cover'],
        'Spatial': [f for f in X.columns if feature_categories.get(f) == 'Spatial'],
        'Temporal': [f for f in X.columns if feature_categories.get(f) == 'Temporal']
    }
    
    # 存储特征类别信息
    # 特征简写映射 - 更新为优化后的特征
    feature_abbreviations = {
        'latitude': 'LAT', 'longitude': 'LONG',
        'temperature': 'TEMP', 'precipitation': 'PREC',  # 去掉pet
        'nightlight': 'NIGH', 'road_density': 'RD', 'mining_density': 'MD', 'population_density': 'PD',
        'elevation': 'ELEV', 'slope': 'SLOP',  # 去掉aspect
        'forest_area_percent': 'FAP', 'cropland_area_percent': 'CAP', 'impervious_area_percent': 'IAP',  # 去掉GAP、SAP、BAP
        'year': 'YEAR'
    }
    
    try:
        X.attrs['feature_categories'] = feature_categories
        X.attrs['feature_categories_grouped'] = feature_categories_grouped
        X.attrs['feature_names'] = list(base_features)
        X.attrs['feature_abbreviations'] = feature_abbreviations
        print("✅ 特征元信息已附加到DataFrame")
    except Exception as e:
        print(f"⚠️ 无法将特征信息附加到DataFrame: {e}")

    # 🔥 打印优化效果摘要
    print(f"\n📈 特征优化效果摘要:")
    print(f"  • 特征数量: 19 → 14 (-5个特征)")
    print(f"  • GeoShapley复杂度: 2^19 → 2^14 (约97%计算量减少)")
    print(f"  • 移除的特征: pet, aspect, grassland/shrubland/bareland覆盖率")
    print(f"  • 保留核心特征: 气候2个, 人类活动4个, 地形2个, 土地覆盖3个, 时空3个")
    
    return X, y 
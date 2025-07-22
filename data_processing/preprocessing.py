#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块：处理H3聚合后的植被健康数据

主要功能：
1. 统一列名格式
2. 提取基础特征
3. 数据质量检查
4. 准备STGPR+GeoShapley框架输入

优化后的特征体系（14个特征）：
- 空间信息：latitude, longitude (2个)
- 气候特征：temperature, precipitation (2个，去掉pet)
- 人类活动特征：nightlight, population_density, road_density, mining_density (4个)
- 地形特征：elevation, slope (2个，去掉aspect)
- 土地覆盖特征（百分比）：forest_area_percent, cropland_area_percent, impervious_area_percent (3个，去掉grassland/shrubland/bareland)
- 时间特征：year (1个)

计算效率提升：GeoShapley复杂度从O(2^19)降低到O(2^14)，约97%的计算量减少

处理的数据结构（37列）：
- 空间标识：h3_index, latitude, longitude, hex_id_res5
- 时间标识：year
- 气候特征：temperature, precipitation, pet
- 植被指标：gpp, lai, fpar, evi
- 人类活动特征：nightlight, population_density, road_density, mining_density
- 土地覆盖特征（面积）：forest_area, cropland_area, grassland_area, shrubland_area, impervious_area, bareland_area
- 土地覆盖特征（百分比）：forest_area_percent, cropland_area_percent, grassland_area_percent, shrubland_area_percent, impervious_area_percent, bareland_area_percent
- 地形特征：elevation, slope, aspect
- 目标变量：VHI
- 其他：total_area_km2, has_valid_data, .geo

作者: Yuandong Wang (wangyuandong@gnnu.edu.cn)
日期: 2025.07.26
"""

import os
import numpy as np
import pandas as pd
import warnings
from typing import Optional, Tuple, Dict, List

def load_complete_dataset(data_dir: str = None, resolutions: List[str] = None, 
                        verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    直接加载2000-2024年完整数据集（包含ARIMA外推数据）
    
    这是论文代码的主要数据加载方式，读者可以清楚看到使用的是完整25年数据集
    
    参数:
    data_dir: 数据目录路径（默认为项目根目录下的data文件夹）
    resolutions: 要加载的分辨率列表，如['res5', 'res6', 'res7']
    verbose: 是否打印详细信息
    
    返回:
    dict: 分辨率为键，DataFrame为值的字典
    """
    if verbose:
        print("=== ⚡ 加载2000-2024年完整数据集 ===")
        print("📊 包含25年数据: 2000-2020观测数据 + 2021-2024 ARIMA外推数据")
    
    # 如果未指定数据目录，使用项目根目录下的data文件夹
    if data_dir is None:
        # 获取项目根目录（data_processing的父目录）
        current_dir = os.path.dirname(os.path.abspath(__file__))  # data_processing目录
        project_root = os.path.dirname(current_dir)  # 项目根目录
        data_dir = os.path.join(project_root, 'data')
    
    # 如果未指定分辨率，则加载所有可用的
    if resolutions is None:
        resolutions = ['res5', 'res6', 'res7']
    
    data_by_resolution = {}
    
    # 文件名模式（完整数据集）
    file_patterns = {
        'res5': 'ALL_DATA_with_VHI_PCA_res5.csv',
        'res6': 'ALL_DATA_with_VHI_PCA_res6.csv',
        'res7': 'ALL_DATA_with_VHI_PCA_res7.csv'
    }
    
    if verbose:
        print(f"📂 数据目录: {data_dir}")
    
    for resolution in resolutions:
        if resolution not in file_patterns:
            if verbose:
                print(f"⚠️  跳过未支持的分辨率: {resolution}")
            continue
            
        file_path = os.path.join(data_dir, file_patterns[resolution])
        
        if not os.path.exists(file_path):
            if verbose:
                print(f"❌ 文件不存在: {file_path}")
            continue
        
        try:
            if verbose:
                print(f"📖 加载 {resolution} 数据: {file_patterns[resolution]}")
            
            # 读取数据
            df = pd.read_csv(file_path)
            
            # 数据验证
            required_columns = [
                'year', 'latitude', 'longitude', 'h3_index', 'VHI',
                'temperature', 'precipitation', 'elevation', 'slope',
                'nightlight', 'population_density', 'road_density', 'mining_density',
                'forest_area_percent', 'cropland_area_percent', 'impervious_area_percent'
            ]
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"⚠️  {resolution} 数据缺失列: {missing_cols}")
            
            # 时间范围验证
            year_range = (df['year'].min(), df['year'].max())
            if verbose:
                print(f"   📅 时间范围: {year_range[0]}-{year_range[1]} ({year_range[1]-year_range[0]+1}年)")
                print(f"   📊 数据形状: {df.shape}")
                print(f"   🎯 特征数量: {len([col for col in df.columns if col not in ['year', 'h3_index', 'VHI', 'latitude', 'longitude']])}个")
            
            # 确认包含完整25年数据
            if year_range != (2000, 2024):
                if verbose:
                    print(f"⚠️  注意: {resolution} 数据时间范围不是预期的2000-2024年")
            
            data_by_resolution[resolution] = df
            
            if verbose:
                print(f"✅ {resolution} 数据加载成功")
        
        except Exception as e:
            if verbose:
                print(f"❌ 加载 {resolution} 数据失败: {str(e)}")
    
    if verbose:
        print(f"\n📋 总结: 成功加载 {len(data_by_resolution)} 个分辨率的数据集")
        for res, df in data_by_resolution.items():
            print(f"  • {res}: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    
    return data_by_resolution

def load_and_check_data(file_path: str, verbose: bool = True) -> pd.DataFrame:
    """读取数据并进行初始检查"""
    if verbose:
        print(f"读取数据文件: {file_path}")
    
    df = pd.read_csv(file_path)
    
    if verbose:
        print(f"原始数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
    
    # 检查空值
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0 and verbose:
        print(f"发现空值的列: {null_counts[null_counts > 0].to_dict()}")
    
    return df

def standardize_h3_index(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """标准化H3索引列"""
    if verbose:
        print("标准化H3索引...")
    
    # 1. 检查hex_id列并重命名为h3_index
    hex_id_columns = [col for col in df.columns if 'hex_id' in col.lower()]
    
    if 'h3_index' not in df.columns and hex_id_columns:
        # 优先选择不带后缀的hex_id
        if 'hex_id' in hex_id_columns:
            df['h3_index'] = df['hex_id']
            if verbose:
                print(f"  使用hex_id列作为h3_index")
        else:
            # 使用第一个找到的hex_id列
            hex_id_col = hex_id_columns[0]
            df['h3_index'] = df[hex_id_col]
            if verbose:
                print(f"  使用{hex_id_col}列作为h3_index")
    
    # 2. 确保h3_index是字符串类型
    if 'h3_index' in df.columns:
        df['h3_index'] = df['h3_index'].astype(str)
        if verbose:
            valid_count = df['h3_index'].notna().sum()
            print(f"  h3_index列状态: {valid_count}/{len(df)} 个有效值")
    
    # 3. 创建备份列
    if 'h3_index' in df.columns and 'original_h3_index' not in df.columns:
        df['original_h3_index'] = df['h3_index'].copy()
        if verbose:
            print("  已创建original_h3_index备份列")
    
    return df

def standardize_landcover_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    标准化土地覆盖特征命名
    
    优化后只处理保留的3个土地覆盖特征（去掉grassland/shrubland/bareland）
    """
    if verbose:
        print("标准化土地覆盖特征...")
    
    # 定义重命名映射（优化后只保留3个特征）
    landcover_mapping = {
        'forest_area': 'forest_area_percent',
        'cropland_area': 'cropland_area_percent', 
        'crop_area': 'cropland_area_percent',
        'impervious_area': 'impervious_area_percent',
        'imperv_area': 'impervious_area_percent',
        # 移除以下映射（优化策略）:
        # 'grassland_area': 'grassland_area_percent',
        # 'grass_area': 'grassland_area_percent',
        # 'shrubland_area': 'shrubland_area_percent',
        # 'shrub_area': 'shrubland_area_percent',
        # 'bareland_area': 'bareland_area_percent',
        # 'bare_area': 'bareland_area_percent'
    }
    
    renamed_cols = 0
    for old_name, new_name in landcover_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
            renamed_cols += 1
            if verbose:
                print(f"  重命名: {old_name} -> {new_name}")
        elif old_name in df.columns and new_name in df.columns:
            # 如果两个列都存在，合并它们
            if verbose:
                print(f"  合并列: {old_name} -> {new_name}")
            # 使用非空值填充
            mask = df[new_name].isna() & df[old_name].notna()
            if mask.any():
                df.loc[mask, new_name] = df.loc[mask, old_name]
            # 删除旧列
            df = df.drop(columns=[old_name])
    
    if verbose and renamed_cols > 0:
        print(f"  总共重命名了 {renamed_cols} 个土地覆盖列")
        print(f"  ✅ 优化：只保留forest/cropland/impervious 3个土地覆盖特征")
    
    return df

def convert_data_types(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """转换数据类型"""
    if verbose:
        print("转换数据类型...")
    
    # 1. 确保H3相关列是字符串类型
    string_columns = ['h3_index', 'original_h3_index', 'hex_id']
    for col in string_columns:
        if col in df.columns:
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                df[col] = df[col].apply(lambda x: str(x) if not pd.isna(x) else x)
                if verbose:
                    print(f"  将列 {col} 转换为字符串类型")
    
    # 2. 确保经纬度是数值类型
    coord_columns = ['latitude', 'longitude']
    for col in coord_columns:
        if col in df.columns:
            try:
                if df[col].dtype != 'float64':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if verbose:
                        print(f"  将列 {col} 转换为数值类型")
            except Exception as e:
                if verbose:
                    print(f"  警告: 无法将 {col} 转换为数值类型: {e}")
    
    # 3. 确保数值特征列是数值类型（优化后去掉pet和aspect）
    numeric_features = [
        'year', 'temperature', 'precipitation',  # 去掉pet
        'gpp', 'lai', 'fpar', 'evi',  # 植被指标（原始数据可能包含，但后续会被排除）
        'nightlight', 'population_density', 'road_density', 'mining_density',
        'elevation', 'slope',  # 去掉aspect
        'VHI', 'total_area_km2'
    ]
    
    # 添加土地覆盖百分比列（优化后只有3个）
    landcover_features = [
        'forest_area_percent', 'cropland_area_percent', 'impervious_area_percent'
    ]
    numeric_features.extend(landcover_features)
    
    for col in numeric_features:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if verbose and df[col].isna().sum() > 0:
                    print(f"  转换 {col} 时产生了 {df[col].isna().sum()} 个NaN值")
            except Exception as e:
                if verbose:
                    print(f"  警告: 转换 {col} 时出错: {e}")
    
    return df

def handle_missing_values(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """处理缺失值"""
    if verbose:
        print("处理缺失值...")
    
    # 1. 检查空值比例
    null_ratios = df.isnull().sum() / len(df)
    high_null_cols = null_ratios[null_ratios > 0.9].index.tolist()
    
    if high_null_cols and verbose:
        print(f"  空值比例超过90%的列: {high_null_cols}")
    
    # 2. 对数值列用均值填充空值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['h3_index', 'original_h3_index'] and df[col].isnull().any():
            if null_ratios[col] <= 0.9:  # 只处理空值比例不太高的列
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
                if verbose:
                    print(f"  用均值 {mean_val:.4f} 填充 {col} 的空值")
    
    return df

def extract_basic_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """提取基础特征，排除不需要的列"""
    if verbose:
        print("提取基础特征...")
    
    # 定义要保留的基础特征类别（更新：去掉5个特征）
    basic_features = {
        # 时间标识
        'temporal': ['year'],
        
        # 气候特征（从3个减少到2个）
        'climate': ['temperature', 'precipitation'],
        
        # 地形特征（从3个减少到2个）
        'terrain': ['elevation', 'slope'],
        
        # 人类活动特征（保持4个不变）
        'human': ['nightlight', 'population_density', 'road_density', 'mining_density'],
        
        # 土地覆盖特征（从6个减少到3个）
        'landcover': [
            'forest_area_percent', 'cropland_area_percent', 'impervious_area_percent'
        ],
        
        # 目标变量
        'target': ['VHI'],
        
        # 位置信息（用于后续分析，但不作为模型特征）
        'location': ['h3_index', 'latitude', 'longitude']
    }
    
    # 收集所有要保留的列
    keep_columns = []
    for category, features in basic_features.items():
        available_features = [f for f in features if f in df.columns]
        keep_columns.extend(available_features)
        if verbose:
            print(f"  {category}: {len(available_features)} 个特征 - {available_features}")
    
    # 排除的特征（避免数据泄露和优化的特征）
    exclude_features = [
        'gpp', 'lai', 'fpar', 'evi',  # 原始植被指标
        'total_area', 'total_area_km2',  # 面积信息
        'has_valid_data', '.geo',  # 元数据
        'original_h3_index',  # 备份列
        # 🔴 新增：明确排除的优化特征
        'pet',  # 潜在蒸散发
        'aspect',  # 坡向
        'grassland_area_percent',  # 草地覆盖百分比
        'shrubland_area_percent',  # 灌木覆盖百分比
        'bareland_area_percent'  # 裸地覆盖百分比
    ]
    
    # 从保留列表中移除要排除的特征
    keep_columns = [col for col in keep_columns if col not in exclude_features]
    
    # 检查是否有其他未分类的数值列
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    unclassified_cols = [col for col in all_numeric_cols if col not in keep_columns and col not in exclude_features]
    
    if unclassified_cols and verbose:
        print(f"  发现未分类的数值列: {unclassified_cols}")
        # 可以选择是否包含这些列
        # keep_columns.extend(unclassified_cols)
    
    # 提取基础特征
    basic_df = df[keep_columns].copy()
    
    if verbose:
        print(f"  最终基础特征数据集: {basic_df.shape[0]} 行 × {basic_df.shape[1]} 列")
        print(f"  保留的列: {list(basic_df.columns)}")
    
    return basic_df

def preprocess_for_basic_features(file_path: str, save_path: Optional[str] = None, 
                                 verbose: bool = True) -> pd.DataFrame:
    """
    完整的基础特征预处理流程
    
    注意：此函数仅用于处理原始数据。主要工作流程请使用load_complete_dataset()函数。
    
    参数:
    file_path (str): 原始CSV文件路径
    save_path (str): 保存处理后数据的路径（可选）
    verbose (bool): 是否打印详细信息
    
    返回:
    DataFrame: 处理后的基础特征数据
    """
    if verbose:
        print("=" * 50)
        print("开始基础特征预处理流程")
        print("💡 提示: 对于常规使用，建议使用load_complete_dataset()函数")
        print("=" * 50)
    
    # 步骤1: 读取数据
    df = load_and_check_data(file_path, verbose)
    
    # 步骤2: H3索引标准化
    df = standardize_h3_index(df, verbose)
    
    # 步骤3: 土地覆盖特征标准化
    df = standardize_landcover_features(df, verbose)
    
    # 步骤4: 数据类型转换
    df = convert_data_types(df, verbose)
    
    # 步骤5: 缺失值处理
    df = handle_missing_values(df, verbose)
    
    # 步骤6: 提取基础特征
    basic_df = extract_basic_features(df, verbose)
    
    # 步骤7: 跳过时间外推 (现在使用预处理好的完整数据集)
    if verbose:
        print(f"\n💡 数据处理完成，时间范围: {basic_df['year'].min()}-{basic_df['year'].max()}")
    
    # 步骤8: 最终检查
    if verbose:
        print("\n最终数据质量检查:")
        print(f"  数据形状: {basic_df.shape}")
        print(f"  时间范围: {basic_df['year'].min()}-{basic_df['year'].max()} ({basic_df['year'].nunique()}年)")
        print(f"  空间范围: {basic_df['h3_index'].nunique()}个H3网格")
        print(f"  空值总数: {basic_df.isnull().sum().sum()}")
        print(f"  数值列数: {len(basic_df.select_dtypes(include=[np.number]).columns)}")
        
        # VHI目标变量检查
        if 'VHI' in basic_df.columns:
            vhi_stats = basic_df.groupby('year')['VHI'].agg(['count', 'mean', lambda x: x.isna().sum()])
            vhi_stats.columns = ['total_records', 'mean_vhi', 'null_count']
            print(f"  🎯 VHI分布:")
            for year, row in vhi_stats.iterrows():
                print(f"    {year}: {row['total_records']}条记录, 均值={row['mean_vhi']:.3f}, 空值={row['null_count']}")
        
        # 显示每个特征的基本统计
        print("\n基础特征统计:")
        for col in basic_df.columns:
            if col not in ['h3_index', 'latitude', 'longitude']:
                if pd.api.types.is_numeric_dtype(basic_df[col]):
                    print(f"  {col}: 均值={basic_df[col].mean():.4f}, 标准差={basic_df[col].std():.4f}")
    
    # 保存处理后的数据
    if save_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            basic_df.to_csv(save_path, index=False)
            if verbose:
                print(f"\n处理后的数据已保存到: {save_path}")
        except Exception as e:
            if verbose:
                print(f"\n保存数据时出错: {e}")
    
    if verbose:
        print("=" * 50)
        print("基础特征预处理完成")
        print("=" * 50)
    
    return basic_df

def process_single_file(file_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    处理单个数据文件：标准化列名、提取基础特征、数据清理
    
    参数:
    file_path: 数据文件路径
    verbose: 是否打印详细信息
    
    返回:
    DataFrame: 处理后的数据框
    """
    if verbose:
        print(f"📖 处理文件: {os.path.basename(file_path)}")
    
    # 读取数据
    df = pd.read_csv(file_path)
    
    if verbose:
        print(f"  原始数据: {len(df)}行 × {df.shape[1]}列")
    
    # 数据清理和标准化
    df_processed = standardize_column_names(df)
    df_processed = extract_basic_features(df_processed, verbose=verbose)
    df_processed = clean_data(df_processed, verbose=verbose)
    
    if verbose:
        print(f"  处理后: {len(df_processed)}行 × {df_processed.shape[1]}列")
        year_range = (df_processed['year'].min(), df_processed['year'].max())
        print(f"  时间范围: {year_range[0]}-{year_range[1]}")
    
    return df_processed

def load_data_files(data_dir: str, resolutions: List[str] = None, 
                   verbose: bool = True, force_reprocess: bool = False) -> Dict[str, pd.DataFrame]:
    """
    加载指定分辨率的数据文件
    
    优化策略：优先从预处理后的文件读取，避免重复预处理
    
    参数:
    data_dir: 数据目录路径
    resolutions: 要加载的分辨率列表，如['res5', 'res6', 'res7']，如果为None则加载所有可用的
    verbose: 是否打印详细信息
    force_reprocess: 是否强制重新处理（忽略已有的预处理文件）
    
    返回:
    dict: 分辨率为键，DataFrame为值的字典
    """
    if verbose:
        print("=== 🚀 加载数据文件 ===")
        if not force_reprocess:
            print("📈 优先使用2000-2024年完整数据集")
    
    # 使用新的load_complete_dataset函数
    return load_complete_dataset(data_dir=data_dir, resolutions=resolutions, verbose=verbose)

def load_processed_data_files(resolutions: List[str] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    直接从预处理后的数据文件加载数据，不进行任何预处理
    
    这是最快的数据加载方式，适合模型已经训练好，只需要快速加载数据的场景
    
    参数:
    resolutions: 要加载的分辨率列表，如['res5', 'res6', 'res7']，如果为None则加载所有可用的
    verbose: 是否打印详细信息
    
    返回:
    dict: 分辨率为键，DataFrame为值的字典
    """
    if verbose:
        print("=== ⚡ 直接加载预处理数据文件 ===")
        print("🎯 跳过所有预处理步骤，直接读取已处理的数据")
    
    # 如果未指定分辨率，则尝试加载所有可用的
    if resolutions is None:
        resolutions = ['res5', 'res6', 'res7']
    
    data_by_resolution = {}
    
    # 预处理数据存储目录
    current_dir = os.path.dirname(os.path.abspath(__file__))  # data_processing目录
    processed_data_dir = os.path.join(current_dir, 'data')
    
    if verbose:
        print(f"📂 预处理数据目录: {processed_data_dir}")
    
    for res in resolutions:
        processed_filename = f'ALL_DATA_with_VHI_PCA_{res}_processed.csv'
        processed_file_path = os.path.join(processed_data_dir, processed_filename)
        
        try:
            if not os.path.exists(processed_file_path):
                if verbose:
                    print(f"⚠️ {res}: 预处理文件不存在: {processed_filename}")
                continue
            
            if verbose:
                print(f"📖 加载 {res}: {processed_filename}")
            
            # 直接读取CSV文件
            df = pd.read_csv(processed_file_path)
            
            # 基本验证
            if len(df) == 0:
                if verbose:
                    print(f"  ❌ {res}: 文件为空")
                continue
            
            # 检查关键列是否存在
            required_cols = ['h3_index', 'latitude', 'longitude', 'year', 'VHI']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                if verbose:
                    print(f"  ⚠️ {res}: 缺少关键列: {missing_cols}")
            
            # 数据摘要
            file_size = os.path.getsize(processed_file_path) / (1024 * 1024)  # MB
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            if 'year' in df.columns:
                year_range = (df['year'].min(), df['year'].max())
                has_extrapolated_data = df['year'].max() >= 2021
                time_status = "含外推数据" if has_extrapolated_data else "仅历史数据"
            else:
                year_range = ("未知", "未知")
                time_status = "无年份信息"
            
            if verbose:
                print(f"  ✅ 成功: {df.shape[0]:,}行 × {df.shape[1]}列")
                print(f"    📁 文件大小: {file_size:.1f}MB")
                print(f"    💾 内存占用: {memory_mb:.1f}MB")
                print(f"    📅 时间范围: {year_range[0]}-{year_range[1]} ({time_status})")
            
            data_by_resolution[res] = df
            
        except Exception as e:
            if verbose:
                print(f"❌ {res}: 加载失败: {e}")
            continue
    
    if verbose:
        total_loaded = len(data_by_resolution)
        print(f"\n✅ 快速加载完成: {total_loaded}/{len(resolutions)} 个分辨率")
        
        if total_loaded == 0:
            print("💡 提示: 如果预处理文件不存在，请先运行完整的数据加载流程生成预处理文件")
            print("       或使用 load_data_files() 函数进行完整的数据处理")
    
    return data_by_resolution

def prepare_features_for_stgpr(df: pd.DataFrame, target: str = 'VHI') -> Tuple[pd.DataFrame, pd.Series]:
    """
    为ST-GPR模型准备特征，优化后使用14个特征（从19个减少）
    
    优化策略：去掉5个相对不重要的特征以大幅提升GeoShapley计算效率
    
    优化后的ST-GPR模型使用14个特征：
    - 空间信息: latitude, longitude (2个)
    - 环境特征 (11个):
      - 气候因素: temperature, precipitation (2个，去掉pet)
      - 人类活动因素: nightlight, road_density, mining_density, population_density (4个)
      - 地形因素: elevation, slope (2个，去掉aspect)
      - 土地覆盖因素: forest_area_percent, cropland_area_percent, impervious_area_percent (3个，去掉grassland/shrubland/bareland)
    - 时间信息: year (1个)
    
    计算效率提升：GeoShapley复杂度从O(2^19)降低到O(2^14)，约97%的计算量减少
    
    参数:
    df: 输入数据框
    target: 目标变量名称
    
    返回:
    tuple: (特征矩阵X, 目标变量y)
    """
    print("🎯 为ST-GPR模型准备优化后的基础特征（14个特征）...")
    
    # 按照优化后设计严格定义基础特征列表
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
    
    # 特征简写映射 - 更新为优化后的特征
    feature_abbreviations = {
        'latitude': 'LAT', 'longitude': 'LONG',
        'temperature': 'TEMP', 'precipitation': 'PREC',  # 去掉pet
        'nightlight': 'NIGH', 'road_density': 'RD', 'mining_density': 'MD', 'population_density': 'PD',
        'elevation': 'ELEV', 'slope': 'SLOP',  # 去掉aspect
        'forest_area_percent': 'FAP', 'cropland_area_percent': 'CAP', 'impervious_area_percent': 'IAP',  # 去掉GAP、SAP、BAP
        'year': 'YEAR'
    }
    
    # 存储特征元信息
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

# 兼容性函数，保持与原有代码的兼容性
def preprocess_data(df: pd.DataFrame, verbose: bool = True, filepath: Optional[str] = None, 
                   resolution: Optional[str] = None, overwrite_original: bool = False) -> pd.DataFrame:
    """
    预处理数据的兼容性包装函数
    
    参数:
    df: 输入数据框
    verbose: 是否打印详细信息
    filepath: 如果提供，会将处理后的数据保存到该路径
    resolution: 分辨率级别（res5, res6, res7）
    overwrite_original: 是否覆盖原始文件，默认为False
    
    返回:
    DataFrame: 处理后的数据框
    """
    if verbose:
        print(f"  🔧 数据预处理 - 初始形状: {df.shape}")
    
    # 使用新的预处理流程
    # 1. H3索引标准化
    df = standardize_h3_index(df, verbose)
    
    # 2. 土地覆盖特征标准化
    df = standardize_landcover_features(df, verbose)
    
    # 3. 数据类型转换
    df = convert_data_types(df, verbose)
    
    # 4. 缺失值处理
    df = handle_missing_values(df, verbose)
    
    if verbose:
        print(f"   ✅ 预处理完成: {df.shape}")
        
    return df

# 为了保持兼容性，保留一些原有的函数签名
def get_data_summary(data_by_resolution: Dict[str, pd.DataFrame], verbose: bool = True) -> Dict:
    """获取数据摘要信息"""
    summary = {}
    
    for res, df in data_by_resolution.items():
        summary[res] = {
            'shape': df.shape,
            'columns': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'null_counts': df.isnull().sum().to_dict()
        }
        
        if verbose:
            print(f"{res}: {df.shape[0]:,}行 × {df.shape[1]}列")
    
    return summary

# 使用示例
if __name__ == "__main__":
    # 处理不同分辨率的数据
    resolutions = ['res7', 'res6', 'res5']
    
    for res in resolutions:
        # 修复：使用绝对路径，确保能找到原始数据文件
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
        input_file = os.path.join(project_root, 'data', f'ALL_DATA_with_VHI_PCA_{res}.csv')
        
        # 修复：将输出文件保存到data_processing/data目录
        current_dir = os.path.dirname(os.path.abspath(__file__))  # data_processing目录
        output_dir = os.path.join(current_dir, 'data')
        output_file = os.path.join(output_dir, f'ALL_DATA_with_VHI_PCA_{res}_basic_features.csv')
        
        if os.path.exists(input_file):
            print(f"\n处理 {res} 分辨率数据...")
            basic_features_df = preprocess_for_basic_features(
                input_file, 
                output_file, 
                verbose=True
            )
        else:
            print(f"文件不存在: {input_file}") 
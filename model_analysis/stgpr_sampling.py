#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
时空高斯过程回归模型 (ST-GPR) - 采样模块

本模块包含ST-GPR模型的采样相关功能：
1. 时空分层采样 (perform_spatiotemporal_sampling)
2. 测试数据采样 (sample_data_for_testing)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def perform_spatiotemporal_sampling(X_samples_df, sample_size, h3_col='h3_index', year_col='year', spatial_coverage=None, random_state=42):
    """
    执行两阶段时空分层采样策略
    
    第一阶段：选择一部分H3网格，并保留这些网格的所有时间点数据
    第二阶段：选择一部分时间截面(年份)，并保留这些年份的所有网格数据
    最终结果：获得选中网格×选中年份的交集样本
    
    参数:
    X_samples_df: DataFrame格式的样本数据
    sample_size: 目标样本数量
    h3_col: H3索引列名
    year_col: 年份列名
    spatial_coverage: 空间覆盖率参数（可选）
    random_state: 随机种子
    
    返回:
    DataFrame: 采样后的数据
    """
    # 检查是否同时存在空间和时间列
    has_h3 = h3_col in X_samples_df.columns
    has_year = year_col in X_samples_df.columns
    
    if has_h3 and has_year:
        # 使用两阶段分层采样策略（先空间后时间）
        print("使用两阶段时空分层采样...")
        
        # 获取基本信息
        unique_h3 = X_samples_df[h3_col].unique()
        unique_years = X_samples_df[year_col].unique()
        n_h3 = len(unique_h3)
        n_years = len(unique_years)
        
        # ⚡ 使用传入的空间覆盖率参数，优先级最高
        if spatial_coverage is not None:
            target_spatial_coverage = spatial_coverage
            print(f"使用指定的空间覆盖率: {target_spatial_coverage*100:.1f}%")
        else:
            # 默认策略（如果没有指定覆盖率）
            if sample_size >= 10000:  # 大数据集（如res7）
                target_spatial_coverage = 0.25  # 25%空间覆盖率
            elif sample_size >= 2000:  # 中等数据集（如res6）
                target_spatial_coverage = 0.35  # 35%空间覆盖率
            else:  # 小数据集（如res5）
                target_spatial_coverage = 0.50  # 50%空间覆盖率
            print(f"使用默认空间覆盖率: {target_spatial_coverage*100:.1f}%")
        
        # 第一阶段：选择H3网格数量（基于空间覆盖率）
        target_h3_count = int(n_h3 * target_spatial_coverage)
        selected_h3_count = min(max(target_h3_count, 10), n_h3)  # 至少10个，最多全部
        
        # 第二阶段：保持全部年份覆盖（25年，2000-2024）
        selected_years_count = n_years  # 使用全部年份，保持完整时间覆盖
        
        print(f"选择策略: {selected_h3_count}网格 × {selected_years_count}年份 = {selected_h3_count * selected_years_count}样本")
        
        try:
            # 第一阶段：随机选择H3网格
            np.random.seed(random_state)
            selected_h3_values = np.random.choice(unique_h3, size=selected_h3_count, replace=False)
            
            # 第二阶段：随机选择年份
            np.random.seed(random_state + 1)  # 不同的随机种子
            selected_years = np.random.choice(unique_years, size=selected_years_count, replace=False)
            
            # 获取交集样本：选中网格 AND 选中年份
            mask_h3 = X_samples_df[h3_col].isin(selected_h3_values)
            mask_year = X_samples_df[year_col].isin(selected_years)
            sampled_df = X_samples_df[mask_h3 & mask_year].copy()
            
            # 🔥 不再进行子采样限制，让实际样本量自然产生
            # 实际样本量 = 选中网格数 × 25年（2000-2024） × 每个网格每年的平均记录数
                
        except Exception as e:
            print(f"时空分层采样失败: {str(e)}")
            print("回退到随机采样...")
            # 随机采样
            indices = np.random.RandomState(random_state).choice(len(X_samples_df), size=min(sample_size, len(X_samples_df)), replace=False)
            sampled_df = X_samples_df.iloc[indices]
    
    elif has_h3:
        # 仅空间分层采样
        print("使用空间分层采样...")
        n_spatial_classes = X_samples_df[h3_col].nunique()
        
        # 计算需要选择的网格数量
        samples_per_h3 = len(X_samples_df) // n_spatial_classes
        selected_h3_count = min(max(sample_size // samples_per_h3, 1), n_spatial_classes)
        
        try:
            # 随机选择H3网格
            selected_h3_values = np.random.RandomState(random_state).choice(
                X_samples_df[h3_col].unique(), 
                size=selected_h3_count, 
                replace=False
            )
            sampled_df = X_samples_df[X_samples_df[h3_col].isin(selected_h3_values)].copy()
            
            # 如果样本量超过目标，进行随机子采样
            if len(sampled_df) > sample_size:
                indices = np.random.RandomState(random_state + 1).choice(
                    len(sampled_df), size=sample_size, replace=False
                )
                sampled_df = sampled_df.iloc[indices].copy()
            
        except Exception as e:
            print(f"空间分层采样失败: {str(e)}")
            print("回退到随机采样...")
            # 回退到随机采样
            sampled_df = X_samples_df.sample(n=sample_size, random_state=random_state)
    
    elif has_year:
        # 仅时间分层采样
        print("使用时间分层采样...")
        try:
            # 获取年份信息
            years = X_samples_df[year_col].round().astype(int).unique()
            n_years = len(years)
            
            # 计算需要选择的年份数量
            samples_per_year = len(X_samples_df) // n_years
            selected_years_count = min(max(sample_size // samples_per_year, 1), n_years)
            
            # 随机选择年份
            selected_years = np.random.RandomState(random_state).choice(
                years, 
                size=selected_years_count, 
                replace=False
            )
            sampled_df = X_samples_df[X_samples_df[year_col].round().astype(int).isin(selected_years)].copy()
            
            # 如果样本量超过目标，进行随机子采样
            if len(sampled_df) > sample_size:
                indices = np.random.RandomState(random_state + 1).choice(
                    len(sampled_df), size=sample_size, replace=False
                )
                sampled_df = sampled_df.iloc[indices].copy()
            
        except Exception as e:
            print(f"时间分层采样失败: {str(e)}")
            print("回退到随机采样...")
            # 回退到随机采样
            sampled_df = X_samples_df.sample(n=sample_size, random_state=random_state)
    else:
        # 没有合适的分层标签，使用随机采样
        print("使用随机采样...")
        sampled_df = X_samples_df.sample(n=sample_size, random_state=random_state)
    
    return sampled_df


def sample_data_for_testing(df, sample_rate=0.1, h3_col='h3_index', year_col='year', min_samples_per_h3=2, res_level=None, seed=42):
    """
    对数据集进行两阶段时空分层采样，保留空间和时间分布特性
    使用与GeoShapley分析相同的分层采样策略
    
    参数:
    df (DataFrame): 原始数据集
    sample_rate (float): 采样比例，默认0.1表示保留10%的数据
    h3_col (str): H3网格索引列名
    year_col (str): 年份列名
    min_samples_per_h3 (int): 每个H3网格至少保留的样本数
    res_level (str): 分辨率级别, 例如'res5'
    seed (int): 随机种子
    
    返回:
    DataFrame: 采样后的数据集
    """
    # 设置随机种子
    np.random.seed(seed)
    
    # 计算初始目标样本量
    raw_sample_size = int(len(df) * sample_rate)
    
    # 根据分辨率自动调整样本大小 - 采用与GeoShapley相同的策略
    sample_size = raw_sample_size
    if res_level is not None:
        if res_level == 'res7':
            # res7分辨率：最大300行
            sample_size = min(raw_sample_size, 300)
        elif res_level == 'res6':
            # res6分辨率：最大150行
            sample_size = min(raw_sample_size, 150)
        elif res_level == 'res5':
            # res5分辨率：最大80行
            sample_size = min(raw_sample_size, 80)
        
        if sample_size != raw_sample_size:
            print(f"{res_level}分辨率: 自动设置采样目标为{sample_size}行 (优化后)")
    
    # 记录原始数据信息
    orig_size = len(df)
    orig_h3_count = df[h3_col].nunique() if h3_col in df.columns else 0
    orig_year_count = df[year_col].nunique() if year_col in df.columns else 0
    
    print(f"原始数据: {orig_size:,}行 ({orig_h3_count}网格 × {orig_year_count}年份)")
    print(f"目标采样: {sample_size}行 (约{sample_size/orig_size*100:.1f}%)")
    
    # 执行时空分层采样
    sampled_df = perform_spatiotemporal_sampling(
        df, sample_size, h3_col=h3_col, year_col=year_col, 
        spatial_coverage=None,  # 使用默认策略
        random_state=seed
    )
    
    # 记录采样后信息
    sampled_size = len(sampled_df)
    sampled_h3_count = sampled_df[h3_col].nunique() if h3_col in sampled_df.columns else 0
    sampled_year_count = sampled_df[year_col].nunique() if year_col in sampled_df.columns else 0
    
    print(f"采样结果: {sampled_size}行 ({sampled_h3_count}网格 × {sampled_year_count}年份) - 实际采样率{sampled_size/orig_size*100:.1f}%")
    
    return sampled_df 
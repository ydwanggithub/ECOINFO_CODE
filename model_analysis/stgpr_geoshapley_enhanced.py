#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ST-GPR GeoShapley增强模块 - 实现六分法分解

⚠️ 重要说明：
============
这是一个实验性的增强模块，为未来的深入分析而准备。
当前ST-GPR项目的标准分析流程不依赖此模块。
此模块提供了更细粒度的SHAP值分解方法，但计算成本较高。

使用场景：
- 需要深入理解时空交互效应时
- 研究特征作用的空间异质性时
- 探索时间维度对特征重要性的调节作用时

注意：此模块的功能尚未集成到主分析流程中，使用时需要单独调用。

将SHAP值分解为六个组成部分：
1. primary: 主要特征效应（特征的直接贡献）
2. geo: 地理位置效应（纯位置的贡献）
3. temporal: 时间效应（纯时间的贡献）
4. geo_feature_interaction: 地理×特征交互效应（位置如何调节特征的影响）
5. temporal_feature_interaction: 时间×特征交互效应（时间如何调节特征的影响）
6. geo_temporal_interaction: 地理×时间交互效应（位置如何调节时间的影响）
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union


def decompose_stgpr_shap_values(
    shap_values: np.ndarray, 
    feature_names: List[str], 
    geo_feature_names: List[str] = ['latitude', 'longitude', 'GEO'],
    temporal_feature_names: List[str] = ['year', 'YEAR', 'time']
) -> Dict:
    """
    将ST-GPR的SHAP值分解为六个组成部分
    
    参数:
    shap_values: numpy数组，形状为(n_samples, n_features)
    feature_names: 特征名称列表
    geo_feature_names: 地理特征的名称
    temporal_feature_names: 时间特征的名称
    
    返回:
    dict: 包含六个分解部分的字典
    """
    n_samples, n_features = shap_values.shape
    
    # 识别三类特征的索引
    geo_indices = []
    temporal_indices = []
    feature_indices = []
    
    for i, feat in enumerate(feature_names):
        if feat in geo_feature_names:
            geo_indices.append(i)
        elif feat in temporal_feature_names:
            temporal_indices.append(i)
        else:
            feature_indices.append(i)
    
    # 初始化结果
    result = {
        # 主效应
        'primary': None,
        'geo': None,
        'temporal': None,
        # 交互效应
        'geo_feature_interaction': None,
        'temporal_feature_interaction': None,
        'geo_temporal_interaction': None,
        # 元信息
        'feature_names': {
            'primary': [feature_names[i] for i in feature_indices],
            'geo': [feature_names[i] for i in geo_indices],
            'temporal': [feature_names[i] for i in temporal_indices]
        },
        'indices': {
            'primary': feature_indices,
            'geo': geo_indices,
            'temporal': temporal_indices
        }
    }
    
    # 简单分解（第一步：分离主效应）
    if len(feature_indices) > 0:
        result['primary'] = shap_values[:, feature_indices]
    else:
        result['primary'] = np.zeros((n_samples, 0))
    
    if len(geo_indices) > 0:
        result['geo'] = shap_values[:, geo_indices].sum(axis=1)
    else:
        result['geo'] = np.zeros(n_samples)
    
    if len(temporal_indices) > 0:
        result['temporal'] = shap_values[:, temporal_indices].sum(axis=1)
    else:
        result['temporal'] = np.zeros(n_samples)
    
    # 初始化交互效应（需要更复杂的计算来估计）
    n_primary = len(feature_indices)
    result['geo_feature_interaction'] = np.zeros((n_samples, n_primary))
    result['temporal_feature_interaction'] = np.zeros((n_samples, n_primary))
    result['geo_temporal_interaction'] = np.zeros(n_samples)
    
    return result


def estimate_spatiotemporal_interactions(
    model_dict: Dict,
    X_samples: Union[pd.DataFrame, np.ndarray],
    shap_decomposition: Dict,
    n_permutations: int = 10,
    sample_size: int = 100,
    random_state: int = 42
) -> Dict:
    """
    估计时空交互效应
    
    通过置换方法估计三种交互效应：
    1. 地理×特征交互
    2. 时间×特征交互  
    3. 地理×时间交互
    """
    from .stgpr_io import predict_with_st_gpr
    
    np.random.seed(random_state)
    
    # 获取索引
    geo_indices = shap_decomposition['indices']['geo']
    temporal_indices = shap_decomposition['indices']['temporal']
    feature_indices = shap_decomposition['indices']['primary']
    
    # 准备数据
    if isinstance(X_samples, pd.DataFrame):
        X_array = X_samples.values
    else:
        X_array = X_samples
    
    # 限制样本大小
    if len(X_array) > sample_size:
        sample_indices = np.random.choice(len(X_array), sample_size, replace=False)
        X_subset = X_array[sample_indices]
    else:
        X_subset = X_array
        sample_indices = np.arange(len(X_array))
    
    # 原始预测
    y_original = predict_with_st_gpr(model_dict, X_subset, return_variance=False)
    
    # 1. 估计地理×特征交互效应
    geo_feature_interactions = np.zeros((len(sample_indices), len(feature_indices)))
    
    for perm in range(n_permutations):
        X_geo_permuted = X_subset.copy()
        # 置换地理特征
        for geo_idx in geo_indices:
            X_geo_permuted[:, geo_idx] = np.random.permutation(X_geo_permuted[:, geo_idx])
        
        # 预测并计算差异
        y_geo_permuted = predict_with_st_gpr(model_dict, X_geo_permuted, return_variance=False)
        
        # 对每个特征计算交互效应
        for i, feat_idx in enumerate(feature_indices):
            # 创建只改变特定特征的数据
            X_feat_geo_permuted = X_geo_permuted.copy()
            X_feat_geo_permuted[:, feat_idx] = X_subset[:, feat_idx]
            
            y_feat_geo = predict_with_st_gpr(model_dict, X_feat_geo_permuted, return_variance=False)
            
            # 交互效应 = 原始 - 地理置换 - 特征效应 + 双重置换
            interaction = y_original - y_geo_permuted - (y_original - y_feat_geo)
            geo_feature_interactions[:, i] += interaction / n_permutations
    
    # 2. 估计时间×特征交互效应
    temporal_feature_interactions = np.zeros((len(sample_indices), len(feature_indices)))
    
    for perm in range(n_permutations):
        X_temp_permuted = X_subset.copy()
        # 置换时间特征
        for temp_idx in temporal_indices:
            X_temp_permuted[:, temp_idx] = np.random.permutation(X_temp_permuted[:, temp_idx])
        
        y_temp_permuted = predict_with_st_gpr(model_dict, X_temp_permuted, return_variance=False)
        
        for i, feat_idx in enumerate(feature_indices):
            X_feat_temp_permuted = X_temp_permuted.copy()
            X_feat_temp_permuted[:, feat_idx] = X_subset[:, feat_idx]
            
            y_feat_temp = predict_with_st_gpr(model_dict, X_feat_temp_permuted, return_variance=False)
            
            interaction = y_original - y_temp_permuted - (y_original - y_feat_temp)
            temporal_feature_interactions[:, i] += interaction / n_permutations
    
    # 3. 估计地理×时间交互效应
    geo_temporal_interactions = np.zeros(len(sample_indices))
    
    for perm in range(n_permutations):
        # 同时置换地理和时间特征
        X_geo_temp_permuted = X_subset.copy()
        
        for geo_idx in geo_indices:
            X_geo_temp_permuted[:, geo_idx] = np.random.permutation(X_geo_temp_permuted[:, geo_idx])
        
        for temp_idx in temporal_indices:
            X_geo_temp_permuted[:, temp_idx] = np.random.permutation(X_geo_temp_permuted[:, temp_idx])
        
        y_geo_temp_permuted = predict_with_st_gpr(model_dict, X_geo_temp_permuted, return_variance=False)
        
        # 地理×时间交互 = 完整效应 - 地理主效应 - 时间主效应
        interaction = y_original - y_geo_temp_permuted
        geo_temporal_interactions += interaction / n_permutations
    
    # 扩展到完整样本（如果使用了子采样）
    if len(sample_indices) < len(X_samples):
        full_geo_feat = np.zeros((len(X_samples), len(feature_indices)))
        full_temp_feat = np.zeros((len(X_samples), len(feature_indices)))
        full_geo_temp = np.zeros(len(X_samples))
        
        full_geo_feat[sample_indices] = geo_feature_interactions
        full_temp_feat[sample_indices] = temporal_feature_interactions
        full_geo_temp[sample_indices] = geo_temporal_interactions
        
        return {
            'geo_feature_interaction': full_geo_feat,
            'temporal_feature_interaction': full_temp_feat,
            'geo_temporal_interaction': full_geo_temp
        }
    
    return {
        'geo_feature_interaction': geo_feature_interactions,
        'temporal_feature_interaction': temporal_feature_interactions,
        'geo_temporal_interaction': geo_temporal_interactions
    }


def create_six_way_summary(
    shap_decomposition: Dict,
    X_samples: Optional[pd.DataFrame] = None
) -> Dict:
    """
    创建六分法的统计摘要
    """
    summary = {
        'primary_effects': {},
        'spatial_effects': {},
        'temporal_effects': {},
        'interaction_effects': {},
        'decomposition_stats': {}
    }
    
    # 1. 主效应统计
    primary = shap_decomposition['primary']
    if primary.shape[1] > 0:
        primary_importance = np.abs(primary).mean(axis=0)
        feature_names = shap_decomposition['feature_names']['primary']
        
        for i, name in enumerate(feature_names):
            summary['primary_effects'][name] = {
                'mean_abs_effect': primary_importance[i],
                'std': np.std(primary[:, i]),
                'min': np.min(primary[:, i]),
                'max': np.max(primary[:, i])
            }
    
    # 2. 空间效应统计
    geo = shap_decomposition['geo']
    summary['spatial_effects'] = {
        'mean': np.mean(geo),
        'std': np.std(geo),
        'min': np.min(geo),
        'max': np.max(geo),
        'spatial_variance_contribution': np.var(geo)
    }
    
    # 3. 时间效应统计
    temporal = shap_decomposition['temporal']
    summary['temporal_effects'] = {
        'mean': np.mean(temporal),
        'std': np.std(temporal),
        'min': np.min(temporal),
        'max': np.max(temporal),
        'temporal_variance_contribution': np.var(temporal)
    }
    
    # 4. 交互效应统计
    geo_feat_inter = shap_decomposition['geo_feature_interaction']
    temp_feat_inter = shap_decomposition['temporal_feature_interaction']
    geo_temp_inter = shap_decomposition['geo_temporal_interaction']
    
    if geo_feat_inter.shape[1] > 0:
        geo_feat_importance = np.abs(geo_feat_inter).mean(axis=0)
        temp_feat_importance = np.abs(temp_feat_inter).mean(axis=0)
        
        for i, name in enumerate(feature_names):
            summary['interaction_effects'][name] = {
                'geo_interaction': geo_feat_importance[i],
                'temporal_interaction': temp_feat_importance[i],
                'total_interaction': geo_feat_importance[i] + temp_feat_importance[i]
            }
    
    summary['interaction_effects']['geo_temporal'] = {
        'mean': np.mean(geo_temp_inter),
        'std': np.std(geo_temp_inter),
        'strength': np.abs(geo_temp_inter).mean()
    }
    
    # 5. 分解统计
    total_variance = (
        np.var(primary) + 
        np.var(geo) + 
        np.var(temporal) +
        np.var(geo_feat_inter) +
        np.var(temp_feat_inter) +
        np.var(geo_temp_inter)
    )
    
    summary['decomposition_stats'] = {
        'primary_contribution': np.var(primary) / total_variance if total_variance > 0 else 0,
        'spatial_contribution': np.var(geo) / total_variance if total_variance > 0 else 0,
        'temporal_contribution': np.var(temporal) / total_variance if total_variance > 0 else 0,
        'interaction_contribution': (
            np.var(geo_feat_inter) + np.var(temp_feat_inter) + np.var(geo_temp_inter)
        ) / total_variance if total_variance > 0 else 0
    }
    
    return summary


def visualize_six_way_decomposition(
    shap_decomposition: Dict,
    feature_name: str,
    X_samples: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    为特定特征创建六分法可视化
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    feature_names = shap_decomposition['feature_names']['primary']
    if feature_name not in feature_names:
        raise ValueError(f"特征 {feature_name} 不在特征列表中")
    
    feat_idx = feature_names.index(feature_name)
    
    # 创建图表布局
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. 主效应
    ax1 = fig.add_subplot(gs[0, 0])
    primary_values = shap_decomposition['primary'][:, feat_idx]
    feature_values = X_samples[feature_name].values
    
    ax1.scatter(feature_values, primary_values, alpha=0.5, s=20)
    ax1.set_xlabel(feature_name)
    ax1.set_ylabel('Primary SHAP value')
    ax1.set_title(f'{feature_name} - Primary Effect')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 2. 地理效应（空间分布）
    ax2 = fig.add_subplot(gs[0, 1])
    if 'latitude' in X_samples.columns and 'longitude' in X_samples.columns:
        scatter = ax2.scatter(
            X_samples['longitude'], 
            X_samples['latitude'],
            c=shap_decomposition['geo'],
            cmap='RdBu_r',
            alpha=0.6,
            s=20
        )
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title('Geographic Effect Distribution')
        plt.colorbar(scatter, ax=ax2)
    
    # 3. 时间效应
    ax3 = fig.add_subplot(gs[0, 2])
    if 'year' in X_samples.columns:
        years = X_samples['year'].values
        temporal_values = shap_decomposition['temporal']
        
        # 按年份聚合
        year_means = pd.DataFrame({
            'year': years,
            'temporal': temporal_values
        }).groupby('year')['temporal'].mean()
        
        ax3.plot(year_means.index, year_means.values, marker='o')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Temporal Effect')
        ax3.set_title('Temporal Effect Trend')
        ax3.grid(True, alpha=0.3)
    
    # 4. 地理×特征交互
    ax4 = fig.add_subplot(gs[1, 0])
    geo_feat_inter = shap_decomposition['geo_feature_interaction'][:, feat_idx]
    ax4.scatter(feature_values, geo_feat_inter, alpha=0.5, s=20, color='green')
    ax4.set_xlabel(feature_name)
    ax4.set_ylabel('Geo×Feature Interaction')
    ax4.set_title(f'{feature_name} - Geographic Interaction')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 5. 时间×特征交互
    ax5 = fig.add_subplot(gs[1, 1])
    temp_feat_inter = shap_decomposition['temporal_feature_interaction'][:, feat_idx]
    ax5.scatter(feature_values, temp_feat_inter, alpha=0.5, s=20, color='orange')
    ax5.set_xlabel(feature_name)
    ax5.set_ylabel('Time×Feature Interaction')
    ax5.set_title(f'{feature_name} - Temporal Interaction')
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 6. 地理×时间交互
    ax6 = fig.add_subplot(gs[1, 2])
    geo_temp_inter = shap_decomposition['geo_temporal_interaction']
    if 'year' in X_samples.columns:
        ax6.scatter(years, geo_temp_inter, alpha=0.5, s=20, color='purple')
        ax6.set_xlabel('Year')
        ax6.set_ylabel('Geo×Time Interaction')
        ax6.set_title('Geographic-Temporal Interaction')
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 7. 效应分解饼图
    ax7 = fig.add_subplot(gs[2, :])
    
    # 计算各部分的贡献
    primary_contrib = np.abs(primary_values).mean()
    geo_contrib = np.abs(shap_decomposition['geo']).mean()
    temp_contrib = np.abs(shap_decomposition['temporal']).mean()
    geo_feat_contrib = np.abs(geo_feat_inter).mean()
    temp_feat_contrib = np.abs(temp_feat_inter).mean()
    geo_temp_contrib = np.abs(geo_temp_inter).mean()
    
    contributions = [
        primary_contrib,
        geo_contrib,
        temp_contrib,
        geo_feat_contrib,
        temp_feat_contrib,
        geo_temp_contrib
    ]
    
    labels = [
        f'Primary\n({primary_contrib:.3f})',
        f'Geographic\n({geo_contrib:.3f})',
        f'Temporal\n({temp_contrib:.3f})',
        f'Geo×Feature\n({geo_feat_contrib:.3f})',
        f'Time×Feature\n({temp_feat_contrib:.3f})',
        f'Geo×Time\n({geo_temp_contrib:.3f})'
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 创建条形图而不是饼图（更适合比较）
    bars = ax7.bar(range(6), contributions, color=colors)
    ax7.set_xticks(range(6))
    ax7.set_xticklabels(labels, rotation=45, ha='right')
    ax7.set_ylabel('Mean Absolute SHAP Value')
    ax7.set_title(f'{feature_name} - Six-Way Effect Decomposition')
    
    # 添加数值标签
    for bar, contrib in zip(bars, contributions):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{contrib:.3f}', ha='center', va='bottom')
    
    plt.suptitle(f'Six-Way Decomposition Analysis for {feature_name}', fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        # plt.show()  # 禁用显示，只保存到文件
        plt.close()  # 确保图形被关闭
    
    return fig


def analyze_spatiotemporal_heterogeneity(
    shap_decomposition: Dict,
    X_samples: pd.DataFrame,
    n_spatial_clusters: int = 5,
    n_temporal_periods: int = 3
) -> Dict:
    """
    分析时空异质性：识别哪些特征的作用在空间和时间上变化最大
    """
    from sklearn.cluster import KMeans
    
    results = {
        'spatial_heterogeneity': {},
        'temporal_heterogeneity': {},
        'spatiotemporal_patterns': {}
    }
    
    # 1. 空间异质性分析
    if 'latitude' in X_samples.columns and 'longitude' in X_samples.columns:
        coords = X_samples[['latitude', 'longitude']].values
        spatial_clusters = KMeans(n_clusters=n_spatial_clusters, random_state=42).fit_predict(coords)
        
        primary = shap_decomposition['primary']
        feature_names = shap_decomposition['feature_names']['primary']
        
        for i, feat_name in enumerate(feature_names):
            # 计算每个空间聚类中的平均SHAP值
            cluster_means = []
            for cluster in range(n_spatial_clusters):
                mask = spatial_clusters == cluster
                cluster_mean = primary[mask, i].mean()
                cluster_means.append(cluster_mean)
            
            # 空间异质性 = 聚类间的标准差
            spatial_heterogeneity = np.std(cluster_means)
            results['spatial_heterogeneity'][feat_name] = {
                'heterogeneity_score': spatial_heterogeneity,
                'cluster_means': cluster_means
            }
    
    # 2. 时间异质性分析
    if 'year' in X_samples.columns:
        years = X_samples['year'].values
        year_range = years.max() - years.min()
        period_length = year_range / n_temporal_periods
        
        temporal_periods = np.floor((years - years.min()) / period_length).astype(int)
        temporal_periods = np.clip(temporal_periods, 0, n_temporal_periods - 1)
        
        for i, feat_name in enumerate(feature_names):
            # 计算每个时间段的平均SHAP值
            period_means = []
            for period in range(n_temporal_periods):
                mask = temporal_periods == period
                if mask.sum() > 0:
                    period_mean = primary[mask, i].mean()
                    period_means.append(period_mean)
            
            # 时间异质性 = 时间段间的标准差
            temporal_heterogeneity = np.std(period_means) if period_means else 0
            results['temporal_heterogeneity'][feat_name] = {
                'heterogeneity_score': temporal_heterogeneity,
                'period_means': period_means
            }
    
    # 3. 识别时空模式
    for feat_name in feature_names:
        spatial_score = results['spatial_heterogeneity'].get(feat_name, {}).get('heterogeneity_score', 0)
        temporal_score = results['temporal_heterogeneity'].get(feat_name, {}).get('heterogeneity_score', 0)
        
        # 分类特征的时空模式
        if spatial_score > np.median([v['heterogeneity_score'] for v in results['spatial_heterogeneity'].values()]):
            if temporal_score > np.median([v['heterogeneity_score'] for v in results['temporal_heterogeneity'].values()]):
                pattern = 'spatiotemporal_varying'  # 时空都变化
            else:
                pattern = 'spatial_varying'  # 仅空间变化
        else:
            if temporal_score > np.median([v['heterogeneity_score'] for v in results['temporal_heterogeneity'].values()]):
                pattern = 'temporal_varying'  # 仅时间变化
            else:
                pattern = 'stable'  # 稳定
        
        results['spatiotemporal_patterns'][feat_name] = {
            'pattern': pattern,
            'spatial_score': spatial_score,
            'temporal_score': temporal_score
        }
    
    return results 
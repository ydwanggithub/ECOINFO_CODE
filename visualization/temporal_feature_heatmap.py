#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
时序特征热图模块: 展示特征重要性随时间的变化模式

该模块为ST-GPR模型创建时序特征热图，展示不同年份数据在统一模型中的
特征重要性模式。每个分辨率（res7/res6/res5）对应一个热图。

注意：这些热图展示的是"统一ST-GPR模型对不同时期数据的解释"，
而不是"每个时期独立的特征-目标关系"。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional

from .base import enhance_plot_style, save_plot_for_publication, ensure_dir_exists, color_map
from .utils import simplify_feature_name_for_plot
from .utils import ensure_spatiotemporal_features

__all__ = ['plot_temporal_feature_heatmap']


def calculate_temporal_shap_values(results_by_resolution: Dict, 
                                 years: Optional[List[int]] = None) -> Dict:
    """
    计算每个年份的平均SHAP值
    
    优化策略：
    1. 优先使用插值后的完整网格SHAP值计算时序模式
    2. 如果插值不可用，回退到使用原始采样数据
    3. 确保与其他SHAP图表保持一致的数据基础
    
    只包含主效应和GEO效应，排除交互效应
    
    参数:
    - results_by_resolution: 包含各分辨率模型结果的字典
    - years: 要分析的年份列表，默认为2000-2024（包含时间外推数据）
    
    返回:
    - temporal_shap_dict: 包含各分辨率时序SHAP值的字典
    """
    if years is None:
        years = list(range(2000, 2025))  # 🔄 更新：包含时间外推数据2000-2024年
    
    print("  🔧 尝试使用插值后的完整网格数据计算时序SHAP值...")
    
    # 尝试使用插值后的完整网格数据
    # 🔥 修复：直接使用原始SHAP值，无需插值或数据增强
    print(f"\n  📊 使用原始SHAP值进行时序分析（保证数据真实性）...")
    
    # 🔥 修复：优先使用原始SHAP值，而不是插值或聚合后的数据
    print(f"  ✅ 使用原始SHAP值进行时序分析（确保数据真实性）")
    final_results = results_by_resolution
    data_source_info = "Original SHAP Values"
    
    temporal_shap_dict = {}
    
    for res, res_data in final_results.items():
        # 🔥 修复：直接使用原始SHAP数据，不使用插值后的数据
        shap_values_by_feature = res_data.get('shap_values_by_feature')
        X_sample = res_data.get('X_sample')
        print(f"    {res}: 使用原始SHAP值 (样本数: {len(X_sample) if X_sample is not None else 'N/A'})")
        
        if shap_values_by_feature is None or X_sample is None:
            print(f"警告: {res}缺少必要的SHAP数据")
            continue
        
        # 从shap_values_by_feature获取特征名称（不包括year和交互效应）
        all_feature_keys = list(shap_values_by_feature.keys())
        feature_names = []
        excluded_features = []
        
        for f in all_feature_keys:
            # 更精确的过滤条件：排除year和任何交互效应特征
            if (f != 'year' and 
                '×' not in f and 
                ' x ' not in f and 
                '_x_' not in f and
                'interaction' not in f.lower()):
                # 只包含主效应和GEO效应
                feature_names.append(f)
            else:
                excluded_features.append(f)
        
        print(f"    📊 {res}原始特征数量: {len(shap_values_by_feature)}个")
        print(f"    📊 {res}过滤后特征数量: {len(feature_names)}个")
        print(f"    📊 {res}包含的特征: {feature_names}")
        print(f"    📊 {res}排除的特征: {excluded_features}")
        
        if not feature_names:
            print(f"警告: {res}没有有效的特征")
            continue
        
        # 确保X_sample包含year列
        if isinstance(X_sample, pd.DataFrame) and 'year' in X_sample.columns:
            # 🔧 修复：获取SHAP值的样本数量，确保数据维度一致
            n_shap_samples = len(next(iter(shap_values_by_feature.values())))
            print(f"    🔧 {res}: SHAP值数组长度: {n_shap_samples}")
            print(f"    🔧 {res}: X_sample年份数据长度: {len(X_sample)}")
            
            # 🛡️ 安全处理：确保年份数据与SHAP值长度匹配
            if len(X_sample) >= n_shap_samples:
                # X_sample长度大于等于SHAP样本数，取前n_shap_samples个
                year_data = X_sample['year'].iloc[:n_shap_samples]
                print(f"    ✅ {res}: 使用X_sample的前{n_shap_samples}个年份数据")
            else:
                # X_sample长度小于SHAP样本数，需要重复或填充
                year_data = X_sample['year']
                # 如果SHAP样本数是X_sample的整数倍，重复年份数据
                repeat_factor = n_shap_samples // len(X_sample)
                if n_shap_samples % len(X_sample) == 0 and repeat_factor > 1:
                    year_data = pd.concat([year_data] * repeat_factor, ignore_index=True)
                    print(f"    🔄 {res}: 重复年份数据{repeat_factor}次以匹配SHAP样本数")
                else:
                    # 使用最近的年份值填充剩余部分
                    remaining = n_shap_samples - len(X_sample)
                    last_year = X_sample['year'].iloc[-1]
                    additional_years = pd.Series([last_year] * remaining)
                    year_data = pd.concat([X_sample['year'], additional_years], ignore_index=True)
                    print(f"    🔧 {res}: 填充{remaining}个年份值({last_year})以匹配SHAP样本数")
            
            # 最终验证
            if len(year_data) != n_shap_samples:
                print(f"    ❌ {res}: 年份数据长度仍不匹配，跳过该分辨率")
                continue
            else:
                print(f"    ✅ {res}: 年份数据长度匹配: {len(year_data)} = {n_shap_samples}")
                
        else:
            print(f"警告: {res}的数据中缺少year列")
            continue
        
        # 初始化时序SHAP矩阵
        n_features = len(feature_names)
        n_years = len(years)
        temporal_shap_matrix = np.zeros((n_features, n_years))
        
        # 计算每个年份的平均GeoShapley值（保持正负号）
        for year_idx, year in enumerate(years):
            year_mask = (year_data == year)
            if np.any(year_mask):
                # 对每个特征计算该年份的平均GeoShapley值（包含正负号）
                for feat_idx, feat_name in enumerate(feature_names):
                    if feat_name in shap_values_by_feature:
                        year_shap = np.array(shap_values_by_feature[feat_name])[year_mask]
                        temporal_shap_matrix[feat_idx, year_idx] = np.mean(year_shap)  # 移除abs()以保持正负号
        
        # 保存结果
        temporal_shap_dict[res] = {
            'matrix': temporal_shap_matrix,
            'features': feature_names,
            'years': years,
            'data_source': data_source_info if 'enhanced_shap_values_by_feature' in res_data else "Sampled Data"
        }
    
    return temporal_shap_dict


def create_bivariate_colorbar(cax, ax, res):
    """
    创建双变量颜色条：展示时间变化(RdBu_r色彩)和重要性(强度)的组合
    """
    import matplotlib.cm as cm
    
    # 清空颜色条轴
    cax.clear()
    
    # 获取RdBu_r colormap的实际红色和蓝色
    rdbu_cmap = cm.get_cmap('RdBu_r')
    red_color = np.array(rdbu_cmap(1.0)[:3])    # 获取最红色 (正值)
    blue_color = np.array(rdbu_cmap(0.0)[:3])   # 获取最蓝色 (负值)
    white_color = np.array([1.0, 1.0, 1.0])    # 白色 (零值)
    
    # 创建小型的双变量色彩矩阵用于图例
    n_temporal = 20  # 时间变化步数
    n_importance = 10  # 重要性步数
    
    legend_rgb = np.zeros((n_importance, n_temporal, 3))
    
    for imp_idx in range(n_importance):
        for temp_idx in range(n_temporal):
            # 时间变化：从-1(蓝)到+1(红)
            temporal_value = (temp_idx / (n_temporal - 1)) * 2 - 1  # [-1, +1]
            
            # 使用RdBu_r colormap + 重要性饱和度梯度
            # temporal_value范围[-1, +1] 映射到 RdBu_r[0, 1]
            colormap_position = (temporal_value + 1) / 2  # 转换为[0, 1]范围
            base_color = np.array(rdbu_cmap(colormap_position)[:3])
            
            # 重要性：从低到高饱和度
            saturation_weight = 0.4 + 0.6 * (imp_idx / (n_importance - 1))  # [0.4, 1.0]
            
            # 应用饱和度调整：通过与白色混合来降低饱和度
            white_color = np.array([1.0, 1.0, 1.0])
            final_color = base_color * saturation_weight + white_color * (1 - saturation_weight)
            
            # 确保RGB值在[0,1]范围内
            final_color = np.clip(final_color, 0, 1)
            legend_rgb[n_importance - 1 - imp_idx, temp_idx, :] = final_color  # 翻转Y轴让高重要性在上
    
    # 在颜色条轴中显示图例
    cax.imshow(legend_rgb, aspect='auto', extent=[-1, 1, 0, 1])
    
    # 添加极简标签，调整间距使其更紧凑
    cax.set_ylabel('Importance\n(Saturation)', fontsize=10, fontweight='bold', rotation=90, va='center', labelpad=-10)
    cax.set_xlabel('Temporal', fontsize=10, fontweight='bold', ha='center', labelpad=3)
    
    # 将Y轴标签移至右侧
    cax.yaxis.set_label_position('right')
    cax.yaxis.tick_right()
    
    # 设置简化刻度
    cax.set_xticks([-1, 0, 1])
    cax.set_xticklabels(['-', '0', '+'], fontsize=9, fontweight='bold')
    cax.set_yticks([0, 1])
    cax.set_yticklabels(['Low', 'High'], fontsize=9, fontweight='bold')
    
    # 美化边框
    for spine in cax.spines.values():
        spine.set_linewidth(1.5)
    
    # 调整刻度参数，使标签更接近colorbar
    cax.tick_params(axis='both', which='major', labelsize=9, width=1.5, length=4, pad=2)
    cax.tick_params(axis='y', which='major', pad=1)  # Y轴标签更紧凑


def plot_temporal_feature_heatmap(results_by_resolution: Dict, 
                                output_dir: Optional[str] = None,
                                top_n_features: int = 15,
                                figsize: Tuple[int, int] = (12, 14),
                                normalization: str = 'log_quantile') -> plt.Figure:
    """
    创建时间特征热力图，展示特征重要性随时间的变化
    
    🔥 修复策略：
    1. 使用原始SHAP值确保数据真实性（不使用插值或聚合数据）
    2. 显示所有主效应和GEO特征（11个环境特征+1个GEO特征=12个）
    3. 提供多种颜色映射归一化方法，平衡不同的显示需求
    4. 保证与其他SHAP图表的数据一致性
    
    Args:
        results_by_resolution: 按分辨率组织的结果字典
        output_dir: 输出目录路径
        top_n_features: 保留参数向后兼容，但实际显示所有主效应和GEO特征
        figsize: 图形大小
        normalization: 归一化方法，可选：
            - 'log_quantile': 对数变换+分位数归一化（推荐，平衡纵横向比较）
            - 'row_normalize': 按行归一化（强调时间变化）
            - 'linear': 线性缩放（保持原始比例，可能被主导特征掩盖）
            - 'percentile': 分位数截断（去除极值影响）
        
    Returns:
        matplotlib.figure.Figure or None
    """
    print("\n🎨 创建时间GeoShapley贡献热力图（使用原始值含正负号）...")
    print(f"  显示所有主效应和GEO特征的时间变化模式（应为12个特征）")
    
    # 准备数据
    temporal_shap_data = calculate_temporal_shap_values(results_by_resolution)
    if not temporal_shap_data:
        print("  ⚠️ 警告: 没有找到有效的时间SHAP数据")
        return None
    
    # 检查数据源信息
    data_source_info = None
    for res_data in temporal_shap_data.values():
        if 'data_source' in res_data:
            data_source_info = res_data['data_source']
            break
    
    # 保存原始rcParams
    original_rcParams = plt.rcParams.copy()
    
    # 创建本地样式字典（参考regionkmeans_plot.py的风格）
    style_dict = {
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'font.weight': 'bold',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'axes.linewidth': 1.5,
        'legend.fontsize': 10,
        'legend.title_fontsize': 11,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'figure.figsize': figsize,
        'figure.constrained_layout.use': False,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.spines.bottom': True,
        'axes.spines.left': True,
    }
    
    # 使用上下文管理器隔离样式设置
    with plt.style.context('default'):
        with plt.rc_context(style_dict):
            
            # 创建图形
            fig = plt.figure(figsize=figsize, dpi=600)
            
            # 添加总标题（已去除数据源后缀）
            # title_suffix = f" ({data_source_info})" if data_source_info else ""
            fig.suptitle('Temporal GeoShapley Contribution Patterns Across Resolutions', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            # 创建GridSpec布局
            gs = gridspec.GridSpec(3, 1, figure=fig, 
                                 height_ratios=[1, 1, 1],
                                 hspace=0.3)
            
            # 分辨率设置
            resolutions = ['res7', 'res6', 'res5']
            res_titles = {
                'res7': 'Resolution 7 (Micro)',
                'res6': 'Resolution 6 (Meso)', 
                'res5': 'Resolution 5 (Macro)'
            }
            
            # 子图标签
            subplot_labels = ['(a)', '(b)', '(c)']
            
            # 创建颜色映射（使用RdBu_r，与SHAP值一致）
            cmap = 'RdBu_r'
            
            # 处理每个分辨率
            for i, res in enumerate(resolutions):
                if res not in temporal_shap_data:
                    # 创建空白子图
                    ax = fig.add_subplot(gs[i])
                    ax.text(0.5, 0.5, f"No data for {res}", 
                           ha='center', va='center', fontsize=14, 
                           transform=ax.transAxes)
                    ax.axis('off')
                    continue
                
                # 获取数据
                data = temporal_shap_data[res]
                shap_matrix = data['matrix']
                features = data['features']
                years = data['years']
                res_data_source = data.get('data_source', 'Unknown')
                
                print(f"    📊 {res}时序热图特征数量: {len(features)}个")
                print(f"    📊 {res}SHAP矩阵形状: {shap_matrix.shape}")
                
                # 🔧 修复：按照特征重要性从上到下排列
                # 计算每个特征在所有年份的平均重要性（使用绝对值排序，但保持原始值的正负号）
                mean_importance = np.mean(np.abs(shap_matrix), axis=1)
                
                # 按重要性降序排列（最重要的在上方）
                sorted_indices = np.argsort(mean_importance)[::-1]
                
                # 重新排列矩阵、特征名称和重要性分数
                final_indices = sorted_indices
                final_shap_matrix = shap_matrix[sorted_indices, :]
                final_features = [features[i] for i in sorted_indices]
                final_mean_importance = mean_importance[sorted_indices]  # 🔥 关键修复：重新排列重要性分数
                
                print(f"    🔧 {res}特征按重要性排序完成:")
                for idx, feat_name in enumerate(final_features[:5]):  # 显示前5个
                    importance = mean_importance[sorted_indices[idx]]
                    print(f"      {idx+1}. {feat_name}: {importance:.4f}")
                if len(final_features) > 5:
                    print(f"      ... (共{len(final_features)}个特征)")
                
                print(f"    📊 {res}最终显示特征数量: {len(final_features)}个")
                
                # 简化特征名称（修复：移除max_length参数）
                simplified_features = [simplify_feature_name_for_plot(f) 
                                     for f in final_features]
                
                # 创建子图
                ax = fig.add_subplot(gs[i])
                
                # 设置轴线宽度
                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)
                
                # 🔥 修复：使用科学的颜色映射方式，类似GIS ArcMap
                # 提供多种归一化方法，平衡不同的显示需求
                
                print(f"    🎨 {res}: 原始数据范围 [{np.min(final_shap_matrix):.4f}, {np.max(final_shap_matrix):.4f}]")
                print(f"    🎨 {res}: 使用归一化方法: {normalization}")
                
                # 根据选择的方法进行归一化
                if normalization == 'log_quantile':
                    # 方法1：对数缩放 + 分位数归一化（推荐）
                    # 既保持特征间相对大小关系，又能显示所有特征的变化
                    
                    # Step 1: 对数变换压缩极值差异
                    epsilon = np.max(final_shap_matrix) * 1e-6
                    log_matrix = np.log10(final_shap_matrix + epsilon)
                    
                    # Step 2: 使用全局分位数归一化
                    flat_values = log_matrix.flatten()
                    percentiles = [5, 25, 50, 75, 95]
                    p5, p25, p50, p75, p95 = np.percentile(flat_values, percentiles)
                    
                    print(f"    🎨 {res}: 对数变换后分位数 p5={p5:.3f}, p50={p50:.3f}, p95={p95:.3f}")
                    
                    # Step 3: 分段线性映射
                    def piecewise_normalize(values, p5, p25, p50, p75, p95):
                        normalized = np.zeros_like(values)
                        mask1 = values <= p25
                        mask2 = (values > p25) & (values <= p50)
                        mask3 = (values > p50) & (values <= p75)
                        mask4 = values > p75
                        
                        normalized[mask1] = 0.2 * (values[mask1] - p5) / (p25 - p5 + 1e-10)
                        normalized[mask2] = 0.2 + 0.3 * (values[mask2] - p25) / (p50 - p25 + 1e-10)
                        normalized[mask3] = 0.5 + 0.3 * (values[mask3] - p50) / (p75 - p50 + 1e-10)
                        normalized[mask4] = 0.8 + 0.2 * (values[mask4] - p75) / (p95 - p75 + 1e-10)
                        
                        return np.clip(normalized, 0, 1)
                    
                    normalized_matrix = piecewise_normalize(log_matrix, p5, p25, p50, p75, p95)
                    colorbar_label = 'Normalized GeoShapley Value'
                    colorbar_ticks = [0.1, 0.35, 0.65, 0.9]
                    colorbar_labels = ['Low\n(≤p25)', 'Medium\n(p25-p50)', 'High\n(p50-p75)', 'Very High\n(>p75)']
                    
                elif normalization == 'row_normalize':
                    # 方法2：分层归一化 - 所有特征可见，但强度不同
                    normalized_matrix = np.zeros_like(final_shap_matrix)
                    
                    # 计算重要性权重（使用反向排名：1/rank）
                    n_features = len(final_features)
                    rank_weights = np.array([1.0 / (i + 1) for i in range(n_features)])  # [1, 1/2, 1/3, ...]
                    # 重要性分层：重要特征用上层色彩范围，不重要特征用下层色彩范围
                    layer_assignments = np.linspace(0.8, 0.2, n_features)  # [0.8, 0.6, 0.4, 0.2] 分层
                    
                    print(f"    🎨 {res}: 特征分层范围 [{np.min(layer_assignments):.2f}, {np.max(layer_assignments):.2f}]")
                    print(f"    🎨 {res}: 前3个特征分层: {layer_assignments[:3]}")
                    
                    for feat_idx in range(final_shap_matrix.shape[0]):
                        row_data = final_shap_matrix[feat_idx, :]
                        row_abs_max = np.max(np.abs(row_data))
                        
                        if row_abs_max > 1e-10:
                            # Step 1: 行归一化保证时间模式可见
                            row_normalized = row_data / row_abs_max  # [-1, +1]
                            
                            # Step 2: 映射到分层色彩范围
                            layer_intensity = layer_assignments[feat_idx]  # 该特征的层级强度
                            
                            # 每个特征在其分配的层级内显示完整的时间模式
                            # 重要特征: [-0.8, +0.8] 范围
                            # 不重要特征: [-0.2, +0.2] 范围
                            normalized_matrix[feat_idx, :] = row_normalized * layer_intensity
                            
                            print(f"      {final_features[feat_idx]}: 排名={feat_idx+1}, 层级强度={layer_intensity:.2f}, 范围=[{-layer_intensity:.2f}, {+layer_intensity:.2f}]")
                        else:
                            normalized_matrix[feat_idx, :] = 0.0
                    
                    # 使用全范围colorbar，但特征在不同层级
                    colorbar_label = 'Normalized GeoShapley Value'
                    colorbar_ticks = [-0.8, -0.4, 0, 0.4, 0.8]
                    colorbar_labels = ['High Neg', 'Low Neg', '0', 'Low Pos', 'High Pos']
                    
                elif normalization == 'percentile':
                    # 方法3：分位数截断（去除极值影响）
                    flat_values = final_shap_matrix.flatten()
                    p5, p95 = np.percentile(flat_values, [5, 95])
                    clipped_matrix = np.clip(final_shap_matrix, p5, p95)
                    normalized_matrix = (clipped_matrix - p5) / (p95 - p5 + 1e-10)
                    
                    colorbar_label = 'Normalized GeoShapley Value'
                    colorbar_ticks = [0, 0.25, 0.5, 0.75, 1.0]
                    colorbar_labels = ['p5', 'p25', 'p50', 'p75', 'p95']
                    
                elif normalization == 'symmetric':
                    # 方法4：对称缩放（保持原始SHAP值的相对关系和正负号）
                    # 这种方法直接使用原始SHAP值，不添加重要性权重
                    abs_max = np.max(np.abs(final_shap_matrix))
                    if abs_max > 0:
                        # 直接按最大绝对值对称缩放，保持零点在中心
                        normalized_matrix = final_shap_matrix / abs_max  # [-1, +1] 范围
                    else:
                        normalized_matrix = np.zeros_like(final_shap_matrix)
                    
                    colorbar_label = 'Normalized GeoShapley Value'
                    colorbar_ticks = [-1, -0.5, 0, 0.5, 1]
                    colorbar_labels = ['-Max', '-50%', '0', '+50%', '+Max']
                    
                elif normalization == 'row_wise':
                    # 方法5：逐行归一化（保持原始排序，显示时间变化模式）
                    # 使用原始SHAP值进行排序，但每行独立归一化以显示时间模式
                    normalized_matrix = np.zeros_like(final_shap_matrix)
                    
                    for feat_idx in range(final_shap_matrix.shape[0]):
                        row_data = final_shap_matrix[feat_idx, :]
                        row_abs_max = np.max(np.abs(row_data))
                        
                        if row_abs_max > 1e-10:
                            # 每行独立归一化，保持正负号和时间变化模式
                            normalized_matrix[feat_idx, :] = row_data / row_abs_max  # [-1, +1]
                        else:
                            normalized_matrix[feat_idx, :] = 0.0
                    
                    colorbar_label = 'Normalized GeoShapley Value'
                    colorbar_ticks = [-1, -0.5, 0, 0.5, 1]
                    colorbar_labels = ['-Max', '-50%', '0', '+50%', '+Max']
                    
                elif normalization == 'importance_weighted':
                    # 方法6：重要性加权的行归一化（显示时间模式+重要性层次）
                    # 既显示时间变化模式，又保持特征间的重要性视觉差异
                    normalized_matrix = np.zeros_like(final_shap_matrix)
                    
                    # 计算重要性权重：基于平均绝对SHAP值
                    importance_scores = final_mean_importance  # 已经按排序后的顺序
                    max_importance = importance_scores[0]  # 最重要特征的分数
                    
                    print(f"    🎨 {res}: 重要性分数范围 [{np.min(importance_scores):.4f}, {np.max(importance_scores):.4f}]")
                    
                    for feat_idx in range(final_shap_matrix.shape[0]):
                        row_data = final_shap_matrix[feat_idx, :]
                        row_abs_max = np.max(np.abs(row_data))
                        
                        if row_abs_max > 1e-10:
                            # Step 1: 行归一化显示时间变化模式
                            row_normalized = row_data / row_abs_max  # [-1, +1]
                            
                            # Step 2: 根据重要性计算显示强度权重（更微妙的差异，保持时间模式突出）
                            importance_ratio = importance_scores[feat_idx] / max_importance
                            # 使用更温和的缩放，强调时间变化而非重要性差异
                            intensity_weight = 0.8 + 0.2 * np.sqrt(importance_ratio)  # [0.8, 1.0] 范围，更微妙的差异
                            
                            # Step 3: 应用重要性权重
                            normalized_matrix[feat_idx, :] = row_normalized * intensity_weight
                            
                            print(f"      {final_features[feat_idx]}: 重要性={importance_scores[feat_idx]:.4f}, 权重={intensity_weight:.2f}, 范围=[{-intensity_weight:.2f}, {+intensity_weight:.2f}]")
                        else:
                            normalized_matrix[feat_idx, :] = 0.0
                    
                    colorbar_label = 'Normalized GeoShapley Value'
                    colorbar_ticks = [-1, -0.5, 0, 0.5, 1]
                    colorbar_labels = ['-Max', '-50%', '0', '+50%', '+Max']
                    
                elif normalization == 'temporal_focus':
                    # 方法7：时间优先归一化（强调时间变化，重要性差异极其微妙）
                    # 最大化时间模式可见性，仅保留极微妙的重要性提示
                    normalized_matrix = np.zeros_like(final_shap_matrix)
                    
                    # 计算重要性权重：基于平均绝对SHAP值
                    importance_scores = final_mean_importance  # 已经按排序后的顺序
                    max_importance = importance_scores[0]  # 最重要特征的分数
                    
                    print(f"    🎨 {res}: 重要性分数范围 [{np.min(importance_scores):.4f}, {np.max(importance_scores):.4f}]")
                    
                    for feat_idx in range(final_shap_matrix.shape[0]):
                        row_data = final_shap_matrix[feat_idx, :]
                        row_abs_max = np.max(np.abs(row_data))
                        
                        if row_abs_max > 1e-10:
                            # Step 1: 行归一化显示时间变化模式（主要效应）
                            row_normalized = row_data / row_abs_max  # [-1, +1]
                            
                            # Step 2: 极微妙的重要性提示（仅5%的差异）
                            importance_ratio = importance_scores[feat_idx] / max_importance
                            # 极小的重要性差异，主要保持时间模式
                            intensity_weight = 0.95 + 0.05 * importance_ratio  # [0.95, 1.0] 范围，极微妙
                            
                            # Step 3: 应用极微妙的重要性权重
                            normalized_matrix[feat_idx, :] = row_normalized * intensity_weight
                            
                            print(f"      {final_features[feat_idx]}: 重要性={importance_scores[feat_idx]:.4f}, 权重={intensity_weight:.3f}, 范围=[{-intensity_weight:.3f}, {+intensity_weight:.3f}]")
                        else:
                            normalized_matrix[feat_idx, :] = 0.0
                    
                    colorbar_label = 'Normalized GeoShapley Value'
                    colorbar_ticks = [-1, -0.5, 0, 0.5, 1]
                    colorbar_labels = ['-Max', '-50%', '0', '+50%', '+Max']
                    
                elif normalization == 'bivariate':
                    # 方法8：双变量颜色映射（时间变化+重要性的三角形色彩空间）
                    # 使用RdBu_r色彩映射：保持与原始色彩方案一致
                    import matplotlib.cm as cm
                    
                    # 获取RdBu_r colormap的实际红色和蓝色
                    rdbu_cmap = cm.get_cmap('RdBu_r')
                    red_color = np.array(rdbu_cmap(1.0)[:3])    # 获取最红色 (正值)
                    blue_color = np.array(rdbu_cmap(0.0)[:3])   # 获取最蓝色 (负值)
                    white_color = np.array([1.0, 1.0, 1.0])    # 白色 (零值)
                    
                    # 计算重要性权重：基于平均绝对SHAP值
                    importance_scores = final_mean_importance  # 已经按排序后的顺序
                    max_importance = importance_scores[0]  # 最重要特征的分数
                    min_importance = importance_scores[-1]  # 最不重要特征的分数
                    
                    print(f"    🎨 {res}: 重要性分数范围 [{min_importance:.4f}, {max_importance:.4f}]")
                    print(f"    🎨 {res}: 使用RdBu_r色彩 - 红色{red_color}, 蓝色{blue_color}")
                    
                    # 创建RGB矩阵用于双变量显示
                    rgb_matrix = np.zeros((final_shap_matrix.shape[0], final_shap_matrix.shape[1], 3))
                    normalized_matrix = np.zeros_like(final_shap_matrix)  # 用于数值显示
                    
                    for feat_idx in range(final_shap_matrix.shape[0]):
                        row_data = final_shap_matrix[feat_idx, :]
                        row_abs_max = np.max(np.abs(row_data))
                        
                        if row_abs_max > 1e-10:
                            # Step 1: 时间变化归一化
                            row_normalized = row_data / row_abs_max  # [-1, +1]
                            normalized_matrix[feat_idx, :] = row_normalized
                            
                            # Step 2: 使用RdBu_r颜色映射 + 重要性饱和度调整
                            # 保持RdBu_r色彩，但根据特征重要性调整饱和度
                            
                            # 计算重要性饱和度权重：从高重要性(1.0)到低重要性(0.4)
                            importance_rank = feat_idx  # 特征已按重要性排序，0=最重要
                            total_features = final_shap_matrix.shape[0]
                            saturation_weight = 1.0 - 0.6 * (importance_rank / (total_features - 1))  # [1.0, 0.4]
                            
                            # Step 3: 使用RdBu_r色彩 + 重要性饱和度
                            for year_idx in range(len(row_normalized)):
                                temporal_value = row_normalized[year_idx]
                                
                                # 获取RdBu_r的基础颜色
                                colormap_position = (temporal_value + 1) / 2  # 转换为[0, 1]范围
                                base_color = np.array(rdbu_cmap(colormap_position)[:3])
                                
                                # 应用饱和度调整：通过与白色混合来降低饱和度
                                white_color = np.array([1.0, 1.0, 1.0])
                                final_color = base_color * saturation_weight + white_color * (1 - saturation_weight)
                                
                                # 确保RGB值在[0,1]范围内
                                final_color = np.clip(final_color, 0, 1)
                                rgb_matrix[feat_idx, year_idx, :] = final_color
                            
                            print(f"      {final_features[feat_idx]}: 重要性={importance_scores[feat_idx]:.4f}, 饱和度权重={saturation_weight:.2f}")
                        else:
                            # 零值为白色
                            rgb_matrix[feat_idx, :, :] = white_color
                            normalized_matrix[feat_idx, :] = 0.0
                    
                    # 保存RGB矩阵用于特殊渲染
                    bivariate_rgb_matrix = rgb_matrix
                    colorbar_label = 'Temporal (RdBu_r) + Importance (Saturation)'
                    colorbar_ticks = [-1, -0.5, 0, 0.5, 1]
                    colorbar_labels = ['-Max', '-50%', '0', '+50%', '+Max']
                    
                else:  # 'linear' (legacy method)
                    # 方法5：线性缩放（映射到0-1范围，0.5为零点）
                    abs_max = np.max(np.abs(final_shap_matrix))
                    if abs_max > 0:
                        normalized_matrix = final_shap_matrix / abs_max * 0.5 + 0.5  # 缩放到0-1，0.5为零点
                    else:
                        normalized_matrix = np.full_like(final_shap_matrix, 0.5)
                    
                    colorbar_label = 'Normalized GeoShapley Value'
                    colorbar_ticks = [0, 0.25, 0.5, 0.75, 1.0]
                    colorbar_labels = ['-Max', '-50%', '0', '+50%', '+Max']
                
                print(f"    🎨 {res}: 归一化后范围 [{np.min(normalized_matrix):.4f}, {np.max(normalized_matrix):.4f}]")
                
                # 使用归一化后的数据绘制
                if normalization == 'row_normalize':
                    # 分层归一化使用固定范围 [-0.8, +0.8]
                    im = ax.imshow(normalized_matrix, 
                                 aspect='auto',
                                 cmap=cmap,
                                 vmin=-0.8,
                                 vmax=0.8,
                                 interpolation='nearest')
                elif normalization == 'bivariate':
                    # 双变量颜色映射使用RGB矩阵直接显示
                    im = ax.imshow(bivariate_rgb_matrix, 
                                 aspect='auto',
                                 interpolation='nearest')
                    # 注意：双变量情况下不需要colormap，因为直接使用RGB值
                elif normalization in ['symmetric', 'row_wise', 'importance_weighted', 'temporal_focus']:
                    # 对称归一化、逐行归一化、重要性加权归一化和时间优先归一化都使用固定范围 [-1, +1]
                    im = ax.imshow(normalized_matrix, 
                                 aspect='auto',
                                 cmap=cmap,
                                 vmin=-1,
                                 vmax=1,
                                 interpolation='nearest')
                else:
                    # 其他归一化方法使用[0, 1]范围
                    im = ax.imshow(normalized_matrix, 
                                 aspect='auto',
                                 cmap=cmap,
                                 vmin=0,
                                 vmax=1,
                                 interpolation='nearest')
                
                # 设置刻度
                ax.set_xticks(np.arange(len(years)))
                ax.set_xticklabels(years, rotation=45, ha='right', fontsize=10)
                ax.set_yticks(np.arange(len(simplified_features)))
                ax.set_yticklabels(simplified_features, fontsize=10)
                
                # 加粗刻度
                ax.tick_params(axis='both', direction='in', width=1.5, length=4)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')
                
                # 添加网格线（在单元格之间）
                for y in range(len(simplified_features) + 1):
                    ax.axhline(y - 0.5, color='white', linewidth=0.5)
                for x in range(len(years) + 1):
                    ax.axvline(x - 0.5, color='white', linewidth=0.5)
                
                # 设置标题和轴标签
                ax.set_title(f'{subplot_labels[i]} {res_titles[res]}',
                           fontsize=14, fontweight='bold', pad=10, loc='left')
                ax.set_xlabel('Year', fontsize=12, fontweight='bold')
                ax.set_ylabel('Features', fontsize=12, fontweight='bold')
                
                # 添加颜色条
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.1)
                
                if normalization == 'bivariate':
                    # 为双变量映射创建特殊的三角形图例
                    create_bivariate_colorbar(cax, ax, res)
                else:
                    # 标准颜色条
                    cbar = plt.colorbar(im, cax=cax)
                    # 🔥 修复：使用动态生成的颜色条标签
                    cbar.set_label(colorbar_label, fontsize=11, fontweight='bold')
                    cbar.ax.tick_params(labelsize=10, width=1.5, length=4)
                    for t in cbar.ax.get_yticklabels():
                        t.set_fontweight('bold')
                    
                    # 添加动态生成的颜色条刻度标签
                    cbar.ax.set_yticks(colorbar_ticks)
                    cbar.ax.set_yticklabels(colorbar_labels, fontweight='bold', fontsize=9)
                
                # 添加数据源信息标注（已注释掉，去除左上角文字标签）
                # ax.text(0.02, 0.98, res_data_source, transform=ax.transAxes,
                #        fontsize=9, ha='left', va='top', fontweight='bold',
                #        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
                
                # 🔥 修复：更新数值标注以显示原始SHAP值（更直观）
                if len(simplified_features) <= 10 and len(years) <= 10:
                    for y in range(len(simplified_features)):
                        for x in range(len(years)):
                            # 显示原始SHAP值而不是归一化值
                            original_value = final_shap_matrix[y, x]
                            normalized_value = normalized_matrix[y, x]
                            
                            # 根据归一化后的背景颜色选择文本颜色
                            if normalization == 'row_normalize':
                                # 分层归一化范围 [-0.8, +0.8]
                                text_color = 'white' if abs(normalized_value) > 0.5 else 'black'
                            elif normalization == 'bivariate':
                                # 双变量颜色映射：基于RGB亮度判断文本颜色
                                rgb_value = bivariate_rgb_matrix[y, x, :]
                                brightness = np.mean(rgb_value)  # 计算亮度
                                text_color = 'white' if brightness < 0.5 else 'black'
                            elif normalization in ['symmetric', 'row_wise', 'importance_weighted', 'temporal_focus']:
                                # 对称归一化、逐行归一化、重要性加权归一化和时间优先归一化范围 [-1, +1]
                                text_color = 'white' if abs(normalized_value) > 0.6 else 'black'
                            else:
                                # 其他范围[0, 1]
                                text_color = 'white' if normalized_value > 0.6 else 'black'
                            
                            # 显示原始值，但使用更简洁的格式
                            if original_value >= 0.01:
                                display_text = f'{original_value:.2f}'
                            elif original_value >= 0.001:
                                display_text = f'{original_value:.3f}'
                            else:
                                display_text = f'{original_value:.1e}'
                            
                            ax.text(x, y, display_text, 
                                   ha='center', va='center',
                                   fontsize=8, fontweight='bold',
                                   color=text_color)
            
            # 调整布局
            plt.tight_layout(rect=[0, 0.02, 1, 0.96])
            
            # 保存图表
            if output_dir:
                if ensure_dir_exists(output_dir):
                    output_path = os.path.join(output_dir, 'temporal_feature_heatmap.png')
                    save_plot_for_publication(output_path, fig)
                    
                    # 输出详细的保存信息
                    print(f"\n  ✅ 时序特征热图（原始SHAP值）已保存至: {output_path}")
                    print(f"    📊 数据真实性: 使用原始SHAP值，确保科学准确性")
                else:
                    print(f"无法创建输出目录: {output_dir}")
    
    # 恢复原始rcParams
    plt.rcParams.update(original_rcParams)
    
    return fig


def plot_temporal_feature_trends(results_by_resolution: Dict,
                               output_dir: Optional[str] = None,
                               top_n_features: int = 5,
                               figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    创建特征重要性趋势图（作为热图的补充）
    
    参数:
    - results_by_resolution: 包含各分辨率模型结果的字典
    - output_dir: 输出目录
    - top_n_features: 显示的顶级特征数量
    - figsize: 图像大小
    
    返回:
    - fig: matplotlib图形对象
    """
    # 计算时序SHAP值
    temporal_shap_data = calculate_temporal_shap_values(results_by_resolution)
    
    if not temporal_shap_data:
        print("错误: 无法计算时序SHAP值")
        return None
    
    # 创建样式设置
    style_dict = {
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'font.weight': 'bold',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'figure.dpi': 600,
        'savefig.dpi': 600,
    }
    
    with plt.style.context('default'):
        with plt.rc_context(style_dict):
            
            # 创建图形
            fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
            fig.suptitle('Top Features Temporal Trends Across Resolutions',
                        fontsize=16, fontweight='bold')
            
            # 分辨率设置
            resolutions = ['res7', 'res6', 'res5']
            res_titles = {
                'res7': 'Resolution 7 (Micro)',
                'res6': 'Resolution 6 (Meso)', 
                'res5': 'Resolution 5 (Macro)'
            }
            
            # 定义颜色调色板
            colors = plt.cm.tab10(np.linspace(0, 1, top_n_features))
            
            # 处理每个分辨率
            for i, (ax, res) in enumerate(zip(axes, resolutions)):
                if res not in temporal_shap_data:
                    ax.text(0.5, 0.5, f"No data for {res}",
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
                    continue
                
                # 获取数据
                data = temporal_shap_data[res]
                shap_matrix = data['matrix']
                features = data['features']
                years = data['years']
                
                # 选择前N个重要特征
                mean_importance = np.mean(shap_matrix, axis=1)
                top_indices = np.argsort(mean_importance)[-top_n_features:][::-1]
                
                # 绘制趋势线
                for j, idx in enumerate(top_indices):
                    feature_name = simplify_feature_name_for_plot(features[idx])
                    importance_trend = shap_matrix[idx]
                    
                    ax.plot(years, importance_trend, 
                           color=colors[j], 
                           linewidth=2,
                           marker='o',
                           markersize=5,
                           label=feature_name)
                
                # 设置标题和标签
                ax.set_title(res_titles[res], fontsize=14, fontweight='bold')
                ax.set_xlabel('Year', fontsize=12, fontweight='bold')
                if i == 0:
                    ax.set_ylabel('Mean |SHAP value|', fontsize=12, fontweight='bold')
                
                # 设置刻度
                ax.tick_params(axis='both', direction='in', width=1.5, length=4)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')
                
                # 添加网格
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # 添加图例
                ax.legend(loc='best', fontsize=10, frameon=True)
                
                # 加粗边框
                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            if output_dir:
                if ensure_dir_exists(output_dir):
                    output_path = os.path.join(output_dir, 'temporal_feature_trends.png')
                    plt.savefig(output_path, dpi=600, bbox_inches='tight')
                    print(f"时序特征趋势图已保存至: {output_path}")
    
    return fig

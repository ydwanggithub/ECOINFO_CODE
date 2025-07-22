#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
年度重要性时序特征热图模块: 按年度重要性排序的双色编码热图

该模块创建新的时序特征热图，其中：
1. 每年特征按该年重要性排序（而非全局重要性）
2. 特征使用离散颜色作为网格边框
3. 网格填充使用原始GeoShapley值
4. 右侧颜色条显示GeoShapley值范围
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
from typing import Dict, List, Tuple, Optional

from .base import enhance_plot_style, save_plot_for_publication, ensure_dir_exists
from .utils import simplify_feature_name_for_plot

__all__ = ['plot_yearly_importance_temporal_heatmap']


def calculate_yearly_shap_importance(results_by_resolution: Dict, 
                                   years: Optional[List[int]] = None) -> Dict:
    """
    计算每年每个特征的重要性，用于年度排序
    
    参数:
    - results_by_resolution: 包含各分辨率模型结果的字典
    - years: 要分析的年份列表，默认为2000-2024
    
    返回:
    - yearly_importance_dict: 包含各分辨率年度重要性数据的字典
    """
    if years is None:
        years = list(range(2000, 2025))
    
    print("  🔧 计算每年特征重要性用于年度排序...")
    
    yearly_importance_dict = {}
    
    for res, res_data in results_by_resolution.items():
        shap_values_by_feature = res_data.get('shap_values_by_feature')
        X_sample = res_data.get('X_sample')
        
        if shap_values_by_feature is None or X_sample is None:
            print(f"警告: {res}缺少必要的SHAP数据")
            continue
        
        # 获取特征名称（排除year和交互效应）
        all_feature_keys = list(shap_values_by_feature.keys())
        feature_names = []
        
        for f in all_feature_keys:
            if (f != 'year' and 
                '×' not in f and 
                ' x ' not in f and 
                '_x_' not in f and
                'interaction' not in f.lower()):
                feature_names.append(f)
        
        print(f"    📊 {res}: 分析{len(feature_names)}个特征的年度重要性")
        
        # 确保年份数据匹配
        n_shap_samples = len(next(iter(shap_values_by_feature.values())))
        if len(X_sample) >= n_shap_samples:
            year_data = X_sample['year'].iloc[:n_shap_samples]
        else:
            year_data = X_sample['year']
            remaining = n_shap_samples - len(X_sample)
            last_year = X_sample['year'].iloc[-1]
            additional_years = pd.Series([last_year] * remaining)
            year_data = pd.concat([X_sample['year'], additional_years], ignore_index=True)
        
        # 计算每年每个特征的重要性
        yearly_importance = {}
        yearly_shap_values = {}
        
        for year in years:
            year_mask = (year_data == year)
            if np.any(year_mask):
                year_importance = {}
                year_shap = {}
                
                for feat_name in feature_names:
                    if feat_name in shap_values_by_feature:
                        feat_shap = np.array(shap_values_by_feature[feat_name])[year_mask]
                        # 重要性：平均绝对值
                        year_importance[feat_name] = np.mean(np.abs(feat_shap))
                        # 原始SHAP值：平均值（保持正负号）
                        year_shap[feat_name] = np.mean(feat_shap)
                
                yearly_importance[year] = year_importance
                yearly_shap_values[year] = year_shap
        
        yearly_importance_dict[res] = {
            'yearly_importance': yearly_importance,
            'yearly_shap_values': yearly_shap_values,
            'feature_names': feature_names,
            'years': years
        }
        
        # 打印前几年的排序示例
        for year in years[:3]:
            if year in yearly_importance:
                sorted_features = sorted(yearly_importance[year].items(), 
                                       key=lambda x: x[1], reverse=True)
                print(f"    📅 {res} {year}年特征排序: {[f[0] for f in sorted_features[:3]]}...")
    
    return yearly_importance_dict


def plot_yearly_importance_temporal_heatmap(results_by_resolution: Dict,
                                          output_dir: Optional[str] = None,
                                          figsize: Tuple[int, int] = (14, 16)) -> plt.Figure:
    """
    创建年度重要性时序热图：每年按重要性排序，双色编码
    
    Args:
        results_by_resolution: 按分辨率组织的结果字典
        output_dir: 输出目录路径
        figsize: 图形大小
        
    Returns:
        matplotlib.figure.Figure or None
    """
    print("\n🎨 创建年度重要性时序热图（双色编码）...")
    print("  📊 每年按该年重要性排序特征")
    print("  🎨 离散颜色边框 + GeoShapley值填充")
    
    # 计算年度重要性数据
    yearly_data = calculate_yearly_shap_importance(results_by_resolution)
    if not yearly_data:
        print("  ⚠️ 警告: 没有找到有效的年度重要性数据")
        return None
    
    # 样式设置
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
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'figure.figsize': figsize,
    }
    
    with plt.style.context('default'):
        with plt.rc_context(style_dict):
            
            # 创建图形
            fig = plt.figure(figsize=figsize, dpi=600)
            fig.suptitle('Yearly-Ranked Temporal GeoShapley Contribution Patterns', 
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
            subplot_labels = ['(a)', '(b)', '(c)']
            
            # 为所有特征分配离散颜色
            all_features = set()
            for res_data in yearly_data.values():
                all_features.update(res_data['feature_names'])
            all_features = sorted(list(all_features))
            
            # 创建离散颜色映射
            n_features = len(all_features)
            feature_colors = plt.cm.Set3(np.linspace(0, 1, n_features))
            feature_color_map = {feat: feature_colors[i] for i, feat in enumerate(all_features)}
            
            print(f"    🎨 为{n_features}个特征分配离散颜色")
            
            # 处理每个分辨率
            for i, res in enumerate(resolutions):
                if res not in yearly_data:
                    ax = fig.add_subplot(gs[i])
                    ax.text(0.5, 0.5, f"No data for {res}", 
                           ha='center', va='center', fontsize=14, 
                           transform=ax.transAxes)
                    ax.axis('off')
                    continue
                
                # 获取数据
                data = yearly_data[res]
                yearly_importance = data['yearly_importance']
                yearly_shap_values = data['yearly_shap_values']
                years = data['years']
                
                print(f"    📊 {res}: 处理{len(years)}年 x {len(data['feature_names'])}特征的数据")
                
                # 创建子图
                ax = fig.add_subplot(gs[i])
                
                # 为每年创建排序后的特征矩阵
                max_features = max(len(yearly_importance[year]) for year in years 
                                 if year in yearly_importance)
                
                # 创建数据矩阵
                shap_matrix = np.full((max_features, len(years)), np.nan)
                feature_matrix = np.full((max_features, len(years)), '', dtype=object)
                
                for year_idx, year in enumerate(years):
                    if year in yearly_importance:
                        # 按该年重要性排序特征
                        sorted_features = sorted(yearly_importance[year].items(), 
                                               key=lambda x: x[1], reverse=True)
                        
                        for feat_rank, (feat_name, importance) in enumerate(sorted_features):
                            if feat_rank < max_features:
                                # 存储SHAP值和特征名
                                shap_matrix[feat_rank, year_idx] = yearly_shap_values[year][feat_name]
                                feature_matrix[feat_rank, year_idx] = feat_name
                
                # 应用行归一化以突出时序变化
                normalized_shap_matrix = np.full_like(shap_matrix, np.nan)
                for row in range(max_features):
                    row_data = shap_matrix[row, :]
                    valid_mask = ~np.isnan(row_data)
                    if np.any(valid_mask):
                        valid_data = row_data[valid_mask]
                        if len(valid_data) > 1:
                            row_abs_max = np.max(np.abs(valid_data))
                            if row_abs_max > 1e-10:
                                # 行归一化到[-1, 1]
                                normalized_shap_matrix[row, valid_mask] = valid_data / row_abs_max
                            else:
                                normalized_shap_matrix[row, valid_mask] = 0.0
                        else:
                            normalized_shap_matrix[row, valid_mask] = 0.0
                
                # 设置颜色范围为归一化后的范围
                vmin, vmax = -1, 1
                
                print(f"    🎨 {res}: 应用行归一化突出时序变化，范围 [{vmin:.1f}, {vmax:.1f}]")
                
                # 绘制网格
                for row in range(max_features):
                    for col in range(len(years)):
                        if not np.isnan(normalized_shap_matrix[row, col]):
                            feat_name = feature_matrix[row, col]
                            normalized_val = normalized_shap_matrix[row, col]
                            
                            # 获取特征的离散颜色（边框）
                            edge_color = feature_color_map[feat_name]
                            
                            # 计算填充颜色（基于行归一化的SHAP值）
                            if normalized_val > 0:
                                # 正值：红色系
                                intensity = min(normalized_val, 1.0)
                                fill_color = plt.cm.Reds(0.3 + 0.7 * intensity)
                            elif normalized_val < 0:
                                # 负值：蓝色系
                                intensity = min(abs(normalized_val), 1.0)
                                fill_color = plt.cm.Blues(0.3 + 0.7 * intensity)
                            else:
                                # 零值：白色
                                fill_color = 'white'
                            
                            # 绘制矩形
                            rect = patches.Rectangle((col, row), 1, 1,
                                                   linewidth=2,
                                                   edgecolor=edge_color,
                                                   facecolor=fill_color,
                                                   alpha=0.8)
                            ax.add_patch(rect)
                
                # 设置坐标轴
                ax.set_xlim(0, len(years))
                ax.set_ylim(0, max_features)
                ax.set_aspect('equal')
                
                # 设置刻度
                ax.set_xticks(np.arange(len(years)) + 0.5)
                ax.set_xticklabels(years, rotation=45, ha='right', fontsize=10)
                
                # Y轴显示排名
                ax.set_yticks(np.arange(max_features) + 0.5)
                ax.set_yticklabels([f'Rank {i+1}' for i in range(max_features)], 
                                 fontsize=10)
                
                # 反转Y轴（最重要的在上方）
                ax.invert_yaxis()
                
                # 设置标题和标签
                ax.set_title(f'{subplot_labels[i]} {res_titles[res]}',
                           fontsize=14, fontweight='bold', pad=40, loc='left')
                ax.set_xlabel('Year', fontsize=12, fontweight='bold')
                ax.set_ylabel('Yearly Importance Rank', fontsize=12, fontweight='bold')
                
                # 创建特征颜色图例（水平布局，位于子图上方）
                # 收集该分辨率出现的所有特征
                res_features = set()
                for row in range(max_features):
                    for col in range(len(years)):
                        if feature_matrix[row, col] != '':
                            res_features.add(feature_matrix[row, col])
                
                res_features = sorted(list(res_features))
                
                # 计算图例位置
                legend_y = 1.12  # 位于子图上方
                legend_height = 0.05
                
                # 创建特征图例
                n_cols = min(len(res_features), 6)  # 每行最多6个特征
                n_rows = (len(res_features) + n_cols - 1) // n_cols
                
                legend_text = ""
                for idx, feat in enumerate(res_features):
                    # 简化特征名称
                    simplified_name = simplify_feature_name_for_plot(feat)
                    
                    # 创建颜色方块
                    rect_x = (idx % n_cols) / n_cols
                    rect_y = legend_y - (idx // n_cols) * 0.03
                    
                    # 绘制颜色方块
                    legend_rect = patches.Rectangle((rect_x, rect_y), 0.015, 0.02,
                                                  transform=ax.transAxes,
                                                  facecolor=feature_color_map[feat],
                                                  edgecolor='black',
                                                  linewidth=1)
                    ax.add_patch(legend_rect)
                    
                    # 添加文本标签
                    ax.text(rect_x + 0.02, rect_y + 0.01, simplified_name,
                           transform=ax.transAxes,
                           fontsize=8, fontweight='bold',
                           verticalalignment='center')
                
                # 加粗刻度
                ax.tick_params(axis='both', direction='in', width=1.5, length=4)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight('bold')
                
                # 设置边框
                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)
                
                # 添加GeoShapley值的颜色条
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.1)
                
                # 创建颜色条
                sm = plt.cm.ScalarMappable(cmap='RdBu_r', 
                                         norm=plt.Normalize(vmin=vmin, vmax=vmax))
                sm.set_array([])
                cbar = plt.colorbar(sm, cax=cax)
                cbar.set_label('Normalized\nTemporal Variation', fontsize=11, fontweight='bold')
                cbar.ax.tick_params(labelsize=10, width=1.5, length=4)
                for t in cbar.ax.get_yticklabels():
                    t.set_fontweight('bold')
            
            # 调整布局，为特征图例留出更多空间
            plt.tight_layout(rect=[0, 0.02, 1, 0.93])
            
            # 保存图表
            if output_dir:
                if ensure_dir_exists(output_dir):
                    output_path = os.path.join(output_dir, 'yearly_importance_temporal_heatmap.png')
                    save_plot_for_publication(output_path, fig)
                    print(f"\n  ✅ 年度重要性时序热图已保存至: {output_path}")
                    print(f"    🎨 双色编码: 离散颜色边框 + 行归一化时序变化填充")
                    print(f"    🏷️ 特征图例: 水平布局于各子图上方")
                else:
                    print(f"无法创建输出目录: {output_dir}")
    
    return fig 
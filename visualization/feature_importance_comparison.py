#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征重要性比较可视化模块

包含多分辨率特征重要性比较功能。

主要功能：
- plot_feature_importance_comparison: 多分辨率特征重要性比较图
- plot_feature_category_comparison: 按类别比较特征重要性
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
import warnings
from typing import Dict, Optional
import seaborn as sns

# 导入核心功能模块
from model_analysis.core import ensure_dir_exists

# 导入基础功能
try:
    from visualization.feature_importance_core import (
        categorize_feature_for_geoshapley_display
    )
    from visualization.utils import simplify_feature_name_for_plot
    from visualization.base import color_map
except ImportError as e:
    warnings.warn(f"导入可视化模块失败: {e}")
    
    def categorize_feature_for_geoshapley_display(feature_name):
        return 'Other'
    
    def simplify_feature_name_for_plot(feature):
        return feature.replace('_', ' ').title()
    
    color_map = {
        'Climate': '#3498db',
        'Human Activity': '#e74c3c',
        'Terrain': '#f39c12',
        'Land Cover': '#9b59b6',
        'Geographic': '#1abc9c',
        'Temporal': '#34495e',
        'Other': '#7f8c8d'
    }


def plot_feature_importance_comparison(feature_importances: Dict,
                                     output_dir: Optional[str] = None,
                                     results: Optional[Dict] = None) -> plt.Figure:
    """
    创建特征重要性比较图，确保使用原始特征值数据保持论文结果一致性
    
    参数:
    - feature_importances: 原始特征重要性字典
    - output_dir: 输出目录
    - results: 完整的结果字典，用于获取原始采样数据
    
    返回:
    - fig: matplotlib图表对象
    """
    print("\n🎨 创建特征重要性比较图...")
    
    # 优先使用原始特征值数据重新计算特征重要性
    original_feature_importances = {}
    original_data_used = False
    
    if results is not None:
        for res in ['res7', 'res6', 'res5']:
            if res not in results:
                continue
            
            res_data = results[res]
            
            # 优先使用原始采样SHAP值数据
            if ('shap_values_by_feature' in res_data and 
                res_data['shap_values_by_feature'] is not None):
                
                print(f"  ✅ {res}: 基于原始采样SHAP值重新计算特征重要性")
                
                original_shap_values = res_data['shap_values_by_feature']
                feature_importance_list = []
                
                for feat_name, shap_vals in original_shap_values.items():
                    importance = np.abs(shap_vals).mean()
                    feature_importance_list.append((feat_name, importance))
                
                feature_importance_list.sort(key=lambda x: x[1], reverse=True)
                original_feature_importances[res] = feature_importance_list
                original_data_used = True
                
            elif res in feature_importances:
                print(f"  🔄 {res}: 使用传入的原始特征重要性数据")
                
                original_importance = feature_importances[res]
                if isinstance(original_importance, dict):
                    original_importance = [(k, v) for k, v in original_importance.items()]
                
                original_importance.sort(key=lambda x: x[1], reverse=True)
                original_feature_importances[res] = original_importance
                original_data_used = True
        
        final_feature_importances = original_feature_importances if original_data_used else feature_importances
    else:
        final_feature_importances = feature_importances
    
    # 定义六大类别的颜色映射
    geoshapley_color_map = {
        'Climate': '#3498db',
        'Human Activity': '#e74c3c',
        'Terrain': '#f39c12',
        'Land Cover': '#9b59b6',
        'Geographic': '#1abc9c',
        'Temporal': '#34495e',
        'Other': '#7f8c8d'
    }
    
    # 样式设置
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
            
            # 创建1×3网格
            fig, axes = plt.subplots(1, 3, figsize=(20, 10), dpi=600)
            
            titles = {
                'res7': 'H3 Resolution 7 (Micro)',
                'res6': 'H3 Resolution 6 (Meso)', 
                'res5': 'H3 Resolution 5 (Macro)'
            }
            
            for idx, res in enumerate(['res7', 'res6', 'res5']):
                ax = axes[idx]
                
                if res not in final_feature_importances:
                    ax.text(0.5, 0.5, f"No data for {res}", 
                           ha='center', va='center', fontsize=14, transform=ax.transAxes)
                    ax.axis('off')
                    continue
                
                # 获取特征重要性数据
                feature_importance = final_feature_importances[res]
                if isinstance(feature_importance, dict):
                    feature_importance = [(k, v) for k, v in feature_importance.items()]
                
                # 按重要性排序
                feature_importance_sorted = sorted(feature_importance, key=lambda x: x[1], reverse=True)
                
                # 计算总重要性用于百分比
                total_importance = sum(imp for _, imp in feature_importance_sorted)
                
                # 准备绘图数据
                features = []
                importances = []
                colors = []
                labels = []
                category_labels = []
                
                for feat, imp in feature_importance_sorted:
                    features.append(feat)
                    importances.append(imp)
                    
                    display_name = simplify_feature_name_for_plot(feat)
                    category = categorize_feature_for_geoshapley_display(feat)
                    colors.append(geoshapley_color_map.get(category, '#7f8c8d'))
                    labels.append(display_name)
                    
                    percentage = (imp / total_importance * 100) if total_importance > 0 else 0
                    category_labels.append(f"{category} {percentage:.1f}%")
                
                # 反转列表，使最重要的特征在顶部
                features = features[::-1]
                importances = importances[::-1]
                colors = colors[::-1]
                labels = labels[::-1]
                category_labels = category_labels[::-1]
                
                # 创建y轴位置
                y_pos = np.arange(len(features))
                
                # 绘制水平条形图
                bars = ax.barh(y_pos, importances, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.0)
        
                # 设置y轴标签
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
                
                # 在条形右侧添加类别和百分比标签
                for i, (bar, cat_label) in enumerate(zip(bars, category_labels)):
                    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                           cat_label, va='center', fontsize=9, fontweight='bold')
        
                # 设置标题和标签
                ax.set_title(f'({chr(97+idx)}) {titles[res]}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Mean |SHAP value|', fontsize=12, fontweight='bold')
                
                # 设置x轴范围
                max_val = max(importances) if importances else 1
                ax.set_xlim(0, max_val * 1.3)
                
                # 添加网格
                ax.grid(axis='x', alpha=0.3, linestyle='--')
                
                # 创建图例 - 计算六大类别的总百分比
                category_percentages = {}
                for feat, imp in feature_importance_sorted:
                    category = categorize_feature_for_geoshapley_display(feat)
                    
                    if category not in category_percentages:
                        category_percentages[category] = 0
                    percentage = (imp / total_importance * 100) if total_importance > 0 else 0
                    category_percentages[category] += percentage
                
                # 创建图例元素
                from matplotlib.patches import Patch
                legend_elements = []
                category_order = ['Climate', 'Human Activity', 'Terrain', 'Land Cover', 'Geographic', 'Temporal']
                for category in category_order:
                    if category in category_percentages:
                        color = geoshapley_color_map.get(category, '#7f8c8d')
                        label = f"{category} {category_percentages[category]:.1f}%"
                        legend_elements.append(Patch(facecolor=color, label=label))
                
                # 添加其他类别
                for category in category_percentages:
                    if category not in category_order:
                        color = geoshapley_color_map.get(category, '#7f8c8d')
                        label = f"{category} {category_percentages[category]:.1f}%"
                        legend_elements.append(Patch(facecolor=color, label=label))
                
                # 🔧 修复：增大图例字体，提高可读性
                ax.legend(handles=legend_elements, loc='lower right', 
                         frameon=True, fontsize=12, title=None,
                         bbox_to_anchor=(0.98, 0.02), 
                         title_fontsize=13, prop={'weight': 'bold'})
    
    # 添加总标题
    fig.suptitle('Feature Importance Comparison Across Resolutions', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图表
    if output_dir:
        ensure_dir_exists(output_dir)
        output_path = os.path.join(output_dir, 'feature_importance_comparison.png')
        plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.1,
                   transparent=False, facecolor='white', edgecolor='none')
        print(f"  ✅ 特征重要性比较图已保存到: {output_path}")
    
    return fig


def plot_feature_category_comparison(results: Dict, 
                                   output_dir: Optional[str] = None, 
                                   save_plot: bool = True, 
                                   figsize: tuple = (12, 8), 
                                   palette: str = 'viridis') -> plt.Figure:
    """
    创建按特征类别分组的特征重要性比较图
    
    参数:
    results: 包含按分辨率组织的model_results字典
    output_dir: 输出目录路径
    save_plot: 是否保存图表
    figsize: 图表大小
    palette: 颜色调色板名称
    
    返回:
    fig: matplotlib图形对象
    """
    print("创建特征类别对比图...")
    
    # 检查必要的键
    required_keys = ['feature_importance', 'feature_categories']
    
    feature_importance_by_res = {}
    feature_category_by_res = {}
    feature_importance_by_category = {}
    
    # 处理不同分辨率的数据
    for res, res_data in results.items():
        if not all(key in res_data for key in required_keys):
            print(f"警告: {res}的特征重要性格式无法识别")
            continue
        
        # 获取特征重要性
        if isinstance(res_data['feature_importance'], dict):
            feature_importance = res_data['feature_importance']
        elif isinstance(res_data['feature_importance'], list):
            if not res_data['feature_importance']:
                print(f"警告: {res}的特征重要性为空")
                continue
                
            if isinstance(res_data['feature_importance'][0], tuple):
                feature_importance = {feat: imp for feat, imp in res_data['feature_importance']}
            else:
                print(f"警告: {res}的特征重要性列表格式无法识别")
                continue
        else:
            print(f"警告: {res}的特征重要性格式无法识别")
            continue
        
        # 获取特征类别
        if isinstance(res_data['feature_categories'], dict):
            feature_categories = res_data['feature_categories']
        else:
            feature_categories = {}
            for feature in feature_importance.keys():
                feature_categories[feature] = categorize_feature_for_geoshapley_display(feature)
        
        feature_importance_by_res[res] = feature_importance
        feature_category_by_res[res] = feature_categories
        
        # 按类别组织特征
        by_category = {}
        for feature, importance in feature_importance.items():
            category = feature_categories.get(feature, 'Other')
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((feature, importance))
        
        # 对每个类别按重要性排序
        for category in by_category:
            by_category[category].sort(key=lambda x: x[1], reverse=True)
        
        feature_importance_by_category[res] = by_category
    
    # 检查是否有可用数据
    if not feature_importance_by_res:
        print("错误: 所有分辨率的特征重要性格式都无法识别")
        return None
    
    # 创建可视化
    resolutions = sorted(feature_importance_by_res.keys(), key=lambda x: int(x[3:]))
    
    # 获取所有存在的类别
    all_categories = set()
    for res_categories in feature_importance_by_category.values():
        all_categories.update(res_categories.keys())
    
    # 确定类别顺序
    category_order = []
    for category in ['Climate', 'Human Activity', 'Terrain', 'Land Cover', 'Geographic', 'Temporal']:
        if category in all_categories:
            category_order.append(category)
    
    # 为没有出现在预定义顺序中的类别添加到末尾
    for category in all_categories:
        if category not in category_order:
            category_order.append(category)
    
    # 设置类别颜色
    category_colors = {
        'Climate': '#3498db',
        'Human Activity': '#e74c3c',
        'Terrain': '#f39c12',
        'Land Cover': '#9b59b6',
        'Geographic': '#1abc9c',
        'Temporal': '#34495e',
        'Other': '#7f8c8d'
    }
    
    # 创建绘图
    fig, axes = plt.subplots(len(resolutions), 1, figsize=figsize, squeeze=False)
    
    # 统一所有子图的y轴比例
    max_importance = 0
    for res, importance_data in feature_importance_by_res.items():
        if importance_data:
            max_importance = max(max_importance, max(importance_data.values()))
    
    # 对每个分辨率绘制特征重要性条形图
    for i, res in enumerate(resolutions):
        ax = axes[i, 0]
        
        # 组织数据用于绘图
        plot_data = []
        for category in category_order:
            if category in feature_importance_by_category[res]:
                # 获取该类别的前3个特征
                for feature, importance in feature_importance_by_category[res][category][:3]:
                    display_name = simplify_feature_name_for_plot(feature)
                    plot_data.append({
                        'feature': display_name,
                        'importance': importance,
                        'category': category
                    })
        
        # 转换为DataFrame
        plot_df = pd.DataFrame(plot_data)
        
        if plot_df.empty:
            ax.text(0.5, 0.5, f"No feature importance data for {res}", 
                  ha='center', va='center', transform=ax.transAxes)
            continue
        
        # 绘制条形图
        sns.barplot(
            data=plot_df, 
            x='importance', 
            y='feature', 
            hue='category',
            palette=category_colors,
            ax=ax
        )
        
        # 设置图表标题和样式
        title = f"Feature Importance by Category ({res.replace('res', 'Resolution ')})"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=10)
        ax.set_ylabel('')
        
        # 统一y轴范围
        ax.set_xlim(0, max_importance * 1.1)
        
        # 添加网格
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        # 调整图例
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title='Categories', loc='lower right')
    
    plt.tight_layout()
    
    # 保存图表
    if output_dir and save_plot:
        output_path = os.path.join(output_dir, 'feature_category_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存特征类别对比图: {output_path}")
    
    return fig 
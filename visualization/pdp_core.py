#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDP核心绘制模块 - 基础PDP可视化功能

从pdp_plots.py重构而来，专注于：
- 单个特征PDP绘制

适配ST-GPR模型的特殊需求
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
from typing import Dict, List, Optional, Union

# 导入通用的绘图函数和工具
try:
    from .base import enhance_plot_style, ensure_dir_exists, save_plot_for_publication, color_map
    from .utils import clean_feature_name_for_plot, categorize_feature, simplify_feature_name_for_plot, enhance_feature_display_name, clean_feature_name, format_pdp_feature_name
    from model_analysis.core import standardize_feature_name
except ImportError:
    # 相对导入失败时尝试绝对导入
    from visualization.base import enhance_plot_style, ensure_dir_exists, save_plot_for_publication, color_map
    from visualization.utils import clean_feature_name_for_plot, categorize_feature, simplify_feature_name_for_plot, enhance_feature_display_name, clean_feature_name, format_pdp_feature_name
    from model_analysis.core import standardize_feature_name

# 设置matplotlib默认字体以支持中文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_pdp(feature, pdp_values, output_dir=None, resolution=None, feature_type=None, 
            hist_data=None, tick_values=None, xlabel=None, title=None, show_hist=True):
    """
    绘制单个特征的部分依赖图，用于分析STGPR模型特征影响
    
    参数:
    feature (str): 特征名称
    pdp_values (DataFrame): 包含ICE/PDP值的数据框，通常有feature和pd_ice列
    output_dir (str): 输出目录
    resolution (str): 分辨率标识
    feature_type (str): 特征类型，用于设置颜色和样式
    hist_data (array): 绘制直方图的数据
    tick_values (list): X轴刻度值
    xlabel (str): X轴标签，如果为None则使用特征名称
    title (str): 图表标题，如果为None则自动生成
    show_hist (bool): 是否显示直方图
    
    返回:
    matplotlib.figure.Figure: 图表对象
    """
    if pdp_values is None or len(pdp_values) == 0:
        print(f"警告: 缺少{feature}的PDP数据")
        return None
    
    # 首先标准化特征名称
    feature = standardize_feature_name(feature)
    
    # 提取x和y值
    if 'feature' in pdp_values.columns and 'pd_ice' in pdp_values.columns:
        x = pdp_values['feature'].values
        y = pdp_values['pd_ice'].values
    else:
        # 尝试提取第一列和第二列
        x = pdp_values.iloc[:, 0].values
        y = pdp_values.iloc[:, 1].values
    
    # 根据特征类型设置颜色
    if feature_type is None:
        feature_type = categorize_feature(feature)
    
    # 使用从base模块导入的color_map
    color = color_map.get(feature_type, '#3498db')
    
    # 用于显示的特征名称
    display_feature = clean_feature_name(feature)
    
    # 创建图表 - 如果要显示直方图，则创建带有两个子图的图表
    if show_hist and hist_data is not None:
        # 创建具有共享x轴的两个子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # 在上方子图绘制PDP曲线
        ax1.plot(x, y, color=color, linewidth=2)
        
        # 设置y轴标签和标题
        ax1.set_ylabel('Partial Dependence', fontsize=12, fontweight='bold')
        
        # 如果提供了标题，使用它，否则生成标题
        if title:
            ax1.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax1.set_title(f'Partial Dependence Plot for {display_feature}', fontsize=14, fontweight='bold')
        
        # 添加网格
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # 在下方子图绘制直方图
        ax2.hist(hist_data, bins=30, alpha=0.6, color=color, edgecolor='black')
        ax2.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        
        # 设置x轴标签
        if xlabel:
            ax2.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        else:
            ax2.set_xlabel(display_feature, fontsize=12, fontweight='bold')
        
        # 如果提供了刻度值，设置它们
        if tick_values is not None:
            ax2.set_xticks(tick_values)
        
        # 调整布局
        plt.tight_layout()
    else:
        # 只创建单个PDP图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制PDP曲线
        ax.plot(x, y, color=color, linewidth=2)
        
        # 设置标签和标题
        ax.set_ylabel('Partial Dependence', fontsize=12, fontweight='bold')
        
        # 如果提供了标题，使用它，否则生成标题
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'Partial Dependence Plot for {display_feature}', fontsize=14, fontweight='bold')
        
        # 设置x轴标签
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        else:
            ax.set_xlabel(display_feature, fontsize=12, fontweight='bold')
        
        # 如果提供了刻度值，设置它们
        if tick_values is not None:
            ax.set_xticks(tick_values)
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
    
    # 保存图表
    if output_dir:
        ensure_dir_exists(output_dir)
        # 清理特征名称以用作文件名
        safe_feature = re.sub(r'[\\/*?:"<>|]', "_", feature)
        
        if resolution:
            fig_path = os.path.join(output_dir, f"{resolution}_pdp_{safe_feature}.png")
        else:
            fig_path = os.path.join(output_dir, f"pdp_{safe_feature}.png")
        
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"已保存PDP图: {fig_path}")
    
    # 返回图表对象而不关闭它，以便调用者可以进一步修改或保存
    return fig 
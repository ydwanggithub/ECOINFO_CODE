#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GeoShapley部分依赖图绘制模块

该模块实现了标准的partial_dependence_plots函数，模仿GeoShapley库的类方法。
支持使用pygam库绘制GAM曲线，以及从pickle文件加载SHAP值进行绘制。

特性：
- 支持多分辨率网格显示
- 使用pygam.LinearGAM绘制红色趋势线
- 从保存的SHAP数据直接绘制
- 兼容GeoShapley三部分分解结构
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

# 尝试导入pygam
try:
    import pygam
    PYGAM_AVAILABLE = True
except ImportError:
    PYGAM_AVAILABLE = False
    print("警告: pygam库未安装，无法绘制GAM曲线")
    print("可以使用以下命令安装: pip install pygam")

from .base import enhance_plot_style, ensure_dir_exists
from .utils import simplify_feature_name_for_plot, enhance_feature_display_name


class GeoShapleyResults:
    """
    模拟GeoShapley库的结果类，用于存储和可视化SHAP值
    """
    
    def __init__(self, primary, geo, X_geo, base_value=None):
        """
        初始化GeoShapley结果对象
        
        参数:
        primary: 主效应SHAP值矩阵 (n_samples, n_primary_features)
        geo: GEO效应SHAP值数组 (n_samples,)
        X_geo: 特征数据DataFrame
        base_value: 基准值
        """
        self.primary = primary
        self.geo = geo if geo.ndim == 1 else geo.flatten()
        self.X_geo = X_geo
        self.base_value = base_value if base_value is not None else 0.0
        
        print(f"✅ 初始化GeoShapley结果:")
        print(f"   - Primary形状: {self.primary.shape}")
        print(f"   - GEO形状: {self.geo.shape}")
        print(f"   - 特征数据形状: {self.X_geo.shape}")
    
    def partial_dependence_plots(self, gam_curve=False, max_cols=3, figsize=None, dpi=200, **kwargs):
        """
        绘制部分依赖图，模仿GeoShapley库的方法
        
        参数:
        gam_curve: 是否绘制GAM平滑曲线
        max_cols: 最大列数
        figsize: 图形大小
        dpi: 图形DPI
        kwargs: 传递给散点图的其他参数
        
        返回:
        matplotlib.figure.Figure: 图形对象
        """
        if not PYGAM_AVAILABLE and gam_curve:
            print("警告: pygam库未安装，将跳过GAM曲线绘制")
            gam_curve = False
        
        k = self.primary.shape[1]  # 主效应特征数量
        
        num_cols = min(k, max_cols)
        num_rows = ceil(k / num_cols)
        
        if figsize is None:
            figsize = (num_cols * 5, num_rows * 4)
        
        fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize, dpi=dpi)
        axs = axs if num_rows > 1 else np.array([axs])
        axs = axs.flatten()
        
        # 设置默认散点图参数
        scatter_kwargs = {
            's': kwargs.get('s', 12),
            'color': kwargs.get('color', "#2196F3"),
            'edgecolors': kwargs.get('edgecolors', "white"),
            'lw': kwargs.get('lw', 0.3),
            'alpha': kwargs.get('alpha', 0.6)
        }
        
        col_counter = 0
        for col in range(k):
            ax = axs[col_counter]
            
            # 添加零线
            ax.axhline(0, linestyle='--', color='black', alpha=0.5, linewidth=1)
            
            # 获取特征值和对应的SHAP值
            x_values = self.X_geo.iloc[:, col].values
            y_values = self.primary[:, col]
            
            # 绘制散点图
            ax.scatter(x_values, y_values, **scatter_kwargs)
            
            # 设置标签
            feature_name = self.X_geo.iloc[:, col].name
            ax.set_ylabel("GeoShapley Value", fontweight='bold')
            ax.set_xlabel(feature_name, fontweight='bold')
            
            # 绘制GAM曲线
            if gam_curve and PYGAM_AVAILABLE:
                try:
                    # 准备数据
                    X_feature = x_values.reshape(-1, 1)
                    y_feature = y_values.reshape(-1, 1)
                    
                    # 网格搜索lambda参数
                    lam = np.logspace(2, 7, 5).reshape(-1, 1)
                    
                    # 拟合GAM模型
                    gam = pygam.LinearGAM(pygam.s(0), fit_intercept=False).gridsearch(
                        X_feature, y_feature, lam=lam
                    )
                    
                    # 生成预测网格
                    XX = gam.generate_X_grid(term=0)
                    pdep, confi = gam.partial_dependence(term=0, X=XX, width=0.95)
                    
                    # 绘制GAM曲线
                    ax.plot(XX, pdep, color="red", lw=2, label='GAM Curve')
                    
                    # 可选：绘制置信区间
                    if 'show_confidence' in kwargs and kwargs['show_confidence']:
                        ax.fill_between(XX.flatten(), 
                                      (pdep - confi).flatten(), 
                                      (pdep + confi).flatten(), 
                                      alpha=0.2, color="red")
                    
                except Exception as e:
                    print(f"警告: 特征 {feature_name} 的GAM曲线拟合失败: {e}")
            
            # 设置网格
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # 增强样式
            enhance_plot_style(ax)
            
            col_counter += 1
        
        # 隐藏多余的子图
        for i in range(col_counter, num_rows * num_cols):
            axs[i].axis('off')
        
        plt.tight_layout()
        return fig


def load_geoshapley_data(res_list=['res5', 'res6', 'res7'], data_dir='output'):
    """
    从pickle文件加载GeoShapley数据
    
    参数:
    res_list: 分辨率列表
    data_dir: 数据目录
    
    返回:
    dict: 包含各分辨率GeoShapley数据的字典
    """
    results = {}
    
    for res in res_list:
        geoshapley_file = os.path.join(data_dir, res, f'{res}_geoshapley_data.pkl')
        
        if os.path.exists(geoshapley_file):
            try:
                with open(geoshapley_file, 'rb') as f:
                    geoshapley_data = pickle.load(f)
                
                results[res] = geoshapley_data
                print(f"✅ 成功加载 {res} 的GeoShapley数据")
                
                # 打印数据键信息
                if isinstance(geoshapley_data, dict):
                    print(f"   包含键: {list(geoshapley_data.keys())}")
                
            except Exception as e:
                print(f"❌ 加载 {res} 的GeoShapley数据失败: {e}")
        else:
            print(f"⚠️ 未找到 {res} 的GeoShapley数据文件: {geoshapley_file}")
    
    return results


def create_geoshapley_results_from_data(geoshapley_data):
    """
    从加载的数据创建GeoShapleyResults对象
    
    参数:
    geoshapley_data: 从pickle文件加载的数据
    
    返回:
    GeoShapleyResults: 结果对象
    """
    try:
        # 检查数据格式
        if 'geoshap_original' in geoshapley_data:
            # 标准GeoShapley格式
            geoshap_orig = geoshapley_data['geoshap_original']
            primary = geoshap_orig['primary']
            geo = geoshap_orig['geo']
            X_sample = geoshapley_data.get('X_sample')
            base_value = geoshap_orig.get('base_value', 0.0)
            
            return GeoShapleyResults(primary, geo, X_sample, base_value)
            
        elif 'shap_values_by_feature' in geoshapley_data:
            # shap_values_by_feature格式
            shap_dict = geoshapley_data['shap_values_by_feature']
            X_sample = geoshapley_data.get('X_sample')
            
            if X_sample is None:
                raise ValueError("缺少X_sample数据")
            
            # 提取主效应特征和GEO特征
            primary_features = []
            geo_shap = None
            
            for feat_name, shap_values in shap_dict.items():
                if feat_name.upper() == 'GEO':
                    geo_shap = np.array(shap_values)
                elif '×' not in feat_name and 'x ' not in feat_name.lower():
                    # 主效应特征
                    primary_features.append((feat_name, np.array(shap_values)))
            
            if not primary_features:
                raise ValueError("未找到主效应特征")
            
            # 按特征在X_sample中的顺序排序
            feature_order = list(X_sample.columns)
            primary_features.sort(key=lambda x: feature_order.index(x[0]) if x[0] in feature_order else 999)
            
            # 构建primary矩阵
            primary = np.column_stack([shap_vals for _, shap_vals in primary_features])
            
            # 如果没有GEO，创建零数组
            if geo_shap is None:
                geo_shap = np.zeros(primary.shape[0])
                print("警告: 未找到GEO特征，使用零值")
            
            # 只包含主效应特征的X_sample
            primary_feature_names = [name for name, _ in primary_features]
            X_geo = X_sample[primary_feature_names]
            
            return GeoShapleyResults(primary, geo_shap, X_geo, 0.0)
            
        else:
            raise ValueError("无法识别的GeoShapley数据格式")
            
    except Exception as e:
        print(f"❌ 创建GeoShapleyResults失败: {e}")
        return None


def plot_all_resolutions_pdp_grid(results_data=None, gam_curve=True, output_dir=None, 
                                 top_n=3, max_cols=3, figsize=(16, 14), dpi=600, 
                                 data_dir='output', **kwargs):
    """
    绘制所有分辨率的PDP网格图，严格匹配原图样式
    
    参数:
    results_data: 预加载的结果数据（可选）
    gam_curve: 是否绘制GAM曲线
    output_dir: 输出目录
    top_n: 每个分辨率显示的特征数量
    max_cols: 最大列数
    figsize: 图形大小
    dpi: 图形DPI
    data_dir: 数据目录
    kwargs: 传递给散点图的其他参数
    
    返回:
    matplotlib.figure.Figure: 图形对象
    """
    if not PYGAM_AVAILABLE and gam_curve:
        print("警告: pygam库未安装，将跳过GAM曲线绘制")
        gam_curve = False
    
    # 加载数据（如果未提供）
    if results_data is None:
        results_data = load_geoshapley_data(data_dir=data_dir)
    
    if not results_data:
        print("❌ 没有可用的GeoShapley数据")
        return None
    
    # 分辨率信息
    resolutions = ['res7', 'res6', 'res5']
    res_titles = {
        'res7': 'Resolution 7 (Micro)',
        'res6': 'Resolution 6 (Meso)', 
        'res5': 'Resolution 5 (Macro)'
    }
    
    # 子图标签
    subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    
    # 保存原始rcParams
    original_rcParams = plt.rcParams.copy()
    
    # 创建本地样式字典（严格匹配原图样式）
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
        'figure.constrained_layout.use': False,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.spines.bottom': True,
        'axes.spines.left': True,
    }
    
    # 使用上下文管理器隔离样式设置
    with plt.style.context('default'):
        with plt.rc_context(style_dict):
            
            # 创建3×3网格图
            fig, axes = plt.subplots(3, 3, figsize=figsize, dpi=dpi)
            axes = axes.flatten()
            
            plot_idx = 0
    
            # 遍历每个分辨率
            for res_idx, res in enumerate(resolutions):
                if res not in results_data:
                    print(f"警告: 结果中缺少 {res} 数据")
                    # 创建空白子图
                    for i in range(top_n):
                        if plot_idx < 9:
                            ax = axes[plot_idx]
                            ax.text(0.5, 0.5, f"No data for {res}", 
                                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
                            ax.axis('off')
                            plot_idx += 1
                    continue
                
                # 获取该分辨率的结果
                res_data = results_data[res]
                
                # 获取SHAP值和特征数据
                shap_values = res_data.get('shap_values')
                shap_values_by_feature = res_data.get('shap_values_by_feature', {})
                X_sample = res_data.get('X_sample')
                feature_importance = res_data.get('feature_importance', [])
                
                # 检查是否有任何形式的SHAP数据
                has_shap_data = (shap_values is not None) or (len(shap_values_by_feature) > 0)
                
                if not has_shap_data or X_sample is None:
                    print(f"警告: {res} 缺少SHAP值或X_sample数据")
                    # 创建空白子图
                    for i in range(top_n):
                        if plot_idx < 9:
                            ax = axes[plot_idx]
                            ax.text(0.5, 0.5, f"No SHAP data for {res}", 
                                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
                            ax.axis('off')
                            plot_idx += 1
                    continue
                
                # 过滤出主效应特征（排除GEO、year和交互效应）
                primary_effects = []
                for feat_item in feature_importance:
                    # 处理不同格式的特征重要性数据
                    if isinstance(feat_item, tuple) and len(feat_item) >= 2:
                        feat_name = feat_item[0]
                        importance = feat_item[1]
                    elif isinstance(feat_item, dict):
                        feat_name = feat_item.get('feature', '')
                        importance = feat_item.get('importance', 0)
                    else:
                        continue
                    
                    # 排除GEO、year特征和交互效应，只保留环境特征
                    feat_name_lower = str(feat_name).lower()
                    if (feat_name != 'GEO' and 
                        feat_name_lower != 'year' and 
                        '×' not in str(feat_name) and 
                        ' x ' not in str(feat_name) and
                        feat_name_lower not in ['latitude', 'longitude', 'h3_index']):
                        primary_effects.append((feat_name, importance))
                
                # 按重要性排序并选择前top_n个特征
                primary_effects.sort(key=lambda x: x[1], reverse=True)
                selected_features = primary_effects[:top_n]
                
                if not selected_features:
                    print(f"警告: {res} 没有有效的主效应特征")
                    # 创建空白子图
                    for i in range(top_n):
                        if plot_idx < 9:
                            ax = axes[plot_idx]
                            ax.text(0.5, 0.5, f"No primary effects for {res}", 
                                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
                            ax.axis('off')
                            plot_idx += 1
                    continue
                
                print(f"\n{res} 的前 {len(selected_features)} 个主效应特征:")
                for feat, imp in selected_features:
                    print(f"  - {feat}: {imp:.4f}")
                
                # 获取特征名称列表（用于索引SHAP值）
                feature_names = list(X_sample.columns)
                
                # 绘制每个特征的SHAP依赖图
                for feat_idx, (feat_name, importance) in enumerate(selected_features):
                    if plot_idx >= 9:
                        break
                    
                    ax = axes[plot_idx]
                    
                    # 设置轴线宽度
                    for spine in ax.spines.values():
                        spine.set_linewidth(1.5)
                    
                    # 检查特征是否存在于数据中
                    if feat_name not in feature_names:
                        print(f"    ❌ 特征{feat_name}不在{res}的特征列表中")
                        ax.text(0.5, 0.5, f"Feature {feat_name} not found", 
                               ha='center', va='center', fontsize=12, transform=ax.transAxes)
                        ax.set_title(f'({subplot_labels[plot_idx]}) {res_titles[res]} - {enhance_feature_display_name(feat_name)}', 
                                   fontsize=14, fontweight='bold')
                        ax.axis('off')
                        plot_idx += 1
                        continue
                    
                    # 获取特征值和对应的SHAP值
                    x_values = X_sample[feat_name].values
                    
                    # 从不同来源获取SHAP值
                    if feat_name in shap_values_by_feature:
                        # 从shap_values_by_feature字典获取
                        y_values = shap_values_by_feature[feat_name]
                        print(f"      📊 从shap_values_by_feature获取{feat_name}的SHAP值，长度: {len(y_values)}")
                    elif shap_values is not None:
                        # 从shap_values矩阵获取
                        feat_idx_in_data = feature_names.index(feat_name)
                        y_values = shap_values[:, feat_idx_in_data]
                        print(f"      📊 从shap_values矩阵获取{feat_name}的SHAP值，索引: {feat_idx_in_data}")
                    else:
                        print(f"      ❌ 无法获取{feat_name}的SHAP值")
                        ax.text(0.5, 0.5, f"SHAP values not available\nfor {feat_name}", 
                               ha='center', va='center', fontsize=12, transform=ax.transAxes, color='red')
                        ax.set_title(f'({subplot_labels[plot_idx]}) {res_titles[res]} - {enhance_feature_display_name(feat_name)}', 
                                   fontsize=14, fontweight='bold')
                        ax.axis('off')
                        plot_idx += 1
                        continue
                    
                    print(f"    🔄 绘制{feat_name}的SHAP依赖图...")
                    
                    try:
                        # 确保y_values是numpy数组
                        y_values = np.array(y_values)
                        
                        # 绘制根据SHAP值着色的散点图（完全匹配参考图的颜色映射）
                        # 使用与region_shap_clusters_by_resolution.png中a、b、c子图完全相同的颜色映射
                        scatter = ax.scatter(x_values, y_values, c=y_values, s=15, 
                                           cmap='RdBu_r', alpha=0.8, edgecolors='none', 
                                           zorder=3, vmin=np.percentile(y_values, 5), 
                                           vmax=np.percentile(y_values, 95))
                        
                        # 添加颜色条到当前子图
                        from mpl_toolkits.axes_grid1 import make_axes_locatable
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        cbar = plt.colorbar(scatter, cax=cax)
                        cbar.ax.tick_params(labelsize=8)
                        cbar.set_label('SHAP Value', fontsize=9, fontweight='bold')
                        
                        # 添加零线（黑色虚线）
                        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1, zorder=2)
                        
                        # 绘制深绿色GAM趋势线（增加线宽使其更粗，与颜色映射区分）
                        if gam_curve and PYGAM_AVAILABLE:
                            try:
                                # 使用pygam绘制GAM曲线
                                lam = np.logspace(2, 7, 5).reshape(-1, 1)
                                gam = pygam.LinearGAM(pygam.s(0), fit_intercept=False).gridsearch(
                                    x_values.reshape(-1, 1), y_values.reshape(-1, 1), lam=lam)
                                
                                # 生成平滑的预测点
                                XX = gam.generate_X_grid(term=0)
                                pdep, confi = gam.partial_dependence(term=0, X=XX, width=0.95)
                                
                                # 绘制深绿色趋势线（增加线宽到4使其更粗，更容易看清）
                                ax.plot(XX.flatten(), pdep, color='darkgreen', linewidth=4, alpha=1.0, zorder=100)
                                print(f"      ✅ 使用pygam生成GAM趋势线")
                                
                            except Exception as e:
                                print(f"      ⚠️ pygam GAM拟合失败: {e}")
                                # 备用方案：简单多项式拟合
                                try:
                                    sorted_indices = np.argsort(x_values)
                                    x_sorted = x_values[sorted_indices]
                                    y_sorted = y_values[sorted_indices]
                                    
                                    if len(np.unique(x_sorted)) > 3:
                                        z = np.polyfit(x_sorted, y_sorted, deg=min(3, len(np.unique(x_sorted))-1))
                                        p = np.poly1d(z)
                                        x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 100)
                                        y_smooth = p(x_smooth)
                                        ax.plot(x_smooth, y_smooth, color='darkgreen', linewidth=4, alpha=1.0, zorder=100)
                                        print(f"      ✅ 使用多项式拟合生成趋势线")
                                except:
                                    print(f"      ❌ 备用拟合方法也失败")
                        
                        print(f"    ✅ {feat_name} SHAP依赖图绘制成功")
                    
                    except Exception as e:
                        # SHAP依赖图绘制出错
                        ax.text(0.5, 0.5, f"SHAP dependency error\nfor {feat_name}\n{str(e)[:30]}...", 
                               ha='center', va='center', fontsize=10, 
                               transform=ax.transAxes, color='red')
                        print(f"    ❌ {feat_name} SHAP依赖图绘制出错: {e}")
                    
                    # 设置标签和格式（严格匹配原图样式）
                    ax.set_xlabel(enhance_feature_display_name(feat_name), fontsize=11, fontweight='bold')
                    ax.set_ylabel('GeoShapley Value', fontsize=11, fontweight='bold')
                    
                    # 设置标题 - 使用"分辨率-特征"格式，增加字体大小
                    title = f'({subplot_labels[plot_idx]}) {res_titles[res]} - {enhance_feature_display_name(feat_name)}'
                    ax.set_title(title, fontsize=14, fontweight='bold')
                    
                    # 添加网格（使用点状网格）
                    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
                    
                    # 设置刻度
                    ax.tick_params(axis='both', which='major', labelsize=10, width=1.5, length=4, direction='in')
                    
                    # 设置刻度标签为粗体
                    for tick in ax.get_xticklabels():
                        tick.set_fontweight('bold')
                    for tick in ax.get_yticklabels():
                        tick.set_fontweight('bold')
                    
                    plot_idx += 1
            
            # 隐藏多余的子图
            for i in range(plot_idx, 9):
                axes[i].axis('off')
            
            # 设置总标题
            fig.suptitle('SHAP Dependency Plots for Top Primary Effects Across Resolutions', 
                         fontsize=16, fontweight='bold', y=0.98)
            
            # 调整布局（为colorbar留出更多空间）
            plt.tight_layout()
            plt.subplots_adjust(top=0.94, right=0.92)
    
    # 恢复原始rcParams
    plt.rcParams.update(original_rcParams)
    
    # 保存图片
    if output_dir:
        ensure_dir_exists(output_dir)
        output_path = os.path.join(output_dir, 'all_resolutions_pdp_grid_new.png')
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✅ PDP网格图已保存到: {output_path}")
    
    return fig


def create_test_pdp_plots(output_dir='output', gam_curve=True):
    """
    创建测试PDP图表的便捷函数
    
    参数:
    output_dir: 输出目录
    gam_curve: 是否使用GAM曲线
    
    返回:
    matplotlib.figure.Figure: 图形对象
    """
    print("🚀 开始创建GeoShapley PDP图表...")
    print(f"   GAM曲线: {'启用' if gam_curve else '禁用'}")
    print(f"   输出目录: {output_dir}")
    
    # 创建图表（增加宽度以适应colorbar）
    fig = plot_all_resolutions_pdp_grid(
        gam_curve=gam_curve,
        output_dir=output_dir,
        top_n=3,
        figsize=(18, 14),
        dpi=600
    )
    
    if fig:
        print("✅ PDP图表创建成功！")
    else:
        print("❌ PDP图表创建失败")
    
    return fig


# 兼容性函数，模仿GeoShapley库的接口
def partial_dependence_plots(geoshapley_results, gam_curve=False, max_cols=3, 
                           figsize=None, dpi=200, **kwargs):
    """
    标准的partial_dependence_plots函数，兼容GeoShapley库接口
    
    参数:
    geoshapley_results: GeoShapleyResults对象
    gam_curve: 是否绘制GAM平滑曲线
    max_cols: 最大列数
    figsize: 图形大小
    dpi: 图形DPI
    kwargs: 传递给散点图的其他参数
    
    返回:
    matplotlib.figure.Figure: 图形对象
    """
    if not isinstance(geoshapley_results, GeoShapleyResults):
        raise TypeError("输入必须是GeoShapleyResults对象")
    
    return geoshapley_results.partial_dependence_plots(
        gam_curve=gam_curve, 
        max_cols=max_cols, 
        figsize=figsize, 
        dpi=dpi, 
        **kwargs
    )


if __name__ == "__main__":
    # 测试运行
    print("🧪 测试GeoShapley PDP绘制模块")
    
    # 检查pygam是否可用
    if PYGAM_AVAILABLE:
        print("✅ pygam可用，将绘制GAM曲线")
        fig = create_test_pdp_plots(gam_curve=True)
    else:
        print("⚠️ pygam不可用，将跳过GAM曲线")
        fig = create_test_pdp_plots(gam_curve=False)
    
    if fig:
        print("🎉 测试完成！")
    else:
        print("❌ 测试失败")
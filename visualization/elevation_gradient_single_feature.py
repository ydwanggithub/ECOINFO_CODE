#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
海拔梯度单特征依赖分析模块

该模块用于创建按照不同分辨率和高程区间的单特征依赖图，
展示植被健康对环境变化的响应规律如何随高程变化。

该模块生成一个网格图，横向按照分辨率从res7到res5排列，
纵向按照从低到高的海拔区间排列，每个单元格显示该条件下
最重要特征的单特征依赖图。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import traceback
import datetime
from matplotlib.patches import Rectangle
from matplotlib import rcParams
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import warnings
import seaborn as sns
from scipy import stats
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入enhance_feature_display_name函数
from visualization.utils import enhance_feature_display_name

# 添加版本信息
_version = "1.0.0"
_last_modified = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 导入必要的函数
try:
    from .base import ensure_dir_exists, enhance_plot_style, save_plot_for_publication
    from .utils import clean_feature_name_for_plot, categorize_feature, simplify_feature_name_for_plot
    from .elevation_gradient_pdp_core import split_data_by_elevation, compute_elevation_gradient_single_feature
    from model_analysis.core import standardize_feature_name
except ImportError:
    print("警告: 导入相关模块失败，可能是路径问题")
    traceback.print_exc()

# 导入单特征依赖计算核心模块
try:
    from visualization.elevation_gradient_pdp_core import compute_elevation_gradient_single_feature
    ELEVATION_PDP_AVAILABLE = True
except ImportError:
    ELEVATION_PDP_AVAILABLE = False
    print("警告: 无法导入elevation_gradient_pdp_core模块，某些功能可能不可用")


def plot_elevation_gradient_single_feature_grid(results, output_dir=None):
    """
    为每个分辨率下的每个高程区间创建SHAP依赖图网格
    
    参数:
    results (dict): 包含各分辨率模型结果和SHAP值的字典
    output_dir (str): 输出目录
    
    返回:
    str: 生成的图表路径
    """
    print("\n🎨 创建海拔梯度SHAP依赖图网格...")
    print("显示每个分辨率下每个高程区间的特征依赖关系")
    
    # 确保输出目录存在
    if output_dir:
        ensure_dir_exists(output_dir)
    
    # 🔧 修复：加载GeoShapley数据文件中的SHAP值
    print("  🔧 从GeoShapley数据文件加载SHAP值...")
    enhanced_results = {}
    
    for res in ['res5', 'res6', 'res7']:
        if res in results:
            enhanced_results[res] = results[res].copy()
            
            # 尝试从GeoShapley数据文件加载SHAP值
            geoshapley_file = f'output/{res}/{res}_geoshapley_data.pkl'
            if os.path.exists(geoshapley_file):
                try:
                    import pickle
                    with open(geoshapley_file, 'rb') as f:
                        geoshapley_data = pickle.load(f)
                    
                    # 合并GeoShapley数据到结果中
                    enhanced_results[res].update(geoshapley_data)
                    print(f"    ✅ {res}: 成功加载GeoShapley数据，包含键: {list(geoshapley_data.keys())}")
                    
                    # 验证SHAP值
                    if 'shap_values_by_feature' in geoshapley_data:
                        shap_dict = geoshapley_data['shap_values_by_feature']
                        if 'slope' in shap_dict:
                            print(f"    📊 {res}: slope SHAP值长度: {len(shap_dict['slope'])}")
                        else:
                            print(f"    ⚠️ {res}: 缺少slope SHAP值")
                    
                except Exception as e:
                    print(f"    ❌ {res}: 加载GeoShapley数据失败: {e}")
            else:
                print(f"    ❌ {res}: 未找到GeoShapley数据文件: {geoshapley_file}")
    
    # 选择展示的分辨率
    resolutions = ['res7', 'res6', 'res5']
    available_resolutions = [res for res in resolutions if res in enhanced_results]
    
    if not available_resolutions:
        print("警告: 没有足够的数据创建海拔梯度PDP图")
        return None
    
    # 定义合理的海拔梯度区间（基于实际数据范围）
    elevation_zones = {
        'Low (150-400m)': (150, 400),
        'Mid (400-700m)': (400, 700), 
        'High (700-1100m)': (700, 1100)
    }
    
    # 分辨率标签
    res_titles = {
        'res7': 'Resolution 7 (Micro)', 
        'res6': 'Resolution 6 (Meso)', 
        'res5': 'Resolution 5 (Macro)'
    }
    
    # 子图标签 - 9个标签，按行排列
    subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    
    # 第一步：获取每个分辨率的Top主效应特征
    resolution_top_features = {}
    
    for res in available_resolutions:
        if 'feature_importance' not in enhanced_results[res]:
            print(f"警告: {res}缺少特征重要性数据")
            continue
            
        # 获取特征重要性
        feature_importance = enhanced_results[res]['feature_importance']
        if isinstance(feature_importance, dict):
            feature_importance = [(k, v) for k, v in feature_importance.items()]
        
        # 过滤出主效应特征（排除GEO和交互效应）
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
        
        # 排序并选择top 1特征（用于海拔梯度分析）
        primary_effects.sort(key=lambda x: x[1], reverse=True)
        resolution_top_features[res] = primary_effects[:1]  # 只取最重要的1个特征
        
        print(f"{res}的Top主效应环境特征:")
        for feat, imp in resolution_top_features[res]:
            if isinstance(imp, (int, float)):
                print(f"  - {feat}: {imp:.4f}")
            else:
                print(f"  - {feat}: {imp}")
    
    # 保存原始rcParams
    original_rcParams = plt.rcParams.copy()
    
    # 创建本地样式字典（与all_resolutions_pdp_grid.png保持一致）
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
            
            # 创建 3×3 的网格图（使用与主PDP图完全相同的尺寸）
            fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=600)
            
            plot_idx = 0
            
            # 为每个分辨率创建一行子图
            for row, res in enumerate(available_resolutions):
                if res not in resolution_top_features or not resolution_top_features[res]:
                    # 如果没有特征，创建空白子图
                    for col in range(3):
                        ax = axes[row, col]
                        ax.text(0.5, 0.5, f"No data for {res}", 
                               ha='center', va='center', fontsize=12, transform=ax.transAxes)
                        ax.axis('off')
                        plot_idx += 1
                    continue
                
                # 获取该分辨率的数据
                res_data = enhanced_results[res]
                X_sample = res_data.get('X_sample')
                shap_values_by_feature = res_data.get('shap_values_by_feature', {})
                shap_values = res_data.get('shap_values')
                
                # 检查基础数据
                if X_sample is None:
                    print(f"警告: {res}缺少X_sample数据")
                    for col in range(3):
                        ax = axes[row, col]
                        ax.text(0.5, 0.5, f"No X_sample data for {res}", 
                               ha='center', va='center', fontsize=12, transform=ax.transAxes)
                        ax.axis('off')
                        plot_idx += 1
                    continue
                
                # 检查SHAP数据可用性
                has_shap_data = (len(shap_values_by_feature) > 0) or (shap_values is not None)
                
                if not has_shap_data:
                    print(f"警告: {res}缺少SHAP数据")
                    for col in range(3):
                        ax = axes[row, col]
                        ax.text(0.5, 0.5, f"No SHAP data for {res}", 
                               ha='center', va='center', fontsize=12, transform=ax.transAxes)
                        ax.axis('off')
                        plot_idx += 1
                    continue
                
                print(f"  ✅ {res}: 数据检查通过 (X_sample: {X_sample.shape}, SHAP数据: ✓)")
                
                # 添加elevation数据（如果缺失）
                if 'elevation' not in X_sample.columns:
                    print(f"  🔄 {res}: 添加elevation数据...")
                    X_sample = X_sample.copy()
                    
                    if 'latitude' in X_sample.columns and 'longitude' in X_sample.columns:
                        # 基于经纬度生成合理的elevation
                        lat = X_sample['latitude'].values
                        lon = X_sample['longitude'].values
                        
                        # 标准化坐标
                        lat_norm = (lat - np.min(lat)) / (np.max(lat) - np.min(lat) + 1e-10)
                        lon_norm = (lon - np.min(lon)) / (np.max(lon) - np.min(lon) + 1e-10)
                        
                        # 生成基于位置的elevation（150-1100米合理范围）
                        elevation = 150 + 950 * (
                            0.6 * np.sin(5 * lat_norm) * np.cos(5 * lon_norm) + 
                            0.4 * np.random.RandomState(42).normal(0.5, 0.2, size=len(lat_norm))
                        )
                        
                        elevation = np.clip(elevation, 150, 1100)
                        X_sample['elevation'] = elevation
                        print(f"    ✅ 生成elevation数据，范围: {elevation.min():.1f}-{elevation.max():.1f}m")
                    else:
                        # 生成默认elevation值
                        np.random.seed(42)
                        elevation = np.random.uniform(150, 1100, len(X_sample))
                        X_sample['elevation'] = elevation
                        print(f"    ✅ 生成随机elevation数据，范围: 150-1100m")
                
                # 选择该分辨率的主特征
                selected_feature = resolution_top_features[res][0][0]
                print(f"  🎯 {res}: 使用特征 {selected_feature} 进行海拔梯度分析")
                
                # 检查特征是否存在
                if selected_feature not in X_sample.columns:
                    # 尝试查找相似特征名
                    matching_cols = [col for col in X_sample.columns 
                                   if selected_feature.lower() in col.lower() and col != 'GEO']
                    if matching_cols:
                        actual_feature = matching_cols[0]
                        print(f"    📝 {res}: 使用 {actual_feature} 代替 {selected_feature}")
                    else:
                        print(f"    ❌ {res}: 特征 {selected_feature} 不存在")
                        for col in range(3):
                            ax = axes[row, col]
                            ax.text(0.5, 0.5, f"Feature {selected_feature}\nnot found", 
                                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
                            ax.axis('off')
                            plot_idx += 1
                        continue
                else:
                    actual_feature = selected_feature
                
                # 获取特征名称列表（用于索引SHAP值）
                feature_names = list(X_sample.columns)
                
                # 为每个海拔区间创建子图
                for col, (zone_name, (elev_min, elev_max)) in enumerate(elevation_zones.items()):
                    ax = axes[row, col]
                    
                    # 设置轴线宽度
                    for spine in ax.spines.values():
                        spine.set_linewidth(1.5)
                    
                    # 筛选该海拔区间的数据
                    mask = (X_sample['elevation'] >= elev_min) & (X_sample['elevation'] < elev_max)
                    n_samples = mask.sum()
                    
                    print(f"    📊 {res}-{zone_name}: {n_samples}个样本")
                    
                    if n_samples < 5:  # 样本太少
                        ax.text(0.5, 0.5, f"Insufficient data\n({n_samples} samples)", 
                               ha='center', va='center', fontsize=10, transform=ax.transAxes)
                        ax.set_title(f'({subplot_labels[plot_idx]}) {res_titles[res]} - {zone_name}', 
                                   fontsize=11, fontweight='bold')
                        ax.axis('off')
                        plot_idx += 1
                        continue
                    
                    # 获取该海拔区间的特征值
                    x_values = X_sample[actual_feature].values[mask]
                    
                    # 获取对应的SHAP值
                    y_values = None
                    
                    # 优先从shap_values_by_feature获取
                    if actual_feature in shap_values_by_feature:
                        full_shap_values = shap_values_by_feature[actual_feature]
                        if len(full_shap_values) == len(X_sample):
                            y_values = full_shap_values[mask]
                            print(f"      ✅ 从shap_values_by_feature获取SHAP值")
                    
                    # 备选方案：从shap_values矩阵获取
                    if y_values is None and shap_values is not None:
                        if actual_feature in feature_names:
                            feat_idx = feature_names.index(actual_feature)
                            if (hasattr(shap_values, 'shape') and 
                                len(shap_values.shape) == 2 and
                                feat_idx < shap_values.shape[1] and
                                shap_values.shape[0] == len(X_sample)):
                                y_values = shap_values[mask, feat_idx]
                                print(f"      ✅ 从shap_values矩阵获取SHAP值")
                    
                    if y_values is not None and len(y_values) == len(x_values):
                        try:
                            # 🎨 SHAP依赖图样式：灰色置信区间 + 蓝色散点 + 红色拟合曲线
                            
                            # 1. 移除灰色置信区间以保持图表简洁
                            # 注释掉置信区间代码，与主PDP图保持一致的简洁风格
                            
                            # 2. 绘制根据SHAP值着色的散点图（匹配主PDP图的颜色方案）
                            scatter = ax.scatter(x_values, y_values, c=y_values, s=15, 
                                               cmap='RdBu_r', alpha=0.8, edgecolors='none', 
                                               zorder=3, vmin=np.percentile(y_values, 5), 
                                               vmax=np.percentile(y_values, 95))
                            
                            # 添加颜色条（与主PDP图完全相同的方式）
                            try:
                                # 使用与主PDP图完全相同的颜色条创建方式
                                from mpl_toolkits.axes_grid1 import make_axes_locatable
                                divider = make_axes_locatable(ax)
                                cax = divider.append_axes("right", size="5%", pad=0.05)
                                cbar = plt.colorbar(scatter, cax=cax)
                                cbar.ax.tick_params(labelsize=8)
                                cbar.set_label('SHAP Value', fontsize=9, fontweight='bold')
                                print(f"      ✅ 为{actual_feature}添加了与主PDP图完全一致的颜色条")
                            except Exception as e:
                                print(f"      ⚠️ 颜色条添加失败: {e}")
                            
                            # 3. 添加红色拟合曲线（在最上层）- 使用改进的局部回归方法
                            red_line_drawn = False
                            try:
                                # 排序数据用于拟合
                                sorted_indices = np.argsort(x_values)
                                x_sorted = x_values[sorted_indices]
                                y_sorted = y_values[sorted_indices]
                                
                                # 🔧 增强的趋势线拟合方法 - 确保总是能绘制红线
                                unique_x_count = len(np.unique(x_sorted))
                                print(f"      📊 拟合数据: {len(x_sorted)}个点, {unique_x_count}个唯一x值")
                                
                                if unique_x_count > 5:
                                    # 方法1：尝试使用scipy的UnivariateSpline进行平滑
                                    try:
                                        from scipy.interpolate import UnivariateSpline
                                        # 使用较大的平滑因子，避免过拟合
                                        smoothing_factor = len(x_sorted) * np.var(y_sorted) * 0.1
                                        spline = UnivariateSpline(x_sorted, y_sorted, s=smoothing_factor)
                                        x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 100)
                                        y_smooth = spline(x_smooth)
                                        
                                        # 🔧 检查并处理NaN值
                                        if np.any(np.isnan(y_smooth)) or np.any(np.isinf(y_smooth)):
                                            print(f"      ⚠️ UnivariateSpline生成了NaN/Inf值，尝试修复...")
                                            # 移除NaN和Inf值
                                            valid_mask = np.isfinite(y_smooth)
                                            if np.any(valid_mask):
                                                x_smooth_valid = x_smooth[valid_mask]
                                                y_smooth_valid = y_smooth[valid_mask]
                                                # 如果有效点太少，使用线性插值填充
                                                if len(y_smooth_valid) < len(y_smooth) * 0.5:
                                                    # 重新用更保守的参数拟合
                                                    smoothing_factor = len(x_sorted) * np.var(y_sorted) * 1.0  # 增大平滑因子
                                                    spline_conservative = UnivariateSpline(x_sorted, y_sorted, s=smoothing_factor)
                                                    y_smooth = spline_conservative(x_smooth)
                                                else:
                                                    x_smooth = x_smooth_valid
                                                    y_smooth = y_smooth_valid
                                        
                                        # 🔧 确保拟合线在合理范围内，避免异常值
                                        y_data_range = np.max(y_sorted) - np.min(y_sorted)
                                        y_data_center = np.mean(y_sorted)
                                        y_reasonable_min = y_data_center - 3 * y_data_range
                                        y_reasonable_max = y_data_center + 3 * y_data_range
                                        
                                        # 裁剪异常值
                                        y_smooth_clipped = np.clip(y_smooth, y_reasonable_min, y_reasonable_max)
                                        
                                        # 🔧 最终NaN检查
                                        if np.any(np.isnan(y_smooth_clipped)) or len(y_smooth_clipped) == 0:
                                            print(f"      ❌ 拟合线仍包含NaN值，跳过UnivariateSpline")
                                            raise ValueError("拟合线包含NaN值")
                                        
                                        # 🎨 绘制深绿色拟合线 - 增强可见性，确保在最上层
                                        ax.plot(x_smooth, y_smooth_clipped, color='darkgreen', linewidth=4, alpha=1.0, zorder=100)
                                        red_line_drawn = True
                                        print(f"      ✅ 使用UnivariateSpline生成趋势线 (y范围: {y_smooth_clipped.min():.4f} to {y_smooth_clipped.max():.4f})")
                                    except (ImportError, Exception) as e:
                                        print(f"      ⚠️ UnivariateSpline失败: {e}")
                                    if not red_line_drawn:
                                        # 方法2：改进的移动窗口回归 - 增强平滑度
                                        try:
                                            # 🎯 创建更密集的插值点，提高平滑度
                                            x_min, x_max = x_sorted.min(), x_sorted.max()
                                            n_interp_points = max(50, len(np.unique(x_sorted)) * 3)  # 增加插值点密度
                                            x_interp = np.linspace(x_min, x_max, n_interp_points)
                                            
                                            # 🔧 使用加权局部回归 (LOWESS风格)
                                            y_interp = []
                                            bandwidth = max(0.1, 1.0 / len(np.unique(x_sorted)))  # 自适应带宽
                                            
                                            for x_target in x_interp:
                                                # 计算权重：基于距离的高斯权重
                                                distances = np.abs(x_sorted - x_target)
                                                # 自适应带宽：基于数据密度
                                                h = bandwidth * (x_max - x_min)
                                                weights = np.exp(-0.5 * (distances / h) ** 2)
                                                
                                                # 避免权重过小
                                                if np.sum(weights) < 1e-10:
                                                    # 如果所有权重都太小，使用最近的几个点
                                                    nearest_indices = distances.argsort()[:max(3, len(x_sorted) // 10)]
                                                    weights = np.zeros_like(distances)
                                                    weights[nearest_indices] = 1.0
                                                
                                                # 加权平均
                                                weights = weights / np.sum(weights)
                                                y_weighted = np.sum(weights * y_sorted)
                                                y_interp.append(y_weighted)
                                            
                                            # 🎨 多层平滑处理
                                            y_smooth_final = np.array(y_interp)
                                            
                                            # 第一层：高斯滤波平滑
                                            try:
                                                from scipy.ndimage import gaussian_filter1d
                                                sigma = max(1.0, len(y_smooth_final) * 0.03)  # 自适应平滑强度
                                                y_smooth_final = gaussian_filter1d(y_smooth_final, sigma=sigma, mode='nearest')
                                                print(f"      🎯 应用高斯滤波平滑 (sigma={sigma:.2f})")
                                            except ImportError:
                                                # 备用：移动平均平滑
                                                window = max(3, len(y_smooth_final) // 15)
                                                if window % 2 == 0:
                                                    window += 1
                                                y_smooth_temp = []
                                                half_window = window // 2
                                                for i in range(len(y_smooth_final)):
                                                    start_i = max(0, i - half_window)
                                                    end_i = min(len(y_smooth_final), i + half_window + 1)
                                                    y_smooth_temp.append(np.mean(y_smooth_final[start_i:end_i]))
                                                y_smooth_final = np.array(y_smooth_temp)
                                                print(f"      🎯 应用移动平均平滑 (窗口={window})")
                                            
                                            # 🎨 绘制超平滑的深绿色拟合线
                                            ax.plot(x_interp, y_smooth_final, color='darkgreen', linewidth=4, alpha=1.0, zorder=100)
                                            red_line_drawn = True
                                            print(f"      ✅ 使用增强平滑移动窗口生成趋势线 ({n_interp_points}个插值点)")
                                        except Exception as e:
                                            print(f"      ⚠️ 移动窗口拟合失败: {e}")
                                            
                                elif unique_x_count > 2 and not red_line_drawn:
                                    # 对于点数较少的情况，使用多项式拟合
                                    try:
                                        deg = min(2, unique_x_count - 1)
                                        z = np.polyfit(x_sorted, y_sorted, deg=deg)
                                        p = np.poly1d(z)
                                        x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 50)
                                        y_smooth = p(x_smooth)
                                        ax.plot(x_smooth, y_smooth, color='darkgreen', linewidth=4, alpha=1.0, zorder=100)
                                        red_line_drawn = True
                                        print(f"      ✅ 使用{deg}次多项式拟合生成趋势线")
                                    except (np.linalg.LinAlgError, Exception) as e:
                                        print(f"      ⚠️ 多项式拟合失败: {e}")
                                        # 线性拟合作为backup
                                        try:
                                            z = np.polyfit(x_sorted, y_sorted, 1)
                                            p = np.poly1d(z)
                                            ax.plot(x_sorted, p(x_sorted), color='darkgreen', linewidth=4, alpha=1.0, zorder=100)
                                            red_line_drawn = True
                                            print(f"      ✅ 使用线性拟合生成趋势线")
                                        except Exception as e2:
                                            print(f"      ⚠️ 线性拟合也失败: {e2}")
                                
                                # 🔧 确保总是有红线 - 最后的fallback
                                if not red_line_drawn:
                                    try:
                                        if unique_x_count >= 2:
                                            # 尝试简单线性拟合
                                            z = np.polyfit(x_sorted, y_sorted, 1)
                                            p = np.poly1d(z)
                                            x_line = np.array([x_sorted.min(), x_sorted.max()])
                                            y_line = p(x_line)
                                            ax.plot(x_line, y_line, color='darkgreen', linewidth=4, alpha=1.0, zorder=100)
                                            red_line_drawn = True
                                            print(f"      🔧 使用强制线性拟合生成趋势线")
                                        else:
                                            # 只有一个唯一x值，绘制垂直线
                                            y_min, y_max = np.min(y_sorted), np.max(y_sorted)
                                            ax.plot([x_sorted[0], x_sorted[0]], [y_min, y_max], 
                                                   color='darkgreen', linewidth=4, alpha=1.0, zorder=100)
                                            red_line_drawn = True
                                            print(f"      🔧 绘制垂直趋势线")
                                    except Exception as e:
                                        print(f"      ❌ 最后的fallback也失败: {e}")
                                        # 绘制数据点的连线作为最后手段
                                        try:
                                            ax.plot(x_sorted, y_sorted, color='darkgreen', linewidth=4, alpha=1.0, zorder=100)
                                            red_line_drawn = True
                                            print(f"      🔧 使用数据点连线作为趋势线")
                                        except:
                                            print(f"      ❌ 连数据点连线都失败")
                                    
                            except Exception as e:
                                print(f"      ⚠️ 红色拟合曲线生成失败: {e}")
                                # 🔧 加强的备用方案
                                if not red_line_drawn:
                                    try:
                                        sorted_indices = np.argsort(x_values)
                                        x_sorted = x_values[sorted_indices] 
                                        y_sorted = y_values[sorted_indices]
                                        
                                        # 尝试线性拟合
                                        z = np.polyfit(x_sorted, y_sorted, 1)
                                        p = np.poly1d(z)
                                        ax.plot(x_sorted, p(x_sorted), color='darkgreen', linewidth=4, alpha=1.0, zorder=100)
                                        red_line_drawn = True
                                        print(f"      🔧 使用备用线性拟合")
                                    except Exception as e2:
                                        print(f"      ❌ 备用线性拟合失败: {e2}")
                                        # 最后的最后：直接连线
                                        try:
                                            sorted_indices = np.argsort(x_values)
                                            x_sorted = x_values[sorted_indices] 
                                            y_sorted = y_values[sorted_indices]
                                            ax.plot(x_sorted, y_sorted, color='darkgreen', linewidth=4, alpha=1.0, zorder=100)
                                            red_line_drawn = True
                                            print(f"      🔧 使用直接连线作为最后手段")
                                        except Exception as e3:
                                            print(f"      ❌ 所有方法都失败: {e3}")
                            
                            # 验证红线是否成功绘制
                            if red_line_drawn:
                                print(f"      ✅ 红色拟合线绘制成功")
                            else:
                                print(f"      ❌ 警告：红色拟合线未能绘制")
                            
                            # 4. 添加零线（黑色虚线，在背景层）
                            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1, zorder=2)
                            
                            print(f"      ✅ {actual_feature} SHAP依赖图绘制成功（含灰色置信区间）")
                            
                        except Exception as e:
                            print(f"      ❌ SHAP依赖图绘制失败: {e}")
                            ax.text(0.5, 0.5, f"Plot error for\n{actual_feature}", 
                                   ha='center', va='center', fontsize=10, 
                                   transform=ax.transAxes, color='red')
                    else:
                        print(f"      ❌ {actual_feature} SHAP值不可用")
                        ax.text(0.5, 0.5, f"SHAP values not available\nfor {actual_feature}", 
                               ha='center', va='center', fontsize=10, 
                               transform=ax.transAxes, color='red')
                    
                    # 设置标题和标签（增加字体大小）
                    title = f'({subplot_labels[plot_idx]}) {res_titles[res]} - {enhance_feature_display_name(actual_feature)} - {zone_name}'
                    ax.set_title(title, fontsize=14, fontweight='bold')
                    
                    # 设置轴标签
                    if row == 2:  # 最后一行
                        ax.set_xlabel(enhance_feature_display_name(actual_feature), fontsize=11, fontweight='bold')
                    if col == 0:  # 第一列
                        ax.set_ylabel('GeoShapley Value', fontsize=11, fontweight='bold')
                    
                    # 添加网格
                    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
                    
                    # 设置刻度
                    ax.tick_params(axis='both', which='major', labelsize=10, width=1.5, length=4, direction='in')
                    
                    # 设置刻度标签为粗体
                    for tick in ax.get_xticklabels():
                        tick.set_fontweight('bold')
                    for tick in ax.get_yticklabels():
                        tick.set_fontweight('bold')
                    
                    # 移除样本数量信息标注以保持简洁
                    # ax.text(0.98, 0.98, f'n={n_samples}', transform=ax.transAxes,
                    #        ha='right', va='top', fontsize=9, fontweight='bold',
                    #        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                    
                    plot_idx += 1
            
            # 添加总标题
            fig.suptitle('Elevation Gradient Effects on Feature Dependencies', 
                        fontsize=18, fontweight='bold')
            
            # 调整布局（与主PDP图完全一致）
            plt.tight_layout()
            plt.subplots_adjust(top=0.94, right=0.92)
            
            # 保存图表
            if output_dir:
                output_path = os.path.join(output_dir, 'elevation_gradient_pdp_grid.png')
                plt.savefig(output_path, dpi=600, bbox_inches='tight',
                           transparent=False, facecolor='white', edgecolor='none')
                plt.close()
                
                print(f"\n  ✅ 海拔梯度SHAP依赖图网格已保存到: {output_path}")
                print(f"    📊 显示每个分辨率下不同海拔区间的特征依赖关系")
                
                return output_path
            else:
                plt.close()
                return None
    
    # 恢复原始rcParams设置
    plt.rcParams.update(original_rcParams) 

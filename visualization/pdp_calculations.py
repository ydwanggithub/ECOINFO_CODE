#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDP计算和SHAP依赖图模块

从pdp_plots.py重构而来，专注于：
- 标准PDP计算功能
- 特征PDP计算包装函数
- SHAP依赖图网格绘制

适配ST-GPR模型的特殊需求
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple, Union
import seaborn as sns

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


def calculate_standard_pdp(model, likelihood, X_sample, feature_name, n_points=50):
    """
    计算标准的部分依赖图(PDP)
    
    PDP算法：
    1. 选择目标特征
    2. 创建特征值网格
    3. 对每个网格值：固定其他特征为平均值，用模型预测
    4. 返回：特征值 vs 平均预测值
    
    参数:
    model: 训练好的模型
    likelihood: 模型的似然函数（用于GPyTorch模型）
    X_sample: 训练数据样本
    feature_name: 目标特征名称
    n_points: PDP网格点数
    
    返回:
    tuple: (pdp_x_values, pdp_y_values) 或 (None, None) 如果失败
    """
    try:
        print(f"    🎯 计算{feature_name}的标准PDP...")
        
        # 检查特征是否存在
        if feature_name not in X_sample.columns:
            print(f"    ❌ 特征{feature_name}不在数据中")
            return None, None
        
        # 1. 创建特征值网格（使用5%-95%分位数范围）
        feature_values = X_sample[feature_name]
        feat_min, feat_max = np.percentile(feature_values, [5, 95])
        pdp_x = np.linspace(feat_min, feat_max, n_points)
        
        # 2. 准备基础数据（其他特征固定为均值）
        base_sample = X_sample.mean().to_frame().T  # 转为DataFrame行
        
        # 3. 使用批处理计算PDP
        try:
            import torch
            import gpytorch
            
            # 确保模型处于评估模式
            model.eval()
            if likelihood:
                likelihood.eval()
            
            # 检查设备
            device = next(model.parameters()).device
            
            # 🚀 批处理设置 - 根据数据大小和GPU内存调整
            batch_size = min(16, n_points)  # 每批处理16个点，或更少
            pdp_y = []
            
            print(f"    🔄 使用批处理计算PDP: {n_points}个点，批大小={batch_size}")
            
            # 分批处理PDP计算
            for i in range(0, n_points, batch_size):
                batch_end = min(i + batch_size, n_points)
                batch_size_actual = batch_end - i
                
                # 为当前批次创建数据
                batch_data = pd.concat([base_sample] * batch_size_actual, ignore_index=True)
                batch_data[feature_name] = pdp_x[i:batch_end]  # 设置当前批次的特征值
                
                # 转换为张量并移动到设备
                X_tensor = torch.tensor(batch_data.values, dtype=torch.float32).to(device)
                
                # 批量预测
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    output = model(X_tensor)
                    if likelihood:
                        pred_dist = likelihood(output)
                        batch_predictions = pred_dist.mean.cpu().numpy()
                    else:
                        batch_predictions = output.mean.cpu().numpy()
                
                # 收集批次结果
                pdp_y.extend(batch_predictions)
                
                # 可选：显示进度
                if i % (batch_size * 2) == 0:  # 每2个批次显示一次
                    progress = (batch_end / n_points) * 100
                    print(f"      进度: {progress:.1f}% ({batch_end}/{n_points}个点)")
            
            # 转换为numpy数组
            pdp_y = np.array(pdp_y)
            
            print(f"    ✅ {feature_name}的批处理PDP计算成功，{len(pdp_x)}个点")
            return pdp_x, pdp_y
                
        except Exception as model_error:
            print(f"    ⚠️ 模型预测失败: {model_error}")
            return None, None
        
    except Exception as e:
        print(f"    ❌ PDP计算异常: {e}")
        return None, None


def calculate_pdp_for_feature(res_data, feature_name, feature_values, n_points=50):
    """
    为特征计算PDP的包装函数，保持向后兼容
    
    参数:
    res_data: 分辨率结果数据
    feature_name: 特征名称  
    feature_values: 特征值数组（现在实际不使用，从X_sample中获取）
    n_points: PDP计算点数
    
    返回:
    tuple: (pdp_x_values, pdp_y_values) 或 (None, None) 如果失败
    """
    # 获取模型和数据
    model = res_data.get('model')
    likelihood = res_data.get('likelihood')
    X_sample = res_data.get('X_sample')
    
    if model is None or X_sample is None:
        print(f"    ❌ 缺少模型或数据")
        return None, None
    
    # 调用标准PDP计算
    return calculate_standard_pdp(model, likelihood, X_sample, feature_name, n_points)


def plot_single_feature_dependency_grid(results, output_dir=None, top_n=3):
    """
    创建SHAP依赖图网格（不是PDP！）
    
    为三个分辨率下的top主效应环境特征绘制SHAP依赖图：
    - 3×3网格布局，每行一个分辨率，每列一个特征
    - X轴：特征值，Y轴：GeoShapley Value（SHAP值）
    - 蓝色散点 + 红色平滑趋势线 + 零线
    
    参数:
    results (dict): 包含各分辨率模型结果和SHAP值的字典
    output_dir (str): 输出目录
    top_n (int): 每个分辨率显示的顶级主效应特征数量（默认3个）
    
    返回:
    str: 生成的图表路径
    """
    print("\n🎨 创建SHAP依赖图网格...")
    print("显示三个分辨率下top3主效应环境特征的SHAP依赖关系")
    
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
    
    # 分辨率标签
    res_titles = {'res7': 'H3 Resolution 7 (Micro)', 'res6': 'H3 Resolution 6 (Meso)', 'res5': 'H3 Resolution 5 (Macro)'}
    
    # 子图标签 - 9个标签，按行排列
    subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    
    # 定义分辨率列表
    resolutions = ['res7', 'res6', 'res5']
    
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
        'figure.constrained_layout.use': False,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.spines.bottom': True,
        'axes.spines.left': True,
    }
    
    # 使用上下文管理器隔离样式设置
    with plt.style.context('default'):
        with plt.rc_context(style_dict):
            
            # 创建 3×3 网格图
            fig, axes = plt.subplots(3, 3, figsize=(16, 14), dpi=600)
            axes = axes.flatten()
            
            plot_idx = 0
            
            # 遍历每个分辨率
            for res_idx, res in enumerate(resolutions):
                if res not in enhanced_results:
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
                res_data = enhanced_results[res]
                
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
                
                # 优先使用shap_values_by_feature，如果没有才使用shap_values矩阵
                if len(shap_values_by_feature) > 0:
                    print(f"\n{res} SHAP依赖图绘制准备，使用shap_values_by_feature格式")
                elif shap_values is not None:
                    print(f"\n{res} SHAP依赖图绘制准备，SHAP值形状: {shap_values.shape}")
                else:
                    print(f"\n{res} 无可用SHAP数据")
                
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
                        res_short = {'res7': 'Resolution 7 (Micro)', 'res6': 'Resolution 6 (Meso)', 'res5': 'Resolution 5 (Macro)'}
                        ax.set_title(f'({subplot_labels[plot_idx]}) {res_short[res]} - {enhance_feature_display_name(feat_name)}', 
                                   fontsize=11, fontweight='bold')
                        ax.axis('off')
                        plot_idx += 1
                        continue
                    
                    # 🎯 获取特征值和对应的SHAP值
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
                        res_short = {'res7': 'Resolution 7 (Micro)', 'res6': 'Resolution 6 (Meso)', 'res5': 'Resolution 5 (Macro)'}
                        ax.set_title(f'({subplot_labels[plot_idx]}) {res_short[res]} - {enhance_feature_display_name(feat_name)}', 
                                   fontsize=11, fontweight='bold')
                        ax.axis('off')
                        plot_idx += 1
                        continue
                    
                    print(f"    🔄 绘制{feat_name}的SHAP依赖图...")
                    
                    try:
                        # 🎨 SHAP依赖图样式：灰色置信区间 + 蓝色散点 + 红色拟合曲线
                        
                        # 1. ✨ 改进的平滑置信区间计算
                        try:
                            # 排序数据以便计算置信区间
                            sorted_indices = np.argsort(x_values)
                            x_sorted = x_values[sorted_indices]
                            y_sorted = y_values[sorted_indices]
                            
                            # 🔧 改进方法：使用移动窗口 + 平滑处理
                            n_points = len(x_sorted)
                            if n_points >= 10:
                                # 自适应窗口大小：确保每个窗口有足够样本但不过大
                                window_size = max(10, min(50, n_points // 5))
                                
                                # 生成平滑的x轴采样点
                                n_smooth_points = min(50, n_points // 2)
                                x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), n_smooth_points)
                                
                                y_lower_smooth = []
                                y_upper_smooth = []
                                
                                for x_target in x_smooth:
                                    # 计算到目标x值的距离
                                    distances = np.abs(x_sorted - x_target)
                                    
                                    # 选择最近的window_size个点
                                    closest_indices = np.argsort(distances)[:window_size]
                                    y_window = y_sorted[closest_indices]
                                    
                                    # 使用加权分位数：距离越近权重越大
                                    weights = 1.0 / (distances[closest_indices] + 1e-8)
                                    weights = weights / np.sum(weights)
                                    
                                    # 计算加权分位数
                                    sorted_window_indices = np.argsort(y_window)
                                    sorted_weights = weights[sorted_window_indices]
                                    cumsum_weights = np.cumsum(sorted_weights)
                                    
                                    # 找到25%和75%分位数对应的值
                                    q25_idx = np.searchsorted(cumsum_weights, 0.25)
                                    q75_idx = np.searchsorted(cumsum_weights, 0.75)
                                    
                                    q25_idx = min(q25_idx, len(y_window) - 1)
                                    q75_idx = min(q75_idx, len(y_window) - 1)
                                    
                                    y_lower_smooth.append(y_window[sorted_window_indices[q25_idx]])
                                    y_upper_smooth.append(y_window[sorted_window_indices[q75_idx]])
                                
                                # 🎯 进一步平滑边界以消除锯齿
                                from scipy.ndimage import gaussian_filter1d
                                # 使用高斯滤波器平滑边界
                                sigma = max(1, len(y_lower_smooth) / 20)  # 自适应平滑强度
                                y_lower_smooth = gaussian_filter1d(y_lower_smooth, sigma=sigma, mode='nearest')
                                y_upper_smooth = gaussian_filter1d(y_upper_smooth, sigma=sigma, mode='nearest')
                                
                                # 绘制平滑的灰色置信区间
                                ax.fill_between(x_smooth, y_lower_smooth, y_upper_smooth, 
                                               color='gray', alpha=0.3, 
                                               label='25%-75% Range', zorder=1)
                                print(f"      ✅ 添加了平滑的灰色置信区间背景（{len(x_smooth)}个点）")
                            
                            elif n_points >= 5:
                                # 数据点较少时，使用简化的全局分位数
                                q25 = np.percentile(y_sorted, 25)
                                q75 = np.percentile(y_sorted, 75)
                                
                                # 绘制水平置信带
                                ax.fill_between([x_sorted.min(), x_sorted.max()], [q25, q25], [q75, q75],
                                               color='gray', alpha=0.2, zorder=1)
                                print(f"      ✅ 数据点较少，使用全局分位数置信带")
                                
                        except ImportError:
                            # 如果没有scipy，使用简化的移动平均方法
                            try:
                                sorted_indices = np.argsort(x_values)
                                x_sorted = x_values[sorted_indices]
                                y_sorted = y_values[sorted_indices]
                                
                                # 简化的移动窗口方法
                                window_size = max(5, len(x_sorted) // 8)
                                n_smooth = min(30, len(x_sorted) // 2)
                                x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), n_smooth)
                                
                                y_lower_simple = []
                                y_upper_simple = []
                                
                                for x_target in x_smooth:
                                    distances = np.abs(x_sorted - x_target)
                                    closest_indices = np.argsort(distances)[:window_size]
                                    y_window = y_sorted[closest_indices]
                                    
                                    y_lower_simple.append(np.percentile(y_window, 30))  # 稍微保守
                                    y_upper_simple.append(np.percentile(y_window, 70))
                                
                                # 简单平滑：移动平均
                                smooth_window = max(1, len(y_lower_simple) // 10)
                                if smooth_window > 1:
                                    y_lower_smooth = np.convolve(y_lower_simple, np.ones(smooth_window)/smooth_window, mode='same')
                                    y_upper_smooth = np.convolve(y_upper_simple, np.ones(smooth_window)/smooth_window, mode='same')
                                else:
                                    y_lower_smooth = y_lower_simple
                                    y_upper_smooth = y_upper_simple
                                
                                ax.fill_between(x_smooth, y_lower_smooth, y_upper_smooth, 
                                               color='gray', alpha=0.3, zorder=1)
                                print(f"      ✅ 使用简化平滑方法生成置信区间")
                                
                            except Exception as e:
                                print(f"      ⚠️ 简化置信区间生成也失败: {e}")
                                
                        except Exception as e:
                            print(f"      ⚠️ 平滑置信区间生成失败: {e}")
                        
                        # 2. 绘制蓝色散点图（在置信区间之上）
                        ax.scatter(x_values, y_values, color='#1f77b4', s=15, 
                                 alpha=0.6, edgecolors='none', zorder=3)
                        
                        # 3. 添加红色拟合曲线（在最上层）- 使用改进的局部回归方法
                        try:
                            # 排序数据用于拟合
                            sorted_indices = np.argsort(x_values)
                            x_sorted = x_values[sorted_indices]
                            y_sorted = y_values[sorted_indices]
                            
                            # 🔧 改进的趋势线拟合方法
                            if len(np.unique(x_sorted)) > 5:
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
                                        print(f"      ⚠️ UnivariateSpline生成了NaN/Inf值，跳过")
                                        raise ValueError("拟合线包含NaN值")
                                    
                                    ax.plot(x_smooth, y_smooth, color='red', linewidth=4, alpha=1.0, zorder=100)
                                    print(f"      ✅ 使用UnivariateSpline生成趋势线")
                                except (ImportError, ValueError):
                                    # 方法2：增强平滑移动窗口回归
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
                                    
                                    # 🎨 绘制超平滑的红色拟合线
                                    ax.plot(x_interp, y_smooth_final, color='red', linewidth=4, alpha=1.0, zorder=100)
                                    print(f"      ✅ 使用增强平滑移动窗口生成趋势线 ({n_interp_points}个插值点)")
                                        
                            elif len(np.unique(x_sorted)) > 2:
                                # 对于点数较少的情况，使用2次多项式拟合
                                try:
                                    z = np.polyfit(x_sorted, y_sorted, deg=min(2, len(np.unique(x_sorted))-1))
                                    p = np.poly1d(z)
                                    x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 50)
                                    y_smooth = p(x_smooth)
                                    ax.plot(x_smooth, y_smooth, color='red', linewidth=4, alpha=1.0, zorder=100)
                                    print(f"      ✅ 使用多项式拟合生成趋势线")
                                except np.linalg.LinAlgError:
                                    # 简单线性拟合作为最后备选
                                    z = np.polyfit(x_sorted, y_sorted, 1)
                                    p = np.poly1d(z)
                                    ax.plot(x_sorted, p(x_sorted), color='red', linewidth=4, alpha=1.0, zorder=100)
                                    print(f"      ✅ 使用线性拟合生成趋势线")
                            else:
                                # 数据点太少，只绘制简单连线
                                ax.plot(x_sorted, y_sorted, color='red', linewidth=4, alpha=1.0, zorder=100)
                                print(f"      ✅ 数据点较少，使用直接连线")
                                
                        except Exception as e:
                            print(f"      ⚠️ 红色拟合曲线生成失败: {e}")
                            # 失败时绘制简单的线性拟合作为备选
                            try:
                                sorted_indices = np.argsort(x_values)
                                x_sorted = x_values[sorted_indices] 
                                y_sorted = y_values[sorted_indices]
                                z = np.polyfit(x_sorted, y_sorted, 1)
                                p = np.poly1d(z)
                                ax.plot(x_sorted, p(x_sorted), color='red', linewidth=4, alpha=1.0, zorder=100)
                                print(f"      🔧 使用备用线性拟合")
                            except:
                                print(f"      ❌ 所有拟合方法都失败")
                        
                        # 4. 添加零线（黑色虚线，在背景层）
                        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1, zorder=2)
                        
                        print(f"    ✅ {feat_name} SHAP依赖图绘制成功（含灰色置信区间）")
                    
                    except Exception as e:
                        # SHAP依赖图绘制出错
                        ax.text(0.5, 0.5, f"SHAP dependency error\nfor {feat_name}\n{str(e)[:30]}...", 
                               ha='center', va='center', fontsize=10, 
                               transform=ax.transAxes, color='red')
                        print(f"    ❌ {feat_name} SHAP依赖图绘制出错: {e}")
                    
                    # 设置标签和格式
                    ax.set_xlabel(enhance_feature_display_name(feat_name), fontsize=11, fontweight='bold')
                    ax.set_ylabel('GeoShapley Value', fontsize=11, fontweight='bold')
                    
                    # 设置标题 - 使用"分辨率-特征"格式
                    res_short = {'res7': 'Resolution 7 (Micro)', 'res6': 'Resolution 6 (Meso)', 'res5': 'Resolution 5 (Macro)'}
                    title = f'({subplot_labels[plot_idx]}) {res_short[res]} - {enhance_feature_display_name(feat_name)}'
                    ax.set_title(title, fontsize=11, fontweight='bold')
                    
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
                
                # 填充剩余的子图（如果特征少于top_n）
                while plot_idx < (res_idx + 1) * top_n and plot_idx < 9:
                    ax = axes[plot_idx]
                    ax.axis('off')
                    plot_idx += 1
            
            # 添加总标题
            fig.suptitle('SHAP Dependency Plots for Top Primary Effects Across Resolutions', 
                        fontsize=18, fontweight='bold')
            
            # 调整布局
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # 保存图表
            if output_dir:
                output_path = os.path.join(output_dir, 'all_resolutions_pdp_grid.png')
                plt.savefig(output_path, dpi=600, bbox_inches='tight', 
                           transparent=False, facecolor='white', edgecolor='none')
                plt.close()
                
                # 输出详细的保存信息
                print(f"\n  ✅ SHAP依赖图网格已保存到: {output_path}")
                print(f"    📊 包含蓝色散点、红色趋势线和零线的标准SHAP依赖图")
                
                return output_path
            else:
                plt.close()  # 确保图形被关闭
                return None
    
    # 恢复原始rcParams设置
    plt.rcParams.update(original_rcParams) 
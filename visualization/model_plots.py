#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model performance and prediction visualization

Contains functions for creating model performance visualizations and comparisons

支持的模型类型：
- ST-GPR (Spatiotemporal Gaussian Process Regression) 模型
- 其他通用机器学习模型

该模块提供了模型性能评估和预测结果的可视化功能，
完全兼容ST-GPR模型的输出格式。
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os

from .base import enhance_plot_style, save_plot_for_publication, ensure_dir_exists, color_map
from .utils import categorize_feature


def plot_model_performance(results, output_dir=None, resolution=None, suffix=None, save_plot=True):
    """
    绘制模型性能指标图表
    
    参数:
    results: 模型结果字典
    output_dir: 输出目录
    resolution: 分辨率标识（如果只绘制一个分辨率的结果）
    suffix: 文件名后缀
    save_plot: 是否保存图表，默认为True
    
    返回:
    matplotlib.figure.Figure: 图表对象
    """
    # 获取性能指标
    metrics = {}
    res_labels = []
    
    # 单个分辨率模式
    if resolution:
        if resolution in results:
            metrics[resolution] = results[resolution].get('metrics', {})
            res_labels = [resolution.upper()]
        else:
            print(f"警告: 找不到分辨率 {resolution} 的结果")
            return None
    else:
        # 多分辨率比较模式
        res_count = 0
        for res in ['res5', 'res6', 'res7']:
            if res in results and 'metrics' in results[res]:
                metrics[res] = results[res]['metrics']
                res_labels.append(res.upper())
                res_count += 1
        
        if res_count == 0:
            print("警告: 没有找到任何有效的分辨率结果")
            return None
    
    # 创建图表 - 使用子图布局以统一样式
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 提取指标
    train_r2 = []
    test_r2 = []
    train_rmse = []
    test_rmse = []
    
    # 对所有分辨率结果进行排序处理
    sorted_metrics = sorted(metrics.items())
    
    for _, metric in sorted_metrics:
        train_r2.append(metric.get('train_r2', 0))
        test_r2.append(metric.get('test_r2', 0))
        train_rmse.append(metric.get('train_rmse', 0))
        test_rmse.append(metric.get('test_rmse', 0))
    
    # 设置x位置
    x = np.arange(len(res_labels))
    width = 0.35
    
    # 绘制R²条形图
    bars1 = ax1.bar(x - width/2, train_r2, width, label='训练集', color='#3498db', edgecolor='black', alpha=0.7)
    bars2 = ax1.bar(x + width/2, test_r2, width, label='测试集', color='#e74c3c', edgecolor='black', alpha=0.7)
    
    # 添加值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 设置R²图表属性
    ax1.set_ylabel('决定系数 (R²)', fontsize=12, fontweight='bold')
    ax1.set_title('模型拟合度 (R²)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(res_labels)
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 绘制RMSE条形图
    bars3 = ax2.bar(x - width/2, train_rmse, width, label='训练集', color='#3498db', edgecolor='black', alpha=0.7)
    bars4 = ax2.bar(x + width/2, test_rmse, width, label='测试集', color='#e74c3c', edgecolor='black', alpha=0.7)
    
    # 添加值标签
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 设置RMSE图表属性
    ax2.set_ylabel('均方根误差 (RMSE)', fontsize=12, fontweight='bold')
    ax2.set_title('预测误差 (RMSE)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(res_labels)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 设置整体标题
    if resolution:
        plt.suptitle(f'模型性能指标 - {resolution.upper()}', fontsize=16, fontweight='bold', y=0.98)
    else:
        plt.suptitle('模型性能指标比较', fontsize=16, fontweight='bold', y=0.98)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 保存图表
    if output_dir:
        ensure_dir_exists(output_dir)
        # 使用suffix优先，其次是resolution
        if suffix:
            fig_path = os.path.join(output_dir, f"model_performance{suffix}.png")
        elif resolution:
            fig_path = os.path.join(output_dir, f"{resolution}_Fig4-6_model_performance.png")
        else:
            fig_path = os.path.join(output_dir, "model_performance.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"已保存模型性能图: {fig_path}")
    
    # 关闭图表，避免在Jupyter中显示
    plt.close(fig)
    
    return fig


def plot_prediction_scatter(results, output_dir=None, resolution=None, suffix=None, save_plot=True):
    """
    绘制预测散点图，显示实际值与预测值的对比
    
    参数:
    results: 模型结果字典
    output_dir: 输出目录
    resolution: 分辨率标识
    suffix: 文件名后缀
    save_plot: 是否保存图表，默认为True
    
    返回:
    matplotlib.figure.Figure: 图表对象
    """
    # 对单个分辨率进行绘图
    if resolution:
        if resolution in results and 'predictions' in results[resolution]:
            predictions = results[resolution]['predictions']
            targets = predictions.get('targets')
            pred_mean = predictions.get('mean')
            
            if targets is None or pred_mean is None:
                print(f"警告: {resolution} 的预测结果缺少必要数据")
                return None
            
            # 计算性能指标
            r2 = r2_score(targets, pred_mean)
            rmse = np.sqrt(mean_squared_error(targets, pred_mean))
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 绘制预测散点图
            scatter = ax.scatter(targets, pred_mean, alpha=0.6, edgecolors='w', 
                              c=get_scatter_color(), cmap='viridis')
            
            # 添加对角线（完美预测线）
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])
            ]
            ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
            
            # 设置轴限制
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            
            # 添加网格线
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 设置标题和轴标签
            title = f'预测值 vs 实际值 - {resolution.upper()}\nR² = {r2:.4f}, RMSE = {rmse:.4f}'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('实际值', fontsize=12, fontweight='bold')
            ax.set_ylabel('预测值', fontsize=12, fontweight='bold')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            if output_dir and save_plot:
                ensure_dir_exists(output_dir)
                # 使用suffix优先，其次是resolution
                if suffix:
                    fig_path = os.path.join(output_dir, f"prediction_scatter{suffix}.png")
                elif resolution:
                    fig_path = os.path.join(output_dir, f"{resolution}_Fig4-7_prediction_scatter.png")
                else:
                    fig_path = os.path.join(output_dir, "prediction_scatter.png")
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"已保存预测散点图: {fig_path}")
            
            # 关闭图表，避免在Jupyter中显示
            plt.close(fig)
            
            return fig
        else:
            print(f"警告: 找不到分辨率 {resolution} 的预测结果")
            return None
    else:
        # 多分辨率合并图表
        print("绘制多分辨率预测散点图...")
        
        # 检查有多少个有效的分辨率
        valid_resolutions = []
        for res in ['res5', 'res6', 'res7']:
            if res in results and 'predictions' in results[res]:
                valid_resolutions.append(res)
        
        if not valid_resolutions:
            print("警告: 没有找到任何有效的分辨率预测结果")
            return None
        
        # 创建子图网格 - 最多3列
        n_cols = min(3, len(valid_resolutions))
        n_rows = (len(valid_resolutions) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5), squeeze=False)
        
        # 确保性能最好的模型（R2最高）显示在第一个位置
        resolution_metrics = {}
        for res in valid_resolutions:
            if 'metrics' in results[res]:
                test_r2 = results[res]['metrics'].get('test_r2', 0)
                resolution_metrics[res] = test_r2
        
        # 按R2降序排序
        sorted_resolutions = sorted(resolution_metrics.items(), key=lambda x: x[1], reverse=True)
        ordered_resolutions = [item[0] for item in sorted_resolutions] + \
                            [res for res in valid_resolutions if res not in resolution_metrics]
        
        # 依次绘制每个分辨率的散点图
        for i, res in enumerate(ordered_resolutions):
            # 计算行和列索引
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            predictions = results[res]['predictions']
            targets = predictions.get('targets')
            pred_mean = predictions.get('mean')
            
            # 计算性能指标
            r2 = r2_score(targets, pred_mean)
            rmse = np.sqrt(mean_squared_error(targets, pred_mean))
            
            # 绘制散点图
            scatter = ax.scatter(targets, pred_mean, alpha=0.6, edgecolors='w',
                             c=get_scatter_color(), cmap='viridis')
            
            # 添加对角线
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])
            ]
            ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
            
            # 设置轴限制
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            
            # 添加网格线
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 设置标题和轴标签
            title = f'{res.upper()}\nR² = {r2:.4f}, RMSE = {rmse:.4f}'
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('实际值', fontsize=10, fontweight='bold')
            ax.set_ylabel('预测值', fontsize=10, fontweight='bold')
        
        # 隐藏空子图
        for i in range(len(ordered_resolutions), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        # 设置整体标题
        plt.suptitle('预测值 vs 实际值 (按分辨率)', fontsize=16, fontweight='bold', y=0.98)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存图表
        if output_dir:
            ensure_dir_exists(output_dir)
            if suffix:
                fig_path = os.path.join(output_dir, f"all_resolution_predictions{suffix}.png")
            else:
                fig_path = os.path.join(output_dir, "all_resolution_predictions.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"已保存多分辨率预测散点图: {fig_path}")
        
        # 关闭图表，避免在Jupyter中显示
        plt.close(fig)
        
        return fig


def plot_residual_histogram(residuals, output_dir=None, resolution=None):
    """
    绘制残差直方图
    
    参数:
    residuals (array): 模型残差
    output_dir (str): 输出目录
    resolution (str): 分辨率标识
    """
    if residuals is None or len(residuals) == 0:
        print("警告: 缺少残差数据")
        return
    
    # 创建直方图
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    n, bins, patches = plt.hist(residuals, bins=30, color='skyblue', 
                              edgecolor='black', alpha=0.7)
    
    # 添加正态分布曲线
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
         np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    y = y * len(residuals) * (bins[1] - bins[0])  # 缩放以匹配直方图
    
    plt.plot(bins, y, 'r--', linewidth=2)
    
    # 设置标题和轴标签
    plt.title('Histogram of Residuals', fontsize=14, fontweight='bold')
    plt.xlabel('Residuals', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    
    # 添加文本框显示均值和标准差
    mu_symbol = r"\mu"
    sigma_symbol = r"\sigma"
    textstr = f'${mu_symbol}={mu:.2f}$\n${sigma_symbol}={sigma:.2f}$'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=props, fontsize=10)
    
    # 添加网格
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if output_dir:
        ensure_dir_exists(output_dir)
        if resolution:
            fig_path = os.path.join(output_dir, f"{resolution}_Fig4-11_residual_histogram.png")
        else:
            fig_path = os.path.join(output_dir, "residual_histogram.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"已保存残差直方图: {fig_path}")
    
    plt.close()


def plot_combined_predictions(results, output_dir=None):
    """
    绘制多个分辨率模型的预测结果对比图
    
    参数:
    results (dict): 包含各分辨率模型结果的字典
    output_dir (str): 输出目录
    """
    if not results:
        print("警告: 没有可用的模型结果")
        return
    
    # 提取各分辨率的测试性能
    resolutions = list(results.keys())
    r2_values = []
    rmse_values = []
    
    for res in resolutions:
        if 'test_metrics' in results[res]:
            r2_values.append(results[res]['test_metrics']['r2'])
            rmse_values.append(results[res]['test_metrics']['rmse'])
        else:
            print(f"警告: 缺少{res}的测试指标")
            r2_values.append(0)
            rmse_values.append(0)
    
    # 创建比较图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 绘制R²比较
    axes[0].bar(resolutions, r2_values, color='#3498db', width=0.6, edgecolor='black', linewidth=1)
    axes[0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Resolution', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('R² Score', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', linestyle='--', alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(r2_values):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10, fontweight='bold')
    
    # 绘制RMSE比较
    axes[1].bar(resolutions, rmse_values, color='#e74c3c', width=0.6, edgecolor='black', linewidth=1)
    axes[1].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Resolution', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('RMSE', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', linestyle='--', alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(rmse_values):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 增强所有子图的样式
    for ax in axes:
        enhance_plot_style(ax)
    
    # 保存图表
    if output_dir:
        ensure_dir_exists(output_dir)
        fig_path = os.path.join(output_dir, "Fig4-13_resolution_comparison.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"已保存分辨率比较图: {fig_path}")
    
    plt.close()


def plot_combined_model_performance_prediction(results, output_dir=None):
    """
    创建组合模型性能与预测图表，比较不同空间分辨率模型的预测性能
    保持标准的学术可视化风格
    
    参数:
    results (dict): 包含各分辨率模型结果的字典
    output_dir (str): 输出目录
    """
    from matplotlib.gridspec import GridSpec
    from sklearn.metrics import r2_score, mean_squared_error
    
    # 不输出过程消息到控制台
    # print("创建组合模型性能与预测图...")
    
    # 检查是否有足够的结果数据
    if not results or len(results) < 1:
        print("警告: 没有足够的结果数据进行模型性能和预测比较")
        return
    
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
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    }
    
    # 使用上下文管理器隔离样式设置
    with plt.style.context('default'):
        with plt.rc_context(style_dict):
            
            # 创建图表
            fig = plt.figure(figsize=(16, 14), dpi=600)
            
            # 设置网格规格：2行、3列
            gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
            
            # 创建子图
            ax_perf = fig.add_subplot(gs[0, :])  # 第一行全部 - 作为a图
            ax_fit1 = fig.add_subplot(gs[1, 0])  # 第二行第一列 - 作为b图
            ax_fit2 = fig.add_subplot(gs[1, 1])  # 第二行第二列 - 作为c图
            ax_fit3 = fig.add_subplot(gs[1, 2])  # 第二行第三列 - 作为d图
            
            # 子图标题字母
            subplot_labels = 'abcd'
            
            # 设置X轴刻度和标签
            x = np.array([0, 1, 2])
            x_labels = ['H3 res7\n(Micro)', 'H3 res6\n(Meso)', 'H3 res5\n(Macro)']
            width = 0.35  # 柱状图宽度
            
            # --- 绘制a图：模型性能对比（上面一行）---
            # R²对比（左Y轴）
            r2_values = []
            rmse_values = []
            
            for res in ['res7', 'res6', 'res5']:
                if res in results and 'test_metrics' in results[res]:
                    metrics = results[res]['test_metrics']
                    r2 = metrics.get('R2', metrics.get('r2', 0))
                    rmse = metrics.get('RMSE', metrics.get('rmse', 0))
                    r2_values.append(r2)
                    rmse_values.append(rmse)
                else:
                    r2_values.append(0)
                    rmse_values.append(0)
            
            # 设置轴线宽度
            for spine in ax_perf.spines.values():
                spine.set_linewidth(1.5)
            
            bars1 = ax_perf.bar(x - width/2, r2_values, width, color='#3498db', label='R²', 
                              edgecolor='black', linewidth=1.5)
            
            # 设置x轴标签
            ax_perf.set_xlabel('Spatial Resolution', fontsize=12, fontweight='bold')
            ax_perf.set_ylabel('Test R²', color='#3498db', fontsize=12, fontweight='bold')
            ax_perf.set_ylim([0, 1])  # R²的范围为0-1
            ax_perf.tick_params(axis='y', labelcolor='#3498db', labelsize=10, width=1.5, length=4, direction='in')
            ax_perf.tick_params(axis='x', labelsize=10, width=1.5, length=4, direction='in')
            ax_perf.set_xticks(x)
            ax_perf.set_xticklabels(x_labels, fontweight='bold')
            ax_perf.grid(axis='y', alpha=0.3, color='#3498db', linestyle='--')
            
            # 添加R²数值标签
            for bar in bars1:
                height = bar.get_height()
                ax_perf.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom', color='#3498db', fontweight='bold')
            
            # RMSE对比（右Y轴）
            ax_perf2 = ax_perf.twinx()
            # 设置右Y轴的轴线宽度
            for spine in ax_perf2.spines.values():
                spine.set_linewidth(1.5)
            
            bars2 = ax_perf2.bar(x + width/2, rmse_values, width, color='#e74c3c', label='RMSE',
                               edgecolor='black', linewidth=1.5)
            
            ax_perf2.set_ylabel('Test RMSE', color='#e74c3c', fontsize=12, fontweight='bold')
            max_rmse = max(rmse_values) * 1.2  # 留出数值标签的空间
            ax_perf2.set_ylim([0, max_rmse])
            ax_perf2.tick_params(axis='y', labelcolor='#e74c3c', labelsize=10, width=1.5, length=4, direction='in')
            ax_perf2.grid(axis='y', alpha=0.3, color='#e74c3c', linestyle=':')
            
            # 添加RMSE数值标签
            for bar in bars2:
                height = bar.get_height()
                ax_perf2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                            f'{height:.3f}', ha='center', va='bottom', color='#e74c3c', fontweight='bold')
            
            # 添加图例
            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], color='#3498db', lw=4),
                Line2D([0], [0], color='#e74c3c', lw=4)
            ]
            legend = ax_perf.legend(custom_lines, ['R² (higher is better)', 'RMSE (lower is better)'], 
                      loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=10)

            # 为图例文本设置粗体
            for text in legend.get_texts():
                text.set_fontweight('bold')
            
            # 添加a子图标题
            ax_perf.set_title(f'(a) Model Performance Comparison Across Resolutions', fontsize=14, fontweight='bold')
            
            # --- 绘制b、c、d图：预测拟合对比（下面一行）---
            axes_fit = [ax_fit1, ax_fit2, ax_fit3]
            resolution_labels = {'res7': 'H3 Resolution 7 (Micro)', 'res6': 'H3 Resolution 6 (Meso)', 'res5': 'H3 Resolution 5 (Macro)'}
            
            # 设置所有子图使用相同的高程颜色映射范围
            elevation_min = float('inf')
            elevation_max = float('-inf')
            
            # 首先获取所有分辨率下的高程值范围
            for res in ['res7', 'res6', 'res5']:
                if res in results and 'X_test' in results[res] and 'elevation' in results[res]['X_test'].columns:
                    elevation_values = results[res]['X_test']['elevation']
                    elevation_min = min(elevation_min, elevation_values.min())
                    elevation_max = max(elevation_max, elevation_values.max())
            
            # 避免无效的min和max值
            if elevation_min == float('inf') or elevation_max == float('-inf'):
                elevation_min, elevation_max = 0, 2000  # 默认值
            
            # 使用统一的颜色标准化器
            norm = plt.Normalize(elevation_min, elevation_max)
            
            for i, (res, ax) in enumerate(zip(['res7', 'res6', 'res5'], axes_fit)):
                # 设置轴线宽度
                for spine in ax.spines.values():
                    spine.set_linewidth(1.5)
                
                if res in results and 'y_test' in results[res] and 'y_pred' in results[res]:
                    y_test = results[res]['y_test']
                    y_pred = results[res]['y_pred']
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    
                    # 获取该分辨率下的高程值
                    if 'X_test' in results[res] and 'elevation' in results[res]['X_test'].columns:
                        elevation = results[res]['X_test']['elevation']
                        
                        # 创建散点图，按高程值着色
                        sc = ax.scatter(y_test, y_pred, alpha=0.7, c=elevation, cmap='terrain', 
                                      norm=norm, edgecolor='black', linewidth=0.3, s=30)
                        
                        # 添加每个子图的颜色条
                        cbar = fig.colorbar(sc, ax=ax)
                        cbar.set_label('Elevation (m)', fontsize=10, fontweight='bold')
                        cbar.ax.tick_params(labelsize=8, width=1.5, length=4)
                        # 设置colorbar标签为粗体
                        for t in cbar.ax.get_yticklabels():
                            t.set_fontweight('bold')
                    else:
                        # 如果没有高程数据，使用默认颜色
                        print(f"  Warning: 'elevation' feature not found in {res} data, using default color.")
                        ax.scatter(y_test, y_pred, alpha=0.5, color='#3498db', edgecolor='black', linewidth=0.3)
                    
                    # 添加对角线
                    ax.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1.5)
                    
                    # 添加标题 - 去掉重复的分辨率数字
                    res_desc = {'res7': 'Micro', 'res6': 'Meso', 'res5': 'Macro'}
                    title = f'({subplot_labels[i+1]}) H3 Resolution {res[-1]} ({res_desc.get(res, "")}) R²={r2:.4f}, RMSE={rmse:.4f}'
                    
                    # 调整标题位置，确保其在边框内
                    ax.set_title(title, fontsize=12, fontweight='bold', pad=10, loc='center')
                    
                    ax.set_xlabel('Actual VHI', fontsize=10, fontweight='bold')
                    ax.set_ylabel('Predicted VHI', fontsize=10, fontweight='bold')
                    
                    # 调整轴刻度标签
                    ax.tick_params(axis='both', which='major', labelsize=10, width=1.5, length=4, direction='in')
                    # 设置刻度标签为粗体
                    for tick in ax.get_xticklabels():
                        tick.set_fontweight('bold')
                    for tick in ax.get_yticklabels():
                        tick.set_fontweight('bold')
                    
                    # 统一设置所有子图的轴范围为0.0-1.0
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.grid(alpha=0.3, linestyle='--')
                else:
                    # 如果没有数据，显示提示文本
                    ax.text(0.5, 0.5, "No prediction data available", 
                          ha='center', va='center', fontsize=10, fontweight='bold', transform=ax.transAxes)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    res_desc = {'res7': 'Micro', 'res6': 'Meso', 'res5': 'Macro'}
                    ax.set_title(f'({subplot_labels[i+1]}) H3 Resolution {res[-1]} ({res_desc.get(res, "")})', 
                                fontsize=12, fontweight='bold', loc='left')
                    ax.tick_params(axis='both', which='major', labelsize=10, width=1.5, length=4, direction='in')
            
            # 确保布局紧凑，但给子图标题和轴标签留出足够空间
            plt.tight_layout(rect=[0, 0, 1, 0.96], pad=2.0, h_pad=2.0, w_pad=1.0)
            
            # 保存组合图
            if output_dir:
                ensure_dir_exists(output_dir)
                combined_path = os.path.join(output_dir, 'model_performance_comparison.png')
                plt.savefig(combined_path, dpi=600, bbox_inches='tight', 
                          transparent=False, facecolor='white', edgecolor='none')
                print(f"✓ 成功保存模型性能跨分辨率比较图: {combined_path}")
            
            plt.close()
    
    # 恢复原始rcParams设置
    plt.rcParams.update(original_rcParams)
    return fig 
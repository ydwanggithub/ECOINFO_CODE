#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDP交互效应绘制模块 - 特征交互可视化功能

从pdp_plots.py重构而来，专注于：
- 识别顶级交互特征对
- 交互效应PDP网格绘制
- 单个交互效应PDP绘制

适配ST-GPR模型的特殊需求和GeoShapley分析
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
from matplotlib.gridspec import GridSpec
from sklearn.inspection import partial_dependence
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches

# 导入通用的绘图函数和工具
try:
    from .base import enhance_plot_style, ensure_dir_exists, save_plot_for_publication, color_map
    from .utils import clean_feature_name_for_plot, categorize_feature, simplify_feature_name_for_plot, enhance_feature_display_name, clean_feature_name, format_pdp_feature_name
    from model_analysis.core import standardize_feature_name
    from visualization.utils import ensure_spatiotemporal_features
except ImportError:
    # 相对导入失败时尝试绝对导入
    from visualization.base import enhance_plot_style, ensure_dir_exists, save_plot_for_publication, color_map
    from visualization.utils import clean_feature_name_for_plot, categorize_feature, simplify_feature_name_for_plot, enhance_feature_display_name, clean_feature_name, format_pdp_feature_name
    from model_analysis.core import standardize_feature_name
    from visualization.utils import ensure_spatiotemporal_features

# 设置matplotlib默认字体以支持中文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def identify_top_interactions(results, top_n=3):
    """
    基于SHAP交互值识别每个分辨率的顶级交互特征对，适用于STGPR模型分析
    
    使用全部18个特征（GeoShapley输出）：
    在GeoShapley分析输出的18个特征中计算交互
    
    参数:
    results (dict): 包含各分辨率模型结果的字典
    top_n (int): 每个分辨率返回的顶级交互特征对数量
    
    返回:
    dict: 每个分辨率的顶级交互特征对字典
    """
    print("识别顶级特征交互对...")
    
    # 存储每个分辨率的顶级交互特征对
    top_interactions = {}
    
    for res, res_data in results.items():
        if 'shap_interaction_values' not in res_data or res_data['shap_interaction_values'] is None:
            print(f"  ❌ {res}: 未找到SHAP交互值")
            print(f"     原因：可能未计算SHAP交互值或计算失败")
            print(f"     建议：检查GeoShapley计算是否成功完成交互值计算")
            print(f"     说明：PDP交互图需要SHAP交互值来识别重要的特征交互对")
            top_interactions[res] = []
            continue
        
        # 获取原始特征名称和交互值
        all_feature_names = list(res_data['X_sample'].columns)
        interaction_values = res_data['shap_interaction_values']
        
        print(f"  {res}: 原始特征数量 = {len(all_feature_names)}")
        
        # 使用全部特征（GeoShapley输出的18个特征）
        # 1. 获取特征重要性（基于SHAP值）
        if 'feature_importance' in res_data:
            # 从计算好的特征重要性中筛选时空特征
            feature_importance_list = res_data['feature_importance']
            
            # 筛选排除'GEO'和'year'的时空特征
            environmental_features = []
            for feat in feature_importance_list:
                if isinstance(feat, tuple):
                    feat_name = feat[0]
                elif isinstance(feat, dict):
                    feat_name = feat['feature']
                else:
                    feat_name = str(feat)
                
                # 排除GEO、year和交互效应，只保留主效应环境特征
                if (feat_name != 'GEO' and 
                    str(feat_name).lower() != 'year' and
                    '×' not in str(feat_name) and 
                    ' x ' not in str(feat_name)):
                    environmental_features.append(feat_name)
            
            print(f"     符合条件的环境特征数量: {len(environmental_features)}")
            
            # 使用前12个环境特征
            selected_features = environmental_features[:12]
            print(f"     选择交互分析的特征数量: {len(selected_features)}")
        else:
            # 使用所有可用特征
            selected_features = all_feature_names
            print(f"     使用所有特征进行交互分析: {len(selected_features)}")
        
        if len(selected_features) < 2:
            print(f"  ❌ {res}: 可用特征数量 ({len(selected_features)}) 不足以进行交互分析")
            top_interactions[res] = []
            continue
        
        # 计算交互值矩阵 - 改进的交互强度计算
        print(f"     计算交互值矩阵，特征数量: {len(selected_features)}")
        
        try:
            # 只计算三角矩阵，避免重复
            interaction_scores = []
            valid_pairs = []
            
            for i in range(len(selected_features)):
                for j in range(i+1, len(selected_features)):
                    feat1 = selected_features[i]
                    feat2 = selected_features[j]
                    
                    # 获取这两个特征在原始特征列表中的索引
                    try:
                        idx1 = all_feature_names.index(feat1)
                        idx2 = all_feature_names.index(feat2)
                        
                        # 计算交互强度：使用Shapley交互值的绝对值平均
                        # SHAP交互值 [i,j] 表示特征i和特征j的交互影响
                        interaction_score = np.abs(interaction_values[:, idx1, idx2]).mean()
                        
                        interaction_scores.append(interaction_score)
                        valid_pairs.append((feat1, feat2, interaction_score))
                        
                    except ValueError as e:
                        print(f"       特征{feat1}或{feat2}在特征列表中未找到，跳过")
                        continue
            
            if not valid_pairs:
                print(f"  ❌ {res}: 没有有效的交互特征对")
                top_interactions[res] = []
                continue
            
            # 按交互强度排序
            valid_pairs.sort(key=lambda x: x[2], reverse=True)
            
            # 选择前top_n个交互对
            top_pairs = valid_pairs[:top_n]
            
            # 存储结果
            top_interactions[res] = [(pair[0], pair[1]) for pair in top_pairs]
            
            print(f"  ✅ {res}: 成功识别出 {len(top_pairs)} 个顶级交互特征对")
            for i, (feat1, feat2, score) in enumerate(top_pairs):
                print(f"     {i+1}. {feat1} × {feat2}: 交互强度 = {score:.4f}")
        
        except Exception as e:
            print(f"  ❌ {res}: 交互值计算出错: {e}")
            top_interactions[res] = []
    
    return top_interactions


def plot_pdp_interaction_grid(results, output_dir=None, top_n=3):
    """
    绘制PDP交互效应网格图，展示不同分辨率下的顶级特征交互对
    
    为每个分辨率的前top_n个交互特征对创建2D PDP热力图：
    - 自动识别基于SHAP交互值的顶级交互对
    - 使用热力图显示特征交互效应
    - 添加等高线和颜色条
    
    参数:
    results (dict): 包含各分辨率模型结果的字典
    output_dir (str): 输出目录
    top_n (int): 每个分辨率显示的顶级交互特征对数量
    
    返回:
    str: 生成的图表路径
    """
    print(f"\n🎨 创建PDP交互效应网格图...")
    print(f"    📊 每个分辨率展示前{top_n}个最重要的特征交互对")
    
    # 确保输出目录存在
    if output_dir:
        ensure_dir_exists(output_dir)
    
    # 1. 识别顶级交互特征对
    top_interactions = identify_top_interactions(results, top_n)
    
    # 检查是否有任何有效的交互对
    has_any_interactions = any(len(interactions) > 0 for interactions in top_interactions.values())
    if not has_any_interactions:
        print("  ❌ 所有分辨率都没有有效的交互特征对，无法生成交互图")
        return None
    
    # 创建子图网格
    resolutions = ['res7', 'res6', 'res5']
    valid_resolutions = [res for res in resolutions if res in results and len(top_interactions.get(res, [])) > 0]
    
    if not valid_resolutions:
        print("  ❌ 没有分辨率包含有效的交互特征对")
        return None
    
    # 计算网格大小
    n_resolutions = len(valid_resolutions)
    cols = top_n
    rows = n_resolutions
    
    # 创建图表
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    plot_count = 0
    
    for res_idx, res in enumerate(valid_resolutions):
        print(f"\n  🔄 处理{res}的交互图...")
        
        res_data = results[res]
        res_interactions = top_interactions[res]
        
        # 获取模型和数据
        model = res_data.get('model')
        likelihood = res_data.get('likelihood')
        X_sample = res_data.get('X_sample')
        
        if not model or X_sample is None:
            print(f"    ❌ {res}: 缺少模型或数据")
            continue
        
        # 为每个交互特征对创建PDP图
        for pair_idx, (feat1, feat2) in enumerate(res_interactions[:top_n]):
            if pair_idx >= cols:  # 防止超出列数
                break
            
            ax = axes[res_idx, pair_idx] if rows > 1 else axes[pair_idx]
            
            print(f"    🎯 绘制 {feat1} × {feat2} 交互图...")
            
            try:
                # 创建PDP预测函数
                if likelihood:
                    # GPyTorch模型
                    def make_gpytorch_predict_fn(model_obj, likelihood_obj):
                        def predict_fn(X):
                            model_obj.eval()
                            likelihood_obj.eval()
                            
                            with torch.no_grad():
                                try:
                                    import torch
                                    if not isinstance(X, torch.Tensor):
                                        X_tensor = torch.tensor(X, dtype=torch.float32)
                                    else:
                                        X_tensor = X
                                    
                                    # 移动到模型设备
                                    device = next(model_obj.parameters()).device
                                    X_tensor = X_tensor.to(device)
                                    
                                    output = model_obj(X_tensor)
                                    pred_dist = likelihood_obj(output)
                                    predictions = pred_dist.mean.cpu().numpy()
                                    
                                    return predictions
                                except Exception as e:
                                    print(f"      预测函数错误: {e}")
                                    return np.zeros(len(X))
                        return predict_fn
                    
                    predict_fn = make_gpytorch_predict_fn(model, likelihood)
                else:
                    # 其他模型类型的安全预测函数
                    def make_safe_predict(model_obj):
                        def safe_predict(X):
                            try:
                                if hasattr(model_obj, 'predict'):
                                    return model_obj.predict(X)
                                elif hasattr(model_obj, '__call__'):
                                    return model_obj(X)
                                else:
                                    print(f"      模型没有predict方法")
                                    return np.zeros(len(X))
                            except Exception as e:
                                print(f"      预测出错: {e}")
                                return np.zeros(len(X))
                        return safe_predict
                    
                    predict_fn = make_safe_predict(model)
                
                # 使用scikit-learn的partial_dependence计算2D PDP
                try:
                    # 获取特征索引
                    feature_names = list(X_sample.columns)
                    feat1_idx = feature_names.index(feat1)
                    feat2_idx = feature_names.index(feat2)
                    
                    # 计算2D PDP - 使用较小的网格以加快计算
                    pdp_result = partial_dependence(
                        predict_fn, X_sample.values, 
                        features=[feat1_idx, feat2_idx], 
                        grid_resolution=15, 
                        kind='average'
                    )
                    
                    # 提取PDP数据
                    pdp_values = pdp_result['average'][0]
                    feat1_values = pdp_result['grid_values'][0]
                    feat2_values = pdp_result['grid_values'][1]
                    
                    # 创建网格
                    f1_mesh, f2_mesh = np.meshgrid(feat1_values, feat2_values)
                    
                    # 绘制热力图
                    contour = ax.contourf(f1_mesh, f2_mesh, pdp_values.T, 
                                        cmap='viridis', alpha=0.9, levels=15)
                    
                    # 添加等高线
                    contour_lines = ax.contour(f1_mesh, f2_mesh, pdp_values.T, 
                                             colors='white', alpha=0.6, levels=8)
                    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.3f')
                    
                    # 设置标签和标题
                    ax.set_xlabel(simplify_feature_name_for_plot(feat1, max_length=4), 
                                fontsize=11, fontweight='bold')
                    ax.set_ylabel(simplify_feature_name_for_plot(feat2, max_length=4), 
                                fontsize=11, fontweight='bold')
                    
                    # 设置标题
                    res_short = {'res7': 'Res7', 'res6': 'Res6', 'res5': 'Res5'}
                    ax.set_title(f'{res_short[res]}: {simplify_feature_name_for_plot(feat1, max_length=4)} × {simplify_feature_name_for_plot(feat2, max_length=4)}', 
                               fontsize=12, fontweight='bold')
                    
                    # 添加颜色条（只为第一列添加）
                    if pair_idx == 0:
                        cbar = plt.colorbar(contour, ax=ax)
                        cbar.set_label('Predicted VHI', fontsize=10, fontweight='bold')
                    
                    print(f"    ✅ {feat1} × {feat2} 交互图绘制成功")
                    plot_count += 1
                
                except Exception as pdp_error:
                    print(f"    ❌ {feat1} × {feat2} PDP计算失败: {pdp_error}")
                    ax.text(0.5, 0.5, f"PDP Error\n{feat1} × {feat2}", 
                           ha='center', va='center', fontsize=10, 
                           transform=ax.transAxes, color='red')
                    ax.set_title(f'{res} - Error', fontsize=12)
            
            except Exception as e:
                print(f"    ❌ {feat1} × {feat2} 整体绘制失败: {e}")
                ax.text(0.5, 0.5, f"Error\n{feat1} × {feat2}", 
                       ha='center', va='center', fontsize=10, 
                       transform=ax.transAxes, color='red')
                ax.set_title(f'{res} - Error', fontsize=12)
    
    # 隐藏空的子图
    for res_idx in range(len(valid_resolutions)):
        for pair_idx in range(len(top_interactions.get(valid_resolutions[res_idx], [])), cols):
            if res_idx < rows and pair_idx < cols:
                ax = axes[res_idx, pair_idx] if rows > 1 else axes[pair_idx]
                ax.axis('off')
    
    # 设置总标题
    plt.suptitle('PDP Interaction Effects Grid Across Resolutions', 
               fontsize=16, fontweight='bold')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图表
    if output_dir and plot_count > 0:
        output_path = os.path.join(output_dir, 'pdp_interaction_grid.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n  ✅ PDP交互效应网格图已保存到: {output_path}")
        print(f"    📊 成功绘制了 {plot_count} 个交互效应图")
        
        return output_path
    else:
        plt.close()
        print(f"\n  ❌ 未生成任何有效的交互图")
        return None


def plot_pdp_single_interaction(feat1, feat2, model, X_sample, output_dir=None, resolution=None):
    """
    绘制单个特征对的2D PDP交互图
    
    参数:
    feat1 (str): 第一个特征名称
    feat2 (str): 第二个特征名称
    model: 训练好的模型
    X_sample (DataFrame): 样本数据
    output_dir (str): 输出目录
    resolution (str): 分辨率标识
    
    返回:
    str或matplotlib.figure.Figure: 图表路径或图表对象
    """
    try:
        print(f"绘制 {feat1} × {feat2} 的PDP交互图...")
        
        # 检查特征是否存在
        feature_names = list(X_sample.columns)
        if feat1 not in feature_names or feat2 not in feature_names:
            print(f"特征 {feat1} 或 {feat2} 不在数据中")
            return None
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 创建预测函数
        try:
            import torch
            import gpytorch
            
            if hasattr(model, 'eval'):  # GPyTorch模型
                def predict_fn(X):
                    model.eval()
                    with torch.no_grad():
                        if not isinstance(X, torch.Tensor):
                            X_tensor = torch.tensor(X, dtype=torch.float32)
                        else:
                            X_tensor = X
                        
                        # 移动到模型设备
                        device = next(model.parameters()).device
                        X_tensor = X_tensor.to(device)
                        
                        output = model(X_tensor)
                        return output.mean.cpu().numpy()
            else:
                # 其他模型类型
                def make_pytorch_predict(model_obj):
                    def pytorch_predict(X):
                        try:
                            if hasattr(model_obj, 'predict'):
                                return model_obj.predict(X)
                            elif hasattr(model_obj, '__call__'):
                                # 对于PyTorch模型
                                if not isinstance(X, torch.Tensor):
                                    X = torch.tensor(X, dtype=torch.float32)
                                return model_obj(X).detach().numpy()
                            else:
                                return np.zeros(len(X))
                        except Exception as e:
                            print(f"预测函数错误: {e}")
                            return np.zeros(len(X))
                    return pytorch_predict
                
                predict_fn = make_pytorch_predict(model)
        
        except ImportError:
            # 如果没有PyTorch，使用通用预测函数
            def predict_fn(X):
                try:
                    if hasattr(model, 'predict'):
                        return model.predict(X)
                    else:
                        return np.zeros(len(X))
                except:
                    return np.zeros(len(X))
        
        # 获取特征索引
        feat1_idx = feature_names.index(feat1)
        feat2_idx = feature_names.index(feat2)
        
        # 使用scikit-learn计算2D PDP
        try:
            pdp_result = partial_dependence(
                predict_fn, X_sample.values, 
                features=[feat1_idx, feat2_idx], 
                grid_resolution=20, 
                kind='average'
            )
            
            # 提取数据
            pdp_values = pdp_result['average'][0]
            feat1_values = pdp_result['grid_values'][0]
            feat2_values = pdp_result['grid_values'][1]
            
            # 创建网格
            f1_mesh, f2_mesh = np.meshgrid(feat1_values, feat2_values)
            
            # 预测VHI值
            vhi_pred = pdp_values.T
            
        except Exception as e:
            print(f"使用scikit-learn计算PDP失败: {e}")
            print("尝试使用自定义方法...")
            
            # 自定义PDP计算
            feat1_range = np.linspace(X_sample[feat1].min(), X_sample[feat1].max(), 20)
            feat2_range = np.linspace(X_sample[feat2].min(), X_sample[feat2].max(), 20)
            
            f1_mesh, f2_mesh = np.meshgrid(feat1_range, feat2_range)
            vhi_pred = np.zeros_like(f1_mesh)
            
            # 基准数据（固定其他特征为均值）
            base_data = X_sample.mean().values
            
            for i in range(len(feat1_range)):
                for j in range(len(feat2_range)):
                    # 创建测试样本
                    test_sample = base_data.copy()
                    test_sample[feat1_idx] = feat1_range[i]
                    test_sample[feat2_idx] = feat2_range[j]
                    
                    # 预测
                    pred = predict_fn(test_sample.reshape(1, -1))
                    vhi_pred[j, i] = pred[0] if len(pred) > 0 else 0
        
        # 创建热力图
        contour = ax.contourf(f1_mesh, f2_mesh, vhi_pred, cmap='viridis', alpha=0.9, levels=15)
        
        # 添加等高线
        contour_lines = ax.contour(f1_mesh, f2_mesh, vhi_pred, colors='white', alpha=0.6, levels=8)
        ax.clabel(contour_lines, inline=True, fontsize=10, fmt='%.3f')
        
        # 获取特征类别
        group1 = categorize_feature(feat1)
        group2 = categorize_feature(feat2)
        
        # 用于图表显示的更清晰特征名称 - 使用与其他图表一致的简化名称
        feat1_display = simplify_feature_name_for_plot(feat1, max_length=4)
        feat2_display = simplify_feature_name_for_plot(feat2, max_length=4)
        
        # 设置标题和轴标签
        if resolution:
            title = f"PDP Interaction: {feat1_display} × {feat2_display}"
        else:
            title = f"PDP Interaction: {feat1_display} × {feat2_display}"
        
        # 设置标题和轴标签
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(feat1_display, fontsize=12, fontweight='bold')
        ax.set_ylabel(feat2_display, fontsize=12, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Predicted VHI', fontsize=12, fontweight='bold')
        
        # 应用增强样式
        enhance_plot_style(ax, xlabel=feat1_display, ylabel=feat2_display)
        
        # 保存图表
        if output_dir:
            # 创建安全文件名
            safe_feat1 = re.sub(r'[\\/*?:"<>|]', "_", feat1)
            safe_feat2 = re.sub(r'[\\/*?:"<>|]', "_", feat2)
            
            if resolution:
                fig_path = os.path.join(output_dir, f"{resolution}_pdp_interaction_{safe_feat1}_{safe_feat2}.png")
            else:
                fig_path = os.path.join(output_dir, f"pdp_interaction_{safe_feat1}_{safe_feat2}.png")
            
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Interaction PDP saved to: {fig_path}")
            
            plt.close(fig)
            return fig_path
        
        return fig
    
    except Exception as e:
        print(f"Error generating PDP for {feat1} × {feat2}: {e}")
        return None 
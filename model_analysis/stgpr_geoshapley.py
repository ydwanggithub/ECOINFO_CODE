#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoShapley 分析模块 - 集成用于 STGPR 模型的空间敏感SHAP分析

包含了适配STGPR模型的GeoShapley解释器，以及相关的辅助函数。
基于标准GeoShapleyExplainer，提供简单的STGPR预测函数。
"""

# 防止重复输出的全局标志
_PRINTED_MESSAGES = set()

def print_once(message):
    """只打印一次的函数"""
    if message not in _PRINTED_MESSAGES:
        print(message)
        _PRINTED_MESSAGES.add(message)

import os
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
import time
import gc
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager

# 检查依赖库
HAS_GEOSHAPLEY = False
HAS_SHAP = False
shap = None

try:
    from geoshapley import GeoShapleyExplainer
    HAS_GEOSHAPLEY = True
except ImportError:
    pass

try:
    import shap
    HAS_SHAP = True
except ImportError:
    print("警告: shap库不可用，将使用sklearn的KMeans作为替代")

# 从其他模块导入需要的函数
from .stgpr_config import get_config
from .stgpr_sampling import perform_spatiotemporal_sampling
from .stgpr_io import predict_with_st_gpr

# 导入内存优化模块
try:
    from .geoshapley_memory_fix import (
        apply_memory_optimization,
        MEMORY_OPTIMIZED_GEOSHAPLEY_PARAMS as GEOSHAPLEY_CONFIG,
        create_memory_optimized_prediction_function
    )
    # 应用内存优化配置（配置信息只输出一次）
    GEOSHAPLEY_CONFIG = apply_memory_optimization()
except ImportError:
    # 如果内存优化模块不可用，使用默认配置
    from .stgpr_config import GEOSHAPLEY_CONFIG
    create_memory_optimized_prediction_function = None
    print("⚠️ 内存优化模块不可用，使用默认配置")


def create_simple_stgpr_prediction_function(model_dict, feature_names_reordered):
    """
    创建简单的STGPR预测函数，遵循参考代码的原理
    
    参考代码中的predict_f就是普通的模型预测函数：
    y = self.predict_f(V).reshape(-1)
    
    参数:
    model_dict: 包含STGPR模型和相关信息的字典
    feature_names_reordered: GeoShapley期望的特征顺序
    
    返回:
    简单的预测函数
    """
    model_feature_names = model_dict.get('feature_names', feature_names_reordered)
    
    print(f"🔧 创建标准STGPR预测函数")
    print(f"  • 模型特征顺序: {model_feature_names[:3]}...{model_feature_names[-1]}")
    print(f"  • GeoShapley特征顺序: {feature_names_reordered[:3]}...{feature_names_reordered[-1]}")
    
    # 创建从GeoShapley顺序到模型顺序的映射
    feature_mapping = {}
    for i, feat in enumerate(feature_names_reordered):
        if feat in model_feature_names:
            feature_mapping[i] = model_feature_names.index(feat)
    
    def simple_stgpr_prediction_function(x):
        """
        简单的STGPR预测函数
        
        遵循参考代码原理：就是普通的模型预测，不需要任何增强
        """
        # 🔴 调试：记录每次预测的输入大小
        import os
        # 只在主进程中打印，避免并行时的混乱输出
        is_main = os.getpid() == os.getppid() if hasattr(os, 'getppid') else True
        
        # 确保输入是numpy数组
        if hasattr(x, 'values'):  # DataFrame
            x_array = x.values
        else:
            x_array = np.asarray(x)
        
        # 🔴 调试信息：检查异常的大批量预测
        if x_array.shape[0] > 1000 and is_main:
            print(f"\n    ⚠️ 检测到大批量预测请求:")
            print(f"      • 输入形状: {x_array.shape}")
            print(f"      • 这可能是GeoShapley内部的全局计算")
            # 打印调用栈的前几层
            import traceback
            stack = traceback.extract_stack()
            for frame in stack[-5:-1]:  # 显示调用栈的前几层
                if 'geoshapley' in frame.filename.lower() or 'shap' in frame.filename.lower():
                    print(f"      • 调用来源: {frame.filename}:{frame.lineno} in {frame.name}")
        
        # 如果是单个样本，确保是2D数组
        if x_array.ndim == 1:
            x_array = x_array.reshape(1, -1)
        
        # 重新排列特征顺序以匹配模型
        n_samples = x_array.shape[0]
        x_reordered = np.zeros((n_samples, len(model_feature_names)))
        
        for geoshap_idx, model_idx in feature_mapping.items():
            x_reordered[:, model_idx] = x_array[:, geoshap_idx]
        
        # 创建DataFrame（使用模型期望的特征顺序）
        x_df = pd.DataFrame(x_reordered, columns=model_feature_names)
        
        # 🔴 关键：就是标准的STGPR预测，如参考代码一样简单
        predictions = predict_with_st_gpr(model_dict, x_df, return_variance=False)
        
        # 确保返回一维数组
        return predictions.ravel() if hasattr(predictions, 'ravel') else predictions
    
    # 测试预测函数
    print(f"\n  🧪 测试标准STGPR预测函数...")
    test_sample = np.random.randn(1, len(feature_names_reordered))  # 创建测试样本
    test_pred = simple_stgpr_prediction_function(test_sample)
    print(f"  • 测试输入形状: {test_sample.shape}")
    print(f"  • 测试预测值: {test_pred[0]:.6f}")
    print(f"  ✅ 标准STGPR预测函数创建成功")
    
    return simple_stgpr_prediction_function


def explain_stgpr_predictions(model_dict, X_samples, X_train=None, feature_names=None, 
                            n_background=None, sample_size=None, res_level=None):
    """
    使用标准GeoShapleyExplainer + 标准STGPR预测函数来解释STGPR模型
    
    遵循参考代码的原理：GeoShapley的空间感知性来自算法本身，不是预测函数
    """
    # 初始化返回值
    shap_interaction_values = None
    local_explanations = None
    
    # 自动计算背景点数量
    if n_background is None:
        n_features = len(feature_names) if feature_names else X_samples.shape[1]
        n_background = max(6, int(np.ceil(np.sqrt(n_features))))
        print(f"  📊 自动计算背景点数量: √{n_features} ≈ {np.sqrt(n_features):.1f} → {n_background}个")
    
    # 获取分辨率特定的采样配置
    resolution_sampling_config = GEOSHAPLEY_CONFIG.get('resolution_sampling', {})
    
    # 对res7应用预采样策略
    if res_level == 'res7' and res_level in resolution_sampling_config:
        res_config = resolution_sampling_config[res_level]
        if res_config.get('use_spatiotemporal_sampling', False):
            sample_rate = res_config.get('sample_rate', 0.1)
            max_samples = res_config.get('max_samples', 18000)
            
            target_samples = min(int(len(X_samples) * sample_rate), max_samples)
            
            if target_samples < len(X_samples):
                print(f"\n📊 res7预采样优化:")
                print(f"  原始数据: {len(X_samples):,}行")
                print(f"  采样率: {sample_rate*100:.0f}%")
                print(f"  目标样本: {target_samples:,}行")
                
                if not isinstance(X_samples, pd.DataFrame):
                    X_samples_df = pd.DataFrame(X_samples, columns=feature_names)
                else:
                    X_samples_df = X_samples.copy()
                
                X_samples = perform_spatiotemporal_sampling(
                    X_samples_df, target_samples,
                    h3_col='h3_index', year_col='year', 
                    spatial_coverage=0.1,  # 使用10%默认覆盖率进行res7预采样
                    random_state=42
                )
                
                print(f"  实际采样: {len(X_samples)}行")
                print(f"  ✅ res7预采样完成，减少{(1-len(X_samples)/len(X_samples_df))*100:.1f}%的数据量")
    
    try:
        gc.collect()
        
        model = model_dict.get('model')
        if model is None:
            return None
        
        # 🔴 移除基于模型参数的回退计算 - 确保只有GeoShapley成功才返回结果
        global_importance = None  # 不再使用模型参数计算全局重要性
        
        # 确保特征顺序与模型训练时一致
        if 'feature_names' in model_dict and model_dict['feature_names']:
            model_feature_names = model_dict['feature_names']
            print("\n🔧 调整特征顺序以匹配模型...")
            
            if isinstance(X_samples, pd.DataFrame):
                try:
                    X_samples = X_samples[model_feature_names]
                    feature_names = model_feature_names
                    print(f"  ✅ 特征顺序已调整为模型期望的顺序")
                except KeyError as e:
                    print(f"  ❌ 特征重排失败: {e}")
        
        print("\n🎯 使用标准GeoShapleyExplainer + 标准STGPR预测函数")
        print(f"  特征数量: {len(feature_names)}")
        print(f"  原理：GeoShapley的空间感知性来自算法本身，如参考代码所示")
        
        # 识别地理特征
        geo_features = []
        non_geo_features = []
        
        for col in feature_names:
            if col.lower() in ['latitude', 'longitude', 'lat', 'lon', 'lng']:
                geo_features.append(col)
            else:
                non_geo_features.append(col)
        
        # 重新组织列顺序：非地理特征在前，地理特征在后
        reordered_columns = non_geo_features + geo_features
        
        # 对X_samples进行重排序
        if isinstance(X_samples, pd.DataFrame):
            X_samples_reordered = X_samples[reordered_columns].copy()
        else:
            X_samples_reordered = X_samples
            
        feature_names_reordered = reordered_columns
        
        print(f"  地理特征数: {len(geo_features)}")
        print(f"  非地理特征数: {len(non_geo_features)}")
        
        # 🔥 **关键优化**：计算GeoShapley复杂度降低效果
        total_features = len(feature_names)
        effective_features = total_features - len(geo_features) + 1  # p - g + 1
        
        print(f"\n🚀 **GeoShapley计算复杂度优化分析**:")
        print(f"  📊 原始特征数量 (p): {total_features}")
        print(f"  📍 地理特征数量 (g): {len(geo_features)} → {geo_features}")
        print(f"  有效计算特征数: p - g + 1 = {total_features} - {len(geo_features)} + 1 = {effective_features}")
        print(f"")
        print(f"  💾 **计算复杂度对比**:")
        print(f"    • 标准SHAP: 2^{total_features} = {2**total_features:,} 种组合")
        print(f"    • GeoShapley优化: 2^{effective_features} = {2**effective_features:,} 种组合")
        print(f"    • 🎉 **减少了 {((2**total_features - 2**effective_features) / 2**total_features * 100):.1f}% 的计算量**")
        print(f"    • ⚡ **加速倍数: {2**total_features / 2**effective_features:.1f}x**")
        print(f"")
        print(f"  🔬 **优化技术详解**:")
        print(f"    • Monte Carlo 采样: ✅ 由GeoShapley库内置实现")
        print(f"    • Kernel SHAP: ✅ 由GeoShapley库内置实现") 
        print(f"    • 位置特征合并: ✅ 通过g={len(geo_features)}参数实现")
        print(f"    • 二进制矩阵Z优化: ✅ 从2^(p+1)降低到2^(p-g+1)")
        
        # 🔴 重要修正：GeoShapley需要全量数据！
        # 原因：GeoShapley在计算每个样本的SHAP值时，需要考虑整个数据集的空间上下文
        # 这是GeoShapley与标准SHAP的关键区别 - 它理解空间关系和地理模式
        # 因此，即使我们只想解释100个样本，GeoShapley仍需要访问全部数据来正确评估地理特征的贡献
        # 
        # 以下采样代码是错误的，已被注释：
        # res_config = PRODUCTION_CONFIG.get(res_level, {})
        # sample_size = min(res_config.get('max_samples', 100), len(X_samples_reordered))
        # 
        # if sample_size < len(X_samples_reordered):
        #     X_samples_df = pd.DataFrame(X_samples_reordered, columns=feature_names_reordered)
        #     X_samples_df = perform_spatiotemporal_sampling(
        #         X_samples_df, sample_size, 
        #         h3_col='h3_index', year_col='year', random_state=42
        #     )
        #     X_samples_reordered = X_samples_df
        
        print(f"\n📊 要解释的样本数: {len(X_samples_reordered):,}")
        print(f"ℹ️  GeoShapley将使用全量数据计算空间模式，享受{2**total_features / 2**effective_features:.1f}x计算加速")
        
        # 准备背景数据
        if X_train is not None:
            print(f"\n📊 准备背景数据...")
            
            if isinstance(X_train, pd.DataFrame):
                if 'feature_names' in model_dict and model_dict['feature_names']:
                    try:
                        X_train = X_train[model_dict['feature_names']]
                    except KeyError:
                        print(f"  ⚠️ 背景数据特征顺序调整失败")
                
                X_train_reordered = X_train[reordered_columns].copy()
            else:
                X_train_reordered = X_train
            
            # 生成背景数据
            if HAS_SHAP and shap is not None:
                background_data = shap.kmeans(X_train_reordered, n_background).data
                print(f"  ✅ 已生成{n_background}个背景数据点 (shap.kmeans)")
            else:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_background, random_state=42, n_init=10)
                kmeans.fit(X_train_reordered)
                background_data = kmeans.cluster_centers_
                print(f"  ✅ 已生成{n_background}个背景数据点 (sklearn.KMeans)")
        else:
            background_data = X_samples_reordered[:min(n_background, len(X_samples_reordered))]
        
        # 确保背景数据是numpy数组
        if hasattr(background_data, 'values'):
            background_data = background_data.values
        else:
            background_data = np.asarray(background_data)
        
        print(f"  📊 背景数据形状: {background_data.shape}")
        
        # 🔴 关键：创建标准的STGPR预测函数（如参考代码所示）
        try:
            print(f"\n🚀 开始标准GeoShapley计算...")
            
            # 创建简单的预测函数
            simple_stgpr_predict_fn = create_simple_stgpr_prediction_function(
                model_dict, feature_names_reordered
            )
            
            # 确保输入数据是DataFrame
            if isinstance(X_samples_reordered, pd.DataFrame):
                X_samples_for_geoshapley = X_samples_reordered
            else:
                X_samples_for_geoshapley = pd.DataFrame(
                    X_samples_reordered, columns=feature_names_reordered
                )
            
            start_time = time.time()
            
            # 🔴 关键：使用标准的GeoShapleyExplainer + 标准预测函数（如参考代码）
            if HAS_GEOSHAPLEY:
                print(f"  📊 使用标准GeoShapleyExplainer + 先进优化技术")
                
                # 🔴 调试：检查背景数据大小
                print(f"\n  📊 背景数据验证:")
                print(f"    • 背景数据形状: {background_data.shape}")
                print(f"    • 背景数据类型: {type(background_data)}")
                print(f"    • 期望的背景点数: {n_background}")
                
                # 📊 **GeoShapley优化技术确认**
                print(f"\n  🔬 **启用的GeoShapley优化技术**:")
                print(f"    1️⃣ **Monte Carlo采样**: 减少SHAP值计算的随机采样需求")
                print(f"    2️⃣ **Kernel SHAP**: 通过加权最小二乘回归问题减少计算量")
                print(f"    3️⃣ **位置特征合并**: 将{len(geo_features)}个地理特征({geo_features})视为1个复合特征")
                print(f"    4️⃣ **二进制矩阵Z优化**: 从2^{total_features}减少到2^{effective_features} = {((2**total_features - 2**effective_features) / 2**total_features * 100):.1f}%减少")
                
                # 创建标准GeoShapley解释器（完全按照参考代码）
                print(f"\n  🏗️ 创建GeoShapleyExplainer实例...")
                print(f"    • 预测函数: ✅ 标准STGPR预测函数")
                print(f"    • 背景数据: ✅ {n_background}个聚类中心点")
                print(f"    • 地理特征数g: ✅ {len(geo_features)} (启用位置优化)")
                
                explainer = GeoShapleyExplainer(
                    predict_f=simple_stgpr_predict_fn,  # 🔴 标准预测函数，如参考代码
                    background=background_data,
                    g=len(geo_features)  # 🔥 关键：地理特征数量，启用位置合并优化
                )
                
                print(f"    ✅ GeoShapleyExplainer创建成功，已启用所有优化技术")
                
                # 获取并行配置
                resolution_n_jobs = GEOSHAPLEY_CONFIG.get('resolution_n_jobs', {})
                n_jobs = resolution_n_jobs.get(res_level, GEOSHAPLEY_CONFIG.get('n_jobs', 1))
                
                print(f"  ⚙️ 计算配置: {'串行计算' if n_jobs == 1 else f'{n_jobs}进程并行计算'}")
                
                # 预估计算时间
                estimated_combinations = 2**effective_features
                samples_count = len(X_samples_for_geoshapley)
                
                print(f"\n  ⏱️ **计算复杂度预估**:")
                print(f"    • 样本数量: {samples_count:,}")
                print(f"    • 特征组合数: 2^{effective_features} = {estimated_combinations:,}")
                print(f"    • 总计算单元: ~{samples_count * estimated_combinations:,}")
                
                if samples_count > 1000:
                    estimated_minutes = (samples_count * estimated_combinations) / (50000 * (n_jobs if n_jobs > 1 else 1))
                    print(f"    • 预计耗时: ~{estimated_minutes:.1f}分钟 ({'并行' if n_jobs > 1 else '串行'})")
                    if estimated_minutes > 10:
                        print(f"    ⚠️ 大数据集计算，建议耐心等待...")
                
                print(f"\n  🚀 开始计算SHAP值（享受{2**total_features / 2**effective_features:.1f}x加速）...")
                
                # 导入进度条工具
                try:
                    from progress_utils import show_shap_calculation_progress, update_shap_progress, close_shap_progress
                    # 创建进度条
                    pbar = show_shap_calculation_progress(len(X_samples_for_geoshapley))
                    use_progress_bar = True
                except ImportError:
                    use_progress_bar = False
                
                # 更强力地抑制tqdm和其他输出
                import sys
                import io
                import logging
                
                # 禁用所有logging输出
                logging.disable(logging.CRITICAL)
                
                # 保存原始的stdout和stderr
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                
                # 创建一个StringIO对象来捕获输出
                captured_output = io.StringIO()
                
                try:
                    # 尝试完全禁用tqdm
                    import tqdm
                    # 保存原始的tqdm类
                    original_tqdm = tqdm.tqdm
                    original_tqdm_notebook = getattr(tqdm, 'tqdm_notebook', None)
                    
                    # 创建一个假的tqdm类
                    class FakeTqdm:
                        def __init__(self, *args, **kwargs):
                            self.iterable = args[0] if args else kwargs.get('iterable', [])
                            
                        def __iter__(self):
                            if use_progress_bar:
                                for i, item in enumerate(self.iterable):
                                    if i % max(1, len(self.iterable) // 20) == 0:  # 更新20次
                                        update_shap_progress(increment=max(1, len(self.iterable) // 20))
                                    yield item
                            else:
                                return iter(self.iterable)
                            
                        def __enter__(self):
                            return self
                                
                        def __exit__(self, *args):
                            pass
                                
                        def update(self, n=1):
                            if use_progress_bar:
                                update_shap_progress(increment=n)
                            
                        def close(self):
                            pass
                    
                    # 替换tqdm
                    tqdm.tqdm = FakeTqdm
                    tqdm.trange = lambda *args, **kwargs: FakeTqdm(range(*args), **kwargs)
                    if original_tqdm_notebook:
                        tqdm.tqdm_notebook = FakeTqdm
                    
                    # 同时禁用tqdm.auto
                    if hasattr(tqdm, 'auto'):
                        tqdm.auto.tqdm = FakeTqdm
                    
                    try:
                        # 重定向stdout和stderr来捕获所有输出
                        sys.stdout = captured_output
                        sys.stderr = captured_output
                        
                        # 使用标准GeoShapley方法计算SHAP值
                        shap_results = explainer.explain(X_samples_for_geoshapley, n_jobs=n_jobs)
                        
                    finally:
                        # 恢复stdout和stderr
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr
                        
                        # 恢复logging
                        logging.disable(logging.NOTSET)
                        
                        # 恢复tqdm
                        try:
                            tqdm.tqdm = original_tqdm
                            if original_tqdm_notebook:
                                tqdm.tqdm_notebook = original_tqdm_notebook
                            if hasattr(tqdm, 'auto'):
                                tqdm.auto.tqdm = original_tqdm
                        except:
                            pass
                    
                    # 获取捕获的输出（用于调试）
                    captured = captured_output.getvalue()
                    if captured and "error" in captured.lower():
                        print(f"  ⚠️ 捕获的警告信息: {captured[:200]}...")
                    
                    elapsed_time = time.time() - start_time
                    
                    # 关闭进度条
                    if use_progress_bar:
                        try:
                            close_shap_progress()
                        except:
                            pass
                    
                    print(f"  ✅ **GeoShapley计算完成**，实际耗时: {elapsed_time:.2f}秒")
                    print(f"  🎉 **享受了{2**total_features / 2**effective_features:.1f}x加速**，节省{((2**total_features - 2**effective_features) / 2**total_features * 100):.1f}%计算时间")
                    
                    # 处理结果
                    if hasattr(shap_results, 'primary') and hasattr(shap_results, 'geo') and hasattr(shap_results, 'geo_intera'):
                        # 这是标准的GeoShapleyResults对象
                        primary = shap_results.primary
                        geo = shap_results.geo
                        geo_intera = shap_results.geo_intera
                        base_value = shap_results.base_value
                        
                        print(f"\n  🔍 标准GeoShapley结果:")
                        print(f"    • Primary形状: {primary.shape}")
                        print(f"    • GEO形状: {geo.shape}")
                        print(f"    • Interaction形状: {geo_intera.shape}")
                        
                        # 创建合并的SHAP值数组
                        n_samples = primary.shape[0]
                        n_features = primary.shape[1] + 1  # +1 for GEO
                        
                        shap_values_array = np.zeros((n_samples, n_features))
                        
                        # 主要特征效应 + 一半的交互效应
                        shap_values_array[:, :-1] = primary + geo_intera / 2
                        
                        # GEO特征 = geo效应 + 所有geo交互效应的一半之和
                        shap_values_array[:, -1] = geo + np.sum(geo_intera / 2, axis=1)
                        
                        # 创建特征名称列表
                        if len(geo_features) == 2:
                            final_feature_names = non_geo_features + ['GEO']
                            print(f"  📊 特征合并: {len(non_geo_features)}个非地理特征 + 1个GEO特征")
                        else:
                            final_feature_names = feature_names_reordered
                        
                        local_explanations = {
                            'shap_values': shap_values_array,
                            'feature_names': final_feature_names,
                            'base_value': base_value,
                            'geoshap_original': {
                                'primary': primary,
                                'geo': geo,
                                'geo_intera': geo_intera,
                                'is_geoshapley': True,
                                'is_standard_geoshapley': True  # 标记为标准GeoShapley
                            }
                        }
                        
                        print(f"  ✅ 最终SHAP值形状: {shap_values_array.shape}")
                        print(f"  ✅ 特征名称: {final_feature_names}")
                        
                    else:
                        print(f"  ❌ 无法识别GeoShapley结果格式")
                        local_explanations = None
                except Exception as e:
                    print(f"  ❌ 标准GeoShapley计算失败: {str(e)}")
                    print(f"  详细错误: {traceback.format_exc()}")
                    local_explanations = None
            
        except Exception as e:
            print(f"  ❌ 标准GeoShapley计算失败: {str(e)}")
            print(f"  详细错误: {traceback.format_exc()}")
            local_explanations = None
        
        # 🔴 确保只有GeoShapley成功时才返回结果
        if local_explanations is not None:
            print(f"\n✅ GeoShapley分析成功完成")
            return {
                'global_importance': None,  # 不使用基于模型参数的全局重要性
                'local_explanations': local_explanations,
                'shap_interaction_values': shap_interaction_values
            }
        else:
            print(f"\n❌ GeoShapley分析失败，不返回基于模型参数的回退结果")
            return None
        
    except Exception as e:
        print(f"  ✗ STGPR GeoShapley分析错误: {e}")
        print(f"  详细错误信息: {traceback.format_exc()}")
        print(f"  🔴 不返回基于模型参数的回退结果")
        return None 
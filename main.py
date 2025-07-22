#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STGPR+GeoShapley 时空高斯过程回归可解释性分析框架 - 主文件

本模块实现了基于STGPR时空高斯过程回归与GeoShapley可解释性分析的建模和分析功能主流程，
用于探究丘陵山地植被健康对环境变化的滞后响应特征及地形调节机制。

作者: Yuandong Wang (wangyuandong@gnnu.edu.cn)
日期: 2025.07.26
"""

# 忽略tqdm的IProgress警告
import warnings
warnings.filterwarnings("ignore", message="IProgress not found")

# 首先设置环境变量和路径
from model_analysis.stgpr_config import setup_environment, configure_python_path, get_config, PROJECT_INFO, DATA_CONFIG

# 设置环境（必须在导入其他模块之前）
setup_environment()

# 配置Python路径
configure_python_path()

# 从工具模块导入函数（去除冗余导入）
from model_analysis.stgpr_utils import (
    clean_pycache,
    check_module_availability,
    create_train_evaluate_wrapper,
    ensure_dir_exists,
    prepare_features_for_stgpr,
    sample_data_for_testing,
    explain_stgpr_predictions,
    perform_spatiotemporal_sampling  # 统一从stgpr_utils导入
)

# 导入其他必要的模块
import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import argparse
import torch
import json

# 从model_analysis导入可视化模块
from model_analysis.stgpr_visualization import create_additional_visualizations

# 清理缓存
clean_pycache()

# 获取配置
CONFIG = get_config()
RANDOM_SEED = CONFIG['random_seed']
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# 打印项目信息
print(f"=== {PROJECT_INFO['name']} ===")
print(f"{PROJECT_INFO['description']}")
print(f"版本: {PROJECT_INFO['version']}\n")

def main(data_dir=None, output_dir=None, plots_to_create=None, use_parallel=False, 
         n_processes=4, data_resolutions=None, use_hyperopt=True, 
         max_hyperopt_evals=10, skip_validation=False, use_processed_data=True):
    """
    主函数：执行STGPR+GeoShapley可解释性分析
    
    参数:
    use_processed_data: 是否优先使用预处理后的数据文件（默认True，大幅提升加载速度）
    """
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    start_time = time.time()
    
    print(f"\n====== 开始STGPR+GeoShapley可解释性分析 ======")
    
    # 显示配置信息
    print("\n📋 当前配置:")
    print(f"  • 随机种子: 42")
    print(f"  • 基础诱导点数量: {CONFIG['model']['num_inducing_points']}")
    print(f"  • 训练迭代次数: {CONFIG['model']['num_iterations']}")
    print(f"  • 批处理大小: {CONFIG['model']['batch_size']}")
    print(f"  • GeoShapley进程数: {CONFIG['geoshapley']['n_jobs']}")
    print(f"  • 背景数据点: 自动计算（≥√特征数，向上取整）")
    if use_processed_data:
        print(f"  • 数据加载优化: ✅ 已启用（优先使用预处理数据）")
    else:
        print(f"  • 数据加载优化: ❌ 已禁用（强制重新预处理）")
    print()
    
    # 检查模块可用性
    print("\n=== 🔍 模块可用性检查 ===")
    modules_status = check_module_availability()
    print("=== 模块检查完成 ===\n")
    
    # 获取训练函数包装器
    train_evaluate_stgpr_model = create_train_evaluate_wrapper()
    
    # 确保输出目录存在
    output_dir = output_dir or DATA_CONFIG['default_output_dir']
    ensure_dir_exists(output_dir)
    
    # 设置目标变量
    target_column = DATA_CONFIG['target_column']
    
    # 设置数据目录
    if data_dir is None:
        data_dir = DATA_CONFIG['default_data_dir']
    
    # 🚀 优化的数据加载流程
    print("\n=== 1. 📊 加载完整数据集 (2000-2024年) ===")
    
    # 导入数据加载模块
    from data_processing.preprocessing import load_complete_dataset, load_data_files, load_processed_data_files, get_data_summary
    
    # 确定要处理的分辨率
    resolutions_to_process = data_resolutions if data_resolutions else DATA_CONFIG['default_resolutions']
    
    all_dfs = {}
    
    # 🎯 主要策略：直接加载2000-2024年完整数据集
    print("🎯 使用完整数据集 (包含2000-2020观测数据 + 2021-2024 ARIMA外推数据)")
    
    try:
        print("\n📊 加载完整数据集...")
        all_dfs = load_complete_dataset(
            data_dir=data_dir,
            resolutions=resolutions_to_process,
            verbose=True
        )
        
        if all_dfs:
            print("✅ 完整数据集加载成功！")
            
            # 显示数据集特征
            print("\n📋 数据集特征确认:")
            for res, df in all_dfs.items():
                year_range = (df['year'].min(), df['year'].max())
                print(f"  • {res}: {len(df):,}行 | 时间: {year_range[0]}-{year_range[1]} | 网格: {df['h3_index'].nunique():,}个")
            
        else:
            print("❌ 未能加载任何数据，尝试备用方法...")
            
    except Exception as e:
        print(f"⚠️ 主要加载方法失败: {e}")
        all_dfs = {}
    
    # 备用策略：使用传统方法
    if not all_dfs:
        print("\n🔄 使用备用数据加载方法...")
        try:
            all_dfs = load_data_files(
                data_dir=data_dir,
                resolutions=resolutions_to_process,
                verbose=True,
                force_reprocess=False
            )
        except Exception as e:
            print(f"❌ 备用加载方法也失败: {e}")
            print("请检查数据文件是否存在且格式正确")
            return
    
    # 🔄 备选策略：如果主要方法失败，使用优化加载方式
    if not all_dfs and use_processed_data:
        print("\n🔄 备选方案1：智能数据加载")
        
        try:
            # 尝试直接从预处理文件加载
            backup_dfs = load_processed_data_files(
                resolutions=resolutions_to_process,
                verbose=True
            )
            
            if backup_dfs:
                all_dfs.update(backup_dfs)
                print("✅ 备选方案1成功")
            
            # 对于仍缺失的分辨率，使用智能加载
            missing_resolutions = [res for res in resolutions_to_process if res not in all_dfs]
            if missing_resolutions:
                print(f"🔄 处理缺失分辨率: {missing_resolutions}")
                smart_load_dfs = load_data_files(
                    data_dir=data_dir, 
                    resolutions=missing_resolutions, 
                    force_reprocess=False,
                    verbose=True
                )
                all_dfs.update(smart_load_dfs)
                
        except Exception as e:
            print(f"❌ 备选方案1失败: {e}")
    
    # 🔄 备选策略2：强制重新预处理
    if not all_dfs and not use_processed_data:
        print("\n🔄 备选方案2：强制重新预处理")
        
        try:
            all_dfs = load_data_files(
                data_dir=data_dir, 
                resolutions=resolutions_to_process, 
                force_reprocess=True,
                verbose=True
            )
            print("✅ 备选方案2成功")
        except Exception as e:
            print(f"❌ 备选方案2失败: {e}")
    
    # ❌ 最终回退：传统加载方式（保持兼容性）
    if not all_dfs:
        print("\n❌ 所有优化方式失败，使用最后的兼容性回退...")
        
        file_patterns = DATA_CONFIG['file_patterns']
        for res in resolutions_to_process:
            if res in file_patterns:
                file_path = os.path.join(data_dir, file_patterns[res])
                if os.path.exists(file_path):
                    try:
                        # 直接读取CSV文件作为最后的手段
                        df = pd.read_csv(file_path)
                        all_dfs[res] = df
                        print(f"✓ 兼容性加载 {res}: {df.shape}")
                    except Exception as load_error:
                        print(f"❌ 兼容性加载 {res} 失败: {load_error}")
    
    # 检查最终结果
    if not all_dfs:
        print("❌ 无法加载任何数据，程序退出")
        print("💡 请检查:")
        print("  1. data/目录中是否存在 ALL_DATA_with_VHI_PCA_{res}.csv 文件")
        print("  2. 文件权限是否正确")
        print("  3. 文件格式是否完整")
        return None
    
    # 显示最终加载结果
    try:
        print(f"\n🎉 数据加载最终结果:")
        print(f"  • 成功加载: {len(all_dfs)}/{len(resolutions_to_process)} 个分辨率")
        
        # 显示加载效率统计
        load_time = time.time() - start_time
        total_rows = sum(len(df) for df in all_dfs.values())
        print(f"  • 加载时间: {load_time:.2f}秒")
        print(f"  • 总数据量: {total_rows:,}行")
        
        # 显示每个分辨率的时间范围
        print(f"\n📅 数据时间范围验证:")
        for res, df in all_dfs.items():
            if 'year' in df.columns:
                year_range = (df['year'].min(), df['year'].max())
                year_count = df['year'].nunique()
                print(f"  {res}: {year_range[0]}-{year_range[1]} ({year_count}年)")
            else:
                print(f"  {res}: 无年份信息")
        
        # 获取数据摘要
        data_summary = get_data_summary(all_dfs, verbose=True)
        print("✅ 数据加载与验证完成")
        
    except Exception as e:
        print(f"⚠️ 获取数据摘要失败: {e}")

    # 使用全量数据进行训练
    dfs = all_dfs
    
    # 为SHAP分析准备采样数据
    shap_dfs = {}
    
    # 🎯 平衡优化：适度增加网格数量，提升空间代表性，控制计算时间在合理范围
    # 根据用户要求：res5=100网格，res6=50网格，res7=200网格
    SHAP_SAMPLE_CONFIG = {
        'res5': {
            'spatial_coverage': 0.45,  # 45%空间覆盖率（99个网格）
            'temporal_coverage': 1.0,  # 100%时间覆盖率（25年，2000-2024）
            'min_networks': 50,        # 最少50个网格
            'description': '小数据集，45%空间覆盖+25年全覆盖（2000-2024），~15分钟'
        },
        'res6': {
            'spatial_coverage': 0.151,  # 15.1%空间覆盖率（200个网格）
            'temporal_coverage': 1.0,   # 100%时间覆盖率（25年，2000-2024）
            'min_networks': 100,        # 最少100个网格
            'description': '中等数据集，15.1%空间覆盖+25年全覆盖（2000-2024），~60分钟'
        },
        'res7': {
            'spatial_coverage': 0.058,  # 5.8%空间覆盖率（500个网格）
            'temporal_coverage': 1.0,   # 100%时间覆盖率（25年，2000-2024）
            'min_networks': 200,        # 最少200个网格
            'description': '大数据集，5.8%空间覆盖+25年全覆盖（2000-2024），~90分钟'
        }
    }
    
    for res, df in all_dfs.items():
        if res in SHAP_SAMPLE_CONFIG:
            config = SHAP_SAMPLE_CONFIG[res]
            
            # 计算目标样本量（基于空间覆盖率和时间覆盖率）
            total_h3 = df['h3_index'].nunique() if 'h3_index' in df.columns else len(df)
            total_years = df['year'].nunique() if 'year' in df.columns else 1
            
            target_h3 = int(total_h3 * config['spatial_coverage'])
            target_years = int(total_years * config['temporal_coverage'])
            
            # 预估目标样本量（仅用于传递给函数，实际样本量由函数自然产生）
            avg_records_per_grid_per_year = len(df) / (total_h3 * total_years) if total_h3 > 0 and total_years > 0 else 1
            estimated_samples = int(target_h3 * target_years * avg_records_per_grid_per_year)
            
            print(f"\n🔧 {res} SHAP采样策略:")
            print(f"    • 原始数据: {len(df):,}行 ({total_h3}网格 × {total_years}年)")
            print(f"    • 空间覆盖: {config['spatial_coverage']*100:.0f}% ({target_h3}网格)")
            print(f"    • 时间覆盖: {config['temporal_coverage']*100:.0f}% ({target_years}年)")
            print(f"    • 预估样本: ~{estimated_samples:,}个")
            print(f"    • 策略说明: {config['description']}")
            
            # 使用改进的时空分层采样（样本量自然产生）
            shap_dfs[res] = perform_spatiotemporal_sampling(
                df, estimated_samples,  # 传入预估值，但函数内部会自然产生实际样本量
                h3_col='h3_index', year_col='year',
                spatial_coverage=config['spatial_coverage'],  # 🚀 传入配置的空间覆盖率
                random_state=42
            )
            
            # 验证采样结果的时空覆盖
            actual_samples = len(shap_dfs[res])
            actual_networks = shap_dfs[res]['h3_index'].nunique() if 'h3_index' in shap_dfs[res].columns else 0
            actual_years = shap_dfs[res]['year'].nunique() if 'year' in shap_dfs[res].columns else 0
            
            # 计算实际覆盖率
            actual_spatial_coverage = actual_networks / total_h3 * 100 if total_h3 > 0 else 0
            actual_temporal_coverage = actual_years / total_years * 100 if total_years > 0 else 0
            
            print(f"    ✅ 采样结果: {actual_samples:,}个样本")
            print(f"    📍 空间覆盖: {actual_networks}/{total_h3} = {actual_spatial_coverage:.1f}%")
            print(f"    📅 时间覆盖: {actual_years}/{total_years} = {actual_temporal_coverage:.1f}%")
            print(f"    📊 实际采样率: {actual_samples/len(df)*100:.2f}%")
            
            # 评估代表性
            if actual_spatial_coverage >= 20 and actual_temporal_coverage >= 90:
                print(f"    🎯 时空代表性: ✅ 优秀")
            elif actual_spatial_coverage >= 10 and actual_temporal_coverage >= 70:
                print(f"    🎯 时空代表性: ⚠️ 良好")
            else:
                print(f"    🎯 时空代表性: ❌ 需要改进")
        else:
            # 默认配置
            default_samples = min(1000, len(df))
            shap_dfs[res] = perform_spatiotemporal_sampling(
                df, default_samples,
                h3_col='h3_index', year_col='year',
                spatial_coverage=0.05,  # 默认5%空间覆盖率
                random_state=42
            )
            print(f"{res}: 使用默认配置，SHAP分析将使用{len(shap_dfs[res])}个采样样本")
    
    # 创建结果字典
    results = {}
    
    # 🔴 新增：验证所有特征是否能被正确分类
    print("\n=== 🔍 特征分类验证 ===")
    from model_analysis.core import validate_all_features_categorized
    
    for res, df in dfs.items():
        print(f"\n{res}分辨率特征验证:")
        # 获取特征列（排除目标变量和非模型特征）
        non_feature_cols = [target_column, 'h3_index', 'original_h3_index', '.geo']
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        
        # 验证特征
        is_valid, validation_result = validate_all_features_categorized(feature_cols)
        
        # 🔥 优化的验证结果处理
        feature_set_type = validation_result.get('feature_set_type', '未知')
        optimization_status = validation_result.get('optimization_status', '未知')
        
        if not is_valid:
            # 根据特征集类型给出不同的处理建议
            if feature_set_type == "GeoShapley优化特征集":
                print(f"⚠️  {res}优化特征集需要调整")
                print(f"💡 建议：检查并调整特征预处理，确保包含14个核心特征")
                
                # 检查是否有意外的移除特征
                optimized_removed_present = validation_result.get('optimized_removed_present', [])
                if optimized_removed_present:
                    print(f"🔧 发现被优化移除的特征仍存在，建议移除以保持优化效果")
                
            elif feature_set_type == "完整特征集":
                print(f"❌ {res}完整特征集验证未通过!")
                print(f"💡 建议：检查特征名称并修正后再运行模型")
            else:
                print(f"❓ {res}特征集类型未知，需要进一步检查")
                print(f"💡 建议：确认特征数量是否符合预期（14个优化特征或19个完整特征）")
            
            # 根据失败类型决定是否继续
            failed_features = validation_result.get('failed', [])
            if failed_features:
                print("🚨 发现无法分类的特征，建议修正后重新运行")
                if skip_validation:
                    print("⚠️ 跳过验证模式已启用，自动继续...")
                else:
                    response = input("是否仍要继续？(y/n): ")
                    if response.lower() != 'y':
                        print("程序终止")
                        return None
            else:
                # 如果只是优化特征集的小问题，可以继续
                if feature_set_type == "GeoShapley优化特征集":
                    print("✅ 优化特征集基本正常，继续运行...")
                else:
                    if skip_validation:
                        print("⚠️ 跳过验证模式已启用，自动继续...")
                    else:
                        response = input("是否仍要继续？(y/n): ")
                        if response.lower() != 'y':
                            print("程序终止")
                            return None
        else:
            # 验证成功的情况
            if feature_set_type == "GeoShapley优化特征集":
                print(f"🎉 {res}GeoShapley优化特征集验证完美通过!")
                print(f"⚡ 已启用三重优化：特征减少 + 位置合并 + 算法优化")
            elif feature_set_type == "完整特征集":
                print(f"✅ {res}完整特征集验证通过!")
                print(f"📊 使用传统19个特征进行建模")
            else:
                print(f"✅ {res}特征集验证通过")
    
    print("=== 特征验证完成 ===\n")
    
    # 对每个分辨率训练一个时空高斯过程模型
    print("🤖 模型训练:")
    for resolution in resolutions_to_process:
        print(f"\n📈 训练{resolution}模型...")
        
        # 根据数据大小和分辨率自动确定诱导点数量
        base_inducing_points = CONFIG['model']['num_inducing_points']
        
        # 获取分辨率特定的配置
        res_config = CONFIG.get('resolution_specific', {}).get(resolution, {})
        inducing_points_factor = res_config.get('num_inducing_points_factor', 1.0)
        
        # 计算实际诱导点数量
        num_inducing_points = int(base_inducing_points * inducing_points_factor)
        
        # 确保诱导点数量不超过数据量
        X, y = prepare_features_for_stgpr(dfs[resolution], target=target_column)
        num_inducing_points = min(num_inducing_points, X.shape[0])
        
        # 计算实际比例
        actual_ratio = num_inducing_points / X.shape[0]
        
        print(f"  📊 诱导点策略:")
        print(f"    • 数据量: {X.shape[0]:,}行")
        print(f"    • 诱导点数量: {num_inducing_points} ({actual_ratio:.1%})")
        print(f"    • 选择方法: KMeans聚类（特征空间代表性点）")
        
        # 🔧 智能GPU选择策略 - 遵循配置文件设置
        use_gpu_for_training = False
        
        # 从配置文件获取GPU偏好设置
        prefer_gpu = res_config.get('prefer_gpu', False)
        
        if torch.cuda.is_available() and prefer_gpu:
            use_gpu_for_training = True
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  🎮 计算设备: GPU ({gpu_name}) - 按配置启用GPU加速")
        else:
            use_gpu_for_training = False
            if not torch.cuda.is_available():
                print(f"  💻 计算设备: CPU (GPU不可用)")
            elif not prefer_gpu:
                print(f"  💻 计算设备: CPU (配置为CPU优先，确保GeoShapley兼容性)")
            else:
                print(f"  💻 计算设备: CPU (未知原因)")
        
        # 获取分辨率特定的超参数优化配置
        actual_max_hyperopt_evals = res_config.get('max_hyperopt_evals', max_hyperopt_evals)
        print(f"  ⚡ 优化策略: 超参数评估次数{actual_max_hyperopt_evals}次")
        
        # 训练模型
        model_output_dir = os.path.join(output_dir, resolution)
        ensure_dir_exists(model_output_dir)
        
        # 选择合适的模型并训练
        if modules_status['HAS_STGPR']:
            result = train_evaluate_stgpr_model(
                dfs[resolution],
                resolution=resolution,
                output_dir=model_output_dir,
                target=target_column,
                use_gpu=use_gpu_for_training,  # 🚀 智能GPU选择
                use_hyperopt=use_hyperopt,
                num_inducing_points=num_inducing_points,
                max_hyperopt_evals=actual_max_hyperopt_evals
            )
            
            # 保存结果
            if result is not None:
                results[resolution] = result
    
            # 🔧 修复：保存模型结果到文件（仅保存可序列化的部分）
            import pickle
            model_file = os.path.join(model_output_dir, f"{resolution}_model_results.pkl")
            
            try:
                # 创建可序列化的结果副本（排除模型对象）
                if result is not None:
                    serializable_result = {}
                    for key, value in result.items():
                        # 跳过包含模型对象的键
                        if key in ['model', 'likelihood', 'mll', 'optimizer']:
                            print(f"    跳过不可序列化的项: {key}")
                            continue
                        
                        # 尝试序列化每个值
                        try:
                            import copy
                            # 深拷贝并测试是否可以序列化
                            temp_value = copy.deepcopy(value)
                            pickle.dumps(temp_value)  # 测试序列化
                            serializable_result[key] = temp_value
                        except Exception as e:
                            print(f"    跳过不可序列化的项 {key}: {str(e)[:100]}")
                            continue
                    
                    # 保存可序列化的结果
                    with open(model_file, 'wb') as f:
                        pickle.dump(serializable_result, f)
                    print(f"  💾 模型结果已保存至: {model_file}")
                    print(f"    保存的项目: {list(serializable_result.keys())}")
                else:
                    print(f"  ⚠️ 模型结果为空，跳过保存")
                    
            except Exception as save_error:
                print(f"  ⚠️ 保存模型结果时出错: {str(save_error)}")
                print(f"  💡 模型训练成功，但结果文件保存失败（不影响后续分析）")
            
            # 简化的性能报告
            if result and 'metrics' in result:
                metrics = result['metrics']
                if 'r2' in metrics and 'rmse' in metrics:
                    print(f"  ✓ 训练完成: 测试R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}")
                else:
                    print(f"  ✓ 训练完成，可用指标: {list(metrics.keys())}")
                
                print(f"    训练样本: {len(result.get('y_train', []))}, 测试样本: {len(result.get('y_test', []))}")
            else:
                print("  ⚠️ 无法获取性能指标")
        else:
            print(f"  ❌ STGPR模型不可用，跳过{resolution}分辨率的训练")
    
    # 检查模型训练结果
    successful_models = [res for res in results.keys() if results[res] is not None]
    if not successful_models:
        print("\n错误: 所有分辨率的模型训练均失败。")
        return
    
    # 使用GeoShapley计算特征重要性
    print("\n=== 2. 使用GeoShapley计算SHAP特征重要性 ===")
    print("🎯 GeoShapley核心特性:")
    print("  • 将经纬度(latitude, longitude)作为联合特征(GEO)处理")
    print("  • 在SHAP分布图中显示为单一的'GEO'特征")
    print("  • 捕捉空间效应的整体贡献，避免经纬度影响分散")
    print("  • 背景数据点自动计算：√特征数（向上取整）")
    
    # 对所有成功训练的模型进行SHAP分析
    for res in successful_models:
        try:
            if results[res] is not None:
                print(f"\n🔍 计算{res}分辨率的GeoShapley值...")
                
                # 获取模型和训练数据
                X_train = results[res].get('X')
                if X_train is None:
                    print(f"  警告: {res}缺少特征矩阵X，无法计算SHAP值")
                    continue
                
                # 使用专门的SHAP采样数据
                print(f"  🔧 使用智能数据策略:")
                print(f"    • 训练数据: {X_train.shape[0]}行 (用于生成背景数据)")
                
                # 获取SHAP专用的采样数据
                if res in shap_dfs:
                    shap_df = shap_dfs[res]
                    print(f"    • SHAP数据: {len(shap_df)}行 (专门采样，保持时空代表性)")
                    
                    # 从SHAP数据中提取特征
                    X_shap, _ = prepare_features_for_stgpr(shap_df, target=target_column)
                    
                    print(f"    • 特征一致性检查: 训练{X_train.shape[1]}列 vs SHAP{X_shap.shape[1]}列")
                    
                    # 确保特征列一致
                    if list(X_train.columns) == list(X_shap.columns):
                        print(f"    ✅ 特征列完全一致")
                        X_samples = X_shap
                    else:
                        print(f"    ⚠️ 特征列不一致，使用训练数据的列顺序")
                        X_samples = X_shap[X_train.columns]
                    
                    # 🔧 关键修复：确保X_samples包含h3_index
                    if 'h3_index' not in X_samples.columns and 'h3_index' in shap_df.columns:
                        print(f"    🔧 添加h3_index到X_samples")
                        X_samples = X_samples.copy()
                        X_samples['h3_index'] = shap_df['h3_index'].values[:len(X_samples)]
                    
                    # 🔧 关键修复：确保数据对应关系正确
                    print(f"    📍 数据对应检查:")
                    print(f"      • X_samples形状: {X_samples.shape}")
                    print(f"      • 原始SHAP数据形状: {shap_df.shape}")
                    print(f"      • h3_index列存在: {'h3_index' in X_samples.columns}")
                else:
                    print(f"    ⚠️ 未找到{res}的SHAP专用数据，回退到训练数据采样")
                    sample_size = min(200, len(X_train))
                    X_samples = X_train.sample(sample_size, random_state=RANDOM_SEED)
                
                print(f"    • 最终SHAP样本: {len(X_samples)}行")
                
                # 验证经纬度特征
                X_samples = validate_geo_features(X_samples, res)
                
                # 添加h3_index列（如果需要）- 修复：接收返回值
                X_samples = add_h3_index_if_needed(X_samples, res, shap_dfs)
                
                # 使用完整的训练数据作为背景数据
                X_train_full = X_train
                
                # 🔧 统一CPU逻辑：所有分辨率都使用相同的成功方案
                print(f"    • {res}使用统一的CPU训练+CPU GeoShapley方案")
                
                # 所有分辨率直接使用原始模型（都是CPU训练）
                geoshapley_model_dict = results[res]
                
                # 计算SHAP解释
                compute_shap_explanations(results, res, X_samples, X_train_full, geoshapley_model_dict)
                
        except Exception as e:
            print(f"  {res}的SHAP分析失败: {e}")
            import traceback
            traceback.print_exc()
            results[res]['feature_importance_failed'] = True
            results[res]['feature_importance'] = None
    
    # 显示每个分辨率的特征重要性
    print("\n=== 3. 特征重要性分析结果 ===")
    display_feature_importance(results, successful_models)
    
    # 创建可视化图表
    print("\n=== 4. 可视化分析 ===")
    create_visualizations(results, output_dir, plots_to_create)
    
    # 完成分析
    elapsed_time = time.time() - start_time
    print(f"\n====== STGPR+GeoShapley可解释性分析完成 ======")
    print(f"总耗时: {elapsed_time/60:.2f} 分钟")
    print(f"分析结果已保存至: {os.path.abspath(output_dir)}")

    # 重新生成修复后的可视化图表
    print("\n🔧 重新生成修复后的可视化图表...")
    try:
        from visualization.feature_plots import plot_feature_importance_comparison
        from visualization.shap_distribution_plots import plot_combined_shap_summary_distribution
        from visualization.geoshapley_spatial_top3 import plot_geoshapley_spatial_top3
        from visualization.regionkmeans_plot import plot_regionkmeans_feature_target_analysis
        
        # 1. 重新生成特征重要性比较图
        print("  📊 重新生成特征重要性比较图...")
        feature_importances_dict = {}
        for res in ['res5', 'res6', 'res7']:
            if res in results:
                feature_importances_dict[res] = results[res]['feature_importance']
        
        plot_feature_importance_comparison(
            feature_importances_dict, 
            output_dir=output_dir,
            results=results
        )
        
        # 2. 重新生成SHAP空间分布图
        print("  🌍 重新生成GeoShapley空间分布图...")
        plot_geoshapley_spatial_top3(
            results,
            output_dir=output_dir
        )
        
        # 3. 重新生成聚类分析图表
        print("  📈 重新生成聚类分析图表...")
        from visualization.regionkmeans_plot import plot_regionkmeans_shap_clusters_by_resolution
        
        # 先生成聚类分析，获取cluster_results
        fig, cluster_results = plot_regionkmeans_shap_clusters_by_resolution(
            results, 
            output_dir=output_dir
        )
        
        # 然后使用cluster_results生成特征贡献分析图
        if cluster_results:
            plot_regionkmeans_feature_target_analysis(
                cluster_results,
                output_dir=output_dir
            )
        else:
            print("  ⚠️ 聚类结果为空，跳过特征贡献分析图")
        
        print("  ✅ 所有修复后的可视化图表已重新生成")

    except Exception as e:
        print(f"  ⚠️ 重新生成图表时出现错误: {e}")
        import traceback
        traceback.print_exc()

def validate_geo_features(X_samples, res):
    """验证地理特征"""
    geo_features = [col for col in X_samples.columns if col.lower() in ['latitude', 'longitude']]
    if len(geo_features) == 2:
        print(f"  🌍 GeoShapley地理特征验证:")
        print(f"    • 发现地理特征: {geo_features}")
        
        # 检查经纬度是否在DataFrame的最后两列（GeoShapley要求）
        last_two_cols = X_samples.columns[-2:].tolist()
        if set(geo_features) == set(last_two_cols):
            print(f"    ✅ 经纬度已在最后两列，符合GeoShapley要求")
        else:
            print(f"    🔄 重新排列特征顺序，将经纬度移至最后...")
            # 重新排列列顺序：非地理特征 + 地理特征
            non_geo_cols = [col for col in X_samples.columns if col not in geo_features]
            new_column_order = non_geo_cols + geo_features
            X_samples = X_samples[new_column_order]
            print(f"    ✅ 特征重排完成: {len(non_geo_cols)}个非地理特征 + 2个地理特征")
    else:
        print(f"  ⚠️ 地理特征不完整: {geo_features}")
        print(f"    GeoShapley需要经纬度两个特征才能正确处理联合地理特征")
    
    return X_samples

def add_h3_index_if_needed(X_samples, res, shap_dfs):
    """添加h3_index列（如果需要）"""
    # 如果X_samples已经有h3_index列，直接返回
    if 'h3_index' in X_samples.columns:
        print(f"  📍 X_samples已包含h3_index列")
        return X_samples
    
    # 尝试从shap_dfs添加h3_index
    if res in shap_dfs and 'h3_index' in shap_dfs[res].columns:
        print(f"  📍 为SHAP计算添加空间标识信息...")
        try:
            shap_df_with_features = shap_dfs[res]
            
            if len(X_samples) == len(shap_df_with_features):
                X_samples = X_samples.copy()
                X_samples['h3_index'] = shap_df_with_features['h3_index'].values
                unique_h3_count = len(set(shap_df_with_features['h3_index']))
                print(f"    ✅ 成功添加h3_index列，共{unique_h3_count}个唯一H3值")
            else:
                print(f"    ⚠️ 数据长度不匹配: X_samples({len(X_samples)}) vs shap_df({len(shap_df_with_features)})")
                
                # 🔥 修复：尝试基于经纬度进行最近邻匹配
                if ('latitude' in X_samples.columns and 'longitude' in X_samples.columns and
                    'latitude' in shap_df_with_features.columns and 'longitude' in shap_df_with_features.columns):
                    
                    print(f"    🔄 尝试基于经纬度进行空间匹配...")
                    from sklearn.neighbors import NearestNeighbors
                    
                    # 构建KNN模型
                    shap_coords = shap_df_with_features[['latitude', 'longitude']].values
                    X_coords = X_samples[['latitude', 'longitude']].values
                    
                    knn = NearestNeighbors(n_neighbors=1, metric='haversine')
                    knn.fit(np.radians(shap_coords))  # 使用弧度单位
                    
                    # 找到最近的点
                    distances, indices = knn.kneighbors(np.radians(X_coords))
                    
                    # 添加h3_index
                    X_samples = X_samples.copy()
                    X_samples['h3_index'] = shap_df_with_features.iloc[indices.flatten()]['h3_index'].values
                    
                    avg_distance = distances.mean() * 6371000  # 转换为米
                    print(f"    ✅ 基于空间匹配成功添加h3_index，平均距离: {avg_distance:.1f}米")
                else:
                    print(f"    ❌ 无法进行空间匹配，缺少经纬度列")
                    
        except Exception as e:
            print(f"    添加h3_index时出错: {str(e)}")
    else:
        print(f"  ⚠️ 无法添加h3_index: res={res}, shap_dfs存在={res in shap_dfs if shap_dfs else False}")
    
    return X_samples

def compute_shap_explanations(results, res, X_samples, X_train_full, geoshapley_model_dict):
    """计算SHAP解释"""
    try:
        # 获取原始的feature_names，确保不包含h3_index
        feature_names = [col for col in X_train_full.columns if col not in ['h3_index']]
        
        print(f"  🚀 开始GeoShapley计算...")
        print(f"    • 样本数量: {len(X_samples)}")
        print(f"    • 特征数量: {len(feature_names)}")
        print(f"    • 背景数据: {len(X_train_full)}行")
        print(f"    • 背景数据点: 自动计算（基于特征数量）")
        print(f"    • 地理特征处理: 经纬度将自动合并为GEO特征")
        
        explanations = explain_stgpr_predictions(
            model_dict=geoshapley_model_dict,
            X_samples=X_samples,
            X_train=X_train_full,
            feature_names=feature_names,
            n_background=None,  # 让函数自动计算背景点数量
            res_level=res
        )
        
        # 处理解释结果
        if explanations:
            results[res]['explanations'] = explanations
            print(f"    ✅ GeoShapley计算成功")
            
            # 提取SHAP解释结果
            if 'local_explanations' in explanations and explanations['local_explanations']:
                local_exp = explanations['local_explanations']
                
                # 提取SHAP值
                if 'shap_values' in local_exp:
                    results[res]['shap_values'] = local_exp['shap_values']
                    results[res]['feature_names'] = local_exp.get('feature_names', feature_names)
                    
                    # 🔴 关键修复：确保X_sample与SHAP值维度一致
                    # 获取实际计算SHAP值的样本数量
                    n_shap_samples = local_exp['shap_values'].shape[0]
                    
                    # 如果原始X_samples比SHAP样本多，说明进行了采样
                    if len(X_samples) > n_shap_samples:
                        print(f"  📊 检测到采样: 原始{len(X_samples)}行 → SHAP计算{n_shap_samples}行")
                        # 需要重新采样X_samples以匹配SHAP值
                        from model_analysis.stgpr_sampling import perform_spatiotemporal_sampling
                        
                        # 🔥 修复：确保采样时保留h3_index列
                        sampling_kwargs = {
                            'random_state': 42  # 使用相同的随机种子
                        }
                        
                        # 检查是否有h3_index列用于采样
                        if 'h3_index' in X_samples.columns:
                            sampling_kwargs['h3_col'] = 'h3_index'
                            print(f"    🗺️ 使用h3_index进行空间分层采样")
                        
                        # 检查是否有year列用于采样
                        if 'year' in X_samples.columns:
                            sampling_kwargs['year_col'] = 'year'
                            print(f"    📅 使用year进行时间分层采样")
                        
                        try:
                            X_samples_matched = perform_spatiotemporal_sampling(
                                X_samples, n_shap_samples, 
                                spatial_coverage=0.05,  # 使用5%默认覆盖率进行重采样
                                **sampling_kwargs
                            )
                            
                            # 验证采样结果
                            if len(X_samples_matched) == n_shap_samples:
                                results[res]['X_sample'] = X_samples_matched
                                print(f"  ✅ X_sample已调整为{len(X_samples_matched)}行，与SHAP值匹配")
                                
                                # 验证h3_index是否保留
                                if 'h3_index' in X_samples.columns:
                                    if 'h3_index' in X_samples_matched.columns:
                                        unique_h3_before = len(set(X_samples['h3_index']))
                                        unique_h3_after = len(set(X_samples_matched['h3_index']))
                                        print(f"    🗺️ h3_index保留完整: {unique_h3_before} → {unique_h3_after}个唯一值")
                                    else:
                                        print(f"    ⚠️ 警告: 采样后丢失了h3_index列")
                            else:
                                print(f"  ❌ 采样结果数量不匹配: 期望{n_shap_samples}, 实际{len(X_samples_matched)}")
                                results[res]['X_sample'] = X_samples  # 使用原始数据
                                
                        except Exception as sampling_error:
                            print(f"  ❌ 采样失败: {sampling_error}")
                            print(f"  🔄 回退：使用原始X_samples的前{n_shap_samples}行")
                            X_samples_matched = X_samples.iloc[:n_shap_samples].copy()
                            results[res]['X_sample'] = X_samples_matched
                    else:
                        results[res]['X_sample'] = X_samples
                    
                    # 创建按特征名称索引的SHAP值字典
                    shap_values_by_feature = {}
                    for i, feat in enumerate(local_exp['feature_names']):
                        shap_values_by_feature[feat] = local_exp['shap_values'][:, i]
                    results[res]['shap_values_by_feature'] = shap_values_by_feature
                
                # 提取GeoShapley的三部分结果
                if 'geoshap_original' in local_exp:
                    geoshap_data = local_exp['geoshap_original']
                    if all(k in geoshap_data for k in ['primary', 'geo', 'geo_intera']):
                        results[res]['geoshapley_values'] = {
                            'primary_effects': geoshap_data['primary'],
                            'geo_effect': geoshap_data['geo'],
                            'interaction_effects': geoshap_data['geo_intera']
                        }
                        print(f"  ✅ 已保存GeoShapley三部分结果")
                        
            # 处理SHAP交互值
            process_shap_interactions(results, res, explanations)
            
            # 计算基于SHAP的特征重要性
            compute_shap_feature_importance(results, res, explanations, feature_names)
        else:
            print(f"    ❌ GeoShapley计算失败，未返回有效结果")
            results[res]['feature_importance_failed'] = True
            results[res]['feature_importance'] = None
            
    except Exception as e:
        print(f"    ❌ GeoShapley计算出错: {str(e)}")
        import traceback
        traceback.print_exc()
        results[res]['feature_importance_failed'] = True
        results[res]['feature_importance'] = None

def process_shap_interactions(results, res, explanations):
    """处理SHAP交互值"""
    if 'local_explanations' in explanations and explanations['local_explanations'] is not None:
        local_expl = explanations['local_explanations']
        if 'shap_interaction_values' in local_expl:
            results[res]['shap_interaction_values'] = local_expl['shap_interaction_values']
            print(f"    ✅ SHAP交互值已保存，形状: {local_expl['shap_interaction_values'].shape}")
            
            if 'interaction_sample_indices' in local_expl:
                results[res]['interaction_sample_indices'] = local_expl['interaction_sample_indices']

def compute_shap_feature_importance(results, res, explanations, feature_names):
    """计算基于SHAP的特征重要性"""
    if explanations is None:
        print(f"    ❌ 没有有效的解释结果")
        results[res]['feature_importance_failed'] = True
        results[res]['feature_importance'] = None
        return
    
    if 'local_explanations' in explanations and explanations['local_explanations'] is not None:
        local_expl = explanations['local_explanations']
        
        if 'shap_values' in local_expl:
            # 直接使用已经处理好的numpy数组（由stgpr_geoshapley.py返回）
            if isinstance(local_expl['shap_values'], np.ndarray):
                print(f"    📊 SHAP值统计: 形状{local_expl['shap_values'].shape}")
            else:
                print(f"    ❌ SHAP值不是numpy数组: {type(local_expl['shap_values'])}")
                results[res]['feature_importance_failed'] = True
                results[res]['feature_importance'] = None
                return
            
            # 使用GeoShapley返回的特征名（已经包含GEO）
            shap_feature_names = local_expl.get('feature_names', feature_names)
            
            # 验证SHAP值质量
            if local_expl['shap_values'].size > 0 and local_expl['shap_values'].ndim == 2:
                process_valid_shap_values(results, res, local_expl['shap_values'], shap_feature_names)
            else:
                print(f"    ❌ SHAP值数组无效或为空")
                print(f"    大小: {local_expl['shap_values'].size}, 维度: {local_expl['shap_values'].ndim}")
                results[res]['feature_importance_failed'] = True
                results[res]['feature_importance'] = None
        else:
            print(f"    ❌ 未找到有效的SHAP值数组")
            results[res]['feature_importance_failed'] = True
            results[res]['feature_importance'] = None
    else:
        print(f"    ❌ 未能生成SHAP解释")
        results[res]['feature_importance_failed'] = True
        results[res]['feature_importance'] = None

def process_valid_shap_values(results, res, shap_values, shap_feature_names):
    """处理有效的SHAP值"""
    shap_range = (shap_values.min(), shap_values.max())
    shap_std = shap_values.std()
    non_zero_ratio = np.count_nonzero(shap_values) / shap_values.size
    
    print(f"    • SHAP值范围: [{shap_range[0]:.6f}, {shap_range[1]:.6f}]")
    print(f"    • SHAP值标准差: {shap_std:.6f}")
    print(f"    • 非零值比例: {non_zero_ratio:.1%}")
    
    if shap_std > 1e-6 and non_zero_ratio > 0.1:
        print(f"    ✅ SHAP值质量良好，包含有意义的变异")
        
        # 计算每个特征的平均绝对SHAP值
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # 确保维度匹配
        if len(mean_abs_shap) != len(shap_feature_names):
            min_len = min(len(mean_abs_shap), len(shap_feature_names))
            mean_abs_shap = mean_abs_shap[:min_len]
            shap_feature_names = shap_feature_names[:min_len]
        
        # 直接使用原始SHAP值（与SHAP分布图保持一致）
        importance_values = mean_abs_shap
        print(f"    📊 使用原始SHAP值（与SHAP分布图保持一致）")
        
        # 创建特征重要性排名
        shap_feature_importance = [(shap_feature_names[i], float(importance)) 
                                  for i, importance in enumerate(importance_values)]
        
        # 按重要性降序排序
        shap_feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # 保存特征重要性
        results[res]['feature_importance'] = shap_feature_importance
        
        # 显示基于SHAP值的特征重要性
        print(f"\n    📊 基于SHAP值的特征重要性:")
        print(f"    {'特征名称':<20} {'重要性值':<10} {'相对百分比':<10}")
        print(f"    {'-'*40}")
        
        # 计算总重要性（用于百分比计算）
        total_importance = sum(imp for _, imp in shap_feature_importance)
        
        # 显示前10个特征
        for i, (feat, imp) in enumerate(shap_feature_importance[:10]):
            percentage = (imp / total_importance * 100) if total_importance > 0 else 0
            print(f"    {feat:<20} {imp:<10.4f} {percentage:<10.1f}%")
        
        if len(shap_feature_importance) > 10:
            print(f"    ... 还有 {len(shap_feature_importance) - 10} 个特征")
        
        # 为可视化添加必要的数据
        results[res]['shap_values'] = shap_values
        results[res]['feature_names'] = shap_feature_names  # 使用GeoShapley返回的特征名
        
        # 创建SHAP值字典 - 使用GeoShapley返回的特征名（包含GEO）
        shap_values_by_feature = {}
        for i, feature in enumerate(shap_feature_names):
            if i < shap_values.shape[1]:
                shap_values_by_feature[feature] = shap_values[:, i]
        results[res]['shap_values_by_feature'] = shap_values_by_feature
        
        print(f"\n    🏆 前5个重要特征（基于SHAP值）:")
        for i, (feat, imp) in enumerate(shap_feature_importance[:5]):
            print(f"      {i+1}. {feat}: {imp:.6f}")
    else:
        print(f"    ⚠️ SHAP值变异太小或零值太多")
        results[res]['feature_importance_failed'] = True
        results[res]['feature_importance'] = None

def display_feature_importance(results, successful_models):
    """显示特征重要性分析结果"""
    print("📊 基于GeoShapley的特征重要性分析:")
    print("  • 经纬度已作为联合GEO特征处理")
    print("  • 特征重要性基于SHAP值计算")
    print("  • 排序按重要性从高到低")
    
    # 导入特征分类函数
    from model_analysis.core import categorize_feature
    
    feature_importances_dict = {}
    
    for res in successful_models:
        print(f"\n🎯 {res} 特征重要性排名:")
        
        # 检查是否有特征重要性失败的标记
        if results[res].get('feature_importance_failed', False):
            print(f"  ❌ {res} 特征重要性计算失败")
            print(f"  ⚠️ 原因：SHAP值计算未成功完成")
            print(f"  💡 建议：")
            print(f"    1. 检查数据预处理是否正确")
            print(f"    2. 确保模型训练成功")
            print(f"    3. 验证特征数量是否匹配")
            print(f"    4. 考虑禁用特征预筛选")
            continue
            
        if 'feature_importance' in results[res] and results[res]['feature_importance']:
            # 获取特征重要性
            feature_importance = results[res]['feature_importance']
            
            # 去重处理
            unique_features = {}
            for feat, imp in feature_importance:
                std_feat = feat.lower() if isinstance(feat, str) else str(feat).lower()
                if std_feat not in unique_features or imp > unique_features[std_feat][1]:
                    unique_features[std_feat] = (feat, imp)
            
            # 转回列表并排序
            feature_importance = [(feat, imp) for _, (feat, imp) in unique_features.items()]
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # 更新结果字典
            feature_importances_dict[res] = feature_importance
            
            # 显示所有特征
            print(f"{'排名':<4} {'特征名称':<25} {'重要性':<10} {'类别':<15}")
            print("-" * 60)
            
            # 计算特征类别信息
            feature_categories = {}
            for feature, importance in feature_importance:
                category = categorize_feature(feature)
                feature_categories[feature] = category
            
            # 保存特征类别信息到结果中
            results[res]['feature_categories'] = feature_categories
            
            for i, (feature, importance) in enumerate(feature_importance, 1):
                category = feature_categories.get(feature, '未分类')
                print(f"{i:<4} {feature:<25} {importance:<10.4f} {category:<15}")
            
            # 统计各类别特征数量
            category_counts = {}
            for feature, importance in feature_importance:
                category = feature_categories.get(feature, '未分类')
                if category not in category_counts:
                    category_counts[category] = []
                category_counts[category].append((feature, importance))
            
            print(f"\n📈 {res} 特征类别统计:")
            for category, features in category_counts.items():
                avg_importance = sum(imp for _, imp in features) / len(features)
                print(f"  {category}: {len(features)}个特征, 平均重要性: {avg_importance:.4f}")
        else:
            print(f"  ❌ {res} 未找到特征重要性数据")
    
    # 跨分辨率特征重要性对比
    if len(feature_importances_dict) > 1:
        cross_resolution_comparison(feature_importances_dict)

def cross_resolution_comparison(feature_importances_dict):
    """跨分辨率特征重要性对比"""
    print(f"\n🔍 跨分辨率特征重要性对比:")
    
    # 收集所有特征
    all_features = set()
    for res_features in feature_importances_dict.values():
        for feat, _ in res_features:
            all_features.add(feat)
    
    print(f"  总共发现 {len(all_features)} 个不同特征")
    
    # 找出在所有分辨率中都重要的特征（前5名）
    consistent_important_features = set()
    for res, features in feature_importances_dict.items():
        top_5_features = set(feat for feat, _ in features[:5])
        if not consistent_important_features:
            consistent_important_features = top_5_features
        else:
            consistent_important_features &= top_5_features
    
    if consistent_important_features:
        print(f"  🏆 在所有分辨率中都排名前5的特征:")
        for feat in consistent_important_features:
            print(f"    • {feat}")
    else:
        print(f"  ℹ️ 没有在所有分辨率中都排名前5的特征")
        
    # 显示各分辨率的最重要特征
    print(f"  📊 各分辨率最重要特征:")
    for res, features in feature_importances_dict.items():
        if features:
            top_feature, top_importance = features[0]
            print(f"    {res}: {top_feature} ({top_importance:.4f})")

def create_visualizations(results, output_dir, plots_to_create):
    """创建可视化图表"""
    vis_success = False
    
    # 🔥 关键修复：确保GeoShapley结果被保存到结果文件中
    print("\n🔧 检查并保存GeoShapley结果...")
    for res in results:
        if res not in ['res5', 'res6', 'res7']:
            continue
            
        res_result = results[res]
        
        # 检查内存中是否有GeoShapley数据
        has_geoshapley = any(key in res_result for key in [
            'geoshapley_values', 'shap_values_by_feature', 'feature_importance'
        ])
        
        if has_geoshapley:
            print(f"  ✅ {res}: 检测到完整的GeoShapley数据")
            
            # 立即保存到文件，确保数据不丢失
            output_res_dir = os.path.join(output_dir, res)
            os.makedirs(output_res_dir, exist_ok=True)
            
            # 🔥 保存关键的GeoShapley数据和空间信息
            geoshapley_data = {}
            
            # 保存GeoShapley核心数据
            for key in ['geoshapley_values', 'shap_values_by_feature', 'feature_importance', 
                       'shap_values', 'feature_names', 'X_sample']:
                if key in res_result:
                    geoshapley_data[key] = res_result[key]
            
            # 🔥 确保保存完整的空间数据
            for spatial_key in ['df', 'raw_data', 'data']:
                if spatial_key in res_result and res_result[spatial_key] is not None:
                    geoshapley_data[spatial_key] = res_result[spatial_key]
                    print(f"    📍 保存空间数据字段: {spatial_key}")
                    break  # 只保存第一个找到的完整空间数据
            
            # 保存到单独的文件
            geoshapley_file = os.path.join(output_res_dir, f'{res}_geoshapley_data.pkl')
            try:
                import pickle
                with open(geoshapley_file, 'wb') as f:
                    pickle.dump(geoshapley_data, f)
                print(f"    💾 GeoShapley数据已保存: {geoshapley_file}")
                
                # 验证保存的数据
                saved_keys = list(geoshapley_data.keys())
                print(f"    📋 保存的数据字段: {saved_keys}")
                
                # 特别检查空间相关字段
                spatial_info = []
                if 'X_sample' in geoshapley_data:
                    X_sample = geoshapley_data['X_sample']
                    if hasattr(X_sample, 'columns'):
                        spatial_cols = [col for col in X_sample.columns if col in ['latitude', 'longitude', 'h3_index']]
                        if spatial_cols:
                            spatial_info.append(f"X_sample包含: {spatial_cols}")
                
                for spatial_key in ['df', 'raw_data', 'data']:
                    if spatial_key in geoshapley_data:
                        spatial_df = geoshapley_data[spatial_key]
                        if hasattr(spatial_df, 'columns'):
                            spatial_cols = [col for col in spatial_df.columns if col in ['latitude', 'longitude', 'h3_index']]
                            if spatial_cols:
                                spatial_info.append(f"{spatial_key}包含: {spatial_cols}")
                                break
                
                if spatial_info:
                    print(f"    🗺️ 空间信息: {'; '.join(spatial_info)}")
                else:
                    print(f"    ⚠️ 警告: 未检测到空间信息字段")
                    
            except Exception as e:
                print(f"    ❌ 保存GeoShapley数据失败: {e}")
        else:
            print(f"  ⚠️ {res}: 未检测到GeoShapley数据")
    
    # 在创建可视化之前先进行数据完整性检查
    try:
        from check_visualization_data import run_comprehensive_check
        print("\n🔍 执行可视化数据完整性检查...")
        data_check_passed = run_comprehensive_check(results)
        if not data_check_passed:
            print("\n⚠️ 警告: 数据检查发现问题，某些图表可能无法生成")
    except ImportError:
        print("ℹ️ 未找到数据检查模块，跳过检查")
    except Exception as e:
        print(f"⚠️ 数据检查时出错: {e}")
    
    try:
        # 尝试导入新接口模块
        from model_analysis import stgpr_visualization
        
        # 准备模型结果用于可视化
        print("\n使用stgpr_visualization模块准备数据并创建可视化...")
        
        # 🔥 直接传递完整的results，确保GeoShapley数据不丢失
        model_results = stgpr_visualization.prepare_stgpr_results_for_visualization(results, output_dir)
        
        # 🔥 验证数据传递是否成功
        print("\n🔍 验证GeoShapley数据传递...")
        for res in model_results:
            if res not in ['res5', 'res6', 'res7']:
                continue
                
            res_data = model_results[res]
            has_shap = any(key in res_data for key in [
                'shap_values_by_feature', 'geoshapley_values', 'feature_importance'
            ])
            
            if has_shap:
                print(f"  ✅ {res}: 可视化数据包含SHAP信息")
                
                # 输出详细信息
                if 'shap_values_by_feature' in res_data:
                    n_features = len(res_data['shap_values_by_feature'])
                    print(f"    📊 shap_values_by_feature: {n_features}个特征")
                
                if 'feature_importance' in res_data:
                    n_importance = len(res_data['feature_importance'])
                    print(f"    🏆 feature_importance: {n_importance}个特征")
                    
                if 'geoshapley_values' in res_data:
                    print(f"    🗺️ geoshapley_values: 三部分结构可用")
            else:
                print(f"  ❌ {res}: 可视化数据缺少SHAP信息！")
        
        # 创建所有可视化图表
        stgpr_visualization.create_all_visualizations(model_results, output_dir)
        
        print("✓ 使用stgpr_visualization模块创建的可视化图表已完成")
        vis_success = True
    except ImportError:
        print("⚠ 警告: 未能导入stgpr_visualization模块")
        print("将使用传统方式创建可视化图表...")
    except Exception as e:
        print(f"⚠ 警告: 使用stgpr_visualization模块时发生错误: {e}")
        traceback.print_exc()
        print("将尝试使用传统方式创建可视化图表...")
    
    # 只有在stgpr_visualization失败时才使用传统方式
    if not vis_success:
        try:
            create_additional_visualizations(
                results, 
                extended_results_by_resolution=None,
                output_dir=output_dir, 
                plots_to_create=plots_to_create
            )
            print("✓ 使用传统方式创建的可视化图表已完成")
        except Exception as e:
            print(f"⚠ 警告: 传统可视化方式失败: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='STGPR+GeoShapley 时空高斯过程回归可解释性分析框架')
        parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
        parser.add_argument('--resolutions', type=str, nargs='+', choices=['res5', 'res6', 'res7'], 
                            help='要处理的分辨率，例如 "res5 res6 res7"')
        parser.add_argument('--skip_validation', action='store_true', help='跳过特征验证询问，自动继续')
        parser.add_argument('--force_reprocess', action='store_true', help='强制重新预处理数据（默认使用预处理文件）')
        
        # 添加一个通用参数来捕获Jupyter发送的特殊参数
        parser.add_argument('--f', type=str, help='Jupyter notebook kernel file (自动忽略)', default=None)
        
        # 使用parse_known_args()而不是parse_args()，忽略未知参数
        args, unknown = parser.parse_known_args()
        
        # 只在有不是以--f开头的未知参数时才输出提示
        unknown_non_jupyter = [arg for arg in unknown if not arg.startswith('--f=')]
        if unknown_non_jupyter:
            print(f"注意：忽略未知参数: {unknown_non_jupyter}")
        
        # 启动主函数
        data_resolutions = args.resolutions if args.resolutions else None
        
        main(
            output_dir=args.output_dir,
            data_resolutions=data_resolutions,
            use_parallel=False,
            use_hyperopt=True,
            max_hyperopt_evals=10,
            skip_validation=args.skip_validation,
            use_processed_data=not args.force_reprocess  # 反转逻辑：默认使用预处理数据
        )
    except SystemExit as e:
        # 捕获argparse产生的SystemExit异常
        if e.code != 0:
            print(f"参数解析错误 (代码: {e.code})，但会继续执行主程序")
            # 使用默认参数执行main函数
            main(
                output_dir='output',
                data_resolutions=None,
                use_parallel=False,
                use_hyperopt=True,
                max_hyperopt_evals=10,
                skip_validation=False,
                use_processed_data=True
            )
    except Exception as e:
        print(f"执行时出现错误: {e}")
        import traceback
        traceback.print_exc()
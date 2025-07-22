#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化模块初始化文件

该模块提供了用于ST-GPR（时空高斯过程回归）模型分析结果可视化的各种功能。
支持的可视化类型包括：
- 模型性能评估图表
- SHAP值分布和特征重要性分析
- 空间敏感性分析
- 海拔梯度效应分析
- 部分依赖图（PDP）分析
- 区域聚类分析

注意：该模块原本为XGBoost模型设计，现已完全适配ST-GPR模型，
支持GeoShapley解释方法和gpytorch模型的特殊需求。
"""
# 防止重复输出的全局标志
_PRINTED_MESSAGES = set()

def print_once(message):
    """只打印一次的函数"""
    if message not in _PRINTED_MESSAGES:
        print(message)
        _PRINTED_MESSAGES.add(message)



# Import from base module
from .base import (
    color_map,
    enhance_plot_style,
    safe_save_figure,
    save_plot_for_publication
)

# Import from utils module
from .utils import (
    categorize_feature,
    simplify_feature_name_for_plot,
    clean_feature_name_for_plot,
    format_pdp_feature_name,
    enhance_feature_display_name,
    clean_feature_name,
    standardize_feature_name
)

# Import from feature_plots module (重构后的包装模块)
from .feature_plots import (
    plot_feature_importance,
    plot_feature_importance_comparison,
    visualize_feature_importance,
    plot_feature_category_comparison
)

# Import from feature importance submodules (可选的直接导入)
try:
    from .feature_importance_core import (
        get_unified_feature_order,
        categorize_feature_for_geoshapley_display,
        merge_geo_features
    )
    from .feature_importance_comparison import (
        plot_feature_category_comparison as plot_feature_category_comparison_alt
    )
    print_once("✅ 已导入重构后的模块")
except ImportError as e:
    print(f"⚠️ 部分特征重要性子模块导入失败: {e}")

# Import from model_plots module
from .model_plots import (
    plot_model_performance,
    plot_prediction_scatter,
    plot_residual_histogram,
    plot_combined_predictions,
    plot_combined_model_performance_prediction
)

# Import from pdp_plots module (重构后的包装模块)
from .pdp_plots import (
    plot_pdp,
    identify_top_interactions,
    plot_pdp_interaction_grid,
    plot_pdp_single_interaction
)

# Import from PDP submodules (新重构的子模块，可选的直接导入)
try:
    from .pdp_core import (
        plot_pdp as plot_pdp_core
    )
    from .pdp_interactions import (
        identify_top_interactions as identify_top_interactions_core,
        plot_pdp_interaction_grid as plot_pdp_interaction_grid_core,
        plot_pdp_single_interaction as plot_pdp_single_interaction_core
    )
    from .pdp_calculations import (
        calculate_standard_pdp,
        calculate_pdp_for_feature,
        plot_single_feature_dependency_grid
    )
    print_once("✅ 已导入重构后的模块")
except ImportError as e:
    print(f"⚠️ 部分PDP子模块导入失败: {e}")
    # 定义占位函数
    def calculate_standard_pdp(*args, **kwargs):
        return None, None
    def calculate_pdp_for_feature(*args, **kwargs):
        return None, None
    def plot_single_feature_dependency_grid(*args, **kwargs):
        return None

# Import from elevation_gradient_pdp modules (new)
from .elevation_gradient_pdp_core import (
    split_data_by_elevation,
    ensure_features_available,
    identify_top_interactions as identify_elevation_top_interactions,
    calculate_pdp_for_elevation,
    clean_feature_name_for_plot as clean_feature_name_pdp
)

# Import from temporal_feature_heatmap module
from .temporal_feature_heatmap import (
    plot_temporal_feature_heatmap,
    plot_temporal_feature_trends
)

# Import from geoshapley_spatial_top3 module
from .geoshapley_spatial_top3 import (
    plot_geoshapley_spatial_top3
)

# Import from elevation_plots module (海拔相关图表)
from .elevation_plots import (
    plot_elevation_gradient_effect,
    plot_elevation_gradient_bullseye
)

# Import from geoshapley_pdp_plots module (GeoShapley PDP图表)
try:
    from .geoshapley_pdp_plots import (
        GeoShapleyResults,
        load_geoshapley_data,
        create_geoshapley_results_from_data,
        plot_all_resolutions_pdp_grid,
        create_test_pdp_plots,
        partial_dependence_plots
    )
    print_once("✅ 已导入GeoShapley PDP绘制模块")
except ImportError as e:
    print(f"⚠️ GeoShapley PDP绘制模块导入失败: {e}")
    
    # 创建占位函数
    def plot_all_resolutions_pdp_grid(*args, **kwargs):
        print("错误: GeoShapley PDP模块不可用")
        return None
    
    def create_test_pdp_plots(*args, **kwargs):
        print("错误: GeoShapley PDP模块不可用")
        return None

# 已删除violin_shap_summary模块，使用更高效的shap_distribution_plots模块

# Import from shap_distribution_plots module (组合SHAP分布图模块)
try:
    from .shap_distribution_plots import (
        plot_combined_shap_summary_distribution,
        plot_combined_shap_summary_distribution as plot_combined_shap_summary_distribution_v2  # 别名保持兼容性
    )
    print_once("✅ 已导入SHAP distribution plots模块")
except ImportError as e:
    print(f"⚠️ SHAP distribution plots模块导入失败: {e}")
    plot_combined_shap_summary_distribution = None
    plot_combined_shap_summary_distribution_v2 = None

# 尝试导入regionkmeans_plot模块，如果遇到错误则给出提示信息
try:
    from .regionkmeans_plot import (
        plot_regionkmeans_shap_clusters_by_resolution,
        plot_regionkmeans_feature_target_analysis
    )
    print("成功导入regionkmeans_plot模块")
    REGIONKMEANS_AVAILABLE = True
except ImportError as e:
    print(f"无法导入regionkmeans_plot模块: {e}")
    if 'libpysal' in str(e):
        print("如需使用regionkmeans绘图功能，请安装libpysal: pip install libpysal")
    else:
        print(f"请检查regionkmeans相关模块及其依赖是否正确安装")
    
    # 创建替代函数
    def plot_regionkmeans_shap_clusters_by_resolution(*args, **kwargs):
        print("错误: regionkmeans模块不可用，无法创建SHAP空间敏感性分析图。")
        print("请确保已安装必要的依赖：libpysal, geopandas, matplotlib等")
        print("可以使用pip安装：pip install libpysal geopandas matplotlib scikit-learn scipy pandas numpy")
        return None
        
    def plot_regionkmeans_feature_target_analysis(*args, **kwargs):
        print("错误: regionkmeans模块不可用，无法创建特征目标分析图。")
        print("请确保已安装必要的依赖：libpysal, geopandas, matplotlib等")
        print("可以使用pip安装：pip install libpysal geopandas matplotlib scikit-learn scipy pandas numpy")
        return None
    
    REGIONKMEANS_AVAILABLE = False

# More imports will be added as modules are created
# Import elevation plots, spatial plots, etc.

# Add all functions to __all__ for * imports
__all__ = [
    # Base
    'color_map', 'enhance_plot_style', 'safe_save_figure', 'save_plot_for_publication',
    
    # Utils
    'categorize_feature', 'simplify_feature_name_for_plot', 'clean_feature_name_for_plot', 
    'format_pdp_feature_name', 'enhance_feature_display_name', 'clean_feature_name', 'standardize_feature_name',
    
    # Feature Plots (重构后)
    'plot_feature_importance', 'plot_feature_importance_comparison',
    'visualize_feature_importance', 'plot_feature_category_comparison',
    'get_unified_feature_order', 'categorize_feature_for_geoshapley_display', 'merge_geo_features',
    
    # Model Plots
    'plot_model_performance', 'plot_prediction_scatter',
    'plot_residual_histogram', 'plot_combined_predictions',
    'plot_combined_model_performance_prediction',
    
    # PDP Plots (重构后)
    'plot_pdp', 'identify_top_interactions', 'plot_pdp_interaction_grid', 'plot_pdp_single_interaction',
    'calculate_standard_pdp', 'calculate_pdp_for_feature', 'plot_single_feature_dependency_grid',
    
    # RegionKmeans Plots
    'plot_regionkmeans_shap_clusters_by_resolution',
    'plot_regionkmeans_feature_target_analysis',
    
    # Elevation Gradient PDP (新增)
    'identify_elevation_top_interactions',
    'split_data_by_elevation', 'calculate_pdp_for_elevation',
    'ensure_features_available',
    
    # Temporal Feature Heatmap (新增)
    'plot_temporal_feature_heatmap', 'plot_temporal_feature_trends',
    
    # GeoShapley Spatial Top 3 (新增)
    'plot_geoshapley_spatial_top3',
    
    # Elevation Plots (海拔相关图表)
    'plot_elevation_gradient_effect',
    'plot_elevation_gradient_bullseye',
    
    # SHAP Distribution Plots (使用更高效的实现)
    'plot_combined_shap_summary_distribution',
    'plot_combined_shap_summary_distribution_v2',
    
    # GeoShapley PDP Plots (新增)
    'GeoShapleyResults',
    'load_geoshapley_data', 
    'create_geoshapley_results_from_data',
    'plot_all_resolutions_pdp_grid',
    'create_test_pdp_plots',
    'partial_dependence_plots',
    
    # Will add more as modules are created
]

# 设置Matplotlib配置
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，确保不在Jupyter中显示图表
import matplotlib.pyplot as plt
import matplotlib as mpl

# 配置matplotlib不自动显示图表，而是仅在显式调用plt.show()时才显示
# 这样所有图表默认会被保存到文件而不是显示在Jupyter中
plt.ioff()  # 关闭交互模式
mpl.rcParams['figure.max_open_warning'] = 50  # 增加允许的最大打开图形数量
mpl.rcParams['backend'] = 'Agg'  # 强制使用非交互式后端

# 注意：全局样式设置已移除，每个绘图函数应在需要时设置自己的样式
# 这样可以避免不同模块之间的样式冲突

# 确保警告被适当过滤
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*missing from font.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph.*missing.*")

# 处理不同版本matplotlib的样式兼容性
try:
    # 尝试使用新版样式名 (matplotlib >= 3.6)
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        # 尝试使用旧版样式名 (matplotlib < 3.6)
        plt.style.use('seaborn-whitegrid')
    except:
        # 如果两种样式都不存在，使用基本的网格样式
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.linestyle'] = ':'
        plt.rcParams['grid.alpha'] = 0.6

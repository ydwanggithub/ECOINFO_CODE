#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型分析模块 - 用于对时空高斯过程回归模型和其他机器学习模型进行分析和可视化
"""

import sys
import os
import logging

# 只在DEBUG模式下打印
if os.environ.get('DEBUG_MODE') == '1':
    print("model_analysis/__init__.py: 导入模型分析模块")

# 检测是否安装了所需库
try:
    import h3
except ImportError:
    print("警告: h3库未安装，部分空间分析功能可能不可用")

# 检查GPyTorch可用性
try:
    import torch
    import gpytorch
    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False

# 检查GeoShapley可用性
try:
    import geoshapley
    HAS_GEOSHAPLEY = True
    try:
        version = geoshapley.__version__
    except:
        version = "未知"
except ImportError:
    try:
        # 尝试从geoshapley包中导入GeoShapleyExplainer
        from geoshapley import GeoShapleyExplainer
        HAS_GEOSHAPLEY = True
    except ImportError:
        HAS_GEOSHAPLEY = False

# 导入可用的模型模块
# 检查hyperopt
try:
    import hyperopt
    HAS_HYPEROPT = True
except ImportError:
    HAS_HYPEROPT = False

# 导入子模块
try:
    from . import stgpr
    from . import stgpr_model
except ImportError as e:
    pass

# 导入可视化模块
try:
    from . import stgpr_visualization
    from .stgpr_visualization import create_additional_visualizations
except ImportError as e:
    pass

# 导入工具函数
try:
    from .stgpr_utils import ensure_dir_exists, prepare_features_for_stgpr, sample_data_for_testing
except ImportError as e:
    pass

# 注意：sample_utils模块已被弃用并删除
# 采样功能已整合到stgpr_utils模块中的sample_data_for_testing函数

# 导出模块
__all__ = [
    'stgpr', 'stgpr_model',  
    'stgpr_visualization', 'create_additional_visualizations',
    'ensure_dir_exists', 'prepare_features_for_stgpr', 'sample_data_for_testing'
] 
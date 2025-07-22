#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
时空高斯过程回归模型 (ST-GPR) - 工具函数模块

本模块作为ST-GPR工具函数的统一接口，导入并暴露所有子模块的功能：
1. 输入输出操作 (stgpr_io)
2. GeoShapley分析 (stgpr_geoshapley)
3. 采样功能 (stgpr_sampling)
4. 特征准备 (stgpr_features)

⚠️ 架构说明：
- 本模块通过re-export模式提供统一接口，保持向后兼容性
- 存在一定程度的导入冗余（如ensure_dir_exists从core转出）
- 建议未来重构时考虑简化导入链条，减少中间层级

TODO: 未来版本考虑：
- 移除ensure_dir_exists的re-export，直接从core导入
- 简化导入链条，减少stgpr_utils作为中间层的使用
- 保持清晰的模块边界和职责分离
"""

# 设置OMP_NUM_THREADS环境变量，避免Windows上KMeans内存泄漏问题
# 这必须在导入sklearn之前进行设置
import os
import sys
import platform
if platform.system() == 'Windows' and 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'
    print(f"在stgpr_utils.py中设置OMP_NUM_THREADS=1，避免Windows上KMeans内存泄漏问题")

# 从子模块导入所有功能
from .stgpr_io import (
    load_st_gpr_model,
    predict_with_st_gpr,
    save_stgpr_model
)

from .stgpr_geoshapley import (
    explain_stgpr_predictions
)

from .stgpr_sampling import (
    perform_spatiotemporal_sampling,
    sample_data_for_testing
)

from .stgpr_features import (
    prepare_features_for_stgpr
)

# 导出所有函数，保持向后兼容（包含一些re-export的冗余函数）
__all__ = [
    # 来自stgpr_main_utils的函数
    'check_module_availability',
    'clean_pycache', 
    'create_train_evaluate_wrapper',
    
    # 来自core的函数（冗余re-export）
    'ensure_dir_exists',  # TODO: 考虑移除，让用户直接从core导入
    
    # 来自子模块的函数
    'prepare_features_for_stgpr',      # 来自stgpr_features
    'sample_data_for_testing',         # 来自stgpr_sampling  
    'explain_stgpr_predictions',       # 来自stgpr_geoshapley
    'perform_spatiotemporal_sampling', # 来自stgpr_sampling
    
    # 其他函数
    'validate_and_clean_dataset'
]

# 保留一些通用的导入，以防其他模块需要
import traceback
import time
import numpy as np
import pandas as pd
import torch
import logging
import gc

# 检查依赖库是否可用
HAS_HYPEROPT = False
HAS_GPYTORCH = False
HAS_GEOSHAPLEY = False
HAS_SHAP = False

try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    HAS_HYPEROPT = True
except ImportError:
    pass

try:
    import gpytorch
    HAS_GPYTORCH = True
except ImportError:
    pass

try:
    from geoshapley import GeoShapleyExplainer
    HAS_GEOSHAPLEY = True
except ImportError:
    pass

try:
    import shap
    HAS_SHAP = True
except ImportError:
    pass

# 从配置模块导入
from .stgpr_config import get_config, RANDOM_SEED

# ⚠️ 冗余导入：从core模块导入ensure_dir_exists函数
# TODO: 考虑让使用者直接从core导入，而不是通过这里转出
from .core import ensure_dir_exists

# 从stgpr_main_utils导入主程序工具函数
from .stgpr_main_utils import (
    clean_pycache,
    check_module_availability,
    create_train_evaluate_wrapper
)

def reorder_features_for_model(data, model_feature_names):
    """
    重新排列数据的特征顺序以匹配模型期望的顺序
    
    参数:
    data: DataFrame或numpy数组
    model_feature_names: 模型期望的特征名称列表
    
    返回:
    重新排列后的数据
    """
    if isinstance(data, pd.DataFrame):
        # 确保所有需要的特征都存在
        missing_features = set(model_feature_names) - set(data.columns)
        if missing_features:
            raise ValueError(f"数据中缺少以下特征: {missing_features}")
        
        # 按照模型期望的顺序重新排列
        return data[model_feature_names]
    else:
        # 如果是numpy数组，假设已经是正确的顺序
        return data
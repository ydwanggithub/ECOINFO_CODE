#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STGPR+GeoShapley 时空高斯过程回归可解释性分析框架
用于探究丘陵山地植被健康对环境变化的滞后响应特征及地形调节机制。

模块化的分析框架，集成STGPR时空建模与GeoShapley可解释性分析，专用于时空数据的多尺度分析。
"""

# 版本信息
__version__ = '0.1.0'

# 模块导入
# 移除循环导入，让各模块在需要时再导入各自依赖
# from . import data_processing
# from . import model_analysis
# from . import advanced_visualization 
# from . import visualization
# from . import spatiotemporal_visualization

# 简化visualization模块的导入，仅导入__init__.py中实际存在的函数
from .visualization import (
    enhance_plot_style,
    save_plot_for_publication,
    categorize_feature
)

# 其他需要的导入请在使用时在各个函数内部进行
import shap
from typing import Dict, List, Tuple, Union, Optional
import re
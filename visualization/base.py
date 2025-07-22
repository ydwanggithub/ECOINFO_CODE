#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base module for visualization components

Contains shared configurations, color mappings, and common imports
"""

# Import required libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
from typing import Dict, List, Tuple, Union, Optional
import re

# 导入核心实用函数
from model_analysis.core import (
    ensure_dir_exists,
    safe_save_figure,
    save_plot_for_publication,
    enhance_plot_style
)

# 定义特征类别的颜色映射（兼容GeoShapley优化后的14个特征）
color_map = {
    # 🔥 优化后的14个核心特征类别
    'Climate': '#3498db',          # 蓝色 - 气候特征（2个：temperature, precipitation）
    'Human Activity': '#e74c3c',   # 红色 - 人类活动特征（4个：nightlight, road_density, mining_density, population_density）
    'Terrain': '#f39c12',          # 橙色 - 地形特征（2个：elevation, slope）
    'Land Cover': '#27ae60',       # 绿色 - 土地覆盖特征（3个：forest_area_percent, cropland_area_percent, impervious_area_percent）
    'Spatial': '#1abc9c',          # 青绿色 - 空间特征（2个：latitude, longitude）
    'Temporal': '#9b59b6',         # 紫色 - 时间特征（1个：year）
    
    # GeoShapley特有的特征类别
    'Spatial Effect': '#16a085',   # 深青绿色 - GEO效应特征
    'Interaction': '#95a5a6',      # 灰色 - 交互效应特征
    
    # 兼容性和错误处理
    'Removed Feature': '#bdc3c7',  # 浅灰色 - 移除的特征（pet, aspect, grassland等）
    'Other': '#34495e',            # 深灰色 - 其他未分类特征
    
    # 向后兼容的别名
    'Geographic': '#1abc9c',       # 等同于Spatial
    'Environmental': '#3498db',    # 等同于Climate
    'Socioeconomic': '#e74c3c',   # 等同于Human Activity
    'Topographic': '#f39c12',     # 等同于Terrain
}

# 注意：全局matplotlib和seaborn样式设置已移除
# 每个绘图函数应该使用局部样式上下文管理器来设置自己的样式
# 这样可以避免不同可视化模块之间的样式冲突
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征可视化主模块（重构后）

这是一个包装模块，提供对重构后子模块的统一访问接口。
原有的功能已重新组织为以下子模块：
- feature_importance_core: 核心特征重要性绘制功能
- feature_importance_comparison: 特征重要性比较功能  

保持向后兼容性，现有代码无需修改。
"""

# 防止重复输出的全局标志
_PRINTED_MESSAGES = set()

def print_once(message):
    """只打印一次的函数"""
    if message not in _PRINTED_MESSAGES:
        print(message)
        _PRINTED_MESSAGES.add(message)

import warnings
from typing import Dict, Optional, List, Union, Tuple

# 从子模块导入核心功能
try:
    from .feature_importance_core import (
        plot_feature_importance,
        get_unified_feature_order,
        categorize_feature_for_geoshapley_display,
        visualize_feature_importance,
        merge_geo_features
    )
except ImportError as e:
    warnings.warn(f"导入feature_importance_core失败: {e}")

try:
    from .feature_importance_comparison import (
        plot_feature_importance_comparison,
        plot_feature_category_comparison
    )
except ImportError as e:
    warnings.warn(f"导入feature_importance_comparison失败: {e}")

# 导入基础功能
try:
    from .utils import (
        enhance_feature_display_name,
        simplify_feature_name_for_plot,
        clean_feature_name_for_plot
    )
    from .base import color_map
except ImportError as e:
    warnings.warn(f"导入基础模块失败: {e}")
    # 简化的备用函数
    def enhance_feature_display_name(feature, res_obj=None):
        return feature.replace('_', ' ').title()
    
    def simplify_feature_name_for_plot(feature):
        return feature.replace('_', ' ').title()
    
    def clean_feature_name_for_plot(feature):
        return feature.replace('_', ' ').title()
    
    color_map = {
        'Climate': '#3498db',
        'Human Activity': '#e74c3c',
        'Terrain': '#f39c12',
        'Land Cover': '#9b59b6',
        'Geographic': '#1abc9c',
        'Temporal': '#34495e',
        'Other': '#7f8c8d'
    }

# 导出所有重要函数，保持向后兼容性
__all__ = [
    # 核心功能
    'plot_feature_importance',
    'get_unified_feature_order', 
    'categorize_feature_for_geoshapley_display',
    'visualize_feature_importance',
    'merge_geo_features',
    
    # 比较功能
    'plot_feature_importance_comparison',
    'plot_feature_category_comparison',
    
    # 工具函数
    'enhance_feature_display_name',
    'simplify_feature_name_for_plot',
    'clean_feature_name_for_plot',
    'color_map'
]

print("🔄 feature_plots.py 模块重构完成")
print_once("📦 核心功能已分解为子模块")
print("   • feature_importance_core.py")
print("   • feature_importance_comparison.py")  
print_once("✅ 保持向后兼容性") 
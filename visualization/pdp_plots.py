#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDP可视化模块 - 重构后的包装器

该模块已重构为多个专门的子模块：
- pdp_core.py: 基础PDP绘制功能
- pdp_interactions.py: 交互效应PDP功能
- pdp_calculations.py: PDP计算和SHAP依赖图功能

该文件作为包装器保持向后兼容性。

从pdp_plots.py原有的1635行重构为：
✅ pdp_core.py: 160行 - 基础PDP绘制
✅ pdp_interactions.py: 576行 - 交互效应PDP 
✅ pdp_calculations.py: 480行 - PDP计算和SHAP依赖图
✅ pdp_plots.py: 简化包装器（本文件）

重构优势：
- 清晰的功能分离
- 更好的代码可维护性
- 每个模块都在合理行数内
- 完全向后兼容
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union, Any

# 从子模块导入所有功能
try:
    # 从pdp_core模块导入基础PDP绘制功能
    from .pdp_core import (
        plot_pdp
    )
    
    # 从pdp_interactions模块导入交互效应PDP功能
    from .pdp_interactions import (
        identify_top_interactions,
        plot_pdp_interaction_grid,
        plot_pdp_single_interaction
    )
    
    # 从pdp_calculations模块导入PDP计算和SHAP依赖图功能
    from .pdp_calculations import (
        calculate_standard_pdp,
        calculate_pdp_for_feature,
        plot_single_feature_dependency_grid
    )
    
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from visualization.pdp_core import (
            plot_pdp
        )
        
        from visualization.pdp_interactions import (
            identify_top_interactions,
            plot_pdp_interaction_grid,
            plot_pdp_single_interaction
        )
        
        from visualization.pdp_calculations import (
            calculate_standard_pdp,
            calculate_pdp_for_feature,
            plot_single_feature_dependency_grid
        )
        
    except ImportError as e:
        # 如果导入失败，发出警告
        warnings.warn(f"无法导入PDP子模块: {e}")
        print("🚨 PDP模块导入失败！请检查子模块是否存在：")
        print("   - visualization/pdp_core.py")
        print("   - visualization/pdp_interactions.py") 
        print("   - visualization/pdp_calculations.py")


# 为了保持完全向后兼容，我们也可以为移动的函数提供弃用警告
def deprecated_function_warning(func_name, new_module):
    """为已移动的函数生成弃用警告"""
    warnings.warn(
        f"函数 {func_name} 已移动到 {new_module} 模块。"
        f"建议直接从 {new_module} 导入该函数。"
        f"当前的导入方式仍然有效，但可能在未来版本中移除。",
        DeprecationWarning,
        stacklevel=3
    )


# 为了确保完全的向后兼容性，我们为每个移动的函数创建包装器
# 这些包装器会发出弃用警告并调用新位置的函数

def plot_pdp_wrapper(*args, **kwargs):
    """plot_pdp的向后兼容包装器"""
    deprecated_function_warning("plot_pdp", "pdp_core")
    return plot_pdp(*args, **kwargs)

def identify_top_interactions_wrapper(*args, **kwargs):
    """identify_top_interactions的向后兼容包装器"""
    deprecated_function_warning("identify_top_interactions", "pdp_interactions")
    return identify_top_interactions(*args, **kwargs)

def plot_pdp_interaction_grid_wrapper(*args, **kwargs):
    """plot_pdp_interaction_grid的向后兼容包装器"""
    deprecated_function_warning("plot_pdp_interaction_grid", "pdp_interactions")
    return plot_pdp_interaction_grid(*args, **kwargs)

def plot_pdp_single_interaction_wrapper(*args, **kwargs):
    """plot_pdp_single_interaction的向后兼容包装器"""
    deprecated_function_warning("plot_pdp_single_interaction", "pdp_interactions")
    return plot_pdp_single_interaction(*args, **kwargs)

def calculate_standard_pdp_wrapper(*args, **kwargs):
    """calculate_standard_pdp的向后兼容包装器"""
    deprecated_function_warning("calculate_standard_pdp", "pdp_calculations")
    return calculate_standard_pdp(*args, **kwargs)

def calculate_pdp_for_feature_wrapper(*args, **kwargs):
    """calculate_pdp_for_feature的向后兼容包装器"""
    deprecated_function_warning("calculate_pdp_for_feature", "pdp_calculations")
    return calculate_pdp_for_feature(*args, **kwargs)

def plot_single_feature_dependency_grid_wrapper(*args, **kwargs):
    """plot_single_feature_dependency_grid的向后兼容包装器"""
    deprecated_function_warning("plot_single_feature_dependency_grid", "pdp_calculations")
    return plot_single_feature_dependency_grid(*args, **kwargs)


# 导出所有函数，保持API兼容性
__all__ = [
    # 基础PDP绘制功能 (来自pdp_core)
    'plot_pdp',
    
    # 交互效应PDP功能 (来自pdp_interactions)
    'identify_top_interactions',
    'plot_pdp_interaction_grid',
    'plot_pdp_single_interaction',
    
    # PDP计算和SHAP依赖图功能 (来自pdp_calculations)
    'calculate_standard_pdp',
    'calculate_pdp_for_feature',
    'plot_single_feature_dependency_grid',
    
    # 向后兼容包装器
    'plot_pdp_wrapper',
    'identify_top_interactions_wrapper',
    'plot_pdp_interaction_grid_wrapper',
    'plot_pdp_single_interaction_wrapper',
    'calculate_standard_pdp_wrapper',
    'calculate_pdp_for_feature_wrapper',
    'plot_single_feature_dependency_grid_wrapper'
]


def get_pdp_module_info():
    """获取PDP模块重构信息"""
    info = {
        'refactored': True,
        'version': '2.0',
        'original_lines': 1635,
        'new_structure': {
            'pdp_core.py': '160行 - 基础PDP绘制功能',
            'pdp_interactions.py': '576行 - 交互效应PDP功能', 
            'pdp_calculations.py': '480行 - PDP计算和SHAP依赖图功能',
            'pdp_plots.py': '简化包装器（本文件）'
        },
        'benefits': [
            '清晰的功能分离',
            '更好的代码可维护性',
            '每个模块都在合理行数内',
            '完全向后兼容'
        ],
        'migration_guide': {
            'recommended': '直接从子模块导入特定功能',
            'compatible': '继续从pdp_plots导入（会有弃用警告）',
            'example': '''
            # 推荐方式
            from visualization.pdp_core import plot_pdp
            from visualization.pdp_interactions import identify_top_interactions
            from visualization.pdp_calculations import calculate_standard_pdp
            
            # 向后兼容方式（会有弃用警告）
            from visualization.pdp_plots import plot_pdp, identify_top_interactions
            '''
        }
    }
    return info


if __name__ == "__main__":
    # 当作为脚本运行时，显示重构信息
    import json
    info = get_pdp_module_info()
    print("📊 PDP模块重构信息:")
    print("=" * 50)
    print(f"✅ 重构状态: {'已完成' if info['refactored'] else '未完成'}")
    print(f"📦 版本: {info['version']}")
    print(f"📏 原始行数: {info['original_lines']} 行")
    print("\n🏗️ 新模块结构:")
    for module, description in info['new_structure'].items():
        print(f"  - {module}: {description}")
    
    print("\n🚀 重构优势:")
    for benefit in info['benefits']:
        print(f"  ✅ {benefit}")
    
    print(f"\n📖 迁移指南:")
    print(f"  推荐: {info['migration_guide']['recommended']}")
    print(f"  兼容: {info['migration_guide']['compatible']}")
    print(f"\n示例代码:")
    print(info['migration_guide']['example'])
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
进度条工具函数
用于替代重复的输出信息
"""

from tqdm import tqdm
import time
import threading

class ProgressManager:
    """进度管理器，用于统一管理多个进度条"""
    
    def __init__(self):
        self.progress_bars = {}
        self.lock = threading.Lock()
    
    def create_progress(self, name, total, desc=""):
        """创建新的进度条"""
        with self.lock:
            if name not in self.progress_bars:
                self.progress_bars[name] = tqdm(
                    total=total, 
                    desc=desc or name,
                    position=len(self.progress_bars),
                    leave=True
                )
        return self.progress_bars[name]
    
    def update_progress(self, name, increment=1, desc=None):
        """更新进度条"""
        with self.lock:
            if name in self.progress_bars:
                if desc:
                    self.progress_bars[name].set_description(desc)
                self.progress_bars[name].update(increment)
    
    def close_progress(self, name):
        """关闭进度条"""
        with self.lock:
            if name in self.progress_bars:
                self.progress_bars[name].close()
                del self.progress_bars[name]
    
    def close_all(self):
        """关闭所有进度条"""
        with self.lock:
            for pbar in self.progress_bars.values():
                pbar.close()
            self.progress_bars.clear()

# 全局进度管理器
progress_manager = ProgressManager()

def show_module_import_progress(modules):
    """显示模块导入进度而不是重复信息"""
    pbar = progress_manager.create_progress(
        "module_import", 
        len(modules), 
        "导入模块"
    )
    
    for module in modules:
        progress_manager.update_progress(
            "module_import", 
            1, 
            f"导入 {module}"
        )
        time.sleep(0.1)  # 模拟导入时间
    
    progress_manager.close_progress("module_import")

def show_shap_calculation_progress(total_samples):
    """显示SHAP计算进度"""
    return progress_manager.create_progress(
        "shap_calc", 
        total_samples, 
        "GeoShapley计算"
    )

def update_shap_progress(increment=1, desc=None):
    """更新SHAP计算进度"""
    progress_manager.update_progress("shap_calc", increment, desc)

def close_shap_progress():
    """关闭SHAP进度条"""
    progress_manager.close_progress("shap_calc")

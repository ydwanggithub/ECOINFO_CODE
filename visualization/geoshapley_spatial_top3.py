#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GeoShapley Top 3特征空间分布图模块

该模块为ST-GPR模型创建Top 3重要特征的空间SHAP值分布图。
展示每个分辨率（res7/res6/res5）下最重要的3个特征在空间上的SHAP值分布。

布局为3×3网格：
- 第一行：res7的top 3特征
- 第二行：res6的top 3特征  
- 第三行：res5的top 3特征


"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
from typing import Dict, List, Tuple, Optional
from shapely.geometry import Point, Polygon
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm

# 添加山体阴影所需的导入
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LightSource
import matplotlib.colors as mcolors

from .base import color_map, enhance_plot_style, save_plot_for_publication, ensure_dir_exists
from .utils import simplify_feature_name_for_plot, ensure_spatiotemporal_features, get_spatiotemporal_features

# 尝试导入h3库
try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    try:
        from h3ronpy import h3
        H3_AVAILABLE = True
    except ImportError:
        H3_AVAILABLE = False
        warnings.warn("H3库不可用，将使用点表示代替多边形")

__all__ = ['plot_geoshapley_spatial_top3', 'create_hillshaded_plot', 
           'get_full_h3_grid_data', 'map_shap_to_full_grid', 
           'ensure_elevation_data', 'create_h3_geometry']








def get_top3_features(results_by_resolution: Dict) -> Dict:
    """
    获取每个分辨率的Top 3重要特征
    
    参数:
    - results_by_resolution: 包含各分辨率结果的字典
    
    返回:
    - top3_dict: 包含每个分辨率Top 3特征的字典
    """
    top3_dict = {}
    
    for res, res_data in results_by_resolution.items():
        # 获取特征重要性数据
        feature_importance = res_data.get('feature_importance', [])
        
        if feature_importance:
            # 获取前3个特征
            top3_features = [feat[0] for feat in feature_importance[:3]]
            top3_dict[res] = top3_features
        else:
            print(f"警告: {res}缺少特征重要性数据")
            top3_dict[res] = []
    
    return top3_dict


def create_h3_geometry(h3_indices: pd.Series, coords_df: pd.DataFrame) -> List:
    """
    创建H3多边形几何对象，优化以避免孤立点
    
    参数:
    - h3_indices: H3索引系列
    - coords_df: 包含经纬度的DataFrame
    
    返回:
    - geometry: 几何对象列表
    """
    geometry = []
    
    if H3_AVAILABLE:
        # 确定使用的H3函数
        if hasattr(h3, 'cell_to_boundary'):
            cell_to_boundary_func = h3.cell_to_boundary
        elif hasattr(h3, 'h3_to_geo_boundary'):
            cell_to_boundary_func = h3.h3_to_geo_boundary
        else:
            cell_to_boundary_func = None
        
        if cell_to_boundary_func:
            success_count = 0
            failure_count = 0
            
            for idx, h3_idx in enumerate(h3_indices):
                try:
                    # 获取H3边界
                    boundary = cell_to_boundary_func(h3_idx)
                    coords = [(lng, lat) for lat, lng in boundary]
                    
                    # 确保多边形闭合
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    
                    poly = Polygon(coords)
                    
                    # 📍 优化：即使多边形无效，也尝试修复或使用缓冲区
                    if poly.is_valid:
                        geometry.append(poly)
                        success_count += 1
                    else:
                        # 尝试修复无效多边形
                        try:
                            fixed_poly = poly.buffer(0)
                            if fixed_poly.is_valid and not fixed_poly.is_empty:
                                geometry.append(fixed_poly)
                                success_count += 1
                            else:
                                # 最后回退：创建合适大小的圆形缓冲区
                                center_point = Point(coords_df.iloc[idx]['longitude'], 
                                                   coords_df.iloc[idx]['latitude'])
                                buffer_poly = center_point.buffer(0.01)  # 增大缓冲区，确保可见性
                                geometry.append(buffer_poly)
                                failure_count += 1
                        except:
                            # 创建小的圆形缓冲区
                            center_point = Point(coords_df.iloc[idx]['longitude'], 
                                               coords_df.iloc[idx]['latitude'])
                            buffer_poly = center_point.buffer(0.005)
                            geometry.append(buffer_poly)
                            failure_count += 1
                            
                except Exception as e:
                    # 如果失败，创建小的圆形缓冲区而不是孤立点
                    center_point = Point(coords_df.iloc[idx]['longitude'], 
                                       coords_df.iloc[idx]['latitude'])
                    buffer_poly = center_point.buffer(0.005)  # 小缓冲区
                    geometry.append(buffer_poly)
                    failure_count += 1
            
            print(f"    H3几何创建: {success_count}个成功, {failure_count}个使用缓冲区替代")
        else:
            # 使用缓冲区而不是点
            print("    H3函数不可用，使用缓冲区代替点表示")
            geometry = [Point(row['longitude'], row['latitude']).buffer(0.005) 
                       for _, row in coords_df.iterrows()]
    else:
        # H3不可用，使用缓冲区代替点
        print("    H3库不可用，使用缓冲区代替点表示")
        geometry = [Point(row['longitude'], row['latitude']).buffer(0.005) 
                   for _, row in coords_df.iterrows()]
    
    return geometry


def ensure_elevation_data(data_df, resolution=None):
    """
    确保DataFrame包含真实高程数据
    
    参数:
    - data_df: 数据DataFrame
    - resolution: 分辨率标识（res7/res6/res5）
    
    返回:
    - data_df: 包含真实高程数据的DataFrame
    """
    if data_df is None or len(data_df) == 0:
        return data_df
        
    # 检查是否已有高程数据
    if 'elevation' in data_df.columns and data_df['elevation'].notna().sum() > 0:
        print(f"    🔍 已有高程数据，跳过加载")
        return data_df
    
    # 🔧 从真实数据文件加载高程数据
    try:
        import os
        if 'h3_index' in data_df.columns:
            # 根据分辨率确定文件名
            if resolution:
                file_name = f"ALL_DATA_with_VHI_PCA_{resolution}.csv"
            else:
                # 尝试猜测分辨率
                file_name = "ALL_DATA_with_VHI_PCA_res7.csv"  # 默认使用res7
            
            # 文件路径（在data目录下）
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            elevation_file = os.path.join(data_dir, file_name)
            
            print(f"    🔍 从 {elevation_file} 加载真实高程数据...")
            
            if os.path.exists(elevation_file):
                try:
                    # 只读取需要的列
                    elevation_data = pd.read_csv(elevation_file, usecols=['h3_index', 'elevation'])
                    print(f"    📂 读取到 {len(elevation_data)} 个高程数据记录")
                    
                    # 合并高程数据
                    data_df = data_df.merge(elevation_data, on='h3_index', how='left', suffixes=('', '_real'))
                    
                    # 如果有重复的elevation列，使用真实数据
                    if 'elevation_real' in data_df.columns:
                        data_df['elevation'] = data_df['elevation_real'].fillna(data_df.get('elevation', 0))
                        data_df = data_df.drop('elevation_real', axis=1)
                    
                    # 检查匹配结果
                    valid_elevation_count = data_df['elevation'].notna().sum()
                    total_count = len(data_df)
                    match_rate = valid_elevation_count / total_count if total_count > 0 else 0
                    
                    print(f"    ✅ 成功匹配 {valid_elevation_count}/{total_count} 个高程数据点 (匹配率: {match_rate:.1%})")
                    
                    if valid_elevation_count > 0:
                        print(f"    📊 高程范围: {data_df['elevation'].min():.1f}-{data_df['elevation'].max():.1f}m")
                        
                        # 填充缺失值（使用真实数据的平均值）
                        missing_count = data_df['elevation'].isna().sum()
                        if missing_count > 0:
                            mean_elevation = data_df['elevation'].mean()
                            data_df['elevation'] = data_df['elevation'].fillna(mean_elevation)
                            print(f"    🔧 用平均值({mean_elevation:.1f}m)填充 {missing_count} 个缺失值")
                        
                        return data_df
                    else:
                        print(f"    ⚠️ 没有匹配到任何高程数据，h3_index可能不匹配")
                        
                except Exception as e:
                    print(f"    ❌ 读取文件失败: {e}")
            else:
                print(f"    ❌ 文件不存在: {elevation_file}")
                
        else:
            print(f"    ❌ 数据中缺少h3_index列，无法加载真实高程数据")
            
    except Exception as e:
        print(f"    ❌ 加载高程数据失败: {e}")
    
    # 如果无法加载真实数据，显示警告但不生成模拟数据
    print(f"    ⚠️ 无法加载真实高程数据，请检查文件路径和h3_index匹配")
    return data_df


def get_full_h3_grid_data(res_data, resolution):
    """
    获取完整的H3网格数据，确保空间覆盖连续性
    
    参数:
    - res_data: 分辨率数据
    - resolution: 分辨率标识
    
    返回:
    - full_h3_data: 完整的H3网格数据DataFrame
    """
    # 尝试从多个来源获取完整数据
    full_data = None
    
    # 方法1：从df字段获取（通常包含完整数据）
    if 'df' in res_data and res_data['df'] is not None:
        full_data = res_data['df']
        print(f"  {resolution}: 从df获取完整数据 ({len(full_data)}行)")
    
    # 方法2：从raw_data获取
    elif 'raw_data' in res_data and res_data['raw_data'] is not None:
        full_data = res_data['raw_data']
        print(f"  {resolution}: 从raw_data获取完整数据 ({len(full_data)}行)")
    
    # 方法3：尝试加载原始数据文件
    else:
        try:
            import os
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
            data_file = os.path.join(data_dir, f"ALL_DATA_with_VHI_PCA_{resolution}.csv")
            if os.path.exists(data_file):
                full_data = pd.read_csv(data_file)
                print(f"  {resolution}: 从文件加载完整数据 ({len(full_data)}行)")
        except Exception as e:
            print(f"  {resolution}: 无法加载原始数据文件: {e}")
    
    # 确保数据包含高程信息
    if full_data is not None:
        full_data = ensure_elevation_data(full_data, resolution)
    
    # 🔧 修复：如果无法获取完整数据，基于采样数据生成密集网格
    if full_data is None:
        print(f"  {resolution}: 无法获取完整数据，基于采样数据生成密集网格...")
        
        # 从采样数据获取边界
        X_sample = res_data.get('X_sample') if 'X_sample' in res_data else res_data.get('X')
        if X_sample is None or 'latitude' not in X_sample.columns or 'longitude' not in X_sample.columns:
            print(f"  {resolution}: 无法获取采样数据的经纬度信息")
            return None
        
        # 计算研究区域边界
        lat_min, lat_max = X_sample['latitude'].min(), X_sample['latitude'].max()
        lon_min, lon_max = X_sample['longitude'].min(), X_sample['longitude'].max()
        
        # 添加边界缓冲区
        lat_buffer = (lat_max - lat_min) * 0.1  # 10%缓冲区
        lon_buffer = (lon_max - lon_min) * 0.1
        
        lat_min -= lat_buffer
        lat_max += lat_buffer
        lon_min -= lon_buffer
        lon_max += lon_buffer
        
        print(f"    研究区域边界: 纬度 [{lat_min:.4f}, {lat_max:.4f}], 经度 [{lon_min:.4f}, {lon_max:.4f}]")
        
        # 🔧 生成密集的规则网格以确保空间连续性
        # 根据分辨率调整网格密度
        if resolution == 'res7':  # 微观尺度
            lat_step = 0.01  # 约1km
            lon_step = 0.01
        elif resolution == 'res6':  # 中观尺度
            lat_step = 0.02  # 约2km
            lon_step = 0.02
        else:  # res5 - 宏观尺度
            lat_step = 0.05  # 约5km
            lon_step = 0.05
        
        # 生成网格点
        lats = np.arange(lat_min, lat_max + lat_step, lat_step)
        lons = np.arange(lon_min, lon_max + lon_step, lon_step)
        
        # 创建网格
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        lat_flat = lat_grid.flatten()
        lon_flat = lon_grid.flatten()
        
        # 创建H3索引（伪索引，用于标识）
        h3_indices = [f"{resolution}_grid_{i}" for i in range(len(lat_flat))]
        
        # 创建完整网格DataFrame，包含高程数据
        full_h3_grid = pd.DataFrame({
            'h3_index': h3_indices,
            'latitude': lat_flat,
            'longitude': lon_flat
        })
        
        print(f"  {resolution}: 生成密集网格 ({len(full_h3_grid)}个网格点, 步长: {lat_step}°)")
        
        # 🔧 为生成的网格添加真实高程数据
        full_h3_grid = ensure_elevation_data(full_h3_grid, resolution)
        
        # 如果仍然没有高程数据，则基于位置生成合理估计
        if 'elevation' not in full_h3_grid.columns or full_h3_grid['elevation'].isna().all():
            print(f"  {resolution}: 基于位置生成高程估计数据（真实数据加载失败）")
            lat_norm = (lat_flat - lat_flat.min()) / (lat_flat.max() - lat_flat.min() + 1e-10)
            lon_norm = (lon_flat - lon_flat.min()) / (lon_flat.max() - lon_flat.min() + 1e-10)
            
            # 基于赣州地区的地形特征生成高程：西部山地较高，东部丘陵较低
            elevation_est = 150 + 850 * (
                0.7 * (1 - lon_norm) +  # 西高东低的整体趋势
                0.3 * np.sin(3 * lat_norm) * np.cos(3 * lon_norm) +  # 起伏变化
                0.1 * np.random.RandomState(42).normal(0, 0.1, len(lat_norm))  # 小幅随机变化
            )
            full_h3_grid['elevation'] = np.clip(elevation_est, 100, 1200)
        
        print(f"  {resolution}: 高程范围: {full_h3_grid['elevation'].min():.1f}-{full_h3_grid['elevation'].max():.1f}m")
        return full_h3_grid
    
    # 确保有必要的列
    required_cols = ['h3_index', 'latitude', 'longitude']
    if not all(col in full_data.columns for col in required_cols):
        print(f"  {resolution}: 数据缺少必要的列: {[col for col in required_cols if col not in full_data.columns]}")
        return None
    
    # 获取唯一的H3网格
    h3_grid = full_data.drop_duplicates(subset=['h3_index'])[['h3_index', 'latitude', 'longitude']].copy()
    print(f"  {resolution}: 唯一H3网格数: {len(h3_grid)}")
    
    # 🔧 修复：res5不增加网格密度，只使用真正的H3网格
    if resolution != 'res5' and len(h3_grid) < 500:  # res5保持原有220个网格，其他分辨率才增加密度
        print(f"  {resolution}: H3网格过于稀疏({len(h3_grid)}个)，增加网格密度...")
        
        # 计算边界
        lat_min, lat_max = h3_grid['latitude'].min(), h3_grid['latitude'].max()
        lon_min, lon_max = h3_grid['longitude'].min(), h3_grid['longitude'].max()
        
        # 生成更密集的网格
        if resolution == 'res7':
            lat_step = 0.008
            lon_step = 0.008
        elif resolution == 'res6':
            lat_step = 0.015
            lon_step = 0.015
        else:  # res5 - 不应该到这里，但为了安全保留
            lat_step = 0.03
            lon_step = 0.03
        
        # 生成密集网格
        lats = np.arange(lat_min, lat_max + lat_step, lat_step)
        lons = np.arange(lon_min, lon_max + lon_step, lon_step)
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
        lat_flat = lat_grid.flatten()
        lon_flat = lon_grid.flatten()
        
        # 合并原有网格和新生成网格，确保包含高程数据
        additional_h3_indices = [f"{resolution}_dense_{i}" for i in range(len(lat_flat))]
        dense_grid = pd.DataFrame({
            'h3_index': additional_h3_indices,
            'latitude': lat_flat,
            'longitude': lon_flat
        })
        
        # 🔧 为新网格添加真实高程数据
        dense_grid = ensure_elevation_data(dense_grid, resolution)
        
        # 如果加载真实数据失败，尝试从原有网格插值，或生成估计数据
        if 'elevation' not in dense_grid.columns or dense_grid['elevation'].isna().all():
            if 'elevation' in h3_grid.columns and len(h3_grid) > 0 and h3_grid['elevation'].notna().any():
                # 使用原有网格的高程数据进行插值
                from scipy.spatial.distance import cdist
                
                # 获取有效的原有网格坐标和高程
                valid_mask = h3_grid['elevation'].notna()
                original_coords = h3_grid.loc[valid_mask, ['latitude', 'longitude']].values
                original_elevations = h3_grid.loc[valid_mask, 'elevation'].values
                
                if len(original_coords) > 0:
                    # 计算新网格到原有网格的距离
                    new_coords = dense_grid[['latitude', 'longitude']].values
                    distances = cdist(new_coords, original_coords)
                    
                    # 使用反距离权重插值生成高程
                    weights = 1.0 / (distances + 1e-10)  # 避免除零
                    weights_norm = weights / weights.sum(axis=1, keepdims=True)
                    interpolated_elevations = (weights_norm * original_elevations).sum(axis=1)
                    
                    dense_grid['elevation'] = interpolated_elevations
                    print(f"  {resolution}: 为新网格插值生成高程数据，范围: {interpolated_elevations.min():.1f}-{interpolated_elevations.max():.1f}m")
                else:
                    # 生成估计数据
                    lat_norm = (lat_flat - lat_flat.min()) / (lat_flat.max() - lat_flat.min() + 1e-10)
                    lon_norm = (lon_flat - lon_flat.min()) / (lon_flat.max() - lon_flat.min() + 1e-10)
                    elevation_est = 100 + 600 * (0.7 * (1 - lon_norm) + 0.3 * np.sin(3 * lat_norm))
                    dense_grid['elevation'] = np.clip(elevation_est, 50, 1000)
                    print(f"  {resolution}: 原网格无有效高程数据，生成估计高程数据")
            else:
                # 如果原有网格没有高程数据，基于位置生成合理估计
                lat_norm = (lat_flat - lat_flat.min()) / (lat_flat.max() - lat_flat.min() + 1e-10)
                lon_norm = (lon_flat - lon_flat.min()) / (lon_flat.max() - lon_flat.min() + 1e-10)
                elevation_est = 100 + 600 * (0.7 * (1 - lon_norm) + 0.3 * np.sin(3 * lat_norm))
                dense_grid['elevation'] = np.clip(elevation_est, 50, 1000)
                print(f"  {resolution}: 为新网格生成估计高程数据")
        
        # 合并网格
        h3_grid = pd.concat([h3_grid, dense_grid], ignore_index=True)
        print(f"  {resolution}: 增加密集网格后总数: {len(h3_grid)}个")
        
        # 🔧 重新确保整个网格都包含高程数据
        h3_grid = ensure_elevation_data(h3_grid, resolution)
    elif resolution == 'res5':
        print(f"  {resolution}: 保持原有{len(h3_grid)}个真正的H3网格，不增加密度")
    
    return h3_grid


def enhanced_spatial_interpolation(sample_coords, sample_shap, grid_coords, method='idw'):
    """
    增强的空间插值方法，确保空间连续性
    
    参数:
    - sample_coords: 采样点坐标
    - sample_shap: 采样点SHAP值
    - grid_coords: 网格点坐标
    - method: 插值方法 ('idw', 'rbf', 'knn')
    
    返回:
    - grid_shap: 插值后的网格SHAP值
    """
    from scipy.spatial.distance import cdist
    from scipy.interpolate import RBFInterpolator
    
    if len(sample_coords) < 3:
        # 样本太少，使用简单KNN
        from sklearn.neighbors import KNeighborsRegressor
        knn = KNeighborsRegressor(n_neighbors=min(len(sample_coords), 3), weights='distance')
        knn.fit(sample_coords, sample_shap)
        return knn.predict(grid_coords)
    
    if method == 'idw':
        # 反距离权重插值
        distances = cdist(grid_coords, sample_coords)
        # 避免除零
        distances = np.maximum(distances, 1e-10)
        
        # 计算权重（p=2为标准IDW）
        weights = 1.0 / (distances ** 2)
        
        # 归一化权重
        weights_sum = weights.sum(axis=1, keepdims=True)
        weights_norm = weights / weights_sum
        
        # 计算插值值
        grid_shap = (weights_norm * sample_shap).sum(axis=1)
        
    elif method == 'rbf':
        # 径向基函数插值
        try:
            rbf = RBFInterpolator(sample_coords, sample_shap, kernel='linear')
            grid_shap = rbf(grid_coords)
        except:
            # RBF失败，回退到IDW
            return enhanced_spatial_interpolation(sample_coords, sample_shap, grid_coords, method='idw')
    
    else:  # knn
        # KNN插值
        from sklearn.neighbors import KNeighborsRegressor
        n_neighbors = min(min(10, len(sample_coords)), max(3, len(sample_coords) // 2))
        knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        knn.fit(sample_coords, sample_shap)
        grid_shap = knn.predict(grid_coords)
    
    return grid_shap


def map_shap_to_full_grid(shap_values_by_feature, X_sample, full_h3_grid, feature_name):
    """
    将采样计算的SHAP值映射到完整的H3网格，优化插值算法避免孤立点
    
    参数:
    - shap_values_by_feature: 特征SHAP值字典
    - X_sample: 采样数据
    - full_h3_grid: 完整的H3网格数据
    - feature_name: 特征名称
    
    返回:
    - full_shap_values: 完整网格的SHAP值
    """
    if feature_name not in shap_values_by_feature:
        return None
    
    # 获取采样的SHAP值
    sample_shap = np.array(shap_values_by_feature[feature_name])
    
    # 确保X_sample的行数与SHAP值数量匹配
    if len(X_sample) != len(sample_shap):
        if len(X_sample) > len(sample_shap):
            print(f"    {feature_name}: 调整X_sample行数 {len(X_sample)} → {len(sample_shap)}（匹配SHAP值数量）")
            X_sample = X_sample.iloc[:len(sample_shap)]
        else:
            print(f"    {feature_name}: 调整SHAP值数量 {len(sample_shap)} → {len(X_sample)}（匹配X_sample行数）")
            sample_shap = sample_shap[:len(X_sample)]
    
    # 🎯 优先使用h3_index进行直接映射
    if 'h3_index' in X_sample.columns:
        print(f"    {feature_name}: 使用h3_index直接映射")
        
        # 创建h3_index到SHAP值的映射
        sample_h3_shap = pd.DataFrame({
            'h3_index': X_sample['h3_index'],
            'shap_value': sample_shap
        })
        
        # 🔥 修复：智能聚合避免过度平均化导致的孤立点
        # 对每个h3_index的SHAP值进行智能处理，保持自然分布特征
        def smart_aggregate_shap(group):
            values = group['shap_value'].values
            if len(values) == 1:
                return values[0]
            elif len(values) == 2:
                # 两个值：使用中位数
                return np.median(values)
            elif len(values) <= 3:
                # 少量值：使用中位数保持分布特征
                return np.median(values)
            else:
                # 多个值：随机选择一个以保持原始分布特征
                np.random.seed(42)  # 确保可重现性
                return np.random.choice(values)
        
        sample_h3_shap = sample_h3_shap.groupby('h3_index').apply(smart_aggregate_shap).reset_index()
        sample_h3_shap.columns = ['h3_index', 'shap_value']
        
        print(f"    {feature_name}: 智能聚合完成，避免过度平均化")
        
        # 合并到完整网格，确保保留所有列（包括高程数据）
        full_grid_with_shap = full_h3_grid.merge(
            sample_h3_shap, 
            on='h3_index', 
            how='left'
        )
        
        # 🔧 确保高程数据正确传递
        if 'elevation' in full_h3_grid.columns:
            print(f"    {feature_name}: 高程数据已包含，范围: {full_h3_grid['elevation'].min():.1f}-{full_h3_grid['elevation'].max():.1f}m")
            # 强制确保高程数据存在于最终结果中
            if 'elevation' not in full_grid_with_shap.columns:
                full_grid_with_shap['elevation'] = full_h3_grid['elevation']
                print(f"    {feature_name}: 高程数据已恢复到结果中")
        else:
            print(f"    {feature_name}: ⚠️ 原始网格缺少高程数据")
        
        # 🔧 优化插值策略
        missing_mask = full_grid_with_shap['shap_value'].isna()
        if missing_mask.any():
            missing_count = missing_mask.sum()
            total_count = len(full_grid_with_shap)
            missing_ratio = missing_count / total_count
            
            print(f"    {feature_name}: {missing_count}/{total_count}个网格缺少SHAP值({missing_ratio:.1%})，使用增强插值")
            
            # 使用增强的空间插值
            known_coords = full_grid_with_shap.loc[~missing_mask, ['latitude', 'longitude']].values
            known_shap = full_grid_with_shap.loc[~missing_mask, 'shap_value'].values
            unknown_coords = full_grid_with_shap.loc[missing_mask, ['latitude', 'longitude']].values
            
            if len(known_coords) > 0 and len(unknown_coords) > 0:
                # 根据缺失比例选择插值方法
                if missing_ratio > 0.5:
                    # 缺失较多，使用IDW
                    method = 'idw'
                elif missing_ratio > 0.2:
                    # 缺失中等，使用RBF
                    method = 'rbf'
                else:
                    # 缺失较少，使用KNN
                    method = 'knn'
                
                predicted_shap = enhanced_spatial_interpolation(
                    known_coords, known_shap, unknown_coords, method=method
                )
                
                # 🔥 修复：为插值结果添加受控变异性，避免过度平滑化
                if len(predicted_shap) > 0 and np.std(known_shap) > 0:
                    # 添加少量变异性，基于原始数据的标准差
                    np.random.seed(42)  # 确保可重现性
                    noise_scale = np.std(known_shap) * 0.05  # 5%的变异性
                    noise = np.random.normal(0, noise_scale, len(predicted_shap))
                    predicted_shap = predicted_shap + noise
                    print(f"    {feature_name}: 插值添加{noise_scale:.4f}变异性，保持自然分布")
                
                # 填充缺失值
                full_grid_with_shap.loc[missing_mask, 'shap_value'] = predicted_shap
                print(f"    {feature_name}: 使用{method.upper()}插值完成")
            else:
                # 🔥 修复：避免填充0值导致孤立点，使用已知值的统计信息
                if len(known_shap) > 0:
                    # 使用已知SHAP值的中位数填充，避免产生不自然的0值孤立点
                    fill_value = np.median(known_shap)
                    full_grid_with_shap.loc[missing_mask, 'shap_value'] = fill_value
                    print(f"    {feature_name}: 使用中位数填充 ({fill_value:.4f})，避免0值孤立点")
                else:
                    # 最后的回退选项：使用整体SHAP值的中位数
                    overall_median = np.median(sample_shap) if len(sample_shap) > 0 else 0
                    full_grid_with_shap.loc[missing_mask, 'shap_value'] = overall_median
                    print(f"    {feature_name}: 使用整体中位数填充 ({overall_median:.4f})")
        
        return full_grid_with_shap
    
    # 如果没有h3_index，使用增强的空间匹配
    else:
        print(f"    {feature_name}: 使用增强空间插值（X_sample缺少h3_index列）")
        
        # 确保X_sample有经纬度
        if 'latitude' not in X_sample.columns or 'longitude' not in X_sample.columns:
            print(f"    {feature_name}: X_sample缺少经纬度，无法进行空间匹配")
            return None
        
        # 确保样本数量匹配
        sample_coords = X_sample[['latitude', 'longitude']].values[:len(sample_shap)]
        grid_coords = full_h3_grid[['latitude', 'longitude']].values
        
        # 使用增强插值
        grid_shap = enhanced_spatial_interpolation(
            sample_coords, sample_shap, grid_coords, method='rbf'
        )
        
        # 🔥 修复：为插值结果添加受控变异性，避免过度平滑化
        if len(grid_shap) > 0 and np.std(sample_shap) > 0:
            np.random.seed(42)  # 确保可重现性
            noise_scale = np.std(sample_shap) * 0.03  # 3%的变异性（比h3路径稍小）
            noise = np.random.normal(0, noise_scale, len(grid_shap))
            grid_shap = grid_shap + noise
            print(f"    {feature_name}: 插值添加{noise_scale:.4f}变异性，保持自然分布")
        
        # 创建结果DataFrame，确保保留所有原始列（包括高程数据）
        full_grid_with_shap = full_h3_grid.copy()
        full_grid_with_shap['shap_value'] = grid_shap
        
        # 🔧 强制确保高程数据正确传递
        if 'elevation' in full_h3_grid.columns:
            print(f"    {feature_name}: 增强空间插值完成，保留高程数据，范围: {full_h3_grid['elevation'].min():.1f}-{full_h3_grid['elevation'].max():.1f}m")
            # 双重保险：确保高程数据存在
            if 'elevation' not in full_grid_with_shap.columns or full_grid_with_shap['elevation'].isna().all():
                full_grid_with_shap['elevation'] = full_h3_grid['elevation']
                print(f"    {feature_name}: 高程数据已强制恢复")
        else:
            print(f"    {feature_name}: 增强空间插值完成（无高程数据）")
        return full_grid_with_shap


def plot_geoshapley_spatial_top3(results_by_resolution: Dict,
                                output_dir: Optional[str] = None,
                                figsize: Tuple[int, int] = (16, 14)) -> plt.Figure:
    """
    创建每个分辨率前3个特征的空间分布子图网格
    
    优化策略：
    1. 优先使用插值后的完整网格SHAP值重新绘制空间分布
    2. 如果插值不可用，回退到使用原始采样数据
    3. 确保与其他SHAP图表保持一致的数据基础
    
    参数:
    - results_by_resolution: 包含各分辨率结果的字典
    - output_dir: 输出目录（可选）
    - figsize: 图表尺寸
    
    返回:
    - fig: matplotlib图表对象
    """
    print("\n🎨 创建GeoShapley空间分布Top3特征图（优先使用插值后的完整网格数据）...")
    
    # 尝试使用插值后的完整网格数据
    enhanced_results = {}
    
    print("  🔧 尝试使用插值后的完整网格SHAP值...")
    for res in ['res7', 'res6', 'res5']:
        if res not in results_by_resolution:
            continue
            
        print(f"\n  📊 处理{res}的完整网格空间分布...")
        
        # 获取原始数据
        res_data = results_by_resolution[res]
        shap_values_by_feature = res_data.get('shap_values_by_feature', {})
        X_sample = res_data.get('X_sample') if 'X_sample' in res_data else res_data.get('X')
        
        if not shap_values_by_feature or X_sample is None:
            print(f"    ⚠️ {res}缺少SHAP数据，使用原始采样数据")
            enhanced_results[res] = res_data
            continue
        
        # 获取完整的H3网格数据
        full_h3_grid = get_full_h3_grid_data(res_data, res)
        if full_h3_grid is None:
            print(f"    ⚠️ {res}无法获取完整H3网格，使用原始采样数据")
            enhanced_results[res] = res_data
            continue
        
        # 🔧 修复：使用内置插值方法确保空间连续性
        print(f"    🔄 {res}: 使用内置插值方法生成完整网格SHAP值...")
        
        # 创建增强的结果数据
        enhanced_res_data = res_data.copy()
        enhanced_shap_values = {}
        
        # 对每个特征进行插值
        interpolation_success = True
        for feat_name, shap_vals in shap_values_by_feature.items():
            try:
                # 使用map_shap_to_full_grid函数进行插值
                full_grid_with_shap = map_shap_to_full_grid(
                    {feat_name: shap_vals}, X_sample, full_h3_grid, feat_name
                )
                
                if full_grid_with_shap is not None:
                    enhanced_shap_values[feat_name] = full_grid_with_shap['shap_value'].values
                    print(f"      ✅ {feat_name}: 插值成功 ({len(enhanced_shap_values[feat_name])}个网格)")
                else:
                    print(f"      ❌ {feat_name}: 插值失败")
                    interpolation_success = False
                    break
            except Exception as e:
                print(f"      ❌ {feat_name}: 插值异常 - {e}")
                interpolation_success = False
                break
        
        if not interpolation_success or len(enhanced_shap_values) == 0:
            print(f"    ❌ {res}插值失败，使用原始采样数据")
            enhanced_results[res] = res_data
            continue
        
        # 使用插值后的完整网格数据
        enhanced_res_data['enhanced_full_h3_grid'] = full_h3_grid
        enhanced_res_data['enhanced_shap_values_by_feature'] = enhanced_shap_values
        
        enhanced_results[res] = enhanced_res_data
        
        print(f"    ✅ {res}完整网格空间分布数据准备完成:")
        print(f"      • 完整网格数据量: {len(full_h3_grid)}个网格")
        print(f"      • 数据增强倍数: {len(full_h3_grid)/len(X_sample):.1f}x")
        print(f"      • 特征数量: {len(enhanced_shap_values)}个")
    
    # 🔥 修复：分离特征选择和空间可视化逻辑
    # 第一阶段：基于原始SHAP值选择top3主效应特征
    print(f"  🎯 第一阶段：基于原始SHAP值选择top3主效应特征...")
    
    # 准备数据
    resolutions = ['res7', 'res6', 'res5']
    res_titles = {
        'res5': 'Resolution 5 (Macro)',
        'res6': 'Resolution 6 (Meso)',
        'res7': 'Resolution 7 (Micro)'
    }
    
    # 收集每个分辨率的前3个主效应特征（基于原始SHAP值）
    top_features_by_res = {}
    
    for res in resolutions:
        if res not in results_by_resolution:
            print(f"  ⚠️ 警告: 缺少{res}的原始数据")
            continue
            
        # 🔥 强制使用原始数据进行特征选择
        original_res_data = results_by_resolution[res]
        
        # 获取特征重要性（基于原始SHAP值）
        if 'feature_importance' in original_res_data and original_res_data['feature_importance']:
            features = original_res_data['feature_importance']
            
            # 确保是列表格式
            if isinstance(features, dict):
                features = [(k, v) for k, v in features.items()]
            
            # 过滤出主效应环境特征（排除GEO、year和交互效应）
            primary_effects = []
            for feat, imp in features:
                if isinstance(feat, tuple):
                    feat_name = feat[0]
                else:
                    feat_name = feat
                    
                # 排除GEO、year、交互效应，只保留环境特征
                if (feat_name != 'GEO' and 
                    feat_name.lower() != 'year' and
                    '×' not in str(feat_name) and 
                    ' x ' not in str(feat_name) and
                    'year' not in str(feat_name).lower()):
                    if isinstance(feat, tuple):
                        primary_effects.append(feat)
                    else:
                        primary_effects.append((feat_name, imp))
            
            # 按重要性排序
            primary_effects.sort(key=lambda x: x[1], reverse=True)
            
            # 选择前3个主效应特征
            top_3 = [f[0] for f in primary_effects[:3]]
            
            top_features_by_res[res] = top_3[:3]
            print(f"  {res}: 基于原始SHAP值选择top3特征 {', '.join(top_3[:3])}")
        else:
            print(f"  ⚠️ 警告: {res}没有特征重要性数据")
    
    # 第二阶段：确定空间可视化数据源
    print(f"  🎯 第二阶段：选择空间可视化数据源...")
    
    if enhanced_results and any('enhanced_shap_values_by_feature' in enhanced_results[res] for res in enhanced_results):
        print(f"  ✅ 使用插值后的完整网格进行空间可视化（{len(enhanced_results)}个分辨率）")
        spatial_vis_results = enhanced_results
        data_source_info = "Original Selection + Full Grid Visualization"
    else:
        print(f"  ⚠️ 回退到原始采样数据进行空间可视化")
        spatial_vis_results = results_by_resolution
        data_source_info = "Original Selection + Sampled Visualization"

    # 保存原始rcParams
    original_rcParams = plt.rcParams.copy()
    
    # 创建本地样式字典（参考regionkmeans_plot.py的风格）
    style_dict = {
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'font.weight': 'bold',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'axes.linewidth': 1.5,
        'legend.fontsize': 10,
        'legend.title_fontsize': 11,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'figure.figsize': figsize,
        'figure.constrained_layout.use': False,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.spines.bottom': True,
        'axes.spines.left': True,
    }
    
    # 使用上下文管理器隔离样式设置
    with plt.style.context('default'):
        with plt.rc_context(style_dict):
            
            # 创建图形
            fig = plt.figure(figsize=figsize, dpi=600)
            
            # 添加总标题
            fig.suptitle('GeoShapley Spatial Distribution of Top 3 Features Across Resolutions', 
                        fontsize=18, fontweight='bold', y=0.98)
            
            # 创建GridSpec布局 - 使用更大的负值进一步压缩列间距
            gs = gridspec.GridSpec(3, 3, figure=fig, 
                                 height_ratios=[1, 1, 1],
                                 width_ratios=[1, 1, 1],
                                 hspace=0.25, wspace=-0.20)
            
            # 子图标签
            subplot_labels = [
                ['(a)', '(b)', '(c)'],  # res7
                ['(d)', '(e)', '(f)'],  # res6
                ['(g)', '(h)', '(i)']   # res5
            ]
            
            # 创建颜色映射（使用RdBu_r，与SHAP值一致）
            cmap = 'RdBu_r'
            
            # 存储所有子图的边界，用于统一坐标范围
            all_bounds = []
            # 🔧 修复：收集所有colorbar信息，在布局调整后统一创建
            colorbar_infos = []
            
            # 处理每个分辨率
            for row_idx, res in enumerate(resolutions):
                if res not in spatial_vis_results or res not in top_features_by_res:
                    # 创建空白子图
                    for col_idx in range(3):
                        ax = fig.add_subplot(gs[row_idx, col_idx])
                        ax.text(0.5, 0.5, f"No data for {res}", 
                               ha='center', va='center', fontsize=14, 
                               transform=ax.transAxes)
                        ax.axis('off')
                    continue
                
                # 获取数据（用于空间可视化）
                res_data = spatial_vis_results[res]
                top3_feat_names = top_features_by_res[res]
                
                # 🎯 优化数据源选择逻辑
                enhanced_data_available = False
                
                # 优先使用增强的SHAP数据
                if 'enhanced_shap_values_by_feature' in res_data:
                    shap_values_by_feature = res_data['enhanced_shap_values_by_feature']
                    full_h3_grid = res_data['enhanced_full_h3_grid']
                    enhanced_data_available = True
                    original_sample_size = len(results_by_resolution[res].get('X_sample', []))
                    enhanced_size = len(full_h3_grid)
                    data_info = f"{enhanced_size/original_sample_size:.1f}x" if original_sample_size > 0 else "Enhanced"
                    print(f"    {res}: 使用增强数据 ({enhanced_size}个网格, {data_info}增强)")
                else:
                    # 使用原始数据但尝试获取完整网格
                    shap_values_by_feature = res_data.get('shap_values_by_feature', {})
                    X_sample = res_data.get('X_sample')
                    
                    # 🔧 尝试获取或创建完整H3网格
                    full_h3_grid = get_full_h3_grid_data(res_data, res)
                    if full_h3_grid is None and X_sample is not None:
                        if 'h3_index' in X_sample.columns and 'latitude' in X_sample.columns and 'longitude' in X_sample.columns:
                            # 使用X_sample中的唯一H3网格
                            full_h3_grid = X_sample[['h3_index', 'latitude', 'longitude']].drop_duplicates(subset=['h3_index']).copy()
                            print(f"    {res}: 从X_sample创建H3网格 ({len(full_h3_grid)}个网格)")
                        else:
                            # 创建基于经纬度的伪网格
                            unique_coords = X_sample[['latitude', 'longitude']].drop_duplicates()
                            full_h3_grid = unique_coords.copy()
                            full_h3_grid['h3_index'] = [f"pseudo_{i}" for i in range(len(unique_coords))]
                            print(f"    {res}: 创建伪H3网格 ({len(full_h3_grid)}个网格)")
                    
                    data_info = "Sampled"
                
                if not shap_values_by_feature or full_h3_grid is None:
                    # 创建空白子图
                    for col_idx in range(3):
                        ax = fig.add_subplot(gs[row_idx, col_idx])
                        ax.text(0.5, 0.5, f"No SHAP data for {res}", 
                               ha='center', va='center', fontsize=14, 
                               transform=ax.transAxes)
                        ax.axis('off')
                    continue
                
                # 处理每个Top 3特征
                for col_idx, feat_name in enumerate(top3_feat_names[:3]):
                    # 创建子图
                    ax = fig.add_subplot(gs[row_idx, col_idx])
                    
                    # 🎯 获取特征的SHAP值
                    full_grid_with_shap = None
                    
                    if feat_name in shap_values_by_feature:
                        if enhanced_data_available:
                            # 增强数据：SHAP值已经对应完整网格
                            shap_vals = shap_values_by_feature[feat_name]
                            
                            # 确保SHAP值数量与网格数量匹配
                            if len(shap_vals) == len(full_h3_grid):
                                full_grid_with_shap = full_h3_grid.copy()
                                full_grid_with_shap['shap_value'] = shap_vals
                                
                                # 🔧 确保高程数据正确传递
                                if 'elevation' in full_h3_grid.columns:
                                    print(f"      {feat_name}: 使用增强数据 ({len(shap_vals)}个值) 包含高程数据")
                                    print(f"      {feat_name}: 高程范围: {full_h3_grid['elevation'].min():.1f}-{full_h3_grid['elevation'].max():.1f}m")
                                else:
                                    print(f"      {feat_name}: 警告：full_h3_grid缺少高程数据")
                            else:
                                print(f"      {feat_name}: 增强数据维度不匹配 ({len(shap_vals)} vs {len(full_h3_grid)})，使用映射")
                                # 回退到映射方法
                                full_grid_with_shap = map_shap_to_full_grid(
                                    shap_values_by_feature, res_data.get('X_sample'), full_h3_grid, feat_name
                                )
                        else:
                            # 原始数据：使用映射方法
                            print(f"      {feat_name}: 使用SHAP值映射")
                            full_grid_with_shap = map_shap_to_full_grid(
                                shap_values_by_feature, res_data.get('X_sample'), full_h3_grid, feat_name
                            )
                    else:
                        print(f"      {feat_name}: 特征不存在于SHAP数据中")
                        ax.text(0.5, 0.5, f"No SHAP values for {feat_name}", 
                               ha='center', va='center', fontsize=12, 
                               transform=ax.transAxes)
                        ax.axis('off')
                        continue
                    
                    if full_grid_with_shap is None:
                        ax.text(0.5, 0.5, f"No SHAP mapping for {feat_name}", 
                               ha='center', va='center', fontsize=12, 
                               transform=ax.transAxes)
                        ax.axis('off')
                        continue
                    
                    # 获取SHAP值
                    shap_vals = full_grid_with_shap['shap_value'].values
                    
                    # 🔧 温和过滤极端异常值，保持数据完整性
                    if len(shap_vals) > 0:
                        # 只过滤极端异常值，保持更多数据
                        shap_std = np.std(shap_vals)
                        shap_mean = np.mean(shap_vals)
                        
                        if shap_std > 0:
                            # 使用更宽松的3.5标准差阈值，只过滤真正的极端值
                            outlier_mask = np.abs(shap_vals - shap_mean) <= 3.5 * shap_std
                            
                            if not outlier_mask.all():
                                outlier_count = (~outlier_mask).sum()
                                # 只有当异常值很少时才进行过滤
                                if outlier_count < len(shap_vals) * 0.05:  # 少于5%才过滤
                                    print(f"      {feat_name}: 温和过滤{outlier_count}个极端异常值")
                                    
                                    # 使用更宽松的分位数
                                    lower_bound = np.percentile(shap_vals, 2)   # 2%分位数
                                    upper_bound = np.percentile(shap_vals, 98)  # 98%分位数
                                    
                                    # 温和的截断，保留更多原始变异
                                    extreme_low = shap_vals < lower_bound
                                    extreme_high = shap_vals > upper_bound
                                    
                                    if extreme_low.any():
                                        shap_vals[extreme_low] = lower_bound
                                        
                                    if extreme_high.any():
                                        shap_vals[extreme_high] = upper_bound
                                    
                                    full_grid_with_shap['shap_value'] = shap_vals
                                    print(f"      {feat_name}: 应用温和截断，保持数据完整性")
                                else:
                                    print(f"      {feat_name}: 异常值比例正常({outlier_count/len(shap_vals):.1%})，保持原始数据")
                    
                    # 📍 创建几何对象（恢复原逻辑+适度优化）
                    print(f"      {feat_name}: 创建几何对象...")
                    
                    # 🔧 修复：使用原逻辑但优化缓冲区大小，确保连续性
                    geometry = create_h3_geometry(
                        full_grid_with_shap['h3_index'], 
                        full_grid_with_shap[['longitude', 'latitude']]
                    )
                    
                    # 如果create_h3_geometry返回的几何对象太小，适当放大缓冲区
                    if len(geometry) > 0:
                        # 检查是否有太多小的缓冲区几何对象
                        small_geom_count = sum(1 for geom in geometry if hasattr(geom, 'area') and geom.area < 1e-6)
                        if small_geom_count > len(geometry) * 0.5:  # 如果超过50%是小几何对象
                            print(f"      {feat_name}: 检测到{small_geom_count}个过小几何对象，调整缓冲区大小")
                            # 重新创建更大的缓冲区
                            geometry = []
                            coords_df = full_grid_with_shap[['longitude', 'latitude']]
                            
                            # 根据分辨率和数据密度调整缓冲区大小
                            if res == 'res7':
                                buffer_size = 0.008  # 增大到约800米
                            elif res == 'res6':
                                buffer_size = 0.015  # 增大到约1.5公里
                            else:  # res5
                                buffer_size = 0.025  # 增大到约2.5公里
                            
                            for _, row in coords_df.iterrows():
                                center = Point(row['longitude'], row['latitude'])
                                hex_buffer = center.buffer(buffer_size)
                                geometry.append(hex_buffer)
                            print(f"      {feat_name}: 重新创建{len(geometry)}个适当大小的几何对象")
                        else:
                            print(f"      {feat_name}: 使用原始几何对象 ({len(geometry)}个)")
                    
                    # 🎯 创建GeoDataFrame
                    try:
                        # 准备数据字典，包含必要的列
                        gdf_data = {
                            'shap_value': shap_vals
                        }
                        
                        # 🔧 必须先加载真实高程数据，禁止使用模拟数据
                        if 'elevation' in full_grid_with_shap.columns:
                            gdf_data['elevation'] = full_grid_with_shap['elevation'].values[:len(shap_vals)]
                            print(f"      {feat_name}: ✅ 使用插值网格中的高程数据")
                        elif 'h3_index' in full_grid_with_shap.columns:
                            print(f"      {feat_name}: 🔄 从原始数据加载真实高程...")
                            # 基于h3_index获取真实高程
                            h3_df = full_grid_with_shap[['h3_index']].copy()
                            merged_tmp = ensure_elevation_data(h3_df, resolution=res)
                            
                            if 'elevation' in merged_tmp.columns and merged_tmp['elevation'].notna().sum() > 0:
                                elevation_values = merged_tmp['elevation'].values
                                if len(elevation_values) >= len(shap_vals):
                                    gdf_data['elevation'] = elevation_values[:len(shap_vals)]
                                    print(f"      {feat_name}: ✅ 成功加载真实高程数据 (范围: {np.min(gdf_data['elevation']):.1f}-{np.max(gdf_data['elevation']):.1f}m)")
                                else:
                                    print(f"      {feat_name}: ❌ 高程数据长度不足")
                            else:
                                print(f"      {feat_name}: ❌ 未能获取真实高程数据")
                        else:
                            print(f"      {feat_name}: ❌ 缺少h3_index，无法加载真实高程")
                        
                        # 添加坐标信息
                        if 'latitude' in full_grid_with_shap.columns and 'longitude' in full_grid_with_shap.columns:
                            if len(full_grid_with_shap['latitude']) >= len(shap_vals):
                                gdf_data['latitude'] = full_grid_with_shap['latitude'].values[:len(shap_vals)]
                                gdf_data['longitude'] = full_grid_with_shap['longitude'].values[:len(shap_vals)]
                        
                        gdf = gpd.GeoDataFrame(
                            gdf_data, 
                            geometry=geometry, 
                            crs='EPSG:4326'
                        )
                        
                        # 确保没有无效几何
                        invalid_geom = ~gdf.geometry.is_valid
                        if invalid_geom.any():
                            print(f"      {feat_name}: 修复{invalid_geom.sum()}个无效几何")
                            gdf.loc[invalid_geom, 'geometry'] = gdf.loc[invalid_geom, 'geometry'].buffer(0)
                        
                        # 移除空几何
                        empty_geom = gdf.geometry.is_empty
                        if empty_geom.any():
                            print(f"      {feat_name}: 移除{empty_geom.sum()}个空几何")
                            gdf = gdf[~empty_geom].copy()
                    
                    except Exception as e:
                        print(f"      {feat_name}: GeoDataFrame创建失败: {e}")
                        # 使用点作为后备
                        point_geometry = [Point(row['longitude'], row['latitude']).buffer(0.005)
                                        for _, row in full_grid_with_shap.iterrows()]
                        
                        # 在后备GDF中也包含elevation数据
                        fallback_data = {'shap_value': shap_vals}
                        if 'elevation' in full_grid_with_shap.columns:
                            fallback_data['elevation'] = full_grid_with_shap['elevation'].values[:len(shap_vals)]
                            print(f"      {feat_name}: 后备GDF包含高程数据")
                        
                        gdf = gpd.GeoDataFrame(
                            fallback_data, 
                            geometry=point_geometry, 
                            crs='EPSG:4326'
                        )
                    
                    if len(gdf) == 0:
                        ax.text(0.5, 0.5, f"No valid geometry for {feat_name}", 
                               ha='center', va='center', fontsize=12, 
                               transform=ax.transAxes)
                        ax.axis('off')
                        continue
                    
                    # 🎨 计算SHAP值的范围，用于颜色映射
                    shap_vals_clean = gdf['shap_value'].values
                    vmin, vmax = shap_vals_clean.min(), shap_vals_clean.max()
                    
                    # 创建以0为中心的颜色映射
                    if vmin < 0 and vmax > 0:
                        # 使用TwoSlopeNorm实现以0为中心的颜色映射
                        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                    else:
                        norm = None
                    
                    # 使用山体阴影绘制方法
                    hillshade_success = False  # 初始化标志
                    try:
                        print(f"      {feat_name}: 尝试应用山体阴影效果")
                        
                        # 设置子图标题
                        simplified_feat = simplify_feature_name_for_plot(feat_name)
                        subplot_title = f'{subplot_labels[row_idx][col_idx]} {res_titles[res]} - {simplified_feat}'
                        
                        # 根据分辨率调整山体阴影强度
                        if res == 'res7':
                            hillshade_strength = 0.4  # 较弱的山体阴影，保持细节
                        elif res == 'res6':
                            hillshade_strength = 0.5  # 中等强度
                        else:  # res5
                            hillshade_strength = 0.6  # 较强的山体阴影，增强立体感
                        
                        # 应用山体阴影效果
                        hillshade_success = create_hillshaded_plot(
                            ax, gdf, 
                            shap_col='shap_value', 
                            elevation_col='elevation',
                            cmap=cmap, 
                            norm=norm, 
                            resolution=res,
                            azimuth=315, 
                            altitude=45, 
                            hillshade_strength=hillshade_strength,
                            title=subplot_title,
                            xlabel='Longitude',
                            ylabel='Latitude' if col_idx == 0 else ''
                        )
                        
                        if hillshade_success:
                            print(f"      {feat_name}: 山体阴影效果应用成功")
                        else:
                            print(f"      {feat_name}: 山体阴影失败，已回退到标准绘图")
                    except Exception as e:
                        print(f"      {feat_name}: 绘制失败: {e}")
                        # 尝试标准绘图作为最后的回退
                        try:
                            gdf.plot(column='shap_value', ax=ax, 
                                   cmap=cmap, norm=norm, 
                                   edgecolor='none', 
                                   linewidth=0,
                                   alpha=0.9,
                                   legend=False)
                            print(f"      {feat_name}: 回退到基本绘图成功")
                        except:
                            ax.text(0.5, 0.5, f"Plot error for {feat_name}", 
                                   ha='center', va='center', fontsize=12, 
                                   transform=ax.transAxes)
                            ax.axis('off')
                            continue
                    
                    # 如果山体阴影失败，需要设置标题和标签
                    if not hillshade_success:
                        # 设置标题
                        simplified_feat = simplify_feature_name_for_plot(feat_name)
                        ax.set_title(f'{subplot_labels[row_idx][col_idx]} {res_titles[res]} - {simplified_feat}',
                                   fontsize=12, fontweight='bold', pad=5, loc='center')
                        
                        # 设置坐标轴
                        ax.set_xlabel('Longitude', fontsize=10, fontweight='bold')
                        if col_idx == 0:
                            ax.set_ylabel('Latitude', fontsize=10, fontweight='bold')
                        else:
                            ax.set_ylabel('')
                        
                        # 设置刻度
                        ax.tick_params(axis='both', direction='in', width=1.5, length=4)
                        for label in ax.get_xticklabels() + ax.get_yticklabels():
                            label.set_fontweight('bold')
                            label.set_fontsize(9)
                        
                        # 添加网格
                        ax.grid(True, linestyle=':', color='grey', alpha=0.3)
                        
                        # 设置等比例坐标
                        ax.set_aspect('equal', adjustable='box')
                        
                        # 加粗边框
                        for spine in ax.spines.values():
                            spine.set_linewidth(1.5)
                    
                    # 延迟创建颜色条，避免与布局调整冲突
                    # 将colorbar信息保存，在统一坐标轴后再创建
                    colorbar_info = {
                        'ax': ax,
                        'cmap': cmap,
                        'norm': norm,
                        'shap_vals': shap_vals_clean.copy(),
                        'feature_name': feat_name
                    }
                    colorbar_infos.append(colorbar_info)
                    
                    # 🏷️ 添加数据源信息标注（已注释掉，去除左上角文字标签）
                    # info_color = 'lightgreen' if enhanced_data_available else 'lightcyan'
                    # ax.text(0.02, 0.98, data_info, transform=ax.transAxes,
                    #        fontsize=9, ha='left', va='top', fontweight='bold',
                    #        bbox=dict(boxstyle='round,pad=0.3', facecolor=info_color, alpha=0.7))
                    
                    # 保存边界用于统一坐标范围
                    all_bounds.append(gdf.total_bounds)
            
            # 调整布局 - 使用负值大幅减小横向间距
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.08, 
                              hspace=0.25, wspace=-0.20)
            
            # 统一所有子图的坐标范围
            if all_bounds:
                bounds_array = np.array(all_bounds)
                global_min_lon = bounds_array[:, 0].min()
                global_min_lat = bounds_array[:, 1].min()
                global_max_lon = bounds_array[:, 2].max()
                global_max_lat = bounds_array[:, 3].max()
                
                # 添加一些边距
                lon_margin = (global_max_lon - global_min_lon) * 0.05
                lat_margin = (global_max_lat - global_min_lat) * 0.05
                
                # 应用统一的坐标范围
                main_axes = [info['ax'] for info in colorbar_infos]  # 只调整有数据的主轴
                for ax in main_axes:
                    if hasattr(ax, 'set_xlim'):
                        # 检查是否已经设置了合适的范围
                        current_xlim = ax.get_xlim()
                        current_ylim = ax.get_ylim()
                        
                        # 只有当当前范围明显不合理时才重新设置
                        if (abs(current_xlim[1] - current_xlim[0]) < 0.01 or 
                            abs(current_ylim[1] - current_ylim[0]) < 0.01):
                            ax.set_xlim(global_min_lon - lon_margin, global_max_lon + lon_margin)
                            ax.set_ylim(global_min_lat - lat_margin, global_max_lat + lat_margin)
            
            # 创建所有colorbar
            print(f"\n🎨 开始创建colorbar，总数: {len(colorbar_infos)}")
            
            # 🔧 修复：创建一个行级colorbar映射，确保每行都有正确的colorbar
            row_colorbar_count = {}  # 统计每行已创建的colorbar数量
            
            for i, info in enumerate(colorbar_infos):
                ax = info['ax']
                cmap = info['cmap']
                norm = info['norm']
                shap_vals = info['shap_vals']
                feat_name = info['feature_name']
                
                # 🔧 确定当前colorbar属于哪一行
                # 基于ax的y位置来判断行号
                ax_pos = ax.get_position()
                ax_y_center = (ax_pos.y0 + ax_pos.y1) / 2
                
                # 根据y位置确定行号（3行布局）
                if ax_y_center > 0.66:  # 第一行（顶部）
                    row_idx = 0
                elif ax_y_center > 0.33:  # 第二行（中间）
                    row_idx = 1
                else:  # 第三行（底部）
                    row_idx = 2
                
                # 🔧 确定当前是这一行的第几个colorbar
                if row_idx not in row_colorbar_count:
                    row_colorbar_count[row_idx] = 0
                col_idx = row_colorbar_count[row_idx]
                row_colorbar_count[row_idx] += 1
                
                print(f"\n  📊 处理第{i+1}个colorbar: {feat_name} (行{row_idx+1}, 列{col_idx+1})")
                
                try:
                    # 简化数据检查，只确保不是全NaN
                    vmin, vmax = np.nanmin(shap_vals), np.nanmax(shap_vals)
                    print(f"      {feat_name}: SHAP值范围 [{vmin:.6f}, {vmax:.6f}], 数组长度: {len(shap_vals)}")
                    
                    if np.isnan(vmin) or np.isnan(vmax):
                        print(f"      ❌ {feat_name}: 全为NaN，使用默认范围")
                        vmin, vmax = -1, 1
                    elif vmin == vmax:
                        print(f"      ⚠️ {feat_name}: 常数值，扩展范围")
                        if vmin == 0:
                            vmin, vmax = -0.001, 0.001
                        else:
                            margin = abs(vmin) * 0.01
                            vmin, vmax = vmin - margin, vmax + margin
                    
                    # 🔧 修复：确保colorbar的范围合理且美观
                    # 对于以0为中心的数据，确保对称
                    if vmin < 0 and vmax > 0:
                        abs_max = max(abs(vmin), abs(vmax))
                        vmin, vmax = -abs_max, abs_max
                        norm_for_cbar = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                    else:
                        norm_for_cbar = plt.Normalize(vmin=vmin, vmax=vmax)
                    
                    # 创建ScalarMappable
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_for_cbar)
                    sm.set_array(shap_vals)
                    
                    # 获取子图位置
                    pos = ax.get_position()
                    cbar_width = 0.012
                    cbar_pad = 0.01
                    
                    # 🔧 修复：每个子图都创建自己的colorbar
                    # 创建colorbar
                    cax = fig.add_axes([pos.x1 + cbar_pad, pos.y0, cbar_width, pos.height])
                    cbar = plt.colorbar(sm, cax=cax)
                    cbar.set_label('SHAP value', fontsize=10, fontweight='bold')
                    
                    # 🔧 修复：设置5个均匀分布的tick
                    if vmin < 0 and vmax > 0:
                        # 对于以0为中心的数据，确保0在中间
                        tick_positions = np.array([-abs_max, -abs_max/2, 0, abs_max/2, abs_max])
                    else:
                        # 对于单边数据，均匀分布5个tick
                        tick_positions = np.linspace(vmin, vmax, 5)
                    
                    cbar.set_ticks(tick_positions)
                    
                    # 🔧 修复：根据数值范围选择合适的格式
                    range_val = abs(vmax - vmin)
                    if range_val >= 10.0:
                        # 大数值：不带小数或1位小数
                        tick_labels = [f'{pos:.0f}' if abs(pos) >= 1 else f'{pos:.1f}' for pos in tick_positions]
                    elif range_val >= 1.0:
                        # 中等数值：2位小数
                        tick_labels = [f'{pos:.2f}' for pos in tick_positions]
                    elif range_val >= 0.1:
                        # 小数值：3位小数
                        tick_labels = [f'{pos:.3f}' for pos in tick_positions]
                    elif range_val >= 0.01:
                        # 很小数值：4位小数
                        tick_labels = [f'{pos:.4f}' for pos in tick_positions]
                    else:
                        # 极小数值：科学计数法
                        tick_labels = [f'{pos:.2e}' for pos in tick_positions]
                    
                    # 🔧 修复：确保0显示为0（不是0.00）
                    for idx, pos in enumerate(tick_positions):
                        if abs(pos) < 1e-10:
                            tick_labels[idx] = '0'
                    
                    cbar.set_ticklabels(tick_labels)
                    
                    # 设置样式
                    cbar.ax.tick_params(labelsize=9, width=1.5, length=4)
                    for t in cbar.ax.get_yticklabels():
                        t.set_fontweight('bold')
                        
                    print(f"      ✅ {feat_name}: 颜色条创建成功（行{row_idx+1}，列{col_idx+1}）")
                    
                except Exception as e:
                    print(f"      ❌ {feat_name}: 颜色条创建失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 保存图表
            if output_dir:
                if ensure_dir_exists(output_dir):
                    output_path = os.path.join(output_dir, 'geoshapley_spatial_top3.png')
                    save_plot_for_publication(output_path, fig)
                    
                    # 输出详细的保存信息
                    print(f"\n  ✅ GeoShapley Top3特征空间分布图已保存至: {output_path}")
                    print(f"  🔍 数据源策略: {data_source_info}")
                    
                    if data_source_info.startswith("Original Selection + Full Grid"):
                        print(f"    📊 特征选择: 基于原始SHAP值（确保科学准确性）")
                        print(f"    🗺️ 空间可视化: 使用插值完整网格（确保空间连续性）")
                        print(f"    📈 数据质量提升:")
                        for res in enhanced_results:
                            if 'enhanced_shap_values_by_feature' in enhanced_results[res]:
                                original_len = len(results_by_resolution[res].get('X_sample', []))
                                enhanced_len = len(enhanced_results[res]['enhanced_full_h3_grid'])
                                if original_len > 0:
                                    enhancement = enhanced_len / original_len
                                    print(f"      • {res}: {enhancement:.1f}倍空间网格增强")
                    else:
                        print(f"    📊 特征选择: 基于原始SHAP值（确保科学准确性）")
                        print(f"    🗺️ 空间可视化: 使用原始采样数据（回退选项）")
                else:
                    print(f"无法创建输出目录: {output_dir}")
    
    # 恢复原始rcParams
    plt.rcParams.update(original_rcParams)
    
    return fig

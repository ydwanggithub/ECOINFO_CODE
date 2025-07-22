#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
区域KMeans聚类算法实现
"""

# 设置OMP_NUM_THREADS环境变量，避免Windows上KMeans内存泄漏问题
# 这必须在导入sklearn之前进行设置
import os
import sys
import platform
if platform.system() == 'Windows' and 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'
    print(f"在regionkmeans_cluster.py中设置OMP_NUM_THREADS=1，避免Windows上KMeans内存泄漏问题")

import numpy as np
import pandas as pd
import libpysal
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
import time
import threading
import logging

# 尝试导入h3库，支持多种版本
try:
    import h3
    H3_AVAILABLE = True
    print("cluster模块: 成功导入h3库")
except ImportError:
    try:
        # 尝试使用h3ronpy作为替代
        from h3ronpy import h3
        H3_AVAILABLE = True
        print("cluster模块: 使用h3ronpy作为h3库替代")
    except ImportError:
        H3_AVAILABLE = False
        print("cluster模块: 未能导入h3库，部分H3功能将不可用")

# 设置日志
logger = logging.getLogger(__name__)

# 给 PySAL W 类添加 to_adjlist 方法，确保 RegionKMeans 正常调用
if not hasattr(libpysal.weights.W, 'to_adjlist'):
    def w_to_adjlist(self):
        # 从稀疈矩阵获取邻接对
        rows, cols = self.sparse.nonzero()
        return pd.DataFrame({'focal': rows, 'neighbor': cols})
    libpysal.weights.W.to_adjlist = w_to_adjlist


def _solve_with_timeout(model, timeout_seconds=300):
    """
    在指定的超时时间内执行聚类求解，使用多线程实现超时检测
    返回: (成功标志, 错误信息)
    """
    result = {"success": False, "error": None, "completed": False}
    
    def solve_func():
        try:
            model.solve()
            result["success"] = True
        except Exception as e:
            result["error"] = str(e)
        finally:
            result["completed"] = True
    
    thread = threading.Thread(target=solve_func)
    thread.daemon = True
    thread.start()
    
    start_time = time.time()
    print(f"RegionKMeans 求解开始 (超时 {timeout_seconds}s)")
    
    while not result["completed"] and (time.time() - start_time) < timeout_seconds:
        time.sleep(1.0)  # 等待完成或超时，忽略中间日志
    
    if not result["completed"]:
        print("✗ RegionKMeans 求解超时")
        return (False, result.get("error") or "超时")
    
    if result["success"]:
        print("✓ RegionKMeans 求解完成")
    else:
        print(f"✗ RegionKMeans 求解失败: {result.get('error')}")
    return (result["success"], result.get("error"))


def create_optimized_spatial_weights(coords_array, k, h3_indices=None):
    """
    创建优化的空间权重矩阵，支持H3 grid_disk或KNN
    
    参数:
    - coords_array: 坐标数组
    - k: KNN邻居数量或H3 grid_disk半径
    - h3_indices: 可选的H3索引，用于基于H3的邻接关系
    
    返回:
    - w: libpysal空间权重对象（行标准化）
    """
    import warnings
    import sys
    from io import StringIO
    
    # 抑制libpysal的孤立点警告信息
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        # 重定向标准输出和错误输出，抑制孤立点警告
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            data_size = len(coords_array)
            if (h3_indices is not None) and (len(h3_indices) == data_size) and H3_AVAILABLE:
                try:
                    # 确认h3库中存在grid_disk方法
                    if hasattr(h3, 'grid_disk') or hasattr(h3, 'k_ring'):
                        grid_disk_func = None
                        if hasattr(h3, 'grid_disk'):
                            grid_disk_func = lambda idx, k: h3.grid_disk(idx, k)
                        elif hasattr(h3, 'k_ring'):
                            grid_disk_func = lambda idx, k: h3.k_ring(idx, k)
                        
                        if grid_disk_func:
                            n = len(h3_indices)
                            adj_matrix = np.zeros((n, n), dtype=int)
                            for i, h3_idx in enumerate(h3_indices):
                                try:
                                    neighbors = grid_disk_func(h3_idx, k)
                                    for neighbor in neighbors:
                                        neighbor_indices = np.where(h3_indices == neighbor)[0]
                                        if len(neighbor_indices) > 0:
                                            for j in neighbor_indices:
                                                adj_matrix[i, j] = 1
                                except:
                                    pass  # 静默处理失败
                            
                            knn_graph = csr_matrix(adj_matrix)
                        else:
                            n_neighbors = min(max(2, min(k, data_size-1)), data_size-1)
                            knn_graph = kneighbors_graph(coords_array, n_neighbors=n_neighbors, mode='connectivity', n_jobs=-1)
                    else:
                        n_neighbors = min(max(2, min(k, data_size-1)), data_size-1)
                        knn_graph = kneighbors_graph(coords_array, n_neighbors=n_neighbors, mode='connectivity', n_jobs=-1)
                except:
                    n_neighbors = min(max(2, min(k, data_size-1)), data_size-1)
                    knn_graph = kneighbors_graph(coords_array, n_neighbors=n_neighbors, mode='connectivity', n_jobs=-1)
            else:
                # 使用KNN构建邻接关系
                n_neighbors = min(max(2, min(k, data_size-1)), data_size-1)
                knn_graph = kneighbors_graph(coords_array, n_neighbors=n_neighbors, mode='connectivity', n_jobs=-1)

            # 构建邻居字典和权重字典
            try:
                neighbors = {i: list(knn_graph.indices[knn_graph.indptr[i]:knn_graph.indptr[i+1]]) 
                             for i in range(knn_graph.shape[0])}
                weights = {i: {j: 1.0 for j in neighbors[i]} for i in range(knn_graph.shape[0])}

                w = libpysal.weights.W(weights)
                w.transform = 'r'
            except:
                # 创建一个简单的邻接矩阵作为备选
                n = data_size
                simple_neighbors = {i: [j for j in range(n) if j != i and np.random.random() < 3.0/n] for i in range(n)}
                for i in range(n):
                    if len(simple_neighbors[i]) == 0:
                        j = (i + 1) % n
                        simple_neighbors[i].append(j)
                        simple_neighbors[j].append(i)
                simple_weights = {i: {j: 1.0 for j in simple_neighbors[i]} for i in range(n)}
                w = libpysal.weights.W(simple_weights)
                w.transform = 'r'

            # 处理孤立点（静默处理）
            if hasattr(w, 'islands') and w.islands:
                for island in w.islands:
                    try:
                        distances = []
                        for j in range(len(coords_array)):
                            if j != island:
                                dist = np.sum((coords_array[island] - coords_array[j])**2)
                                distances.append((j, dist))
                        distances.sort(key=lambda x: x[1])
                        nearest = distances[:2]
                        neighbors[island] = [n[0] for n in nearest]
                        weights[island] = {n[0]: 1.0 for n in nearest}
                        for n in nearest:
                            if n[0] not in neighbors:
                                neighbors[n[0]] = [island]
                                weights[n[0]] = {island: 1.0}
                            else:
                                neighbors[n[0]].append(island)
                                weights[n[0]][island] = 1.0
                    except:
                        pass  # 静默处理失败
                
                try:
                    w = libpysal.weights.W(weights)
                    w.transform = 'r'
                except:
                    pass  # 静默处理失败

            # 检查连通性（静默）
            try:
                n_components, _ = connected_components(w.sparse, directed=False)
                if n_components > 1:
                    # 自适应邻居数
                    if data_size < 500:
                        k_used = min(10, data_size - 1)
                    else:
                        k_used = min(6, data_size - 1)
                    try:
                        knn_graph_new = kneighbors_graph(coords_array, n_neighbors=k_used, mode='connectivity', n_jobs=-1)
                        neighbors_new = {i: list(knn_graph_new.indices[knn_graph_new.indptr[i]:knn_graph_new.indptr[i+1]])
                                        for i in range(knn_graph_new.shape[0])}
                        weights_new = {i: {j: 1.0 for j in neighbors_new[i]} for i in range(knn_graph_new.shape[0])}
                        w = libpysal.weights.W(weights_new)
                        w.transform = 'r'
                    except:
                        pass  # 静默处理失败
            except:
                pass  # 静默处理失败

            # Monkey-patch to_adjlist 方法
            if not hasattr(w, 'to_adjlist'):
                def to_adjlist():
                    adj = []
                    for i, neis in w.neighbors.items():
                        for j in neis:
                            adj.append((i, j))
                    return pd.DataFrame(adj, columns=['focal', 'neighbor'])
                w.to_adjlist = to_adjlist

    finally:
        # 恢复标准输出和错误输出
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    print("✓ 空间权重矩阵构建完成")
    return w


def create_subgraph(w, indices):
    """
    手动创建空间权重矩阵的子图
    
    参数:
    - w: 原始空间权重矩阵 (libpysal.weights.W对象)
    - indices: 子图包含的点的索引
    
    返回:
    - subw: 子图的空间权重矩阵
    """
    # 创建索引映射，从原始索引到子图索引
    idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
    
    # 创建新的邻居字典和权重字典
    new_neighbors = {}
    new_weights = {}
    
    for i, idx in enumerate(indices):
        # 获取当前点的邻居
        neighbors_in_subgraph = []
        weights_in_subgraph = {}
        
        for neighbor in w.neighbors[idx]:
            # 只保留也在子图中的邻居
            if neighbor in idx_map:
                neighbors_in_subgraph.append(idx_map[neighbor])
                weights_in_subgraph[idx_map[neighbor]] = w.weights[idx][neighbor]
        
        new_neighbors[i] = neighbors_in_subgraph
        new_weights[i] = weights_in_subgraph
    
    # 创建新的W对象
    subw = libpysal.weights.W(new_weights, id_order=range(len(indices)))
    
    # 如果原始W有sparse属性，为子图也创建sparse
    if hasattr(w, 'sparse') and hasattr(w.sparse, 'toarray'):
        import scipy.sparse as sp
        n = len(indices)
        data = []
        row = []
        col = []
        
        for i in range(n):
            for j in new_neighbors[i]:
                row.append(i)
                col.append(j)
                data.append(new_weights[i][j])
        
        subw.sparse = sp.csr_matrix((data, (row, col)), shape=(n, n))
    
    return subw


def spatial_constrained_postprocessing(initial_labels, w, data, target_clusters):
    """
    对标准KMeans结果进行空间后处理，合并空间相邻且属性相似的聚类
    
    参数:
    - initial_labels: 初始聚类标签
    - w: 空间权重矩阵
    - data: 数据矩阵
    - target_clusters: 目标聚类数量
    
    返回:
    - final_labels: 处理后的聚类标签
    """
    unique_labels = np.unique(initial_labels)
    if len(unique_labels) <= target_clusters:
        return initial_labels
    
    centers = {}
    for label in unique_labels:
        mask = (initial_labels == label)
        if np.any(mask):
            centers[label] = np.mean(data[mask], axis=0)
    
    cluster_adjacency = {}
    for i in range(len(initial_labels)):
        label_i = initial_labels[i]
        neighbors = w.neighbors[i]
        for j in neighbors:
            if j < len(initial_labels):
                label_j = initial_labels[j]
                if label_i != label_j:
                    if label_i not in cluster_adjacency:
                        cluster_adjacency[label_i] = set()
                    cluster_adjacency[label_i].add(label_j)
    
    for label in cluster_adjacency:
        cluster_adjacency[label] = list(cluster_adjacency[label])
    
    current_labels = initial_labels.copy()
    current_unique = np.unique(current_labels)
    max_iterations = 100
    iteration = 0
    
    while len(current_unique) > target_clusters and iteration < max_iterations:
        iteration += 1
        # 更复杂的合并策略，同时考虑特征距离和连通性
        best_pair = None
        min_distance = float('inf')
        
        for label_i in current_unique:
            if label_i in cluster_adjacency:
                neighbors = [n for n in cluster_adjacency[label_i] if n in current_unique]
                # 如果邻居列表为空但仍需合并
                if not neighbors and len(current_unique) > target_clusters:
                    # 找到特征空间最近的簇
                    min_feat_dist = float('inf')
                    closest_label = None
                    for label_j in current_unique:
                        if label_i != label_j and label_i in centers and label_j in centers:
                            dist = np.linalg.norm(centers[label_i] - centers[label_j])
                            if dist < min_feat_dist:
                                min_feat_dist = dist
                                closest_label = label_j
                    if closest_label is not None:
                        # 添加为虚拟邻居
                        neighbors.append(closest_label)
                        print(f"添加特征空间最近的簇 {closest_label} 作为簇 {label_i} 的虚拟邻居")

                for label_j in neighbors:
                    if label_i in centers and label_j in centers:
                        # 引入空间近似度加权的特征距离度量
                        feat_dist = np.linalg.norm(centers[label_i] - centers[label_j])
                        
                        # 估计空间近似度（如果可用）
                        space_weight = 1.0  # 默认权重
                        # 对于小簇赋予更高权重，避免孤立点成为独立簇
                        mask_i = (current_labels == label_i)
                        mask_j = (current_labels == label_j)
                        size_i = np.sum(mask_i)
                        size_j = np.sum(mask_j)
                        # 小簇的合并优先级更高
                        if size_i < 5 or size_j < 5:
                            space_weight = 0.7  # 提高合并概率
                        
                        # 计算加权距离
                        dist = feat_dist * space_weight
                        
                        if dist < min_distance:
                            min_distance = dist
                            best_pair = (label_i, label_j)
        
        if best_pair is None:
            if len(current_unique) >= 2:
                updated_adjacency = False
                for i, label_i in enumerate(current_unique):
                    for j, label_j in enumerate(current_unique):
                        if i != j:
                            mask_i = (current_labels == label_i)
                            mask_j = (current_labels == label_j)
                            indices_i = np.where(mask_i)[0]
                            indices_j = np.where(mask_j)[0]
                            if len(indices_i) > 0 and len(indices_j) > 0:
                                max_checks = min(100, len(indices_i) * len(indices_j))
                                check_count = 0
                                for idx_i in indices_i:
                                    for idx_j in indices_j:
                                        check_count += 1
                                        if check_count > max_checks or updated_adjacency:
                                            break
                                    if check_count > max_checks or updated_adjacency:
                                        break
                if updated_adjacency:
                    continue
                min_feature_dist = float('inf')
                closest_pair = None
                for i, label_i in enumerate(current_unique):
                    for j, label_j in enumerate(current_unique):
                        if i < j and label_i in centers and label_j in centers:
                            dist = np.linalg.norm(centers[label_i] - centers[label_j])
                            if dist < min_feature_dist:
                                min_feature_dist = dist
                                closest_pair = (label_i, label_j)
                if closest_pair:
                    best_pair = closest_pair
                else:
                    random_indices = np.random.choice(len(current_unique), 2, replace=False)
                    best_pair = (current_unique[random_indices[0]], current_unique[random_indices[1]])
            else:
                break
        
        label_keep, label_merge = best_pair
        mask_merge = (current_labels == label_merge)
        current_labels[mask_merge] = label_keep
        
        if label_keep in centers and label_merge in centers:
            mask_keep = (current_labels == label_keep)
            centers[label_keep] = np.mean(data[mask_keep], axis=0)
            if label_merge in centers:
                del centers[label_merge]
        
        current_unique = np.unique(current_labels)
    
    if len(current_unique) > target_clusters:
        print(f"警告: 空间后处理未能达到目标聚类数 {target_clusters}，当前聚类数 {len(current_unique)}")
    
    final_labels = np.zeros_like(current_labels)
    for i, label in enumerate(np.unique(current_labels)):
        final_labels[current_labels == label] = i
    
    return final_labels


def _perform_clustering(shap_features, n_clusters=3):
    """
    对SHAP特征执行聚类并计算相关统计信息
    
    参数:
    - shap_features: SHAP特征DataFrame
    - n_clusters: 聚类数量
    
    返回:
    - clusters: 聚类标签
    - centers: 聚类中心
    - standardized_features: 标准化后的特征
    """
    # 标准化特征
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(shap_features)
    
    # 执行K-means聚类，调整参数以提高聚类稳定性
    max_iter = 1000  # 增加最大迭代次数
    n_init = 30     # 增加启动次数
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=n_init, max_iter=max_iter)
    clusters = kmeans.fit_predict(standardized_features)
    centers = kmeans.cluster_centers_
    
    # 计算每个聚类的SHAP总影响
    abs_shap_sum = np.abs(shap_features).sum(axis=1)
    cluster_impacts = []
    
    for cluster in range(n_clusters):
        mask = (clusters == cluster)
        if np.any(mask):  # 确保有点属于这个聚类
            impact = np.mean(abs_shap_sum[mask])
            cluster_impacts.append((cluster, impact))
    
    # 按影响力排序（高->低）
    cluster_impacts.sort(key=lambda x: x[1], reverse=True)
    
    # 创建重映射：高影响->0, 中等影响->1, 低影响->2
    remapping = {old_idx: new_idx for new_idx, (old_idx, _) in enumerate(cluster_impacts)}
    
    # 应用重映射
    remapped_clusters = np.array([remapping[c] for c in clusters])
    
    return remapped_clusters, centers, standardized_features


def force_large_continuous_regions(labels, coords, w, X, target_clusters):
    """
    强制形成大块连续的聚类区域，进一步合并小聚类并优化空间连续性
    
    参数:
    - labels: 聚类标签
    - coords: 坐标数组
    - w: 空间权重矩阵
    - X: 特征矩阵
    - target_clusters: 目标聚类数量
    
    返回:
    - enhanced_labels: 强化后的聚类标签
    """
    print("  🔧 强制形成大块连续区域...")
    
    enhanced_labels = labels.copy()
    unique_labels, counts = np.unique(enhanced_labels, return_counts=True)
    
    # 计算每个聚类的特征中心
    cluster_centers = {}
    for label in unique_labels:
        mask = (enhanced_labels == label)
        if np.any(mask):
            cluster_centers[label] = np.mean(X[mask], axis=0)
    
    # 🔥 适度的最小聚类大小：每个聚类占合理比例
    min_region_size = len(enhanced_labels) // (target_clusters * 5)  # 每个聚类至少占1/(5*n_clusters)
    print(f"    适度最小区域大小: {min_region_size}")
    
    # 迭代合并小聚类到最近的大聚类
    max_iterations = 50
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        unique_labels, counts = np.unique(enhanced_labels, return_counts=True)
        
        # 找到最小的聚类
        small_clusters = [label for label, count in zip(unique_labels, counts) if count < min_region_size]
        
        if not small_clusters:
            break
            
        print(f"    迭代{iteration}: 发现{len(small_clusters)}个小聚类需要合并")
        
        # 逐个合并最小的聚类
        for small_label in small_clusters:
            small_mask = (enhanced_labels == small_label)
            small_indices = np.where(small_mask)[0]
            
            if len(small_indices) == 0:
                continue
                
            # 找到这个小聚类的空间邻居
            neighbor_labels = set()
            for idx in small_indices:
                if idx in w.neighbors:
                    for neighbor_idx in w.neighbors[idx]:
                        if neighbor_idx < len(enhanced_labels):
                            neighbor_label = enhanced_labels[neighbor_idx]
                            if neighbor_label != small_label:
                                neighbor_labels.add(neighbor_label)
            
            # 在邻居中找到最大的聚类
            best_neighbor = None
            max_neighbor_size = 0
            min_feature_distance = float('inf')
            
            for neighbor_label in neighbor_labels:
                neighbor_size = np.sum(enhanced_labels == neighbor_label)
                if neighbor_size > max_neighbor_size:
                    # 还要检查特征相似性
                    if (small_label in cluster_centers and 
                        neighbor_label in cluster_centers):
                        feature_dist = np.linalg.norm(
                            cluster_centers[small_label] - cluster_centers[neighbor_label]
                        )
                        if neighbor_size > max_neighbor_size or feature_dist < min_feature_distance:
                            max_neighbor_size = neighbor_size
                            min_feature_distance = feature_dist
                            best_neighbor = neighbor_label
            
            # 如果没找到合适的邻居，找特征最相似的聚类
            if best_neighbor is None and small_label in cluster_centers:
                min_dist = float('inf')
                for other_label in unique_labels:
                    if (other_label != small_label and 
                        other_label in cluster_centers and
                        np.sum(enhanced_labels == other_label) >= min_region_size):
                        dist = np.linalg.norm(
                            cluster_centers[small_label] - cluster_centers[other_label]
                        )
                        if dist < min_dist:
                            min_dist = dist
                            best_neighbor = other_label
            
            # 执行合并
            if best_neighbor is not None:
                enhanced_labels[small_mask] = best_neighbor
                print(f"      合并聚类{small_label}({np.sum(small_mask)}个点) → 聚类{best_neighbor}")
                
                # 更新聚类中心
                if best_neighbor in cluster_centers:
                    new_mask = (enhanced_labels == best_neighbor)
                    cluster_centers[best_neighbor] = np.mean(X[new_mask], axis=0)
                
                # 删除被合并的聚类中心
                if small_label in cluster_centers:
                    del cluster_centers[small_label]
    
    # 重新编号以确保连续性
    final_unique = np.unique(enhanced_labels)
    label_mapping = {old: new for new, old in enumerate(final_unique)}
    final_labels = np.array([label_mapping[label] for label in enhanced_labels])
    
    print(f"    ✅ 大块区域强化完成，最终聚类数: {len(np.unique(final_labels))}")
    print(f"    最终聚类大小分布: {dict(zip(*np.unique(final_labels, return_counts=True)))}")
    
    return final_labels


def fix_spatial_discontinuity(labels, coords, w, X):
    """
    修复空间聚类中的不连续性问题，确保聚类结果具有更好的空间连续性
    
    参数:
    - labels: 原始聚类标签
    - coords: 坐标数组
    - w: 空间权重矩阵
    - X: 特征矩阵
    
    返回:
    - fixed_labels: 修复后的聚类标签
    """
    print("  🔧 修复空间不连续性...")
    
    fixed_labels = labels.copy()
    n_clusters = len(np.unique(labels))
    
    # 检测并修复孤立点
    isolated_count = 0
    for i in range(len(labels)):
        current_label = labels[i]
        neighbors = w.neighbors[i] if i in w.neighbors else []
        
        # 计算邻居的标签分布
        neighbor_labels = [labels[j] for j in neighbors if j < len(labels)]
        
        if neighbor_labels:
            # 如果当前点的标签与所有邻居都不同，则为孤立点
            if current_label not in neighbor_labels:
                # 找到邻居中最频繁的标签
                from collections import Counter
                most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
                
                # 计算与最常见邻居标签的特征相似性
                same_label_indices = [j for j in range(len(labels)) if labels[j] == most_common_label]
                if same_label_indices:
                    # 计算特征距离
                    point_feature = X[i]
                    cluster_center = np.mean(X[same_label_indices], axis=0)
                    feature_distance = np.linalg.norm(point_feature - cluster_center)
                    
                    # 如果特征距离不太大，则重新分配标签
                    threshold = np.std(X) * 2  # 2倍标准差作为阈值
                    if feature_distance < threshold:
                        fixed_labels[i] = most_common_label
                        isolated_count += 1
    
    # 🔥 仅处理真正的孤立点（1-2个点的聚类），保持SHAP值分布的自然性
    unique_labels, counts = np.unique(fixed_labels, return_counts=True)
    # 只合并真正的孤立点
    min_cluster_size = 3  # 只合并小于3个点的聚类
    print(f"  仅处理真正的孤立点，最小大小: {min_cluster_size}")
    print(f"  当前聚类大小分布: {dict(zip(unique_labels, counts))}")
    
    for label, count in zip(unique_labels, counts):
        if count < min_cluster_size:
            # 找到特征最相似的聚类进行合并
            label_indices = np.where(fixed_labels == label)[0]
            if len(label_indices) == 0:
                continue
                
            # 计算当前小聚类的平均特征
            small_cluster_features = np.mean(X[label_indices], axis=0)
            
            min_feature_distance = float('inf')
            closest_label = None
            
            for other_label in unique_labels:
                if other_label != label and np.sum(fixed_labels == other_label) >= min_cluster_size:
                    other_indices = np.where(fixed_labels == other_label)[0]
                    other_features = np.mean(X[other_indices], axis=0)
                    
                    # 使用特征相似性而不是空间距离
                    feature_distance = np.linalg.norm(small_cluster_features - other_features)
                    if feature_distance < min_feature_distance:
                        min_feature_distance = feature_distance
                        closest_label = other_label
            
            if closest_label is not None:
                fixed_labels[fixed_labels == label] = closest_label
                print(f"    合并孤立点聚类 {label} 到特征相似的聚类 {closest_label} (大小: {count})")
    
    # 重新编号标签以确保连续性
    unique_labels = np.unique(fixed_labels)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    final_labels = np.array([label_mapping[label] for label in fixed_labels])
    
    if isolated_count > 0:
        print(f"  ✓ 修复了 {isolated_count} 个孤立点，最终聚类数: {len(np.unique(final_labels))}")
    
    return final_labels


def perform_spatial_clustering(shap_features, coords_df, n_clusters=3, grid_disk_k=1):
    """
    使用AgglomerativeClustering执行空间约束聚类
    
    参数:
    - shap_features: DataFrame格式的SHAP特征
    - coords_df: 包含latitude, longitude和可选h3_index的DataFrame
    - n_clusters: 聚类数量
    - grid_disk_k: H3 grid_disk半径（邻居数）
    
    返回:
    - labels: 聚类标签数组
    - X: 标准化并加权后的特征矩阵
    """
    # 🔥 检查是否强制使用KMeans
    import os
    force_kmeans = os.getenv('FORCE_KMEANS_CLUSTERING', 'false').lower() == 'true'
    
    if force_kmeans:
        print(f"🎯 强制使用普通KMeans聚类(n_clusters={n_clusters})，跳过空间约束")
    else:
        print(f"对SHAP特征进行空间约束聚类(n_clusters={n_clusters}, grid_disk_k={grid_disk_k})")
    
    try:
        # 🔧 修复：确保shap_features是DataFrame
        if isinstance(shap_features, np.ndarray):
            # 如果是numpy数组，转换为DataFrame
            if hasattr(coords_df, 'columns') and len(coords_df.columns) > 3:
                # 尝试从coords_df推断特征名
                feature_cols = [col for col in coords_df.columns if col not in ['h3_index', 'latitude', 'longitude']]
                if len(feature_cols) == shap_features.shape[1]:
                    shap_features = pd.DataFrame(shap_features, columns=feature_cols)
                else:
                    shap_features = pd.DataFrame(shap_features, columns=[f'feature_{i}' for i in range(shap_features.shape[1])])
            else:
                shap_features = pd.DataFrame(shap_features, columns=[f'feature_{i}' for i in range(shap_features.shape[1])])
        
        # 标准化SHAP特征
        scaler = StandardScaler()
        X = scaler.fit_transform(shap_features)
        
        # 🔧 修复：根据平均绝对SHAP值计算特征重要性权重
        if hasattr(shap_features, 'values'):
            # pandas DataFrame
            importance = np.abs(shap_features).mean(axis=0).values
        else:
            # numpy array
            importance = np.abs(shap_features).mean(axis=0)
        
        # 归一化最大值为1
        if importance.max() > 0:
            importance_norm = importance / importance.max()
        else:
            importance_norm = np.ones_like(importance)
        
        # 对标准化后的特征按列加权，使平均SHAP值更高的特征在距离计算中更重要
        X = X * importance_norm
        
        # 🔥 如果强制使用KMeans，直接跳到KMeans分支
        if force_kmeans:
            print("🚀 直接使用KMeans聚类，基于SHAP值自然分布")
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                print(f"✅ KMeans聚类完成，共生成{len(np.unique(labels))}个聚类")
                
                # 输出聚类分布统计
                unique_labels, counts = np.unique(labels, return_counts=True)
                for i, (label, count) in enumerate(zip(unique_labels, counts)):
                    percentage = count / len(labels) * 100
                    print(f"  聚类{label}: {count}个网格 ({percentage:.1f}%)")
                
                return labels, X
                
            except Exception as e:
                print(f"❌ KMeans聚类失败: {e}")
                # 继续执行原有逻辑作为备份
                pass
        
        # 构建空间权重，基于坐标
        coords = coords_df[['latitude', 'longitude']].values
        h3_indices = coords_df['h3_index'].values if 'h3_index' in coords_df.columns else None
        
        try:
            # 🔥 根据grid_disk_k直接设置邻居数量，不强制最小值
            if grid_disk_k == 1:
                enhanced_k = 6   # res7: 最小空间约束
            elif grid_disk_k == 2:
                enhanced_k = 12  # res6: 中等空间约束
            else:  # grid_disk_k == 3
                enhanced_k = 18  # res5: 较强空间约束
            print(f"  根据grid_disk_k={grid_disk_k}使用空间邻居数：{enhanced_k}")
            w = create_optimized_spatial_weights(coords, k=enhanced_k, h3_indices=h3_indices)
            conn = w.sparse
            print(f"  创建空间权重成功，平均邻居数: {w.mean_neighbors:.1f}")
        except Exception as e:
            print(f"创建空间权重失败: {e}，将使用无约束聚类")
            conn = None
        
        # 🔧 使用标准空间约束聚类
        print("开始执行标准空间约束层次聚类...")
        if conn is not None:
            try:
                # 🎯 使用标准ward linkage
                model = AgglomerativeClustering(
                    n_clusters=n_clusters, 
                    connectivity=conn, 
                    linkage='ward'  # 使用标准ward linkage
                )
                labels = model.fit_predict(X)
                print(f"标准空间约束聚类完成，共生成{len(np.unique(labels))}个聚类")
                
                # 🔥 仅修复明显的孤立点，保持自然分布
                labels = fix_spatial_discontinuity(labels, coords, w, X)
                
            except Exception as e:
                print(f"空间约束聚类失败: {e}，使用无约束聚类")
                try:
                    # 回退到无约束聚类
                    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                    labels = model.fit_predict(X)
                    print(f"无约束聚类完成，共生成{len(np.unique(labels))}个聚类")
                    
                    # 🔥 仅修复明显的孤立点
                    if w is not None:
                        labels = fix_spatial_discontinuity(labels, coords, w, X)
                    
                except Exception as e2:
                    print(f"无约束聚类也失败: {e2}，使用KMeans")
                    model = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = model.fit_predict(X)
                    print(f"KMeans聚类完成，共生成{len(np.unique(labels))}个聚类")
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            labels = model.fit_predict(X)
            print(f"无空间约束聚类完成，共生成{len(np.unique(labels))}个聚类")
            
            # 🔥 仅在有空间权重时轻度修复孤立点
            if 'w' in locals() and w is not None:
                labels = fix_spatial_discontinuity(labels, coords, w, X)
        
        # 检查是否有空聚类
        unique_labels = np.unique(labels)
        if len(unique_labels) < n_clusters:
            print(f"警告: 聚类数量({len(unique_labels)})小于目标数量({n_clusters})，将尝试重新平衡聚类")
            # 重新分配最大聚类的部分点以达到目标聚类数
            counts = np.bincount(labels)
            largest_cluster = np.argmax(counts)
            largest_indices = np.where(labels == largest_cluster)[0]
            
            if len(largest_indices) > n_clusters:
                # 从大簇中拆分出点创建新簇
                missing_clusters = n_clusters - len(unique_labels)
                points_per_new_cluster = len(largest_indices) // (missing_clusters + 1)
                
                # 使用KMeans在大簇内部再次聚类
                sub_X = X[largest_indices]
                kmeans = KMeans(n_clusters=missing_clusters+1, random_state=42)
                sub_labels = kmeans.fit_predict(sub_X)
                
                # 重新分配标签
                new_labels = labels.copy()
                for i, new_cluster_id in enumerate(range(len(unique_labels), n_clusters)):
                    new_labels[largest_indices[sub_labels == i]] = new_cluster_id
                    
                labels = new_labels
                print(f"聚类重新平衡完成，现有{len(np.unique(labels))}个聚类")
    
    except Exception as e:
        print(f"空间聚类过程中出错: {e}，将使用简单KMeans")
        # 使用简单KMeans作为后备方案
        try:
            # 🔧 修复：确保数据类型兼容
            if isinstance(shap_features, np.ndarray):
                X = StandardScaler().fit_transform(shap_features)
            else:
                X = StandardScaler().fit_transform(shap_features.values if hasattr(shap_features, 'values') else shap_features)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X)
            print(f"使用KMeans完成聚类，共生成{len(np.unique(labels))}个聚类")
            
            # 🔥 为KMeans聚类创建空间权重并轻度修复孤立点
            try:
                w_backup = create_optimized_spatial_weights(coords, k=8, h3_indices=h3_indices if 'h3_indices' in locals() else None)
                labels = fix_spatial_discontinuity(labels, coords, w_backup, X)
                print("KMeans聚类轻度修复完成")
            except Exception as e3:
                print(f"KMeans后处理失败: {e3}")
        except Exception as e2:
            print(f"KMeans聚类也失败: {e2}，使用随机分配")
            # 最后的后备方案：随机分配标签
            labels = np.random.randint(0, n_clusters, size=len(shap_features))
            X = StandardScaler().fit_transform(shap_features.values if hasattr(shap_features, 'values') else shap_features)
            print("使用随机分配完成聚类")
            
    return labels, X


__all__ = [
    'create_optimized_spatial_weights',
    'create_subgraph', 
    'spatial_constrained_postprocessing',
    '_solve_with_timeout',
    '_perform_clustering',
    'perform_spatial_clustering',
    'force_large_continuous_regions',
    'fix_spatial_discontinuity'
] 
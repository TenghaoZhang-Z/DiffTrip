import torch
import numpy as np
import networkx as nx
from typing import Dict, Tuple, List
import logging
# 移除了 cdist，改用自定义的 numpy 实现
import pandas as pd

logger = logging.getLogger(__name__)


class GlobalGraphBuilder:
    """构建全局空间邻接图"""

    def __init__(self, distance_threshold_km: float = 3.0):
        """
        初始化全局图构建器

        Args:
            distance_threshold_km: 距离阈值(km)，小于此值的POI会连接
        """
        self.distance_threshold_km = distance_threshold_km
        self.R = 6371.0  # 地球半径(km)

    def _haversine_matrix(self, coords: np.ndarray) -> np.ndarray:
        """
        使用向量化 Haversine 公式计算两两之间的球面距离

        Args:
            coords: numpy array, shape [N, 2], columns are [lat, lon]

        Returns:
            distance_matrix: numpy array, shape [N, N], unit: km
        """
        # 将角度转换为弧度
        coords_rad = np.radians(coords)
        lat = coords_rad[:, 0]
        lon = coords_rad[:, 1]

        # 利用广播机制计算差值
        # lat[:, np.newaxis] shape is (N, 1)
        # lat[np.newaxis, :] shape is (1, N)
        # 结果 dlat shape is (N, N)
        dlat = lat[:, np.newaxis] - lat[np.newaxis, :]
        dlon = lon[:, np.newaxis] - lon[np.newaxis, :]

        # Haversine 公式
        a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, np.newaxis]) * np.cos(lat[np.newaxis, :]) * np.sin(dlon / 2) ** 2

        # 限制数值在 [-1, 1] 范围内，防止浮点误差导致 arcsin 报错
        a = np.clip(a, 0, 1)

        c = 2 * np.arcsin(np.sqrt(a))

        return self.R * c

    def build_from_data(self, poi_data: pd.DataFrame) -> nx.Graph:
        """
        从POI数据构建全局图

        Args:
            poi_data: POI数据DataFrame，需包含poi_id, lat, lon列

        Returns:
            networkx.Graph: 全局空间图
        """
        logger.info("开始构建全局空间图...")

        # 提取POI ID和坐标
        poi_ids = poi_data['poi_id'].values
        coords = poi_data[['lat', 'lon']].values

        # 计算距离矩阵 (使用 Haversine 球面距离)
        # 注意：这里直接返回的是千米，不再需要乘以 111
        distance_km = self._haversine_matrix(coords)

        # 构建邻接矩阵
        adj_matrix = (distance_km <= self.distance_threshold_km).astype(float)
        np.fill_diagonal(adj_matrix, 0)  # 移除自环

        # 创建networkx图
        G = nx.from_numpy_array(adj_matrix)

        # 添加POI属性
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['poi_id'] = int(poi_ids[i])
            G.nodes[node]['lat'] = float(coords[i, 0])
            G.nodes[node]['lon'] = float(coords[i, 1])

            # 如果有类别信息，也添加
            if 'category' in poi_data.columns:
                G.nodes[node]['category'] = poi_data.iloc[i]['category']
            if 'cat_id' in poi_data.columns:
                G.nodes[node]['cat_id'] = int(poi_data.iloc[i]['cat_id'])

        # 计算图统计信息
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = 2 * num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0

        logger.info(f"全局图构建完成: {num_nodes}节点, {num_edges}边, 密度: {density:.4f}")

        return G

    def get_graph_statistics(self, G: nx.Graph) -> Dict:
        """获取图统计信息"""
        if G.number_of_nodes() == 0:
            return {}

        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_clustering': nx.average_clustering(G),
            'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
            'is_connected': nx.is_connected(G) if G.number_of_nodes() > 0 else False,
            'num_components': nx.number_connected_components(G)
        }

        return stats

    def save_to_csv(self, G: nx.Graph, output_path: str):
        """将图保存为CSV文件（边列表）"""
        edges = []
        for u, v, data in G.edges(data=True):
            edges.append({
                'source_poi': G.nodes[u]['poi_id'],
                'target_poi': G.nodes[v]['poi_id'],
                'source_lat': G.nodes[u]['lat'],
                'source_lon': G.nodes[u]['lon'],
                'target_lat': G.nodes[v]['lat'],
                'target_lon': G.nodes[v]['lon']
            })

        if edges:
            edges_df = pd.DataFrame(edges)
            edges_df.to_csv(output_path, index=False)
            logger.info(f"全局图边列表已保存到: {output_path}")
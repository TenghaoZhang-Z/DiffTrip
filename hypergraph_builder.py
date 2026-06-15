import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict
import logging
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import warnings

logger = logging.getLogger(__name__)


class HypergraphBuilder:
    def __init__(self, k: int = 10, alpha: float = 0.85, max_iter: int = 100):
        self.k = k
        self.alpha = alpha
        self.max_iter = max_iter
        warnings.filterwarnings('ignore', category=RuntimeWarning)

    def build_user_hypergraphs(self, trajectories: pd.DataFrame) -> Dict[int, Dict]:
        logger.info("开始构建用户轨迹超图...")
        user_hypergraphs = {}
        grouped = trajectories.groupby('userID')
        for user_id, user_trajs in grouped:
            hyperedges = []
            for idx, traj in user_trajs.iterrows():
                poi_seq = self._parse_sequence(traj['poi_sequence'])
                if len(poi_seq) < 2:
                    continue
                hyperedge = {
                    'hyperedge_id': f"user{user_id}_traj{idx}",
                    'poi_seq': poi_seq,
                    'size': len(poi_seq),
                    'start_poi': int(traj['start_poi']),
                    'end_poi': int(traj['end_poi']),
                    'seq_len': int(traj['sequence_length'])
                }
                if 'cat_sequence' in traj:
                    hyperedge['cat_seq'] = self._parse_sequence(traj['cat_sequence'])
                if 'time_sequence' in traj:
                    time_seq = str(traj['time_sequence']).split(';')
                    hyperedge['time_seq'] = [float(t) for t in time_seq
                                             if t and t.strip() and t != 'nan']

                hyperedges.append(hyperedge)
            if hyperedges:
                user_hypergraphs[user_id] = {
                    'user_id': user_id,
                    'num_hyperedges': len(hyperedges),
                    'hyperedges': hyperedges,
                    'unique_pois': list(set(poi for he in hyperedges for poi in he['poi_seq']))
                }

        logger.info(f"用户超图构建完成: {len(user_hypergraphs)}个用户")
        return user_hypergraphs

    def build_importance_hypergraph(self, transition_matrix: np.ndarray,
                                    poi_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("开始构建重要性超图...")

        num_nodes = len(poi_ids)
        H = self._build_knn_hypergraph(transition_matrix)

        if H is None or H.shape[0] == 0 or H.shape[1] == 0:
            logger.warning("超图构建失败，返回均匀重要性")
            importance_dict = {poi_id: 0.5 for poi_id in poi_ids}
            return np.zeros((0, 0)), importance_dict

        node_importance = self._hypergraph_pagerank(H)

        if node_importance is None or len(node_importance) == 0:
            logger.warning("PageRank计算失败，返回均匀重要性")
            importance_dict = {poi_id: 0.5 for poi_id in poi_ids}
            return H, importance_dict

        importance_dict = {poi_id: float(importance)
                           for poi_id, importance in zip(poi_ids, node_importance)}

        logger.info(f"重要性超图构建完成: {num_nodes}个节点")
        logger.info(f"重要性统计 - 均值: {np.mean(node_importance):.4f}, "
                    f"中位数: {np.median(node_importance):.4f}, "
                    f"最大值: {np.max(node_importance):.4f}, "
                    f"最小值: {np.min(node_importance):.4f}")

        # 记录重要性大于0.9的节点
        high_importance_nodes = [(poi_ids[i], node_importance[i])
                                 for i in range(len(node_importance))
                                 if node_importance[i] > 0.9]
        if high_importance_nodes:
            logger.warning(f"有{len(high_importance_nodes)}个节点的重要性大于0.9:")
            for poi_id, imp in sorted(high_importance_nodes, key=lambda x: x[1], reverse=True)[:10]:
                logger.warning(f"  POI {poi_id}: {imp:.4f}")

        return H, importance_dict

    def _parse_sequence(self, seq_str: Any) -> List[int]:
        if pd.isna(seq_str):
            return []
        seq = str(seq_str).strip()
        if not seq:
            return []
        for sep in [';', ',', '|', ' ', '->', '→']:
            if sep in seq:
                return [int(x.strip()) for x in seq.split(sep) if x.strip()]
        try:
            return [int(seq)]
        except:
            return []

    def _build_knn_hypergraph(self, features: np.ndarray) -> np.ndarray:
        """构建KNN超图"""
        num_nodes = len(features)

        if num_nodes == 0:
            logger.warning("节点数为0，无法构建KNN超图")
            return None

        k = min(self.k, num_nodes - 1)
        if k <= 0:
            # 只有一个节点，返回单位矩阵
            H = np.eye(num_nodes, dtype=np.float32)
            return H

        try:
            # 计算余弦相似度
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1  # 避免除零
            features_normalized = features / norms

            # 计算余弦相似度矩阵
            similarity_matrix = np.dot(features_normalized, features_normalized.T)

            # 构建KNN超图
            H = np.zeros((num_nodes, num_nodes), dtype=np.float32)

            for i in range(num_nodes):
                # 获取当前节点的相似度向量
                similarities = similarity_matrix[i]

                # 排除自身
                similarities[i] = -1

                # 找到最相似的k个节点
                if k > 0:
                    # 使用argsort获取相似度最高的k个节点
                    top_k_indices = np.argsort(similarities)[-k:]
                else:
                    top_k_indices = []

                # 构建超边：包含当前节点和其k个最近邻
                hyperedge_nodes = [i] + list(top_k_indices)

                for node in hyperedge_nodes:
                    H[node, i] = 1.0

        except Exception as e:
            logger.error(f"KNN超图构建失败: {e}")
            # 返回一个简单的单位矩阵作为fallback
            H = np.eye(num_nodes, dtype=np.float32)

        return H

    def _hypergraph_pagerank(self, H: np.ndarray) -> np.ndarray:
        """在超图上计算PageRank - 修复版本"""
        if H is None or H.size == 0:
            logger.warning("超图为空，返回None")
            return None

        n_nodes, n_hyperedges = H.shape

        if n_nodes == 0 or n_hyperedges == 0:
            logger.warning("超图维度为0，返回均匀分布")
            return np.ones(n_nodes) / n_nodes

        logger.info(f"计算PageRank: {n_nodes}节点, {n_hyperedges}超边")

        # 节点度和超边度
        d_v = H.sum(axis=1)  # 每个节点属于多少超边
        d_e = H.sum(axis=0)  # 每个超边包含多少节点

        # 避免除零
        d_v = np.where(d_v == 0, 1, d_v)  # 将度为0的节点设为1
        d_e = np.where(d_e == 0, 1, d_e)  # 将度为0的超边设为1

        d_v_inv = 1.0 / d_v
        d_e_inv = 1.0 / d_e

        # 初始化PageRank - 使用均匀分布
        pr = np.ones(n_nodes) / n_nodes

        # 迭代计算
        for iteration in range(self.max_iter):
            # 节点 -> 超边
            edge_msg = H.T @ (pr * d_v_inv)

            # 超边 -> 节点
            node_msg = H @ (edge_msg * d_e_inv)

            # PageRank更新公式
            pr_new = self.alpha * node_msg + (1 - self.alpha) / n_nodes

            # 检查收敛
            diff = np.linalg.norm(pr_new - pr, 1)
            if diff < 1e-8:
                logger.info(f"PageRank在{iteration + 1}次迭代后收敛")
                break

            pr = pr_new

        # 记录PageRank原始统计
        logger.info(f"PageRank原始值 - 均值: {np.mean(pr):.6f}, "
                    f"标准差: {np.std(pr):.6f}, "
                    f"最大值: {np.max(pr):.6f}, "
                    f"最小值: {np.min(pr):.6f}")

        pr_exp = np.exp(pr - np.max(pr))  # 减去最大值避免数值溢出
        pr_normalized = pr_exp / np.sum(pr_exp)
        pr_min = np.min(pr)
        pr_max = np.max(pr)
        if pr_max > pr_min:
            pr_scaled = (pr - pr_min) / (pr_max - pr_min)
            # 避免log(0)
            pr_scaled = np.clip(pr_scaled, 1e-10, 1)
            pr_log = np.log(pr_scaled + 1)
            pr_normalized = pr_log / np.sum(pr_log)
        else:
            # 所有值相等，返回均匀分布
            pr_normalized = np.ones_like(pr) / len(pr)

        # 方法3: 使用分位数归一化 (更稳健)
        # 但这里我们使用方法1，因为它能保持相对顺序

        logger.info(f"归一化后 - 均值: {np.mean(pr_normalized):.6f}, "
                    f"最大值: {np.max(pr_normalized):.6f}, "
                    f"最小值: {np.min(pr_normalized):.6f}")

        return pr_normalized

    def get_hypergraph_statistics(self, user_hypergraphs: Dict) -> Dict:
        """获取超图统计信息"""
        if not user_hypergraphs:
            return {}

        stats = {
            'num_users': len(user_hypergraphs),
            'total_hyperedges': 0,
            'avg_hyperedge_size': 0.0,
            'max_hyperedge_size': 0,
            'min_hyperedge_size': float('inf'),
            'unique_pois': set()
        }

        hyperedge_sizes = []

        for user_id, hypergraph in user_hypergraphs.items():
            stats['total_hyperedges'] += hypergraph['num_hyperedges']

            for hyperedge in hypergraph['hyperedges']:
                size = hyperedge['size']
                hyperedge_sizes.append(size)
                stats['max_hyperedge_size'] = max(stats['max_hyperedge_size'], size)
                stats['min_hyperedge_size'] = min(stats['min_hyperedge_size'], size)
                stats['unique_pois'].update(hyperedge['poi_seq'])

        if hyperedge_sizes:
            stats['avg_hyperedge_size'] = np.mean(hyperedge_sizes)
            stats['min_hyperedge_size'] = stats['min_hyperedge_size'] if stats['min_hyperedge_size'] != float(
                'inf') else 0

        stats['unique_pois'] = len(stats['unique_pois'])

        return stats

    def save_importance_scores(self, importance_dict: Dict[int, float], output_path: str):
        """保存节点重要性分数"""
        if not importance_dict:
            logger.warning("重要性字典为空，跳过保存")
            return

        df = pd.DataFrame([
            {'poi_id': poi_id, 'importance': importance}
            for poi_id, importance in importance_dict.items()
        ])

        # 按重要性降序排序
        df = df.sort_values('importance', ascending=False)
        df.to_csv(output_path, index=False)

        # 添加统计信息
        logger.info(f"节点重要性统计:")
        logger.info(f"  总数: {len(df)}")
        logger.info(f"  均值: {df['importance'].mean():.6f}")
        logger.info(f"  标准差: {df['importance'].std():.6f}")
        logger.info(f"  最大值: {df['importance'].max():.6f}")
        logger.info(f"  最小值: {df['importance'].min():.6f}")

        # 检查是否有异常值
        high_imp = df[df['importance'] > 0.9]
        if len(high_imp) > 0:
            logger.warning(f"  有{len(high_imp)}个节点的重要性大于0.9")
            logger.warning(f"  最高重要性节点: {high_imp.iloc[0]['poi_id']} = {high_imp.iloc[0]['importance']:.6f}")

        logger.info(f"节点重要性已保存到: {output_path}")
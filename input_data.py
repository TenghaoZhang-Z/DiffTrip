import pandas as pd
import numpy as np
import torch
import os
import json
import random
from typing import Dict, List, Tuple, Any, Optional
import logging
from collections import defaultdict
import warnings
from copy import deepcopy
from datetime import datetime

# 导入自定义模块
from global_graph_builder import GlobalGraphBuilder
from hypergraph_builder import HypergraphBuilder

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class UserHypergraphAugmenter:
    """用户超图增强器 - 基于节点重要性进行超图增强"""

    def __init__(self, importance_dict: Dict[int, float], config: Dict):
        """
        初始化用户超图增强器

        Args:
            importance_dict: 节点重要性字典 {poi_id: importance_score}
            config: 配置参数
        """
        self.importance_dict = importance_dict
        self.config = config

        # 超图增强参数
        self.node_mask_prob = config.get('node_mask_prob', 0.2)  # 节点掩码概率
        self.edge_mask_prob = config.get('edge_mask_prob', 0.3)  # 边掩码概率
        self.node_add_prob = config.get('node_add_prob', 0.1)  # 节点添加概率
        self.edge_add_prob = config.get('edge_add_prob', 0.2)  # 边添加概率

        # 构建重要性相似性查找表
        self._build_similarity_lookup()

    def _build_similarity_lookup(self):
        """构建基于重要性的相似POI查找表"""
        self.similarity_dict = {}

        if not self.importance_dict:
            return

        # 按重要性分组
        importance_values = list(self.importance_dict.values())
        poi_ids = list(self.importance_dict.keys())

        # 计算重要性分位数
        if len(importance_values) >= 5:
            bins = np.percentile(importance_values, [0, 20, 40, 60, 80, 100])
        else:
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

        # 分组POI
        groups = defaultdict(list)
        for poi_id, importance in self.importance_dict.items():
            for i in range(len(bins) - 1):
                if bins[i] <= importance <= bins[i + 1]:
                    groups[i].append(poi_id)
                    break

        # 构建相似性字典
        for poi_id, importance in self.importance_dict.items():
            # 找到所属组
            target_group = None
            for i in range(len(bins) - 1):
                if bins[i] <= importance <= bins[i + 1]:
                    target_group = i
                    break

            if target_group is not None:
                # 同一组内的POI视为相似
                similar_pois = groups[target_group].copy()
                if poi_id in similar_pois:
                    similar_pois.remove(poi_id)

                if similar_pois:
                    # 按重要性差异排序
                    similar_pois.sort(key=lambda x: abs(
                        self.importance_dict[x] - importance
                    ))
                    self.similarity_dict[poi_id] = similar_pois

    def _get_mask_probability(self, poi_id: int) -> float:
        """根据节点重要性返回掩码概率"""
        importance = self.importance_dict.get(poi_id, 0.5)
        # 重要性越低，掩码概率越高
        return self.node_mask_prob + (1.0 - importance) * 0.1

    def _find_similar_poi(self, poi_id: int) -> Optional[int]:
        """找到重要性相似的POI"""
        if poi_id in self.similarity_dict and self.similarity_dict[poi_id]:
            return random.choice(self.similarity_dict[poi_id])
        return None

    def augment_user_hypergraph(self, user_hypergraph: Dict) -> Tuple[Dict, Dict]:
        """
        对用户超图进行增强，生成两个增强视图

        Args:
            user_hypergraph: 用户超图字典

        Returns:
            (augmented_view1, augmented_view2): 两个增强视图
        """
        # 复制原始超图结构
        view1 = deepcopy(user_hypergraph)
        view2 = deepcopy(user_hypergraph)

        # 对view1应用增强策略1：基于重要性的节点掩码
        self._importance_based_node_mask(view1)

        # 对view2应用增强策略2：基于重要性的边扰动
        self._importance_based_edge_perturb(view2)

        return view1, view2

    def _importance_based_node_mask(self, hypergraph: Dict):
        """基于重要性的节点掩码增强"""
        hyperedges = hypergraph.get('hyperedges', [])

        for hyperedge in hyperedges:
            poi_seq = hyperedge.get('poi_seq', [])

            # 对超边中的每个节点应用掩码
            new_poi_seq = []
            for poi_id in poi_seq:
                mask_prob = self._get_mask_probability(poi_id)

                if random.random() < mask_prob:
                    # 掩码这个节点
                    continue
                new_poi_seq.append(poi_id)

            # 更新超边
            hyperedge['poi_seq'] = new_poi_seq
            hyperedge['size'] = len(new_poi_seq)

    def _importance_based_edge_perturb(self, hypergraph: Dict):
        """基于重要性的边扰动增强"""
        hyperedges = hypergraph.get('hyperedges', [])

        for hyperedge in hyperedges:
            # 以一定概率添加相似节点
            if random.random() < self.node_add_prob and hyperedge['poi_seq']:
                # 随机选择一个节点
                idx = random.randint(0, len(hyperedge['poi_seq']) - 1)
                target_poi = hyperedge['poi_seq'][idx]
                similar_poi = self._find_similar_poi(target_poi)

                if similar_poi and similar_poi not in hyperedge['poi_seq']:
                    # 在随机位置插入相似节点
                    insert_pos = random.randint(0, len(hyperedge['poi_seq']))
                    hyperedge['poi_seq'].insert(insert_pos, similar_poi)
                    hyperedge['size'] += 1

            # 以一定概率移除不重要节点
            if random.random() < self.edge_mask_prob and len(hyperedge['poi_seq']) > 2:
                # 找到重要性最低的节点
                poi_importance = [
                    (poi, self.importance_dict.get(poi, 0.5))
                    for poi in hyperedge['poi_seq']
                ]
                poi_importance.sort(key=lambda x: x[1])

                # 移除重要性最低的节点
                if poi_importance:
                    min_poi = poi_importance[0][0]
                    hyperedge['poi_seq'] = [p for p in hyperedge['poi_seq'] if p != min_poi]
                    hyperedge['size'] = len(hyperedge['poi_seq'])


class DataProcessor:
    """主数据处理类 - 包含用户超图对比学习的数据处理"""

    def __init__(self, config: Dict = None):
        """
        初始化数据处理器

        Args:
            config: 配置参数
        """
        self.config = config or {}

        # 设置参数
        self.data_dir = self.config.get('data_dir', 'data/Glas')
        self.output_dir = self.config.get('output_dir', f'{self.data_dir}/processed')

        # 数据集划分比例
        self.val_ratio = self.config.get('val_ratio', 0.2)
        self.test_ratio = self.config.get('test_ratio', 0.1)

        # 用户超图对比学习相关参数
        self.user_hypergraph_cl_enabled = self.config.get('user_hypergraph_cl_enabled', True)
        self.num_aug_views_per_user = self.config.get('num_aug_views_per_user', 2)

        # 图构建参数
        self.distance_threshold = self.config.get('distance_threshold_km', 3.0)
        self.k_neighbors = self.config.get('k_neighbors', 10)

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        # 数据存储
        self.poi_data = None
        self.trajectories = None
        self.transition_matrix = None
        self.poi_ids = None
        self.poi_to_cat_id = {}
        self.num_cats = 0

        # [NEW] 新增的物理与逻辑约束张量
        self.distance_matrix = None
        self.time_cat_matrix = None
        self.poi_to_cat_tensor = None

        # 图结构
        self.global_graph = None
        self.importance_hypergraph = None
        self.importance_dict = None

        # 用户超图
        self.user_hypergraphs = None

        # 数据集
        self.train_data = []
        self.val_data = []
        self.test_data = []

        # 增强用户超图对（用于图级对比学习）
        self.augmented_user_hypergraphs = {}

        # 统计信息
        self.stats = {}

    def process(self) -> Dict:
        """主处理流程"""
        logger.info("开始数据处理流程")

        # 执行处理步骤
        steps = [
            self._load_raw_data,
            self._preprocess_data,
            self._build_and_save_transition_matrix,
            self._build_distance_matrix,  # [NEW] 构建空间距离矩阵
            self._build_time_category_matrix,  # [NEW] 构建时序-类别矩阵
            self._build_poi_to_cat_mapping,  # [NEW] 构建 POI-类别 映射张量
            self._build_global_graph,
            self._build_importance_hypergraph,
            self._build_user_hypergraphs,
            self._generate_dataset,
            self._generate_augmented_user_hypergraphs,
            self._save_results
        ]

        for step in steps:
            step_name = step.__name__.replace('_', ' ').title()
            logger.info(f"步骤: {step_name}")
            step()

        logger.info("数据处理完成")
        return self._get_results()

    def _load_raw_data(self):
        """加载原始数据文件"""
        # 加载POI数据
        poi_path = f"{self.data_dir}/poi-Osak.csv"
        if not os.path.exists(poi_path):
            raise FileNotFoundError(f"POI文件不存在: {poi_path}")

        self.poi_data = pd.read_csv(poi_path)

        # 标准化列名
        col_rename = {
            'poiID': 'poi_id', 'poiCat': 'category',
            'poiLon': 'lon', 'poiLat': 'lat', 'catID': 'cat_id'
        }
        self.poi_data.rename(columns={k: v for k, v in col_rename.items() if k in self.poi_data.columns}, inplace=True)

        # 加载轨迹数据
        traj_files = ["sequences_inf.csv", "sequences_with_users.csv"]
        for f in traj_files:
            if os.path.exists(f"{self.data_dir}/{f}"):
                self.trajectories = pd.read_csv(f"{self.data_dir}/{f}")
                break
        else:
            raise FileNotFoundError("未找到轨迹文件")

        # 加载转移矩阵
        trans_path = f"{self.data_dir}/transition_matrix.csv"
        if os.path.exists(trans_path):
            trans_df = pd.read_csv(trans_path, index_col=0)
            self.transition_matrix = trans_df.values.astype(np.float32)
            self.poi_ids = trans_df.index.tolist()

        logger.info(f"POI数据: {len(self.poi_data)}行")
        logger.info(f"轨迹数据: {len(self.trajectories)}行")
        logger.info(
            f"转移矩阵 (Raw CSV): {self.transition_matrix.shape if self.transition_matrix is not None else '无'}")

    def _preprocess_data(self):
        """数据预处理"""
        # POI数据预处理
        self.poi_data['poi_id'] = self.poi_data['poi_id'].astype(int)
        self.poi_data['lat'] = self.poi_data['lat'].astype(float)
        self.poi_data['lon'] = self.poi_data['lon'].astype(float)

        # 构建POI到类别的映射
        cats = set()
        for _, row in self.poi_data.iterrows():
            if 'cat_id' in row and pd.notna(row['cat_id']):
                cid = int(row['cat_id'])
                self.poi_to_cat_id[row['poi_id']] = cid
                cats.add(cid)

        self.num_cats = max(cats) + 1 if cats else 0

        # 轨迹数据预处理
        if 'sequence_length' not in self.trajectories.columns:
            self.trajectories['sequence_length'] = self.trajectories['poi_sequence'].apply(
                lambda x: len(str(x).split(';')) if pd.notna(x) else 0
            )

        # 过滤无效轨迹
        initial_len = len(self.trajectories)
        self.trajectories = self.trajectories[
            (self.trajectories['sequence_length'] >= 2) &
            self.trajectories[['start_poi', 'end_poi']].notna().all(axis=1)
            ].copy()

        # 提取POI ID
        if not self.poi_ids:
            all_pois = set()
            for seq in self.trajectories['poi_sequence']:
                if pd.isna(seq): continue
                pois = [int(p) for p in str(seq).split(';') if p.strip()]
                all_pois.update(pois)
            self.poi_ids = sorted(all_pois)

        logger.info(f"轨迹过滤: {initial_len} -> {len(self.trajectories)}")
        logger.info(f"POI数量: {len(self.poi_ids)}")
        logger.info(f"类别数量: {self.num_cats}")

    def _build_and_save_transition_matrix(self):
        """构建并保存标准化的转移概率矩阵"""
        logger.info("开始构建并保存转移概率矩阵...")
        if not self.poi_ids:
            logger.warning("未找到POI ID列表，无法构建转移矩阵")
            return

        max_poi_id = max(self.poi_ids)
        num_nodes = max_poi_id + 1

        counts = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        count_valid_pairs = 0

        for _, row in self.trajectories.iterrows():
            seq_str = row['poi_sequence']
            if pd.isna(seq_str): continue
            pois = [int(p) for p in str(seq_str).split(';') if p.strip()]

            if len(pois) < 2: continue
            for i in range(len(pois) - 1):
                u, v = pois[i], pois[i + 1]
                if u < num_nodes and v < num_nodes:
                    counts[u, v] += 1
                    count_valid_pairs += 1

        epsilon = 1e-5
        counts += epsilon
        counts[0, :] = epsilon
        counts[0, 0] = 1.0

        row_sums = counts.sum(dim=1, keepdim=True)
        probs = counts / row_sums

        output_path = f"{self.output_dir}/transition_probs.pt"
        torch.save(probs, output_path)
        logger.info(f"转移概率矩阵已保存: {output_path}")

    # ================= [NEW] 物理与逻辑约束构建区块 =================

    def _build_distance_matrix(self):
        """
        [NEW] 构建基于球面距离（Haversine）的全局物理距离矩阵
        """
        logger.info("构建全局空间距离矩阵...")
        max_poi_id = max(self.poi_ids) if self.poi_ids else 0
        num_nodes = max_poi_id + 1

        # 将 POI 坐标映射为 Numpy 数组，默认 0.0 (防止 Padding ID 报错)
        coords = np.zeros((num_nodes, 2))
        for _, row in self.poi_data.iterrows():
            poi_id = int(row['poi_id'])
            if poi_id < num_nodes:
                coords[poi_id] = [float(row['lat']), float(row['lon'])]

        # 向量化 Haversine 距离计算
        coords_rad = np.radians(coords)
        lat = coords_rad[:, 0]
        lon = coords_rad[:, 1]

        dlat = lat[:, np.newaxis] - lat[np.newaxis, :]
        dlon = lon[:, np.newaxis] - lon[np.newaxis, :]

        a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, np.newaxis]) * np.cos(lat[np.newaxis, :]) * np.sin(dlon / 2) ** 2
        a = np.clip(a, 0, 1)  # 防御性编程，避免浮点溢出
        c = 2 * np.arcsin(np.sqrt(a))
        R = 6371.0  # 地球半径 (km)

        distance_km = R * c
        self.distance_matrix = torch.tensor(distance_km, dtype=torch.float32)
        logger.info(f"空间距离矩阵构建完成，Shape: {self.distance_matrix.shape}")

    def _build_time_category_matrix(self):
        """
        [NEW] 构建时序-类别活跃度约束矩阵 P(Category | TimeSlot)
        """
        logger.info("构建时序-类别活跃度矩阵...")
        num_cats = self.num_cats
        # 48 个时间槽 (每半小时一个)
        counts = torch.zeros((48, num_cats), dtype=torch.float32)

        # 统计频次
        for _, row in self.trajectories.iterrows():
            seq_data = self._parse_sequences(row)
            time_slots = seq_data['time_slots']
            cat_seq = seq_data['cat_seq']

            for t, c in zip(time_slots, cat_seq):
                if 0 <= t < 48 and 0 <= c < num_cats:
                    counts[t, c] += 1

        # Add-epsilon 平滑与对数概率化
        epsilon = 1e-5
        counts += epsilon
        row_sums = counts.sum(dim=1, keepdim=True)
        probs = counts / row_sums

        # 存储为对数概率空间
        self.time_cat_matrix = torch.log(probs)
        logger.info(f"时序-类别矩阵构建完成，Shape: {self.time_cat_matrix.shape}")

    def _build_poi_to_cat_mapping(self):
        """
        [NEW] 构建 POI_ID -> Category_ID 的映射张量 (用于解码期广播)
        """
        logger.info("构建 POI 到类别的映射张量...")
        max_poi_id = max(self.poi_ids) if self.poi_ids else 0
        num_nodes = max_poi_id + 1

        poi_to_cat = torch.zeros(num_nodes, dtype=torch.long)
        for poi_id, cat_id in self.poi_to_cat_id.items():
            if 0 <= poi_id < num_nodes:
                poi_to_cat[poi_id] = cat_id

        self.poi_to_cat_tensor = poi_to_cat
        logger.info(f"POI映射张量构建完成，Shape: {self.poi_to_cat_tensor.shape}")

    # ====================================================================

    def _build_global_graph(self):
        """构建全局空间图"""
        builder = GlobalGraphBuilder(distance_threshold_km=self.distance_threshold)
        self.global_graph = builder.build_from_data(self.poi_data)
        self.stats.update(builder.get_graph_statistics(self.global_graph))
        logger.info(f"全局图: {self.global_graph.number_of_nodes()}节点, {self.global_graph.number_of_edges()}边")

    def _build_importance_hypergraph(self):
        """构建重要性超图并计算重要性"""
        if self.transition_matrix is None:
            self.importance_dict = {}
            return

        builder = HypergraphBuilder(k=self.k_neighbors, alpha=0.85, max_iter=100)
        H, self.importance_dict = builder.build_importance_hypergraph(self.transition_matrix, self.poi_ids)
        self.importance_hypergraph = H

    def _build_user_hypergraphs(self):
        """构建用户超图"""
        if self.trajectories is None:
            return

        builder = HypergraphBuilder(k=self.k_neighbors, alpha=0.85, max_iter=100)
        self.user_hypergraphs = builder.build_user_hypergraphs(self.trajectories)

        stats = builder.get_hypergraph_statistics(self.user_hypergraphs)
        self.stats.update(stats)

    def _timestamp_to_time_slot(self, timestamp: float) -> int:
        """将Unix时间戳转换为48小时时间槽"""
        try:
            dt = datetime.fromtimestamp(timestamp)
            minutes = dt.hour * 60 + dt.minute
            time_slot = min(47, minutes // 30)
            return time_slot
        except:
            return 0

    def _parse_sequences(self, traj_row: pd.Series) -> Dict:
        """解析轨迹的三个序列"""

        def parse_seq(seq_str, dtype=float):
            if pd.isna(seq_str): return []
            return [dtype(p.strip()) for p in str(seq_str).split(';') if p.strip()]

        poi_seq = parse_seq(traj_row['poi_sequence'], int)
        cat_seq = parse_seq(traj_row.get('cat_sequence'), int) if 'cat_sequence' in traj_row else []
        raw_time = parse_seq(traj_row.get('time_sequence'), float) if 'time_sequence' in traj_row else []

        time_slots = []
        if raw_time:
            for ts in raw_time:
                time_slots.append(self._timestamp_to_time_slot(ts))
        elif poi_seq:
            seq_len = len(poi_seq)
            time_slots = list(np.linspace(0, 47, seq_len, dtype=int)) if seq_len > 1 else [0]

        if not cat_seq and poi_seq and self.poi_to_cat_id:
            cat_seq = [self.poi_to_cat_id.get(poi, 0) for poi in poi_seq]

        if len(cat_seq) < len(poi_seq):
            cat_seq = cat_seq + [0] * (len(poi_seq) - len(cat_seq))

        return {'poi_seq': poi_seq, 'cat_seq': cat_seq, 'time_slots': time_slots}

    def _create_sample(self, idx: int, traj: pd.Series, is_augmented: bool = False) -> Optional[Dict]:
        """创建单个样本"""
        seq_data = self._parse_sequences(traj)
        if len(seq_data['poi_seq']) < 2:
            return None

        return {
            'sample_id': f"{'aug_' if is_augmented else ''}traj_{idx}",
            'user_id': int(traj['userID']),
            'start_poi': int(traj['start_poi']),
            'end_poi': int(traj['end_poi']),
            'expected_length': int(traj.get('sequence_length', len(seq_data['poi_seq']))),
            'poi_sequence': ';'.join(map(str, seq_data['poi_seq'])),
            'cat_sequence': ';'.join(map(str, seq_data['cat_seq'])),
            'time_sequence': ';'.join(map(str, seq_data['time_slots'])),
            'start_time': seq_data['time_slots'][0] if seq_data['time_slots'] else 0,
            'end_time': seq_data['time_slots'][-1] if seq_data['time_slots'] else 0,
            'is_augmented': is_augmented
        }

    def _generate_dataset(self):
        """生成训练/验证/测试集"""
        samples = [self._create_sample(i, row) for i, row in self.trajectories.iterrows()]
        samples = [s for s in samples if s]

        user_samples = defaultdict(list)
        for s in samples:
            user_samples[s['user_id']].append(s)

        users = list(user_samples.keys())
        random.shuffle(users)

        n_users = len(users)
        train_end = int(n_users * (1 - self.val_ratio - self.test_ratio))
        val_end = int(n_users * (1 - self.test_ratio))

        train_users = set(users[:train_end])
        val_users = set(users[train_end:val_end])
        test_users = set(users[val_end:])

        self.train_data = [s for s in samples if s['user_id'] in train_users]
        self.val_data = [s for s in samples if s['user_id'] in val_users]
        self.test_data = [s for s in samples if s['user_id'] in test_users]
        random.shuffle(self.train_data)

    def _generate_augmented_user_hypergraphs(self):
        """生成用于图级对比学习的增强用户超图"""
        if not self.user_hypergraph_cl_enabled or not self.user_hypergraphs or not self.importance_dict:
            return

        augmenter = UserHypergraphAugmenter(self.importance_dict, self.config)
        train_user_ids = set([s['user_id'] for s in self.train_data])

        for user_id, user_hypergraph in self.user_hypergraphs.items():
            if user_id in train_user_ids:
                view1, view2 = augmenter.augment_user_hypergraph(user_hypergraph)
                self.augmented_user_hypergraphs[user_id] = {
                    'original': user_hypergraph,
                    'view1': view1,
                    'view2': view2
                }
        self._convert_hypergraphs_to_edge_index()

    def _convert_hypergraphs_to_edge_index(self):
        """将用户超图转换为边索引格式"""
        if not self.augmented_user_hypergraphs:
            return

        poi_id_to_idx = {poi_id: idx for idx, poi_id in enumerate(self.poi_ids)}

        for user_id, hypergraphs in self.augmented_user_hypergraphs.items():
            for view_type in ['original', 'view1', 'view2']:
                hypergraph = hypergraphs[view_type]
                hyperedges = hypergraph.get('hyperedges', [])
                edge_index = []

                for edge_idx, hyperedge in enumerate(hyperedges):
                    poi_seq = hyperedge.get('poi_seq', [])
                    for poi_id in poi_seq:
                        if poi_id in poi_id_to_idx:
                            node_idx = poi_id_to_idx[poi_id]
                            edge_index.append([node_idx, edge_idx])

                if edge_index:
                    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t()
                else:
                    edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
                hypergraphs[f'{view_type}_edge_index'] = edge_index_tensor

    def _save_results(self):
        """保存处理结果"""
        logger.info("保存处理结果...")

        datasets = {'train': self.train_data, 'val': self.val_data, 'test': self.test_data}
        req_cols = ['sample_id', 'user_id', 'start_poi', 'end_poi', 'expected_length',
                    'start_time', 'end_time', 'poi_sequence', 'cat_sequence',
                    'time_sequence', 'is_augmented']

        for name, data in datasets.items():
            if not data: continue
            df = pd.DataFrame(data)
            for col in req_cols:
                if col not in df.columns:
                    df[col] = ''
            df = df.reindex(columns=req_cols)
            df['is_augmented'] = df.get('is_augmented', False)
            df.to_csv(f"{self.output_dir}/{name}.csv", index=False)

        if self.importance_dict:
            with open(f"{self.output_dir}/node_importance.json", 'w', encoding='utf-8') as f:
                json.dump(self.importance_dict, f, indent=2, ensure_ascii=False)

        if self.augmented_user_hypergraphs:
            train_user_hypergraphs = {}
            for user_id, hypergraphs in self.augmented_user_hypergraphs.items():
                train_user_hypergraphs[user_id] = {
                    'original_edge_index': hypergraphs.get('original_edge_index',
                                                           torch.empty((2, 0), dtype=torch.long)),
                    'view1_edge_index': hypergraphs.get('view1_edge_index', torch.empty((2, 0), dtype=torch.long)),
                    'view2_edge_index': hypergraphs.get('view2_edge_index', torch.empty((2, 0), dtype=torch.long))
                }
            torch.save(train_user_hypergraphs, f"{self.output_dir}/user_hypergraphs.pt")

        self._save_graph_structures()

        # [NEW] 保存三维软约束的预计算张量
        if self.distance_matrix is not None:
            torch.save(self.distance_matrix, f"{self.output_dir}/distance_matrix.pt")
        if self.time_cat_matrix is not None:
            torch.save(self.time_cat_matrix, f"{self.output_dir}/time_cat_matrix.pt")
        if self.poi_to_cat_tensor is not None:
            torch.save(self.poi_to_cat_tensor, f"{self.output_dir}/poi_to_cat.pt")
        logger.info("  [NEW] 物理与逻辑约束查询张量已成功持久化保存。")

        self._save_config_and_stats()

    def _save_graph_structures(self):
        """保存图结构为PyTorch张量格式"""
        if self.global_graph:
            edge_index, edge_weights = [], []
            for u, v, data in self.global_graph.edges(data=True):
                edge_index.append([u, v])
                edge_weights.append(data.get('weight', 1.0))

            if edge_index:
                edge_tensor = torch.tensor(edge_index, dtype=torch.long).t()
                weight_tensor = torch.tensor(edge_weights, dtype=torch.float)
                torch.save({
                    'edge_index': edge_tensor,
                    'edge_weights': weight_tensor,
                    'num_nodes': self.global_graph.number_of_nodes()
                }, f"{self.output_dir}/global_graph.pt")

        if self.importance_hypergraph is not None:
            H = self.importance_hypergraph
            node_indices, hyperedge_indices = np.where(H > 0)
            hyper_edge_index = torch.tensor(np.stack([node_indices, hyperedge_indices]), dtype=torch.long)
            torch.save({
                'hyper_edge_index': hyper_edge_index,
                'num_nodes': len(self.poi_ids),
                'num_hyperedges': H.shape[1]
            }, f"{self.output_dir}/hypergraph.pt")

    def _save_config_and_stats(self):
        """保存配置和统计信息"""
        all_samples = self.train_data + self.val_data + self.test_data
        hypergraph_info = {}
        if self.augmented_user_hypergraphs:
            hypergraph_info = {
                'num_users_with_hypergraphs': len(self.augmented_user_hypergraphs),
                'num_train_users': len(set([s['user_id'] for s in self.train_data]))
            }

        current_stats = {
            'num_pois': len(self.poi_ids) if self.poi_ids else 0,
            'num_cats': self.num_cats,
            'num_users': len(set([s['user_id'] for s in all_samples])),
            'max_seq_len': max([len(s['poi_sequence'].split(';')) for s in all_samples]) if all_samples else 0,
            'dataset_split': {
                'train': len(self.train_data),
                'val': len(self.val_data),
                'test': len(self.test_data)
            },
            'user_hypergraph_info': hypergraph_info,
            'config': self.config
        }
        self.stats.update(current_stats)

        with open(f"{self.output_dir}/dataset_info.json", 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        with open(f"{self.output_dir}/config.json", 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def _get_results(self) -> Dict:
        """获取处理结果"""
        return {
            'config': self.config,
            'stats': self.stats,
            'train_data': self.train_data,
            'val_data': self.val_data,
            'test_data': self.test_data,
        }


def main():
    """测试用主函数"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/Edin")
    args = parser.parse_args()

    config = {
        'data_dir': args.data_dir,
        'user_hypergraph_cl_enabled': True,
        'num_aug_views_per_user': 2,
    }

    logging.basicConfig(level=logging.INFO)
    processor = DataProcessor(config)
    processor.process()


if __name__ == "__main__":
    main()
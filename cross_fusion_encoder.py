import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HypergraphConv
import math
from typing import Dict, Optional, Tuple, Any, List
import logging

logger = logging.getLogger(__name__)


class DualGraphNodeEncoder(nn.Module):

    def __init__(self, num_pois: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.poi_embedding = nn.Embedding(num_pois, hidden_dim, padding_idx=0)
        self.wgat = GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True, dropout=dropout, edge_dim=1)
        self.hgnn = HypergraphConv(hidden_dim, hidden_dim, use_attention=False, dropout=dropout)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, all_node_ids, spatial_edge_index, spatial_edge_weight, hyper_edge_index):
        """
        输入是整张图的信息，输出是所有POI的增强向量
        """
        # 初始特征
        x = self.poi_embedding(all_node_ids)  # [Num_POIs, Dim]

        # 空间视角流 (WGAT)
        if spatial_edge_weight.dim() == 1:
            spatial_edge_weight = spatial_edge_weight.unsqueeze(-1)
        x_spatial = self.wgat(x, spatial_edge_index, edge_attr=spatial_edge_weight)

        # 语义视角流 (HGNN) - 全局超图
        x_semantic = self.hgnn(x, hyper_edge_index)

        # 融合
        combined = torch.cat([x_spatial, x_semantic], dim=-1)
        enriched_embeddings = self.fusion(combined)  # [Num_POIs, Dim]

        return enriched_embeddings


class UserHypergraphEncoder(nn.Module):
    """
    用户超图编码器
    用于对每个用户的超图进行编码，得到图级别的表示
    """

    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 多层超图卷积
        self.hgnn_layers = nn.ModuleList([
            HypergraphConv(hidden_dim, hidden_dim, use_attention=False, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 图池化层（用于从节点表示得到图表示）
        self.graph_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # 投影头（用于对比学习）
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, node_features: torch.Tensor, user_hyper_edge_index: torch.Tensor) -> torch.Tensor:
        """
        对单个用户超图进行编码，返回图级别的表示
        """
        x = node_features

        # 多层超图卷积
        for hgnn in self.hgnn_layers:
            x = hgnn(x, user_hyper_edge_index)
            x = F.relu(x)

        # 图池化：使用注意力池化
        attention_scores = torch.sigmoid(self.graph_pool(x))  # [num_nodes, 1]

        # 加权求和得到图表示
        graph_emb = torch.sum(x * attention_scores, dim=0) / torch.sum(attention_scores)

        # 投影
        graph_emb_proj = self.projection_head(graph_emb)

        return graph_emb_proj

    def forward_batch(self, node_features: torch.Tensor,
                      user_hyper_edge_indices: List[torch.Tensor]) -> torch.Tensor:
        """
        批量处理用户超图
        """
        graph_embs = []
        for edge_index in user_hyper_edge_indices:
            # 处理单个超图
            if edge_index.size(1) > 0:  # 确保边索引非空
                graph_emb = self.forward(node_features, edge_index)
            else:
                # 如果超图为空，使用零向量
                graph_emb = torch.zeros(self.hidden_dim, device=node_features.device)
            graph_embs.append(graph_emb)

        return torch.stack(graph_embs, dim=0)


class ContextTransformer(nn.Module):

    def __init__(self, num_cats: int, max_len: int = 50, hidden_dim: int = 128, nhead: int = 4, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.cat_emb = nn.Embedding(num_cats, hidden_dim, padding_idx=0)
        self.time_emb = nn.Embedding(48, hidden_dim)  # 0-47 时间槽
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        # 门控融合机制
        self.cat_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.time_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, poi_emb_seq, cat_seq, time_seq, padding_mask=None):
        batch_size, seq_len, _ = poi_emb_seq.shape
        cat_features = self.cat_emb(cat_seq)
        time_features = self.time_emb(time_seq)

        # 门控融合
        fused_features = poi_emb_seq + \
                         (cat_features * self.cat_gate(cat_features)) + \
                         (time_features * self.time_gate(time_features))

        # 添加位置编码
        positions = torch.arange(seq_len, device=poi_emb_seq.device).unsqueeze(0)
        fused_features = fused_features + self.pos_emb(positions)
        context = self.transformer(fused_features, src_key_padding_mask=padding_mask)
        context = self.output_norm(context)

        return context


class TriModalFusionLayer(nn.Module):
    """
    【融合层】自适应三模态融合
    作用：显式融合全局图特征、用户个性化特征和动态轨迹上下文，为GRU生成粗糙轨迹提供综合信息。
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # 用于计算注意力的线性投影
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        # 针对三个模态的 Key 投影
        self.key_global = nn.Linear(hidden_dim, hidden_dim)
        self.key_user = nn.Linear(hidden_dim, hidden_dim)
        self.key_traj = nn.Linear(hidden_dim, hidden_dim)

        # 注意力评分向量
        self.attn_vec = nn.Linear(hidden_dim, 1, bias=False)

        # 最终融合后的投影
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )

    def forward(self, h_global, h_user, h_traj):
        """
        Args:
            h_global: [Batch, Seq_Len, Dim] - 全局图增强的节点特征（未经时序混合）
            h_user:   [Batch, Dim]           - 用户超图特征（个性化长期偏好）
            h_traj:   [Batch, Seq_Len, Dim]  - Transformer 输出的上下文（动态时序意图）
        Returns:
            h_fused:  [Batch, Seq_Len, Dim]
        """
        B, L, D = h_traj.shape

        # 1. 对齐维度：将用户特征广播到序列长度
        # [B, D] -> [B, 1, D] -> [B, L, D]
        h_user_expanded = h_user.unsqueeze(1).expand(-1, L, -1)

        # 2. 堆叠三个模态 [Batch, Seq_Len, 3, Dim]
        stack_feat = torch.stack([h_global, h_user_expanded, h_traj], dim=2)

        # 3. 计算注意力分数
        # 使用 h_traj 作为 Query (因为它是当前的动态意图)，去查询三个模态的重要性
        Q = self.query_proj(h_traj).unsqueeze(2)  # [B, L, 1, D]

        # Keys
        K_global = self.key_global(h_global)
        K_user = self.key_user(h_user_expanded)
        K_traj = self.key_traj(h_traj)
        K = torch.stack([K_global, K_user, K_traj], dim=2)  # [B, L, 3, D]

        # Attention Energy: tanh(Q + K) * V
        # [B, L, 3, 1]
        energy = self.attn_vec(torch.tanh(Q + K))

        # Weights: [B, L, 3, 1]
        attn_weights = F.softmax(energy, dim=2)

        # 4. 加权求和
        # sum( [B, L, 3, 1] * [B, L, 3, D] ) -> [B, L, D]
        h_fused = torch.sum(attn_weights * stack_feat, dim=2)

        return self.out_proj(h_fused)


class HGC_BERT(nn.Module):
    def __init__(self, config: Dict, global_edge_index: torch.Tensor, global_edge_weights: torch.Tensor,
                 hyper_edge_index: torch.Tensor):
        super().__init__()
        self.config = config
        self.hidden_dim = config['hidden_dim']

        logger.info(f"Initializing HGC_BERT with Tri-Modal Adaptive Fusion...")

        # 注册图数据为 Buffer
        self.register_buffer('global_edge_index', global_edge_index)
        self.register_buffer('global_edge_weights', global_edge_weights)
        self.register_buffer('hyper_edge_index', hyper_edge_index)  # 全局超图
        self.register_buffer('all_node_ids', torch.arange(config['num_pois']))

        # 1. 基础编码器
        self.graph_encoder = DualGraphNodeEncoder(config['num_pois'], self.hidden_dim)

        # 用户超图编码器（用于对比学习及个性化融合）
        self.user_hypergraph_encoder = UserHypergraphEncoder(
            hidden_dim=self.hidden_dim,
            num_layers=config.get('user_hgnn_layers', 2),
            dropout=config.get('dropout', 0.1)
        )

        self.seq_encoder = ContextTransformer(
            num_cats=config['num_categories'],
            max_len=config['max_seq_len'],
            hidden_dim=self.hidden_dim
        )

        # 2. [NEW] 三模态融合层
        self.fusion_layer = TriModalFusionLayer(self.hidden_dim, dropout=config.get('dropout', 0.1))

        # 3. 任务头
        self.mlm_head = nn.Linear(self.hidden_dim, config['num_pois'])

        # 粗糙轨迹生成器 (GRU + Classifier)
        self.coarse_rnn = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.coarse_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        self.coarse_classifier = nn.Linear(self.hidden_dim, config['num_pois'])
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, batch_data, masked_indices=None, return_user_graph_emb=False,
                user_hyper_edge_indices1=None, user_hyper_edge_indices2=None):
        """
        前向传播逻辑
        """
        batch_size = batch_data['poi_sequence'].size(0)

        # --- 1. [Modal A] 全局图特征 (Global Graph Features) ---
        enriched_node_table = self.graph_encoder(
            self.all_node_ids,
            self.global_edge_index,
            self.global_edge_weights,
            self.hyper_edge_index
        )

        # --- 2. [Modal B] 用户个性化特征 (User Personalized Features) ---
        user_emb1, user_emb2 = None, None

        # [MODIFIED FOR ABLATION] 根据开关决定是否计算对比学习特征
        enable_cl = self.config.get('enable_cl', True)
        if enable_cl and return_user_graph_emb:
            if user_hyper_edge_indices1 is not None:
                user_emb1 = self.user_hypergraph_encoder.forward_batch(enriched_node_table, user_hyper_edge_indices1)
            if user_hyper_edge_indices2 is not None:
                user_emb2 = self.user_hypergraph_encoder.forward_batch(enriched_node_table, user_hyper_edge_indices2)

        # 平均池化作为融合特征
        if user_emb1 is not None and user_emb2 is not None:
            user_graph_emb = (user_emb1 + user_emb2) / 2.0
        elif user_emb1 is not None:
            user_graph_emb = user_emb1
        elif user_emb2 is not None:
            user_graph_emb = user_emb2
        else:
            # 当未开启 CL 或没有有效数据时，使用全 0 向量代替
            user_graph_emb = torch.zeros(batch_size, self.hidden_dim, device=enriched_node_table.device)

        # --- 3. 准备序列数据 ---
        poi_seq = batch_data['poi_sequence']
        cat_seq = batch_data['cat_sequence']
        time_seq = batch_data['time_sequence']
        padding_mask = (batch_data['attention_mask'] == 0)

        # 获取序列中每个位置的全局图嵌入 [Batch, Seq_Len, Dim]
        poi_emb_seq = F.embedding(poi_seq, enriched_node_table)

        # 备份一份作为 h_global 输入给融合层 (保留纯粹的POI语义，未被Mix)
        h_global = poi_emb_seq.clone()

        # 应用掩码 (仅影响 Transformer 的输入，用于 MLM 学习)
        if masked_indices is not None:
            mask_bool = masked_indices.unsqueeze(-1)
            mask_token_expand = self.mask_token.expand_as(poi_emb_seq)
            poi_emb_seq_masked = torch.where(mask_bool, mask_token_expand, poi_emb_seq)
        else:
            poi_emb_seq_masked = poi_emb_seq

        # --- 4. [Modal C] 轨迹上下文特征 (Dynamic Trajectory Context) ---
        h_traj = self.seq_encoder(
            poi_emb_seq_masked, cat_seq, time_seq,
            padding_mask=padding_mask
        )

        # --- 5. [Fusion] 三模态显式自适应融合 ---
        # h_global: [B, L, D]
        # user_graph_emb: [B, D] (averaged from views)
        # h_traj: [B, L, D]
        fused_context = self.fusion_layer(h_global, user_graph_emb, h_traj)

        # --- 6. 任务输出 ---

        # [MODIFIED FOR ABLATION] 仅当开启 MLM 时才计算 MLM head，否则返回 None
        enable_mlm = self.config.get('enable_mlm', True)
        if enable_mlm:
            mlm_logits = self.mlm_head(h_traj)
        else:
            mlm_logits = None

        # Task 2: Coarse Trajectory Generation (使用融合特征，侧重个性化生成)
        rnn_out, _ = self.coarse_rnn(fused_context)

        coarse_traj_emb = self.coarse_proj(rnn_out)
        coarse_logits = self.coarse_classifier(rnn_out)

        # Task 3: Contrastive Learning (需要原始分离视图)
        # 这里 user_emb1/2 已经在上面计算过，直接返回即可
        return {
            'mlm_logits': mlm_logits,
            'context_emb': h_traj,  # SDE 的 Context Condition
            'coarse_traj_emb': coarse_traj_emb,  # SDE 的 Coarse Condition (个性化增强)
            'coarse_logits': coarse_logits,
            'user_graph_emb1': user_emb1,
            'user_graph_emb2': user_emb2,
            'enriched_node_table': enriched_node_table
        }

    def get_diffusion_context(self, start_poi, end_poi, start_time, end_time, length,
                              user_hyper_edge_indices1=None, user_hyper_edge_indices2=None):
        """
        推理阶段：准备扩散模型所需的上下文和初始条件
        [MODIFIED] 增加 user_hyper_edge_index 参数以支持个性化融合
        """
        self.eval()
        batch_size = start_poi.size(0)
        device = start_poi.device
        max_len = self.config['max_seq_len']

        with torch.no_grad():
            # 1. 实时更新图特征
            enriched_node_table = self.graph_encoder(
                self.all_node_ids,
                self.global_edge_index,
                self.global_edge_weights,
                self.hyper_edge_index
            )

            # 2. 准备基础序列
            cat_seq = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
            time_seq = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
            attn_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)

            # 3. 构建 POI Embedding 序列 (即 h_global)
            # 使用 Mask Token 初始化整个序列
            poi_emb_seq = self.mask_token.expand(batch_size, max_len, -1).clone()

            for i in range(batch_size):
                cur_len = min(int(length[i].item()), max_len)
                attn_mask[i, :cur_len] = False

                # 填入起点和终点的真实 Embedding
                start_vec = enriched_node_table[start_poi[i]]
                poi_emb_seq[i, 0] = start_vec

                if cur_len > 1:
                    end_vec = enriched_node_table[end_poi[i]]
                    poi_emb_seq[i, cur_len - 1] = end_vec

                # 填充时间序列
                if cur_len > 1:
                    s_t = float(start_time[i])
                    e_t = float(end_time[i])
                    times = torch.linspace(s_t, e_t, steps=cur_len)
                    time_seq[i, :cur_len] = times.long()
                else:
                    time_seq[i, 0] = start_time[i]

            # h_global (用于融合的原始图特征序列)
            h_global = poi_emb_seq.clone()

            # 4. 计算用户特征 (h_user)
            # [MODIFIED FOR ABLATION] 保持推理时的一致性
            user_emb1, user_emb2 = None, None
            enable_cl = self.config.get('enable_cl', True)

            if enable_cl:
                if user_hyper_edge_indices1 is not None:
                    user_emb1 = self.user_hypergraph_encoder.forward_batch(enriched_node_table,
                                                                           user_hyper_edge_indices1)
                if user_hyper_edge_indices2 is not None:
                    user_emb2 = self.user_hypergraph_encoder.forward_batch(enriched_node_table,
                                                                           user_hyper_edge_indices2)

            if user_emb1 is not None and user_emb2 is not None:
                user_graph_emb = (user_emb1 + user_emb2) / 2.0
            elif user_emb1 is not None:
                user_graph_emb = user_emb1
            elif user_emb2 is not None:
                user_graph_emb = user_emb2
            else:
                user_graph_emb = torch.zeros(batch_size, self.hidden_dim, device=device)

            # 5. 编码序列获取上下文 (h_traj)
            context_emb = self.seq_encoder(
                poi_emb_seq, cat_seq, time_seq,
                padding_mask=attn_mask
            )

            # 6. [Fusion] 执行三模态融合
            fused_context = self.fusion_layer(h_global, user_graph_emb, context_emb)

            # 7. 生成个性化粗糙轨迹
            rnn_out, _ = self.coarse_rnn(fused_context)
            coarse_traj_emb = self.coarse_proj(rnn_out)

            # 返回字典，包含上下文和粗糙轨迹
            return {
                'context_emb': context_emb,  # SDE 的 Transformer Context
                'coarse_traj_emb': coarse_traj_emb  # 融合了个性化的 Coarse Trajectory
            }


class UserHypergraphContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, user_graph_embs1: torch.Tensor, user_graph_embs2: torch.Tensor,
                user_ids: torch.Tensor = None) -> torch.Tensor:
        batch_size = user_graph_embs1.size(0)

        if batch_size <= 1:
            return torch.tensor(0.0, device=user_graph_embs1.device)
        z1 = F.normalize(user_graph_embs1, dim=-1)
        z2 = F.normalize(user_graph_embs2, dim=-1)
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(batch_size, device=user_graph_embs1.device)
        return self.criterion(sim_matrix, labels)


class MaskedPredictionLoss(nn.Module):
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(predictions.view(-1, predictions.size(-1)), targets.view(-1))
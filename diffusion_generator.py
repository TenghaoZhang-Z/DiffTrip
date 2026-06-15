import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import numpy as np
import torchsde

# Import custom modules
from cross_fusion_encoder import HGC_BERT, MaskedPredictionLoss, UserHypergraphContrastiveLoss

logger = logging.getLogger(__name__)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DenoisingTransformer(nn.Module):
    def __init__(self, hidden_dim=128, nhead=4, num_layers=4, dropout=0.1, max_len=100):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.local_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.local_norm = nn.LayerNorm(hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x_t, t, context, coarse_traj=None, src_key_padding_mask=None):
        B, L, D = x_t.shape
        device = x_t.device

        t_emb = self.time_mlp(t)
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        pos_emb = self.pos_emb(positions)

        x = self.input_proj(x_t) + t_emb.unsqueeze(1) + pos_emb
        if coarse_traj is not None:
            x = x + coarse_traj

        global_feat = self.transformer(
            tgt=x,
            memory=context,
            tgt_key_padding_mask=src_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        conv_input = global_feat.transpose(1, 2)
        local_feat = self.local_conv(conv_input)
        local_feat = F.gelu(local_feat)
        local_feat = local_feat.transpose(1, 2)
        local_feat = self.local_norm(local_feat)

        combined_feat = global_feat + local_feat
        return self.output_proj(combined_feat)


class ContextAwareWeightNetwork(nn.Module):
    """上下文感知权重网络"""

    def __init__(self, config):
        super().__init__()

        # 特征维度
        self.time_dim = 3  # 时间特征维度
        self.spatial_dim = 3  # 空间特征维度
        self.user_dim = 32  # 用户特征维度
        self.state_dim = 10  # 状态特征维度
        self.confidence_dim = 2  # 置信度特征维度

        total_input_dim = self.time_dim + self.spatial_dim + self.user_dim + self.state_dim + self.confidence_dim

        # 编码器网络
        self.encoder = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        # 三个独立的权重头
        self.dist_weight_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.time_weight_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.trans_weight_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 全局约束强度预测
        self.global_strength_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, context_features):
        # 编码上下文
        encoded = self.encoder(context_features)

        # 生成各约束权重
        w_dist = self.dist_weight_head(encoded)
        w_time = self.time_weight_head(encoded)
        w_trans = self.trans_weight_head(encoded)

        # 生成全局约束强度
        global_strength = self.global_strength_head(encoded)

        # 归一化各约束相对权重
        weights = torch.cat([w_dist, w_time, w_trans], dim=-1)
        weights = F.softmax(weights, dim=-1)

        return weights, global_strength


class SDEFunc(torchsde.SDEIto):
    def __init__(self, model, context, coarse_traj, padding_mask,
                 beta_min, beta_max, trans_log_prob=None, guidance_scale=0.0,
                 p=2.0, all_node_embs=None):
        super().__init__(noise_type="diagonal")
        self.model = model
        self.context = context
        self.coarse_traj = coarse_traj
        self.padding_mask = padding_mask
        self.batch_size, self.seq_len, self.dim = context.shape
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.trans_log_prob = trans_log_prob
        self.guidance_scale = guidance_scale
        self.p = p  # Time-varying guidance polynomial power
        self.all_node_embs = all_node_embs

    def f(self, s, y):
        t_diff = 1.0 - s
        y_reshaped = y.view(self.batch_size, self.seq_len, self.dim)
        B, L, D = y_reshaped.shape
        beta_t = self.beta_min + t_diff * (self.beta_max - self.beta_min)
        t_expand = t_diff.expand(B)

        with torch.set_grad_enabled(True):
            y_reshaped.requires_grad_(True)
            eps_pred = self.model(y_reshaped, t_expand, self.context, self.coarse_traj, self.padding_mask)
            total_score = -eps_pred

            if self.trans_log_prob is not None and self.guidance_scale > 0 and self.all_node_embs is not None:
                gamma_t = self.guidance_scale * (s ** self.p)

                y_norm = F.normalize(y_reshaped, dim=-1)
                emb_norm = F.normalize(self.all_node_embs, dim=-1)
                logits = torch.matmul(y_norm, emb_norm.t())
                probs = F.softmax(logits, dim=-1)

                probs_prev = torch.roll(probs, shifts=1, dims=1)
                probs_prev[:, 0, :] = 0.0

                expected_probs = torch.matmul(probs_prev, torch.exp(self.trans_log_prob))
                log_expected = torch.log(expected_probs + 1e-9)

                energy = (probs * log_expected).sum()
                guidance_grad = torch.autograd.grad(energy, y_reshaped)[0]

                # Apply time-varying guidance gradient
                total_score = total_score - gamma_t * guidance_grad

        drift = 0.5 * beta_t * (y_reshaped + total_score)
        return drift.view(self.batch_size, -1)

    def g(self, s, y):
        t_diff = 1.0 - s
        beta_t = self.beta_min + t_diff * (self.beta_max - self.beta_min)
        return torch.sqrt(beta_t).expand_as(y)


class TrajectorySDE(nn.Module):
    """SDE-based Diffusion Manager with Context-Aware Constraints"""

    def __init__(self, encoder, denoiser, config, transition_matrix=None,
                 distance_matrix=None, time_cat_matrix=None, poi_to_cat_tensor=None,
                 poi_coords=None, user_embeddings=None):
        super().__init__()
        self.encoder = encoder
        self.denoiser = denoiser
        self.config = config

        self.beta_min = config.get('beta_min', 0.1)
        self.beta_max = config.get('beta_max', 20.0)
        self.guidance_power = config.get('guidance_power', 2.0)

        # 注册约束矩阵
        if transition_matrix is not None:
            self.register_buffer('trans_log_prob', torch.log(transition_matrix + 1e-9))
        else:
            self.register_buffer('trans_log_prob', None)

        if distance_matrix is not None:
            self.register_buffer('distance_matrix', distance_matrix)
        else:
            self.register_buffer('distance_matrix', None)

        if time_cat_matrix is not None:
            self.register_buffer('time_cat_matrix', time_cat_matrix)
        else:
            self.register_buffer('time_cat_matrix', None)

        if poi_to_cat_tensor is not None:
            self.register_buffer('poi_to_cat_tensor', poi_to_cat_tensor)
        else:
            self.register_buffer('poi_to_cat_tensor', None)

        # 注册坐标和用户嵌入
        if poi_coords is not None:
            self.register_buffer('poi_coords', poi_coords)
        else:
            self.register_buffer('poi_coords', None)

        if user_embeddings is not None:
            self.register_buffer('user_embeddings', user_embeddings)
        else:
            self.register_buffer('user_embeddings', None)

        # 上下文感知权重网络 (将在阶段2端到端训练)
        self.context_weight_net = ContextAwareWeightNetwork(config)

        # 约束参数
        self.gamma = config.get('dist_gamma', 0.1)
        self.base_lambda = config.get('constraint_base_lambda', 0.8)

        self.input_norm = nn.LayerNorm(config['hidden_dim'])

        # Losses
        self.mlm_criterion = MaskedPredictionLoss()
        self.cl_criterion = UserHypergraphContrastiveLoss(temperature=config.get('cl_temperature', 0.1))
        self.mse_criterion = nn.MSELoss()
        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.lambda_mlm = config.get('lambda_mlm', 0.1)
        self.lambda_cl = config.get('lambda_cl', 0.01)
        self.lambda_coarse = config.get('lambda_coarse', 1.0)

    def _extract_parallel_context_features(self, x_t, t_expand, batch_user_ids, enriched_node_table):
        """为联合训练提取并行的连续状态上下文特征"""
        B, L, D = x_t.shape
        device = x_t.device
        features = torch.zeros(B, L, 50, device=device)

        # 1. 扩散时间进度特征
        features[:, :, 0] = t_expand.unsqueeze(1).expand(B, L)
        features[:, :, 1] = 0.5  # Time dummy slot
        features[:, :, 2] = 0.0  # Weekend dummy

        # 2. 用户特征映射 (维度: 32)
        if self.user_embeddings is not None and batch_user_ids is not None:
            user_embs = self.user_embeddings[batch_user_ids]
            features[:, :, 6:38] = user_embs.unsqueeze(1).expand(B, L, 32)

        # 3. 提取连续对齐的信息熵特征
        norm_xt = F.normalize(x_t, dim=-1)
        norm_emb = F.normalize(enriched_node_table, dim=-1)
        logits = torch.matmul(norm_xt, norm_emb.t())
        probs = F.softmax(logits, dim=-1)

        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        max_entropy = math.log(logits.size(-1))
        features[:, :, 48] = (entropy / max_entropy).clamp(0, 1)

        # 展平以便输入 MLP
        return features.view(B * L, 50), probs

    def forward(self, batch, stage='all'):
        device = batch['poi_sequence'].device
        batch_size, seq_len = batch['poi_sequence'].shape

        enable_mlm = self.config.get('enable_mlm', True)
        enable_cl = self.config.get('enable_cl', True)

        encoder_out = self.encoder(
            batch,
            masked_indices=None,
            return_user_graph_emb=(stage in ['pretrain', 'all']),
            user_hyper_edge_indices1=batch.get('user_hyper_edge_indices1'),
            user_hyper_edge_indices2=batch.get('user_hyper_edge_indices2')
        )

        total_loss = torch.tensor(0.0, device=device)
        results = {}
        enriched_node_table = encoder_out['enriched_node_table']
        x_start_raw = F.embedding(batch['poi_sequence'], enriched_node_table)
        x_start = self.input_norm(x_start_raw)

        if stage in ['pretrain', 'all']:
            padding_mask = (batch['attention_mask'] == 0)

            if enable_mlm and encoder_out['mlm_logits'] is not None:
                mlm_labels = batch['poi_sequence'].clone()
                mlm_labels[padding_mask] = -100
                mlm_loss = self.mlm_criterion(encoder_out['mlm_logits'], mlm_labels)
                total_loss += self.lambda_mlm * mlm_loss
                results['mlm_loss'] = mlm_loss

            if enable_cl and encoder_out['user_graph_emb1'] is not None:
                cl_loss = self.cl_criterion(encoder_out['user_graph_emb1'], encoder_out['user_graph_emb2'],
                                            batch['user_id'])
                total_loss += self.lambda_cl * cl_loss
                results['cl_loss'] = cl_loss

            coarse_traj = encoder_out['coarse_traj_emb']
            coarse_mse_loss = self.mse_criterion(coarse_traj[~padding_mask], x_start[~padding_mask])

            coarse_logits = encoder_out['coarse_logits']
            coarse_ce_loss = self.ce_criterion(
                coarse_logits.view(-1, coarse_logits.size(-1)),
                batch['poi_sequence'].view(-1)
            )

            total_coarse_loss = coarse_mse_loss + 2.0 * coarse_ce_loss
            total_loss += self.lambda_coarse * total_coarse_loss
            results['coarse_loss'] = total_coarse_loss

        if stage in ['diffusion', 'all']:
            context = encoder_out['context_emb']
            coarse_traj = encoder_out['coarse_traj_emb']
            padding_mask = (batch['attention_mask'] == 0)

            t = torch.rand(batch_size, device=device) * (1.0 - 1e-5) + 1e-5

            integral_beta = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t ** 2)
            alpha_t = torch.exp(-0.5 * integral_beta).view(-1, 1, 1)
            std_t = torch.sqrt(1.0 - torch.exp(-integral_beta)).view(-1, 1, 1)

            epsilon = torch.randn_like(x_start)
            x_t = x_start * alpha_t + epsilon * std_t

            # [MODIFIED] 为了让 MLP 能接收到反向传播的梯度，必须跟踪 x_t 的运算图
            x_t.requires_grad_(True)

            eps_pred = self.denoiser(x_t, t, context, coarse_traj, padding_mask)

            # --- 联合能量与分数的端到端优化机制 (Joint Score and Energy Modeling) ---
            guidance_scale = self.config.get('guidance_scale', 2.0)

            if guidance_scale > 0:
                norm_node_table = self.input_norm(enriched_node_table)
                # 提取批量并行的上下文特征及软对齐概率
                ctx_feats, probs = self._extract_parallel_context_features(x_t, t, batch.get('user_id'),
                                                                           norm_node_table)

                # MLP 前向计算自适应动态权重
                weights, strength = self.context_weight_net(ctx_feats)
                weights = weights.view(batch_size, seq_len, 3)
                strength = strength.view(batch_size, seq_len)

                w_dist, w_time, w_trans = weights[:, :, 0], weights[:, :, 1], weights[:, :, 2]
                energy_total = torch.tensor(0.0, device=device, requires_grad=True).clone()

                probs_prev = torch.roll(probs, shifts=1, dims=1)
                probs_prev[:, 0, :] = 0.0

                # 约束A: 转移概率能量
                if self.trans_log_prob is not None:
                    expected_probs = torch.matmul(probs_prev, torch.exp(self.trans_log_prob))
                    log_expected = torch.log(expected_probs + 1e-9)
                    energy_trans = (probs * log_expected).sum(dim=-1)
                    energy_total = energy_total + (strength * w_trans * energy_trans).sum()

                # 约束B: 地理空间距离能量
                if self.distance_matrix is not None:
                    dist_exp = torch.matmul(probs_prev, self.distance_matrix)
                    energy_dist = -(probs * dist_exp).sum(dim=-1) * self.gamma
                    energy_total = energy_total + (strength * w_dist * energy_dist).sum()

                # 生成能量梯度并修正预测噪声 (create_graph=True 保留计算图使 MLP 受益于 L_diff 梯度更新)
                if energy_total.requires_grad:
                    grad_E = torch.autograd.grad(energy_total, x_t, create_graph=True)[0]
                    # 修改去噪网络预测：将引导梯度施加于预测噪声，迫使 MLP 学习最优分配以最小化最终误差
                    eps_pred = eps_pred - std_t * grad_E * guidance_scale

            # 计算最终整合了能量干预的 MSE 损失
            loss_score = F.mse_loss(eps_pred[~padding_mask], epsilon[~padding_mask])

            total_loss += loss_score
            results['diff_loss'] = loss_score

        results['loss'] = total_loss
        return results

    def _extract_context_features(self, step_idx, traj_len, prev_poi_id, start_poi_id,
                                  end_poi_id, start_time, end_time, current_time,
                                  visited_pois, user_id, model_entropy, top_k_conf):
        """提取当前步骤的上下文特征 (用于推理阶段离散解码)"""
        features = []

        time_slot = current_time % 48
        features.append(time_slot / 47.0)
        time_progress = step_idx / (traj_len - 1) if traj_len > 1 else 0.0
        features.append(time_progress)
        is_weekend = 0.0
        features.append(is_weekend)

        if self.poi_coords is not None:
            prev_coord = self.poi_coords[prev_poi_id]
            start_coord = self.poi_coords[start_poi_id]
            end_coord = self.poi_coords[end_poi_id]
            dist_to_start = torch.norm(prev_coord - start_coord)
            dist_to_end = torch.norm(prev_coord - end_coord)
            max_dist = 50.0
            features.append(dist_to_start.item() / max_dist)
            features.append(dist_to_end.item() / max_dist)
            area_type = 0.0
            features.append(area_type)
        else:
            features.extend([0.5, 0.5, 0.0])

        if self.user_embeddings is not None and user_id < len(self.user_embeddings):
            user_emb = self.user_embeddings[user_id]
            features.extend(user_emb.tolist())
        else:
            features.extend([0.0] * 32)

        features.append(len(visited_pois) / 20.0)
        visited_cat_diversity = 0.5
        features.append(visited_cat_diversity)
        features.extend([
            step_idx / 20.0,
            (traj_len - step_idx) / 20.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])

        features.append(model_entropy)
        features.append(top_k_conf)

        return torch.tensor(features, dtype=torch.float32)

    @torch.no_grad()
    def sample_trajectory(self, start_poi, end_poi, start_time, end_time, length,
                          user_hyper_edge_indices1=None,
                          user_hyper_edge_indices2=None,
                          guidance_scale=2.0,
                          user_ids=None,
                          temperature=1.0,
                          decode_top_k=5):
        """SDE采样 + 上下文感知约束解码"""
        B = start_poi.size(0)
        device = start_poi.device

        encoder_out = self.encoder.get_diffusion_context(
            start_poi, end_poi, start_time, end_time, length,
            user_hyper_edge_indices1=user_hyper_edge_indices1,
            user_hyper_edge_indices2=user_hyper_edge_indices2
        )
        context = encoder_out['context_emb']
        coarse_traj = encoder_out['coarse_traj_emb']

        enriched_node_table = self.encoder.graph_encoder(
            self.encoder.all_node_ids,
            self.encoder.global_edge_index,
            self.encoder.global_edge_weights,
            self.encoder.hyper_edge_index
        )
        all_node_embs = self.input_norm(enriched_node_table)

        sde_func = SDEFunc(
            model=self.denoiser,
            context=context,
            coarse_traj=coarse_traj,
            padding_mask=None,
            beta_min=self.beta_min,
            beta_max=self.beta_max,
            trans_log_prob=self.trans_log_prob,
            guidance_scale=guidance_scale,
            p=self.guidance_power,
            all_node_embs=all_node_embs
        )

        y0 = torch.randn_like(context).view(B, -1)
        ts = torch.linspace(0.0, 1.0, steps=100, device=device)
        ys = torchsde.sdeint(sde_func, y0, ts, method='euler', dt=1.0 / 100)
        x_final = ys[-1].view(B, context.size(1), context.size(2))

        start_emb = self.input_norm(F.embedding(start_poi, enriched_node_table))
        end_emb = self.input_norm(F.embedding(end_poi, enriched_node_table))

        x_final[:, 0, :] = start_emb
        for b in range(B):
            l = min(int(length[b].item()), context.size(1))
            if l > 1:
                x_final[b, l - 1, :] = end_emb[b]

        normalized_table = F.normalize(all_node_embs, dim=-1)
        normalized_x = F.normalize(x_final, dim=-1)
        scores = torch.matmul(normalized_x, normalized_table.t())

        predicted_ids = torch.zeros(B, context.size(1), dtype=torch.long, device=device)
        num_pois = scores.size(-1)

        for b in range(B):
            traj_len = min(int(length[b].item()), context.size(1))
            visited = set()

            s_id = start_poi[b].item()
            predicted_ids[b, 0] = s_id
            visited.add(s_id)

            e_id = end_poi[b].item()
            if traj_len > 1:
                predicted_ids[b, traj_len - 1] = e_id

            user_id = user_ids[b].item() if user_ids is not None else 0

            for t in range(1, traj_len - 1):
                current_scores = scores[b, t].clone()

                probs = torch.softmax(current_scores, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-9))
                max_entropy = max(math.log(num_pois), 1.0)
                normalized_entropy = (entropy / max_entropy).clamp(0, 1)

                top_k = 5
                top_values, _ = torch.topk(current_scores, k=top_k)
                top_k_conf = torch.var(top_values).item() if top_k > 1 else 0.0

                prev_id = predicted_ids[b, t - 1].item()
                s_t = start_time[b].float()
                e_t = end_time[b].float()
                current_time = s_t + (e_t - s_t) * (t / (traj_len - 1))

                context_features = self._extract_context_features(
                    step_idx=t,
                    traj_len=traj_len,
                    prev_poi_id=prev_id,
                    start_poi_id=s_id,
                    end_poi_id=e_id,
                    start_time=s_t,
                    end_time=e_t,
                    current_time=current_time,
                    visited_pois=visited,
                    user_id=user_id,
                    model_entropy=normalized_entropy,
                    top_k_conf=top_k_conf
                ).unsqueeze(0).to(device)

                weights, global_strength = self.context_weight_net(context_features)
                w_dist = weights[0, 0].item()
                w_time = weights[0, 1].item()
                w_trans = weights[0, 2].item()
                strength = global_strength[0, 0].item()

                dist_bonus, time_cat_bonus, trans_bonus = 0.0, 0.0, 0.0

                if self.distance_matrix is not None:
                    dist_bonus = -self.gamma * self.distance_matrix[prev_id]

                if self.time_cat_matrix is not None and self.poi_to_cat_tensor is not None:
                    current_time_slot = max(0, min(47, int(current_time.round().item())))
                    cat_log_probs = self.time_cat_matrix[current_time_slot]
                    time_cat_bonus = cat_log_probs[self.poi_to_cat_tensor]

                if self.trans_log_prob is not None:
                    trans_bonus = self.trans_log_prob[prev_id]

                constraint_adjustment = (
                        w_dist * dist_bonus +
                        w_time * time_cat_bonus +
                        w_trans * trans_bonus
                )

                current_scores = current_scores + strength * constraint_adjustment

                if visited:
                    mask_indices = torch.tensor(list(visited), device=device)
                    current_scores[mask_indices] = -float('inf')

                if s_id != e_id:
                    current_scores[e_id] = -float('inf')

                scaled_scores = current_scores / temperature

                valid_mask = scaled_scores > -float('inf')
                num_valid = valid_mask.sum().item()
                actual_k = max(1, min(decode_top_k, num_valid))

                if actual_k == 1:
                    next_id = torch.argmax(scaled_scores).item()
                else:
                    top_k_vals, top_k_indices = torch.topk(scaled_scores, k=actual_k)
                    top_k_probs = torch.softmax(top_k_vals, dim=-1)
                    sampled_idx = torch.multinomial(top_k_probs, 1).item()
                    next_id = top_k_indices[sampled_idx].item()

                predicted_ids[b, t] = next_id
                visited.add(next_id)

        return predicted_ids


class CompleteTrajectoryDiffusionModel(nn.Module):
    """完整轨迹扩散模型"""

    def __init__(self, config, num_pois, global_graph_data, hypergraph_data,
                 transition_matrix=None, distance_matrix=None, time_cat_matrix=None,
                 poi_to_cat_tensor=None, poi_coords=None, user_embeddings=None):
        super().__init__()
        self.config = config
        g_edge_index = global_graph_data['edge_index']
        g_edge_weights = global_graph_data['edge_weights']
        h_edge_index = hypergraph_data['hyper_edge_index']

        self.encoder = HGC_BERT(config, g_edge_index, g_edge_weights, h_edge_index)

        self.denoiser = DenoisingTransformer(
            hidden_dim=config['hidden_dim'],
            nhead=config.get('nhead', 4),
            num_layers=config.get('num_layers', 4),
            dropout=config.get('dropout', 0.1),
            max_len=config.get('max_seq_len', 100)
        )

        self.generator = TrajectorySDE(
            self.encoder, self.denoiser, config,
            transition_matrix=transition_matrix,
            distance_matrix=distance_matrix,
            time_cat_matrix=time_cat_matrix,
            poi_to_cat_tensor=poi_to_cat_tensor,
            poi_coords=poi_coords,
            user_embeddings=user_embeddings
        )

    def forward(self, batch, mode='train', stage='all'):
        if mode == 'train':
            return self.generator(batch, stage=stage)
        else:
            raise ValueError("Use .sample() for inference")

    @torch.no_grad()
    def sample(self, start_poi, end_poi, start_time, end_time, length,
               user_hyper_edge_indices1=None,
               user_hyper_edge_indices2=None,
               guidance_scale=2.0,
               user_ids=None,
               temperature=1.0,
               decode_top_k=5):
        return self.generator.sample_trajectory(
            start_poi, end_poi, start_time, end_time, length,
            user_hyper_edge_indices1=user_hyper_edge_indices1,
            user_hyper_edge_indices2=user_hyper_edge_indices2,
            guidance_scale=guidance_scale,
            user_ids=user_ids,
            temperature=temperature,
            decode_top_k=decode_top_k
        )
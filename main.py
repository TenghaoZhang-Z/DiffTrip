import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import logging
import os
import json
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import networkx as nx
from node2vec import Node2Vec
import time  # 导入时间模块计算 Epoch 耗时

# --- 1. Import Custom Modules ---
from input_data import DataProcessor
from evaluate import TrajectoryEvaluator
from cross_fusion_encoder import HGC_BERT
from diffusion_generator import DenoisingTransformer, TrajectorySDE

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log")
    ]
)
logger = logging.getLogger(__name__)


def pretrain_node_embeddings(transition_matrix, num_pois, hidden_dim):
    """
    使用 Node2Vec 基于转移矩阵预训练 POI Embedding
    """
    logger.info("Starting Node2Vec pre-training to initialize embeddings...")

    G = nx.Graph()
    rows, cols = torch.where(transition_matrix > 0)
    rows = rows.cpu().numpy()
    cols = cols.cpu().numpy()
    weights = transition_matrix[rows, cols].cpu().numpy()

    edge_list = []
    for r, c, w in zip(rows, cols, weights):
        edge_list.append((r, c, float(w)))

    if not edge_list:
        logger.warning("Transition matrix is empty! Skipping pre-training.")
        return None

    G.add_weighted_edges_from(edge_list)

    for i in range(num_pois):
        if i not in G:
            G.add_node(i)

    node2vec = Node2Vec(G, dimensions=hidden_dim, walk_length=10, num_walks=20, workers=1, quiet=True)
    model = node2vec.fit(window=5, min_count=1, batch_words=4)

    pretrained_weights = torch.zeros(num_pois, hidden_dim)
    count_initialized = 0

    for i in range(num_pois):
        key = str(i)
        if key in model.wv:
            pretrained_weights[i] = torch.tensor(model.wv[key])
            count_initialized += 1
        else:
            torch.nn.init.xavier_uniform_(pretrained_weights[i].unsqueeze(0))

    logger.info(f"Node2Vec pre-training complete. Initialized {count_initialized}/{num_pois} nodes.")
    return pretrained_weights


# --- 2. Model Wrapper ---
class CompleteTrajectoryDiffusionModel(nn.Module):
    # 增加物理约束张量的传参接口
    def __init__(self, config, num_pois, global_graph_data, hypergraph_data, transition_matrix=None,
                 distance_matrix=None, time_cat_matrix=None, poi_to_cat_tensor=None):
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

        # 将新张量传递给生成器
        self.generator = TrajectorySDE(
            self.encoder, self.denoiser, config,
            transition_matrix=transition_matrix,
            distance_matrix=distance_matrix,
            time_cat_matrix=time_cat_matrix,
            poi_to_cat_tensor=poi_to_cat_tensor
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
               user_ids=None,  # [NEW]
               temperature=1.0,  # [NEW]
               decode_top_k=5):  # [NEW]
        return self.generator.sample_trajectory(
            start_poi, end_poi, start_time, end_time, length,
            user_hyper_edge_indices1=user_hyper_edge_indices1,
            user_hyper_edge_indices2=user_hyper_edge_indices2,
            guidance_scale=guidance_scale,
            user_ids=user_ids,  # [NEW] 传导给 SDE 采样器
            temperature=temperature,  # [NEW]
            decode_top_k=decode_top_k  # [NEW]
        )


# --- 3. Dataset Definition ---
class TrajectoryDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_collate_fn(user_hypergraphs):
    def collate_fn(batch):
        sample_ids = [item['sample_id'] for item in batch]

        def parse(s, dtype=int):
            if not s: return []
            return [dtype(x) for x in str(s).split(';') if x]

        poi_seqs = [torch.tensor(parse(item['poi_sequence']), dtype=torch.long) for item in batch]
        cat_seqs = [torch.tensor(parse(item['cat_sequence']), dtype=torch.long) for item in batch]
        time_seqs = [torch.tensor(parse(item['time_sequence']), dtype=torch.long) for item in batch]

        poi_padded = torch.nn.utils.rnn.pad_sequence(poi_seqs, batch_first=True, padding_value=0)
        cat_padded = torch.nn.utils.rnn.pad_sequence(cat_seqs, batch_first=True, padding_value=0)
        time_padded = torch.nn.utils.rnn.pad_sequence(time_seqs, batch_first=True, padding_value=0)

        lengths = torch.tensor([len(s) for s in poi_seqs])
        if len(lengths) > 0:
            max_len = lengths.max()
            mask = torch.arange(max_len)[None, :] < lengths[:, None]
        else:
            mask = torch.zeros(0, 0)

        user_ids = torch.tensor([item['user_id'] for item in batch], dtype=torch.long)
        user_hyper_edge_indices1 = []
        user_hyper_edge_indices2 = []

        for item in batch:
            user_id = item['user_id']
            if user_id in user_hypergraphs:
                edge_idx1 = user_hypergraphs[user_id]['view1_edge_index']
                edge_idx2 = user_hypergraphs[user_id]['view2_edge_index']
            else:
                edge_idx1 = torch.empty((2, 0), dtype=torch.long)
                edge_idx2 = torch.empty((2, 0), dtype=torch.long)
            user_hyper_edge_indices1.append(edge_idx1)
            user_hyper_edge_indices2.append(edge_idx2)

        return {
            'poi_sequence': poi_padded,
            'cat_sequence': cat_padded,
            'time_sequence': time_padded,
            'attention_mask': mask.long(),
            'lengths': lengths,
            'sample_ids': sample_ids,
            'user_id': user_ids,
            'user_hyper_edge_indices1': user_hyper_edge_indices1,
            'user_hyper_edge_indices2': user_hyper_edge_indices2
        }

    return collate_fn


# --- 4. Trainer Class ---
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(config['output_dir'], exist_ok=True)
        logger.info(f"Using device: {self.device}")

        # Step 1: Process Data
        logger.info("Step 1: Processing Data...")
        self.processor = DataProcessor(config)
        self.data_res = self.processor.process()

        # Step 2: Load Graphs & Transition Matrix
        logger.info("Step 2: Loading Graph Structures & Physical Constraints...")
        try:
            global_graph = torch.load(f"{config['output_dir']}/global_graph.pt")
            hypergraph = torch.load(f"{config['output_dir']}/hypergraph.pt")
            self.user_hypergraphs = torch.load(f"{config['output_dir']}/user_hypergraphs.pt")
            logger.info(f"Loaded user hypergraphs: {len(self.user_hypergraphs)} users")

            # Load Transition Matrix
            trans_path = f"{config['output_dir']}/transition_probs.pt"
            if os.path.exists(trans_path):
                self.transition_matrix = torch.load(trans_path)
                logger.info(f"Loaded transition matrix: {self.transition_matrix.shape}")
            else:
                self.transition_matrix = None
                logger.warning("Transition matrix not found. Running without guidance.")

            # Load Distance Matrix
            dist_path = f"{config['output_dir']}/distance_matrix.pt"
            if os.path.exists(dist_path):
                self.distance_matrix = torch.load(dist_path)
                logger.info(f"Loaded distance matrix: {self.distance_matrix.shape}")
            else:
                self.distance_matrix = None

            # Load Time-Category Matrix
            time_cat_path = f"{config['output_dir']}/time_cat_matrix.pt"
            if os.path.exists(time_cat_path):
                self.time_cat_matrix = torch.load(time_cat_path)
                logger.info(f"Loaded time-category matrix: {self.time_cat_matrix.shape}")
            else:
                self.time_cat_matrix = None

            # Load POI-to-Category Mapping
            poi_cat_path = f"{config['output_dir']}/poi_to_cat.pt"
            if os.path.exists(poi_cat_path):
                self.poi_to_cat_tensor = torch.load(poi_cat_path)
                logger.info(f"Loaded POI-to-Cat tensor: {self.poi_to_cat_tensor.shape}")
            else:
                self.poi_to_cat_tensor = None

        except FileNotFoundError as e:
            logger.error(f"Required files not found: {e}")
            raise

        config['num_pois'] = self.data_res['stats']['num_pois']
        config['num_categories'] = self.data_res['stats']['num_cats']
        config['max_seq_len'] = max(self.data_res['stats'].get('max_seq_len', 50), 50)

        # Step 3: Build Model
        logger.info("Step 3: Building Model...")
        self.model = CompleteTrajectoryDiffusionModel(
            config,
            config['num_pois'],
            global_graph,
            hypergraph,
            transition_matrix=self.transition_matrix,
            distance_matrix=self.distance_matrix,
            time_cat_matrix=self.time_cat_matrix,
            poi_to_cat_tensor=self.poi_to_cat_tensor
        ).to(self.device)

        # Apply Pre-trained Embeddings
        if self.transition_matrix is not None:
            pretrained_emb = pretrain_node_embeddings(
                self.transition_matrix,
                config['num_pois'],
                config['hidden_dim']
            )
            if pretrained_emb is not None:
                self.model.encoder.graph_encoder.poi_embedding.weight.data.copy_(pretrained_emb)
                logger.info("Applied pre-trained Node2Vec weights to POI Embeddings.")

        # Step 4: DataLoader
        collate_fn = create_collate_fn(self.user_hypergraphs)
        self.train_loader = DataLoader(
            TrajectoryDataset(self.data_res['train_data']),
            batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn
        )
        self.val_loader = DataLoader(
            TrajectoryDataset(self.data_res['val_data']),
            batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn
        )
        self.test_loader = DataLoader(
            TrajectoryDataset(self.data_res['test_data']),
            batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn
        )

        self.evaluator = TrajectoryEvaluator()
        self.optimizer = None
        self.scheduler = None

    def _prepare_batch(self, batch):
        """Prepare batch data by moving tensors to device"""
        prepared_batch = {}
        for key, value in batch.items():
            if key in ['user_hyper_edge_indices1', 'user_hyper_edge_indices2']:
                if isinstance(value, list):
                    prepared_batch[key] = [edge_idx.to(self.device) for edge_idx in value]
                else:
                    prepared_batch[key] = value.to(self.device) if isinstance(value, torch.Tensor) else value
            else:
                prepared_batch[key] = value.to(self.device) if isinstance(value, torch.Tensor) else value
        return prepared_batch

    def _freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def _unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True
        module.train()

    def train_epoch(self, stage):
        if stage == 'pretrain':
            self.model.encoder.train()
            self.model.denoiser.eval()
        elif stage == 'diffusion':
            self.model.encoder.train()
            self.model.denoiser.train()

        total_loss = 0
        metrics = {'mlm': 0, 'cl': 0, 'diff': 0, 'coarse': 0}

        pbar = tqdm(self.train_loader, desc=f"Training [{stage}]")
        for batch in pbar:
            batch = self._prepare_batch(batch)
            self.optimizer.zero_grad()

            output = self.model(batch, mode='train', stage=stage)
            loss = output['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

            if 'mlm_loss' in output: metrics['mlm'] += output['mlm_loss'].item()
            if 'cl_loss' in output: metrics['cl'] += output['cl_loss'].item()
            if 'diff_loss' in output: metrics['diff'] += output['diff_loss'].item()
            if 'coarse_loss' in output: metrics['coarse'] += output['coarse_loss'].item()

            logs = {'L': f"{loss.item():.4f}"}
            if stage == 'pretrain':
                logs['M'] = f"{output.get('mlm_loss', torch.tensor(0.0)).item():.3f}"
                logs['Co'] = f"{output['coarse_loss'].item():.3f}"
            else:
                logs['D'] = f"{output['diff_loss'].item():.3f}"
            pbar.set_postfix(logs)

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, metrics

    @torch.no_grad()
    def validate(self, stage):
        self.model.eval()
        total_loss = 0
        if len(self.val_loader) == 0:
            return 0.0

        for batch in self.val_loader:
            batch = self._prepare_batch(batch)
            output = self.model(batch, mode='train', stage=stage)
            total_loss += output['loss'].item()

        return total_loss / len(self.val_loader)

    def run(self):
        logger.info("=" * 60)
        logger.info(f" STAGE 1: Encoder Pre-training ({self.config['epochs_stage1']} epochs)")
        logger.info(" Goal: Learn representations via MLM + CL + Coarse Trajectory Prediction")
        logger.info("=" * 60)

        self._freeze_module(self.model.denoiser)
        self._unfreeze_module(self.model.encoder)

        self.optimizer = optim.AdamW(
            self.model.encoder.parameters(),
            lr=self.config['lr_stage1']
        )

        for epoch in range(self.config['epochs_stage1']):
            start_time = time.time()  # 记录开始时间

            avg_loss, metrics = self.train_epoch(stage='pretrain')

            epoch_time = time.time() - start_time  # 计算耗时

            logger.info(
                f"Stage 1 | Epoch {epoch + 1} | "
                f"Time: {epoch_time:.2f}s | "
                f"Loss: {avg_loss:.4f} "
                f"(M:{metrics['mlm'] / len(self.train_loader):.3f} "
                f"C:{metrics['cl'] / len(self.train_loader):.3f} "
                f"Co:{metrics['coarse'] / len(self.train_loader):.3f})"
            )

        torch.save(self.model.state_dict(), f"{self.config['output_dir']}/stage1_encoder.pth")
        logger.info("Stage 1 complete.")

        logger.info("\n" + "=" * 60)
        logger.info(f"STAGE 2: SDE Training ({self.config['epochs_stage2']} epochs)")
        logger.info(" Goal: Train Score Matching (Denoiser) + Fine-tune encoder")
        logger.info("=" * 60)

        self._unfreeze_module(self.model.encoder)
        self._unfreeze_module(self.model.denoiser)

        optimizer_grouped_parameters = [
            {'params': self.model.denoiser.parameters(), 'lr': self.config['lr_stage2']},
            {'params': self.model.encoder.parameters(), 'lr': self.config['lr_stage2'] * 0.1}
        ]

        self.optimizer = optim.AdamW(optimizer_grouped_parameters)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        best_val_loss = float('inf')
        early_stop_cnt = 0

        for epoch in range(self.config['epochs_stage2']):
            start_time = time.time()  # 记录开始时间

            avg_loss, metrics = self.train_epoch(stage='diffusion')
            val_loss = self.validate(stage='diffusion')

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            epoch_time = time.time() - start_time  # 计算耗时

            logger.info(
                f"Stage 2 | Epoch {epoch + 1} | "
                f"Time: {epoch_time:.2f}s | "
                f"Loss: {avg_loss:.4f} (Score:{metrics['diff'] / len(self.train_loader):.3f}) | "
                f"Val: {val_loss:.4f} | LR: {current_lr:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f"{self.config['output_dir']}/best_model.pth")
                logger.info(f" >>> Saved Best Model (Val Loss: {best_val_loss:.4f})")
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

            if early_stop_cnt >= 20:
                logger.info("Early stopping triggered in Stage 2.")
                break

        self.final_evaluation()

    @torch.no_grad()
    def final_evaluation(self):
        load_path = os.path.join(self.config['output_dir'], "best_model.pth")
        if not os.path.exists(load_path):
            logger.error("No best model found!")
            return

        logger.info(f"Loading best model from {load_path}")
        self.model.load_state_dict(torch.load(load_path))
        self.model.eval()

        global_stats = {
            'batch_f1_max': [],
            'batch_f1_mean': [],
            'batch_pairs_f1_max': [],
            'batch_pairs_f1_mean': []
        }

        # [NEW] 读取新增的推理参数
        guidance_scale = self.config.get('guidance_scale', 2.0)
        temperature = self.config.get('temperature', 1.0)
        decode_top_k = self.config.get('decode_top_k', 5)

        logger.info(
            f"Running SDE Sampling | Guidance Scale: {guidance_scale} | Temp: {temperature} | Top-K: {decode_top_k}")
        logger.info("-" * 80)
        logger.info(f"{'Batch':<6} | {'F1 Max':<10} | {'Pairs Max':<10}")
        logger.info("-" * 80)

        for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
            batch_dev = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            u_indices1 = None
            if 'user_hyper_edge_indices1' in batch:
                u_indices1 = [idx.to(self.device) for idx in batch['user_hyper_edge_indices1']]

            u_indices2 = None
            if 'user_hyper_edge_indices2' in batch:
                u_indices2 = [idx.to(self.device) for idx in batch['user_hyper_edge_indices2']]

            lens = batch_dev['lengths']
            start_poi = batch_dev['poi_sequence'][:, 0]
            idx = (lens - 1).view(-1, 1)
            end_poi = batch_dev['poi_sequence'].gather(1, idx).squeeze(1)
            start_time = batch_dev['time_sequence'][:, 0]
            end_time = batch_dev['time_sequence'].gather(1, idx).squeeze(1)

            # [NEW] 提取 user_ids 提供给模型推理阶段获取全局约束
            user_ids = batch_dev.get('user_id', None)

            preds = self.model.sample(
                start_poi=start_poi,
                end_poi=end_poi,
                start_time=start_time,
                end_time=end_time,
                length=lens,
                user_hyper_edge_indices1=u_indices1,
                user_hyper_edge_indices2=u_indices2,
                guidance_scale=guidance_scale,
                user_ids=user_ids,  # [NEW]
                temperature=temperature,  # [NEW]
                decode_top_k=decode_top_k  # [NEW]
            )

            targets = batch_dev['poi_sequence']
            metrics = self.evaluator.evaluate(preds, targets, lens)

            logger.info(
                f"{batch_idx:<6} | "
                f"{metrics['f1_max']:.4f} | {metrics['pairs_f1_max']:.4f}"
            )

            if metrics['f1_max'] > 0 or metrics['pairs_f1_max'] > 0:
                global_stats['batch_f1_max'].append(metrics['f1_max'])
                global_stats['batch_f1_mean'].append(metrics['f1_mean'])
                global_stats['batch_pairs_f1_max'].append(metrics['pairs_f1_max'])
                global_stats['batch_pairs_f1_mean'].append(metrics['pairs_f1_mean'])

        if len(global_stats['batch_f1_max']) > 0:
            avg_batch_max_f1 = np.mean(global_stats['batch_f1_max'])
            avg_batch_max_pairs = np.mean(global_stats['batch_pairs_f1_max'])

            avg_batch_mean_f1 = np.mean(global_stats['batch_f1_mean'])
            avg_batch_mean_pairs = np.mean(global_stats['batch_pairs_f1_mean'])

            logger.info("=" * 60)
            logger.info("Final Test Results:")
            logger.info(f"F1 Score (Avg Max): {avg_batch_max_f1:.4f} ")
            logger.info(f"PairsF1 Score (Avg Max): {avg_batch_max_pairs:.4f} ")
        else:
            logger.warning("No valid metrics computed.")


# --- 5. Main Entry ---
if __name__ == "__main__":
    config = {
        'data_dir': 'data/Osak',
        'output_dir': 'data/Osak/processed',
        'val_ratio': 0.2,
        'test_ratio': 0.1,

        # 消融实验开关 (Ablation Flags)
        'enable_mlm': True,
        'enable_cl': True,

        # Graph Params
        'user_hypergraph_cl_enabled': True,
        'num_aug_views_per_user': 2,
        'node_mask_prob': 0.2,
        'edge_mask_prob': 0.3,
        'node_add_prob': 0.1,
        'edge_add_prob': 0.2,
        'distance_threshold_km': 3.0,
        'k_neighbors': 10,

        # Model Params
        'hidden_dim': 64,
        'nhead': 8,
        'num_layers': 2,
        'dropout': 0.3,
        'user_hgnn_layers': 2,
        'max_seq_len': 10,

        # SDE Params (VP-SDE)
        'beta_min': 0.1,
        'beta_max': 20.0,
        'guidance_scale': 5.0,
        'guidance_power': 2.0,  # [NEW] 控制 gamma(t) 时变强度的多项式幂
        'temperature': 1.0,  # [NEW] 采样温度（越大探索性越强）
        'decode_top_k': 5,  # [NEW] 截断采样的候选池大小
        'lambda_mlm': 0.5,
        'lambda_cl': 0.1,
        'lambda_coarse': 5.0,
        'cl_temperature': 0.1,

        # 物理与逻辑约束超参数
        'dist_gamma': 0.1,
        'constraint_base_lambda': 0.8,

        'batch_size': 4,
        'epochs_stage1': 150,
        'lr_stage1': 1e-3,

        # Stage 2
        'epochs_stage2': 100,
        'lr_stage2': 5e-4,
    }

    trainer = Trainer(config)
    trainer.run()
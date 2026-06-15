import torch
import numpy as np
from typing import List, Set, Tuple, Dict


class TrajectoryEvaluator:
    """
    轨迹推荐评价器
    计算 F1 (集合重叠) 和 PairsF1 (序列相对顺序重叠 - 宽松模式)
    """

    def __init__(self, top_k: int = 1):
        self.top_k = top_k

    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor = None) -> Dict[
        str, float]:
        """
        Args:
            predictions: [Batch, Seq_Len]
            targets: [Batch, Seq_Len]
            lengths: [Batch]
        Returns:
            Dict: 包含 mean 和 max 指标
        """
        pred_list = predictions.cpu().numpy().tolist()
        target_list = targets.cpu().numpy().tolist()

        f1_scores = []
        pairs_f1_scores = []

        batch_size = len(pred_list)

        for i in range(batch_size):
            # 1. 数据清洗 (去除 Padding)
            if lengths is not None:
                length = int(lengths[i].item())
                # 确保长度至少为2，否则无法计算Pairs
                length = max(length, 2)
                p_seq = pred_list[i][:length]
                t_seq = target_list[i][:length]
            else:
                p_seq = [p for p in pred_list[i] if p != 0]
                t_seq = [t for t in target_list[i] if t != 0]

            # 2. 计算单条轨迹指标
            f1 = self._compute_f1(p_seq, t_seq)
            # 使用修改后的宽松 PairsF1 计算
            pairs_f1 = self._compute_pairs_f1(p_seq, t_seq)

            f1_scores.append(f1)
            pairs_f1_scores.append(pairs_f1)

        # 3. 汇总统计 (计算均值和最大值)
        return {
            "f1_mean": np.mean(f1_scores) if f1_scores else 0.0,
            "f1_max": np.max(f1_scores) if f1_scores else 0.0,
            "pairs_f1_mean": np.mean(pairs_f1_scores) if pairs_f1_scores else 0.0,
            "pairs_f1_max": np.max(pairs_f1_scores) if pairs_f1_scores else 0.0,
            "best_idx": np.argmax(pairs_f1_scores) if pairs_f1_scores else 0
        }

    def _compute_f1(self, pred: List[int], target: List[int]) -> float:
        """计算基于集合的 F1 分数 (保持不变)"""
        if len(pred) == 0 or len(target) == 0:
            return 0.0
        set_pred = set(pred)
        set_target = set(target)
        intersection = len(set_pred & set_target)
        precision = intersection / len(set_pred)
        recall = intersection / len(set_target)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _compute_pairs_f1(self, pred: List[int], target: List[int]) -> float:
        """
        [修改] 计算基于相对顺序的 PairsF1 分数 (宽松模式)
        逻辑：只要 pred 中的 POI 对 (A, B) 在 target 中也是 A 在 B 前面，即视为匹配。
        """
        n_pred = len(pred)
        n_target = len(target)

        # 如果轨迹长度小于2，无法构成对，直接返回0
        if n_pred < 2 or n_target < 2:
            return 0.0

        # 1. 计算总对数 (组合数 C(n, 2))
        total_pred_pairs = n_pred * (n_pred - 1) / 2
        total_target_pairs = n_target * (n_target - 1) / 2

        # 2. 构建真实轨迹的顺序映射 {POI_ID: Index}
        # 如果有重复POI，字典会保留最后一次出现的索引，这在宽松评估中通常是可以接受的
        target_order = {poi: i for i, poi in enumerate(target)}

        # 3. 统计预测轨迹中符合真实相对顺序的对数
        correct_pairs = 0

        # 双重循环遍历预测轨迹的所有可能对 (i, j) 其中 i < j
        for i in range(n_pred):
            for j in range(i + 1, n_pred):
                poi_a = pred[i]
                poi_b = pred[j]

                # 只有当两个点都不相同，且都在真实轨迹中出现过，才进行比较
                if poi_a != poi_b and poi_a in target_order and poi_b in target_order:
                    # 检查真实轨迹中 A 是否在 B 前面
                    if target_order[poi_a] < target_order[poi_b]:
                        correct_pairs += 1

        # 4. 计算 Precision, Recall, F1
        if total_pred_pairs == 0:
            precision = 0.0
        else:
            precision = correct_pairs / total_pred_pairs

        if total_target_pairs == 0:
            recall = 0.0
        else:
            recall = correct_pairs / total_target_pairs

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)
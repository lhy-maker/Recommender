import numpy as np
from collections import defaultdict
from typing import List, Dict, Union

class AccuracyMetric:
    def __init__(self, top_ks: List[int] = [5, 10, 20]):
        self.top_ks = sorted(top_ks)
        self.reset()
        
    def reset(self):
        self.hits = {k: defaultdict(int) for k in self.top_ks}
        self.positions = {k: defaultdict(int) for k in self.top_ks}
        self.test_users = 0
        
    def update(self, recommendations: List[Union[str, int]], 
              ground_truth: List[Union[str, int]], user_id: str):
        """更新指标状态"""
        self.test_users += 1
        gt_set = set(ground_truth)
        
        for k in self.top_ks:
            topk = recommendations[:k]
            hits = len(gt_set & set(topk))
            self.hits[k][user_id] = hits
            self.positions[k][user_id] = [i+1 for i, x in enumerate(topk) if x in gt_set]
            
    def compute(self) -> Dict[str, float]:
        """计算所有准确性指标"""
        metrics = {}
        for k in self.top_ks:
            hit_count = sum(self.hits[k].values())
            
            # Recall@K
            recall = hit_count / (len(ground_truth) * self.test_users) if self.test_users > 0 else 0
            metrics[f'recall@{k}'] = recall
            
            # Precision@K
            precision = hit_count / (k * self.test_users) if self.test_users > 0 else 0
            metrics[f'precision@{k}'] = precision
            
            # MRR@K
            mrr = np.mean([1./p for p in self.positions[k].values() if p]) if self.positions[k] else 0
            metrics[f'mrr@{k}'] = mrr
            
        return metrics
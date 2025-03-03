from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DiversityMetric:
    def __init__(self, item_embeddings: Dict[str, np.ndarray]):
        self.item_embeddings = item_embeddings
        
    def _get_embedding(self, item_id: str) -> np.ndarray:
        return self.item_embeddings.get(item_id, np.zeros(300))
    
    def intra_list_similarity(self, recommendations: List[List[str]]) -> float:
        """计算推荐列表内的平均相似度"""
        total_sim = 0.0
        total_pairs = 0
        
        for rec_list in recommendations:
            embeddings = [self._get_embedding(i) for i in rec_list]
            sim_matrix = cosine_similarity(embeddings)
            np.fill_diagonal(sim_matrix, 0)
            total_sim += np.sum(sim_matrix) / 2  # 上三角元素和
            total_pairs += len(rec_list) * (len(rec_list)-1) / 2
            
        return total_sim / total_pairs if total_pairs > 0 else 0
    
    def inter_list_diversity(self, all_recommendations: List[List[str]]) -> float:
        """计算推荐列表间的覆盖度"""
        all_items = set()
        recommended_items = set()
        
        for rec_list in all_recommendations:
            recommended_items.update(rec_list)
            all_items.update(self.item_embeddings.keys())
            
        return len(recommended_items) / len(all_items) if all_items else 0
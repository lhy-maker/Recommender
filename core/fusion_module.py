# core/fusion_module.py
import numpy as np
import torch
from typing import List, Dict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from omegaconf import DictConfig

class HybridRanker:
    def __init__(self, config: DictConfig):
        self.config = config.fusion
        self.scaler = MinMaxScaler()
        self._init_weights()

    def _init_weights(self):
        """初始化融合权重"""
        self.weights = {
            'semantic': self.config.weights.semantic,
            'collaborative': self.config.weights.collaborative,
            'diversity': self.config.weights.diversity,
            'freshness': self.config.weights.freshness
        }

    def rerank(self, 
              candidates: List[Dict], 
              user_profile: Dict) -> List[Dict]:
        """多维度重新排序"""
        # 特征标准化
        features = self._extract_features(candidates, user_profile)
        scaled_features = self.scaler.fit_transform(features)
        
        # 计算各维度得分
        scores = self._calculate_scores(scaled_features)
        
        # 多样性控制
        if self.config.diversity.enabled:
            scores = self._apply_diversity(scores, candidates)
        
        # 生成最终排序
        sorted_indices = np.argsort(-scores)
        return [candidates[i] for i in sorted_indices]

    def _extract_features(self, candidates: List[Dict], user: Dict) -> np.ndarray:
        """提取多维特征"""
        features = []
        for candidate in candidates:
            # 语义相关性
            semantic = candidate.get('semantic_score', 0.5)
            
            # 协同过滤得分
            collaborative = self._cosine_sim(
                user['embedding'], 
                candidate['embedding']
            )
            
            # 多样性得分（占位符）
            diversity = 1.0  # 初始值
            
            # 时效性得分
            freshness = np.exp(-0.1 * (user['timestamp'] - candidate['timestamp']))
            
            features.append([semantic, collaborative, diversity, freshness])
        return np.array(features)

    def _calculate_scores(self, features: np.ndarray) -> np.ndarray:
        """加权融合得分"""
        return np.dot(features, list(self.weights.values()))

    def _apply_diversity(self, scores: np.ndarray, candidates: List[Dict]) -> np.ndarray:
        """应用多样性优化"""
        similarity_matrix = self._build_similarity_matrix(candidates)
        return scores * (1 - self.config.diversity.lambda_ * similarity_matrix.max(axis=1))

    def _build_similarity_matrix(self, candidates: List[Dict]) -> np.ndarray:
        """构建物品相似度矩阵"""
        embeddings = np.array([c['embedding'] for c in candidates])
        return cosine_similarity(embeddings)

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def mmr_rerank(self, candidates: List[Dict], lambda_param: float = 0.5) -> List[Dict]:
        """最大边际相关性重排序"""
        selected = []
        remaining = candidates.copy()
        
        while remaining:
            # 计算每个候选的MMR得分
            scores = [
                (lambda_param * candidate['score'] - 
                 (1 - lambda_param) * max([self._cosine_sim(candidate['embedding'], s['embedding']) 
                                        for s in selected] or ))
                for candidate in remaining
            ]
            
            # 选择最高分候选
            best_idx = np.argmax(scores)
            selected.append(remaining.pop(best_idx))
            
        return selected

if __name__ == "__main__":
    from configs import load_config
    import numpy as np
    
    # 示例配置
    config = load_config()
    ranker = HybridRanker(config)
    
    # 模拟候选数据
    mock_candidates = [{
        'item_id': f'N{i}',
        'semantic_score': np.random.uniform(0.3, 1.0),
        'embedding': np.random.randn(256),
        'timestamp': 1620000000 + i*10000
    } for i in range(100)]
    
    # 模拟用户画像
    mock_user = {
        'embedding': np.random.randn(256),
        'timestamp': 1620000000 + 100000
    }
    
    # 执行重排序
    sorted_items = ranker.rerank(mock_candidates, mock_user)
    print("Top5推荐结果:")
    for item in sorted_items[:5]:
        print(f"ID: {item['item_id']} | 得分: {item['semantic_score']:.3f}")
import yaml
from tqdm import tqdm
import logging
from typing import Dict, Any
from evaluation.metrics.accuracy import AccuracyMetric
from evaluation.metrics.diversity import DiversityMetric
from evaluation.metrics.novelty import NoveltyMetric
from evaluation.metrics.coverage import CoverageMetric

class OfflineEvaluator:
    def __init__(self, config_path: str = "configs/eval_config.yaml"):
        self.config = self._load_config(config_path)
        self._init_metrics()
        self._load_data()
        
    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_metrics(self):
        self.metrics = {
            'accuracy': AccuracyMetric(top_ks=self.config['accuracy']['top_ks']),
            'diversity': DiversityMetric(item_embeddings=self.item_embeddings),
            'novelty': NoveltyMetric(item_popularity=self.item_popularity),
            'coverage': CoverageMetric(catalog_size=self.config['coverage']['catalog_size'])
        }
        
    def _load_data(self):
        """加载测试数据集和必要辅助数据"""
        # 示例加载逻辑
        self.test_data = {}  # {user_id: [ground_truth_items]}
        self.item_embeddings = {}  # {item_id: np.array}
        self.item_popularity = Counter()  # {item_id: count}
        
    def evaluate(self, recommender: Any) -> Dict[str, Any]:
        """执行完整评估流程"""
        for user_id, ground_truth in tqdm(self.test_data.items(), 
                                         desc="Processing Users"):
            try:
                # 生成推荐结果
                recs = recommender.recommend(user_id, topk=self.config['max_topk'])
                
                # 更新所有指标
                self.metrics['accuracy'].update(recs, ground_truth, user_id)
                self.metrics['diversity'].update([recs])
                self.metrics['novelty'].update([recs])
                self.metrics['coverage'].update(recs)
                
            except Exception as e:
                logging.error(f"Error evaluating user {user_id}: {str(e)}")
                
        return {
            name: metric.compute() 
            for name, metric in self.metrics.items()
        }
# core/recommender.py
import logging
import time
from typing import Dict, List, Optional
import numpy as np
from omegaconf import DictConfig
from .data_processor import MultimodalProcessor
from .diffusion_model import DiffusionRecommender
from .llm_agent import LLMReasoningEngine
from .fusion_module import HybridRanker

class GenerationPipeline:
    def __init__(self, config: DictConfig):
        self.config = config
        self._init_components()
        self._setup_caches()
        logging.basicConfig(level=logging.INFO)

    def _init_components(self):
        """初始化所有子系统"""
        self.data_processor = MultimodalProcessor(self.config)
        self.diffusion_engine = DiffusionRecommender(self.config)
        self.llm_engine = LLMReasoningEngine(self.config)
        self.ranker = HybridRanker(self.config)
        
        logging.info("所有组件初始化完成")

    def _setup_caches(self):
        """设置缓存系统"""
        self.user_profile_cache = {}
        self.item_feature_cache = {}
        self.hot_items = self._load_hot_items()

    def _load_hot_items(self) -> List[Dict]:
        """加载热门物品作为冷启动备用"""
        # 示例实现，实际应从数据库读取
        return [{
            "item_id": "hot_123",
            "title": "热门新闻",
            "abstract": "当前最受关注的新闻内容...",
            "embedding": np.random.randn(2816)
        }]

    def get_user_profile(self, user_id: str) -> Dict:
        """获取或构建用户画像"""
        if user_id in self.user_profile_cache:
            return self.user_profile_cache[user_id]
        
        # 从数据库加载原始数据
        raw_profile = self._load_raw_profile(user_id)
        processed = self.data_processor.build_user_profiles([raw_profile])
        self.user_profile_cache[user_id] = processed
        return processed

    def realtime_recommend(self, user_id: str, context: Optional[Dict] = None) -> List[Dict]:
        """实时推荐主流程"""
        start_time = time.time()
        
        try:
            # 获取用户画像
            user_profile = self.get_user_profile(user_id)
            
            # 生成候选
            candidates = self._generate_candidates(user_profile)
            
            # 语义增强
            enhanced = self._enhance_candidates(candidates, user_profile)
            
            # 融合排序
            ranked = self._rank_candidates(enhanced, user_profile)
            
            # 添加系统信息
            for item in ranked:
                item['latency'] = time.time() - start_time
                
            return ranked[:self.config.system.topk]
            
        except Exception as e:
            logging.error(f"推荐失败: {str(e)}")
            return self._fallback_recommend()

    def _generate_candidates(self, user_profile: Dict) -> List[Dict]:
        """生成初始候选集"""
        # 扩散模型生成
        diffusion_candidates = self.diffusion_engine.generate_candidates(
            user_emb=user_profile['embedding'],
            topk=self.config.diffusion.topk
        )
        
        # 添加缓存特征
        return [{
            **cand,
            'embedding': self.item_feature_cache.get(cand['item_id'], np.zeros(2816))
        } for cand in diffusion_candidates]

    def _enhance_candidates(self, candidates: List[Dict], user: Dict) -> List[Dict]:
        """LLM语义增强"""
        return self.llm_engine.generate_reasons(
            candidates=candidates,
            user_profile=user
        )

    def _rank_candidates(self, candidates: List[Dict], user: Dict) -> List[Dict]:
        """多维度融合排序"""
        return self.ranker.rerank(
            candidates=candidates,
            user_profile=user
        )

    def _fallback_recommend(self) -> List[Dict]:
        """降级推荐策略"""
        return self.hot_items[:self.config.system.fallback_topk]

    def process_feedback(self, user_id: str, feedback: Dict):
        """处理实时反馈"""
        # 记录用户行为
        logging.info(f"用户 {user_id} 反馈: {feedback}")
        
        # 短期画像更新
        if user_id in self.user_profile_cache:
            self._update_profile(user_id, feedback)
            
        # 触发模型增量更新（示例）
        if feedback.get('retrain_signal'):
            self._trigger_retrain()

    def _update_profile(self, user_id: str, feedback: Dict):
        """实时更新用户画像"""
        profile = self.user_profile_cache[user_id]
        profile['history'].append(feedback['item_id'])
        profile['embedding'] = self._moving_average(
            profile['embedding'],
            self.item_feature_cache[feedback['item_id']],
            weight=0.1
        )

    def _moving_average(self, base: np.ndarray, new: np.ndarray, weight: float) -> np.ndarray:
        """滑动平均更新向量"""
        return (1 - weight) * base + weight * new

    def _trigger_retrain(self):
        """触发模型增量训练（示例）"""
        logging.warning("模型重训练信号触发")
        # 实际应提交训练任务到任务队列

if __name__ == "__main__":
    from configs import load_config
    
    # 初始化配置
    config = load_config()
    pipeline = GenerationPipeline(config)
    
    # 模拟用户请求
    mock_user = "user_123"
    recommendations = pipeline.realtime_recommend(mock_user)
    
    print("\n实时推荐结果:")
    for i, rec in enumerate(recommendations[:5]):
        print(f"{i+1}. {rec['title']}")
        print(f"   理由: {rec['reason']}")
        print(f"   得分: {rec.get('score', 0):.2f}")
        print("-" * 50)
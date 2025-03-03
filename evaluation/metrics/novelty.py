from typing import List, Dict
from collections import Counter

class NoveltyMetric:
    def __init__(self, item_popularity: Dict[str, float]):
        self.item_popularity = item_popularity
        
    def average_popularity(self, recommendations: List[List[str]]) -> float:
        """计算推荐物品的平均流行度"""
        total_pop = 0.0
        total_items = 0
        
        for rec_list in recommendations:
            for item in rec_list:
                total_pop += self.item_popularity.get(item, 0)
                total_items += 1
                
        return total_pop / total_items if total_items > 0 else 0
    
    def unexpectedness(self, recommendations: List[List[str]], 
                      historical_interactions: Dict[str, List[str]]) -> float:
        """计算推荐物品的意外性"""
        unexpected_count = 0
        total_items = 0
        
        for user_id, rec_list in recommendations.items():
            user_history = set(historical_interactions.get(user_id, []))
            for item in rec_list:
                if item not in user_history:
                    unexpected_count += 1
                total_items += 1
                
        return unexpected_count / total_items if total_items > 0 else 0
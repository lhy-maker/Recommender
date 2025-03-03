from typing import List, Dict

class CoverageMetric:
    def __init__(self, catalog_size: int):
        self.catalog_size = catalog_size
        self.covered_items = set()
        
    def update(self, recommendations: List[str]):
        self.covered_items.update(recommendations)
        
    def catalog_coverage(self) -> float:
        return len(self.covered_items) / self.catalog_size
    
    def gini_coefficient(self, recommendation_counts: Dict[str, int]) -> float:
        """计算推荐分布的基尼系数"""
        counts = sorted(recommendation_counts.values())
        n = len(counts)
        cum_sum = sum(counts)
        
        if cum_sum == 0:
            return 0.0
            
        numerator = sum((i+1)*counts[i] for i in range(n))
        return (2 * numerator) / (n * cum_sum) - (n + 1)/n
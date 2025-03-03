import numpy as np
from scipy import stats
from typing import Dict, List

class ABTester:
    def __init__(self, variants: List[Dict[str, Any]]):
        self.variants = variants
        self.results = {v['name']: {'samples': []} for v in variants}
        
    def add_sample(self, variant_name: str, metric_value: float):
        self.results[variant_name]['samples'].append(metric_value)
        
    def analyze_results(self, metric_name: str) -> Dict[str, Any]:
        """执行统计显著性检验"""
        analysis = {}
        baseline = None
        
        for name, data in self.results.items():
            samples = data['samples']
            analysis[name] = {
                'mean': np.mean(samples),
                'std': np.std(samples),
                'n': len(samples)
            }
            
            if baseline is None:
                baseline = name
            else:
                # 执行t检验
                t_stat, p_value = stats.ttest_ind(
                    self.results[baseline]['samples'],
                    samples
                )
                analysis[name]['p_value'] = p_value
                analysis[name]['significant'] = p_value < 0.05
                
        return {
            'metric': metric_name,
            'analysis': analysis
        }
# core/diffusion_model.py
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional
from omegaconf import DictConfig
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ConditionalDiffuser(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.cfg = config
        self.device = torch.device(config.device)
        self._build_network()
        self._init_scheduler()
        
    def _build_network(self):
        """构建条件扩散网络"""
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(2816, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024)
        )
        
        # 时间步编码
        self.time_embed = nn.Embedding(1000, 256)
        
        # 去噪网络
        encoder_layers = TransformerEncoderLayer(
            d_model=1024, 
            nhead=8,
            dim_feedforward=2048
        )
        self.denoiser = TransformerEncoder(encoder_layers, num_layers=6)
        
        # 最终投影层
        self.output_proj = nn.Linear(1024, 2816)
        
    def _init_scheduler(self):
        """初始化噪声调度器"""
        self.num_timesteps = self.cfg.scheduler.num_train_timesteps
        beta = np.linspace(
            self.cfg.scheduler.beta_start,
            self.cfg.scheduler.beta_end,
            self.num_timesteps
        )
        self.beta = torch.from_numpy(beta).float().to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """前向扩散过程"""
        # 编码条件
        cond_emb = self.condition_encoder(condition)
        
        # 时间步嵌入
        t_emb = self.time_embed(t)
        
        # 拼接输入
        x = x + cond_emb + t_emb.unsqueeze(1)
        
        # 去噪过程
        x = self.denoiser(x)
        return self.output_proj(x)
    
    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_samples: int = 100,
        guidance_scale: float = 3.0,
        progress: bool = True
    ) -> torch.Tensor:
        """条件生成采样"""
        shape = (num_samples, 2816)  # 匹配特征维度
        x = torch.randn(shape, device=self.device)
        
        iterator = tqdm(
            reversed(range(self.num_timesteps)),
            desc="Sampling",
            disable=not progress
        )
        
        for t in iterator:
            ts = torch.full((num_samples,), t, device=self.device)
            model_out = self.forward(x, condition, ts)
            
            if guidance_scale > 0:
                # 分类器自由引导
                uncond_out = self.forward(x, torch.zeros_like(condition), ts)
                model_out = uncond_out + guidance_scale * (model_out - uncond_out)
            
            # DDIM更新规则
            alpha_bar = self.alpha_bar[t]
            eps = (x - torch.sqrt(alpha_bar) * model_out) / torch.sqrt(1 - alpha_bar)
            x = torch.sqrt(alpha_bar) * model_out + torch.sqrt(1 - alpha_bar) * eps
            
        return x

class DiffusionRecommender:
    def __init__(self, config: DictConfig):
        self.config = config
        self.model = self._load_pretrained()
        self.model.eval()
        self.index = self._build_index()
        
    def _load_pretrained(self) -> ConditionalDiffuser:
        """加载预训练模型"""
        model = ConditionalDiffuser(self.config)
        state_dict = torch.load(self.config.model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        return model.to(self.config.device)
    
    def _build_index(self) -> Dict[str, np.ndarray]:
        """构建物品特征索引"""
        # 从预处理结果加载
        return np.load(self.config.index_path, allow_pickle=True).item()
    
    def _knn_search(self, query: np.ndarray, k: int = 100) -> List[str]:
        """近似最近邻搜索"""
        # 使用FAISS加速
        index = faiss.IndexFlatIP(2816)
        index.add(np.stack(list(self.index.values())))
        distances, indices = index.search(query, k)
        return [list(self.index.keys())[i] for i in indices]
    
    def generate_candidates(
        self,
        user_emb: np.ndarray,
        topk: int = 100,
        condition_scale: float = 3.0
    ) -> List[Dict]:
        """生成推荐候选"""
        # 转换输入格式
        condition = torch.from_numpy(user_emb).float().to(self.config.device)
        
        # 条件生成
        generated = self.model.sample(
            condition.unsqueeze(0),
            num_samples=1,
            guidance_scale=condition_scale
        )
        
        # 转换为numpy
        generated_emb = generated.cpu().numpy()
        
        # 检索相似物品
        item_ids = self._knn_search(generated_emb, k=topk)
        
        return [{
            "item_id": item_id,
            "similarity": float(self._cosine_sim(generated_emb, self.index[item_id])),
            "features": self.index[item_id]
        } for item_id in item_ids]
    
    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == "__main__":
    # 示例用法
    from configs import load_config
    
    config = load_config(["diffusion=latent"])
    recommender = DiffusionRecommender(config.diffusion)
    
    # 模拟用户特征
    user_emb = np.random.randn(2816)
    
    # 生成推荐
    candidates = recommender.generate_candidates(user_emb)
    print(f"生成 {len(candidates)} 个候选推荐")
    print("Top5推荐:")
    for item in candidates[:5]:
        print(f"- {item['item_id']} (相似度: {item['similarity']:.3f})")
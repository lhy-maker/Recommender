import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModel
from torch import nn
from tqdm import tqdm
from configs import load_config

class MindDataProcessor:
    def __init__(self, config_path: str = "configs/mind.yaml"):
        self.cfg = load_config(config_path)
        self.device = torch.device(self.cfg.system.device)
        self._init_text_encoder()
        self._load_entity_embeddings()
        self._prepare_dirs()
        
        # 加载核心数据
        self.news_df = self._load_news_data()
        self.behavior_df = self._load_behavior_data()

    def _init_text_encoder(self):
        """初始化文本编码模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.text_encoder.name)
        self.text_model = AutoModel.from_pretrained(
            self.cfg.text_encoder.name
        ).to(self.device).eval()

    def _load_entity_embeddings(self) -> Dict[str, np.ndarray]:
        """加载实体嵌入向量"""
        entity_path = Path(self.cfg.data.path) / "entity_embedding.vec"
        embeddings = {}
        with open(entity_path, 'r') as f:
            for line in tqdm(f, desc="加载实体嵌入"):
                parts = line.strip().split()
                if len(parts) < 2: continue
                entity_id = parts
                vector = np.array(parts[1:], dtype=np.float32)
                embeddings[entity_id] = vector
        self.entity_embeddings = embeddings
        return embeddings

    def _load_news_data(self) -> pd.DataFrame:
        """加载新闻元数据"""
        news_path = Path(self.cfg.data.path) / "news.tsv"
        df = pd.read_csv(news_path, sep='\t', 
                        names=['news_id', 'title', 'abstract', 'entities', 'publish_time', 'category'])
        
        # 解析实体列表
        df['entity_list'] = df.entities.str.strip("[]").str.split(',')
        
        # 生成实体嵌入
        df['entity_emb'] = df.entity_list.apply(
            lambda x: np.mean([self.entity_embeddings.get(e, np.zeros(100)) 
                              for e in x], axis=0)
        )
        return df

    def _load_behavior_data(self) -> pd.DataFrame:
        """加载用户行为数据"""
        behavior_path = Path(self.cfg.data.path) / "behaviors.tsv"
        df = pd.read_csv(behavior_path, sep='\t',
                        names=['user_id', 'timestamp', 'news_sequence', 'click_sequence'])
        
        # 解析新闻序列
        df['news_list'] = df.news_sequence.str.strip("[]").str.split(',')
        df['click_labels'] = df.click_sequence.str.strip("[]").str.split(',').apply(
            lambda x: list(map(int, x)))
        return df

    def _text_to_embedding(self, text: str) -> np.ndarray:
        """生成文本特征向量"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.text_model(**inputs)
        return outputs.last_hidden_state[:,0,:].cpu().numpy()

    def process_news_features(self):
        """生成新闻多模态特征"""
        features = {}
        for _, row in tqdm(self.news_df.iterrows(), total=len(self.news_df), desc="处理新闻"):
            # 文本特征
            text_emb = self._text_to_embedding(f"{row['title']} [SEP] {row['abstract']}")
            
            # 实体特征
            entity_emb = row['entity_emb']
            
            # 特征融合
            combined_emb = np.concatenate([
                text_emb.squeeze(),
                entity_emb
            ], axis=0)
            
            features[row['news_id']] = {
                "text_emb": text_emb,
                "entity_emb": entity_emb,
                "combined_emb": combined_emb,
                "category": row['category']
            }
        self.news_features = features
        return features

    def build_user_profiles(self):
        """构建用户画像"""
        user_profiles = {}
        for user_id, group in tqdm(self.behavior_df.groupby('user_id'), desc="构建用户画像"):
            # 获取点击新闻的特征
            clicked_news = [
                news_id 
                for news_ids, clicks in zip(group.news_list, group.click_labels)
                for news_id, click in zip(news_ids, clicks) 
                if click == 1
            ]
            
            # 聚合特征
            if len(clicked_news) > 0:
                embeddings = [self.news_features[n]['combined_emb'] for n in clicked_news if n in self.news_features]
                user_emb = np.mean(embeddings, axis=0) if embeddings else self._cold_start()
            else:
                user_emb = self._cold_start()
            
            user_profiles[user_id] = user_emb
        return user_profiles

    def _cold_start(self) -> np.ndarray:
        """冷启动处理"""
        return np.concatenate([
            np.zeros(768),  # 文本特征维度
            np.zeros(100)   # 实体特征维度
        ])

    def _prepare_dirs(self):
        """创建输出目录"""
        os.makedirs(self.cfg.data.processed_path, exist_ok=True)

if __name__ == "__main__":
    # 示例用法
    processor = MindDataProcessor()
    
    # 处理新闻特征
    news_features = processor.process_news_features()
    
    # 构建用户画像
    user_profiles = processor.build_user_profiles()
    
    # 保存处理结果
    processor.save_processed_data()
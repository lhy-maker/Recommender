# 数据集方案：Hybrid-MMRec Dataset v1.0
1. 核心数据源
基础数据集：Microsoft News Recommendation Dataset (MIND)
    下载地址：https://msnews.github.io/
选择理由：
 + 包含1,000,000+新闻阅读记录
 + 同时具备文本内容（标题/摘要）和图片URL
 + 用户隐式反馈数据（点击/未点击）

> 增强数据：
 >+ 文本增强：HuggingFace Datasets中的CNN_dailymail（新闻长文本）
 >+ 图像增强：Unsplash公开图片库（补充失效图片链接）

```
├── configs/                  # 配置文件
│   ├── diffusion.yaml        # 扩散模型参数
│   └── llm.yaml              # 大语言模型参数
│
├── core/                     # 核心逻辑
│   ├── data_processor.py     # 数据预处理模块
│   ├── diffusion_model.py    # 扩散模型接口
│   ├── llm_agent.py          # 大语言模型接口
│   ├── fusion_module.py      # 多模态融合模块
│   └── recommender.py        # 推荐生成主逻辑
│
├── evaluation/               # 评估模块
│   ├── quality_metrics.py    # 生成质量评估
│   └── relevance_metrics.py  # 推荐相关性评估
│
├── utils/                    # 工具函数
│   ├── api_connector.py      # 外部API连接
│   └── logging.py            # 日志系统
│
└── main.py                   # 系统入口
```
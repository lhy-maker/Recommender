# main.py
import os
import sys
import signal
import argparse
import logging
from typing import Optional
from omegaconf import OmegaConf, DictConfig
from utils.log.logger import setup_logger
from utils.path import PathManager
from core.recommender import GenerationPipeline
from evaluation.testers.offline_tester import OfflineEvaluator

# 全局变量
global_config: Optional[DictConfig] = None
recommender: Optional[GenerationPipeline] = None

def signal_handler(sig, frame):
    """处理中断信号"""
    logging.info("接收到终止信号，清理资源...")
    if recommender is not None:
        # 执行清理操作
        pass
    sys.exit(0)

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多模态推荐系统入口")
    
    # 主运行模式
    parser.add_argument("mode", choices=["train", "recommend", "evaluate", "api"],
                        help="运行模式: 训练/推荐/评估/启动API服务")
    
    # 配置参数
    parser.add_argument("-c", "--config", default="configs/base.yaml",
                       help="主配置文件路径")
    parser.add_argument("--overrides", nargs="+", default=[],
                       help="配置覆写参数，格式为 key=value")
    
    # 推荐相关参数
    parser.add_argument("-u", "--user-id", help="需要推荐的用户ID")
    parser.add_argument("-o", "--output", default="results.json",
                       help="推荐结果输出路径")
    
    # 训练参数
    parser.add_argument("--resume", action="store_true",
                       help="从检查点恢复训练")
    parser.add_argument("--dist-url", default="env://",
                       help="分布式训练URL")
    
    return parser.parse_args()

def initialize_system(config_path: str, overrides: list) -> DictConfig:
    """系统初始化流程"""
    global global_config
    
    # 加载基础配置
    base_conf = OmegaConf.load(config_path)
    
    # 加载环境变量覆盖
    env_conf = OmegaConf.from_env_vars()
    
    # 命令行覆写
    cli_conf = OmegaConf.from_dotlist(overrides)
    
    # 合并配置
    global_config = OmegaConf.merge(base_conf, env_conf, cli_conf)
    
    # 初始化路径系统
    path_manager = PathManager(global_config)
    
    # 配置日志系统
    logger = setup_logger(
        name=global_config.system.logger.name,
        log_dir=path_manager.get_path("log_dir")
    )
    logging.getLogger().setLevel(global_config.system.logger.level)
    
    # 注册中断信号
    signal.signal(signal.SIGINT, signal_handler)
    
    return global_config

def run_training(config: DictConfig):
    """训练模式主流程"""
    logging.info("初始化训练环境...")
    
    # 分布式初始化
    if config.system.distributed:
        from utils.parallel.distributed import init_distributed_mode
        init_distributed_mode(config)
    
    # 初始化推荐系统
    global recommender
    recommender = GenerationPipeline(config)
    
    logging.info("开始模型训练...")
    # 训练逻辑实现...
    
def run_recommendation(config: DictConfig, user_id: str, output_path: str):
    """推荐模式主流程"""
    logging.info(f"为用户 {user_id} 生成推荐...")
    
    # 初始化推荐系统
    recommender = GenerationPipeline(config)
    
    # 获取推荐结果
    results = recommender.realtime_recommend(user_id)
    
    # 保存结果
    from utils.io.data_loader import DataLoader
    DataLoader.save_json(results, output_path)
    logging.info(f"推荐结果已保存至 {output_path}")

def run_evaluation(config: DictConfig):
    """评估模式主流程"""
    logging.info("初始化评估系统...")
    
    # 初始化评估器和推荐系统
    evaluator = OfflineEvaluator(config)
    recommender = GenerationPipeline(config)
    
    # 执行评估
    metrics = evaluator.evaluate(recommender)
    
    # 生成报告
    from evaluation.visualizer.report_generator import ReportGenerator
    ReportGenerator().generate_html_report(
        metrics,
        config.evaluation.report_path
    )
    logging.info(f"评估报告已生成于 {config.evaluation.report_path}")

def start_api_server(config: DictConfig):
    """启动API服务"""
    from fastapi import FastAPI
    app = FastAPI(title=config.api.name)
    
    @app.on_event("startup")
    def init_recommender():
        global recommender
        recommender = GenerationPipeline(config)
    
    @app.get("/recommend/{user_id}")
    async def recommend(user_id: str, topk: int = 10):
        return recommender.realtime_recommend(user_id)[:topk]
    
    import uvicorn
    uvicorn.run(app, host=config.api.host, port=config.api.port)

def main():
    args = parse_args()
    
    try:
        # 初始化系统
        config = initialize_system(args.config, args.overrides)
        
        # 执行对应模式
        if args.mode == "train":
            run_training(config)
        elif args.mode == "recommend":
            if not args.user_id:
                raise ValueError("推荐模式需要指定用户ID")
            run_recommendation(config, args.user_id, args.output)
        elif args.mode == "evaluate":
            run_evaluation(config)
        elif args.mode == "api":
            start_api_server(config)
            
    except Exception as e:
        logging.critical(f"系统运行失败: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        # 资源清理逻辑
        pass

if __name__ == "__main__":
    main()
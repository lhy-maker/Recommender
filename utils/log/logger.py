import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logger(name: str, log_dir: str = "./logs"):
    """配置全局日志记录器"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # 文件Handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f"{log_dir}/{name}_{timestamp}.log")
    file_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(name)s: %(message)s'
    ))
    
    # 控制台Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s - %(message)s'
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
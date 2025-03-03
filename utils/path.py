import os
from pathlib import Path
from typing import Union
from omegaconf import DictConfig

class PathManager:
    def __init__(self, config: DictConfig):
        self.config = config
        self._verify_paths()
        
    def _verify_paths(self):
        """创建必要目录"""
        Path(self.config.system.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.system.cache_dir).mkdir(parents=True, exist_ok=True)
        
    def get_path(self, path_type: str) -> str:
        """获取配置路径"""
        return os.path.join(
            self.config.system.base_dir,
            self.config.system.paths[path_type]
        )
    
    @staticmethod
    def resolve_path(path: Union[str, Path]) -> str:
        """解析相对路径为绝对路径"""
        return str(Path(path).expanduser().resolve())
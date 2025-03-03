# configs/__init__.py
from omegaconf import OmegaConf
import os

_BASE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "base.yaml")

def load_config(overrides=[]):
    base_cfg = OmegaConf.load(_BASE_CONFIG_PATH)
    cli_cfg = OmegaConf.from_cli(overrides)
    return OmegaConf.merge(base_cfg, cli_cfg)

def print_config(cfg):
    print(OmegaConf.to_yaml(cfg))
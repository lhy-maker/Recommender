import torch
from pathlib import Path
from typing import Dict, Union

class ModelIO:
    @staticmethod
    def save_checkpoint(model: torch.nn.Module, 
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       path: Union[str, Path],
                       **kwargs):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            **kwargs
        }
        torch.save(state, path)
        
    @staticmethod
    def load_checkpoint(path: Union[str, Path], 
                       device: str = 'cpu') -> Dict:
        return torch.load(path, map_location=device)
    
    @staticmethod
    def count_parameters(model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
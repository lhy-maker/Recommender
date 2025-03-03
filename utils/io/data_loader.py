import json
import pickle
import pandas as pd
from typing import Any, Dict

class DataLoader:
    @staticmethod
    def load_json(path: str) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_json(data: Any, path: str):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
    @staticmethod
    def load_csv(path: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(path, **kwargs)
    
    @staticmethod
    def save_csv(df: pd.DataFrame, path: str, **kwargs):
        df.to_csv(path, index=False, **kwargs)
        
    @staticmethod
    def load_pickle(path: str) -> Any:
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_pickle(data: Any, path: str):
        with open(path, 'wb') as f:
            pickle.dump(data, f)
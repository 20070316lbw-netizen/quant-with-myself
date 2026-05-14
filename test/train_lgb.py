from utils.config import get_db, load_config
import lightgbm as lgb
import numpy as np
import pandas as pd


cfg = load_config()
model_params = cfg["model"]
train_start = cfg["data"]["splits"]["train"]["start"]

def train_lgb(
        
) -> lgb.Booster:
    pass
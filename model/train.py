import sys
import lightgbm as lgb
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from utils.config import get_db, load_config
from features.make_features import make_features
from labels.make_labels import make_label 

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
cfg = load_config()
model_params = cfg["model"]
train_start = cfg["data"]["splits"]["train"]["start"]

FEATURE_COL = ["mon_1d", "mon_5d", "std_5d"]
LABEL_COL = "label_5d"

def bulid_dataset() -> pd.DataFrame:
    with get_db() as con:
        df = con.execute("""
                        SELECT * FROM prices
                         """).fetchdf()
        df = make_features(df)
        
        
    
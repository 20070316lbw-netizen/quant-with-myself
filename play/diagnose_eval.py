import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
import lightgbm as lgb
from utils.config import get_db, load_config
from features import build_feature_cols, make_features
from labels.make_labels import make_label
from model.train import build_dataset, train_lgb
from labels.processors import build_processors, apply_label_processors

def run_diagnose(use_cs_rank=False):
    cfg = load_config()
    if use_cs_rank:
        cfg["label_processors"] = ["cs_rank"]
    else:
        cfg["label_processors"] = []

    feature_cols = build_feature_cols(cfg)

    with get_db() as con:
        df = con.execute("SELECT * FROM prices ORDER BY ticker, date").fetchdf()
    
    df = make_features(df, feature_cols)
    df["label_5d"] = make_label(df, n_periods=5, gap=1)
    
    train_df, test_df = build_dataset(df, cfg)
    model = train_lgb(train_df, cfg, feature_cols=feature_cols)
    
    # Predict on test set
    feature_cols = model.feature_name()
    test_clean = test_df.dropna(subset=feature_cols + ["label_5d"]).copy()
    test_clean["pred"] = model.predict(test_clean[feature_cols])
    
    print(f"\n=== DIAGNOSING: cs_rank={use_cs_rank} ===")
    print(f"Test clean rows: {len(test_clean)}")
    print(f"Unique dates: {test_clean['date'].nunique()}")
    
    # Analyze daily stock count
    daily_counts = test_clean.groupby("date").size()
    print(f"Daily stock count: min={daily_counts.min()}, max={daily_counts.max()}, mean={daily_counts.mean():.1f}")
    
    # Analyze unique prediction values per day
    daily_unique_preds = test_clean.groupby("date")["pred"].nunique()
    print(f"Daily unique predictions: min={daily_unique_preds.min()}, max={daily_unique_preds.max()}, mean={daily_unique_preds.mean():.1f}")
    
    # Let's count how many days actually fail the qcut or get dropped
    n_groups = 5
    def _grp_of_day(s: pd.Series):
        if len(s) < n_groups * 4:
            return pd.Series(float("nan"), index=s.index)
        try:
            res = pd.qcut(s, n_groups, labels=False, duplicates="drop")
            return res
        except Exception as e:
            return pd.Series(float("nan"), index=s.index)
            
    test_clean["grp"] = test_clean.groupby("date")["pred"].transform(_grp_of_day)
    valid_grp = test_clean.dropna(subset=["grp"])
    print(f"Rows after grp dropna: {len(valid_grp)}")
    print(f"Unique dates with valid grp: {valid_grp['date'].nunique()}")
    
    # Let's see some sample days
    for date, group in test_clean.groupby("date"):
        pred_unique = group["pred"].nunique()
        grp_unique = group["grp"].dropna().nunique()
        if grp_unique < n_groups and grp_unique > 0:
            print(f"Date {date.strftime('%Y-%m-%d')}: stocks={len(group)}, unique_preds={pred_unique}, unique_grps={grp_unique}")
            print(f"Predictions sample: {group['pred'].value_counts().head(5).to_dict()}")
            break

if __name__ == "__main__":
    print("Starting diagnosis for Baseline:")
    run_diagnose(use_cs_rank=False)
    print("\nStarting diagnosis for cs_rank:")
    run_diagnose(use_cs_rank=True)

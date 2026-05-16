# model/train.py —— 注意:模块顶层不读配置、不 sys.path hack

import lightgbm as lgb
import pandas as pd

FEATURE_COLS = ["mom_1d", "mom_5d", "std_5d"]   # 修正拼写
LABEL_COL = "label_5d"

def build_dataset(df: pd.DataFrame, cfg: dict) -> tuple:
    """
    df: 已经过 make_features + make_label 的全量长格式数据
    cfg: 传进来的配置(不在这里读文件)
    返回: (train_df, test_df) —— 按配置的时间边界切好
    """
    splits = cfg["data"]["splits"]
    tr = splits["train"]
    te = splits["test"]

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 这就是"配置的日期"和"训练"对齐的那一行:
    # 配置只声明边界,这里用边界切数据
    train_df = df[(df["date"] >= tr["start"]) & (df["date"] <= tr["end"])]
    test_df  = df[(df["date"] >= te["start"]) & (df["date"] <= te["end"])]

    return train_df, test_df


def train_lgb(train_df: pd.DataFrame, cfg: dict) -> lgb.Booster:
    """只负责训练。配置从参数来,不自己读。"""
    X = train_df[FEATURE_COLS]
    y = train_df[LABEL_COL]

    # 关键:特征/标签算出来的 NaN 行(窗口没填满 + 未来不存在)必须先丢
    # 否则 LightGBM 虽然能处理 NaN 特征,但 NaN 的 label 会污染训练
    mask = y.notna() & X.notna().all(axis=1)
    X, y = X[mask], y[mask]

    dataset = lgb.Dataset(X, label=y)
    model = lgb.train(cfg["model"], dataset)
    return model
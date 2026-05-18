# model/train.py —— 注意:模块顶层不读配置、不 sys.path hack

import lightgbm as lgb
import pandas as pd

from features.make_features import FEATURE_COLS   # single source of truth,不再手抄
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
    embargo_days = splits.get("embargo", {}).get("days", 0)

    # 验证隔离带长度，避免未来函数泄漏（label_5d = gap 1 + n_periods 5 = 6 交易日）
    assert embargo_days >= 6, f"隔离带必须 >= 6 个交易日以防泄露，当前设置为 {embargo_days}"

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 获取所有实际的交易日，并计算真实的 test.start
    trading_days = pd.Series(df["date"].unique()).sort_values().tolist()
    train_end_ts = pd.to_datetime(tr["end"])
    
    # 找到小于等于 train_end 的最后一个交易日作为实际的结束点
    actual_train_ends = [d for d in trading_days if d <= train_end_ts]
    if actual_train_ends:
        idx = trading_days.index(actual_train_ends[-1])
        # 往后数 embargo_days 个交易日得到 test.start
        if idx + embargo_days < len(trading_days):
            test_start_ts = trading_days[idx + embargo_days]
        else:
            raise ValueError(f"加上 {embargo_days} 天隔离带后，超出了数据总范围！")
    else:
        # 如果历史数据里没有找到对应的结束日期，退回默认行为
        test_start_ts = pd.to_datetime(te["start"])

    # 这就是"配置的日期"和"训练"对齐的那一行:
    # 训练集: 落在配置指定的训练范围内
    train_df = df[(df["date"] >= pd.to_datetime(tr["start"])) & (df["date"] <= train_end_ts)]
    
    # 测试集: 起点是动态算出的 test_start_ts，终点是配置的值
    test_df  = df[(df["date"] >= test_start_ts) & (df["date"] <= pd.to_datetime(te["end"]))]

    return train_df, test_df

def train_lgb(train_df: pd.DataFrame, cfg: dict) -> lgb.Booster:
    """只负责训练。配置从参数来,不自己读。"""
    X = train_df[FEATURE_COLS]
    y = train_df[LABEL_COL]

    # 只丢 NaN label。NaN 特征不丢:LightGBM 原生支持缺失值,会为缺失值学一个
    # 默认分裂方向。rolling 因子(std_5d/KLEN 等)每只股票开头天然是 NaN,
    # 若连特征 NaN 行一起丢,会白白损失约 15% 的早期样本。
    # 之前代码是 
    # `mask = y.notna() & X.notna().all(axis=1)`
    # std_5d、KLEN 这些是 rolling 窗口算的,每只股票最早那几十行天然是 NaN,这半段会把它们全删掉
    # 所以删掉后半段就可以
    mask = y.notna()
    X, y = X[mask], y[mask]

    dataset = lgb.Dataset(X, label=y)
    model = lgb.train(cfg["model"], dataset)
    return model
# model/train.py —— 注意:模块顶层不读配置、不 sys.path hack

import lightgbm as lgb
import pandas as pd
import numpy as np

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

def train_lgb(train_df: pd.DataFrame, cfg: dict, return_history: bool = False):
    """
    训练 + 时间切验证集 + early stopping。配置从参数来。

    为什么验证集要这么切(面试深水区):
    1) 不能随机切——金融数据有时间结构,拿未来验证过去=作弊。
       验证集必须是训练集时间上最靠后的一段。
    2) 训练段与验证段之间还要留 embargo。label_5d 要看未来 6
       个交易日,若两段紧贴,训练集末尾样本的「答案」会覆盖
       到验证期,早停信号被污染——你会以为没过拟合,其实是泄露撑着。
    3) 这套 embargo 逻辑与 build_dataset 里 train->test 的切法一致,
       不引入新逻辑,保持口径统一。
    """
    tc = cfg.get("train_control", {})
    embargo = cfg["data"]["splits"].get("embargo", {}).get("days", 6)
    valid_frac = tc.get("valid_frac", 0.2)
    num_round = tc.get("num_boost_round", 1000)
    stop_round = tc.get("early_stopping", 50)

    df = train_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 按交易日排序,从末尾切出 valid_frac 做验证,中间留 embargo
    days = np.sort(df["date"].unique())
    n_valid = max(1, int(len(days) * valid_frac))
    valid_days = days[-n_valid:]                         # 末尾一段做验证
    train_days = days[: len(days) - n_valid - embargo]   # 前段做训练,中间空 embargo
    if len(train_days) == 0:
        raise ValueError(
            f"训练日为空:总{len(days)}天 - 验证{n_valid}天 - embargo{embargo}天 <= 0"
        )

    tr = df[df["date"].isin(train_days)]
    va = df[df["date"].isin(valid_days)]

    # 只丢 NaN label。NaN 特征不丢:LightGBM 原生支持缺失值,会为缺失值学一个
    # 默认分裂方向。rolling 因子(std_5d/KLEN 等)每只股票开头天然是 NaN,
    # 若连特征 NaN 行一起丢,会白白损失早期样本。
    # 之前代码是 `mask = y.notna() & X.notna().all(axis=1)`,后半段会误删。
    def _xy(frame):
        m = frame[LABEL_COL].notna()
        return frame[FEATURE_COLS][m], frame[LABEL_COL][m]

    Xtr, ytr = _xy(tr)
    Xva, yva = _xy(va)

    dtrain = lgb.Dataset(Xtr, label=ytr)
    dvalid = lgb.Dataset(Xva, label=yva, reference=dtrain)

    # record_evaluation 把每轮 train/valid 误差存进 history,用于画过拟合曲线
    history = {}
    model = lgb.train(
        cfg["model"],
        dtrain,
        num_boost_round=num_round,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=stop_round),
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(history),
        ],
    )
    if return_history:
        return model, history
    return model
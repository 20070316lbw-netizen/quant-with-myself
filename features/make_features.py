"""
目前这里只做测试,完整的特征采用以下特征:
FEATURE_COLS = [
    "KMID", "KLEN", "KMID2", "KUP", "KLOW", "KSFT",
    "ROC5", "ROC20", "ROC60",
    "MA5", "MA20", "MA60",
    "STD5", "STD20", "STD60",
]

解释:
---
kbar 类(6 个）—— 描述当日多空力量博弈

KMID (实体涨跌幅）：今天总体是涨是跌、力度多大。最直接的当日强弱信号。
KLEN (振幅）：今天的波动有多剧烈。高振幅可能意味着不确定性增加或情绪激烈。
KMID(实体占比）：如果实体大、影线小，说明多空一方完胜；如果实体小、影线长，说明拉锯激烈、方向未定。
KUP (上影线）：上方有抛压，冲高回落，看跌信号。
KLOW (下影线）：下方有支撑，探底回升，看涨信号。
KSFT (收盘在区间的位置）：收盘越靠近最高，说明多头掌控；越靠近最低，空头掌控。

KMID  = (close - open) / open
KLEN  = (high - low) / open
KMID2 = (close - open) / (high - low + 1e-12)
KUP   = (high - max(open, close)) / open          # 上影线
KLOW  = (min(open, close) - low) / open           # 下影线
KSFT  = (2*close - high - low) / open             # 收盘在当日区间的相对位置

这些都是技术分析里 K 线形态学的数学化，几十年来交易员就在看这些东西。

---
ROC 类 (3 个）—— 动量因子

ROC5 / ROC20 / ROC60: 过去 5/20/60 天的累计涨跌。

背后逻辑是动量效应 (momentum) 学术界公认的异象之一:过去涨得好的股票,未来一段时间倾向于继续涨;过去跌得惨的, 倾向于继续跌。三个窗口分别对应短期、中期、长期动量。Jegadeesh & Titman 1993 那篇经典论文专门研究的就是这个。
ROC5  = close.shift(5)  / close
ROC20 = close.shift(20) / close
ROC60 = close.shift(60) / close

---
MA 类 (3 个）—— 趋势位置

MA5 / MA20 / MA60 (均线 / 当前价）：当前价相对均线的位置。

如果 MA20/close > 1,说明当前价低于 20 日均线，可能是回调；< 1 则在均线之上，趋势向上。背后是均值回归 vs 趋势跟随的博弈——这个特征本身不预设方向，让模型自己学在什么情况下倾向哪个。
MA5  = close.rolling(5).mean()  / close
MA20 = close.rolling(20).mean() / close
MA60 = close.rolling(60).mean() / close

---
STD 类 (3 个）—— 波动率

STD5 / STD20 / STD60 (标准差 / 当前价）：归一化的波动率。

背后逻辑是低波动异象 (low volatility anomaly): 长期看，低波动股票的风险调整后收益反而高于高波动股票，违反了 CAPM 的预期。这是另一个被广泛研究的异象。
STD5  = close.rolling(5).std()  / close
STD20 = close.rolling(20).std() / close
STD60 = close.rolling(60).std() / close
"""

import pandas as pd
import numpy as np

# ---- 每个特征 = 一个独立纯函数,只做一件事 ----
# 约定:输入排好序的 df,返回一个 Series(和 df 行对齐)

def _mom_1d(df: pd.DataFrame) -> pd.Series:
    return df.groupby('ticker')['close'].pct_change(1)

def _std_5d(df: pd.DataFrame) -> pd.Series:
    # 注意:它依赖 mom_1d 这一列已经存在 —— 这个依赖现在是【显式】的
    return (
        df.groupby('ticker')['mom_1d']
        .rolling(5).std()
        .reset_index(level=0, drop=True)
    )

# KMID  = (close - open) / open
# KLEN  = (high - low) / open
# KUP   = (high - max(open, close)) / open          # 上影线
# KLOW  = (min(open, close) - low) / open           # 下影线
# KSFT  = (2*close - high - low) / open             # 收盘在当日区间的相对位置
# KMID2 = (close - open) / (high - low + 1e-12)
def _kmid(df: pd.DataFrame) -> pd.Series:
    return (df['close'] - df['open']) / df['open']

def _klen(df: pd.DataFrame) -> pd.Series:
    return (df['high'] - df['low']) / df['open']

def _kup(df: pd.DataFrame) -> pd.Series:
    return (df["high"] - np.maximum(df["open"], df["close"])) / df["open"]

def _klow(df: pd.DataFrame) -> pd.Series:
    return (np.minimum(df["open"], df["close"]) - df["low"]) / df["open"]

def _ksft(df: pd.DataFrame) -> pd.Series:
    return (2 * df["close"] - df["high"] - df["low"]) / df["open"]

def _kmid_2(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-12)


# ROC5  = close / close.shift(5)    # 今价/5天前价,涨了 >1(与 J&T1993 惯例一致)
# ROC20 = close / close.shift(20)
# ROC60 = close / close.shift(60)
# 必须 groupby('ticker'):否则 shift 会越过 ticker 边界取到上一只股票的价格
def _roc_5(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df.groupby("ticker")["close"].shift(5)

def _roc_20(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df.groupby("ticker")["close"].shift(20)

def _roc_60(df: pd.DataFrame) -> pd.Series:
    return df["close"] / df.groupby("ticker")["close"].shift(60)


# MA5  = close.rolling(5).mean()  / close
# MA20 = close.rolling(20).mean() / close
# MA60 = close.rolling(60).mean() / close
# 必须 groupby('ticker'):否则 rolling 窗口会跨股票边界混入上一只股票的尾部数据
def _ma_5(df: pd.DataFrame) -> pd.Series:
    ma = df.groupby("ticker")["close"].rolling(5).mean().reset_index(level=0, drop=True)
    return ma / df["close"]

def _ma_20(df: pd.DataFrame) -> pd.Series:
    ma = df.groupby("ticker")["close"].rolling(20).mean().reset_index(level=0, drop=True)
    return ma / df["close"]

def _ma_60(df: pd.DataFrame) -> pd.Series:
    ma = df.groupby("ticker")["close"].rolling(60).mean().reset_index(level=0, drop=True)
    return ma / df["close"]



# STD5  = close.rolling(5).std()  / close
# STD20 = close.rolling(20).std() / close
# STD60 = close.rolling(60).std() / close
# 必须 groupby('ticker'):同 MA,防 rolling 跨股票污染
def _std_5(df: pd.DataFrame) -> pd.Series:
    s = df.groupby("ticker")["close"].rolling(5).std().reset_index(level=0, drop=True)
    return s / df["close"]

def _std_20(df: pd.DataFrame) -> pd.Series:
    s = df.groupby("ticker")["close"].rolling(20).std().reset_index(level=0, drop=True)
    return s / df["close"]

def _std_60(df: pd.DataFrame) -> pd.Series:
    s = df.groupby("ticker")["close"].rolling(60).std().reset_index(level=0, drop=True)
    return s / df["close"]

def _rank_mom_5d(df: pd.DataFrame) -> pd.Series:
    # 自给自足:内部自算 5 日动量再做截面排名,不再依赖 mom_5d 列。
    # 这样 mom_5d 可以从 registry 彻底删掉(它与本因子高度共线)。
    mom5 = df.groupby("ticker")["close"].pct_change(5)
    return mom5.groupby(df["date"]).rank(pct=True)

# ---- 横截面因子:看个股在「当天全市场」里的相对位置 ----
# 设计原则:均基于「已验证干净」的时序因子做截面 rank,
# 不引入新的原始计算——控制变量,出错面最小。
# rank 默认 na_option='keep':源因子为 NaN 的行 rank 后仍是 NaN,
# 不会被错误地排进百分位。必须 groupby("date"):只在同一
# 交易日内比较,绝不跨日期。

def _rank_roc20(df: pd.DataFrame) -> pd.Series:
    # 中期动量的横截面强弱(Jegadeesh-Titman 动量异象)
    return df.groupby("date")["ROC20"].rank(pct=True)

def _rank_std20(df: pd.DataFrame) -> pd.Series:
    # 低波动异象——截面上波动率最低的那批股票
    return df.groupby("date")["STD20"].rank(pct=True)

def _rank_kmid(df: pd.DataFrame) -> pd.Series:
    # 当日实体涨跌的相对强弱(短期反转 / 日内强度)
    return df.groupby("date")["KMID"].rank(pct=True)

def _rank_ma20(df: pd.DataFrame) -> pd.Series:
    # 相对均线位置的截面排序(趋势 vs 回调)
    return df.groupby("date")["MA20"].rank(pct=True)

# ---- 登记表:这就是"做选择"的地方,谁先算谁后算一目了然 ----
# 顺序有意义:std_5d 在 mom_1d 之后,因为它依赖 mom_1d
FEATURE_REGISTRY = {
    "mom_1d": _mom_1d,
    "std_5d": _std_5d,
    "KMID": _kmid,
    # "KMID2": _kmid_2,
    "KLEN": _klen,
    # "KUP": _kup,
    # "KLOW": _klow,
    # "KSFT": _ksft,
    "ROC5": _roc_5,
    "ROC20": _roc_20,
    "ROC60": _roc_60,
    # "MA5": _ma_5,
    # "MA20": _ma_20,
    # "MA60": _ma_60,
    "STD5": _std_5,
    "STD20": _std_20,
    "STD60": _std_60,
    "RANK_MOM_5D": _rank_mom_5d,
    # 横截面 rank 因子——必须注册在其依赖的源因子之后。
    # ROC20/STD20/KMID/MA20 均在上方已注册,make_features 按
    # 此顺序逐个计算,放末尾保证依赖列已存在。
    "RANK_ROC20": _rank_roc20,
    "RANK_STD20": _rank_std20,
    "RANK_KMID": _rank_kmid,
    # "RANK_MA20": _rank_ma20,
}

# ===== single source of truth =====
# 特征名一律从这里取。train.py / evaluate.py 不准再自己手抄一份。
# 取的是 FEATURE_REGISTRY 实际注册的键——以「实际能算的」为准,
# 而不是以文档/记忆为准。以后加因子只需在 REGISTRY 里加一项,
# train 和 evaluate 自动同步,结构上不可能再出现错位。
FEATURE_COLS = list(FEATURE_REGISTRY.keys())

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """按登记表依次计算所有特征,挂回 df"""
    # reset_index(drop=True) 必须加:groupby-rolling 返回 MultiIndex,
    # 内层靠行号对齐回 df。不重置索引的话 sort_values 留下的
    # 乱序原索引会让 ma/df['close'] 的除法对齐出错。
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True).copy()
    for name, fn in FEATURE_REGISTRY.items():
        df[name] = fn(df)
    return df


if __name__ == "__main__":
    # 直接运行本文件时，先把项目根加到 sys.path，才能 import utils
    # 被别的模块 import 时不走这里，不会污染 sys.path
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from utils.config import get_db

    with get_db() as con:
        df = con.execute("SELECT * FROM prices ORDER BY ticker, date").fetchdf()

    df_with_features = make_features(df)

    print(df_with_features.head(10))
    print(df_with_features.tail(10))
    print(f"\n共 {len(df_with_features)} 行")
    print(f"NaN 数量:\n{df_with_features.isnull().sum()}")
    print("(每只股票前几行会有 NaN, 是因为 window 还没填满, 这是正常的)")
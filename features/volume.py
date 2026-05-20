"""
Volume 量价与上涨一致性因子（基于 Alpha158 089, 099, 129 等）。
"""

import pandas as pd
import numpy as np


def _corr5(df: pd.DataFrame) -> pd.Series:
    """
    CORR5: 5日内收盘价与 log(成交量 + 1) 的相关性。
    正值为量价齐升，负值为背离。
    """
    # groupby ticker + group_keys=False apply 保证完美的索引对齐
    res = df.groupby("ticker", group_keys=False).apply(
        lambda x: x["close"].rolling(5).corr(np.log1p(x["volume"]))
    )
    # 针对 pandas 在单 ticker 时可能返回 DataFrame/MultiIndex 的防御性处理
    if isinstance(res, pd.DataFrame):
        res = res.iloc[0]
    return res


def _cntp5(df: pd.DataFrame) -> pd.Series:
    """
    CNTP5: 过去5日中收盘上涨天数的占比。
    反应近期上涨的一致性强弱。
    """
    close_up = (df["close"] > df.groupby("ticker")["close"].shift(1)).astype(float)
    tmp = pd.DataFrame({"ticker": df["ticker"], "close_up": close_up})
    return tmp.groupby("ticker")["close_up"].rolling(5).mean().reset_index(level=0, drop=True)


def _vma5(df: pd.DataFrame) -> pd.Series:
    """
    VMA5: 5日均量 / 当日成交量。
    >1 表示当日成交量低于均量（缩量）。
    """
    mean_v = df.groupby("ticker")["volume"].rolling(5).mean().reset_index(level=0, drop=True)
    return mean_v / (df["volume"] + 1e-12)


VOLUME_REGISTRY = {
    "CORR5": _corr5,
    "CNTP5": _cntp5,
    "VMA5":  _vma5,
}

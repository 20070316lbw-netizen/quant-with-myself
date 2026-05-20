"""
横截面百分位 Rank 特征（基于 Alpha158 核心基础特征）。
每个特征函数均具备自给自足能力：如果 df 中已有预先计算的源时序特征，则直接复用；
否则，动态调用基础子模块进行临时计算，避免强制依赖 features.active。
"""

import pandas as pd


# ---- 动态/缓存源特征获取方法 ----

def _get_kmid(df: pd.DataFrame) -> pd.Series:
    if "KMID" in df.columns:
        return df["KMID"]
    from .kbar import _kmid
    return _kmid(df)


def _get_klen(df: pd.DataFrame) -> pd.Series:
    if "KLEN" in df.columns:
        return df["KLEN"]
    from .kbar import _klen
    return _klen(df)


def _get_roc5(df: pd.DataFrame) -> pd.Series:
    if "ROC5" in df.columns:
        return df["ROC5"]
    from .roc import _roc_5
    return _roc_5(df)


def _get_roc20(df: pd.DataFrame) -> pd.Series:
    if "ROC20" in df.columns:
        return df["ROC20"]
    from .roc import _roc_20
    return _roc_20(df)


def _get_roc60(df: pd.DataFrame) -> pd.Series:
    if "ROC60" in df.columns:
        return df["ROC60"]
    from .roc import _roc_60
    return _roc_60(df)


def _get_std5(df: pd.DataFrame) -> pd.Series:
    if "STD5" in df.columns:
        return df["STD5"]
    from .std import _std_5
    return _std_5(df)


def _get_std20(df: pd.DataFrame) -> pd.Series:
    if "STD20" in df.columns:
        return df["STD20"]
    from .std import _std_20
    return _std_20(df)


def _get_std60(df: pd.DataFrame) -> pd.Series:
    if "STD60" in df.columns:
        return df["STD60"]
    from .std import _std_60
    return _std_60(df)


# ---- 截面 Rank 因子定义 (在当日 date 截面比较，[0, 1] 比例排名) ----

def _rank_kmid(df: pd.DataFrame) -> pd.Series:
    return _get_kmid(df).groupby(df["date"]).rank(pct=True)


def _rank_klen(df: pd.DataFrame) -> pd.Series:
    return _get_klen(df).groupby(df["date"]).rank(pct=True)


def _rank_roc5(df: pd.DataFrame) -> pd.Series:
    return _get_roc5(df).groupby(df["date"]).rank(pct=True)


def _rank_roc20(df: pd.DataFrame) -> pd.Series:
    return _get_roc20(df).groupby(df["date"]).rank(pct=True)


def _rank_roc60(df: pd.DataFrame) -> pd.Series:
    return _get_roc60(df).groupby(df["date"]).rank(pct=True)


def _rank_std5(df: pd.DataFrame) -> pd.Series:
    return _get_std5(df).groupby(df["date"]).rank(pct=True)


def _rank_std20(df: pd.DataFrame) -> pd.Series:
    return _get_std20(df).groupby(df["date"]).rank(pct=True)


def _rank_std60(df: pd.DataFrame) -> pd.Series:
    return _get_std60(df).groupby(df["date"]).rank(pct=True)


CS_RANK_REGISTRY = {
    "RANK_KMID":  _rank_kmid,
    "RANK_KLEN":  _rank_klen,
    "RANK_ROC5":  _rank_roc5,
    "RANK_ROC20": _rank_roc20,
    "RANK_ROC60": _rank_roc60,
    "RANK_STD5":  _rank_std5,
    "RANK_STD20": _rank_std20,
    "RANK_STD60": _rank_std60,
}

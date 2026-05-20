"""
ROC 动量因子（Alpha158 014-018）。
"""

import pandas as pd


def _roc_5(df):   return df["close"] / df.groupby("ticker")["close"].shift(5)
def _roc_10(df):  return df["close"] / df.groupby("ticker")["close"].shift(10)
def _roc_20(df):  return df["close"] / df.groupby("ticker")["close"].shift(20)
def _roc_30(df):  return df["close"] / df.groupby("ticker")["close"].shift(30)
def _roc_60(df):  return df["close"] / df.groupby("ticker")["close"].shift(60)


ROC_REGISTRY = {
    "ROC5":  _roc_5,
    "ROC10": _roc_10,
    "ROC20": _roc_20,
    "ROC30": _roc_30,
    "ROC60": _roc_60,
}

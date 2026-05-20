"""
MA 趋势均线因子（Alpha158 019-023）。
"""

import pandas as pd


def _ma_5(df):
    ma = df.groupby("ticker")["close"].rolling(5).mean().reset_index(level=0, drop=True)
    return ma / df["close"]

def _ma_10(df):
    ma = df.groupby("ticker")["close"].rolling(10).mean().reset_index(level=0, drop=True)
    return ma / df["close"]

def _ma_20(df):
    ma = df.groupby("ticker")["close"].rolling(20).mean().reset_index(level=0, drop=True)
    return ma / df["close"]

def _ma_30(df):
    ma = df.groupby("ticker")["close"].rolling(30).mean().reset_index(level=0, drop=True)
    return ma / df["close"]

def _ma_60(df):
    ma = df.groupby("ticker")["close"].rolling(60).mean().reset_index(level=0, drop=True)
    return ma / df["close"]


MA_REGISTRY = {
    "MA5":  _ma_5,
    "MA10": _ma_10,
    "MA20": _ma_20,
    "MA30": _ma_30,
    "MA60": _ma_60,
}

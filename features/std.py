"""
STD 波动率因子（Alpha158 024-028）。
"""

import pandas as pd


def _std_5(df):
    s = df.groupby("ticker")["close"].rolling(5).std().reset_index(level=0, drop=True)
    return s / df["close"]

def _std_10(df):
    s = df.groupby("ticker")["close"].rolling(10).std().reset_index(level=0, drop=True)
    return s / df["close"]

def _std_20(df):
    s = df.groupby("ticker")["close"].rolling(20).std().reset_index(level=0, drop=True)
    return s / df["close"]

def _std_30(df):
    s = df.groupby("ticker")["close"].rolling(30).std().reset_index(level=0, drop=True)
    return s / df["close"]

def _std_60(df):
    s = df.groupby("ticker")["close"].rolling(60).std().reset_index(level=0, drop=True)
    return s / df["close"]


STD_REGISTRY = {
    "STD5":  _std_5,
    "STD10": _std_10,
    "STD20": _std_20,
    "STD30": _std_30,
    "STD60": _std_60,
}

"""
K 线形态因子（Alpha158 001-009）。
每个函数只做一件事，无副作用，无全局状态。
"""

import pandas as pd
import numpy as np


def _kmid(df):   return (df["close"] - df["open"]) / df["open"]
def _klen(df):   return (df["high"] - df["low"]) / df["open"]
def _kmid2(df):  return (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-12)
def _kup(df):    return (df["high"] - np.maximum(df["open"], df["close"])) / df["open"]
def _kup2(df):   return (df["high"] - np.maximum(df["open"], df["close"])) / (df["high"] - df["low"] + 1e-12)
def _klow(df):   return (np.minimum(df["open"], df["close"]) - df["low"]) / df["open"]
def _klow2(df):  return (np.minimum(df["open"], df["close"]) - df["low"]) / (df["high"] - df["low"] + 1e-12)
def _ksft(df):   return (2 * df["close"] - df["high"] - df["low"]) / df["open"]
def _ksft2(df):  return (2 * df["close"] - df["high"] - df["low"]) / (df["high"] - df["low"] + 1e-12)


KBAR_REGISTRY = {
    "KMID":  _kmid,
    "KLEN":  _klen,
    "KMID2": _kmid2,
    "KUP":   _kup,
    "KUP2":  _kup2,
    "KLOW":  _klow,
    "KLOW2": _klow2,
    "KSFT":  _ksft,
    "KSFT2": _ksft2,
}

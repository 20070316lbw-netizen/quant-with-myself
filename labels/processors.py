"""
Processor 基类 + 标签处理器。
设计原则完全对齐 Qlib：
  - fit() 只在训练集上调用，学统计量
  - __call__() 应用变换
  - 测试集永远不走 fit()，也可以选择不走 __call__()（标签类）

防泄漏铁律（写在这里而不是散在调用方）：
  1. 任何 groupby 必须是 groupby("date")，严禁跨日期
  2. fit() 的统计量只能来自训练集
  3. 标签类 Processor（is_label=True）在推理侧完全跳过
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class Processor(ABC):
    is_label = False  # 子类若依赖 label，设为 True；推理侧自动跳过

    def fit(self, series: pd.Series, date_col: pd.Series) -> "Processor":
        """在训练集上学参数。无参数的处理器可以不 override。"""
        return self

    @abstractmethod
    def __call__(self, series: pd.Series, date_col: pd.Series) -> pd.Series:
        """应用变换。必须实现。"""
        ...

    def sanity_check(self, series: pd.Series, date_col: pd.Series):
        """
        数据健康检查：调用前先跑，发现问题立即 raise。
        防止数据越来越少的问题在这里集中暴露。
        """
        n_before = series.notna().sum()
        if n_before == 0:
            raise ValueError(f"{self.__class__.__name__}: 输入全是 NaN，检查上游")
        # 返回给调用方，方便对比处理前后的行数
        return n_before


class CSRankNorm(Processor):
    """
    截面 rank → 标准化到零均值对称分布（对标 Qlib CSRankNorm）。

    公式：(rank(pct=True) - 0.5) * 3.46
    为什么 3.46：[0,1] 均匀分布的标准差 = 1/sqrt(12) ≈ 0.2887，
    乘以 3.46 ≈ sqrt(12) 后 std → 1。L2 回归天然适配零均值对称分布。

    防泄漏自检：
      ✓ groupby("date") 严格按单日截面
      ✓ rank 不需要 fit（纯序，无跨样本统计量）
      ✓ is_label=True，推理侧自动跳过
    """
    is_label = True

    def __call__(self, series: pd.Series, date_col: pd.Series) -> pd.Series:
        n_before = self.sanity_check(series, date_col)

        ranked = series.groupby(date_col).rank(pct=True)
        result = (ranked - 0.5) * 3.46

        # ── 行数守卫 ──────────────────────────────────────
        n_after = result.notna().sum()
        if n_after < n_before * 0.95:
            raise ValueError(
                f"CSRankNorm 后非 NaN 行数从 {n_before} 降到 {n_after}，"
                f"超过 5% 损失，检查 groupby 或输入"
            )
        return result


class CSZScoreNorm(Processor):
    """
    截面 z-score 标准化（对标 Qlib CSZScoreNorm，Qlib 默认选项）。

    公式：(x - 当日截面均值) / 当日截面标准差
    vs CSRankNorm：保留相对数值差距，同时减掉大盘 beta。
    留作 Phase 3 对照实验。

    防泄漏自检：
      ✓ fit() 只在训练集调用，stats_ 来自训练截面
      ✓ __call__() 用 fit 时学到的 stats_，不用当日自己的（防测试集泄漏）
      ✗ 注意：若在测试集上重新 groupby 算均值/std，是泄漏！
    """
    is_label = True

    def fit(self, series: pd.Series, date_col: pd.Series) -> "CSZScoreNorm":
        self.stats_ = (
            series.groupby(date_col)
            .agg(["mean", "std"])
            .rename(columns={"mean": "mu", "std": "sigma"})
        )
        return self

    def __call__(self, series: pd.Series, date_col: pd.Series) -> pd.Series:
        if not hasattr(self, "stats_"):
            raise RuntimeError("CSZScoreNorm 必须先 fit() 再 __call__()")
        n_before = self.sanity_check(series, date_col)

        mu = date_col.map(self.stats_["mu"])
        sigma = date_col.map(self.stats_["sigma"]).replace(0, np.nan)
        result = (series - mu) / sigma

        n_after = result.notna().sum()
        if n_after < n_before * 0.90:
            raise ValueError(
                f"CSZScoreNorm 后非 NaN 行数从 {n_before} 降到 {n_after}，超过 10% 损失"
            )
        return result


# ── 注册表：yaml 里的字符串 → 类 ─────────────────────────
PROCESSOR_REGISTRY = {
    "cs_rank":   CSRankNorm,
    "cs_zscore": CSZScoreNorm,
}


def build_processors(names: list[str]) -> list[Processor]:
    """
    把 yaml 里的 ['cs_rank', ...] 实例化成 Processor 列表。
    拼错 / 未注册 → 启动即报错，不等到运行时。
    """
    if names is None:
        return []
    unknown = [n for n in names if n not in PROCESSOR_REGISTRY]
    if unknown:
        raise ValueError(f"label_processors 中未知处理器: {unknown}，可用: {list(PROCESSOR_REGISTRY)}")
    return [PROCESSOR_REGISTRY[n]() for n in names]


def apply_label_processors(
    processors: list[Processor],
    df: pd.DataFrame,
    label_col: str,
    date_col: str = "date",
    fit: bool = False,
) -> pd.DataFrame:
    """
    对 label 列依次应用处理器链。

    fit=True  → 训练集，先 fit() 再 __call__()
    fit=False → 测试集，is_label=True 的处理器直接跳过（防泄漏铁律）

    行数守卫：每个 Processor 应用后检查行数，防止数据悄悄变少。
    """
    df = df.copy()
    series = df[label_col]
    dates = df[date_col]

    n_start = series.notna().sum()

    for proc in processors:
        if not fit and proc.is_label:
            # 测试集：标签处理器完全跳过，label 原样保留
            continue
        if fit:
            proc.fit(series, dates)
        series = proc(series, dates)

    n_end = series.notna().sum()
    # 全链结束后再做一次总体守卫
    if n_end < n_start * 0.90:
        raise ValueError(
            f"标签处理链执行后有效行数从 {n_start} → {n_end}，损失超 10%，检查处理器链"
        )

    df[label_col] = series
    return df

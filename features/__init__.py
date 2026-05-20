"""
全局因子注册表。这是唯一合法的因子来源。
train.py / evaluate.py / main.py 只从这里取，不准自己手抄。

防泄漏自检（启动时执行）：
  - yaml 点名的因子必须在 REGISTRY 里（拼错 → 启动即报错）
  - 因子函数计算前后行数守卫在 make_features() 里执行
"""

import pandas as pd
from .kbar   import KBAR_REGISTRY
from .roc    import ROC_REGISTRY
from .ma     import MA_REGISTRY
from .std    import STD_REGISTRY
from .volume import VOLUME_REGISTRY
from .cs_rank import CS_RANK_REGISTRY


FEATURE_REGISTRY = {
    **KBAR_REGISTRY,
    **ROC_REGISTRY,
    **MA_REGISTRY,
    **STD_REGISTRY,
    **VOLUME_REGISTRY,
    **CS_RANK_REGISTRY,
}


def build_feature_cols(cfg: dict) -> list[str]:
    """
    从 yaml features.active 取因子列表，启动时做两项校验：
    1. yaml 点名的每个因子都在 REGISTRY 里（防拼错/漏注册）
    2. 列表不能为空（防无因子训练）
    """
    active = cfg.get("features", {}).get("active", [])
    if not active:
        raise ValueError("config.yaml features.active 为空，至少选一个因子")
    unknown = [f for f in active if f not in FEATURE_REGISTRY]
    if unknown:
        raise ValueError(f"yaml 中的因子未注册: {unknown}，全集: {list(FEATURE_REGISTRY)}")
    return active


def make_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    按 feature_cols 顺序计算因子，挂回 df。
    顺序有意义：有依赖关系的因子（如 std_5d 依赖 mom_1d）必须靠注册顺序保证。

    行数守卫：计算前后总行数不能变，因子计算不得 dropna / filter 行。
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    n_before = len(df)

    for name in feature_cols:
        fn = FEATURE_REGISTRY[name]
        df[name] = fn(df)

    n_after = len(df)
    if n_after != n_before:
        raise RuntimeError(
            f"make_features 前后行数不一致: {n_before} → {n_after}，"
            "因子函数不允许 dropna/filter 行"
        )
    return df

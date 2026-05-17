# model/evaluate.py
import pandas as pd

FEATURE_COLS = ["mom_1d", "mom_5d", "std_5d", "KMID", "KLEN"]
LABEL_COL = "label_5d"

def evaluate_rank_ic(model, test_df: pd.DataFrame) -> dict:
    """
    横截面 Rank IC 评估。
    返回逐日 IC 序列的统计:均值、标准差、ICIR、>0 的占比
    """
    df = test_df.dropna(subset=FEATURE_COLS + [LABEL_COL]).copy()
    df["pred"] = model.predict(df[FEATURE_COLS])

    # 核心:按【每个交易日】分组,各算当天的 Rank IC
    def _daily_ic(group):
        # 当天股票太少,排序无意义,跳过
        if len(group) < 20:
            return None
        return group["pred"].corr(group[LABEL_COL], method="spearman")

    daily_ic = (
        df.groupby("date")
        .apply(_daily_ic)
        .dropna()
    )

    mean_ic = daily_ic.mean()
    std_ic = daily_ic.std()
    return {
        "mean_rank_ic": mean_ic,
        "std_rank_ic": std_ic,
        "icir": mean_ic / std_ic if std_ic > 0 else float("nan"),
        "ic_positive_ratio": (daily_ic > 0).mean(),  # IC 为正的天数占比
        "n_days": len(daily_ic),
    }

# model/evaluate.py 里,加在 evaluate_rank_ic 旁边

def evaluate_quantile_returns(model, test_df: pd.DataFrame, n_groups: int = 5) -> dict:
    """
    分组收益评估。
    每个交易日按预测值把股票分 n_groups 组,
    看每组的平均真实收益,以及"最高组 - 最低组"的多空收益。
    """
    df = test_df.dropna(subset=FEATURE_COLS + [LABEL_COL]).copy()
    df["pred"] = model.predict(df[FEATURE_COLS])

    def _assign_group(group):
        # 当天股票太少,分组无意义
        if len(group) < n_groups * 4:
            return None
        # qcut: 按预测值分位数切成 n_groups 组,标 0..n_groups-1
        # labels=False 直接给组号; duplicates='drop' 防止边界重复值报错
        group = group.copy()
        group["grp"] = pd.qcut(
            group["pred"], n_groups, labels=False, duplicates="drop"
        )
        return group

    df = (
        df.groupby("date", group_keys=False)
        .apply(_assign_group)
        .dropna(subset=["grp"])
    )

    # 每组的平均真实收益(跨所有日期、所有股票)
    group_ret = df.groupby("grp")[LABEL_COL].mean()

    top = group_ret.iloc[-1]    # 预测最高组的真实收益
    bottom = group_ret.iloc[0]  # 预测最低组的真实收益

    # 单调性检查:理想情况组号越大收益越高
    is_monotonic = group_ret.is_monotonic_increasing

    return {
        "group_returns": group_ret.to_dict(),  # 每组平均收益
        "long_short": top - bottom,            # 多空收益:最关键指标
        "monotonic": is_monotonic,             # 阶梯是否单调(信号质量)
    }
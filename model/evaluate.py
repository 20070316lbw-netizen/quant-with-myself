import pandas as pd

from features.make_features import FEATURE_COLS   # single source of truth,不再手抄
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
        df.groupby("date")          # df.groupby("date"), IC 是逐日算的,分组收益却是 pooled,两套口径不一致
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

def evaluate_quantile_returns(model, test_df: pd.DataFrame, n_groups: int = 5) -> dict:
    """
    分组收益评估。
    每个交易日按预测值把股票分 n_groups 组,也就是5个桶子,
    看每组的平均真实收益,以及"最高组 - 最低组"的多空收益。
    """
    df = test_df.dropna(subset=FEATURE_COLS + [LABEL_COL]).copy()
    df["pred"] = model.predict(df[FEATURE_COLS])

    # 逐日打组号:用 groupby(...).transform,逐行返回、不改 df 结构,
    # date 始终是普通列——避开 pandas 3.0 下 groupby-apply 把分组键
    # 吸进 index 导致下一次 groupby("date") 报 KeyError 的问题。
    def _grp_of_day(s: pd.Series) -> pd.Series:
        # s = 某一天所有股票的 pred。当天股票太少则不分组(返回 NaN)
        if len(s) < n_groups * 4:
            return pd.Series(float("nan"), index=s.index)
        # 把当天股票按 pred 从低到高切成 n_groups 等份,组号 0..n_groups-1
        return pd.qcut(s, n_groups, labels=False, duplicates="drop")

    df = df.copy()
    df["grp"] = df.groupby("date")["pred"].transform(_grp_of_day)
    df = df.dropna(subset=["grp"])

    # ===== per-day:每天先算「当天的多空收益」,得到一条逐日序列 =====
    # 为什么必须逐日算:胜率 = 赚钱天数/总天数,波动 = 标准差,
    # 这些都是定义在「一条序列」上的统计量。一旦直接对全样本
    # .mean(),逐日序列在那一瞬间被销毁,波动/胜率/夏普将永远
    # 无法还原——这就是 pooled 的致命缺陷。
    # 另注:这里的「逐日」与 evaluate_rank_ic 口径完全对称,两套评估
    # 口径从此统一(之前 IC 逐日、分组收益 pooled 是不一致的)。
    #
    # 实现上不用任何 groupby-apply:先按 (date, grp) 聚合出“每天每组
    # 均收益”,pivot 成「行=date、列=grp」的表,最高组减最低组
    # 就是每天的多空收益。纯聚合+向量运算,不踩 apply 的任何坑。
    daily_grp = (
        df.groupby(["date", "grp"])[LABEL_COL]
        .mean()
        .unstack("grp")          # 行=date, 列=grp(0..n_groups-1)
        .sort_index(axis=1)      # 确保列按组号从小到大
    )
    # 每天: 最高组(最后一列) - 最低组(第一列)
    daily_ls = (daily_grp.iloc[:, -1] - daily_grp.iloc[:, 0]).dropna()

    # 在「序列」上做统计 —— 这几个 pooled 永远算不出来
    mean_ls = daily_ls.mean()
    std_ls = daily_ls.std()
    win_rate = (daily_ls > 0).mean()              # 胜率:多空>0 的天数占比
    sharpe = (mean_ls / std_ls) if std_ls > 0 else float("nan")

    # 各组平均收益:仅用于看分组形状(pooled 视角),不用于多空
    group_ret = df.groupby("grp")[LABEL_COL].mean()

    # 单调性:用「组号 vs 收益的秩相关」替代脆弱的布尔。
    # 取值 -1~+1:+1=完全单调递增,0=无序,-1=完全递减。
    # 比 is_monotonic_increasing 稳健:某一组 0.001% 的噪声不会把它
    # 从 True 翻成 False,且无需任何魔法阈值。
    grp_idx = pd.Series(group_ret.index, index=group_ret.index)
    monotonic_corr = grp_idx.corr(group_ret, method="spearman")

    return {
        "group_returns": group_ret.to_dict(),
        "long_short": mean_ls,              # 逐日均值,不再是 pooled
        "long_short_std": std_ls,          # 日波动(pooled 看不到)
        "win_rate": win_rate,              # 胜率(pooled 看不到)
        "sharpe": sharpe,                  # 夏普(最关键,pooled 看不到)
        "n_days": len(daily_ls),           # 有效天数
        "monotonic_corr": monotonic_corr,  # 秩相关,替代脆布尔 monotonic
    }

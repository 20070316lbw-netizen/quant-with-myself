from utils.config import get_db, load_config
from features import build_feature_cols, make_features
from labels.make_labels import make_label
from model.train import build_dataset, train_lgb


def main():
    cfg = load_config()

    # 启动即校验：yaml 因子名都在注册表里，否则直接 raise
    feature_cols = build_feature_cols(cfg)

    with get_db() as con:
        df = con.execute("SELECT * FROM prices ORDER BY ticker, date").fetchdf()
    print(f"[1] 原始数据: {len(df)} 行, {df['ticker'].nunique()} 只股票, "
          f"日期 {df['date'].min()} ~ {df['date'].max()}")

    # make_features 现在接受 feature_cols 参数（不再全量算）
    df = make_features(df, feature_cols)
    df["label_5d"] = make_label(df, n_periods=5, gap=1)
    print(f"[2] 特征+标签后: {len(df)} 行")

    # ── 数据量快照（在切分前）──────────────────────────
    _print_data_health(df, feature_cols)

    train_df, test_df = build_dataset(df, cfg)
    print(f"[3] 切分后: train={len(train_df)} 行, test={len(test_df)} 行")
    
    # 确认切出来的不是空的
    if len(train_df) == 0:
        print("    ❌ 训练集是空的!日期边界和数据范围对不上")
        return

    model, history = train_lgb(train_df, cfg, feature_cols=feature_cols, return_history=True)
    print(f"[4] 训练完成, 早停后实际树数 = {model.num_trees()} 棵(上限是配置的 num_boost_round)")
    
    # 把每轮 train/valid 误差存下来,供画过拟合曲线用
    import json
    with open("train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"    误差曲线已存到 train_history.json(train/valid 逐轮 RMSE)")

    from model.evaluate import evaluate_rank_ic
    metrics = evaluate_rank_ic(model, test_df)
    print(f"[5] Rank IC 评估:")
    print(f"    平均 Rank IC : {metrics['mean_rank_ic']:.4f}")
    print(f"    ICIR         : {metrics['icir']:.4f}")
    print(f"    IC>0 占比    : {metrics['ic_positive_ratio']:.2%}")
    print(f"    评估天数     : {metrics['n_days']}")

    from model.evaluate import evaluate_quantile_returns
    q = evaluate_quantile_returns(model, test_df, n_groups=5)
    print(f"[6] 分组收益(5组,按预测值从低到高):")
    for g, r in sorted(q["group_returns"].items()):
        print(f"    第{int(g)}组: 平均5日真实收益 = {r:.4%}")
    print(f"[7] 多空组合(per-day 口径,逐日算再统计):")
    print(f"    多空收益(逐日均值): {q['long_short']:.4%}")
    print(f"    日波动(标准差)   : {q['long_short_std']:.4%}")
    print(f"    胜率(>0 天占比)  : {q['win_rate']:.2%}")
    print(f"    夏普(mean/std)   : {q['sharpe']:.4f}")
    print(f"    有效天数         : {q['n_days']}")
    print(f"    分组秩相关(-1~+1): {q['monotonic_corr']:.4f}")


def _print_data_health(df, feature_cols):
    """
    每次运行都打印数据健康摘要，方便发现'数据越来越少'的问题。
    不 raise，只是可见性工具。
    """
    total = len(df)
    label_nan = df["label_5d"].isna().sum()
    print(f"\n── 数据健康检查 ──────────────────────────")
    print(f"   总行数       : {total}")
    print(f"   label NaN   : {label_nan} ({label_nan/total:.1%})")
    for col in feature_cols:
        n = df[col].isna().sum()
        if n / total > 0.10:   # 超过 10% NaN 才报警，否则只是 rolling 头部正常缺失
            print(f"   ⚠ {col}: {n} NaN ({n/total:.1%})，超过 10% 请检查")
    print(f"──────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
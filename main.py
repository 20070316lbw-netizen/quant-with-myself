from utils.config import get_db, load_config
from features.make_features import make_features
from labels.make_labels import make_label
from model.train import build_dataset, train_lgb

def main():
    cfg = load_config()

    with get_db() as con:
        df = con.execute("SELECT * FROM prices ORDER BY ticker, date").fetchdf()
    print(f"[1] 原始数据: {len(df)} 行, {df['ticker'].nunique()} 只股票, "
          f"日期 {df['date'].min()} ~ {df['date'].max()}")

    df = make_features(df)
    df["label_5d"] = make_label(df, n_periods=5, gap=1)
    from model.train import FEATURE_COLS
    print(f"[2] 加特征+标签后: {len(df)} 行")
    nan_info = ", ".join([f"{col}={df[col].isna().sum()}" for col in FEATURE_COLS])
    print(f"    特征NaN: {nan_info}, label={df['label_5d'].isna().sum()}")

    train_df, test_df = build_dataset(df, cfg)
    print(f"[3] 切分后: train={len(train_df)} 行, test={len(test_df)} 行")
    # 这一行最关键 —— 确认切出来的不是空的
    if len(train_df) == 0:
        print("    ❌ 训练集是空的!日期边界和数据范围对不上")
        return

    model = train_lgb(train_df, cfg)
    print(f"[4] 训练完成, 模型有 {model.num_trees()} 棵树")

    # 临时:在测试集上预测,看一眼数值合理不(还不是正式 evaluate)
    from model.train import FEATURE_COLS
    valid = test_df.dropna(subset=FEATURE_COLS + ["label_5d"])
    pred = model.predict(valid[FEATURE_COLS])
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
    print(f"    多空收益(最高组 - 最低组): {q['long_short']:.4%}")
    print(f"    收益随分组单调递增: {q['monotonic']}")

if __name__ == "__main__":
    main()
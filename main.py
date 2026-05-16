from utils.config import get_db, load_config
from features.make_features import make_features
from labels.make_labels import make_label
from model.train import build_dataset, train_lgb

def main():
    cfg = load_config()                    # 配置只在这里读一次

    with get_db() as con:
        df = con.execute("SELECT * FROM prices ORDER BY ticker, date").fetchdf()

    df = make_features(df)
    df["label_5d"] = make_label(df, n_periods=5, gap=1).values

    train_df, test_df = build_dataset(df, cfg)   # 配置传进去
    model = train_lgb(train_df, cfg)
    # evaluate(model, test_df, cfg) ← 下一步

if __name__ == "__main__":
    main()
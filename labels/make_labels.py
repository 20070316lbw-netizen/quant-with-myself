import pandas as pd
import numpy as np

# 注：get_db 的 import 放在 __main__ 里，不在这里。
# 这样 make_label() 本身不依赖 utils，被别的模块 import 时不会报错。


# =====================================================================
# 1) 打标签:未来 N 日收益率
# =====================================================================

def make_label(df: pd.DataFrame,
               n_periods: int = 5,
               price_col: str = "close",
               gap: int = 1,
               ) -> pd.Series:
    """
    Args:
        df: 含有 date, ticker, price_col 的普通 DataFrame
            (date、ticker 是普通列,不是 index)
        n_periods: 持仓天数 (预测未来多少天的累计收益)
        price_col: 用哪列价 (默认 close)
        gap: 当前 t 和进场之间隔几期 (默认 1, 即 t+1 开仓, t+1+N 平仓)
    Returns:
        Series, 与输入 df 行对齐, 值是未来 N 日收益率
        (NaN 会出现在每只股票末尾的 gap + n_periods 行,因为未来数据不存在)
    """
    # 先按 ticker 和 date 排序,保证 shift 的方向正确
    # 不排序的话,如果 df 是乱序的,shift 出来的"未来"可能是别的日期
    df = df.sort_values(['ticker', 'date'])

    # 按 ticker 分组对 price_col 做 shift
    # 这里用列名 'ticker' 而不是 level=,跟 make_features.py 保持接口一致
    by_inst = df.groupby('ticker')[price_col]

    # shift(-k) 表示把 k 期之后的值挪到当前行
    #   enter:  t + gap 时刻的价格(进场价)
    #   exit_:  t + gap + n_periods 时刻的价格(平仓价)
    enter = by_inst.shift(-gap)
    exit_ = by_inst.shift(-(gap + n_periods))

    return (exit_ / enter - 1).rename(f"label_{n_periods}d")


if __name__ == "__main__":
    # 直接运行本文件时，先把项目根加到 sys.path，才能 import utils
    # 被别的模块 import 时不走这里，不会污染 sys.path
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from utils.config import get_db

    # 简单自测:从数据库读价格,打标签,看结果
    with get_db() as con:
        df = con.execute("SELECT * FROM prices ORDER BY ticker, date").fetchdf()

    label = make_label(df, n_periods=5, gap=1)
    df_with_label = df.assign(label_5d=label.values)

    print(df_with_label[['date', 'ticker', 'close', 'label_5d']].head(10))
    print(df_with_label[['date', 'ticker', 'close', 'label_5d']].tail(10))
    print(f"\n共 {len(df_with_label)} 行")
    print(f"label NaN 数量: {df_with_label['label_5d'].isna().sum()}")
    print(f"(每只股票末尾会有 gap + n_periods = {1 + 5} 行 NaN,因为未来还没发生)")

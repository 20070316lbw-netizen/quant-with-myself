"""
目前这里只做测试,完整的特征采用以下特征:
FEATURE_COLS = [
    "KMID", "KLEN", "KMID2", "KUP", "KLOW", "KSFT",
    "ROC5", "ROC20", "ROC60",
    "MA5", "MA20", "MA60",
    "STD5", "STD20", "STD60",
]

解释:
---
kbar 类(6 个）—— 描述当日多空力量博弈

KMID (实体涨跌幅）：今天总体是涨是跌、力度多大。最直接的当日强弱信号。
KLEN (振幅）：今天的波动有多剧烈。高振幅可能意味着不确定性增加或情绪激烈。
KMID(实体占比）：如果实体大、影线小，说明多空一方完胜；如果实体小、影线长，说明拉锯激烈、方向未定。
KUP (上影线）：上方有抛压，冲高回落，看跌信号。
KLOW (下影线）：下方有支撑，探底回升，看涨信号。
KSFT (收盘在区间的位置）：收盘越靠近最高，说明多头掌控；越靠近最低，空头掌控。

KMID  = (close - open) / open
KLEN  = (high - low) / open
KMID2 = (close - open) / (high - low + 1e-12)
KUP   = (high - max(open, close)) / open          # 上影线
KLOW  = (min(open, close) - low) / open           # 下影线
KSFT  = (2*close - high - low) / open             # 收盘在当日区间的相对位置

这些都是技术分析里 K 线形态学的数学化，几十年来交易员就在看这些东西。

---
ROC 类 (3 个）—— 动量因子

ROC5 / ROC20 / ROC60: 过去 5/20/60 天的累计涨跌。

背后逻辑是动量效应 (momentum) 学术界公认的异象之一:过去涨得好的股票,未来一段时间倾向于继续涨;过去跌得惨的, 倾向于继续跌。三个窗口分别对应短期、中期、长期动量。Jegadeesh & Titman 1993 那篇经典论文专门研究的就是这个。
ROC5  = close.shift(5)  / close
ROC20 = close.shift(20) / close
ROC60 = close.shift(60) / close

---
MA 类 (3 个）—— 趋势位置

MA5 / MA20 / MA60 (均线 / 当前价）：当前价相对均线的位置。

如果 MA20/close > 1,说明当前价低于 20 日均线，可能是回调；< 1 则在均线之上，趋势向上。背后是均值回归 vs 趋势跟随的博弈——这个特征本身不预设方向，让模型自己学在什么情况下倾向哪个。
MA5  = close.rolling(5).mean()  / close
MA20 = close.rolling(20).mean() / close
MA60 = close.rolling(60).mean() / close

---
STD 类 (3 个）—— 波动率

STD5 / STD20 / STD60 (标准差 / 当前价）：归一化的波动率。

背后逻辑是低波动异象 (low volatility anomaly): 长期看，低波动股票的风险调整后收益反而高于高波动股票，违反了 CAPM 的预期。这是另一个被广泛研究的异象。
STD5  = close.rolling(5).std()  / close
STD20 = close.rolling(20).std() / close
STD60 = close.rolling(60).std() / close
"""

import pandas as pd
import numpy as np

# 注：get_db 的 import 放在 __main__ 里，不在这里。
# 这样 make_features() 本身不依赖 utils，被别的模块 import 时不会报错。


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入: 含有 date, ticker, open, high, low, close, volume 的 DataFrame
    输出: 在原 DataFrame 基础上, 新增 mom_1d, mom_5d, std_5d 三列

    命名说明:
      mom_1d / mom_5d  — 过去 1 天 / 5 天的 close 变化率 (动量因子, 是特征)
      std_5d           — 过去 5 天 mom_1d 的标准差 (波动率, 是特征)
      不用 return_5d 这种名字, 是为了跟 labels/ 里的"未来 N 日收益 (label_5d)"划清界限。
      二者都是 close 的 pct_change 计算, 但用途相反:
        mom_5d   是当前已知的 (t-5 → t)        → 用作特征输入模型
        label_5d 是未来要预测的 (t+1 → t+6) → 用作训练目标

    ⚠️ 关键点: 特征要按 ticker 分组计算!
    不能把 NVDA 的昨天和 INTC 的今天混在一起算。
    """
    # 先按 ticker 和 date 排序, 确保 shift/rolling 的顺序正确
    df = df.sort_values(['ticker', 'date']).copy()

    # 动量: 过去 N 天的 close 变化率
    # pct_change(N) = (close_t / close_{t-N}) - 1
    df['mom_1d'] = df.groupby('ticker')['close'].pct_change(1)
    df['mom_5d'] = df.groupby('ticker')['close'].pct_change(5)

    # 波动率: 过去 5 天 mom_1d 的标准差
    # rolling 后会多出一层 group key 索引, reset_index 用来拉平, 才能赋值回 df
    df['std_5d'] = (
        df.groupby('ticker')['mom_1d']
        .rolling(5)
        .std()
        .reset_index(level=0, drop=True)
    )

    return df


if __name__ == "__main__":
    # 直接运行本文件时，先把项目根加到 sys.path，才能 import utils
    # 被别的模块 import 时不走这里，不会污染 sys.path
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from utils.config import get_db

    with get_db() as con:
        df = con.execute("SELECT * FROM prices ORDER BY ticker, date").fetchdf()

    df_with_features = make_features(df)

    print(df_with_features.head(10))
    print(df_with_features.tail(10))
    print(f"\n共 {len(df_with_features)} 行")
    print(f"NaN 数量:\n{df_with_features.isnull().sum()}")
    print("(每只股票前几行会有 NaN, 是因为 window 还没填满, 这是正常的)")
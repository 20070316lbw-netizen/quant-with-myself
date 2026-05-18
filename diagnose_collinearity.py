"""
共线性诊断脚本 —— 只读,不改任何现有代码。

目的:用数据验证"加了19个特征后指标砍半,是不是共线性放大了噪声"。
做三件事:
  1. 复用项目自己的 make_features 算出全部特征(口径与模型完全一致)
  2. 相关性矩阵 + 揪出 |corr| > 0.8 的高共线特征对
  3. 每个特征的 VIF(方差膨胀因子),量化"这个特征能被其他特征线性表示的程度"
     —— VIF 用 numpy 手算(= 用其它特征回归它的 R²),不依赖 statsmodels

跑法(在项目根目录):
    uv run python diagnose_collinearity.py

注意:本脚本只 print,不写数据库、不改 config、不碰 model/ 下任何文件。
"""

import numpy as np
import pandas as pd

from utils.config import get_db
from features.make_features import make_features, FEATURE_COLS


def main():
    # ---- 1. 读原始价格,用项目自己的特征函数算特征(口径完全一致)----
    with get_db() as con:
        df = con.execute("SELECT * FROM prices ORDER BY ticker, date").fetchdf()
    print(f"原始数据: {len(df)} 行, {df['ticker'].nunique()} 只股票")

    df = make_features(df)
    print(f"特征数: {len(FEATURE_COLS)} 个")
    print(f"特征列表: {FEATURE_COLS}\n")

    # 只取特征列,丢掉任何含 NaN 的行(算相关性/VIF 需要干净矩阵)
    X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"用于诊断的干净样本: {len(X)} 行(丢弃含 NaN/inf 后)\n")

    # ---- 2. 相关性矩阵 + 高共线对 ----
    corr = X.corr()

    print("=" * 70)
    print("【高共线特征对】 |相关系数| > 0.8 (这些是冗余嫌疑最大的)")
    print("=" * 70)
    pairs = []
    cols = FEATURE_COLS
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr.iloc[i, j]
            if abs(c) > 0.8:
                pairs.append((abs(c), cols[i], cols[j], c))
    pairs.sort(reverse=True)
    if not pairs:
        print("  (没有 |corr|>0.8 的特征对)")
    else:
        for _, a, b, c in pairs:
            print(f"  {a:<14} <-> {b:<14}  corr = {c:+.3f}")

    print("\n" + "=" * 70)
    print("【中度相关】 0.6 < |相关系数| <= 0.8 (次级嫌疑)")
    print("=" * 70)
    mid = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr.iloc[i, j]
            if 0.6 < abs(c) <= 0.8:
                mid.append((abs(c), cols[i], cols[j], c))
    mid.sort(reverse=True)
    if not mid:
        print("  (无)")
    else:
        for _, a, b, c in mid:
            print(f"  {a:<14} <-> {b:<14}  corr = {c:+.3f}")

    # ---- 3. VIF:用其余特征线性回归每个特征,VIF = 1/(1-R^2) ----
    # R^2 用最小二乘闭式解算,纯 numpy,不需要 statsmodels。
    # VIF 解读:  1=完全独立; >5 值得警惕; >10 严重共线,基本是冗余特征。
    print("\n" + "=" * 70)
    print("【VIF 方差膨胀因子】 越高 = 越能被其他特征替代 = 越冗余")
    print("  经验阈值: VIF>10 严重冗余, 5~10 偏高, <5 健康")
    print("=" * 70)

    # 标准化(VIF 对量纲敏感,先 z-score)
    Xz = (X - X.mean()) / X.std(ddof=0)
    Xz = Xz.values
    n, p = Xz.shape

    vifs = []
    for k in range(p):
        y = Xz[:, k]
        # 设计矩阵 = 其余所有特征 + 截距
        idx = [t for t in range(p) if t != k]
        A = np.column_stack([np.ones(n), Xz[:, idx]])
        # 最小二乘: beta = (A'A)^-1 A'y ,用 lstsq 更稳
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        yhat = A @ beta
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = 1.0 / (1.0 - r2) if r2 < 1 else float("inf")
        vifs.append((vif, FEATURE_COLS[k], r2))

    vifs.sort(reverse=True)
    for vif, name, r2 in vifs:
        flag = "  <-- 严重冗余" if vif > 10 else ("  <- 偏高" if vif > 5 else "")
        vif_str = "inf" if vif == float("inf") else f"{vif:8.2f}"
        print(f"  {name:<14} VIF = {vif_str}   (被其他特征解释 R^2={r2:.3f}){flag}")

    print("\n" + "=" * 70)
    print("怎么读这份报告:")
    print("  - 高共线对里,每一对通常只需保留一个(留更基础/更稳的那个)")
    print("  - VIF>10 的特征是删除的首要候选")
    print("  - 删之前先看它和谁共线:成对出现才说明是'重复',单独高 VIF 要谨慎")
    print("=" * 70)


if __name__ == "__main__":
    main()
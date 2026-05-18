from utils.config import get_db
import numpy as np

with get_db() as con:
    df = con.execute("SELECT * FROM prices ORDER BY ticker, date").fetchdf()

print("open 有多少个 0 或负数:", ((df['open'] <= 0).sum()))
print("open 有多少 NaN:", df['open'].isna().sum())

kmid = (df['close'] - df['open']) / df['open']
print("KMID 里 inf 的数量:", np.isinf(kmid).sum())
print("KMID 里 nan 的数量:", kmid.isna().sum())
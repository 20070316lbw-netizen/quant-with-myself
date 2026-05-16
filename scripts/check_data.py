"""
一次性数据体检脚本：确认 fetch_data 有没有静默丢票。
跑法： uv run python -m scripts.check_data
确认完问题后这个文件可以删掉。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import get_db

# 这些是日志里报过 "possibly delisted" 的票，全是没退市的活跃大盘股
SUSPECT = ["SBUX", "TXN", "PANW", "PRU", "ROK", "ROL",
           "PPG", "PFG", "TTWO", "TT", "TPL"]

with get_db() as con:
    # 1) 库里一共多少只股票、多少行、日期范围
    total_tickers = con.execute(
        "SELECT COUNT(DISTINCT ticker) FROM prices"
    ).fetchone()[0]
    total_rows = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    date_min, date_max = con.execute(
        "SELECT MIN(date), MAX(date) FROM prices"
    ).fetchone()

    print("=" * 55)
    print(f"库内股票总数 : {total_tickers}  (S&P500 应在 ~500 左右)")
    print(f"库内总行数   : {total_rows}")
    print(f"日期范围     : {date_min}  ->  {date_max}")
    print("=" * 55)

    # 2) AAPL 作为正常对照，看一只健康股票应该有多少行
    aapl_rows = con.execute(
        "SELECT COUNT(*) FROM prices WHERE ticker = 'AAPL'"
    ).fetchone()[0]
    print(f"对照组 AAPL 行数: {aapl_rows}  (其它票应该和这个接近)")
    print("-" * 55)

    # 3) 逐个检查被报失败的票，到底在不在库里
    print("被报 'possibly delisted' 的票，实际在库情况:")
    rows = con.execute(
        """
        SELECT ticker, COUNT(*) AS n
        FROM prices
        WHERE ticker IN (
            'SBUX','TXN','PANW','PRU','ROK','ROL',
            'PPG','PFG','TTWO','TT','TPL'
        )
        GROUP BY ticker
        ORDER BY ticker
        """
    ).fetchall()

    found = {t: n for t, n in rows}
    for t in sorted(SUSPECT):
        if t in found:
            print(f"  {t:6s}: {found[t]:>6d} 行   ✅ 在库")
        else:
            print(f"  {t:6s}:      0 行   ❌ 完全缺失（被静默丢弃）")

    print("=" * 55)
    missing = [t for t in SUSPECT if t not in found]
    if missing:
        print(f"结论：{len(missing)} 只票完全缺失 -> {missing}")
        print("fetch 逻辑确实在静默吞失败，需要修。")
    else:
        print("结论：这些票最终都进库了。")
        print("说明某次重跑时它们成功了，但日志不可靠这点仍需修。")
    print("=" * 55)
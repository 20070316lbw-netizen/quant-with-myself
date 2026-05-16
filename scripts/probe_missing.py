"""
隔离测试：单独把那 11 只"缺失"的票小批量拉一遍。
目的——区分两种可能：
  (A) 单独拉能成功  -> 是批量请求触发 Yahoo 限流，需改 fetch 策略
  (B) 单独拉还是失败 -> 这些 ticker 本身有问题（格式/数据源），另想办法

跑法： uv run python -m scripts.probe_missing
确认完可以删掉。
"""
import time
import yfinance as yf

MISSING = ["SBUX", "TXN", "PANW", "PRU", "ROK", "ROL",
           "PPG", "PFG", "TTWO", "TT", "TPL"]

START = "2016-05-18"
END = "2026-05-16"


def probe_one(ticker: str) -> int:
    """单独拉一只，返回拿到的行数（0 表示失败）"""
    try:
        df = yf.download(
            ticker,
            start=START,
            end=END,
            auto_adjust=True,
            progress=False,
        )
        return 0 if df is None or df.empty else len(df)
    except Exception as e:
        print(f"    异常: {e}")
        return 0


print("=" * 55)
print("逐只单独拉取（每只之间 sleep 1.5 秒，避免连续请求被限）")
print("=" * 55)

ok, bad = [], []
for t in MISSING:
    n = probe_one(t)
    if n > 0:
        print(f"  {t:6s}: {n:>6d} 行   ✅ 单独拉成功")
        ok.append(t)
    else:
        print(f"  {t:6s}:      0 行   ❌ 单独拉也失败")
        bad.append(t)
    time.sleep(1.5)

print("=" * 55)
if ok and not bad:
    print(f"全部 {len(ok)} 只单独拉都成功。")
    print("结论(A)：问题是批量请求触发限流，不是股票本身。")
    print("修复方向：缩小 batch_size + 加重试 + 批间 sleep。")
elif bad and not ok:
    print(f"全部 {len(bad)} 只单独拉也失败。")
    print("结论(B)：可能是 ticker 格式或数据源问题，需要逐个深查。")
else:
    print(f"成功 {len(ok)} 只: {ok}")
    print(f"失败 {len(bad)} 只: {bad}")
    print("结论：混合情况。成功的属限流问题，失败的需单独查。")
print("=" * 55)
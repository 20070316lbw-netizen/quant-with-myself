import pandas as pd
import numpy as np
from utils.config import get_db


class DataChecker:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def check_ohlcv(self) -> dict:
        """检查是否包含必要的 OHLCV 列"""
        required = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in self.df.columns]
        ok = len(missing) == 0
        return {"ok": ok, "message": "OHLCV 列完整" if ok else f"缺失列: {missing}"}

    def check_missing_data(self) -> dict:
        """检查是否有空值"""
        total_missing = self.df.isnull().sum().sum()
        ok = total_missing == 0
        return {"ok": ok, "message": "无缺失值" if ok else f"共缺失 {total_missing} 处"}

    def check_date_column(self) -> dict:
        """检查 date 列是否存在且格式正确"""
        if "date" not in self.df.columns:
            return {"ok": False, "message": "缺少 date 列"}
        try:
            pd.to_datetime(self.df["date"])
            return {"ok": True, "message": "date 列格式正常"}
        except Exception as e:
            return {"ok": False, "message": f"date 列格式异常: {e}"}

    def check_duplicates(self) -> dict:
        """检查 (date, ticker) 是否有重复"""
        if not {"date", "ticker"}.issubset(self.df.columns):
            return {"ok": False, "message": "缺少 date 或 ticker 列，无法检查重复"}
        dupes = self.df.duplicated(subset=["date", "ticker"]).sum()
        ok = dupes == 0
        return {"ok": ok, "message": "无重复记录" if ok else f"发现 {dupes} 条重复记录"}

    def check_date_continuity(self) -> dict:
        """检查每只股票的日期是否有断层（缺失交易日）"""
        if not {"date", "ticker"}.issubset(self.df.columns):
            return {"ok": False, "message": "缺少 date 或 ticker 列，无法检查连续性"}

        df = self.df.copy()
        df["date"] = pd.to_datetime(df["date"])
        issues = []

        for ticker, group in df.groupby("ticker"):
            dates = group["date"].sort_values()
            # 用交易日历粗估：超过 5 天的间隔认为有断层
            gaps = dates.diff().dropna()
            large_gaps = gaps[gaps > pd.Timedelta(days=5)]
            if not large_gaps.empty:
                issues.append(f"{ticker}: {len(large_gaps)} 处断层")

        ok = len(issues) == 0
        return {"ok": ok, "message": "日期连续" if ok else f"发现断层 -> {'; '.join(issues)}"}
    
    def count_trading_days(self) -> int:
        """返回数据库中共有多少个交易日"""
        with get_db() as con:
            result = con.execute("SELECT COUNT(DISTINCT date) FROM prices").fetchone()
        return result[0]

    def get_trading_days(self) -> list:
        """返回数据库中所有交易日的列表，按日期排序"""
        with get_db() as con:
            result = con.execute("""
                SELECT date
                FROM prices
                GROUP BY date
                ORDER BY date
            """).df()
        return result["date"].tolist()
    
    def check_table_name(self) -> list:
        with get_db() as con:
            result = con.execute("""
                DESCRIBE prices
            """).df()
            # DESCRIBE 返回的 DataFrame 有个 column_name 列，.tolist() 转成列表就行了
        return result["column_name"].tolist()

    def run_all(self) -> None:
        """运行全部检查，打印结果"""
        checks = [
            self.check_ohlcv(),
            self.check_missing_data(),
            self.check_date_column(),
            self.check_duplicates(),
            self.check_date_continuity(),
        ]
        for result in checks:
            status = "✅" if result["ok"] else "❌"
            print(f"{status} {result['message']}")
        print(f"📅 共 {self.count_trading_days()} 个交易日")
        print(f"📋 列名字有{self.check_table_name()}")
        # 📋 🗂️ 📊 🏷️  候选emoji...莫名其妙

if __name__ == "__main__":
    with get_db() as con:
        df = con.execute("SELECT * FROM prices").df()

    checker = DataChecker(df)
    checker.run_all()

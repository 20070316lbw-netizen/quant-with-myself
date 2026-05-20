import duckdb
import yaml
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import get_db, load_config

# 将数据按照先名字再日期的顺序排列
with get_db() as con:
    df = con.execute("""
                     SELECT *
                     FROM prices
                     ORDER BY ticker, date
                     """).df()
    



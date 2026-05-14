使用config.py导入配置
---
```python
from utils.config import get_db, load_config

cfg = load_config()
model_params = cfg["model"]
train_start = cfg["data"]["splits"]["train"]["start"]
```

## 数据与字段命名总结

这个项目约定所有数据都是**长格式**（date、ticker 是普通列，不是 MultiIndex），按 `['ticker', 'date']` 排序。

### 1) 原始价格数据 — `prices` 表（DuckDB）

来源：yfinance 拉取的标普 500 日线（前复权），写入 `database/sp500.duckdb`。

| 列名 | 类型 | 含义 |
|---|---|---|
| `date` | DATE | 交易日 |
| `ticker` | VARCHAR | 股票代码（如 AAPL，BRK.B 已转为 BRK-B） |
| `open` | DOUBLE | 开盘价 |
| `high` | DOUBLE | 最高价 |
| `low` | DOUBLE | 最低价 |
| `close` | DOUBLE | 收盘价（auto_adjust=True，已复权） |
| `volume` | BIGINT | 成交量 |

主键：`(date, ticker)`。

### 2) 特征 — `make_features()` 输出

在 `prices` 表的基础上添加以下列（都是按 `ticker` 分组计算，不会跨股票泄露）：

| 列名 | 含义 | 公式 |
|---|---|---|
| `mom_1d` | 过去 1 天动量 | `close.pct_change(1)` |
| `mom_5d` | 过去 5 天动量 | `close.pct_change(5)` |
| `std_5d` | 过去 5 天 `mom_1d` 的标准差 | `mom_1d.rolling(5).std()` |

> 待扩展：KMID / KLEN / KMID2 / KUP / KLOW / KSFT（K 线形态类）、ROC5/20/60、MA5/20/60、STD5/20/60——完整公式见 `features/make_features.py` 顶部注释。

### 3) 标签 — `make_label()` 输出

| 列名 | 含义 | 公式（默认 `n_periods=5, gap=1`） |
|---|---|---|
| `label_5d` | 未来 5 日收益率 | `close.shift(-6) / close.shift(-1) - 1` |

> 默认设定是 `t+1` 开仓、`t+6` 平仓（gap=1 避免使用当日收盘价造成未来函数）。可传参改变持仓期。

### ⚠️ 特征 vs 标签：为什么不都叫 `return`?

`mom_5d` 和 `label_5d` 计算上很像，但方向相反：

- **`mom_5d`**：看**过去**5 天（`t-5 → t`）——当前已知信息→ 特征
- **`label_5d`**：看**未来**5 天（`t+1 → t+6`）——要预测的量 → 标签

如果两者都叫 `return_5d`，训练时一不小心把标签当特征用了，就是经典的数据泄露。

### 4) 配置文件 — `config.yaml`

```yaml
data:
  start: '2024-05-12'
  end:   '2025-05-12'
  splits:
    train:
      start: '2024-05-12'
      end:   '2025-01-22'
    test:
      start: '2025-01-23'
      end:   '2025-05-12'

model:
  objective: regression
  num_leaves: 31
  learning_rate: 0.05
  # …其他 lightgbm 参数
```

- `data.start` / `data.end`：拉取数据的总范围
- `data.splits.train` / `data.splits.test`：训练集 / 测试集的时间划分（按时间切，不是随机切）
- `model`：LightGBM 超参（可手改）
- `scripts/generate_config.py` 只负责生成/更新 `data` 这一块，不会动 `model`。

### 常用路径速查

```
fetcher/get_sp500_list.py   # 从维基拿成分股列表
fetcher/fetch_prices.py     # 批量拉价格数据
scripts/fetch_data.py       # 一键：拉列表 + 拉价 + 入库
data/load_data.py           # prices 表的建表 / 写入 / 读取
data/check_data.py          # 数据质量检查（DataChecker）
features/make_features.py   # 特征计算
labels/make_labels.py       # 标签计算
scripts/generate_config.py  # 生成 data 部分的配置
utils/config.py             # get_db() / load_config()
```
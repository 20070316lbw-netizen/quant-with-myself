使用config.py导入配置
---
```python
from utils.config import get_db, load_config

cfg = load_config()
model_params = cfg["model"]
train_start = cfg["data"]["splits"]["train"]["start"]
```

---
## 最新进展
```bash
PS C:\Users\liu\Desktop\quant> uv run python main.py
[1] 原始数据: 1230607 行, 503 只股票, 日期 2016-05-18 00:00:00 ~ 2026-05-15 00:00:00
[2] 加特征+标签后: 1230607 行
    特征NaN: mom_1d=503, mom_5d=2515, std_5d=2515, KMID=0, KLEN=0, label=3018
[3] 切分后: train=979153 行, test=246945 行
[4] 训练完成, 模型有 100 棵树
[5] Rank IC 评估:
    平均 Rank IC : 0.0159
    ICIR         : 0.0930
    IC>0 占比    : 54.53%
    评估天数     : 486
[6] 分组收益(5组,按预测值从低到高):
    第0组: 平均5日真实收益 = 0.2533%
    第1组: 平均5日真实收益 = 0.2884%
    第2组: 平均5日真实收益 = 0.3010%
    第3组: 平均5日真实收益 = 0.3194%
    第4组: 平均5日真实收益 = 0.5524%
    多空收益(最高组 - 最低组): 0.2990%
    收益随分组单调递增: True
```
---
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
  start: '2016-05-18'
  end: '2026-05-15'
  splits:
    train:
      start: '2016-05-18'
      end: '2024-05-15'      # 前 8 年
    embargo:
      days: 10   # TODO: 此值当前未被代码使用!test.start 是手填的,
                 # embargo 改任何值都不影响实际行为。需让 build_dataset
                 # 按交易日从 train.end + embargo 计算 test.start,
                 # 并加 assert 验证隔离带 >= label窗口(6交易日)。                
      start: '2024-05-24'    # = train.end + 6 个交易日，不是紧接着
      end: '2026-05-15'      # 后 2 年
model:
  objective: regression
  num_leaves: 31
  learning_rate: 0.05
  # …其他 lightgbm 参数
```
TODO: 让 embargo 从"装饰"变成"真正起作用的机制"

正确的设计,是 test.start 不该手填,而应该由代码从 train.end + embargo 算出来。这样"声明"(embargo=10)和"执行"(代码据此算 test.start)就对齐了,配置改一个数,行为真的跟着变。
但这里有个你必须知道的细节,也是你注释里其实已经意识到的:embargo 应该按"交易日"算,不是"日历日"。 因为周末、节假日不交易,日历日 10 天可能只含 6 个交易日,泄露窗口算错。你的 label 是未来 gap+n_periods = 1+5 = 6 个交易日,所以 embargo 至少要 ≥ 6 个交易日才能真正隔断泄露(train 最后一天的 label 用到未来6个交易日的数据,这6天必须落在隔离带里,不能进 test)。
给你修改方向(不直接给你写死的完整代码,因为你现在有能力自己实现,而且自己实现这个你印象会更深):
build_dataset 里,不要再从配置读 te["start"]。改成:从你数据里取出所有实际交易日(你 check_data.py 里那个 get_trading_days() 正好能干这事——你早就写好了这个工具,现在用上),找到 train.end 在交易日序列里的位置,往后数 embargo.days 个交易日,那个日期才是 test.start。配置里 test 段只保留 end,start 由代码算。

伪代码:
```python
trading_days = 所有交易日排序列表  # 用你已有的 get_trading_days()
train_end = tr["end"]
idx = trading_days 中 train_end 的位置
test_start = trading_days[idx + embargo_days]   # 往后数 embargo 个交易日
# 然后 test_df = df[(date >= test_start) & (date <= te["end"])]
# 同时 train_df 末尾也要确保不含 label 会越过边界的样本(进阶,可后做)
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
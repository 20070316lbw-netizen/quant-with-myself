# quant-with-myself

S&P 500 横截面选股的端到端流水线:数据 → 特征 → 标签 → 时间切分 →
LightGBM → 横截面评估。单人自学项目,重点不在跑出高收益,
而在**把每一步做对、并诚实记录哪里曾经做错**。

当前结果是诚实但偏弱的(Rank IC ≈ 0.015,夏普 ≈ 0.17,扣成本后大概率
不显著),这符合仅用 17 个公开技术因子的预期。把弱结果如实写出来,
比粉饰一个好看的数字更重要。

---

## 工程踩坑与设计决策

> 这一节是这个项目的核心。每条都是真实发生的问题:怎么发现、
> 为什么错、怎么修、修完的可量化影响。详细复现代码见我的 cookbook。

### 1. 多空收益:pooled vs per-day(评估口径不一致)

`evaluate.py` 里 Rank IC 是逐日算的(`groupby("date")`),分组收益却是
所有交易日一锅炖(pooled)。**同一个文件,两套口径。**

pooled 的致命处不在均值偏差(平时凑巧接近),而在它求那次均值的瞬间
就销毁了逐日序列 —— 波动、胜率、夏普永远无法还原,而量化最看夏普。

修复:改成 per-day(每天先算多空,得到逐日序列,再统计)。
顺带把脆弱的布尔 `is_monotonic_increasing` 换成组号-收益的秩相关
(连续值,不会被某一组 0.001% 的噪声从 True 翻成 False)。

### 2. 加因子反而让 IC 下降 —— 真正问题是"单次 IC 不可靠"

扩了 MA5/20/60 后,单次跑分 Rank IC 从 0.0138 掉到 0.0097,
五个指标全线下降。直觉解释是"MA 与 ROC 共线,弱信号下稀释信号"。

但这个归因下得太满。为验证它,做了受控合成实验(见 cookbook
`pitfalls/add_features_hurts_ic`):用 LightGBM 换 5 个随机种子,
"加共线因子"对 IC 的影响**方向不稳定** —— seed 0/1 是降
(-0.0015),seed 2/3/4 反而是升(+0.0039)。

结论修正:信号极弱(夏普 ~0.1)时,加一个因子对单次 IC 的扰动
量级**比随机波动还小**。那次 0.0138→0.0097,很可能本身就落在
噪声 band 里,单次运行无法区分"共线稀释"和"纯随机"。

教训:① 单次 IC 涨跌不能用来判断因子好坏,必须多种子/多切分
重复 + 因子相关性 + 特征重要性;② 用 toy 复现验证猜想时,
过度简化(如等权平均)会把要研究的现象本身抹掉。"我归因 Y →
受控实验发现单次观察不足以支撑 Y"本身才是这条的价值。

### 3. early stopping 证明过拟合(而非仅仅防止)

训练原本写死 100 棵树。引入时间切分验证集 + early stopping 后,
模型在**第 27 棵树**就触发早停 —— 即第 27 棵之后的 73 棵全在背噪声。
砍掉后 Rank IC 0.0097 → 0.0150,夏普 0.098 → 0.165,全线回升。
验证曲线呈教科书形状:训练误差单调降,验证误差第 27 轮触底后回升。

关键设计:验证集**按时间切**(不能随机切,拿未来验证过去=作弊),
且训练段与验证段之间**留 embargo**(label 看未来 6 个交易日,
两段紧贴会让早停信号被泄露污染)。这套 embargo 逻辑与
`build_dataset` 里 train→test 的切法一致,不引入新逻辑。

已知局限:early stopping 用 RMSE 做指标,但项目真正关心 Rank IC,
RMSE 最优 ≠ Rank IC 最优。这是有意接受的简化,待用自定义指标改进。
(注:结合 #2,单次早停结果也应多种子复核,不能只看一次。)

### 4. 特征列表三处手抄 → single source of truth

特征名曾在 `make_features` / `train` / `evaluate` 三处各写一份。
扩因子时只改了两处,导致训练用 11 特征、预测用 5 特征,直接崩。
修复:`FEATURE_COLS = list(FEATURE_REGISTRY.keys())` 由 REGISTRY
派生导出,其余文件 import 它,结构上杜绝再次错位。

### 5. notna 误删特征行(注释与实现自相矛盾)

旧代码 `mask = y.notna() & X.notna().all(axis=1)`,后半段把任何含
NaN 特征的行也删了 —— 但注释明说"LightGBM 能处理 NaN 特征"。
靠发现注释与实现矛盾查出。修复:只 `y.notna()`,NaN 特征交给
LightGBM 原生处理。当前真实影响 ≈0.2%(rolling 窗口短),
但会随长窗口因子线性放大,属于拆掉的定时炸弹。

### 6. pandas 3.0 的 groupby-apply 行为变化

`groupby(...).apply()` 后分组键被吸进 index,下一次 `groupby("date")`
报 `KeyError: 'date'`。教训:与其反复赌 apply 参数,不如换成不依赖
该行为的纯聚合写法(`transform` + `unstack`)—— 面对不确定的依赖,
选确定性路径。

### 方法论:可疑 ≠ 有错(一个反面教材)

发现 K 线因子 NaN 全为 0,怀疑有 inf 污染(除以 open=0)。
**没有凭感觉加分母保护**,而是写脚本用真实数据验证 → 确认 open
无 0 / 无负 / 无 NaN,inf 数为 0。结论:不加保护(给不可能触发的
情况加保护是虚假的安心)。换数据源时重跑验证脚本再议。
原则:可疑必须验证,但行动由证据决定,不由直觉决定。

---

## 怎么跑

```bash
uv run python main.py
```

输出 `[1]`~`[7]`:数据概览 → 特征/标签 NaN → 时间切分 →
训练(早停后实际树数) → Rank IC → 分组收益 → per-day 多空统计。
训练历史存到 `train_history.json`(用于画过拟合曲线)。

## 数据约定

所有中间数据是**长格式**:`date`、`ticker` 是普通列(非 MultiIndex),
按 `['ticker','date']` 排序。

### 原始价格 — `prices` 表(DuckDB)

yfinance 拉取的 S&P 500 日线(`auto_adjust=True` 前复权),
写入 `database/sp500.duckdb`。列:`date, ticker, open, high, low,
close, volume`,主键 `(date, ticker)`。

### 特征 — `make_features()`

特征清单由 `features/make_features.py` 的 `FEATURE_REGISTRY`
**唯一定义**(single source of truth)。当前 17 个:

| 类别 | 因子 |
|---|---|
| 动量 | mom_1d, mom_5d |
| 波动(收益) | std_5d |
| K 线形态 | KMID, KLEN, KUP, KLOW, KSFT |
| 动量(ROC) | ROC5, ROC20, ROC60 |
| 均线位置 | MA5, MA20, MA60 |
| 波动(价格) | STD5, STD20, STD60 |

> 已知冗余:`std_5d`(mom_1d 的滚动 std)与 `STD5`(close 的滚动 std)
> 经济含义相近,是一对共线因子 —— 留作"因子筛选"的现实理由。
> 公式见 `make_features.py` 顶部注释(注:该注释规划了 KMID2,
> 实际 REGISTRY 未注册,以 REGISTRY 为准)。

### 标签 — `make_label()`

`label_5d` = 未来 5 日收益率,默认 `n_periods=5, gap=1`:
`t+1` 开仓、`t+6` 平仓(gap=1 避免用当日收盘价造成未来函数)。

> 已知残留:`make_labels.py` 的 `__main__` 自测块仍有 `label.values`
> (cookbook `pandas_assign_values_misalign` 那个坑),主流程已修,
> 自测块待清理。如实记录,不藏。

### 特征 vs 标签:为什么不都叫 return

`mom_5d` 看**过去** 5 天(已知信息→特征),`label_5d` 看**未来**
5 天(待预测→标签)。若都叫 `return_5d`,极易把标签当特征用,
经典泄露。

### 配置 — `config.yaml`

`data.splits` 时间切分,`embargo.days` 隔离带(已在 `build_dataset`
里按**交易日**真实生效:从 train.end 在交易日序列里往后数 embargo
个交易日得 test.start,并 assert ≥ 6)。`model` 是 LightGBM 超参,
`train_control` 是训练过程控制(num_boost_round / early_stopping /
valid_frac),两者分开避免污染 `cfg["model"]`。

## 路径速查

```
fetcher/        成分股列表 + 价格拉取
scripts/        数据入库 / 配置生成 / 数据检查
features/make_features.py   特征(FEATURE_REGISTRY 是唯一特征源)
labels/make_labels.py       标签
model/train.py              build_dataset(时间切分+embargo) / train_lgb(早停)
model/evaluate.py           Rank IC + per-day 分组收益
utils/config.py             get_db() / load_config()
main.py                     端到端入口
```

## 下一步规划(按取舍而非愿望排序)

这些不是空头清单,每条都来自实际踩坑或设计讨论,且已想过取舍、
有意识地决定"先不做"的理由。

### 1. 特征是否物化进数据库

现状:`make_features` 每次跑都全量重算 17 个因子。
- **重算派**(现状):逻辑单一来源,改因子立即生效,无"数据库
  与代码不一致"风险;代价是每次跑都重算,因子多/数据大时慢。
- **物化派**:特征算一次写进 DuckDB,后续直接读长表,快;
  代价是多一层"代码与已存特征是否一致"的接缝(正是 single
  source of truth 那个坑的同类风险)。

倾向:特征稳定后再物化,并配一个"特征版本号 / 哈希"校验,
让"读到的特征"和"当前 make_features 逻辑"对得上。
**未做的理由**:因子还在频繁增改,过早物化等于在流沙上盖楼。

### 2. yaml 驱动的因子子集 + 启动校验

现状:`FEATURE_COLS` 由 REGISTRY 全量导出,想做"只用 K 线类
跑一版"这种因子消融实验,得改代码。
计划:config 里写 `features: [...]` 选子集,启动时校验
"yaml 点名的每个因子都在 REGISTRY 全集里",拼错/漏注册立即报错
(把"静默错位"变成"启动即暴露",正是踩坑 #4 的教训延伸)。
**未做的理由**:这是个独立功能而非补丁,值得单独认真做、单独讲。

### 3. 因子筛选机制(踩坑 #2 的直接出口)

"加 MA 反而 IC 下降"暴露了缺一个客观标尺。计划:因子两两
相关性矩阵 + LightGBM 特征重要性 + 单因子 IC,合起来回答
"某因子该不该留",而不是靠"加完跑一遍肉眼比 IC"。

### 4. 自定义 Rank IC 早停指标(已知局限的出口)

当前 early stopping 用 RMSE,但项目真正关心 Rank IC。
计划:给 `lgb.train` 传自定义 eval 函数,直接用截面 Rank IC
做早停判据,让"停在哪"对齐"我们真正优化的目标"。

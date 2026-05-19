# 横截面标签实验设计 — Qlib 0.9.7 源码对照版

> 这份不是"思路"。每一条都有 Qlib 0.9.7 源码原文背书,行号可复查。
> 用途:回到电脑前照此实现 / commit 时引用 / 收进第二篇笔记。

---

## 0. 这件事的来历(写给面试时的你)

- 昨晚:自己提出假设——横截面相对特征 vs 绝对收益标签,根本不匹配,
  大盘 beta 淹没信号,同时解释"IC 极低"和"U 型分组恒在"。
- 今早:没看任何资料,独立又推到同一结论,并多想一层:回归任务本身
  (L2 拟合绝对收益数值)也是错配的一环。
- 今天:下载 Qlib 0.9.7 源码逐行验证,发现——
  **我推导出的"修复方案",正是 Qlib 跑 LightGBM 的默认范式,
  写死在 handler 里、默认开启。**

这个故事的价值:不是"我照着 Qlib 抄了个配置",是
"我先独立踩坑、独立推导,再发现成熟框架早就用某个设计规避了这个坑"。
前者证明懂为什么,后者只证明会抄。这条主线,而非任何 IC 数字,是面试要讲的。

---

## 1. Qlib 源码铁证(全部 0.9.7,行号可复查)

### 标签公式(qlib/contrib/data/handler.py:152)
```python
def get_label_config(self):
    return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]
```
= close[t+2]/close[t+1]-1,**绝对收益率**。和我的 label_5d 同类,只是
持有期 1 天 vs 5 天。结论:Qlib 的 label 也是绝对收益,我没改错方向。

### 默认处理链(handler.py:37-45)—— 核心铁证
```python
_DEFAULT_LEARN_PROCESSORS = [          # 训练链
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
_DEFAULT_INFER_PROCESSORS = [          # 推理/评估链
    {"class": "ProcessInf"},
    {"class": "ZScoreNorm"},           # 只动 feature,绝不碰 label
    {"class": "Fillna"},
]
```
三个铁证:
1. Qlib 跑 LightGBM 默认就对 label 做截面标准化 → 验证昨晚假设。
2. 默认用 CSZScoreNorm(截面 z-score),不是 CSRankNorm。它做
   `(label − 当天截面均值)/当天截面标准差`——那个"减当天均值"
   就是昨晚方向A说的"减掉大盘 beta",除标准差又把日波动归一。
   → 我提的方向A/方向B,Qlib 默认选了介于两者、更偏A的方案。
3. DropnaLabel + CSZScoreNorm(label) 只在 LEARN 链,INFER 链
   完全不碰 label → 防泄漏不是洁癖,是架构强制。

### 截面标准化实现(qlib/data/dataset/processor.py:322)
```python
df[cols] = df[cols].groupby("datetime", group_keys=False).apply(self.zscore_func)
```
`groupby("datetime")` = 严格按单交易日截面,绝不跨日期。
**自检只需一句话:我的处理前面有没有 groupby(日期)?**

### CSRankNorm 实现(processor.py:352-358)—— 备选方案
```python
t = df[cols].groupby("datetime", group_keys=False).rank(pct=True)
t -= 0.5
t *= 3.46   # 让 [0,1] 均匀分布 → 均值0、标准差≈1 的对称分布
```
`-0.5 *3.46` 的意义:rank 后是 [0,1] 均匀分布(均值0.5、std≈1/3.46),
标准化成零均值对称分布 → **L2 回归天生适合拟合这种目标**
(高斯假设下 MSE 即极大似然)。这正面回答了今早"用回归拟合 rank"
的可行性:可行,但要先标准化成对称分布,Qlib 这两行就是干这个的。

### 防泄漏纪律(handler.py:109-111,DropnaLabel)
```python
def is_for_infer(self) -> bool:
    """The samples are dropped according to label. So it is not usable for inference"""
    return False
```
任何依赖 label 的处理,只能在训练侧。预测时不知道未来 label,
若用 label 筛/排样本 → 训练-推理分布不一致 → IC 虚高型泄漏,看不出来。

---

## 2. 今天的实验设计(单变量铁律)

### 三层错配,今天只动第一层

| 层 | 现状(错配) | 对齐后 | 今天动? |
|---|---|---|---|
| 标签处理 | 绝对收益直接喂模型 | 进模型前过截面标准化 | ✅ 唯一变量 |
| 任务/早停 | L2 回归 / RMSE | 回归(冻结) | ❌ 冻结 |
| 评估口径 | — | 用原始绝对收益算分组收益 | ❌ 不变 |

### 实验表

| 实验 | 标签公式 | 横截面处理 | 其余 | 单变量? |
|---|---|---|---|---|
| 基线 | label_5d 不动 | 无 | 全集/切分/超参/RMSE 冻结 | — |
| 今天 | label_5d **公式完全不动** | 训练集加 CS 标准化 | 同上,全冻结 | ✅ 只动这一层 |

关键决策:**不改 make_labels.py 的公式。** Qlib 范式证明正确做法是
保留绝对收益公式 + 加一个可挂载的处理步骤。这样 Phase 1 守住的
无泄漏完全保留,新增泄漏面只剩那一个 groupby。

CS 标准化选哪种(二选一,今天先选一个,另一个留 Phase 3 对照):
- **CSRankNorm 式**(纯排名→标准化):对极端值最鲁棒,完全只保留序。
  和"选股排序"终极目标最同构。**建议今天先用这个。**
- **CSZScoreNorm 式**(减均值除标准差):Qlib 默认,保留相对数值差距,
  减掉大盘 beta。留作 Phase 3 对照(就是昨晚的方向A)。

### 证伪条件(先承诺,跑前写死,防自圆其说)
- IC 没动 + U 型还在 → **假设被证伪**,回查更底层(evaluate 口径)。
- IC 温和回升(0.011 → 0.02~0.03 量级)+ U 型减弱/消失 → 假设成立。
- IC 暴涨到 0.05+ → **先查泄漏,不是先高兴**。横截面做对了是温和
  回升;暴涨是 rank 跨日期或成分股用了未来名单的典型特征。
- IC 回升但 U 型仍在 → 假设只对一半,U 型另有成因(最有信息量)。

---

## 3. yaml 开关架构(兑现"一直想做"的三条规划)

Qlib 不把 processor 散在 yaml 里写 if-else,而是:
- 默认值集中写在一处(`_DEFAULT_LEARN_PROCESSORS`)= single source of truth
- yaml 通过 handler kwargs 整体覆盖

照搬到本项目:
1. config.yaml 加 `label_processors`,默认 `[cs_rank]`。
2. build_dataset 时间切分后:**训练集**按 label_processors 施加处理;
   **测试集不施加任何 label 处理**(对应 Qlib LEARN/INFER 分离)。
3. 跑基线 vs 实验:`label_processors: []` vs `[cs_rank]`,改一行。
4. 一步兑现三条规划:① yaml 驱动开关 ② 横截面标签实验
   ③ 处理链单一来源(治 README 记的"代码与配置不一致"老坑)。

单变量铁律在此架构下天然成立:唯一差别是那个列表空不空。

---

## 4. 回到电脑前的执行清单

- [ ] 写 CS 标准化函数:`groupby(交易日)` 后 rank(pct=True),再 `-0.5 *3.46`
- [ ] 自检:rank 前面有没有 groupby(日期)?(防跨日期泄漏,唯一自检点)
- [ ] 接进 build_dataset:**只对训练集施加**,测试集 label 原样保留
- [ ] config.yaml 加 label_processors 开关,默认列表 + yaml 可覆盖
- [ ] evaluate 确认:分组真实收益用**原始 label_5d 绝对收益**,不是处理后的值
- [ ] 跑基线(开关空)→ 记 IC/U型 → 跑实验(开关=cs_rank)→ 对照
- [ ] 按证伪条件判定,**先看是不是泄漏(IC暴涨)再下结论**
- [ ] 实验记录写进第二篇笔记:假设→独立推导→Qlib源码验证→结果
- [ ] commit 分开写:① 加开关架构 ② 加CS处理 ③ 实验结果,
      每个 message 写清"为什么这么改"

---

## 5. 一句话总线(面试脱稿用)

"我项目 IC 趴在 0.01,先自己怀疑特征共线,做消融实验把这个假设
推翻了;再推导出可能是横截面特征配了绝对收益标签,任务还是回归;
后来读 Qlib 源码,发现它跑 LightGBM 默认就对 label 做截面标准化,
我独立推导出的修复,正是业界标准管线写死在 handler 里的默认行为。
我现在能讲清楚它为什么要那个 `*3.46`,以及为什么 label 处理
必须只在训练侧。"

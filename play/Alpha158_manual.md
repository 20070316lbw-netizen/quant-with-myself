Alpha158 因子实现手册
qlib Alpha158 → 自定义 pandas/numpy 实现进度追踪
因子总数：158   窗口期：5 / 10 / 20 / 30 / 60 日   数据列：close, open, high, low, vwap, volume
使用说明
• 完成列：实现后在空格内手写勾 ✓，或直接在代码里备注 Done
• 表达式中 Ref(x,d) = x 向前平移 d 日；Max/Min 为滚动最大/最小；Mean/Std 为滚动均值/标准差
• Greater(a,b) = max(a,b)；Less(a,b) = min(a,b)；所有因子以当日 $close 归一化，消除量纲
• 实现顺序建议：K线(9) → ROC/MA/STD(15) → CORR/CORD/CNT*(15) → RSI族(15) → 量因子(20)

编号
因子名称
qlib 表达式
经济含义 / 信号解读
完成
▌ 一、K线形态因子（Kbar）
001
KMID
($close-$open)/$open
实体涨幅：收盘相对开盘的涨跌幅，反映当日多空力量
☐
002
KLEN
($high-$low)/$open
振幅：全日波动范围除以开盘价，反映当日波动强度
☐
003
KMID2
($close-$open)/($high-$low+1e-12)
实体占振幅比：实体涨幅 / 全日振幅，反映方向性强度
☐
004
KUP
($high-max($open,$close))/$open
上影线长度 / 开盘价，反映上方压力（上影越长压力越大）
☐
005
KUP2
($high-max($open,$close))/($high-$low+1e-12)
上影线占振幅比，同上影信号但归一化到 [0,1]
☐
006
KLOW
(min($open,$close)-$low)/$open
下影线长度 / 开盘价，反映下方支撑（下影越长支撑越强）
☐
007
KLOW2
(min($open,$close)-$low)/($high-$low+1e-12)
下影线占振幅比，同下影信号但归一化到 [0,1]
☐
008
KSFT
(2*$close-$high-$low)/$open
收盘偏移度：收盘相对于当日高低均值的偏移，正值偏强
☐
009
KSFT2
(2*$close-$high-$low)/($high-$low+1e-12)
收盘偏移占振幅比，与 KSFT 含义相同但规模统一
☐
▌ 二、原始价格因子（Price）
010
OPEN0
$open/$close
当日开盘价相对收盘价，>1表示低开
☐
011
HIGH0
$high/$close
当日最高价相对收盘价，反映当日最强点位
☐
012
LOW0
$low/$close
当日最低价相对收盘价，反映当日最弱点位
☐
013
VWAP0
$vwap/$close
VWAP 相对收盘价，>1表示收盘弱于均值成本
☐
▌ 三、价格动量 ROC（Rate of Change）
014
ROC5
Ref($close,5)/$close
5日前收盘价 / 当日收盘：<1表示近期上涨，>1表示近期下跌（注意方向与命名相反）
☐
015
ROC10
Ref($close,10)/$close
10日前收盘价 / 当日收盘：<1表示近期上涨，>1表示近期下跌（注意方向与命名相反）
☐
016
ROC20
Ref($close,20)/$close
20日前收盘价 / 当日收盘：<1表示近期上涨，>1表示近期下跌（注意方向与命名相反）
☐
017
ROC30
Ref($close,30)/$close
30日前收盘价 / 当日收盘：<1表示近期上涨，>1表示近期下跌（注意方向与命名相反）
☐
018
ROC60
Ref($close,60)/$close
60日前收盘价 / 当日收盘：<1表示近期上涨，>1表示近期下跌（注意方向与命名相反）
☐
▌ 四、移动平均 MA（Moving Average）
019
MA5
Mean($close,5)/$close
5日均价 / 当日收盘：<1表示股价高于均线（上穿），>1表示低于均线（下穿）
☐
020
MA10
Mean($close,10)/$close
10日均价 / 当日收盘：<1表示股价高于均线（上穿），>1表示低于均线（下穿）
☐
021
MA20
Mean($close,20)/$close
20日均价 / 当日收盘：<1表示股价高于均线（上穿），>1表示低于均线（下穿）
☐
022
MA30
Mean($close,30)/$close
30日均价 / 当日收盘：<1表示股价高于均线（上穿），>1表示低于均线（下穿）
☐
023
MA60
Mean($close,60)/$close
60日均价 / 当日收盘：<1表示股价高于均线（上穿），>1表示低于均线（下穿）
☐
▌ 五、收盘波动率 STD
024
STD5
Std($close,5)/$close
5日收盘价标准差 / 当日收盘，衡量近期波动强度（去单位化）
☐
025
STD10
Std($close,10)/$close
10日收盘价标准差 / 当日收盘，衡量近期波动强度（去单位化）
☐
026
STD20
Std($close,20)/$close
20日收盘价标准差 / 当日收盘，衡量近期波动强度（去单位化）
☐
027
STD30
Std($close,30)/$close
30日收盘价标准差 / 当日收盘，衡量近期波动强度（去单位化）
☐
028
STD60
Std($close,60)/$close
60日收盘价标准差 / 当日收盘，衡量近期波动强度（去单位化）
☐
▌ 六、线性趋势斜率 BETA
029
BETA5
Slope($close,5)/$close
5日收盘价线性回归斜率 / 收盘价，正值代表上升趋势强度
☐
030
BETA10
Slope($close,10)/$close
10日收盘价线性回归斜率 / 收盘价，正值代表上升趋势强度
☐
031
BETA20
Slope($close,20)/$close
20日收盘价线性回归斜率 / 收盘价，正值代表上升趋势强度
☐
032
BETA30
Slope($close,30)/$close
30日收盘价线性回归斜率 / 收盘价，正值代表上升趋势强度
☐
033
BETA60
Slope($close,60)/$close
60日收盘价线性回归斜率 / 收盘价，正值代表上升趋势强度
☐
▌ 七、线性拟合度 RSQR（R²）
034
RSQR5
Rsquare($close,5)
5日收盘价线性回归 R²，越接近1代表趋势越线性/稳定
☐
035
RSQR10
Rsquare($close,10)
10日收盘价线性回归 R²，越接近1代表趋势越线性/稳定
☐
036
RSQR20
Rsquare($close,20)
20日收盘价线性回归 R²，越接近1代表趋势越线性/稳定
☐
037
RSQR30
Rsquare($close,30)
30日收盘价线性回归 R²，越接近1代表趋势越线性/稳定
☐
038
RSQR60
Rsquare($close,60)
60日收盘价线性回归 R²，越接近1代表趋势越线性/稳定
☐
▌ 八、线性残差 RESI
039
RESI5
Resi($close,5)/$close
5日线性回归残差 / 收盘价，衡量价格偏离线性趋势的程度
☐
040
RESI10
Resi($close,10)/$close
10日线性回归残差 / 收盘价，衡量价格偏离线性趋势的程度
☐
041
RESI20
Resi($close,20)/$close
20日线性回归残差 / 收盘价，衡量价格偏离线性趋势的程度
☐
042
RESI30
Resi($close,30)/$close
30日线性回归残差 / 收盘价，衡量价格偏离线性趋势的程度
☐
043
RESI60
Resi($close,60)/$close
60日线性回归残差 / 收盘价，衡量价格偏离线性趋势的程度
☐
▌ 九、N日最高价 MAX
044
MAX5
Max($high,5)/$close
5日内最高价 / 当日收盘，反映价格离历史高点的距离
☐
045
MAX10
Max($high,10)/$close
10日内最高价 / 当日收盘，反映价格离历史高点的距离
☐
046
MAX20
Max($high,20)/$close
20日内最高价 / 当日收盘，反映价格离历史高点的距离
☐
047
MAX30
Max($high,30)/$close
30日内最高价 / 当日收盘，反映价格离历史高点的距离
☐
048
MAX60
Max($high,60)/$close
60日内最高价 / 当日收盘，反映价格离历史高点的距离
☐
▌ 十、N日最低价 MIN
049
MIN5
Min($low,5)/$close
5日内最低价 / 当日收盘，反映价格离历史低点的距离
☐
050
MIN10
Min($low,10)/$close
10日内最低价 / 当日收盘，反映价格离历史低点的距离
☐
051
MIN20
Min($low,20)/$close
20日内最低价 / 当日收盘，反映价格离历史低点的距离
☐
052
MIN30
Min($low,30)/$close
30日内最低价 / 当日收盘，反映价格离历史低点的距离
☐
053
MIN60
Min($low,60)/$close
60日内最低价 / 当日收盘，反映价格离历史低点的距离
☐
▌ 十一、80%分位数 QTLU
054
QTLU5
Quantile($close,5,0.8)/$close
5日收盘价80%分位 / 当日收盘，反映当前价格相对近期高位的位置
☐
055
QTLU10
Quantile($close,10,0.8)/$close
10日收盘价80%分位 / 当日收盘，反映当前价格相对近期高位的位置
☐
056
QTLU20
Quantile($close,20,0.8)/$close
20日收盘价80%分位 / 当日收盘，反映当前价格相对近期高位的位置
☐
057
QTLU30
Quantile($close,30,0.8)/$close
30日收盘价80%分位 / 当日收盘，反映当前价格相对近期高位的位置
☐
058
QTLU60
Quantile($close,60,0.8)/$close
60日收盘价80%分位 / 当日收盘，反映当前价格相对近期高位的位置
☐
▌ 十二、20%分位数 QTLD
059
QTLD5
Quantile($close,5,0.2)/$close
5日收盘价20%分位 / 当日收盘，反映当前价格相对近期低位的位置
☐
060
QTLD10
Quantile($close,10,0.2)/$close
10日收盘价20%分位 / 当日收盘，反映当前价格相对近期低位的位置
☐
061
QTLD20
Quantile($close,20,0.2)/$close
20日收盘价20%分位 / 当日收盘，反映当前价格相对近期低位的位置
☐
062
QTLD30
Quantile($close,30,0.2)/$close
30日收盘价20%分位 / 当日收盘，反映当前价格相对近期低位的位置
☐
063
QTLD60
Quantile($close,60,0.2)/$close
60日收盘价20%分位 / 当日收盘，反映当前价格相对近期低位的位置
☐
▌ 十三、价格百分位排名 RANK
064
RANK5
Rank($close,5)
当日收盘在过去5日的百分位 [0,1]，0.5表示处于中位，越高代表近期强势
☐
065
RANK10
Rank($close,10)
当日收盘在过去10日的百分位 [0,1]，0.5表示处于中位，越高代表近期强势
☐
066
RANK20
Rank($close,20)
当日收盘在过去20日的百分位 [0,1]，0.5表示处于中位，越高代表近期强势
☐
067
RANK30
Rank($close,30)
当日收盘在过去30日的百分位 [0,1]，0.5表示处于中位，越高代表近期强势
☐
068
RANK60
Rank($close,60)
当日收盘在过去60日的百分位 [0,1]，0.5表示处于中位，越高代表近期强势
☐
▌ 十四、KDJ随机值 RSV
069
RSV5
($close-Min($low,5))/(Max($high,5)-Min($low,5)+1e-12)
当日收盘在5日高低区间内的位置 [0,1]，是 KDJ 中 RSV 的原始形式
☐
070
RSV10
($close-Min($low,10))/(Max($high,10)-Min($low,10)+1e-12)
当日收盘在10日高低区间内的位置 [0,1]，是 KDJ 中 RSV 的原始形式
☐
071
RSV20
($close-Min($low,20))/(Max($high,20)-Min($low,20)+1e-12)
当日收盘在20日高低区间内的位置 [0,1]，是 KDJ 中 RSV 的原始形式
☐
072
RSV30
($close-Min($low,30))/(Max($high,30)-Min($low,30)+1e-12)
当日收盘在30日高低区间内的位置 [0,1]，是 KDJ 中 RSV 的原始形式
☐
073
RSV60
($close-Min($low,60))/(Max($high,60)-Min($low,60)+1e-12)
当日收盘在60日高低区间内的位置 [0,1]，是 KDJ 中 RSV 的原始形式
☐
▌ 十五、距最高点天数 IMAX（Aroon）
074
IMAX5
IdxMax($high,5)/5
过去5日最高点距今天数 / 5，接近0表示最高点在近期（Aroon 上轨原型）
☐
075
IMAX10
IdxMax($high,10)/10
过去10日最高点距今天数 / 10，接近0表示最高点在近期（Aroon 上轨原型）
☐
076
IMAX20
IdxMax($high,20)/20
过去20日最高点距今天数 / 20，接近0表示最高点在近期（Aroon 上轨原型）
☐
077
IMAX30
IdxMax($high,30)/30
过去30日最高点距今天数 / 30，接近0表示最高点在近期（Aroon 上轨原型）
☐
078
IMAX60
IdxMax($high,60)/60
过去60日最高点距今天数 / 60，接近0表示最高点在近期（Aroon 上轨原型）
☐
▌ 十六、距最低点天数 IMIN（Aroon）
079
IMIN5
IdxMin($low,5)/5
过去5日最低点距今天数 / 5，接近0表示最低点在近期（Aroon 下轨原型）
☐
080
IMIN10
IdxMin($low,10)/10
过去10日最低点距今天数 / 10，接近0表示最低点在近期（Aroon 下轨原型）
☐
081
IMIN20
IdxMin($low,20)/20
过去20日最低点距今天数 / 20，接近0表示最低点在近期（Aroon 下轨原型）
☐
082
IMIN30
IdxMin($low,30)/30
过去30日最低点距今天数 / 30，接近0表示最低点在近期（Aroon 下轨原型）
☐
083
IMIN60
IdxMin($low,60)/60
过去60日最低点距今天数 / 60，接近0表示最低点在近期（Aroon 下轨原型）
☐
▌ 十七、高低点时序差 IMXD
084
IMXD5
(IdxMax($high,5)-IdxMin($low,5))/5
(距最高点天数 - 距最低点天数) / 5，正值表示最低点更近，暗示下行动量
☐
085
IMXD10
(IdxMax($high,10)-IdxMin($low,10))/10
(距最高点天数 - 距最低点天数) / 10，正值表示最低点更近，暗示下行动量
☐
086
IMXD20
(IdxMax($high,20)-IdxMin($low,20))/20
(距最高点天数 - 距最低点天数) / 20，正值表示最低点更近，暗示下行动量
☐
087
IMXD30
(IdxMax($high,30)-IdxMin($low,30))/30
(距最高点天数 - 距最低点天数) / 30，正值表示最低点更近，暗示下行动量
☐
088
IMXD60
(IdxMax($high,60)-IdxMin($low,60))/60
(距最高点天数 - 距最低点天数) / 60，正值表示最低点更近，暗示下行动量
☐
▌ 十八、价量相关（绝对价） CORR
089
CORR5
Corr($close,Log($volume+1),5)
5日内收盘价与 log(成交量) 的相关性，正值为量价齐升，负值为背离
☐
090
CORR10
Corr($close,Log($volume+1),10)
10日内收盘价与 log(成交量) 的相关性，正值为量价齐升，负值为背离
☐
091
CORR20
Corr($close,Log($volume+1),20)
20日内收盘价与 log(成交量) 的相关性，正值为量价齐升，负值为背离
☐
092
CORR30
Corr($close,Log($volume+1),30)
30日内收盘价与 log(成交量) 的相关性，正值为量价齐升，负值为背离
☐
093
CORR60
Corr($close,Log($volume+1),60)
60日内收盘价与 log(成交量) 的相关性，正值为量价齐升，负值为背离
☐
▌ 十九、价量变化相关 CORD
094
CORD5
Corr($close/Ref($close,1),Log($volume/Ref($volume,1)+1),5)
5日内日涨跌幅与日成交量变化率的相关性，更稳健的量价关系信号
☐
095
CORD10
Corr($close/Ref($close,1),Log($volume/Ref($volume,1)+1),10)
10日内日涨跌幅与日成交量变化率的相关性，更稳健的量价关系信号
☐
096
CORD20
Corr($close/Ref($close,1),Log($volume/Ref($volume,1)+1),20)
20日内日涨跌幅与日成交量变化率的相关性，更稳健的量价关系信号
☐
097
CORD30
Corr($close/Ref($close,1),Log($volume/Ref($volume,1)+1),30)
30日内日涨跌幅与日成交量变化率的相关性，更稳健的量价关系信号
☐
098
CORD60
Corr($close/Ref($close,1),Log($volume/Ref($volume,1)+1),60)
60日内日涨跌幅与日成交量变化率的相关性，更稳健的量价关系信号
☐
▌ 二十、上涨天数占比 CNTP
099
CNTP5
Mean($close>Ref($close,1),5)
过去5日中收盘上涨的天数占比，越高表示近期上涨一致性越强
☐
100
CNTP10
Mean($close>Ref($close,1),10)
过去10日中收盘上涨的天数占比，越高表示近期上涨一致性越强
☐
101
CNTP20
Mean($close>Ref($close,1),20)
过去20日中收盘上涨的天数占比，越高表示近期上涨一致性越强
☐
102
CNTP30
Mean($close>Ref($close,1),30)
过去30日中收盘上涨的天数占比，越高表示近期上涨一致性越强
☐
103
CNTP60
Mean($close>Ref($close,1),60)
过去60日中收盘上涨的天数占比，越高表示近期上涨一致性越强
☐
▌ 二十一、下跌天数占比 CNTN
104
CNTN5
Mean($close<Ref($close,1),5)
过去5日中收盘下跌的天数占比，越高表示近期下跌一致性越强
☐
105
CNTN10
Mean($close<Ref($close,1),10)
过去10日中收盘下跌的天数占比，越高表示近期下跌一致性越强
☐
106
CNTN20
Mean($close<Ref($close,1),20)
过去20日中收盘下跌的天数占比，越高表示近期下跌一致性越强
☐
107
CNTN30
Mean($close<Ref($close,1),30)
过去30日中收盘下跌的天数占比，越高表示近期下跌一致性越强
☐
108
CNTN60
Mean($close<Ref($close,1),60)
过去60日中收盘下跌的天数占比，越高表示近期下跌一致性越强
☐
▌ 二十二、涨跌天数差 CNTD
109
CNTD5
Mean($close>Ref($close,1),5)-Mean($close<Ref($close,1),5)
过去5日涨天占比 - 跌天占比，正值偏多，负值偏空
☐
110
CNTD10
Mean($close>Ref($close,1),10)-Mean($close<Ref($close,1),10)
过去10日涨天占比 - 跌天占比，正值偏多，负值偏空
☐
111
CNTD20
Mean($close>Ref($close,1),20)-Mean($close<Ref($close,1),20)
过去20日涨天占比 - 跌天占比，正值偏多，负值偏空
☐
112
CNTD30
Mean($close>Ref($close,1),30)-Mean($close<Ref($close,1),30)
过去30日涨天占比 - 跌天占比，正值偏多，负值偏空
☐
113
CNTD60
Mean($close>Ref($close,1),60)-Mean($close<Ref($close,1),60)
过去60日涨天占比 - 跌天占比，正值偏多，负值偏空
☐
▌ 二十三、上涨幅度占比 SUMP（RSI型）
114
SUMP5
Sum(max($close-Ref($close,1),0),5)/(Sum(Abs($close-Ref($close,1)),5)+1e-12)
5日总涨幅 / 总绝对涨跌幅，类似 RSI 分子，值越高代表买方力量越强
☐
115
SUMP10
Sum(max($close-Ref($close,1),0),10)/(Sum(Abs($close-Ref($close,1)),10)+1e-12)
10日总涨幅 / 总绝对涨跌幅，类似 RSI 分子，值越高代表买方力量越强
☐
116
SUMP20
Sum(max($close-Ref($close,1),0),20)/(Sum(Abs($close-Ref($close,1)),20)+1e-12)
20日总涨幅 / 总绝对涨跌幅，类似 RSI 分子，值越高代表买方力量越强
☐
117
SUMP30
Sum(max($close-Ref($close,1),0),30)/(Sum(Abs($close-Ref($close,1)),30)+1e-12)
30日总涨幅 / 总绝对涨跌幅，类似 RSI 分子，值越高代表买方力量越强
☐
118
SUMP60
Sum(max($close-Ref($close,1),0),60)/(Sum(Abs($close-Ref($close,1)),60)+1e-12)
60日总涨幅 / 总绝对涨跌幅，类似 RSI 分子，值越高代表买方力量越强
☐
▌ 二十四、下跌幅度占比 SUMN（RSI型）
119
SUMN5
Sum(max(Ref($close,1)-$close,0),5)/(Sum(Abs($close-Ref($close,1)),5)+1e-12)
5日总跌幅 / 总绝对涨跌幅，SUMN = 1 - SUMP，值越高代表卖方力量越强
☐
120
SUMN10
Sum(max(Ref($close,1)-$close,0),10)/(Sum(Abs($close-Ref($close,1)),10)+1e-12)
10日总跌幅 / 总绝对涨跌幅，SUMN = 1 - SUMP，值越高代表卖方力量越强
☐
121
SUMN20
Sum(max(Ref($close,1)-$close,0),20)/(Sum(Abs($close-Ref($close,1)),20)+1e-12)
20日总跌幅 / 总绝对涨跌幅，SUMN = 1 - SUMP，值越高代表卖方力量越强
☐
122
SUMN30
Sum(max(Ref($close,1)-$close,0),30)/(Sum(Abs($close-Ref($close,1)),30)+1e-12)
30日总跌幅 / 总绝对涨跌幅，SUMN = 1 - SUMP，值越高代表卖方力量越强
☐
123
SUMN60
Sum(max(Ref($close,1)-$close,0),60)/(Sum(Abs($close-Ref($close,1)),60)+1e-12)
60日总跌幅 / 总绝对涨跌幅，SUMN = 1 - SUMP，值越高代表卖方力量越强
☐
▌ 二十五、涨跌力量差 SUMD（RSI型）
124
SUMD5
(Sum(max($close-Ref($close,1),0),5)-Sum(max(Ref($close,1)-$close,0),5))/(Sum(Abs($close-Ref($close,1)),5)+1e-12)
(总涨幅 - 总跌幅) / 总绝对涨跌幅，等价于 2×SUMP-1，即 RSI 的线性变换
☐
125
SUMD10
(Sum(max($close-Ref($close,1),0),10)-Sum(max(Ref($close,1)-$close,0),10))/(Sum(Abs($close-Ref($close,1)),10)+1e-12)
(总涨幅 - 总跌幅) / 总绝对涨跌幅，等价于 2×SUMP-1，即 RSI 的线性变换
☐
126
SUMD20
(Sum(max($close-Ref($close,1),0),20)-Sum(max(Ref($close,1)-$close,0),20))/(Sum(Abs($close-Ref($close,1)),20)+1e-12)
(总涨幅 - 总跌幅) / 总绝对涨跌幅，等价于 2×SUMP-1，即 RSI 的线性变换
☐
127
SUMD30
(Sum(max($close-Ref($close,1),0),30)-Sum(max(Ref($close,1)-$close,0),30))/(Sum(Abs($close-Ref($close,1)),30)+1e-12)
(总涨幅 - 总跌幅) / 总绝对涨跌幅，等价于 2×SUMP-1，即 RSI 的线性变换
☐
128
SUMD60
(Sum(max($close-Ref($close,1),0),60)-Sum(max(Ref($close,1)-$close,0),60))/(Sum(Abs($close-Ref($close,1)),60)+1e-12)
(总涨幅 - 总跌幅) / 总绝对涨跌幅，等价于 2×SUMP-1，即 RSI 的线性变换
☐
▌ 二十六、成交量均值 VMA
129
VMA5
Mean($volume,5)/($volume+1e-12)
5日均量 / 当日成交量，>1表示当日成交量低于均量（缩量）
☐
130
VMA10
Mean($volume,10)/($volume+1e-12)
10日均量 / 当日成交量，>1表示当日成交量低于均量（缩量）
☐
131
VMA20
Mean($volume,20)/($volume+1e-12)
20日均量 / 当日成交量，>1表示当日成交量低于均量（缩量）
☐
132
VMA30
Mean($volume,30)/($volume+1e-12)
30日均量 / 当日成交量，>1表示当日成交量低于均量（缩量）
☐
133
VMA60
Mean($volume,60)/($volume+1e-12)
60日均量 / 当日成交量，>1表示当日成交量低于均量（缩量）
☐
▌ 二十七、成交量波动率 VSTD
134
VSTD5
Std($volume,5)/($volume+1e-12)
5日成交量标准差 / 当日成交量，衡量成交量的相对波动程度
☐
135
VSTD10
Std($volume,10)/($volume+1e-12)
10日成交量标准差 / 当日成交量，衡量成交量的相对波动程度
☐
136
VSTD20
Std($volume,20)/($volume+1e-12)
20日成交量标准差 / 当日成交量，衡量成交量的相对波动程度
☐
137
VSTD30
Std($volume,30)/($volume+1e-12)
30日成交量标准差 / 当日成交量，衡量成交量的相对波动程度
☐
138
VSTD60
Std($volume,60)/($volume+1e-12)
60日成交量标准差 / 当日成交量，衡量成交量的相对波动程度
☐
▌ 二十八、量价加权波动率 WVMA
139
WVMA5
Std(Abs($close/Ref($close,1)-1)*$volume,5)/(Mean(Abs($close/Ref($close,1)-1)*$volume,5)+1e-12)
5日量价乘积的变异系数，波动率高且成交量大时该值更高
☐
140
WVMA10
Std(Abs($close/Ref($close,1)-1)*$volume,10)/(Mean(Abs($close/Ref($close,1)-1)*$volume,10)+1e-12)
10日量价乘积的变异系数，波动率高且成交量大时该值更高
☐
141
WVMA20
Std(Abs($close/Ref($close,1)-1)*$volume,20)/(Mean(Abs($close/Ref($close,1)-1)*$volume,20)+1e-12)
20日量价乘积的变异系数，波动率高且成交量大时该值更高
☐
142
WVMA30
Std(Abs($close/Ref($close,1)-1)*$volume,30)/(Mean(Abs($close/Ref($close,1)-1)*$volume,30)+1e-12)
30日量价乘积的变异系数，波动率高且成交量大时该值更高
☐
143
WVMA60
Std(Abs($close/Ref($close,1)-1)*$volume,60)/(Mean(Abs($close/Ref($close,1)-1)*$volume,60)+1e-12)
60日量价乘积的变异系数，波动率高且成交量大时该值更高
☐
▌ 二十九、成交量上升占比 VSUMP
144
VSUMP5
Sum(max($volume-Ref($volume,1),0),5)/(Sum(Abs($volume-Ref($volume,1)),5)+1e-12)
5日量增幅 / 量总变动幅度，成交量 RSI 型因子，反映资金流入强度
☐
145
VSUMP10
Sum(max($volume-Ref($volume,1),0),10)/(Sum(Abs($volume-Ref($volume,1)),10)+1e-12)
10日量增幅 / 量总变动幅度，成交量 RSI 型因子，反映资金流入强度
☐
146
VSUMP20
Sum(max($volume-Ref($volume,1),0),20)/(Sum(Abs($volume-Ref($volume,1)),20)+1e-12)
20日量增幅 / 量总变动幅度，成交量 RSI 型因子，反映资金流入强度
☐
147
VSUMP30
Sum(max($volume-Ref($volume,1),0),30)/(Sum(Abs($volume-Ref($volume,1)),30)+1e-12)
30日量增幅 / 量总变动幅度，成交量 RSI 型因子，反映资金流入强度
☐
148
VSUMP60
Sum(max($volume-Ref($volume,1),0),60)/(Sum(Abs($volume-Ref($volume,1)),60)+1e-12)
60日量增幅 / 量总变动幅度，成交量 RSI 型因子，反映资金流入强度
☐
▌ 三十、成交量下降占比 VSUMN
149
VSUMN5
Sum(max(Ref($volume,1)-$volume,0),5)/(Sum(Abs($volume-Ref($volume,1)),5)+1e-12)
5日量减幅 / 量总变动幅度，VSUMN = 1 - VSUMP，反映资金流出强度
☐
150
VSUMN10
Sum(max(Ref($volume,1)-$volume,0),10)/(Sum(Abs($volume-Ref($volume,1)),10)+1e-12)
10日量减幅 / 量总变动幅度，VSUMN = 1 - VSUMP，反映资金流出强度
☐
151
VSUMN20
Sum(max(Ref($volume,1)-$volume,0),20)/(Sum(Abs($volume-Ref($volume,1)),20)+1e-12)
20日量减幅 / 量总变动幅度，VSUMN = 1 - VSUMP，反映资金流出强度
☐
152
VSUMN30
Sum(max(Ref($volume,1)-$volume,0),30)/(Sum(Abs($volume-Ref($volume,1)),30)+1e-12)
30日量减幅 / 量总变动幅度，VSUMN = 1 - VSUMP，反映资金流出强度
☐
153
VSUMN60
Sum(max(Ref($volume,1)-$volume,0),60)/(Sum(Abs($volume-Ref($volume,1)),60)+1e-12)
60日量减幅 / 量总变动幅度，VSUMN = 1 - VSUMP，反映资金流出强度
☐
▌ 三十一、量增减力量差 VSUMD
154
VSUMD5
(Sum(max($volume-Ref($volume,1),0),5)-Sum(max(Ref($volume,1)-$volume,0),5))/(Sum(Abs($volume-Ref($volume,1)),5)+1e-12)
(量增幅 - 量减幅) / 量总变动幅度，成交量的多空力量差，正值代表放量趋势
☐
155
VSUMD10
(Sum(max($volume-Ref($volume,1),0),10)-Sum(max(Ref($volume,1)-$volume,0),10))/(Sum(Abs($volume-Ref($volume,1)),10)+1e-12)
(量增幅 - 量减幅) / 量总变动幅度，成交量的多空力量差，正值代表放量趋势
☐
156
VSUMD20
(Sum(max($volume-Ref($volume,1),0),20)-Sum(max(Ref($volume,1)-$volume,0),20))/(Sum(Abs($volume-Ref($volume,1)),20)+1e-12)
(量增幅 - 量减幅) / 量总变动幅度，成交量的多空力量差，正值代表放量趋势
☐
157
VSUMD30
(Sum(max($volume-Ref($volume,1),0),30)-Sum(max(Ref($volume,1)-$volume,0),30))/(Sum(Abs($volume-Ref($volume,1)),30)+1e-12)
(量增幅 - 量减幅) / 量总变动幅度，成交量的多空力量差，正值代表放量趋势
☐
158
VSUMD60
(Sum(max($volume-Ref($volume,1),0),60)-Sum(max(Ref($volume,1)-$volume,0),60))/(Sum(Abs($volume-Ref($volume,1)),60)+1e-12)
(量增幅 - 量减幅) / 量总变动幅度，成交量的多空力量差，正值代表放量趋势
☐

来源：qlib/contrib/data/loader.py · Alpha158DL.get_feature_config()  |  整理：quant-with-myself
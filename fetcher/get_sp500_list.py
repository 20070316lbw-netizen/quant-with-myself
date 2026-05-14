"""
拉取标普500成分股列表
来源: 维基百科
"""
import pandas as pd
import requests
from io import StringIO


def get_sp500_tickers():
    """
    从维基百科拉取当前的标普500成分股
    返回: DataFrame,包含 symbol、security、sector 等字段
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    # 加个 User-Agent 假装自己是浏览器,避免被拒绝
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }
    
    # 先用 requests 把网页抓下来
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # 如果出错了,抛异常
    
    # 让 pandas 解析网页里的表格
    # StringIO 是把字符串当文件用,pd.read_html 需要这个
    tables = pd.read_html(StringIO(response.text))
    
    # 第一个表格就是成分股列表
    sp500 = tables[0]
    
    # 把列名改成统一的小写下划线风格,方便后面用
    sp500 = sp500.rename(columns={
        'Symbol': 'symbol',
        'Security': 'security',
        'GICS Sector': 'sector',
        'GICS Sub-Industry': 'sub_industry',
    })
    
    # 只保留我们关心的几列
    sp500 = sp500[['symbol', 'security', 'sector', 'sub_industry']]
    
    # yfinance 不认 BRK.B 这种点号,要改成 BRK-B
    sp500['symbol'] = sp500['symbol'].str.replace('.', '-', regex=False)
    
    return sp500


if __name__ == "__main__":
    # 测试一下:跑这个文件就执行下面的代码
    df = get_sp500_tickers()
    print(f"成功拉到 {len(df)} 只股票")
    print("\n前 10 只:")
    print(df.head(10))
    print("\n按行业分布:")
    print(df['sector'].value_counts())
"""
批量拉取股票池的历史行情数据
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

 

def fetch_prices_batch(tickers, start_date, end_date, batch_size=50):
    """
    分批拉取多只股票的历史日线数据
    
    参数:
        tickers: 股票代码列表,例如 ['AAPL', 'MSFT', ...]
        start_date: 开始日期,字符串 'YYYY-MM-DD'
        end_date: 结束日期,字符串 'YYYY-MM-DD'
        batch_size: 每批拉多少只,默认 50
    
    返回:
        长格式的 DataFrame,列: date, ticker, open, high, low, close, volume
    """
    all_data = []  # 用来收集每一批的结果
    
    # 把列表切成若干批
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        
        print(f"正在拉第 {batch_num}/{total_batches} 批 ({len(batch)} 只股票)...")
        
        try:
            # yfinance 批量下载
            # auto_adjust=True: 用前复权价格(分红、拆股调整后)
            # progress=False: 关掉进度条,自己打印更清爽
            # group_by='ticker' 不用,默认的"按字段分组"反而方便我们后面 stack
            data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
            )
            
            if data.empty:
                print(f"  ⚠️ 这一批没拉到数据,跳过")
                continue
            
            # 把宽格式转成长格式
            # stack(level=1) 会把第二层列(股票代码)变成行索引
            data_long = data.stack(level=1, future_stack=True).reset_index()
            
            # 整理列名
            data_long.columns = [c.lower() for c in data_long.columns]
            data_long = data_long.rename(columns={'ticker': 'ticker'})
            
            all_data.append(data_long)
            print(f"  ✅ 成功,共 {len(data_long)} 行")
            
        except Exception as e:
            print(f"  ❌ 出错: {e}")
            continue
    
    # 把所有批次合起来
    if not all_data:
        print("一条数据都没拉到")
        return pd.DataFrame()
    
    result = pd.concat(all_data, ignore_index=True)
    
    # 去掉 close 是 NaN 的行(说明那只股票当天没数据,比如还没上市)
    result = result.dropna(subset=['close'])
    
    # 列顺序整理
    result = result[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
    
    return result


if __name__ == "__main__":
    # 测试一下:先拉 5 只股票看看
    test_tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN']
    
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"拉取范围: {start_date} 到 {end_date}")
    print(f"测试股票: {test_tickers}\n")
    
    df = fetch_prices_batch(test_tickers, start_date, end_date)
    
    print(f"\n总共拉到 {len(df)} 行")
    print("\n前 10 行:")
    print(df.head(10))
    print("\n每只股票的数据量:")
    print(df['ticker'].value_counts())
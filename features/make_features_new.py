import pandas as pd
import numpy as np

class MakeFeature:
    # ===========================================
    # 动量因子
    # ===========================================

    def mom_1d(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['ticker', 'date']).copy()

        df["mon_1d"] = df.groupby('ticker')["close"].pct_change(1)
        return df['mom_1d']
    
    def mom_5d(df: pd.DataFrame) -> pd.DataFrame:
            df = df.sort_values(['ticker', 'date']).copy()

            df["mon_5d"] = df.groupby('ticker')["close"].pct_change(5)
            return df['mom_5d']
    # ===========================================
    # 波动因子
    # ===========================================
    
    def std_5d(df:pd.DataFrame) -> pd.DataFrame:
         df = df.sort_values(['ticker', 'date']).copy()

         df["std_5d"] = (
              df.groupby('ticker')["mom_1d"]
              .rolling(5)
              .std()
              .reset_index(level=0, drop=True)
         )
         return df['std_5d']
    


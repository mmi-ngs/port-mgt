import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import perf_metrics as pm

os.chdir('S:\Trustee\Investments\INVESTMENTS\Portfolio Construction\port-mgt'
         '\Projects\AUD_20210621')
os.getcwd()

aud_data = pd.read_excel('aud original and rsi.xlsx', index_col=0,
                         parse_dates=True)

'''---------------Strategy One----------------
Long: Heuristic_Z > 0.5 & RSI cross over 30
Close: RSI cross over 70

Short: Heuristic_Z < -0.5 & RSI cross down 70
Close: RSI cross down 30
'''
df = aud_data.pct_change().iloc[1:, :4]
df.loc[:, 'rf'] = 0.10 / (100 * 365)
df.columns = ['Z_1', 'Z_1 & RSI', 'Z_0.5_avg', 'Z_0.5_avg & RSI', 'rf']

pm_aud = pm.df_perf_metrics(df, 'Z_1', 'rf', freq='daily', period=999,
                            cols_incl=4, trading_day=False)
pm_aud.to_clipboard()
pm.cum_ret_chart(df, freq='daily', trading_day=False)

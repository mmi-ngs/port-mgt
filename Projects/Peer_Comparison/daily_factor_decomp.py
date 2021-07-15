import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import factor_analysis as fa
import utils as ut
import perf_metrics as pm

os.chdir('.\Projects\Peer_Comparison')
os.getcwd()

# Read data
peer_unit_prices = pd.ExcelFile('Peer Daily Unit Prices 20210713.xlsx')

aus = pd.read_excel(peer_unit_prices, 'AusSuper', index_col=0, parse_dates=True)
uni = pd.read_excel(peer_unit_prices, 'UniSuper', index_col=0, parse_dates=True)
aware = pd.read_excel(peer_unit_prices, 'AwareSuper', index_col=0,
                      parse_dates=True)
sun = pd.read_excel(peer_unit_prices, 'SunSuper', index_col=0, parse_dates=True)
hostplus = pd.read_excel(peer_unit_prices, 'Hostplus', index_col=0,
                         parse_dates=True)
ngs = pd.read_excel(peer_unit_prices, 'NGS', index_col=0, parse_dates=True)

# Data Clean - Daily Return
aus_ret = aus / 100
uni_ret = (uni/100+1).pct_change().iloc[1:, :]
aware_ret = aware.pct_change().iloc[1:, :]
sun_ret = sun.pct_change().iloc[1:, :]
hostplus_ret = hostplus.replace('-', np.nan).pct_change().iloc[1:, :]
# ngs 2020-11-21 start daily unit prices
ngs_ret = ngs.pct_change().iloc[1:, :]

'''------------------Factor Decomposition---------------'''
# Specify start and end date and df_ret to decompose
df = sun_ret
df = ut.first_common_date(df.copy())[1]

start = pd.to_datetime('2020-11-21')
df = df.loc[df.index > start]

y = df.iloc[:, 0] - df.loc[:, 'Cash'].values
x = df.iloc[:, 1:-1]

# Check nan
print('nan: {}'.format(df.isna().sum().sum()))

# Initiate factor model
factor_model = fa.FactorModel(y, x, freq='daily', period=999)
params, tstats, r2_adj = factor_model.df_factor_decomp()

df_params, df_tstats, series_r2_adj = \
    fa.rolling_factor_decomp(y, x, 1/12, roll_freq='daily', trading_day=False)
df_params.to_clipboard()

df_params.iloc[:, [1, 2, 3, 4]].plot()

fa.feature_importance_plot(x, y, x.columns, 0.01, True)

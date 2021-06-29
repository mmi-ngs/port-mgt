import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from port_opt.port_opt import SAAOpt
import perf_metrics as pm
import utils as ut

os.chdir(r'.\Projects\SAA_20210406')
os.getcwd()

# Read SAA data
xls = pd.ExcelFile('ngs_saa_clean.xlsx')

data_map = pd.read_excel(xls, 'data_map', index_col=0)
data_bloomberg = pd.read_excel(xls, 'data_bloomberg', index_col=0,
                               parse_dates=True)
data_cambridge = pd.read_excel(xls, 'data_cambridge', index_col=0,
                               parse_dates=True)
cma = pd.read_excel(xls, 'cma', index_col=0)

assets = [*data_map.index]
N = len(assets)

# Convert data_bloomberg to monthly/quarterly return series
df_bbg_monthly = data_bloomberg.resample(
    'M', convention='end').last().pct_change()
df_bbg_quarterly = data_bloomberg.resample(
    'Q', convention='end').last().pct_change()

df_quarterly = pd.concat([df_bbg_quarterly, data_cambridge], axis=1)
df_quarterly_ordered = df_quarterly[assets]
df_quarterly_ordered_ex = df_quarterly_ordered.subtract(
    df_quarterly_ordered.loc[:, 'Cash'], axis=0)

# Calculate perf metrics for full sample
df_calc = df_quarterly_ordered

pm_full_sample = \
    pm.df_perf_metrics(df_calc, 'Int. Equities - DM (U/H)',
                       'Cash', freq='quarterly', period=999, cols_incl=25)
pm_full_sample.to_clipboard()
hist_ret_vol = pm_full_sample.iloc[:2, :].T
hist_ret_vol.to_clipboard()

hist_corr = df_calc.corr()
hist_corr.to_clipboard()

# Calculate rolling 10-yr ret & vol to get 95% HPDI: (2.5th, 97.5th)
roll_yr = 10

roll_ret = df_calc.rolling(4 * roll_yr, axis=0).apply(
    lambda x: np.prod(x + 1) ** (1 * 4 / x.size) - 1)
roll_vol = df_calc.rolling(4 * roll_yr, axis=0).apply(
    lambda x: np.std(x, ddof=1) * np.sqrt(4))
roll_corr = df_calc.rolling(4 * roll_yr, axis=0).corr()
roll_corr_list = []
for idx, df_i in roll_corr.groupby(level=0):
    print(idx)
    roll_corr_list.append(df_i)

roll_corr_3d = np.array(roll_corr_list)
np.quantile(roll_corr_3d, 0.975, axis=0)

# Ret
roll_ret_range = pd.Series(zip(round(roll_ret.quantile(0.025), 4),
                               round(roll_ret.quantile(0.975), 4)))
roll_ret_range.to_clipboard()

# Vol
roll_vol_range = pd.Series(zip(round(roll_vol.quantile(0.025), 4),
                               round(roll_vol.quantile(0.975), 4)))
roll_vol_range.to_clipboard()

"""
---Draw the boxplot to show the CMA range---
* Frontier - *
* CMA Mean - d
* Historical Mean - x
* Box1 - CMA Sources
* Box2 - Rolling 10yr Ret

Ret
Vol
"""

# Plot 1: Frontier CMA vs Full Sample Historical
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Frontier CMA vs Historical Full Sample')

sns.boxplot(data=roll_ret, ax=axs[0])
sns.swarmplot(data=cma[['Exp_Ret_Frontier']].T, marker='*', color='blue',
              size=10, ax=axs[0])
axs[0].set_ylabel('Return')
axs[0].set_xlabel('')

sns.boxplot(data=roll_vol, ax=axs[1])
sns.swarmplot(data=cma[['Exp_Vol_Frontier']].T, marker='*', color='blue',
              size=10, ax=axs[1])
axs[1].set_ylabel('Volatility')

plt.xticks(np.arange(N), assets, rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Plot 2: Frontier CMA vs Alternative CMAs
alt_cma_ret = cma.iloc[:, [4, 6, 8, 10, 12, 14, 16, 18]].T
alt_cma_vol = cma.iloc[:, [5, 7, 9, 11, 13, 15, 17, 19]].T

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Frontier CMA vs Alternative CMAs')

sns.boxplot(data=alt_cma_ret, ax=axs[0])
sns.swarmplot(data=cma[['Exp_Ret_Frontier']].T, marker='*', color='blue',
              size=10, ax=axs[0])
axs[0].set_ylabel('Return')
axs[0].set_xlabel('')

sns.boxplot(data=alt_cma_vol, ax=axs[1])
sns.swarmplot(data=cma[['Exp_Vol_Frontier']].T, marker='*', color='blue',
              size=10, ax=axs[1])
axs[1].set_ylabel('Volatility')

plt.xticks(np.arange(N), assets, rotation=45, ha='right')
plt.tight_layout()
plt.show()

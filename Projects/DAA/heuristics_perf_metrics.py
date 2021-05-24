import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import perf_metrics as pm

os.chdir(
    'S:\Trustee\Investments\INVESTMENTS\Portfolio Construction\Projects\DAA')
os.getcwd()

'''---------------DAA----------------'''
# Read data
heuristics_daa = pd.read_excel('heuristic quarterly signal (ex-momentum).xlsx',
                               sheet_name='clean', index_col=0)
heuristics_daa['rf'] = 0.10 / (100 * 365)
# Perf metrics

pm_df = pm.df_perf_metrics(heuristics_daa, 'SAA', 'rf', freq='quarterly',
                           period=1, cols_incl=2)
pm_df.to_clipboard()

'''---------------Individual Instruments----------------'''
heuristics_indices = pd.read_excel(
    'heuristic timeseries (MSCI World, MSCI EM, ASX).xlsx',
    sheet_name='clean', index_col=0)
df_ret = heuristics_indices.pct_change().iloc[1:, :]
df_ret['rf'] = 0.10 / (100 * 365)

world_str = ['world_bh', 'world_gold_cross', 'world_t_0', 'world_t_0.5',
             'world_t_1', 'world_t_1.5', 'rf']
em_str = ['em_bh', 'em_gold_cross', 'em_t_0', 'em_t_0.5', 'em_t_1',
          'em_t_1.5', 'rf']
asx_str = ['asx_bh', 'asx_gold_cross', 'asx_t_0', 'asx_t_0.5',
           'asx_t_1', 'asx_t_1.5', 'rf']

world = df_ret.loc[:, world_str]
em = df_ret.loc[:, em_str]
asx = df_ret.loc[:, asx_str]

pm_world = pm.df_perf_metrics(world, 'world_bh', 'rf', freq='daily',
                              trading_day=False,
                              period=999, cols_incl=6)
pm_world.to_clipboard()

pm_em = pm.df_perf_metrics(em, 'em_bh', 'rf', freq='daily',
                           trading_day=False,
                           period=999, cols_incl=6)
pm_em.to_clipboard()

pm_asx = pm.df_perf_metrics(asx, 'asx_bh', 'rf', freq='daily',
                            trading_day=False,
                            period=999, cols_incl=6)
pm_asx.to_clipboard()

'''----------------REITS and HY----------------------'''
reits_hy = pd.ExcelFile(
    'heuristic timeseries (ASX REITs, Bloomberg Barclays US High Yield).xlsx')

reits = pd.read_excel(reits_hy, 'REITs', index_col=0, parse_dates=True)
hy = pd.read_excel(reits_hy, 'High yield', index_col=0, parse_dates=True)

df_reits = reits.pct_change().iloc[1:, :]
df_hy = hy.pct_change().iloc[1:, :]

df_reits['rf'] = 0.10 / (100 * 365)
df_hy['rf'] = 0.10 / (100 * 365)

pm_reits = pm.df_perf_metrics(df_reits, 'Buyhold', 'rf', freq='daily',
                              trading_day=False,
                              period=999, cols_incl=6)
pm_reits.to_clipboard()

pm_hy = pm.df_perf_metrics(df_hy, 'Buyhold', 'rf', freq='daily',
                           trading_day=False,
                           period=999, cols_incl=6)
pm_hy.to_clipboard()

'''-------------New DAA (indexed growth)--------------'''
# Read data
heuristics_daa = pd.read_excel('heuristic quarterly signal (ex-momentum) '
                               '- Indexed Growth - HY REIT.xlsx',
                               sheet_name='clean', index_col=0)
daa_ret = heuristics_daa.pct_change().iloc[1:, :]
daa_ret = daa_ret[['DAA', 'SAA']]
daa_ret['rf'] = 0.10 / (100 * 4)

# Multi-period Perf Metrics
ls_period = [1, 3, 5, 10, 15, 999]
ls_df = []
for i in ls_period:
    pm_df_i = pm.df_perf_metrics(daa_ret, 'SAA', 'rf', freq='quarterly',
                                 period=i, cols_incl=2)
    ls_df.append(pm_df_i)
pm_df = pd.concat(ls_df, axis=1)
pm_df.to_clipboard()

# Plot the chart
pm.multi_period_perf_comparison_chart(
    pm_df, title_str='Multi-Period Performance Comparison - DAA vs SAA ('
                     'Simulated Indexed Growth)')


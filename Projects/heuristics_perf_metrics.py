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
heuristics_daa['rf'] = 0.01 / 100
# Perf metrics

pm_df = pm.df_perf_metrics(heuristics_daa, 'SAA', 'rf', freq='quarterly',
                           period=1, cols_incl=2)
pm_df.to_clipboard()

'''---------------Individual Instruments----------------'''
heuristics_indices = pd.read_excel(
    'heuristic timeseries (MSCI World, MSCI EM, ASX).xlsx',
    sheet_name='clean', index_col=0)
df_ret = heuristics_indices.pct_change().iloc[1:, :]
df_ret['rf'] = 0.01 / 100

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

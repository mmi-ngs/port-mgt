import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import perf_metrics as pm

os.chdir(
    r"S:\Trustee\Investments\INVESTMENTS\Portfolio "
    r"Construction\Projects\DAA\NDR")
os.getcwd()

'''-----------------------------------'''
# Read data
xls = pd.ExcelFile('NDR Data.xlsx')
gbam = pd.read_excel(xls, 'Global Balanced Account Model', index_col=0)
grem = pd.read_excel(xls, 'Global Regional Equity Model', index_col=0)
usm = pd.read_excel(xls, 'US Sector Model', index_col=0)

# Specify parameters
ls_period = [1, 3, 5, 10, 15, 20, 999]

# Global Balanced Account Model
gbam_ret = gbam.iloc[:, :2].pct_change().iloc[1:, :]
gbam_ret.columns = ['Global Balanced Account Model', 'Benchmark (55/35/10)']
gbam_ret['rf'] = 0.0010 / 12

pm_gbam = pm.multi_period_df_perf_metrics(
    gbam_ret, 'Benchmark (55/35/10)', 'rf', freq='monthly',
    ls_period=ls_period, cols_incl=2)
pm_gbam.to_clipboard()

pm.multi_period_perf_comparison_chart(
    pm_gbam, title_str='Multi-Period Performance Comparison - NDR Global '
                       'Balanced Account Model')

# Global Regional Equity Model
grem_ret = grem.iloc[:, :2].pct_change().iloc[1:, :]
grem_ret.columns = ['Global Regional Equity Model', 'Benchmark (ACWI Weight)']
grem_ret['rf'] = 0.0010 / 12

pm_grem = pm.multi_period_df_perf_metrics(
    grem_ret, 'Benchmark (ACWI Weight)', 'rf', freq='monthly',
    ls_period=ls_period, cols_incl=2)
pm_grem.to_clipboard()

pm.multi_period_perf_comparison_chart(
    pm_grem, title_str='Multi-Period Performance Comparison - NDR Global '
                       'Regional Equity Model')

# US Sector Model
usm_ret = usm.iloc[:, :2].pct_change().iloc[1:, :]
usm_ret.columns = ['US Sector Model', 'Benchmark (S&P Weight)']
usm_ret['rf'] = 0.0010 / 12

pm_usm = pm.multi_period_df_perf_metrics(
    usm_ret, 'Benchmark (S&P Weight)', 'rf', freq='monthly',
    ls_period=ls_period, cols_incl=2)
pm_usm.to_clipboard()

pm.multi_period_perf_comparison_chart(
    pm_usm, title_str='Multi-Period Performance Comparison - NDR US Sector '
                      'Model')

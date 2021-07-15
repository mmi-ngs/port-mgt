import numpy as np
import pandas as pd
import os

import perf_metrics as pm

os.chdir(r'.\Projects\SAA_20210406')
os.getcwd()

# Read SAA data
xls = pd.ExcelFile('ngs_saa_clean.xlsx')
map_saa = pd.read_excel(xls, 'map', index_col=0)
saa = pd.read_excel(xls, 'saa', index_col=0)
cons_saa = pd.read_excel(xls, 'cons', index_col=0)
option_map = pd.read_excel(xls, 'option_map', index_col=0)
tax_map = pd.read_excel(xls, 'tax_map', index_col=0)
cma = pd.read_excel(xls, 'cma', index_col=0)
exp_rho_frontier = pd.read_excel(xls, 'exp_corr', index_col=0)
hist_rho_fs = pd.read_excel(xls, 'hist_corr', index_col=0)

growth_wgt = map_saa.loc[:, 'Growth_PDS']
fx_wgt = map_saa.loc[:, 'FX_PDS']
illiq_wgt = map_saa.loc[:, 'Illiquidity_PDS']

N = map_saa.index.size

curr_saa = saa.iloc[:N, :]

# Specify CMA assumptions
exp_ret_vol_frontier = cma.loc[:, ['Exp_Ret_Frontier', 'Exp_Vol_Frontier']]
exp_ret_vol_frontier_20yr = cma.loc[:, ['Exp_Ret_Frontier_20yr',
                                        'Exp_Vol_Frontier_20yr']]
exp_ret_vol_mean = cma.loc[:, ['Exp_Ret_Mean', 'Exp_Vol_Mean']]
hist_ret_vol_fs = cma.loc[:, ['Hist_Ret_FS', 'Hist_Vol_FS']]
exp_ret_vol_ngs = cma.loc[:, ['Exp_Ret_NGS', 'Exp_Vol_NGS']]

exp_cov_frontier = \
    np.diag(exp_ret_vol_frontier.iloc[:, 1]) @ exp_rho_frontier @ \
    np.diag(exp_ret_vol_frontier.iloc[:, 1])
exp_cov_frontier_20yr = \
    np.diag(exp_ret_vol_frontier_20yr.iloc[:, 1]) @ exp_rho_frontier @ \
    np.diag(exp_ret_vol_frontier_20yr.iloc[:, 1])
exp_cov_mean = np.diag(exp_ret_vol_mean.iloc[:, 1]) @ exp_rho_frontier\
                  @ np.diag(exp_ret_vol_mean.iloc[:, 1])
hist_cov_fs = np.diag(hist_ret_vol_fs.iloc[:, 1]) @ hist_rho_fs\
                  @ np.diag(hist_ret_vol_fs.iloc[:, 1])
exp_cov_ngs = np.diag(exp_ret_vol_ngs.iloc[:, 1]) @ hist_rho_fs\
                  @ np.diag(exp_ret_vol_ngs.iloc[:, 1])

assets = [*map_saa.index]
options = [*cons_saa.columns[2:]]


'''---------Specify which set of CMA assumptions to use----------------'''
exp_ret_vol = exp_ret_vol_ngs
exp_corr = exp_rho_frontier.values

exp_ret = exp_ret_vol.iloc[:, 0].values
exp_vol = exp_ret_vol.iloc[:, 1].values
exp_cov = np.diag(exp_vol) @ exp_corr @ np.diag(exp_vol)

# After-tax cma assumptions
exp_ret_at, exp_vol_at = pm.after_tax_exp_ret_vol(tax_map, exp_ret_vol)
exp_cov_at = np.diag(exp_vol_at) @ exp_corr @ np.diag(exp_vol_at)

"""-----------Calculate Perf Metrics----------"""
# Specify saa to use
# Note: Use the same option names as the ones in the map (ngs_saa_clean)
saa_i = curr_saa

# Get the obj_list and obj_horizon_list for the options in the curr_saa
obj_list_curr_saa = []
obj_horizon_list_curr_saa = []
account_curr_saa = []

for i in saa_i.columns:
    obj_list_curr_saa.append(option_map.loc[i, 'Objective'])
    obj_horizon_list_curr_saa.append(option_map.loc[i, 'Objective_Horizon'])
    account_curr_saa.append(option_map.loc[i, 'Account'])

# Perf Metrics
saa_i_metrics = pm.df_option_perf_metrics(
    saa_i, exp_ret, exp_vol, exp_corr, obj_list_curr_saa,
    obj_horizon_list_curr_saa,
    growth_wgt, fx_wgt, illiq_wgt, after_tax=True, tax_map=tax_map,
    tax_account_list=account_curr_saa)
saa_i_metrics.to_clipboard()

# Get curr_saa and opt_saa out with the right format
saa_extend_i = pd.concat([map_saa.iloc[:, :2], saa_i], axis=1)
saa_inv_i = saa_extend_i.groupby('Sector_InvTeam').sum()
saa_inv_i.to_clipboard()

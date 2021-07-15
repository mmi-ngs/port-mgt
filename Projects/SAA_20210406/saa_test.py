import numpy as np
import pandas as pd
import os
import seaborn as sns

from port_opt.port_opt import SAAOpt
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
prop_saa = saa.iloc[N+1:2*N+1, :]

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

# Set optimization parameters
sub_sectors = [*map_saa.index]
options = [*cons_saa.columns[2:]]
option_i = 'Diversified'

bounds = (0, 1)

'''---------Specify which set of CMA assumptions to use----------------'''
exp_ret_vol = exp_ret_vol_ngs
exp_corr = hist_rho_fs.values

exp_ret = exp_ret_vol.iloc[:, 0].values
exp_vol = exp_ret_vol.iloc[:, 1].values
exp_cov = np.diag(exp_vol) @ exp_corr @ np.diag(exp_vol)

exp_ret_at, exp_vol_at = pm.after_tax_exp_ret_vol(tax_map, exp_ret_vol)
exp_cov_at = np.diag(exp_vol_at) @ exp_corr @ np.diag(exp_vol_at)

# Test all options with one objective function
obj_list = ['max_sharpe']
args_obj = {'max_sharpe': (exp_ret, exp_cov, 0.0025, True)}

saa_opt = SAAOpt(map_saa, cons_saa, options, option_map, tax_map,
                 exp_ret, exp_vol, exp_corr, n_assets=N, tickers=sub_sectors,
                 bounds=bounds, te_bmk={'TE_PDS': curr_saa})
res_multi_options = saa_opt.options_opt_run(obj_list, args_obj,
                                            after_tax_opt=True,
                                            after_tax_output=True,
                                            print_res=False,
                                            check_constraints=True,
                                            opt_perf_metrics=True)
export_multi_options = saa_opt.export_saa_output(
    save_excel=True, filename='saa_output_max_sr_ngs_at.xlsx')

opt_saa = export_multi_options['Asset_Wgt']
opt_saa.columns = options

# Efficient Frontier
df_ef, df_random = saa_opt.visualize_efficient_frontier(
    (exp_ret_at, exp_cov_at), n_samples=5000)

# Split the efficient frontier line to 4 quartiles
dict_split = {}
for str_x in ['df_ef', 'df_random']:
    df_split = eval(str_x)
    dict_split[str_x] = df_split
    df_split.loc[:, 'Vol_Quartile'] = pd.cut(df_split.loc[:, 'Vol'], 4,
                                             labels=[1, 2, 3, 4])
    x = df_split.groupby('Vol_Quartile').agg({'min', 'max'}).T

    df_split_range = pd.DataFrame(np.nan, index=df_split.columns[:-1],
                                  columns=[1, 2, 3, 4])
    for a in df_split_range.index:
        for b in df_split_range.columns:
            if a == 'Sharpe':
                min_ab = round(x.loc[(a, 'min'), b], 2)
                max_ab = round(x.loc[(a, 'max'), b], 2)
                df_split_range.loc[a, b] = '[{}, {}]'.format(min_ab, max_ab)
            else:
                min_ab = round(x.loc[(a, 'min'), b] * 100, 2)
                max_ab = round(x.loc[(a, 'max'), b] * 100, 2)
                df_split_range.loc[a, b] = '[{}%, {}%]'.format(min_ab, max_ab)
    dict_split[str_x + '_split'] = df_split
    dict_split[str_x + '_split_range'] = df_split_range

# Export to excel
with pd.ExcelWriter('ef_simulation_output_ngs_at.xlsx') as writer:
    for i in dict_split:
        dict_split[i].to_excel(writer, sheet_name=i)

"""-----------Calculate Perf Metrics (Frontier CMA vs Mean_CMA)----------"""
# Specify saa to use
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

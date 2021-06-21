import numpy as np
import pandas as pd
import os
import seaborn as sns

from port_opt.port_opt import SAAOpt
import port_opt.objective_functions as obj
import perf_metrics as pm

os.chdir(r'.\Projects\SAA_20210406')
os.getcwd()

# Read SAA data
xls = pd.ExcelFile('ngs_saa_clean.xlsx')
map_saa = pd.read_excel(xls, 'map', index_col=0)
saa = pd.read_excel(xls, 'saa', index_col=0)
cons_saa = pd.read_excel(xls, 'cons', index_col=0)
option_map = pd.read_excel(xls, 'option_map', index_col=0)
exp_ret_vol = pd.read_excel(xls, 'exp_ret_vol', index_col=0)
exp_rho = pd.read_excel(xls, 'exp_corr', index_col=0)

exp_cov = np.diag(exp_ret_vol.iloc[:, 1]) @ exp_rho @ \
    np.diag(exp_ret_vol.iloc[:, 1])

# Set optimization parameters
N = map_saa.index.size
sub_sectors = [*map_saa.index]
options = [*cons_saa.columns[2:]]
option_i = 'Diversified'

bounds = (0, 1)

exp_ret = exp_ret_vol.iloc[:, 0].values
exp_cov = exp_cov.values

# Test all options and three objective functions
# obj_list = ['max_sharpe', 'min_var', 'max_quad_utility']
# args_obj = {'max_sharpe': (exp_ret, exp_cov, 0.0025, True),
#             'min_var': (exp_cov, ),
#             'max_quad_utility': (exp_ret, exp_cov, 3)}

# Test all options with one objective function
obj_list = ['max_sharpe']
args_obj = {'max_sharpe': (exp_ret, exp_cov, 0.0025, True)}

saa_opt = SAAOpt(map_saa, cons_saa, options, option_map, n_assets=N,
                 tickers=sub_sectors, bounds=bounds)
res_multi_options = saa_opt.options_opt_run(obj_list, args_obj, bounds=bounds,
                                            print_res=False,
                                            check_constraints=True,
                                            opt_perf_metrics=True,
                                            exp_ret=exp_ret, exp_cov=exp_cov)

export_multi_options = saa_opt.export_saa_output(
    save_excel=True, filename='saa_output_max_sr.xlsx')

df_ef = saa_opt.visualize_efficient_frontier((exp_ret, exp_cov),
                                             n_samples=3000)

# Split the efficient frontier line to 4 quartiles
df_ef.loc[:, 'Vol_Quartile'] = pd.cut(df_ef.loc[:, 'Vol'], 4,
                                      labels=[1, 2, 3, 4])
x = df_ef.groupby('Vol_Quartile').agg({'min', 'max'}).T

df_ef_range = pd.DataFrame(np.nan, index=df_ef.columns[:-1],
                           columns=[1, 2, 3, 4])
for a in df_ef_range.index:
    for b in df_ef_range.columns:
        min_ab = round(x.loc[(a, 'min'), b] * 100, 2)
        max_ab = round(x.loc[(a, 'max'), b] * 100, 2)
        df_ef_range.loc[a, b] = '[{}%, {}%]'.format(min_ab, max_ab)
df_ef_range.to_clipboard()
df_ef.to_clipboard()
# g = sns.catplot(kind='box', data=x.iloc[:, 2:],
#                 col_wrap=2)
# (g.set_xticklabels(sub_sectors, rotation=45, fontsize=10)
#  .tight_layout())


# Test the OptionPerfMetrics
x = export_multi_options['Asset_Wgt']
growth_wgt = map_saa[['Growth_PDS']]
fx_wgt = map_saa[['FX_PDS']]
illiq_wgt = map_saa[['Illiquidity_PDS']]
x_metrics = pm.OptionPerfMetrics(x.iloc[:, 0], exp_ret_vol.iloc[:, :2],
                                 exp_cov, 0.03, 5, growth_wgt, fx_wgt,
                                 illiq_wgt).standard_metrics()

"""
# Test one option and one objective function
saa_opt = SAAOpt(map_saa, cons_saa, option_i, n_assets=N,
                 tickers=sub_sectors, bounds=bounds)
saa_opt.map_saa_constraints(option_i)

args_sharpe = (exp_ret, exp_cov, 0.0025, True)
res_sharpe = saa_opt.opt_run('max_sharpe', args_sharpe, bounds=bounds,
                             print_res=True)
wgt_sharpe = saa_opt.export_wgt()

# Test one option and three objective functions
saa_opt = SAAOpt(map_saa, cons_saa, option_i, n_assets=N,
                 tickers=sub_sectors, bounds=bounds)
saa_opt.map_saa_constraints(option_i)


res_multi_obj = saa_opt.opt_run(obj_list, args_obj, bounds=bounds,
                                print_res=True)
wgt_multi_obj = saa_opt.export_wgt()
"""



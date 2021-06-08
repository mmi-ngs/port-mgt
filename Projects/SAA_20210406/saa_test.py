import numpy as np
import pandas as pd
import os

from port_opt.port_opt import SAAOpt
import port_opt.objective_functions as obj

os.chdir(r'.\Projects\SAA_20210406')
os.getcwd()

# Read SAA data
xls = pd.ExcelFile('ngs_saa_clean.xlsx')
map_saa = pd.read_excel(xls, 'map', index_col=0)
saa = pd.read_excel(xls, 'saa', index_col=0)
cons_saa = pd.read_excel(xls, 'cons', index_col=0)
exp_ret_vol = pd.read_excel(xls, 'exp_ret_vol', index_col=0)
exp_rho = pd.read_excel(xls, 'exp_cov', index_col=0)

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
obj_list = ['max_sharpe', 'min_var', 'max_quad_utility']
args_obj = {'max_sharpe': (exp_ret, exp_cov, 0.0025, True),
            'min_var': (exp_cov, ),
            'max_quad_utility': (exp_ret, exp_cov, 3)}

saa_opt = SAAOpt(map_saa, cons_saa, options, n_assets=N,
                 tickers=sub_sectors, bounds=bounds)
res_multi_options = saa_opt.options_opt_run(obj_list, args_obj, bounds=bounds,
                                            print_res=False,
                                            check_constraints=True)

export_multi_options = saa_opt.export_saa_output(
    save_excel=False, filename='saa_output_tight.xlsx')

x = saa_opt.visualize_efficient_frontier((exp_ret, exp_cov))


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

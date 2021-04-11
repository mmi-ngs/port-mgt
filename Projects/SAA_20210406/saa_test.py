import numpy as np
import pandas as pd
import os

from port_opt import port_opt


os.chdir(r'.\Projects\SAA_20210406')
os.getcwd()

# Read SAA data
xls = pd.ExcelFile('ngs_saa_clean.xlsx')
map = pd.read_excel(xls, 'map', index_col=0)
saa = pd.read_excel(xls, 'saa', index_col=0)
cons = pd.read_excel(xls, 'cons', index_col=0)
exp_ret_vol = pd.read_excel(xls, 'exp_ret_vol', index_col=0)
exp_cov = pd.read_excel(xls, 'exp_cov', index_col=0)

# Set optimization
N = map.index.size
saa_opt = port_opt.PortOpt(N, tickers=map.index, bounds=(0, 1))
saa_opt.add_constraint('eq', 'lambda x: x.sum()-1')

args_sharpe = (exp_ret_vol.iloc[:, 0].values, exp_cov.values, 0.0025, True)
res_sharpe = saa_opt.opt_run('max_sharpe', args_sharpe, bounds=(0, 1),
                             print_res=True)
res_wgt = saa_opt.export_wgt()

args_quadutil = (exp_ret_vol.iloc[:, 0].values, exp_cov.values, 3)
res_quadutil = saa_opt.opt_run('max_quad_utility', args_quadutil,
                               bounds=(0, 1), print_res=True)
wgt_quadutil = saa_opt.export_wgt()

import port_opt.objective_functions as obj
sr = obj.sharpe_ratio(saa_opt.wgt,
                      exp_ret_vol.iloc[:, 0].values,
                      exp_cov.values, negative=False)

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

args_sharpe = (exp_ret_vol.iloc[:, 0].values, exp_cov.values, 0.0025, True)


import port_opt.objective_functions as obj
x = obj.call_obj_functions('max_sharpe')
import types
from collections import Callable
isinstance(x, types.FunctionType)
isinstance(x, Callable)
y = x(np.array([1/30] * 30), exp_ret_vol.iloc[:, 0].values, exp_cov.values)
yy = obj.sharpe_ratio(np.array([1/30] * 30), exp_ret_vol.iloc[:, 0].values,
                      exp_cov.values)

res_sharpe = saa_opt.opt_run('max_sharpe', args_sharpe, bounds=(0.01, 1),
                             print_res=True)

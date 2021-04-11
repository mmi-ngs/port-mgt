import numpy as np
import pandas as pd
import os

from port_opt import port_opt


os.chdir(r'.\Projects\SAA_20210406')
os.getcwd()

# Read SAA data
xls = pd.ExcelFile('ngs_saa_clean.xlsx')
map_saa = pd.read_excel(xls, 'map', index_col=0)
saa = pd.read_excel(xls, 'saa', index_col=0)
cons = pd.read_excel(xls, 'cons', index_col=0)
exp_ret_vol = pd.read_excel(xls, 'exp_ret_vol', index_col=0)
exp_rho = pd.read_excel(xls, 'exp_cov', index_col=0)

exp_cov = np.diag(exp_ret_vol.iloc[:, 1]) @ exp_rho @ \
    np.diag(exp_ret_vol.iloc[:, 1])

# Set optimization parameters
N = map_saa.index.size
sectors = [*map_saa.index]
options = [*cons.columns[2:]]
bounds = (0, 0.25)

saa_opt = port_opt.PortOpt(N, tickers=sectors, bounds=bounds)

# Add constraints (require input cons, map_saa, option_i)
option_i = 'Diversified'

for i in range(len(cons.index)):
    con_i = cons.index[i]
    con_type_i = cons['Con_Type'].iloc[i]
    type_i = cons['Type'].iloc[i]
    if (con_type_i == 'Sum') & (type_i == 'Sum'):
        sum_i = cons.loc[con_i, option_i]
        saa_opt.add_constraint(
            'eq', 'lambda x: x.sum() - {}'.format(sum_i))
    elif con_i in map_saa.columns[2:]:
        if type_i == 'Min':
            x_min_i = cons[option_i].iloc[i]
            saa_opt.add_constraint(
                'ineq', "lambda x: x.T @ map_saa.loc[:, con_type_i] - {}".format(
                    x_min_i))
        elif type_i == 'Max':
            x_max_i = cons[option_i].iloc[i]
            saa_opt.add_constraint(
                'ineq', "lambda x: {} - x.T @ map_saa.loc[:, con_type_i]".format(
                    x_max_i))
        else:
            raise Warning(
                'Constraint {} is not added for wrong format.'.format(con_i))
    elif con_type_i == 'Sector_PDS':
        # Get the location of subsectors that belong to the sector.
        # E.g., for Australian Equities, [0, 1] in the map_saa.Sector_PDS vector.
        loc_sector_pds_i = np.where(map_saa.Sector_PDS == con_i)

        # Check min or max
        if type_i == 'Min':
            x_min_i = cons[option_i].iloc[i]
            saa_opt.add_constraint(
                'ineq', "lambda x: x[{}].sum() - {}".format(
                    loc_sector_pds_i, x_min_i))
        elif type_i == 'Max':
            x_max_i = cons[option_i].iloc[i]
            saa_opt.add_constraint(
                'ineq', "lambda x: {} - x[{}].sum()".format(
                    x_max_i, loc_sector_pds_i))
        else:
            raise Warning(
                'Constraint {} is not added for wrong format.'.format(con_i))

# saa_opt.add_constraint('eq', 'lambda x: x.sum()-1')

args_sharpe = (exp_ret_vol.iloc[:, 0].values, exp_cov.values, 0.0025, True)
res_sharpe = saa_opt.opt_run('max_sharpe', args_sharpe, bounds=(0, 1),
                             print_res=True)
wgt_sharpe = saa_opt.export_wgt()

# Max Util
args_quadutil = (exp_ret_vol.iloc[:, 0].values, exp_cov.values, 3)
res_quadutil = saa_opt.opt_run('max_quad_utility', args_quadutil,
                               bounds=(0, 1), print_res=True)
wgt_quadutil = saa_opt.export_wgt()

# Min Var
args_minvar = (exp_cov.values, )
res_minvar = saa_opt.opt_run('min_var', args_minvar, bounds=(0, 1),
                             print_res=True)
wgt_minvar = saa_opt.export_wgt()

import port_opt.objective_functions as obj
sr = obj.sharpe_ratio(saa_opt.wgt,
                      exp_ret_vol.iloc[:, 0].values,
                      exp_cov.values, negative=False)
obj.port_var(saa_opt.wgt, exp_cov.values)
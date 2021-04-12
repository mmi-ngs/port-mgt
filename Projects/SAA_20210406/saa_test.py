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
                                            print_res=False)
wgt_multi_options = saa_opt.export_wgt(save_csv=True,
                                       filename='saa_test_run_result.csv')


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


"""-------Storage for map_saa_constraint()-----
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
        # E.g., for Australian Equities,
        # [0, 1] in the map_saa.Sector_PDS vector.
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
                
# Test - add constraint one by one
saa_opt.add_constraint('eq', "lambda x: x.sum() - 1")

con_i = 'Australian Equities'
x_min_i = 0.15
x_max_i = 0.4

loc_sector_pds_i = np.where(map_saa.Sector_PDS == con_i)[0]
saa_opt.add_constraint('ineq', "lambda x: x[{}].sum() - {}".format(
        list(loc_sector_pds_i), x_min_i))
saa_opt.add_constraint('ineq', "lambda x: {} - x[{}].sum()".format(
        x_max_i, list(loc_sector_pds_i)))

saa_opt.add_constraint('ineq', "lambda x: x.T @ np.array({}) - {}".format(
                            list(map_saa.loc[:, con_i]), x_min_i))
saa_opt.add_constraint('ineq', "lambda x: {} - x.T @ np.array({})".format(
                            x_max_i, list(map_saa.loc[:, con_i])))
"""
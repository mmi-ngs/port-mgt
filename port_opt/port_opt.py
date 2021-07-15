"""
The port_opt submodule houses the PortOpt parent class, which includes several
embedded objective functions with flexibility in imposing constraints and
bounds.

The parent class PortOpt can be inherited to several use cases, e.g., SAA, DAA,
within-sector optimization. The child classes are also maintained here.

The optimization algorithm is based on scipy.optimize.shgo/slsqp. shgo is the
go-to approach for relatively low dimensional settings while slsqp is used
for high dimensional settings.
(ref: https://www.microprediction.com/blog/optimize)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.optimize import shgo, minimize, dual_annealing

import port_opt.objective_functions as obj
import port_opt.simulation as sim
import perf_metrics as pm
import utils as ut


class PortOptBase:
    """
    PortOptBase class to incorporate basic variables and functions for an
    instance of base optimization.
    ------------------
    ### Instance variables ###
    n_assets: [int] number of assets
    tickers: [list(str)] list of strings of asset names
    long_short: [boolean] market-neutral if True, weight sum to 0

    wgt: [np.array] np.array of asset weights
    wgt_dict: [dict] dictionary of all optimized or input clean weights

    ### Public Methods ###
    add_clean_wgt(): rounds up to 2 decimals, add as a series to the wgt_dict
    export_wgt(): save weights to csv file
    """

    def __init__(self, n_assets, tickers=None, long_short=False):
        """
        :param n_assets: [int] number of assets
        :param tickers: [list(str)] list of strings of asset names
        :param long_short: [boolean] market-neutral if True, weight sum to 0
        """
        self.n_assets = n_assets
        if tickers is None:
            self.tickers = list(range(n_assets))
        elif len(tickers) != n_assets:
            raise AttributeError('tickers length is not the same as n_assets.')
        else:
            self.tickers = tickers

        if long_short:
            self._wgt_sum = 0
        else:
            self._wgt_sum = 1

        # Outputs
        self.wgt = None
        self.wgt_dict = {}

    def add_clean_wgt(self, input_wgt=None, wgt_name=None, cutoff=1e-4,
                      decimal=4):
        """
        Helper function to check the weights sum to specified value (1/0),
        clean the raw weights, setting any weights whose absolute values are
        below the cutoff to zero, and round the rest. Then check the sum
        again before making it a pd.Series output.

        :param input_wgt: [list/array] the input weights to be made the
                          default weight and added to the weight dictionary
        :param wgt_name: [str] the name of the weight, usually defined as the
                         objective function name, e.g., 'max_sharpe'.
        :param cutoff: [float] the lower bound, default to 1e-4
        :param decimal: [int] number of decimal places to round weights,
                      default 2
                      * Check: positive integer (int & >1)
        :return: wgt [pd.Series] from ._make_wgt_series() method
        """
        # Check if there is new input_wgt
        if input_wgt is not None:
            self._set_wgt(input_wgt)

        # Check sum
        self._wgt_check()

        # Clean
        if self.wgt is None:
            raise AttributeError('Weights not yet computed')
        clean_wgt = self.wgt.copy()
        clean_wgt[np.abs(clean_wgt) < cutoff] = 0
        if not isinstance(decimal, int) or decimal < 1:
            raise ValueError('Param decimal must be a positive integer')
        else:
            clean_wgt = np.round(clean_wgt, decimal)

        # If clean_wgt doesn't sum up to specified sum, change the max wgt by
        # its difference and check again.
        if clean_wgt.sum() != self._wgt_sum:
            _wgt_ex = clean_wgt.sum() - self._wgt_sum
            # Allow 5 bps error range
            if abs(_wgt_ex) <= 0.0005:
                # Use list.index(max(list)) to avoid multiple max from array
                clean_wgt_list = [*clean_wgt]
                clean_wgt[clean_wgt_list.index(max(clean_wgt_list))] -= _wgt_ex
            else:
                warnings.warn('New weight adds up to {}'.format(
                    clean_wgt.sum()))

        self._wgt_check(clean_wgt)

        # Add clean_wgt to self.wgt_dict
        if wgt_name is None:
            wgt_name = 'Wgt_{}'.format(len(self.wgt_dict))

        self.wgt_dict[wgt_name] = \
            self._make_wgt_series(clean_wgt, wgt_name=wgt_name)

    def export_wgt(self, save_csv=False, filename="wgt.csv"):
        """
        Utility function to export weights to a DataFrame and save weights to a
        csv file.

        :param save_csv: [boolean] save to a csv file if True.
        :param filename: [str] name of file. Should be csv.
        """
        if self.wgt_dict is None or self.wgt_dict == {}:
            raise Exception('Empty dictionary of weight outputs.')
        elif any(isinstance(i, dict) for i in self.wgt_dict.values()):
            # Nested dictionary
            df_output = ut.nested_dict_to_df(self.wgt_dict)
        else:
            df_output = pd.DataFrame.from_dict(self.wgt_dict)

        if save_csv:
            df_output.to_csv(filename)

        return df_output

    def _set_wgt(self, wgt):
        """
        Utility function to set the self.wgt to the input wgt.

        :param wgt: [list/array] the input weight
        """
        if not (isinstance(wgt, list) or (isinstance(wgt, np.ndarray))):
            raise ValueError('Input wgt is not a list or np.ndarray.')
        else:
            if len(wgt) != self.n_assets:
                raise AttributeError('Input wgt length is not matching '
                                     'n_assets.')
            else:
                self._wgt_check(wgt)
                self.wgt = wgt

    def _wgt_check(self, wgt=None):
        """
        Utility function to check whether final weights sum up to 1.
        * If leveraged or long-short, allows change of parameter to sum up to
          0 or else.
        """
        if wgt is not None:
            if round(wgt.sum(), 2) != self._wgt_sum:
                warnings.warn(
                    'New weights do not sum up to {} (sum={})'.format(
                        self._wgt_sum, wgt.sum()))
            else:
                pass
        else:
            if self.wgt is not None:
                if round(self.wgt.sum(), 2) != self._wgt_sum:
                    warnings.warn(
                        'Weights do not sum up to {} (sum={})'.format(
                            self._wgt_sum, self.wgt.sum()))
            else:
                raise AttributeError('Weights not yet computed')

    def _make_wgt_series(self, wgt=None, wgt_name=None):
        """
        Utility function to make output wgt pd.Series from wgt np.array.
        Use self.tickers and self.wgt if no wgt argument passed.
        """
        if wgt is None:
            wgt = self.wgt

        return pd.Series(wgt, index=self.tickers, name=wgt_name)


class PortOpt(PortOptBase):
    """
    The PortOpt class inherits the parent PortOptBase class and uses the
    scipy.optimize.shgo global minimum optimizer to generate optimized weights.
    Additional features include constraints and built-in objective functions.

    ### Instance variables ###
    n_assets: [int] number of assets
    tickers: [list(str)] list of strings of asset names
    long_short: [boolean] market-neutral if True, weight sum to 0
    bounds: [tuple] or [list(tuple)]: min and max weight of each asset or
            single pair if all identical, default (0, 1), need to be changed
            to (-1, 1) if allow shorting.

    wgt: [np.array] np.array of asset weights
    wgt_dict: [dict] dictionary of all optimized or input clean weights

    ### Public Methods ###
    add_constraint(): add a constraint for optimization
    check_constraint(): check if the constraints are met
    add_clean_wgt(): rounds up to 2 decimals, add as a series to the wgt_dict
    export_wgt(): save weights to csv file

    """

    def __init__(self, n_assets, tickers=None, long_short=False, bounds=(0, 1)):
        """

        :param n_assets: [int] number of assets
        :param tickers: [list(str)] list of strings of asset names
        :param long_short: [boolean] market-neutral if True, weight sum to 0
        :param bounds: [np.array/tuple] np.array of asset weight bounds or a
                        single tuple that fits for all assets
        """
        super().__init__(n_assets, tickers, long_short)

        # Optimization variables
        if isinstance(bounds, list) and len(bounds) == self.n_assets:
            self.bounds = bounds
        elif isinstance(bounds, tuple):
            self.bounds = [bounds] * self.n_assets
        else:
            self.bounds = [(None, None)] * self.n_assets
            warnings.warn('Bounds are not defined.')

        # Protected variables
        self._wgt = None
        self._wgt0 = np.array([1/self.n_assets] * self.n_assets)
        self._objective = None
        self._objective_list = []
        self._args = None
        self._args_dict = {}
        self._constraints = []
        self._res_opt = {}

    def add_constraint(self, type_str, fun_str):
        """
        Add a constraint to scipy.shgo/slsqp optimizer. The format of
        constraint is:
        ({'type': 'eq', 'fun': lambda x: x.sum() - 1},
         {'type': 'ineq', 'fun': lambda x: x - wgt_min},
         {'type': 'ineq', 'fun': lambda x: wgt_max - x})

        :param type_str: [str] 'eq' OR 'ineq';
                     determine whether constraint is an equality constraint or
                     inequality constraint.
        :param fun_str: [str] a well-defined python function or lambda
                        function written in string, will be decoded using
                        eval() to
        """
        if not callable(eval(fun_str)):
            raise TypeError(
                "New constraint must be provided as a function")
        self._constraints.append({'type': type_str, 'fun': eval(fun_str)})

    def opt_run(self, obj_str, args, bounds=None, print_res=False):
        """
        Execute optimization based on defined objective functions and added
        constraints.

        :param obj_str: [str] or [list(str)]
                Name[s] of pre-defined objective functions
                e.g., obj_str = 'max_sharpe' / ['max_sharpe', 'min_cvar']
                *If a list is passed, generate outcomes from different objective
                functions.
        :param args: [tuple] or [dict(tuple)]
                Arguments passed on for scipy.optimize.shgo/slqlp optimizers.
                e.g., For 'max_sharpe', args = (ret, cov, rf=0.0025,
                negative=True).
                *If a dictionary of tuples is passed, the keys should
                match the objective functions respectively.
                e.g., args = {
                'max_sharpe': (ret, cov, rf=0,0025, negative=True),
                'min_var': (cov)}
        :param bounds: [list/tuple] redefine the bounds for the optimization run
                *If None, use instance variable self.bounds.
        :param print_res: [boolean] print results if True

        :return self._res_opt
        """
        # Objective functions
        if isinstance(obj_str, list):
            if not isinstance(args, dict):
                raise TypeError('When obj_str is a list, args should be a '
                                'dictionary matching the objective functions.')
            self._objective_list = obj_str
        elif isinstance(obj_str, str):
            self._objective_list.append(obj_str)
        else:
            raise TypeError('obj_str is not a list or string.')

        # Args
        if isinstance(args, tuple):
            self._args_dict[obj_str] = args
        elif isinstance(args, dict):
            if not [*args.keys()] == self._objective_list:
                raise KeyError('The keys of dictionary args should be the '
                               'same as the obj_str list.')
            else:
                self._args_dict = args

        # Bounds
        if bounds is not None:
            if isinstance(bounds, list) and len(bounds) == self.n_assets:
                self.bounds = bounds
            elif isinstance(bounds, tuple):
                self.bounds = [bounds] * self.n_assets
            else:
                raise TypeError('Bounds are not list or tuple, or the length '
                                'of list does not match the n_assets.')

        # Loop through objective functions for optimization
        for i in self._objective_list:
            self._objective = obj.call_obj_functions(i)
            self._args = self._args_dict[i]

            # Run optimization
            res_i = self.opt_selector()

            # Set weight and standard weight check and clean
            self.wgt = res_i.x
            self.add_clean_wgt(wgt_name=i)
            self._res_opt[i] = res_i

        # Print optimized weights
        if print_res:
            for a, b in self.wgt_dict.items():
                print(b)

        return self._res_opt

    def opt_perf(self, initial_wgt=None, plot_ef=False):
        """
        Conduct performance comparison between initial weight and optimized
        weight (or use set_weight() method to compare an initial weight and a
        user input weight).

        :param initial_wgt: [list/array] list or array of the current weights
                            for performance comparison; no comparison if not
                            provide, pure optimization performance
        :param plot_ef: [boolean] plot efficient frontier if True
        :return: df_perf: [pd.DataFrame] DataFrame of optimized weights and
                          optimized performance
        """
        if initial_wgt is not None:
            if len(initial_wgt) != self.n_assets:
                raise ValueError('Length of list not equal to number of assets')

        return

    def opt_selector(self, set_opt=None):
        """
        Helper function to select the correct optimization function based on
        the dimension of the problem.
        :param set_opt: [str] name of the optimization method.
                        e.g. 'shgo', 'SLQLP'.

        If set_opt is None:
            Automatically set it based on the condition below:
            If self.n_assets <= 10, use shgo. Else: use SLSQP.
        Else:
            use set_opt. (UD)
        :return:
        """
        opt_avail = ['shgo', 'Powell', 'SLSQP']
        if set_opt is None:
            if self.n_assets <= 10:
                res = shgo(self._objective,
                           bounds=self.bounds,
                           args=self._args,
                           constraints=self._constraints,
                           options={'maxiter': 10000})
            else:
                res = minimize(self._objective,
                               x0=self._wgt0,
                               bounds=self.bounds, args=self._args,
                               method='SLSQP',
                               constraints=self._constraints,
                               options={'maxiter': 10000, 'disp': True})
                # res = dual_annealing(self._objective,
                #                      x0=self._wgt0,
                #                      bounds=self.bounds, args=self._args)
        else:
            if set_opt not in opt_avail:
                raise ValueError('Undefined optimization method.')
            else:
                res = None
                print('Manually setting optimizer is still under development.')

        return res

    def clear_constraints(self):
        """
        Clear the self._constraints list.
        """
        self._constraints = []


class SAAOpt(PortOpt):
    """
    The SAAOpt class is a child class from PortOpt class. It inherits all the
    variables and functions from PortOpt but include some specific features
    that are unique to SAA optimization, e.g. map_constraints().

    ### Unique Instance Variables ###
    map_saa: [pd.DataFrame] the table that include all the mapping information.
    con_saa: [pd.DataFrame] the table that include all the constraints.
    options: [str] / [list(str)] options for optimization

    # From map_saa and con_saa we retrieve the two variables below:
    sectors: [list] sectors used for output, one level higher than tickers
                    (sub-sector). Default: map_saa['Sector_PDS']

    ### Unique public methods ###:
    map_saa_constraints(): Automatically map the SAA constraints to the
                            optimizer with two key prefixed tables.
    check_saa_constraints(): Check and export whether each constraint is met
                             and export the outcome.
    saa_perf_metrics(): Compute the SAA-related performance metrics.
    export_saa_output(): save all useful dataframes to a single excel file.

    ### Instance variables ###
    n_assets: [int] number of assets
    tickers: [list(str)] list of strings of asset names
    long_short: [boolean] market-neutral if True, weight sum to 0
    bounds: [tuple] or [list(tuple)]: min and max weight of each asset or
            single pair if all identical, default (0, 1), need to be changed
            to (-1, 1) if allow shorting.

    wgt: [np.array] np.array of asset weights
    wgt_dict: [dict] dictionary of all optimized or input clean weights
    df_wgt: [pd.DataFrame] DataFrame of wgt_dict
    df_constraint_check: [pd.DataFrame] DataFrame of saa constraints check

    ### Public Methods ###
    add_constraint(): add a constraint for optimization
    add_clean_wgt(): rounds up to 2 decimals, add as a series to the wgt_dict
    export_wgt(): save weights to csv file
    """

    def __init__(self, map_saa, cons_saa, options, option_map, tax_map,
                 exp_ret, exp_vol, exp_corr, n_assets, tickers=None,
                 long_short=False, bounds=(0, 1), te_bmk=None, **kwargs):
        """
        :param map_saa: [pd.DataFrame]
               TABLE RESTRICTIONS:
               1. index == tickers/assets/sub-sectors
               2. first two columns are ['Sector_PDS', 'Sector_InvTeam']
               3. third column onwards are the ticker's exposure to the
                  constraint bucket
               4. have columns ['Growth_PDS', 'FX_PDS', Illiquidity_PDS']
                  for weight assignments
        :param cons_saa: [pd.DataFrame] index=assets/sub-sectors
               TABLE RESTRICTIONS:
               1. index == constraint name
               2. first two columns are ['Con_Type', 'Type']
               3. third column onwards are the option names that can be
                  called to read the specific constraint parameter for each
                  option.
        :param options: [str] / [list(str)] the option name used for
                        optimization. Can be a list. The name has to be in
                        the con_saa columns to read the specified constraint.
        :param option_map: [pd.DataFrame] index=options
               TABLE RESTRICTIONS:
               1. index == options
               2. include columns ['Objective', 'Objective_Horizon',
                                   'Accumulation', 'Income']
        :param tax_map: [pd.DataFrame] (2 * n_assets x 10)
               All tax-related parameters to calculate after-tax measures.
               TABLE RESTRICTIONS:
               1. index == tickers/assets/sub-sectors
               2. first column ['Account'] has both 'Income' and 'Accumulation'
                  each account has the a full table with all assets listed.
               3. rest of the columns:
                  ['CRS', 'PFr', 'Tc', 'PD', 'DTc', 'Def Tax', 'Ts', 'Tp', 'Ta']
        :param exp_ret [array] (N x 1)
        :param exp_vol [array] (N x 1)
        :param exp_corr [array] (N x N)
        :param n_assets: [int] number of assets, should be the same as the
                         index size of map_saa and con_saa.
        :param tickers: [list(str)] list of strings of asset names
        :param long_short: [boolean] market-neutral if True, weight sum to 0
        :param bounds: [np.array/tuple] np.array of asset weight bounds or a
                       single tuple that fits for all assets
        :param te_bmk: [dict] the benchmark SAA used for calculating ex-ante
                       tracking error constraint.
               RESTRICTIONS:
               1. keys = tracking error constraint name in cons_saa, e.g.,
                         TE_PDS (tracking error relative to PDS SAA)
               2. items = df_wgt_bmk [pd.DataFrame] (n_assets, n_options),
                          e.g., PDS SAA weight
                          Each option with a max TE_PDS in cons_saa must have a
                          bmk weight column.
               Note: Can have multiple tracking error constraint key-item pairs.
        """
        super().__init__(n_assets, tickers, long_short, bounds)

        self.map_saa = map_saa
        self.cons_saa = cons_saa
        self.assets = [*self.map_saa.index]
        self.sectors = [*self.map_saa['Sector_PDS'].unique()]
        self.tax_map = tax_map
        self.te_bmk = te_bmk

        # Feature Weights
        self.growth_wgt = self.map_saa.loc[:, 'Growth_PDS']
        self.fx_wgt = self.map_saa.loc[:, 'FX_PDS']
        self.illiq_wgt = self.map_saa.loc[:, 'Illiquidity_PDS']

        # CMA assumptions used for performance metrics calculation
        self.exp_ret = exp_ret
        self.exp_vol = exp_vol
        self.exp_ret_vol = np.vstack([self.exp_ret, self.exp_vol]).T
        self.exp_corr = exp_corr
        self.exp_cov = np.diag(self.exp_vol) @ self.exp_corr @ np.diag(
            self.exp_vol)

        # Check whether options are all in the cons_saa columns:
        if isinstance(options, list):
            for i in options:
                if i not in self.cons_saa.columns[2:]:
                    raise ValueError(
                        '{} is not in the cons_saa table columns.'.format(i))
            self.options = options
        elif isinstance(options, str):
            if options not in self.cons_saa.columns[2:]:
                raise ValueError(
                    '{} is not in the cons_saa table columns.'.format(options))
            self.options = [options]
        else:
            raise TypeError('Input options is not string or list.')

        # Check whether options and option_map index are matching
        if self.options != [*option_map.index]:
            raise Exception('option_map input has incorrect options list.')
        else:
            self.option_map = option_map

        # Check tracking error constraints
        self.te_constraints = [*self.cons_saa.loc[self.cons_saa.Con_Type ==
                                                  'Option_TE'].index]
        if len(self.te_constraints) > 0:
            if self.te_bmk is None:
                raise ValueError('Need to provide the benchmark weight for '
                                 'tracking error constraint calculation.')
            elif self.te_constraints != [*self.te_bmk.keys()]:
                raise ValueError('The provided te_bmk keys should match the '
                                 'tracking error constraints in cons_saa.')

        # Protected variables
        self._option = None
        self._option_obj = None
        self._option_obj_horizon = None
        self._option_tax_account = None
        self._bounds_option = None
        self._bounds_dict = {}
        self._constraints_option = None
        self._wgt_dict_options = {}

        # Optimization output
        self.df_wgt = None
        self.df_constraint_check = None
        self.df_perf_metrics = None
        self.dict_export = {}
        self.res_opt_options = {}

    def set_option(self, option):
        """
        Set the option to go through SAA optimization.

        :param option: [str] the option name to read the specified
                             constraints.
        """

        # Check input option type and value
        if not isinstance(option, str):
            raise TypeError(
                'Input option ({}) is not a string.'.format(option))
        elif option not in self.options:
            raise ValueError(
                'Input option ({}) not in the options list.'.format(option))
        else:
            self._option = option

    def map_saa_constraints(self, option=None):
        """
        A unique method to map all saa-related constraints to acceptable
        format that can be read by the optimizers. The two key tables are
        map_saa and cons_saa.

        self.map_saa is the table that include all the mapping information.
        self.cons_saa is the table that include all the constraints.

        See the restrictions on the two tables in the class definition.

        :param option [str] the option name to read the specified
                                 constraints. If None, use self._option.

        """
        # Set option name or check current option name.
        if option is not None:
            self.set_option(option)
        else:
            if self._option is None:
                raise ValueError('Option name is not defined yet.')

        # Clear existing constraints
        if self._constraints is not None:
            self.clear_constraints()

        # Add constraints for the specified option
        for i in range(len(self.cons_saa.index)):
            con_i = self.cons_saa.index[i]
            con_type_i = self.cons_saa['Con_Type'].iloc[i]
            type_i = self.cons_saa['Type'].iloc[i]

            if np.isnan(self.cons_saa[self._option].iloc[i]):
                warnings.warn('No {} constraint for {} option'.format(
                    con_i, self._option))
                continue

            if (con_i == 'Sum') & (type_i == 'Eq'):
                sum_i = self.cons_saa.loc[con_i, self._option]
                self.add_constraint(
                    'eq', 'lambda x: x.sum() - {}'.format(sum_i))
            elif con_i == 'Prob_Meet_Objective':
                self._option_obj = self.option_map.loc[
                    self._option, 'Objective']
                self._option_obj_horizon = self.option_map.loc[
                    self._option, 'Objective_Horizon']
                x_min_i = self.cons_saa[self._option].iloc[i]
                self.add_constraint(
                    'ineq', 'lambda x: pm.prob_meet_obj(x.T @ np.array({}), '
                            'np.sqrt(x.T @ np.frombuffer({}).reshape({}, {}) '
                            '@ x), {}, {}) - {}'.format(
                        list(self.exp_ret), self.exp_cov.tostring(),
                        self.n_assets, self.n_assets, self._option_obj,
                        self._option_obj_horizon, x_min_i))
            elif con_i in self.map_saa.columns[2:]:
                if con_type_i == 'Option_PDS':
                    if type_i == 'Min':
                        x_min_i = self.cons_saa[self._option].iloc[i]
                        self.add_constraint(
                            'ineq', "lambda x: x.T @ np.array({}) - {}".format(
                                list(self.map_saa.loc[:, con_i]), x_min_i))
                    elif type_i == 'Max':
                        x_max_i = self.cons_saa[self._option].iloc[i]
                        self.add_constraint(
                            'ineq', "lambda x: {} - x.T @ np.array({})".format(
                                x_max_i, list(self.map_saa.loc[:, con_i])))
                    else:
                        warnings.warn(
                            'Constraint {} is not added for wrong '
                            'format.'.format(con_i))
                elif con_type_i == 'Sub_Sector':
                    # Figure out the sector it belongs to
                    loc_sub_sector_i = \
                        np.where(self.map_saa.loc[:, con_i] > 0)[0]
                    sector_pds_i = self.map_saa.Sector_PDS[
                        loc_sub_sector_i].unique()[0]
                    loc_sector_pds_i = \
                        np.where(self.map_saa.Sector_PDS == sector_pds_i)[0]

                    # Add constraint
                    if type_i == 'Min':
                        x_min_i = self.cons_saa[self._option].iloc[i]
                        self.add_constraint(
                            'ineq', "lambda x: "
                            "x.T @ np.array({}) / x[{}].sum() - {}".format(
                                list(self.map_saa.loc[:, con_i]),
                                list(loc_sector_pds_i),
                                x_min_i))
                    elif type_i == 'Max':
                        x_max_i = self.cons_saa[self._option].iloc[i]
                        self.add_constraint(
                            'ineq', "lambda x: "
                            "{} - x.T @ np.array({}) / x[{}].sum()".format(
                                x_max_i,
                                list(self.map_saa.loc[:, con_i]),
                                list(loc_sector_pds_i)))
                    else:
                        warnings.warn(
                            'Constraint {} is not added for wrong '
                            'format.'.format(con_i))
                else:
                    raise Exception(
                        'Unexpected constraint type {}.'.format(con_type_i))
            elif con_type_i == 'Option_TE':
                # Tracking error constraint
                bmk_i = self.te_bmk[con_i].loc[:, self._option].values
                if type_i == 'Max':
                    x_max_i = self.cons_saa[self._option].iloc[i]
                    self.add_constraint(
                        'ineq', "lambda x: {} - (x-np.array({})).T @ "
                                "np.frombuffer({}).reshape({}, {}) "
                                "@ (x-np.array({}))".format(
                            x_max_i ** 2, list(bmk_i), self.exp_cov.tostring(),
                            self.n_assets, self.n_assets, list(bmk_i)))

            elif con_type_i == 'Sector_PDS':
                # Get the location of sub-sectors that belong to the sector.
                # E.g., for Australian Equities,
                # [0, 1] in the self.map_saa.Sector_PDS vector.
                loc_sector_pds_i = np.where(self.map_saa.Sector_PDS == con_i)[0]

                # Check min or max
                if type_i == 'Min':
                    x_min_i = self.cons_saa[self._option].iloc[i]
                    self.add_constraint(
                        'ineq', "lambda x: x[{}].sum() - {}".format(
                            list(loc_sector_pds_i), x_min_i))
                elif type_i == 'Max':
                    x_max_i = self.cons_saa[self._option].iloc[i]
                    self.add_constraint(
                        'ineq', "lambda x: {} - x[{}].sum()".format(
                            x_max_i, list(loc_sector_pds_i)))
                else:
                    warnings.warn(
                        'Constraint {} is not added for wrong format.'.format(
                            con_i))
            elif con_type_i == 'Asset_Bound':
                # 'Asset_Bound' constraint is added as weight bounds into the
                # optimizer.
                continue
            else:
                warnings.warn(
                  'Constraint {} is not added for wrong format.'.format(con_i))

        # Add asset bounds for optimization
        mask_asset_bounds = self.cons_saa.Con_Type == 'Asset_Bound'
        df_asset_bounds = self.cons_saa.loc[mask_asset_bounds, :]

        # Check dimension
        if df_asset_bounds.shape[0] != 2 * self.n_assets:
            raise ValueError(
                'The Asset_Bound constraints do not match the number of assets!'
                ' {} != 2*{}'.format(df_asset_bounds.shape[0], self.n_assets))

        for a in self.options:
            bound_min_a = df_asset_bounds.loc[df_asset_bounds.Type == 'Min', a]
            bound_max_a = df_asset_bounds.loc[df_asset_bounds.Type == 'Max', a]
            self._bounds_dict[a] = list(zip(bound_min_a, bound_max_a))

    def check_saa_constraints(self):
        """
        Check whether the saa constraints are met in the output weight and
        export the results. Used after the self.options_opt_run() function.

        :return: [DataFrame]
        """

        # Check if the optimization has been run
        if self.wgt_dict is None or self.wgt_dict == {}:
            raise Exception('Empty dictionary of weight outputs.')
        elif any(isinstance(i, dict) for i in self.wgt_dict.values()):
            # Nested dictionary
            self.df_wgt = ut.nested_dict_to_df(self.wgt_dict)
        else:
            # None-Nested dictionary
            self.df_wgt = pd.DataFrame.from_dict(self.wgt_dict)

        # Initiate a dictionary to store output
        dict_constraint_check = {}

        # Check constraints for each column in the self.df_wgt
        # Basically, each optimization output
        # multi-index dataframe: (a, b) = (option name, obj function)
        for (a, b) in self.df_wgt.columns:
            df_wgt_i = self.df_wgt.loc[:, (a, b)]
            df_constraint_check_i = \
                pd.DataFrame(np.nan, index=[*self.cons_saa.index.unique()],
                             columns=['Actual', 'Constraint', 'Check'])
            # Go through the same procedure as map_saa_constraint() to check
            # if each constraint is met for this optimization output
            for i in range(len(df_constraint_check_i.index)):
                # Name of the constraint (index column)
                con_i = df_constraint_check_i.index[i]
                # print(con_i)
                df_con_i = self.cons_saa.loc[con_i, :]
                # (falseValue, trueValue)[test==True]
                con_type_i = (df_con_i['Con_Type'], df_con_i[
                    'Con_Type'][0])[type(df_con_i) != pd.Series]

                # Check if it's all np.nan for this constraint
                if np.isnan(self.cons_saa.loc[con_i, a]).sum() == \
                        self.cons_saa.loc[con_i, a].size:
                    warnings.warn('No {} constraint for {} option'.format(
                        con_i, a))
                    df_constraint_check_i.loc[con_i, :] = np.nan
                    continue

                # Calculate column 1 'Actual'
                if con_i == 'Sum':
                    df_constraint_check_i.loc[con_i, 'Actual'] = df_wgt_i.sum()
                elif con_i == 'Prob_Meet_Objective':
                    df_constraint_check_i.loc[con_i, 'Actual'] = \
                        pm.prob_meet_obj(
                            df_wgt_i.T @ self.exp_ret,
                            np.sqrt(df_wgt_i.T @ self.exp_cov @ df_wgt_i),
                            self.option_map.loc[a, 'Objective'],
                            self.option_map.loc[a, 'Objective_Horizon'])
                elif con_i in self.map_saa.columns[2:]:
                    if con_type_i == 'Option_PDS':
                        df_constraint_check_i.loc[con_i, 'Actual'] = \
                            df_wgt_i.T @ self.map_saa.loc[:, con_i]
                    elif con_type_i == 'Sub_Sector':
                        # Figure out the sector it belongs to
                        loc_sub_sector_i = \
                            np.where(self.map_saa.loc[:, con_i] > 0)[0]
                        sector_pds_i = self.map_saa.Sector_PDS[
                            loc_sub_sector_i].unique()[0]
                        loc_sector_pds_i = \
                            np.where(self.map_saa.Sector_PDS == sector_pds_i)[0]

                        # Calculate actual sub_sector/sector_pds weight
                        df_constraint_check_i.loc[con_i, 'Actual'] = \
                            df_wgt_i.T @ self.map_saa.loc[:, con_i] / \
                            df_wgt_i.values[list(loc_sector_pds_i)].sum()
                    else:
                        raise Exception(
                            'Unexpected constraint type {}.'.format(con_type_i))
                elif con_type_i == 'Option_TE':
                    # Tracking error constraint
                    bmk_i = self.te_bmk[con_i].loc[:, a].values
                    df_constraint_check_i.loc[con_i, 'Actual'] = \
                        np.sqrt((df_wgt_i - bmk_i).T @ self.exp_cov @ (
                                df_wgt_i - bmk_i))
                elif con_type_i == 'Sector_PDS':
                    # Get the location of sub-sectors that belong to the sector.
                    # E.g., for Australian Equities,
                    # [0, 1] in the self.map_saa.Sector_PDS vector.
                    loc_sector_pds_i = \
                        np.where(self.map_saa.Sector_PDS == con_i)[0]
                    df_constraint_check_i.loc[con_i, 'Actual'] = \
                        df_wgt_i.values[list(loc_sector_pds_i)].sum()
                elif con_type_i == 'Asset_Bound':
                    # Use df_wgt_i directly for actual
                    df_constraint_check_i.loc[con_i, 'Actual'] = \
                        df_wgt_i.loc[con_i]
                else:
                    Exception('Unchecked constraint {} of type {}.'.format(
                        con_i, con_type_i))

                # Map column 2 and 3
                if type(df_con_i) == pd.Series:
                    # Type is either 'Eq' or 'Max'/'Min' Alone
                    if df_con_i['Type'] == 'Eq':
                        # Col 2
                        df_constraint_check_i.loc[con_i, 'Constraint'] = \
                            df_con_i[a]
                        # Col 3
                        if np.round(df_constraint_check_i.loc[con_i,
                                                              'Actual'], 4) == \
                            df_con_i[a]:
                            df_constraint_check_i.loc[con_i, 'Check'] = 'Yes'
                        else:
                            df_constraint_check_i.loc[con_i, 'Check'] = 'No'
                    elif df_con_i['Type'] == 'Min':
                        # Col 2
                        df_constraint_check_i.loc[con_i, 'Constraint'] = \
                            '[{}, ]'.format(df_con_i[a])
                        # Col 3
                        if np.round(df_constraint_check_i.loc[con_i,
                                                              'Actual'], 4) >= \
                            df_con_i[a]:
                            df_constraint_check_i.loc[con_i, 'Check'] = 'Yes'
                        else:
                            df_constraint_check_i.loc[con_i, 'Check'] = 'No'
                    elif df_con_i['Type'] == 'Max':
                        # Col 2
                        df_constraint_check_i.loc[con_i, 'Constraint'] = \
                            '[, {}]'.format(df_con_i[a])
                        # Col 3
                        if np.round(df_constraint_check_i.loc[con_i,
                                                              'Actual'], 4) <= \
                                df_con_i[a]:
                            df_constraint_check_i.loc[con_i, 'Check'] = 'Yes'
                        else:
                            df_constraint_check_i.loc[con_i, 'Check'] = 'No'
                    else:
                        raise Exception('Unexpected constraint type other '
                                        'than Eq, Min, or Max!')
                elif df_con_i.index.size == 2:
                    # Type is 'Min' and 'Max' together
                    min_mask_i = (df_con_i.Type == 'Min')
                    max_mask_i = (df_con_i.Type == 'Max')
                    # Col 2
                    df_constraint_check_i.loc[con_i, 'Constraint'] = \
                        '[{}, {}]'.format(df_con_i.loc[min_mask_i, a][0],
                                          df_con_i.loc[max_mask_i, a][0])
                    # Col 3
                    if (np.round(df_constraint_check_i.loc[con_i, 'Actual'],
                                 4) >= df_con_i.loc[min_mask_i, a][0]) & (
                        np.round(df_constraint_check_i.loc[con_i, 'Actual'],
                                 4) <= df_con_i.loc[max_mask_i, a][0]):
                        df_constraint_check_i.loc[con_i, 'Check'] = 'Yes'
                    else:
                        df_constraint_check_i.loc[con_i, 'Check'] = 'No'
                else:
                    raise Exception('Constraint {} has unexpected number of '
                                    'rows! Should be 1 or 2.'.format(con_i))
            dict_constraint_check[(a, b)] = df_constraint_check_i

        # Put the nested dictionary to dataframe
        self.df_constraint_check = ut.nested_dict_to_df(dict_constraint_check)

        return self.df_constraint_check

    def options_opt_run(self, obj_str, args, print_res=False,
                        after_tax_opt=True, after_tax_output=True,
                        check_constraints=True, opt_perf_metrics=True,
                        **kwargs):
        """
        Loop through all the options in the self.options for optimization.
        Run the self.opt_run() method for each option and each objective
        function defined.
        Store the result to res_opt_options for all options and objective
        functions defined.

        :param: after_tax_opt: [boolean] If true, convert args to after-tax
                for optimization.
        :param: after_tax_output: [boolean] If true, use the instance
                self.exp_ret, self.exp_vol, self.exp_corr to calculate the
                 after-tax performance metrics.
        
        :return: res_opt_options [dict] a nested dictionary stores all
                 options optimization results in the format: e.g.,
                {'Diversified': {{'max_sharpe': res_opt}, {'min_var': res_opt}},
                'Balanced': {{'max_sharpe': res_opt}, {'min_var': res_opt}}}
        """
        for i in self.options:
            self.set_option(i)
            print(self._option)

            # Clear and Add constraint
            self.clear_constraints()
            self.map_saa_constraints(self._option)

            if self._bounds_dict[self._option] is not None:
                self._bounds_option = self._bounds_dict[self._option]
            else:
                self._bounds_option = self.bounds

            # Convert before-tax exp_ret into after-tax
            if after_tax_opt and obj_str == ['max_sharpe']:
                # Currently only works for 'max_sharpe' otherwise the
                # order of arguments is not correct
                for k in args.keys():
                    # Construct a list since tuple cannot be changed
                    list_arg_k = list(args[k])
                    exp_ret_at_k, exp_vol_at_k = pm.after_tax_exp_ret_vol(
                        self.tax_map, self.exp_ret_vol,
                        account=self.option_map.loc[self._option,
                                                    'Account'])
                    list_arg_k[0] = exp_ret_at_k
                    exp_cov_at_k = np.diag(exp_vol_at_k) @ self.exp_corr @ \
                                   np.diag(exp_vol_at_k)
                    list_arg_k[1] = exp_cov_at_k
                    args[k] = tuple(list_arg_k)

            # Run optimization and Store results for each objective function
            self.res_opt_options[self._option] = \
                self.opt_run(obj_str, args, bounds=self._bounds_option,
                             print_res=print_res)

            # Add self.wgt_dict to the self._wgt_dict_options dictionary
            self._wgt_dict_options[self._option] = self.wgt_dict

            # Clean the self._res_opt and self.wgt_dict for the next
            # optimization run
            self._res_opt = {}
            self.wgt_dict = {}
        self.wgt_dict = self._wgt_dict_options

        # Check SAA Constraints
        if check_constraints:
            # Results stored in self.df_constraint_check
            self.check_saa_constraints()

        # Generate Regulatory Performance Metrics
        if opt_perf_metrics:
            # Results stored in self.df_perf_metrics
            self.opt_perf_metrics(after_tax=after_tax_output, **kwargs)

        return self.res_opt_options

    def opt_perf_metrics(self, fx_hedge=0.5, cpi=0.024, after_tax=True):
        """
        Calculate the standard performance metrics for optimized weights.
        :param exp_ret: [pd.Series] (N x ), expected return,
        :param exp_cov: [pd.DataFrame] (N x N) expected covariance matrix
        :param fx_hedge: [float] foreign currency hedging ratio, default = 0.5
        :param cpi: [float] 10-yr cpi assumption
        :param after_tax: [boolean] provide after-tax performance metrics

        :return: df_perf_metrics
        """
        # Check if the optimization has been run
        if self.wgt_dict is None or self.wgt_dict == {}:
            raise Exception('Empty dictionary of weight outputs.')
        elif any(isinstance(i, dict) for i in self.wgt_dict.values()):
            # Nested dictionary
            self.df_wgt = ut.nested_dict_to_df(self.wgt_dict)
        else:
            # None-Nested dictionary
            self.df_wgt = pd.DataFrame.from_dict(self.wgt_dict)

        # Initiate a dictionary to store output
        list_perf_metrics = []
        for (a, b) in self.df_wgt.columns:
            self._option = a
            self._option_obj = self.option_map.loc[self._option, 'Objective']
            self._option_obj_horizon = self.option_map.loc[self._option,
                                                           'Objective_Horizon']
            self._option_tax_account = self.option_map.loc[self._option,
                                                           'Account']
            perf_metrics_i = pm.OptionPerfMetrics(
                self.df_wgt.loc[:, (a, b)], self.exp_ret, self.exp_vol,
                self.exp_corr,
                self._option_obj,
                self._option_obj_horizon, self.map_saa.loc[:, 'Growth_PDS'],
                self.map_saa.loc[:, 'FX_PDS'],
                self.map_saa.loc[:, 'Illiquidity_PDS'],
                fx_hedge=fx_hedge, cpi=cpi, after_tax=after_tax,
                tax_map=self.tax_map,
                tax_account=self._option_tax_account).standard_metrics()

            list_perf_metrics.append(perf_metrics_i)

        self.df_perf_metrics = pd.concat(list_perf_metrics, axis=1)
        return self.df_perf_metrics

    def export_saa_output(self, save_excel=False, filename="saa_output.xlsx"):
        """
        Utility function to export all output to a DataFrame and save weights
        to an excel file.
        The exported dataframes include:
        1. Asset_Wgt
        2. Sector_PDS_Wgt
        3. Sector_InvTeam_Wgt
        4. Constraints_Check
        5. Regulatory_Metrics (OptionPerfMetrics)
        *6. Perf_Metrics (needs data)

        :param: save_excel [boolean] save to an Excel file if True.
        :param: filename [str] name of file. Should be xlsx.

        :return: self.dict_export [dict] a dictionary of all export dataframes
        """

        # Get the export_wgt output first (asset level - 1st hierarchy weights)
        if self.df_wgt is None:
            raise Exception('Check SAA Constraints not run yet. Set '
                            'check_constraints kwarg to True and rerun '
                            'options_opt_run().')
        else:
            self.dict_export['Asset_Wgt'] = self.df_wgt

        # Group wgts into different hierarchies ('Sector_PDS', 'Sector_InvTeam')
        # Specified in self.map_saa first two cols
        for i in ['Sector_PDS', 'Sector_InvTeam']:
            df_i = pd.concat([self.df_wgt, self.map_saa[[i]]], axis=1)
            df_wgt_i = df_i.groupby(by=i, sort=False).sum()
            df_wgt_i.loc['Sum', :] = df_wgt_i.sum(axis=0)
            self.dict_export['{}_Wgt'.format(i)] = df_wgt_i

        # Add constraints check to self.dict_export
        self.dict_export['Constraints_Check'] = self.df_constraint_check

        # Add regulatory metrics to self.dict_export
        self.dict_export['Regulatory_Metrics'] = self.df_perf_metrics

        if save_excel:
            with pd.ExcelWriter(filename) as writer:
                for i in self.dict_export:
                    self.dict_export[i].to_excel(writer, sheet_name=i)

        return self.dict_export

    def visualize_efficient_frontier(self, args, ef_type='Mean-Var',
                                     constrained=False, rf=0.0025,
                                     n_samples=1000):
        """
        This function allows visualization of the efficient
        frontier with simulated random portfolios. The function also returns
        a dataframe summarizing the efficient frontier points (ret, vol,
        wgt) as well as random portfolio points (ret, vol, wgt).
        * The default opt_type is mean-variance optimization using
          'max_quad_utility' function.
        * The default plot is to plot all existing optimization results on the
          same efficient frontier.

        :param args [tuple] different arguments required for different ef_type
               'Mean-Var': args = (exp_ret, exp_cov)
               'Mean-CVaR': args = (df_ret, exp_ret, exp_cov)
               'Mean-TE': args = (exp_ret, exp_cov, bmk_wgt)
        :param ef_type [str] default 'Mean-Var'
               Efficient frontier type.
               Available types: 'Mean-Var', 'Mean-CVaR', 'Mean-TE'
               *under development: 'Mean-CVar', 'Mean-TE'
        :param constrained [boolean] default False
               Define whether the random portfolio weights are also constrained.
        :param rf [float] risk-free rate used to calculate Sharpe Ratio
        :param n_samples [int] number of random weights generated from
               Dirichlet distribution
        :return: df_ef, df_random [pd.DataFrame]
        """
        # Get len(a) and len(b)
        #   len(a): Number of options - color
        color_dict = {}
        color_list = ut.cmap_rgba('tab20', len(self.options))
        for i in range(len(self.options)):
            # np.random.seed(i)
            color_dict[self.options[i]] = color_list[i]

        #   len(b): Number of objective_functions - marker
        marker_list = ['*', 'P', 'X', 'o', 'd', 's']
        marker_dict = {}
        for i in range(len(self._objective_list)):
            marker_dict[self._objective_list[i]] = marker_list[i]

        if ef_type == 'Mean-Var':
            # Start with the default 'Mean-Var' optimization and return a
            # 'Exp_Ret-Volatility' chart
            # Minimum Requirement: args = (exp_ret, exp_cov)
            exp_ret, exp_cov = args

            # Ret-Vol Scatter Plot
            plt.figure(figsize=(10, 8))
            plt.title('Efficient Frontier with Random Portfolios')

            for (a, b) in self.df_wgt.columns:
                # Plot the dots - random weights
                df_wgt_random = sim.random_weights(self.n_assets, n_samples,
                                                   alpha=0.3)
                vol_random = np.sqrt(np.diag(df_wgt_random @ exp_cov @
                                             df_wgt_random.T))
                ret_random = df_wgt_random.dot(exp_ret)
                sharpe_random = (ret_random - rf) / vol_random
                plt.scatter(vol_random, ret_random, marker=".", alpha=0.5,
                            c=sharpe_random, cmap='viridis', zorder=0)

                # Plot the stars - optimized weights
                df_wgt_i = self.df_wgt.loc[:, (a, b)]
                vol_i = pm.port_vol(df_wgt_i, exp_cov, freq='yearly')
                ret_i = pm.port_ret(df_wgt_i, exp_ret, freq='yearly')
                plt.scatter(vol_i, ret_i, c=color_dict[a],
                            marker=marker_dict[b], s=250,
                            label='{}-{}'.format(a, b), zorder=10)

            # Plot the line - efficient frontier
            wgt_ef = []
            ret_ef = []
            vol_ef = []
            for i in np.linspace(1, 1000, 1000):
                args_i = (exp_ret, exp_cov, i)
                self._objective = obj.call_obj_functions('max_quad_utility')
                res_i = minimize(self._objective,
                                 x0=self._wgt0,
                                 bounds=[(0, 1)] * self.n_assets,
                                 args=args_i, method='SLSQP',
                                 constraints=
                                 {'type': 'eq', 'fun': lambda x: x.sum() - 1},
                                 options={'maxiter': 10000, 'disp': True})
                wgt_ef.append(res_i.x)
                ret_ef.append(np.dot(res_i.x, exp_ret))
                vol_ef.append(np.sqrt(res_i.x.T @ exp_cov @ res_i.x))
            plt.plot(vol_ef, ret_ef, c='blue', linestyle='dashed', zorder=10,
                     label='Unconstrained Efficient Frontier')

            # Plot the rest
            plt.colorbar()
            plt.xlabel('Volatility')
            plt.ylabel('Return')
            plt.legend(loc='upper left', bbox_to_anchor=(1.15, 0.9))
            plt.show()
        elif ef_type == 'Mean-CVaR' or ef_type == 'Mean-TE':
            print('Mean-CVaR and Mean-TE approaches are under construction.')

        # Prepare the output dataframe
        # 1. EF portfolios
        df_wgt_ef = pd.DataFrame(wgt_ef, columns=self.tickers)
        df_ret_ef = pd.Series(ret_ef, name='Ret')
        df_vol_ef = pd.Series(vol_ef, name='Vol')
        df_sharpe_ef = pd.Series((df_ret_ef - rf) / df_vol_ef, name='Sharpe')
        df_growth_ef = pd.Series(df_wgt_ef @ self.growth_wgt, name='Growth')
        df_illiq_ef = pd.Series(df_wgt_ef @ self.illiq_wgt, name='Illiquidity')
        df_fx_ef = pd.Series(df_wgt_ef @ self.fx_wgt, name='FX')

        df_ef = pd.concat([df_ret_ef, df_vol_ef, df_sharpe_ef, df_growth_ef,
                           df_illiq_ef, df_fx_ef, df_wgt_ef], axis=1)

        # 2. Random portfolios
        df_wgt_random = pd.DataFrame(df_wgt_random, columns=self.tickers)
        df_ret_random = pd.Series(ret_random, name='Ret')
        df_vol_random = pd.Series(vol_random, name='Vol')
        df_sharpe_random = pd.Series(sharpe_random, name='Sharpe')
        df_growth_random = pd.Series(df_wgt_random @ self.growth_wgt,
                                     name='Growth')
        df_illiq_random = pd.Series(df_wgt_random @ self.illiq_wgt,
                                    name='Illiquidity')
        df_fx_random = pd.Series(df_wgt_random @ self.fx_wgt, name='FX')

        df_random = pd.concat([df_ret_random, df_vol_random,
                               df_sharpe_random, df_growth_random,
                               df_illiq_random, df_fx_random, df_wgt_random],
                              axis=1)

        return df_ef, df_random

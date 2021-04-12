"""
The port_opt submodule houses the PortOpt parent class, which includes several
embedded objective functions with flexibility in imposing constraints and
bounds.

The parent class PortOpt can be inherited to several use cases, e.g., SAA, DAA,
within-sector optimization. The child classes are also maintained here.

The optimization algorithm is based on scipy.optimize.shgo, which is the
go-to approach for relatively low dimensional settings. For SAA, DAA,
and within-sector optimization, it seems to be the best performer.
(ref: https://www.microprediction.com/blog/optimize)
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import shgo, minimize, dual_annealing

import port_opt.objective_functions as obj
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
                raise ValueError(
                    'New weights do not sum up to {} (sum={})'.format(
                        self._wgt_sum, wgt.sum()))
            else:
                pass
        else:
            if self.wgt is not None:
                if round(self.wgt.sum(), 2) != self._wgt_sum:
                    raise ValueError(
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
        Add a constraint to scipy.shgo optimizer. The format of constraint is:
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
                Arguments passed on for scipy.optimize.shgo optimizers.
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
    saa_perf_metrics(): Compute the SAA-related performance metrics.

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
    add_clean_wgt(): rounds up to 2 decimals, add as a series to the wgt_dict
    export_wgt(): save weights to csv file
    """

    def __init__(self, map_saa, cons_saa, options, n_assets,
                 tickers=None, long_short=False, bounds=(0, 1)):
        """
        :param map_saa: [pd.DataFrame]
               TABLE RESTRICTIONS:
               1. index == tickers/assets/sub-sectors
               2. first two columns are ['Sector_PDS', 'Sector_InvTeam']
               3. third column onwards are the ticker's exposure to the
                  constraint bucket
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
        :param n_assets: [int] number of assets, should be the same as the
                         index size of map_saa and con_saa.
        :param tickers: [list(str)] list of strings of asset names
        :param long_short: [boolean] market-neutral if True, weight sum to 0
        :param bounds: [np.array/tuple] np.array of asset weight bounds or a
                       single tuple that fits for all assets
        """
        super().__init__(n_assets, tickers, long_short, bounds)

        self.map_saa = map_saa
        self.cons_saa = cons_saa
        self.sectors = [*self.map_saa['Sector_PDS'].unique()]

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

        # Protected variables
        self._option = None
        self._constraints_option = None
        self._wgt_dict_options = {}

        # Optimization output
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

            if (con_i == 'Sum') & (type_i == 'Sum'):
                sum_i = self.cons_saa.loc[con_i, self._option]
                self.add_constraint(
                    'eq', 'lambda x: x.sum() - {}'.format(sum_i))
            elif con_i in self.map_saa.columns[2:]:
                if con_type_i == 'Sub_Sector':
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
            else:
                warnings.warn(
                  'Constraint {} is not added for wrong format.'.format(con_i))

    def options_opt_run(self, obj_str, args, bounds=None, print_res=False):
        """
        Loop through all the options in the self.options for optimization.
        Run the self.opt_run() method for each option and each objective
        function defined.
        Store the result to res_opt_options for all options and objective
        functions defined.
        
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

            # Run optimization and Store results for each objective function
            self.res_opt_options[self._option] = \
                self.opt_run(obj_str, args, bounds=bounds, print_res=print_res)

            # Add self.wgt_dict to the self._wgt_dict_options dictionary
            self._wgt_dict_options[self._option] = self.wgt_dict

            # Clean the self._res_opt and self.wgt_dict for the next
            # optimization run
            self._res_opt = {}
            self.wgt_dict = {}
        self.wgt_dict = self._wgt_dict_options

        return self.res_opt_options


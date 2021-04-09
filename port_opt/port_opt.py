"""
The port_opt submodule houses the PortOpt parent class, which includes several
embedded objective functions with flexibility in imposing constraints and
bounds.

The parent class PortOpt can be inherited to several use cases, e.g., SAA, DAA,
within-sector optimizaton. The child classes are also maintained here.

The optimization algorithm is based on scipy.optimize.shgo, which is the
go-to approach for relatively low dimensional settings. For SAA, DAA,
and within-sector optimization, it seems to be the best performer.
(ref: https://www.microprediction.com/blog/optimize)
"""

import collections
import numpy as np
import pandas as pd
from scipy.optimize import shgo


class PortOptBase:
    """
    PortOptBase class to incorporate basic variables and functions for an
    instance of base optimization.
    ------------------
    ### Instance variables ###
    n_assets: [int] number of assets
    tickers: [list(str)] list of strings of asset names
    wgt: [np.array] np.array of asset weights

    ### Public Methods ###
    set_wgt(): set weights from user input
    clean_wgt(): rounds up to 2 decimals
    export_wgt(): save weights to csv file
    """

    def __init__(self, n_assets, tickers=None):
        """
        :param n_assets: [int] number of assets
        :param tickers: [list(str)] list of strings of asset names
        """
        self.n_assets = n_assets
        if tickers is None:
            self.tickers = list(range(n_assets))
        self._rf = None
        # Outputs
        self.wgt = None

    def _make_wgt_series(self, wgt=None):
        """
        Utility function to make output wgt pd.Series from wgt np.array.
        Use self.tickers and self.wgt if no wgt argument passed.
        """
        if wgt is None:
            wgt = self.wgt

        return pd.Series(wgt, index=self.tickers)

    def set_wgt(self, input_wgt):
        """
        Utility function to set wgt attribute from user input
        * Check: len(input_wgt) == self.n_assets

        :param input_wgt: [list/array] list/array of weights

        """
        if len(input_wgt) != self.n_assets:
            raise ValueError('Length of list not equal to number of assets')
        self.wgt = np.array(input_wgt)

    def clean_wgt(self, cutoff=1e-4, decimal=4):
        """
        Helper function to clean the raw weights, setting any weights whose
        absolute values are below the cutoff to zero, and round the rest.

        :param cutoff: [float] the lower bound, default to 1e-4
        :param decimal: [int] number of decimal places to round weights,
                      default 4
                      * Check: positive integer (int & >1)
        :return: wgt [dict] from ._make_output_wgt() method
        """
        if self.wgt is None:
            raise AttributeError('Weights not yet computed')
        clean_wgt = self.wgt.copy()
        clean_wgt[np.abs(clean_wgt) < cutoff] = 0
        if not isinstance(decimal, int) or decimal < 1:
            raise ValueError('Param decimal must be a positive integer')
        else:
            clean_wgt = np.round(clean_wgt, decimal)

        return self._make_wgt_series(clean_wgt)

    def export_wgt(self, filename="wgt.csv"):
        """
        Utility function to save weights to a csv file.

        :param filename: [str] name of file. Should be csv.
        """
        self.clean_wgt().to_csv(filename, header=False)

    def _wgt_check(self, wgt_sum=1):
        """
        Utility function to check whether final weights sum up to 1.
        * If leveraged or long-short, allows change of parameter to sum up to
          0 or else.
        """
        if self.wgt is not None:
            if round(self.wgt.sum(), 2) != wgt_sum:
                raise ValueError('Weights do not sum up to {} (sum={})'.format(
                    wgt_sum, self.wgt.sum()))
        else:
            raise AttributeError('Weights not yet computed')


class PortOpt(PortOptBase):
    """
    The PortOpt class inherits the parent PortOptBase class and uses the
    scipy.optimize.shgo global minimum optimizer to generate optimized weights.
    Additional features include constraints and built-in objective functions.

    ### Instance variables ###
    bounds: [tuple] or [list(tuple)]: min and max weight of each asset or
            single pair if all identical, default (0, 1), need to be changed
            to (-1, 1) if allow shorting.
    n_assets: [int] number of assets
    tickers: [list(str)] list of strings of asset names
    wgt: [np.array] np.array of asset weights

    ### Public Methods ###
    add_constraint(): add a constraint for optimization
    set_wgt(): set weights from user input
    clean_wgt(): rounds up to 2 decimals
    export_wgt(): save weights to csv file

    """

    def __init__(self, n_assets, tickers=None, bounds=(0, 1),
                 obj_str='max_sharpe'):
        """

        :param n_assets: [int] number of assets
        :param tickers: [list(str)] list of strings of asset names
        :param bounds: [np.array/tuple] np.array of asset weight bounds or a
                        single tuple that fits for all assets
        :param obj_str: [str] or [list(str)] name of objective functions
                        e.g., obj_str = 'max_sharpe' OR
                        ['max_sharpe', 'min_cvar']
                        * If a list is passed, generate outcomes from different
                        objective functions.
        """
        super().__init__(n_assets, tickers)

        # Optimization variables
        if isinstance(bounds, list) and len(bounds) == self.n_assets:
            self.bounds = bounds
        elif isinstance(bounds, tuple):
            self.bounds = [bounds] * self.n_assets
        else:
            self.bounds = [(None, None)] * self.n_assets
            raise Warning('Bounds are not defined.')

        # Local variables
        self._wgt = None
        self._objective = None
        self._objective_list = []
        self._constraints = []
        self._lower_bounds = None
        self._upper_bounds = None

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

    def opt_run(self, obj_str, bounds=None, plot_ef=False, print_res=False):
        """
        Execute optimization based on defined objective functions and added
        constraints.

        :param obj_str: [str] or [list(str)]
                Name[s] of pre-defined objective functions
                e.g., obj_str = 'max_sharpe' / ['max_sharpe', 'min_cvar']
                *If a list is passed, generate outcomes from different objective
                functions.
        :param bounds: [list/tuple] redefine the bounds for the optimization run
                *If None, use instance variable self.bounds.
        :param plot_ef: [boolean] plot efficient frontier if True
        :param print_res: [boolean] print results if True
        """
        # Objective functions
        if isinstance(obj_str, list):
            self._objective_list = obj_str
        else:
            self._objective_list.append(obj_str)

        # Define bounds
        if bounds is not None:
            if isinstance(bounds, list) and len(bounds) == self.n_assets:
                self.bounds = bounds
            elif isinstance(bounds, tuple):
                self.bounds = [bounds] * self.n_assets
            else:
                raise TypeError('Bounds are not list or tuple, or the length '
                                'of list does not match the n_assets.')

        # Loop through objective functions for optimization

        # Might Need a function to call those objective functions

        # Need a efficient frontier class to plot the results.
        return abc

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

        return abc
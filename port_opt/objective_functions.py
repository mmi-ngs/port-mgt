"""
The objective_function submodule houses the key classical objective functions
used in the portfolio management context. No class implemented to allow
flexibility to add more functions and allow different parameter inputs for
different objective functions.

The implemented objective functions include:
* Max Sharpe
* Max IR
* Min Variance
* Min CVaR
* Max Calmar
* Hierarchical Risk Parity (Marcos De Lopez, 2016)

"""

from types import FunctionType
import numpy as np
import perf_metrics as pm
import utils as ut


def call_obj_functions(obj_str):
    """
    Help function to summarize all available objective functions to be called
    in the PortOpt instances.

    :param obj_str: [str] the name of the objective functions
    :return: obj_func: [FunctionType] a callable objective function.
    """
    obj_avail = {'min_var': port_var,
                 'max_sharpe': sharpe_ratio,
                 'max_quad_utility': quadratic_utility}
    if obj_str in obj_avail.keys():
        obj_func = obj_avail[obj_str]
    else:
        raise ValueError(
            '{} is an undefined objective function. The available '
            'objective functions include {}.'.format(obj_str,
                                                     [*obj_avail.keys()]))

    if not isinstance(obj_func, FunctionType):
        raise TypeError('{} is not a FunctionType.'.format(obj_avail[obj_str]))
    return obj_func


def port_var(wgt, cov):
    """
    Calculate the portfolio variance based on the weight and covariance matrix.

    :param wgt: [np.ndarray] (Nx1) weight of assets
    :param cov: [np.ndarray] (NxN) covariance matrix
    :return: var: [float] variance of the portfolio
    """
    if not (isinstance(wgt, np.ndarray) & isinstance(cov, np.ndarray)):
        raise TypeError('One input out of [wgt, cov] is not np.ndarray.')

    if not ((wgt.ndim == 1) & (cov.ndim == 2)):
        raise AttributeError('Dimensions of [wgt, cov] are not [1, 2].')

    return wgt.T @ cov @ wgt


def sharpe_ratio(wgt, ret, cov, rf=0.0025, negative=True):
    """
    Calculate the Sharpe ratio based on a static risk-free rate.

    :param wgt: [np.ndarray] (Nx1) weight of assets
    :param ret: [np.ndarray] (Nx1) return of assets
    :param cov: [np.ndarray] (NxN) covariance matrix
    :param rf: [float] static risk-free rate
    :param negative: [boolean] set True if return negative Sharpe Ratio
    :return: sharpe: [float] Sharpe Ratio of the portfolio
    """
    if not (isinstance(wgt, np.ndarray) & isinstance(ret, np.ndarray) &
            isinstance(cov, np.ndarray) & isinstance(rf, float)
            & isinstance(negative, bool)):
        raise TypeError(
            'At least one input out of [wgt, ret, cov, rf, negative] is not '
            'matching [np.ndarray, np.ndarray, np.ndarray, float, bool].')

    if not ((wgt.ndim == 1) & (ret.ndim == 1) & (cov.ndim == 2)):
        raise AttributeError('Dimensions of [wgt, ret, cov] are not [1, 1, 2].')

    sign = -1 if negative else 1
    sharpe = sign * (wgt.T @ ret - rf) / np.sqrt(wgt.T @ cov @ wgt)
    return sharpe


def quadratic_utility(wgt, ret, cov, risk_aversion, rf=0.0025, negative=True):
    """
    Quadratic utility function,
    i.e :math:`w.T @ r - 0.5 * A * w.T @ Sigma @ w`.

    :param wgt: [np.ndarray] (Nx1) weight of assets
    :param ret: [np.ndarray] (Nx1) return of assets
    :param cov: [np.ndarray] (NxN) covariance matrix
    :param risk_aversion: [float] risk aversion coefficient. Higher
                          risk_aversion, lower portfolio risk.
    :param rf: [float] static risk-free rate
    :param negative: [boolean] set True if return negative value
    :return: quad_utility: [float] value of the objective function
    """
    if not (isinstance(wgt, np.ndarray) & isinstance(ret, np.ndarray) &
            isinstance(cov, np.ndarray) &
            (isinstance(risk_aversion, float) or isinstance(risk_aversion, int))
            & isinstance(rf, float) & isinstance(negative, bool)):
        raise TypeError(
            'At least one input out of '
            '[wgt, ret, cov, risk_aversion, rf, negative]'
            ' is not matching '
            '[np.ndarray, np.ndarray, np.ndarray, float/int, float, bool].')

    if not ((wgt.ndim == 1) & (ret.ndim == 1) & (cov.ndim == 2)):
        raise AttributeError('Dimensions of [wgt, ret, cov] are not [1, 1, 2].')

    sign = -1 if negative else 1
    quad_utility = \
        sign * (wgt.T @ ret - rf - 0.5 * risk_aversion * wgt.T @ cov @ wgt)
    return quad_utility



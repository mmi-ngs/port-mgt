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


import numpy as np
import perf_metrics as pm
import utils as ut


def _input_type_check():
    """
    Help function to loop through the inputs for a function adn checks the
    input type. Otherwise raise TypeError.
    """
    return abc


def _sum_obj_functions(obj_str):
    """
    Help function to summarize all available objective functions to be called
    in the PortOpt instances.

    :param obj_str: [str] the name of the objective functions
    :return:
    """
    obj_avail = ['neg_sharpe_ratio']
    return abc


def sharpe_ratio(wgt, ret, cov, rf=0.0025):
    """
    Calculate the Sharpe ratio based on a static risk-free rate.

    :param wgt: [np.ndarray], (Nx1) weight of assets
    :param ret: [np.ndarray], (Nx1) return of assets
    :param cov: [np.ndarray], (NxN) covariance matrix
    :param rf: [float], static risk-free rate
    :return: sharpe: [float], Sharpe Ratio of the portfolio
    """
    if not (isinstance(wgt, np.ndarray) & isinstance(ret, np.ndarray) &
            isinstance(cov, np.ndarray)):
        raise TypeError('One input out of [wgt, ret, cov] is not np.ndarray.')

    if not ((wgt.ndim == 1) & (ret.ndim == 1) & (cov.ndim == 2)):
        raise AttributeError('Dimensions of [wgt, ret, cov] are not [1, 1, 2].')

    sharpe = (wgt.T @ ret - rf) / (wgt.T @ cov @ wgt)
    return sharpe


def neg_sharpe_ratio(wgt, ret, cov, rf=0.0025):
    """
    Negative Sharpe ratio based on a static risk-free rate, for optimization
    purpose.

    :param wgt: [np.ndarray], (Nx1) weight of assets
    :param ret: [np.ndarray], (Nx1) return of assets
    :param cov: [np.ndarray], (NxN) covariance matrix
    :param rf: [float], static risk-free rate
    :return: neg_sharpe: [float], Negative Sharpe Ratio of the portfolio
    """
    return -sharpe_ratio(wgt, ret, cov, rf)

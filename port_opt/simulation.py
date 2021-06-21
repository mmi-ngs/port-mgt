"""
The simulation submodule is used for Monte Carlo simulation and generate
simulated outcomes, such as predicative distribution, expected moments of
distribution.

The simulation models vary across Frequentist and Bayesian approaches, include
univariate and multivariate models, and uses copulas for advanced
distributional models.
"""

import numpy as np
import scipy


def random_weights(n_assets, n_samples, alpha=1):
    """
    This function generates random weights for a portfolio consists of n_assets.

    Use dirichlet distribution with alpha is 1 for all assets to ensure
    uniformly distributed random weights drawn for each asset.

    :param n_assets [int] number of assets in the portfolio
    :param n_samples [int] number of samples to generate
    :param alpha [float] / [list of float]
                 parameter for dirichlet distribution, assumed the
                 same for all assets, default = 1.
           NOTE: If want more weights concentrated on specific assets (
           corner), use alpha < 1. If want more weights split between assets
           (middle), use alpha > 1.
           See "https://stats.stackexchange.com/questions/244917/
           what-exactly-is-the-alpha-in-the-dirichlet-distribution"

           Can use a list of alphas to ensure the best shape of the random
           weights in an efficient frontier.

    :return df_wgt_random [np.array] (n_samples x n_assets)
    """
    if isinstance(alpha, float) or isinstance(alpha, int):
        df_wgt_random = np.random.dirichlet(np.ones(n_assets) * alpha,
                                            n_samples)
    elif isinstance(alpha, list):
        list_wgt_random = []
        for i in alpha:
            df_wgt_random_i = np.random.dirichlet(np.ones(n_assets) * i,
                                                  int(n_samples/len(alpha)))
            list_wgt_random.append(df_wgt_random_i)
        df_wgt_random = np.concatenate(list_wgt_random, axis=0)
    else:
        raise ValueError('Alpha is not int/float or a list of int/float.')

    return df_wgt_random

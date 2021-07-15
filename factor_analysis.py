import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import utils as ut

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram


class FactorModel:
    """
    Use LinearModels LinearFactorModel to decompose manager returns with
    FF-5-factor model.
    --inputs--
    df_eq_ex:   series/dataframe:
                excess return dataframe of managers/portfolios
    df_factors: FF-5 factors for different regions
    freq:       str; 'monthly' or 'daily'
    period:     nt or tuple
                1) int: the most recent x years
                2) tuple: (start_date, end_date)
    """

    def __init__(self, df_eq_ex, df_factors, freq='monthly', period=999):
        # Data frequency adjustment
        self.freq = freq
        self.Q = ut.freq_adj(freq)

        # Data Period adjustment
        df = pd.concat([df_eq_ex, df_factors], axis=1)
        df = ut.first_common_date(df)[1].dropna()
        self.period = period

        if type(df_eq_ex) == pd.Series:
            self.N = 1
            self.df_eq_ex = ut.period_adj(df.iloc[:, 0], period, freq)
            self.eq_str = [df_eq_ex.name]
        else:
            self.eq_str = [*df_eq_ex.columns]
            self.N = len(self.eq_str)
            self.df_eq_ex = ut.period_adj(df.loc[:, self.eq_str], period, freq)

        factors_str = df_factors.columns
        self.df_factors = ut.period_adj(df.loc[:, factors_str], period, freq)

    def df_factor_decomp(self):
        if self.N == 1:
            box_params, box_tstats, box_r2_adj = factor_decomp(
                self.df_eq_ex, self.df_factors)
        else:
            box_params, box_tstats, box_r2_adj = [], [], []
            for i in range(self.N):
                params_i, tstats_i, r2_adj_i = factor_decomp(
                    self.df_eq_ex.iloc[:, i], self.df_factors)
                box_params.append(params_i)
                box_tstats.append(tstats_i)
                box_r2_adj.append(r2_adj_i)

        df_params = pd.DataFrame(box_params)
        df_tstats = pd.DataFrame(box_tstats)
        series_r2_adj = pd.Series(box_r2_adj,
                                  index=self.eq_str,
                                  name='Adjusted R Square')

        fig, axs = plt.subplots(1, 3, figsize=(28, 23))
        fig.suptitle('Factors Decomposition ({} - {})'.format(
                self.df_eq_ex.index[0].strftime('%Y-%m-%d'),
                self.df_eq_ex.index[-1].strftime('%Y-%m-%d')), fontsize=15)

        cmap = sns.diverging_palette(10, 220, as_cmap=True)
        sns.heatmap(df_params.round(2), annot=True, annot_kws={"size": 5},
                    cmap=cmap, ax=axs[0], vmin=-1, vmax=1)
        axs[0].set_yticklabels(axs[0].get_yticklabels(), rotation=0)
        axs[0].set_title('Factor Loadings')

        # 'YlGnBu'
        sns.heatmap(df_tstats.round(2), annot=True, annot_kws={"size": 5},
                    cmap=cmap, ax=axs[1], vmin=-3, vmax=3)
        axs[1].set_yticklabels(axs[1].get_yticklabels(), rotation=0)
        axs[1].set_title('Factor t-stats')

        wid = np.arange(len(series_r2_adj.index))[::-1]
        axs[2].barh(wid, series_r2_adj)
        for i in range(len(wid)):
            axs[2].text(x=series_r2_adj.iloc[i]/2, y=wid[i],
                        s=round(series_r2_adj.iloc[i], 2))
        axs[2].set_yticks(wid)
        axs[2].set_yticklabels(series_r2_adj.index)
        axs[2].set_title('Adjusted R-square')

        fig.tight_layout()

        return df_params, df_tstats, series_r2_adj


def rolling_factor_decomp(y, x, roll_period, roll_freq, trading_day=True):
    """
    y: series of 1 variable y with name
    x: df of n factors x
    roll_period: float, in years: rolling N-yr factor decomposition
    roll_freq: string, the frequency of the roll_period,
               e.g. daily/monthly/yearly.
    trading_day: boolean, True or False, used for 'daily' freq only.
        if True: q_factor = 252;
        if False: q_factor = 365.
    """
    # Period adjustment
    window = int(ut.freq_adj(roll_freq, trading_day) * roll_period)
    dates = y.iloc[int(window-1):].index

    # Rolling regression
    box_params, box_tstats, box_r2_adj = [], [], []
    for i in range(window, int(y.index.size+1)):
        y_i = y.iloc[int(i-window):i]
        x_i = x.iloc[int(i-window):i, :]
        params_i, tstats_i, r2_adj_i = factor_decomp(
            y_i, x_i, print_summary=False)
        box_params.append(params_i)
        box_tstats.append(tstats_i)
        box_r2_adj.append(r2_adj_i)

    # Pack results into dataframes
    df_params = pd.DataFrame(box_params, index=dates)
    df_tstats = pd.DataFrame(box_tstats, index=dates)
    series_r2_adj = pd.Series(box_r2_adj, index=dates)

    # Plot the three subplots (params, tstats, r2_adj)
    fig, axs = plt.subplots(3, 1, figsize=(28, 20))
    fig.suptitle('{} Rolling {}-{} Factors Decomposition ({} - {})'.format(
        y.name,
        window, roll_freq[0].capitalize(),
        y.index[0].strftime('%Y-%m-%d'),
        y.index[-1].strftime('%Y-%m-%d')), fontsize=15)

    # Subplot 0 - Parameters
    [axs[0].plot(df_params.index, df_params.iloc[:, i], label=df_params.columns[i])
     for i in range(df_params.columns.size)]
    axs[0].set_ylabel('Factor Coefficient')
    axs[0].set_title('Rolling Factor Parameters')
    axs[0].legend()

    # Subplot 1 - t-stats
    [axs[1].plot(df_tstats.index, df_tstats.iloc[:, i], label=df_tstats.columns[i])
     for i in range(df_tstats.columns.size)]
    axs[1].set_ylabel('Factor t-stats')
    axs[1].set_title('Rolling Factor t-stats')
    axs[1].legend()
    axs[1].axhline(y=2, linestyle='dashed', c='gray', alpha=0.5)
    axs[1].axhline(y=-2, linestyle='dashed', c='gray', alpha=0.5)

    # Subplot 2 - adj-R square
    axs[2].plot(series_r2_adj.index, series_r2_adj.values, label=series_r2_adj.name)
    axs[2].set_ylabel('Adjusted R-square')
    axs[2].set_title('Rolling Adjusted R-square')

    fig.tight_layout()
    plt.show()

    return df_params, df_tstats, series_r2_adj


def factor_decomp(y, x, print_summary=True):
    x = sm.add_constant(x)
    res = sm.OLS(y, x).fit()
    if print_summary:
        print(res.summary())

    params = pd.Series(res.params.values, index=x.columns, name=y.name)
    tstats = pd.Series(res.tvalues.values, index=x.columns, name=y.name)
    r2_adj = res.rsquared_adj

    return params, tstats, r2_adj


def cross_val_plot(X, y, mod_type='Lasso', normalize=False,
                   alpha_space=np.logspace(-4, 0, 50), cv=10):
    """
    The function plots the CV score in a predefined alpha space for Lasso
    and Ridge Regression
    ------
    X: np.array (T x k)
    y: np.array (T x 1)
    mod_type: str, 'Lasso'/'Ridge'/'LinearRegression'
    alpha_space: default, np.logspace(-4, 0, 50) for alpha parameter in the
        model
    cv: int, N-fold cross validation
    """
    # Define the model
    if mod_type == 'Lasso':
        mod = Lasso(normalize=normalize)
    elif mod_type == 'Ridge':
        mod = Ridge(normalize=normalize)
    elif mod_type == 'LinearRegression':
        mod = LinearRegression(normalize=normalize)
    else:
        raise Exception('Undefined Model Type')

    # Fit the model and store the coefficients
    mod_scores = []
    mod_scores_std = []

    for alpha in alpha_space:
        if mod_type != 'LinearRegression':
            mod.alpha = alpha
        mod_cv_scores = cross_val_score(mod, X, y, cv=cv)
        mod_scores.append(np.mean(mod_cv_scores))
        mod_scores_std.append(np.std(mod_cv_scores))

    # Plot CV scores with std
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alpha_space, mod_scores)

    std_error = mod_scores_std / np.sqrt(cv)

    ax.fill_between(alpha_space, mod_scores + std_error, mod_scores -
                    std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(mod_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()


def feature_importance_plot(X, y, feature_names, lasso_alpha, normalize=False):
    """
    The function plots the feature importance using the lasso regression
    coefficient.
    """
    lasso = Lasso(alpha=lasso_alpha, normalize=normalize)
    lasso_coef = lasso.fit(X, y).coef_
    lasso_r2 = round(lasso.score(X, y), 2)

    plt.figure()
    plt.bar(range(len(feature_names)), lasso_coef)
    plt.xticks(range(len(feature_names)), feature_names, rotation=60)
    plt.axhline(y=0, c='gray', linestyle='dashed', alpha=0.5)
    plt.ylabel('Coefficients')
    plt.xlabel('Features')
    plt.title('Feature Importance with Lasso Regression (alpha={}, '
              'r_square={})'.format(lasso_alpha, lasso_r2))
    plt.show()


def hierarchical_clustering(df, method='complete',
                            ts=True, period=999,
                            freq='monthly',
                            title='Hierarchical Clustering'):
    """
    The function plots the hierarchical clustering of different managers or
    assets
    ----
    df: pd.DataFrame (T x N),
        where T is the number of observations at different time periods,
        N is the number of managers or assets.
        *It will be transposed to fit the scipy model
    method: see scipy.cluster.hierarchy 'method'
    title: str
    """

    df = ut.period_adj(df, period, freq)
    df = ut.first_common_date(df)[1]
    dt_str = df.index[0].strftime('%Y-%m') + ' - ' + df.index[-1].strftime(
        '%Y-%m')

    df_transpose = df.T
    mergings = linkage(df_transpose, method=method)

    plt.figure(figsize=(10, 8))
    dendrogram(mergings, labels=df.columns, leaf_rotation=60,
               leaf_font_size=10)
    plt.title(title, fontsize=12)
    plt.show()
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from cycler import cycler

from dateutil.relativedelta import relativedelta
from tabulate import tabulate

import utils as ut


class PerfMetrics:
    """
    Calculate a variety of performance metrics for a given portfolio series
    """

    def __init__(self, mgr, bmk, rf, freq, period, trading_day=True):
        """
        ---Inputs---
        :param mgr: Series, (T x ), single asset/mgr time series return
        :param bmk: Series, (T x ), bmk time series return
        :param rf: Series, (T x ), rf time series return
        :param freq: str, 'monthly' or 'daily'
        :param period: int or tuple,
                        1) int, recent n years
                        2) tuple, (start_date, end_date)
        :param trading_day: boolean, True or False;
                see ut.freq_adj() parameter description.
        """
        self.metrics_list = ['Annual Return', 'Annual Vol', 'Alpha',
                             "Jenson's Alpha",
                             'Beta',
                             'Upside Capture', 'Downside Capture',
                             'Skewness', 'Kurtosis', 'Max Drawdown',
                             'VaR', 'CVaR', 'Sharpe Ratio',
                             'Infor Ratio', 'Calmar Ratio',
                             'Tracking Error', 'Yrs', 'Bmk']

        # Time period adjustments
        df_raw = pd.concat([mgr, bmk, rf], axis=1)
        df_raw_adj = ut.period_adj(df_raw, period, freq, trading_day)
        _, df_new = ut.first_common_date(df_raw_adj)

        self.mgr = df_new.iloc[:, 0]
        self.bmk = df_new.iloc[:, 1]
        self.rf = df_new.iloc[:, 2]
        self.freq = freq
        self.Q = ut.freq_adj(freq, trading_day)
        self.yrs = np.round(self.mgr.size / self.Q, 2)

        if mgr.shape != bmk.shape or mgr.shape != rf.shape:
            raise Exception('Error! Inputs shape not compatible')

        if self.mgr.size < 2:
            raise Exception('Error! Mgr has only one data point.')

    def holding_period_ret(self):
        """Holding period return"""
        return np.prod(self.mgr + 1) - 1

    def cum_ret(self, x=1):
        """Cumulative Return (Growth of X)"""
        dates = self.mgr.index.to_pydatetime()
        if self.freq == 'monthly':
            dates = np.insert(dates, 0, dates[0] - relativedelta(months=1))
        elif self.freq == 'daily':
            dates = np.insert(dates, 0, dates[0] - relativedelta(days=1))
        else:
            pass

        # index is dates + 1 (for starting X)
        ret_cum = pd.Series(np.nan, index=dates)
        for i in range(ret_cum.size):
            if i == 0:
                ret_cum.iloc[i] = x
            else:
                ret_cum.iloc[i] = ret_cum.iloc[i - 1] * (
                        1 + self.mgr.iloc[i - 1])

        return ret_cum

    def ann_ret(self):
        """Cumulative annualized gross return - CAGR"""
        return (self.holding_period_ret() + 1) ** (1 / (
                self.mgr.size / self.Q)) - 1

    def ann_vol(self):
        """Annualized Volatility"""
        return np.std(self.mgr, ddof=1) * np.sqrt(self.Q)

    def alpha(self):
        """Direct Alpha / Excess Return"""
        ret_mgr = np.prod(self.mgr + 1) ** (1 / (self.mgr.size / self.Q)) - 1
        ret_bmk = np.prod(self.bmk + 1) ** (1 / (self.bmk.size / self.Q)) - 1
        return ret_mgr - ret_bmk

    def ann_downside_vol(self):
        """Annualized Downside Deviation / Volatility"""
        mgr_down = self.mgr[self.mgr < 0]
        return np.std(mgr_down, ddof=1) * np.sqrt(self.Q)

    def single_index_model(self):
        """Return alpha and beta of single_index_model"""
        if self.mgr.size < 2:
            raise Exception('Error! Single data point for single_index_model')
        res = sm.OLS(self.mgr, sm.add_constant(self.bmk)).fit()
        return res.params

    def up_down_capture(self, rolling_yr=999):
        """
        Calculate the upside capture for a rolling Q years
        --- INPUTS ---
        mgr: Series
        bmk: Series
        rolling_yr: Number of years for calculation
            *999 means since inception

        --- OUTPUTS ---
        upside_capture, downside_capture, ratio
        """
        df_raw = pd.concat([self.mgr, self.bmk], axis=1)
        df_new = ut.first_common_date(df_raw)[1]
        mgr_new, bmk_new = df_new.iloc[:, 0], df_new.iloc[:, 1]

        if rolling_yr == 999:
            rolling_yr = mgr_new.size / self.Q  # Since inception, no change
        else:
            mgr_new = mgr_new.iloc[-rolling_yr * self.Q:]
            bmk_new = bmk_new.iloc[-rolling_yr * self.Q:]

        bmk_up = bmk_new[bmk_new >= 0]
        mgr_up = mgr_new[bmk_new >= 0]

        if bmk_up.size != mgr_up.size:
            raise Exception('Size not Matching')

        bmk_down = bmk_new[bmk_new < 0]
        mgr_down = mgr_new[bmk_new < 0]

        if bmk_down.size != mgr_down.size:
            raise Exception('Size not Matching')

        up_capture = ((holding_period_ret(mgr_up) + 1) **
                      (1 / rolling_yr) - 1) / (
                             (holding_period_ret(bmk_up) + 1) ** (
                                 1 / rolling_yr) - 1)

        if holding_period_ret(mgr_down) == -1:
            down_capture = np.nan
            up_down_ratio = np.nan
        else:
            down_capture = ((holding_period_ret(mgr_down) + 1) **
                            (1 / rolling_yr) - 1) / (
                                   (holding_period_ret(bmk_down) + 1) **
                                   (1 / rolling_yr) - 1)

            up_down_ratio = up_capture / down_capture

        return up_capture, down_capture, up_down_ratio

    def skewness(self):
        """
        Fisher-Pearson coefficient of skewness: 0 for normal distribution
        """
        return skew(self.mgr)

    def kurt(self):
        """
        Fisher Kurtosis: 0 for normal distribution
        """
        return kurtosis(self.mgr)

    def max_drawdown(self):
        p = self.cum_ret(x=1)
        return (p / p.expanding(min_periods=0).max()).min() - 1

    def top_n_drawdowns(self):
        p = self.cum_ret(x=1)
        dd = p / p.expanding(min_periods=0).max() - 1
        return dd

    def var(self, xi=0.05):
        var_xi = np.percentile(self.mgr, xi * 100.0)
        if self.freq == 'monthly':
            return var_xi
        elif self.freq == 'daily':
            return var_xi * 21
        elif self.freq == 'quarterly':
            return var_xi / 4
        else:
            return np.nan

    def cvar(self, xi=0.05):
        var_mgr = self.var(xi=xi)
        pct = sum(self.mgr <= var_mgr) / self.mgr.size
        y = self.mgr * (self.mgr <= var_mgr)
        cvar_mgr = 1.0 / xi * (y.mean() + var_mgr * (xi - pct))
        if self.freq == 'monthly':
            return cvar_mgr
        elif self.freq == 'daily':
            return cvar_mgr * 21
        elif self.freq == 'quarterly':
            return cvar_mgr / 4
        else:
            return np.nan

    def sharpe_ratio(self):
        return (self.mgr.mean() - self.rf.mean()) / self.mgr.std(
            ddof=1) * np.sqrt(self.Q)

    def tracking_error(self):
        return np.std(self.mgr - self.bmk, ddof=1) * np.sqrt(self.Q)

    def infor_ratio(self):
        return self.alpha() / self.tracking_error()

    def calmar_ratio(self):
        return self.ann_ret() / abs(self.max_drawdown())

    def metrics(self):
        data = pd.Series(np.nan, index=self.metrics_list, name=self.mgr.name)

        data.iloc[0] = round(self.ann_ret(), 4)
        data.iloc[1] = round(self.ann_vol(), 4)
        data.iloc[2] = round(self.alpha(), 4)
        data.iloc[3] = round(self.single_index_model()[0] * self.Q, 4)
        data.iloc[4] = round(self.single_index_model()[1], 4)
        data.iloc[5] = round(self.up_down_capture()[0], 4)
        data.iloc[6] = round(self.up_down_capture()[1], 4)
        data.iloc[7] = round(self.skewness(), 4)
        data.iloc[8] = round(self.kurt(), 4)
        data.iloc[9] = round(self.max_drawdown(), 4)
        data.iloc[10] = round(self.var(), 4)
        data.iloc[11] = round(self.cvar(), 4)
        data.iloc[12] = round(self.sharpe_ratio(), 4)
        data.iloc[13] = round(self.infor_ratio(), 4)
        data.iloc[14] = round(self.calmar_ratio(), 4)
        data.iloc[15] = round(self.tracking_error(), 4)
        data.iloc[16] = round(self.yrs, 2)
        data.iloc[17] = self.bmk.name
        return data


class PortPerfMetrics(PerfMetrics):
    def __init__(self, wgt, df_mgr, bmk, rf, freq, trading_day, period):
        """
        Calculate the portfolio performance metrics.
        ---Inputs---
        :param wgt: Series, (N x ), portfolio weight
        :param df_mgr: DataFrame, (T x N), panel data of N managers or assets
        :param bmk: Series, (T x ), bmk time series return
        :param rf: Series, (Tx ), rf time series return
        :param freq: str, 'monthly' or 'daily'
        :param trading_day: boolean, True or False;
                see ut.freq_adj() parameter description.
        :param period: int or tuple,
                        1) int, recent n years
                        2) tuple, (start_date, end_date)
        """
        # Calculate portfolio time series return
        port = pd.Series(df_mgr @ wgt, name='Port Ret')
        super().__init__(port, bmk, rf, freq, trading_day, period)
        self.wgt = wgt
        self.df_mgr = df_mgr
        self.port = port

    def infor_ratio_exp(self, ret_exp):
        """
        Use the expected return to calculate expected information ratio.
        :param ret_exp: Series, (N x ), manager/asset expected return
        :return: infor_ratio_exp, float
        """
        return (self.wgt.T @ ret_exp - self.bmk.mean() * self.Q) / \
                self.tracking_error()


def sep_perf_metrics(df_mgr, bmk, rf, freq, trading_day, period):
    output_list = []
    for i in range(df_mgr.columns.size):
        output_list.append(
            PerfMetrics(df_mgr.iloc[:, i], bmk, rf,
                        freq, trading_day, period).metrics())
    df_res = pd.concat(output_list, axis=1).round(2)
    return df_res


def df_perf_metrics(df, bmk_str, rf_str, freq='monthly',
                    period=999, trading_day=True,
                    cols_incl=7,
                    monetary_regime_str='NA'):
    """
    Calculate the performance metrics for a DataFrame of portfolio returns
    :param df: df, (N x m)
    :param bmk_str: str, the column name of bmk
    :param rf_str: str, the column name of rf
    :param freq: str, 'monthly' or 'daily'
    :param period: int/tuple, recent n years / (start_date, end_date)
    :param trading_day: boolean, True or False, True 252 days, False 365 days
    :param cols_incl: int, number of columns included in the calculation
    :param monetary_regime_str: str, 'NA'/'Easing'/'Tightening'/'Flat'
           *Note: requires an additional column in df to use this
           classification tool
    :return: df_res:
    """
    if monetary_regime_str == 'NA':
        pass
    elif monetary_regime_str != 'NA':
        if monetary_regime_str not in ['Easing', 'Tightening', 'Flat']:
            raise Exception('Error! Undefined monetary regime.')
        else:
            df = df[df['Monetary_Regime'] == monetary_regime_str]

    output_list = []
    for i in df.iloc[:, :cols_incl].columns:
        print(i)
        col = df.loc[:, i]
        rf = df[rf_str]
        bmk = df[bmk_str]
        output_list.append(PerfMetrics(col, bmk, rf, freq,
                                       period, trading_day).metrics())
    df_res = pd.concat(output_list, axis=1).round(2)
    return df_res


def perf_scatter(res_str, legend_str, color_str, axs_str, col_str, marker_str,
                 xlim, ylim):
    """
    Plot the return-volatility scatter plot and up-down capture scatter
    plot using performance metrics of different regimes

    ---Inputs---
    res_str: list of performance metrics results for comparison
            (full sample, easing, tightening, flat)
    legend_str = ['Full Sample', 'Easing', 'Tightening', 'Flat']
    color_str = ['black', 'g', 'r', 'grey']
    axs_str = [(0,0), (0,1), (1,0), (1,1)] # subplots
    col_str: list of columns of performance metrics results (indices, mgrs)
    marker_str: markers of the columns (indices, mgrs)
    xlim: tuple (a,b) - subplots
    ylim: tuple (a,b) - subplots

    """

    # Check shape
    if len(res_str) == len(legend_str) == len(color_str) == len(axs_str):
        pass
    else:
        raise Exception(
            'Error! Lengths of regime types lists are not compatible!')

    if len(col_str) != len(marker_str):
        raise Exception('Error! Column numbers are not compatible')

    # Return-Vol Scatter Plot
    plt.figure(figsize=(10, 8))
    [plt.scatter(eval(res_str[i]).loc['Annual Vol', col_str[j]],
                 eval(res_str[i]).loc['Annual Return', col_str[j]],
                 label=eval(res_str[i])[col_str[j]].name, marker=marker_str[j],
                 c=color_str[i], s=100) for i in range(len(res_str)) for j in
     range(len(col_str))]
    patches = [mpatches.Patch(color=color_str[i], label=legend_str[i]) for i in
               range(len(res_str))]
    leg_1 = plt.legend(handles=patches, loc='best')
    lines = [mlines.Line2D([], [], color='black', marker=marker_str[j],
                           markersize=10, label=col_str[j], linestyle='None')
             for j in
             range(len(col_str))]
    plt.legend(handles=lines, loc='center right')
    plt.axhline(y=0, color='grey', linestyle='dashed', alpha=0.5)
    plt.gca().add_artist(leg_1)
    plt.xlabel('Vol')
    plt.ylabel('Return')
    plt.title('Return-Vol Scatter Plot under Different Monetary Regimes',
              fontsize=15)
    plt.show()

    # Upside-Downside Capture Subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    [axs[axs_str[i]].scatter(
        eval(res_str[i]).loc['Downside Capture', col_str[j]],
        eval(res_str[i]).loc['Upside Capture', col_str[j]],
        label=eval(res_str[i])[col_str[j]].name, marker=marker_str[j],
        c=color_str[i], s=100) for i in range(len(res_str)) for j in
        range(len(col_str))]

    [axs[axs_str[i]].axhline(y=1, color='grey', linestyle='dashed', alpha=0.5)
     for i in range(len(res_str))]
    [axs[axs_str[i]].axvline(x=1, color='grey', linestyle='dashed', alpha=0.5)
     for i in range(len(res_str))]
    [axs[axs_str[i]].set_xlim(xlim[0], xlim[1]) for i in range(len(res_str))]
    [axs[axs_str[i]].set_ylim(ylim[0], ylim[1]) for i in range(len(res_str))]
    [axs[axs_str[i]].set_xlabel('Downside Capture') for i in
     range(len(res_str))]
    [axs[axs_str[i]].set_ylabel('Upside Capture') for i in range(len(res_str))]
    [axs[axs_str[i]].legend(
        handles=[mpatches.Patch(color=color_str[i], label=legend_str[i])]) for
        i in range(len(res_str))]

    lines = [mlines.Line2D([], [], color='black', marker=marker_str[j],
                           markersize=10, label=col_str[j], linestyle='None')
             for j in range(len(col_str))]
    fig.legend(handles=lines, loc='lower left', bbox_to_anchor=(0.32, 0.01),
               borderaxespad=0)
    fig.suptitle('Up-Down Capture under Different Monetary Regimes',
                 fontsize=15)
    fig.tight_layout(rect=(0.05, 0.15, 0.90, 0.95))

    return


def port_ret(wgt, df_ret, freq, trading_day=True):
    """
    Calculate the portfolio return based on weight and return, where freq is
    for data frequency adjustments.
    *Note: if ret is annualized expected return (e.g. ret_bl), freq = 'yearly'
    """
    q_factor = ut.freq_adj(freq, trading_day)
    return wgt.T @ df_ret * q_factor


def port_vol(wgt, cov_hist, freq, trading_day=True):
    """
    Calculate the portfolio volatility based on weight and covariance
    matrix, where freq is for data frequency adjustments.
    e.g. monthly: *np.sqrt(12)
    """
    q_factor = ut.freq_adj(freq, trading_day)
    return np.sqrt(wgt.T @ cov_hist @ wgt) * np.sqrt(q_factor)


def holding_period_ret(ret_series):
    """Holding period return"""
    return np.prod(ret_series + 1) - 1


def rolling_ret(ret_series, period, freq, trading_day=True):
    q_factor = ut.freq_adj(freq, trading_day)
    T = int(period * q_factor)
    return (ret_series.rolling(T).apply(
        holding_period_ret, raw=False) + 1) ** (1 / period) - 1


def rolling_alpha(ret_series, bmk, period, freq, trading_day=True):
    return rolling_ret(ret_series, period, freq, trading_day) - \
           rolling_ret(bmk, period, freq, trading_day)


def cum_ret(ret_series, freq, x=1):
    """Cumulative Return (Growth of X)"""
    dates = ret_series.index.to_pydatetime()
    if freq == 'monthly':
        dates = np.insert(dates, 0, dates[0] - relativedelta(months=1))
    elif freq == 'daily':
        dates = np.insert(dates, 0, dates[0] - relativedelta(days=1))
    else:
        raise Exception('Error! Data frequency not monthly or daily')

    # index is dates + 1 (for starting X)
    ret_cum = pd.Series(np.nan, index=dates)
    for i in range(ret_cum.size):
        if i == 0:
            ret_cum.iloc[i] = x
        else:
            ret_cum.iloc[i] = ret_cum.iloc[i - 1] * (
                    1 + ret_series.iloc[i - 1])
    return ret_cum


def df_cum_ret(ret_df, freq, x=1):
    _, ret_df_common = ut.first_common_date(ret_df)
    df_output = []
    for i in range(ret_df_common.columns.size):
        ret_i = ret_df_common.iloc[:, i]
        ret_cum_i = cum_ret(ret_i, freq, x=x)
        df_output.append(ret_cum_i)
    df_output = pd.concat(df_output, axis=1, join='outer')
    df_output.columns = ret_df.columns
    return df_output


def forward_looking_alpha(mgr, bmk, rolling_year, pred_year_list,
                          freq='monthly', plot=True, table=True):
    """
    Calculate the forward-looking alpha conditional on the rolling alpha

    ### INPUTS ###
    1. mgr: (T x 1) return series
    2. bmk: (T x 1) return series
    3. rolling_year: rolling_year
    4. pred_year_list: list e.g. [0.5, 1, 3]

    ### OUTPUTS ###
    df_merged: future alpha & counts
    """
    # global T, df, box, rolling_alpha_t, rank_t, x
    df_raw = pd.concat([mgr, bmk], axis=1)
    dt_0, df = ut.first_common_date(df_raw)
    T, _ = df.shape
    df_plot = pd.DataFrame(np.nan, index=df.index,
                           columns=['alpha', 'mean', 'mean+std', 'mean+2std',
                                    'mean-std', 'mean-2std'])
    df_plot.iloc[:, 0] = rolling_alpha(df.iloc[:, 0], df.iloc[:, 1],
                                       rolling_year, freq)

    # global box, df_R, x
    frame, frame_grouped = [], []
    for k in pred_year_list:
        box = []
        if 12 * (rolling_year + 2) + k * 12 >= T:
            continue

        for t in range(int(12 * (rolling_year + 2)), T):
            rolling_alpha_t = rolling_alpha(df.iloc[:t + 1, 0],
                                            df.iloc[:t + 1, 1],
                                            rolling_year, freq)
            rank_t, alpha_t, mean_t, std_t = ut.last_day_class(rolling_alpha_t)
            if k == pred_year_list[0]:
                df_plot.iloc[t, 1] = mean_t
                df_plot.iloc[t, 2] = mean_t + std_t
                df_plot.iloc[t, 3] = mean_t + 2 * std_t
                df_plot.iloc[t, 4] = mean_t - std_t
                df_plot.iloc[t, 5] = mean_t - 2 * std_t

            if (t >= 12 * (rolling_year + 2)) and (
                    t + k * 12 <= T - 1):
                # higher than rolling years + 2
                box.append([rank_t, rolling_alpha(
                    df.iloc[:, 0], df.iloc[:, 1], k, freq).iloc[
                    int(t + k * 12)]])
            else:
                box.append([rank_t, np.nan])

            # Note: We leave an additional month gap for
            # implementation as it's supposed to be .iloc[int(t+k*12-1)]
        df_box_k = pd.DataFrame(box)
        frame.append(df_box_k)
        df_box_k.columns = ['rank', str(k) + ' years']
        df_box_k_grouped = df_box_k.groupby('rank').agg(
            ['mean', 'count']).sort_index(ascending=False)
        frame_grouped.append(df_box_k_grouped)
    df_merged = pd.concat(frame, axis=1, join='outer')
    df_merged_grouped = pd.concat(frame_grouped, axis=1,
                                  join='outer').sort_index(ascending=False)

    # Plot the alpha
    if plot is True:
        plt.figure(figsize=(10, 8))
        [plt.plot(df_plot.iloc[:, i], label=df_plot.iloc[:, i].name) for i in
         range(df_plot.columns.size)]
        plt.scatter(df_plot.index[-1], df_plot.alpha[-1], s=50, color='red',
                    label='Current Rank: ' + str(int(df_merged.iloc[-1, 0])))
        plt.legend()
        plt.title('Rolling ' + str(rolling_year)
                  + ' year Alpha Chart - Expanding Mean & Std - '
                  + mgr.name + ' vs ' + bmk.name)

    # Print the merged table
    if table is True:
        print('Conditional Forward-Looking Alphas - ' + mgr.name + ' vs '
              + bmk.name + ' (Rolling ' + str(rolling_year) + '-years)')
        print(tabulate(df_merged_grouped, headers=df_merged_grouped.columns,
                       tablefmt='github'))
        print(mgr.name + ' vs ' + bmk.name + ' Current Rank: ' + str(
            int(df_merged.iloc[-1, 0])))
    return df_merged, df_merged_grouped, df_plot


def bulk_compute_forward_looking_alpha(Source_EQ, Source_Code,
                                       sub_sector_str_list, rolling_year,
                                       pred_year_list, freq='monthly',
                                       sep_bmk=False):
    """
    This function calculates each manager's forward alpha in the source equity
    matrix.
    ### INPUTS ###
    Source_EQ: Source_AEQ/ Source_IEQ/ Segments of AEQ or IEQ
    Source_Code: Source_Code matrix with five columns
    (Mgr, Sector, Sub_Sector, sep_bmk, Agg_bmk)
    sub_sector_str_list: e.g. ['AEQ LC', 'AEQ SC']
    rolling_year: rolling years for alpha calculation
    pred_year_list: list of forward-looking years for alpha calculation
    e.g. [0.5, 1,3]

    ### OUTPUTS ###
    print df_merged_grouped for each manager
    plot df_plot for each manager
    """
    df_alpha_list, df_alpha_merged_list, df_alpha_plot_list = [], [], []
    mgr_str = list(ut.sub_sector_mgr_str(Source_Code, sub_sector_str_list))
    EQ = Source_EQ[mgr_str]
    for i in range(EQ.columns.size):
        mgr_i = EQ.iloc[:, i]
        if not sep_bmk:
            # If False -> Use aggregate benchmark for the selected managers
            bmk_str = \
                Source_Code[Source_Code.Mgr == mgr_str[i]]['Agg_bmk'].iloc[0]
        elif sep_bmk:
            bmk_str = \
                Source_Code[Source_Code.Mgr == mgr_str[i]]['Sep_bmk'].iloc[0]
        else:
            raise Exception('Error! Parameter sep_bmk is not boolean.')

        bmk_i = Source_EQ[bmk_str]

        print(mgr_i.name + ' (' + bmk_str + ')')
        df_alpha_i, df_alpha_merged_i, df_alpha_plot_i = forward_looking_alpha(
            mgr_i, bmk_i, rolling_year, pred_year_list, freq)
        df_alpha_list.append(df_alpha_i)
        df_alpha_merged_list.append(df_alpha_merged_i)
        df_alpha_plot_list.append(df_alpha_plot_i)

    return df_alpha_list, df_alpha_merged_list, df_alpha_plot_list


# Alpha Correlation Heatmap
def alpha_corr_heatmap(Source_EQ, Source_Code, sub_sector_str_list, yr=999,
                       sep_bmk=False):
    '''
    Purpose: Calculate the alpha correlation for the given matrix & plot heatmap

    ### INPUTS ###
    Source_EQ: (T x k) return matrix  e.g. Source_AEQ/Source_IEQ
    Source_Code
    sub_sector_str_list: list of sub_sectors to be selected
    yr: default to be 999, include the full history with common dates
        if not 0, will be defined to be the latest N years e.g. 3 -> last 3 years
    Sep_bmk: default to be False, use Agg_bmk
        if True, use Sep_bmk

    ### OUTPUT ###
    Alpha_Corr_matrix
    Plot the heatmap
    '''

    df = ut.sub_sector_df(Source_EQ, Source_Code, sub_sector_str_list)

    if yr == 999:  # Full history involved
        EQ = df
    else:  # Latest N years
        EQ = df.iloc[-yr * 12:, :]

    T = EQ.index.size
    dt_str = EQ.index[0].strftime('%Y-%m') + ' - ' + EQ.index[-1].strftime(
        '%Y-%m')
    # Find the agg/sep benchmark of the selected sub-sector
    mgr_str = EQ.columns

    if not sep_bmk:
        # Use aggregate benchmark for the selected managers
        bmk_str = Source_Code[Source_Code.Mgr.isin(list(mgr_str))][
            'Agg_bmk'].unique()
    else:
        bmk_str = Source_Code[Source_Code.Mgr.isin(list(mgr_str))][
            'Sep_bmk'].unique()
        if bmk_str.size > 1:
            raise Exception(
                'Error! No unified separate benchmark for the selected '
                'mgrs')

    corr_matrix = EQ.subtract(Source_EQ[str(bmk_str[0])].iloc[-T:],
                              axis=0).corr()

    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=(10, 8))
    plt.title("Alpha Correlation Matrix - against " + str(
        bmk_str[0]) + ' (' + dt_str + ')')
    plt.tight_layout()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix.round(2), mask=mask, annot=True, cmap=cmap, vmax=1,
                vmin=-1, center=0, square=True, linewidths=.5,
                cbar_kws={"shrink": .5})
    return corr_matrix


def corr_heatmap(df, bmk=None, period=999, freq='monthly', title=None,
                 trading_day=True):
    """
    :param df: DataFrame of Returns
    :param bmk: Series of benchmark returns -> If not None, calculate alpha
                correlation over benchmark returns
    :return: corr_matrix: Correlation Matrix
    """

    if bmk is not None:
        df = df.subtract(bmk, axis=0)

    df = ut.period_adj(df, period, freq, trading_day)
    df = ut.first_common_date(df)[1]
    dt_str = df.index[0].strftime('%Y-%m') + ' - ' + df.index[-1].strftime(
        '%Y-%m')

    corr_matrix = df.corr()
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=(10, 8))

    if title is None:
        if bmk is not None:
            plt.title("Alpha Correlation Matrix - against " + bmk.name +
                      ' (' + dt_str + ')')
        else:
            plt.title("Correlation Matrix " + ' (' + dt_str + ')')
    else:
        plt.title(title)

    plt.tight_layout()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix.round(2), mask=mask, annot=True, cmap=cmap, vmax=1,
                vmin=-1, center=0, square=True, linewidths=.5,
                cbar_kws={"shrink": .5})
    return corr_matrix


def pairplot(df, tail_classify=False, title='Pair Plot', **kwargs):
    """
    Draw the pair plot with kde on the diagonal
    ---------------
    df: Dataframe with columns
    tail_classify: default=False;
        optional argument:
        * if False, no tail specification required;
        * if True, specify tail_classify = tuple(threshold, mkt_col_str,
                                                 classified_str)
            e.g., tail_threshold=(-0.02, 'SR Median', [])
        the dataframe would be reclassified into tail months and non-tail
        months, and draw the colors differently.
    title: Figure title
    **kwargs: all key word arguments passed directly to sns.pairplot function
    """

    df_common = ut.first_common_date(df)[1]
    if tail_classify is not False:
        threshold = tail_classify[0]
        mkt_col_str = tail_classify[1]
        classified_str = tail_classify[2]
        df_common.loc[df_common.loc[:, mkt_col_str] < threshold, 'Type'] = \
            classified_str[0]
        df_common.loc[df_common.loc[:, mkt_col_str] >= threshold, 'Type']\
            = \
            classified_str[1]

        hue = 'Type'
    else:
        hue = False

    # plt.figure(figsize=(10, 8))
    pp = sns.pairplot(df_common, diag_kind='kde', hue=hue, **kwargs)
    plt.suptitle(title, fontsize=15)
    return


def cum_ret_chart(df, period=999, freq='monthly', x=1, trading_day=True):
    df = ut.period_adj(df, period, freq, trading_day)
    df_cum = df_cum_ret(df, freq, x)
    dt_str = df_cum.index[0].strftime('%Y-%m') + ' - ' + df_cum.index[
        -1].strftime('%Y-%m')

    c_cycle = cycler(color=['b', 'g', 'r', 'y', 'c', 'm', 'k'])
    l_cycle = cycler(linestyle=['-', '--', ':', '-.'])
    cycle = l_cycle * c_cycle

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_prop_cycle(cycle)
    # fig.rc('axes', prop_cycle=cycle)
    for i in range(df_cum.columns.size):
        ax.plot(df_cum.iloc[:, i], label=df_cum.columns[i])
    ax.legend()
    ax.set_title('Cumulative Return Chart (' + dt_str + ')')
    return df_cum


def ret_vol_scatter(pm_df):
    # Only take annual return & annual vol rows
    df = pm_df.loc[['Annual Return', 'Annual Vol', 'Yrs'], :].T
    labels = [df.index[i] + ' (' + str(df['Yrs'][i]) + ' Yrs)'
              for i in range(df.index.size)]

    plt.figure(figsize=(10, 8))
    [plt.scatter(df.iloc[i, 1], df.iloc[i, 0], label=labels[i]) for i in \
     range(df.index.size)]

    plt.xlabel('Annual Vol')
    plt.ylabel('Annual Return')
    plt.legend()
    plt.title('Return-Volatility Scatter Plot')
    return

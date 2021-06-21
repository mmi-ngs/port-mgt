import os
import numpy as np
import pandas as pd
import math
import functools
import matplotlib
from dateutil.relativedelta import relativedelta


def my_date_parser(x):
    """change date-like index to datetime strings"""
    return pd.to_datetime(x).strftime('%Y%m%d')


def first_common_date(df):
    """
    Purpose: get the first common date with all assets having a valid return/
        wgt/target item
    Input: df: DataFrame (index: dates; column: assets)
    Outputs:
        1) dt: first common date (datetime)
        2) df_new: new df starting with the first common date (DataFrame)
        3) i: int
    """
    n = 0
    for i in range(df.index.size):
        if df.iloc[i, :].isnull().any() == False:
            n = i
            break
    dt = pd.to_datetime(df.index[n])
    df_new = df.iloc[n:, :]
    return dt, df_new


def freq_adj(freq, trading_day=True):
    """
    Define the frequency factor used to calculate annualized return, vol, cov.
    Inputs:
    freq: str, 'monthly' or 'daily'
    trading_day: boolean, True or False, used for 'daily' freq only.
        if True: q_factor = 252;
        if False: q_factor = 365.
    Outputs:
    q: int, 252 or 12
    """
    if freq == 'monthly':
        q_factor = 12
    elif freq == 'daily':
        if trading_day:
            q_factor = 252
        else:
            q_factor = 365
    elif freq == 'yearly':
        q_factor = 1
    elif freq == 'quarterly':
        q_factor = 4
    else:
        raise Exception('Error! Undefined frequency.')
    return q_factor


def period_adj(df, period, freq, trading_day=True):
    """
    Make time period adjustments
    ---Inputs---
    :param df: df/series, (T x N) index must be datetime, columns: assets
    :param period: int or tuple
        1) int: the most recent x years
        2) tuple: (start_date, end_date)
    :param freq: str, 'monthly'/'daily'/'yearly'
    :param trading_day: boolean, True or False, used for 'daily' freq only.
        if True: q_factor = 252;
        if False: q_factor = 365.
    ---Outputs---
    :return: df_adj
    """
    Q = freq_adj(freq, trading_day)
    if type(df) is pd.DataFrame:
        if type(period) is int:
            df_adj = df.iloc[int(-period * Q):, :]
        elif type(period) is tuple:
            start = pd.to_datetime(period[0])
            end = pd.to_datetime(period[1])
            df_adj = df.loc[start:end, :]
        else:
            raise Exception(
                'TypeError! Parameter period is neither int or tuple.')
    elif type(df) is pd.Series:
        if type(period) is int:
            df_adj = df.iloc[int(-period * Q):]
        elif type(period) is tuple:
            start = pd.to_datetime(period[0])
            end = pd.to_datetime(period[1])
            df_adj = df.loc[start:end]
        else:
            raise Exception(
                'TypeError! Parameter period is neither int or tuple.')
    else:
        raise Exception(
            'TypeError! Parameter df is neither DataFrame or Series.')
    return df_adj


def mth_start_to_end(df):
    """Fix month-beginning date index to month-end"""
    df.index = df.index.to_pydatetime() + relativedelta(months=1) - \
        relativedelta(days=1)
    return


def moving_avg(ret_series, n_periods):
    """Calculate the moving average of a return series"""
    return ret_series.rolling(window=n_periods).mean()


def last_day_class(rolling_alpha):
    """
    Classify the latest rolling alpha into 6 ranges

    INPUT: rolling-alpha series

    OUTPUTS:
    rank: int, range 1- range 6
    last_alpha: the latest N-yr rolling alpha
    mean, std: expanding window historical mean & std
    """
    last_alpha = rolling_alpha.iloc[-1]  # latest rolling_alpha
    mean = rolling_alpha.mean()
    std = rolling_alpha.std(ddof=1)

    if last_alpha >= mean + std * 2:
        rank = 6
    elif mean + std <= last_alpha < mean + std * 2:
        rank = 5
    elif mean <= last_alpha < mean + std:
        rank = 4
    elif mean - std < last_alpha < mean:
        rank = 3
    elif mean - std * 2 < last_alpha <= mean - std:
        rank = 2
    elif last_alpha <= mean - std * 2:
        rank = 1
    else:
        rank = np.nan
    return rank, last_alpha, mean, std


def sub_sector_mgr_str(Source_Code, sub_sector_str_list):
    """Return the list of mgr names (str) in the specified sub_sector"""
    return Source_Code[Source_Code.Sub_Sector.isin(sub_sector_str_list)]['Mgr']


def sub_sector_df(Source_EQ, Source_Code, sub_sector_str_list):
    """
    Purpose: Get the df (T x k) of mgrs selected for a specified sub-sector
    """
    mgr_str_list = sub_sector_mgr_str(Source_Code, sub_sector_str_list)
    _, df = first_common_date(Source_EQ[mgr_str_list])
    return df


def print_trade_analysis(analyzer):
    """
    Function to print the Technical Analysis results in a nice format.
    """
    # Get the results we are interested in
    total_open = analyzer.total.open

    try:
        total_closed = analyzer.total.closed
    except KeyError:
        raise Exception('Insufficient closed deals for analysis.')

    total_won = analyzer.won.total
    total_lost = analyzer.lost.total
    win_streak = analyzer.streak.won.longest
    lose_streak = analyzer.streak.lost.longest
    pnl_net = round(analyzer.pnl.net.total, 2)
    strike_rate = round((total_won / total_closed) * 100, 2)

    # Designate the rows
    h1 = ['Total Open', 'Total Closed', 'Total Won', 'Total Lost']
    h2 = ['Strike Rate', 'Win Streak', 'Losing Streak', 'PnL Net']
    r1 = [total_open, total_closed, total_won, total_lost]
    r2 = [strike_rate, win_streak, lose_streak, pnl_net]

    # Check which set of headers is the longest.
    if len(h1) > len(h2):
        header_length = len(h1)
    else:
        header_length = len(h2)

    # Print the rows
    print_list = [h1, r1, h2, r2]
    row_format = "{:<15}" * (header_length + 1)
    print("Trade Analysis Results:")
    for row in print_list:
        print(row_format.format("", *row))


def fx_pair_switch(fx_str):
    """Switch the AUD/USD into USD/AUD or any other string patterns"""
    a, b = fx_str.split('/')
    fx_rev = b + '/' + a
    return fx_rev


def matplotlib_backend_selection():
    import matplotlib
    gui_env = ['TKAgg', 'GTKAgg', 'Qt4Agg', 'WXAgg']
    for gui in gui_env:
        try:
            print("testing", gui)
            matplotlib.use(gui, warn=False, force=True)
            from matplotlib import pyplot as plt
            break
        except:
            continue
    print("Using:", matplotlib.get_backend())
    return


def multi_df_merge(df_list):
    """
    The function merges many data frames into one based on similar indices
    """
    return functools.reduce(lambda left, right: pd.merge(left, right,
                                                         left_index=True,
                                                         right_index=True,
                                                         how='outer'), df_list)


def compile_excel(df, filename, path=None):
    """
    The function to store data (can be large) into excel file
    * If the dataframe is big, it will be fitted into several sheets in one file
    Variables:
    ---------
    df: original dataframe to be stored into excel
    filename: str.xlsx
    path: default wording directory, stored path if specified
        e.g. '.\\Projects\\Market Microstructure'
    """
    path_cwd = os.getcwd()
    if path is None:
        pass
    else:
        os.chdir(path)

    # Excel max row: 1048576
    if df.index.size < 1048570:
        df.to_excel("{}.xlsx".format(filename))
    else:
        writer = pd.ExcelWriter("{}.xlsx".format(filename))
        for i in range(math.ceil(df.index.size / 1000000)):
            print("{}/{}".format(i, int(math.ceil(df.index.size / 1000000)-1)))
            df_i = df.iloc[int(1000000 * i):int(1000000 * (i + 1)), :]
            df_i.to_excel(writer, sheet_name='sheet_{}'.format(i))
            writer.save()

    # Change back to its current working directory after storage
    os.chdir(path_cwd)
    return


def mem_usage_check(df):
    """
    Check the memory usage of a pd.DataFrame
    """
    print('dense : {:0.2f} bytes'.format(df.memory_usage().sum() / 1e3))


def nested_dict_to_df(input_dict, orient='columns'):
    """
    Convert a nested dictionary to pd.DataFrame.

    :param input_dict: [dict] a nested dictionary
    :param orient: [str] 'index'/'column', define the orientation of the
                   returned DataFrame
    :return: df: [pd.DataFrame] a multi-index DataFrame
    """
    df = pd.DataFrame.from_dict(
        {(i, j): input_dict[i][j] for i in input_dict.keys()
         for j in input_dict[i].keys()}, orient=orient)
    return df


def random_rgba(alpha=1):
    """
    Create an autocycler based on the RGB color tuple rules.

    :return: rgba color tuple, e.g., (0.1, 0.2, 0.5, 0.3).
    """
    r = np.random.random()
    g = np.random.random()
    b = np.random.random()
    rgba = (r, g, b, alpha)
    return rgba


def cmap_rgba(cmap, n_colors):
    """
    Use matplotlib.cm.get_cmap to get cmap and use evenly spaced
    color on the color line.

    :param: cmap [str] matplotlib cmap name, e.g., 'rainbow'
    :param: n_colors [int] number of colors needed
    :return: rgba_list [list] a list of rgba colors selected from rainbow cmap
    """
    x = matplotlib.cm.get_cmap(cmap)
    n = np.linspace(0, 1, n_colors)
    rgba_list = []
    for i in n:
        rgba_list.append(x(i))
    return rgba_list

import numpy as np
import pandas as pd
from numpy.linalg import inv
import scipy.optimize as sco
from scipy.stats import norm
from mixedvines.mixedvine import MixedVine
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

import perf_metrics as pm
import utils as ut


# noinspection PyTypeChecker
class BlackLitterman:
    """
    Black-Litterman Optimisation Model
    ### Inputs ###
    df_hist:       df; historical return df
                    (index: date, columns: assets)
    wgt_eql:        df; target weight or equilibrium weight
                    (default to be wgt_current)
    df_view:        df (N x 2); view return and view confidence df
                    (index: assets, columns: ['View', 'Confidence of View'])
                    *Note: Leave as 0 if there is no view.
    freq:           str; 'monthly' or 'daily'
    risk_aversion:  int; default = 3.07
    wgt_ex:         series; similar to wgt_eql, but like wgt_em for ieq and
                            wgt_sc for aeq
    max_ex:         float: define a max_ex for ieq, max_sc for aeq
    period:         int or tuple:
                        1) int: the most recent x years
                        2) tuple: (start_date, end_date)

    """

    def __init__(self, df_hist, bmk, rf, wgt_eql, df_view,
                 wgt_ex=None, max_ex=None, period=999,
                 freq='monthly', risk_aversion=3.07, tau=0.25):

        # Test data types and data size consistency
        if (df_hist.index.size != rf.size) | (df_hist.index.size != bmk.size):
            print('Error! The length of historical return DataFrame is '
                  'different from the length of rf or bmk')
            raise Exception('Exit')

        if type(df_view) != pd.DataFrame:
            print('Error!: The view input is not a DataFrame')
            raise Exception('Exit')

        if np.count_nonzero(df_view.iloc[:, 0]) != \
                np.count_nonzero(df_view.iloc[:, 1]):
            print('Error!: The number of views and confidence of views are '
                  'not matching! Please re-enter the views')
            raise Exception('exit')

        # Data frequency adjustment
        self.freq = freq
        self.Q = ut.freq_adj(freq)

        # Data Period adjustment
        self.period = period
        self.df_hist = ut.period_adj(df_hist, period, freq)
        self.bmk = ut.period_adj(bmk, period, freq)
        self.rf = ut.period_adj(rf, period, freq)
        self.wgt_ex = wgt_ex
        self.max_ex = max_ex

        # Calc after period adjustment
        self.ret_hist = self.df_hist.mean() * self.Q
        self.df_hist_ex = self.df_hist.subtract(self.rf, axis=0)
        self.cov_hist = self.df_hist_ex.cov() * self.Q

        # Variables declaration
        self.wgt_eql = wgt_eql
        self.df_view = df_view
        self.N = df_hist.columns.size
        self.mgr_list = df_hist.columns
        self.A = risk_aversion
        self.tau = tau

        # Define view vector, view confidence vector, P matrix,
        # view covariance matrix
        self.k = np.count_nonzero(df_view.iloc[:, 0])
        self.q_view = df_view.iloc[:, 0]
        self.view_list = self.q_view[self.q_view != 0]
        view_conf = df_view.iloc[:, 1]

        self.q_view_conf = pd.DataFrame(0, index=df_hist.columns, columns=[
                'View ' + str(x + 1) for x in range(self.k)])
        for x in range(self.k):
            self.q_view_conf.loc[self.view_list.index[x], 'View ' + str(x+1)] \
                = view_conf[view_conf.index == self.view_list.index[x]].iloc[0]

        self.p_matrix = pd.DataFrame(0, index=['View ' + str(x+1) for x in
                                     range(self.k)], columns=df_hist.columns)
        for x in range(self.k):
            self.p_matrix.loc['View '+str(x+1), self.view_list.index[x]] = 1
        self.omega_error = self.get_omega

        # BL return
        self.ret_bl_active = self.get_ret_bl[0]
        self.ret_bl = self.get_ret_bl[1]

    @property
    def get_omega(self):
        """Calculate the view covariance matrix - omega_error"""
        if self.k == 0:
            exit()

        ret_eql = np.zeros((self.N, ))
        omega_error = pd.DataFrame(0, index=['View ' + str(x + 1) for x in
                                             range(self.k)], columns=[
            'View ' + str(x + 1) for x in range(self.k)])

        for i in range(self.k):
            p_k = np.array(self.p_matrix.iloc[i, :]).reshape((1, self.N))
            q_k = self.view_list.iloc[i]
            ret_k_100 = np.array(ret_eql).reshape((self.N, 1)) + \
                self.tau * self.cov_hist @ p_k.T @ inv(
                p_k * self.tau @ self.cov_hist @ p_k.T) @ \
                (q_k - p_k @ np.array(ret_eql).reshape((self.N, 1)))
            wgt_k_100 = inv(self.A * self.cov_hist) @ ret_k_100
            d_k_100 = wgt_k_100 - np.array(self.wgt_eql).reshape((self.N, 1))
            c_k = self.q_view_conf.iloc[:, i]
            tilt_k = np.asarray(d_k_100).reshape(-1) * c_k
            wgt_k_pct = np.array(self.wgt_eql + tilt_k).reshape((self.N, 1))
            ret_eql_reshaped = np.array(ret_eql).reshape((self.N, 1))

            # Get the w_k (k-th element in the diagonal of omega_error)
            args = (self.A, self.cov_hist, self.tau, p_k, ret_eql_reshaped,
                    q_k, wgt_k_pct, self.N)
            ini_guess = np.diag(omega_error)[i] + 0.0000000001
            bounds = [(0, None)]
            res_k = sco.minimize(self.min_squared_diff, x0=ini_guess,
                                 args=args,
                                 bounds=bounds, method='SLSQP',
                                 options={'maxiter': 10000, 'disp': True})
            omega_error.iloc[i, i] = res_k.x[0]
        return omega_error

    @staticmethod
    def min_squared_diff(omega, risk_aversion, cov_hist, tau, p, r_eql, q,
                         wgt_pct, n):
        """Objective function to calculate view covariance matrix -
        omega_error"""
        wgt_k = inv(risk_aversion * cov_hist) @ inv(inv(tau * cov_hist) +
                                                    p.T * (1 / omega) * p) @ (
                    np.reshape(inv(tau * cov_hist) @ r_eql, (n, 1)) + p.T * (
                        1 / omega) * q)
        var_diff = np.square(np.array(wgt_pct) - wgt_k).sum()
        return var_diff

    @property
    def get_ret_bl(self):
        """Get Black-Litterman Return Vector"""
        ret_active = self.cov_hist @ self.p_matrix.T @ \
            inv(np.array(self.omega_error) / self.tau + self.p_matrix @
                self.cov_hist @ self.p_matrix.T) @ np.array(self.view_list)
        ret_bl = ret_active + self.bmk.mean() * self.Q
        return ret_active, ret_bl

    def port_opt(self, how, wgt_min, wgt_max, period=3, bounded=False,
                 pairplot=False, args='default', te=0.025):
        """
        Perform three portfolio optimisation tasks after calculate BL
        return
        --- Inputs ---
        how: str,
            'Max IR', 'Max SR', 'Min cVaR'
        wgt_min: Series (N x )
        wgt_max: Series (N x )
        freq: 'monthly'
        period: int, default = 3
        bounded: boolean, default = False
        args: str/tuple, (if args=False, use default settings; else set args)
        """
        # Period adjustment: Use the information within period for optimisation
        if (type(period) is int) & (period > self.period):
            raise Exception(
                'The period is less than the BlackLitterman class period')

        df_hist_adj = ut.period_adj(self.df_hist, period, self.freq)
        bmk_adj = ut.period_adj(self.bmk, period, self.freq)
        rf_adj = ut.period_adj(self.rf, period, self.freq)

        ini_guess = self.wgt_eql
        if self.wgt_ex is not None:
            cons = ({'type': 'eq', 'fun': lambda x: x.sum() - 1},
                    {'type': 'ineq', 'fun': lambda x: x - wgt_min},
                    {'type': 'ineq', 'fun': lambda x: wgt_max - x},
                    {'type': 'ineq', 'fun': lambda x: pm.PortPerfMetrics(
                               x, df_hist_adj, bmk_adj, rf_adj,
                               self.freq, period).tracking_error() - te},
                    {'type': 'ineq', 'fun': lambda x: self.max_ex -
                     x.T @ self.wgt_ex})
        else:
            cons = ({'type': 'eq', 'fun': lambda x: x.sum() - 1},
                    {'type': 'ineq', 'fun': lambda x: x - wgt_min},
                    {'type': 'ineq', 'fun': lambda x: wgt_max - x},
                    {'type': 'ineq', 'fun': lambda x: pm.PortPerfMetrics(
                               x, df_hist_adj, bmk_adj, rf_adj,
                               self.freq, period).tracking_error() - te})
        bounds = tuple((self.wgt_eql.iloc[x]-0.03, self.wgt_eql.iloc[
                 x]+0.03) for x in range(self.N))
        if args == 'default':
            args = (self.ret_bl, df_hist_adj, bmk_adj, rf_adj,
                    self.freq, period)
        else:
            args = args

        if bounded is False:    # Unbounded Optimisation
            if how == 'Max IR':
                # noinspection PyTypeChecker
                res_max_ir = sco.minimize(self.neg_ir_exp,
                                          x0=ini_guess, args=args,
                                          constraints=cons,
                                          method='SLSQP',
                                          options={'maxiter': 10000,
                                                   'disp': True})
                wgt_opt = res_max_ir.x
                perf_metrics_opt = pm.PortPerfMetrics(wgt_opt, df_hist_adj,
                                                      bmk_adj, rf_adj,
                                                      self.freq,
                                                      period).metrics()

            elif how == 'Max SR':
                res_max_sr = sco.minimize(self.neg_sr_exp,
                                          x0=ini_guess, args=args,
                                          constraints=cons,
                                          method='SLSQP',
                                          options={'maxiter': 10000,
                                                   'disp': True})
                wgt_opt = res_max_sr.x
                perf_metrics_opt = pm.PortPerfMetrics(wgt_opt, df_hist_adj,
                                                      bmk_adj, rf_adj,
                                                      self.freq,
                                                      period).metrics()

            elif how == 'Min cVaR':
                # Fit and Simulation on historical data
                is_coutinous = [True] * self.N
                vine = MixedVine.fit(np.array(df_hist_adj), is_coutinous)

                # Incorporate views
                vine_exp = vine
                vol_hist = df_hist_adj.std(ddof=1) * np.sqrt(self.Q)
                for i in range(self.N):
                    vine_exp.set_marginal(i, norm(
                        self.ret_bl.iloc[i]/self.Q,
                        vol_hist.iloc[i]/np.sqrt(self.Q)))
                samples_exp = vine_exp.rvs(size=10000)
                samples_exp_df = pd.DataFrame(samples_exp,
                                              columns=df_hist_adj.columns)
                if pairplot is False:
                    pass
                else:
                    sns.pairplot(samples_exp_df, diag_kind='kde')

                # Min cVaR
                args_cvar = (samples_exp_df, 0.05)
                res_min_cvar = sco.minimize(self.port_cvar_mc,
                                            x0=ini_guess, args=args_cvar,
                                            constraints=cons,
                                            method='SLSQP',
                                            options={'maxiter': 10000,
                                                     'disp': True})
                wgt_opt = res_min_cvar.x
                perf_metrics_opt = pm.PortPerfMetrics(wgt_opt, df_hist_adj,
                                                      bmk_adj, rf_adj,
                                                      self.freq,
                                                      period).metrics()
            else:
                raise Exception('Error! Undefined optimization method.')
        elif bounded is True:  # Bounded Optimisation
            if how == 'Max IR':
                res_max_ir = sco.minimize(self.neg_ir_exp,
                                          x0=ini_guess, args=args,
                                          constraints=cons,
                                          bounds=bounds,
                                          method='SLSQP',
                                          options={'maxiter': 10000,
                                                   'disp': True})
                wgt_opt = res_max_ir.x
                perf_metrics_opt = pm.PortPerfMetrics(wgt_opt, df_hist_adj,
                                                      bmk_adj, rf_adj,
                                                      self.freq,
                                                      period).metrics()
            elif how == 'Max SR':
                res_max_sr = sco.minimize(self.neg_sr_exp,
                                          x0=ini_guess, args=args,
                                          constraints=cons,
                                          bounds=bounds,
                                          method='SLSQP',
                                          options={'maxiter': 10000,
                                                   'disp': True})
                wgt_opt = res_max_sr.x
                perf_metrics_opt = pm.PortPerfMetrics(wgt_opt, df_hist_adj,
                                                      bmk_adj, rf_adj,
                                                      self.freq,
                                                      period).metrics()
            elif how == 'Min cVaR':
                # Fit and Simulation on historical data
                is_coutinous = [True] * self.N
                vine = MixedVine.fit(np.array(df_hist_adj), is_coutinous)

                # Incorporate views
                vine_exp = vine
                vol_hist = df_hist_adj.std(ddof=1) * np.sqrt(self.Q)
                for i in range(self.N):
                    vine_exp.set_marginal(i, norm(
                        self.ret_bl.iloc[i]/self.Q,
                        vol_hist.iloc[i]/np.sqrt(self.Q)))
                samples_exp = vine_exp.rvs(size=10000)
                samples_exp_df = pd.DataFrame(samples_exp,
                                              columns=df_hist_adj.columns)
                if pairplot is False:
                    pass
                else:
                    sns.pairplot(samples_exp_df, diag_kind='kde')

                # Min cVaR
                args_cvar = (samples_exp_df, 0.05)
                res_min_cvar = sco.minimize(self.port_cvar_mc,
                                            x0=ini_guess, args=args_cvar,
                                            constraints=cons,
                                            bounds=bounds,
                                            method='SLSQP',
                                            options={'maxiter': 10000,
                                                     'disp': True})
                wgt_opt = res_min_cvar.x
                perf_metrics_opt = pm.PortPerfMetrics(wgt_opt, df_hist_adj,
                                                      bmk_adj, rf_adj,
                                                      self.freq,
                                                      period).metrics()
            else:
                raise Exception('Error! Undefined optimization method.')
        else:
            raise Exception('Error! Wrong argument for parameter bounded.')
        wgt_opt = pd.Series(wgt_opt, index=self.mgr_list)
        return perf_metrics_opt, wgt_opt

    def port_opt_comparison(self, wgt_min, wgt_max, period=3, plot=True,
                            te=0.025, rf_exp=0.0025):
        """
        Compile the results from three optimisation process and make
        comparisons by drawing plots on wgt and perf_metrics
        """
        bmk_adj = ut.period_adj(self.bmk, period, self.freq)

        res_max_IR_unbounded = self.port_opt('Max IR', wgt_min, wgt_max,
                                             period=period, bounded=False,
                                             te=te)
        res_max_SR_unbounded = self.port_opt('Max SR', wgt_min, wgt_max,
                                             period=period, bounded=False,
                                             te=te)
        res_min_cvar_unbounded = self.port_opt('Min cVaR', wgt_min, wgt_max,
                                               period=period, bounded=False,
                                               te=te)
        res_max_IR_bounded = self.port_opt('Max IR', wgt_min, wgt_max,
                                           period=period, bounded=True, te=te)
        res_max_SR_bounded = self.port_opt('Max SR', wgt_min, wgt_max,
                                           period=period, bounded=True, te=te)
        res_min_cvar_bounded = self.port_opt('Min cVaR', wgt_min, wgt_max,
                                             period=period, bounded=True,
                                             te=te)
        cols = ['Max IR Unbounded', 'Max SR Unbounded', 'Min cVaR Unbounded',
                'Max IR Bounded', 'Max SR Bounded', 'Min cVaR Bounded']

        df_perf_metrics_opt = pd.concat([res_max_IR_unbounded[0],
                                        res_max_SR_unbounded[0],
                                        res_min_cvar_unbounded[0],
                                        res_max_IR_bounded[0],
                                        res_max_SR_bounded[0],
                                        res_min_cvar_bounded[0]], axis=1)
        df_perf_metrics_opt.columns = cols

        df_wgt_opt = pd.concat([res_max_IR_unbounded[1],
                                res_max_SR_unbounded[1],
                                res_min_cvar_unbounded[1],
                                res_max_IR_bounded[1],
                                res_max_SR_bounded[1],
                                res_min_cvar_bounded[1]], axis=1)
        df_wgt_opt.columns = cols

        if plot is True:
            # 1. Weight Comparison - Unbounded and Bounded
            fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 8))
            ind = np.arange(self.N) * 2
            width = 0.35

            axs[0].set_title('Unbounded Optimisation')
            axs[0].set_xticks(ind)
            axs[0].set_xticklabels(self.mgr_list)
            axs[0].bar(ind - 1.5 * width, self.wgt_eql, width,
                       label='Current weight', color='gray')
            axs[0].bar(ind - 0.5 * width, df_wgt_opt.iloc[:, 0], width,
                       label=df_wgt_opt.iloc[:, 0].name, color='green')
            axs[0].bar(ind + 0.5 * width, df_wgt_opt.iloc[:, 1], width,
                       label=df_wgt_opt.iloc[:, 1].name, color='orange')
            axs[0].bar(ind + 1.5 * width, df_wgt_opt.iloc[:, 2], width,
                       label=df_wgt_opt.iloc[:, 2].name, color='red')
            axs[0].legend()

            axs[1].set_title('Bounded Optimisation')
            axs[1].set_xticks(ind)
            axs[1].set_xticklabels(self.mgr_list)
            axs[1].bar(ind - 1.5 * width, self.wgt_eql, width,
                       label='Current weight', color='gray')
            axs[1].bar(ind - 0.5 * width, df_wgt_opt.iloc[:, 3], width,
                       label=df_wgt_opt.iloc[:, 3].name, color='green')
            axs[1].bar(ind + 0.5 * width, df_wgt_opt.iloc[:, 4], width,
                       label=df_wgt_opt.iloc[:, 4].name, color='orange')
            axs[1].bar(ind + 1.5 * width, df_wgt_opt.iloc[:, 5], width,
                       label=df_wgt_opt.iloc[:, 5].name, color='red')
            axs[1].legend()

            fig.suptitle('Optimised Weight Comparison: Current | Max IR | '
                         'Max SR | Min cVaR', fontsize=20)
            fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
            plt.setp(axs[0].get_xticklabels(), rotation=30,
                     horizontalalignment='right', fontsize='small')
            plt.setp(axs[1].get_xticklabels(), rotation=30,
                     horizontalalignment='right', fontsize='small')

            # 2. Efficient Frontier - Unbounded and Bounded
            con_efficient_frontier_exp = \
                self.constrained_efficient_frontier(wgt_min, wgt_max,
                                                    period=period)

            # Separate unbounded and bounded (order follows cols above)
            colors = ['green', 'orange', 'red', 'green', 'orange', 'red']
            alphas = [1, 1, 1, 0.5, 0.5, 0.5]

            plt.figure(figsize=(10, 8))
            plt.plot(con_efficient_frontier_exp.iloc[:, 1],
                     con_efficient_frontier_exp.iloc[:, 0],
                     label='Expected Efficient Frontier')
            [plt.scatter(
                pm.port_vol(df_wgt_opt.iloc[:, i], self.cov_hist, 'yearly'),
                pm.port_ret(df_wgt_opt.iloc[:, i], self.ret_bl, 'yearly'),
                s=250, marker='*', alpha=alphas[i], c=colors[i],
                label=df_wgt_opt.iloc[:, i].name)
                for i in range(len(cols))]
            plt.scatter(self.bmk.std(ddof=1)*np.sqrt(self.Q),
                        self.bmk.mean()*self.Q, s=200, c='red', marker='o',
                        label=self.bmk.name + ' - Historical ' +
                        str(round(self.bmk.size/self.Q, 2)) + ' years')
            plt.scatter(pm.port_vol(self.wgt_eql, self.cov_hist, 'yearly'),
                        pm.port_ret(self.wgt_eql, self.ret_bl, 'yearly'),
                        s=200, c='gray', marker='d',
                        label='Expected Return with Current Weight')
            plt.title('Optimised Risk-Return Comparison on the '
                      'Expected Efficient Frontier')
            plt.xlabel('Vol')
            plt.ylabel('E(R)')
            plt.legend()
        else:
            pass

        # Add expected performance measures
        df_perf_metrics_exp = pd.DataFrame(np.zeros((3, 6)),
                                           index=['Expected Return',
                                                  'Expected Vol',
                                                  'Expected Sharpe'],
                                           columns=cols)
        for i in range(len(cols)):
            df_perf_metrics_exp.iloc[0, i] = pm.port_ret(
                df_wgt_opt.iloc[:, i], self.ret_bl, 'yearly')
            df_perf_metrics_exp.iloc[1, i] = pm.port_vol(
                df_wgt_opt.iloc[:, i], self.cov_hist, 'yearly')
            df_perf_metrics_exp.iloc[2, i] = (df_perf_metrics_exp.iloc[0, i] -
                                              rf_exp)/df_perf_metrics_exp.iloc[
                1, i]
        return df_perf_metrics_opt, df_wgt_opt, df_perf_metrics_exp

    def constrained_efficient_frontier(self, wgt_min, wgt_max, period=3,
                                       te=0.025,
                                       risk_aversion_range=(-0.05, 100.05,
                                                            3)):
        """Calculate the constrained efficient frontier with unbounded wgt"""
        A_rand = np.arange(risk_aversion_range[0], risk_aversion_range[1],
                           risk_aversion_range[2])
        eff_port = pd.DataFrame(np.nan, index=np.arange(0, A_rand.size, 1),
                                columns=['Expected Ret', 'Std', 'Sharpe'])

        # Period adjustment
        df_hist_adj = ut.period_adj(self.df_hist, period, self.freq)
        bmk_adj = ut.period_adj(self.bmk, period, self.freq)
        rf_adj = ut.period_adj(self.rf, period, self.freq)

        df_hist_ex_adj = ut.period_adj(self.df_hist_ex, period, self.freq)
        cov_hist_adj = df_hist_ex_adj.cov() * self.Q

        ini_guess = self.wgt_eql
        cons = ({'type': 'eq', 'fun': lambda x: x.sum() - 1},
                {'type': 'ineq', 'fun': lambda x: x - wgt_min},
                {'type': 'ineq', 'fun': lambda x: wgt_max - x},
                {'type': 'ineq', 'fun': lambda x: pm.PortPerfMetrics(
                           x, df_hist_adj, bmk_adj, rf_adj,
                           self.freq, period).tracking_error() - te})
        for i in range(A_rand.size):
            args_i = (self.ret_bl, cov_hist_adj, A_rand[i])
            res_max_utility = sco.minimize(self.neg_quad_utility,
                                           x0=ini_guess, args=args_i,
                                           constraints=cons, method='SLSQP',
                                           options={'maxiter': 10000,
                                                    'disp': True})

            wgt_opt = res_max_utility.x
            eff_port.iloc[i, 0] = pm.port_ret(wgt_opt, self.ret_bl, 'yearly')
            eff_port.iloc[i, 1] = pm.port_vol(wgt_opt, cov_hist_adj,
                                              'yearly')
            eff_port.iloc[i, 2] = (eff_port.iloc[i, 0] - rf_adj.mean() *
                                   self.Q) / eff_port.iloc[i, 1]
        return eff_port

    def backtest(self, wgt_min, wgt_max, rolling_period=3, initial_period=5,
                 freq_rebal='quarterly', plot=True, how='Historical Mean',
                 wgt=None):

        start, df = ut.first_common_date(self.df_hist)
        bmk_adj = self.bmk.loc[start:]
        rf_adj = self.rf.loc[start:]

        T = df.shape[0]
        T_test = T - initial_period * self.Q
        period_test = int(T_test / self.Q)
        df_test = df.iloc[-T_test:, :]
        bmk_test = bmk_adj.iloc[-T_test:]
        rf_test = rf_adj.iloc[-T_test:]

        q_rebal = ut.freq_adj(freq_rebal)
        if initial_period <= rolling_period:
            raise Exception('Error! Leave a longer initial period, '
                            'cannot be less than rolling period')
        if (initial_period + rolling_period) > (T / self.Q):
            raise Exception('Error! Insufficient data points for backtest. '
                            'Adjust the period setting or Increase sample '
                            'size.')

        # Set up storage
        df_wgt_ir = pd.DataFrame(np.nan, columns=df.columns,
                                 index=df_test.index)
        df_wgt_sr = pd.DataFrame(np.nan, columns=df.columns,
                                 index=df_test.index)
        df_wgt_cvar = pd.DataFrame(np.nan, columns=df.columns,
                                   index=df_test.index)

        ret_ir = pd.Series(np.nan, index=df_test.index, name='Max IR')
        ret_sr = pd.Series(np.nan, index=df_test.index, name='Max SR')
        ret_cvar = pd.Series(np.nan, index=df_test.index, name='Min cVaR')

        # Calculate wgt_opt every rebalance period
        for i in range(T_test):
            df_roll = df.iloc[i:int(i+rolling_period*self.Q), :]
            bmk_roll = bmk_adj.iloc[i:int(i+rolling_period*self.Q)]
            rf_roll = rf_adj.iloc[i:int(i+rolling_period*self.Q)]

            df_expand = df.iloc[:int(i+initial_period*self.Q), :]
            bmk_expand = bmk_adj.iloc[:int(i+initial_period*self.Q)]
            rf_expand = rf_adj.iloc[:int(i+rolling_period*self.Q)]

            if i%q_rebal == 0:      # Rebalance Frequency
                # Setup forecast strategy for optimisation
                # self.forecast function require df_expand, bmk_expand
                args = (self.forecast(df_expand, bmk_expand, rf_expand,
                                      how=how, wgt=wgt,
                                      rolling_period=rolling_period),
                        df_roll, bmk_roll, rf_roll, self.freq, rolling_period)
                res_max_ir_i = self.port_opt('Max IR', wgt_min, wgt_max,
                                             period=rolling_period, args=args)
                res_max_sr_i = self.port_opt('Max SR', wgt_min, wgt_max,
                                             period=rolling_period, args=args)
                res_min_cvar_i = self.port_opt('Min cVaR', wgt_min, wgt_max,
                                               period=rolling_period,
                                               args=args)

                wgt_opt_ir_i = np.array(res_max_ir_i[1])
                wgt_opt_sr_i = np.array(res_max_sr_i[1])
                wgt_opt_cvar_i = np.array(res_min_cvar_i[1])

                df_wgt_ir.iloc[i, :] = wgt_opt_ir_i
                df_wgt_sr.iloc[i, :] = wgt_opt_sr_i
                df_wgt_cvar.iloc[i, :] = wgt_opt_cvar_i
            else:
                df_wgt_ir.iloc[i, :] = wgt_opt_ir_i
                df_wgt_sr.iloc[i, :] = wgt_opt_sr_i
                df_wgt_cvar.iloc[i, :] = wgt_opt_cvar_i

            # Calculate the portfolio return
            ret_ir.iloc[i] = wgt_opt_ir_i.T @ df_test.iloc[i, :]
            ret_sr.iloc[i] = wgt_opt_sr_i.T @ df_test.iloc[i, :]
            ret_cvar.iloc[i] = wgt_opt_sr_i.T @ df_test.iloc[i, :]

            # Performance metrics and wgt list
            df_output = pd.concat([ret_ir, ret_sr, ret_cvar], axis=1)
            df_perf_metrics = pm.sep_perf_metrics(df_output, bmk_test,
                                                  rf_test, self.freq,
                                                  period_test)
            df_wgt_list = [df_wgt_ir, df_wgt_sr, df_wgt_cvar]

        if plot is True:
            plt.figure(figsize=(10, 8))
            plt.title('Cumulative Return Comparison on Rebalancing Strategy - '
                      + how)
            plt.plot(pm.PerfMetrics(ret_ir, bmk_test, rf_test, self.freq,
                                    period_test).cum_ret(), label=ret_ir.name)
            plt.plot(pm.PerfMetrics(ret_sr, bmk_test, rf_test, self.freq,
                                    period_test).cum_ret(), label=ret_sr.name)
            plt.plot(pm.PerfMetrics(ret_cvar, bmk_test, rf_test, self.freq,
                                    period_test).cum_ret(),
                     label=ret_cvar.name)
            plt.plot(pm.PerfMetrics(bmk_test, bmk_test, rf_test, self.freq,
                                    period_test).cum_ret(),
                     label=bmk_test.name, color='red')
            plt.plot(pm.PerfMetrics(df_test @ self.wgt_eql, bmk_test,
                                    rf_test, self.freq,
                                    period_test).cum_ret(),
                     label='Current Wgt (Static)', color='gray')
            plt.legend()

            return df_perf_metrics, df_wgt_list

    def te_sensitivity(self, how, wgt_min, wgt_max, bounded=False, period=3,
                       te_list=None, print_table=True):

        if te_list is None:
            te_list = [1.5, 2, 2.5, 3, 3.5, 4]

        df_pm = []
        df_wgt = []
        for i in range(len(te_list)):
            df_pm_i, df_wgt_i = self.port_opt(how, wgt_min, wgt_max,
                                              period=period,
                                              bounded=bounded,
                                              te=te_list[i]/100)
            df_pm.append(df_pm_i)
            df_wgt.append(df_wgt_i)

        df_pm = pd.concat(df_pm, axis=1)
        df_wgt = pd.concat(df_wgt, axis=1)
        df_wgt = df_wgt.round(4)

        df_pm.columns = ['TE: {}%'.format(i) for i in te_list]
        df_wgt.columns = ['TE: {}%'.format(i) for i in te_list]

        if print_table:
            print(tabulate(df_pm, headers=df_pm.columns,
                           tablefmt='github'))
            print(tabulate(df_wgt, headers=df_pm.columns,
                           tablefmt='github'))

        return df_pm, df_wgt

    def plot_te_sensitivity(self, wgt_min, wgt_max, hows=None, bounded=False,
                            period=3, te_list=None, print_table=False):
        if te_list is None:
            te_list = [1.5, 2, 2.5, 3, 3.5, 4]

        if hows is None:
            hows = ['Max IR', 'Max SR', 'Min cVaR']
            colors = ['green', 'orange', 'red']

        if bounded:
            b_str = 'bounded'
        else:
            b_str = 'unbounded'

        df_pm = []
        df_wgt = []
        for i in range(len(hows)):
            df_pm_i, df_wgt_i = self.te_sensitivity(hows[i], wgt_min, wgt_max,
                                                    bounded=bounded,
                                                    period=period,
                                                    te_list=te_list,
                                                    print_table=print_table)
            df_pm.append(df_pm_i)
            df_wgt.append(df_wgt_i)

        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 8))
        fig.suptitle('Portfolio Comparison on Different Tracking Error '
                     'Levels ({})'.format(b_str), fontsize=15)

        for i in range(len(hows)):
            df_i = df_pm[i]
            for j in range(len(te_list)):
                axs[0].scatter(df_i.loc['Tracking Error', df_i.columns[j]],
                               df_i.loc['Annual Return', df_i.columns[j]],
                               label='{}({})'.format(
                                   hows[i], df_i.columns[j]),
                               marker='*', s=te_list[j] * 50, c=colors[i])
                axs[0].set_xlabel('Tracking Error')
                axs[0].set_ylabel('Annual Return')

                axs[1].scatter(df_i.loc['Annual Vol', df_i.columns[j]],
                               df_i.loc['Annual Return', df_i.columns[j]],
                               label='{}({})'.format(
                                   hows[i], df_i.columns[j]),
                               marker='*', s=te_list[j] * 50, c=colors[i])
                axs[1].set_xlabel('Annual Vol')
                axs[1].set_ylabel('Annual Return')

        axs[0].set_title('Annual Return vs Tracking Error')
        axs[1].set_title('Annual Return vs Annual Vol')
        axs[1].legend(framealpha=0.5)
        return df_pm, df_wgt

    @staticmethod
    def forecast(df_hist, bmk, rf, how, wgt=None, rolling_period=3,
                 freq='monthly'):
        Q = ut.freq_adj(freq)

        df_roll = ut.period_adj(df_hist, rolling_period, freq)
        bmk_roll = ut.period_adj(bmk, rolling_period, freq)
        rf_roll = ut.period_adj(rf, rolling_period, freq)

        if how == 'Historical Mean':
            ret_pred = df_roll.mean() * Q
        elif how == 'Mean Reversion':
            ret_pred = pd.Series(np.nan, index=df_hist.columns)
            ls_dis = [1.2, 1.1, 1.05, 0.95, 0.9, 0.8]
            for i in range(df_hist.columns.size):
                # Expanding df_hist and bmk
                rolling_alpha_i = pm.rolling_alpha(df_hist.iloc[:, i], bmk,
                                                   rolling_period, freq)
                class_i = ut.last_day_class(rolling_alpha_i)[0]
                if class_i is np.nan:
                    dis_i = np.nan
                else:
                    dis_i = ls_dis[int(class_i-1)]
                ret_pred.iloc[i] = df_roll.iloc[:, i].mean() * dis_i
        elif how == 'Historical Percentile':        # Wendy's approach -> BL
            df_view_pred = pd.DataFrame(0, index=df_hist.columns,
                                        columns=['Expected Alpha (1yr)',
                                                 'Confidence'])
            for i in range(df_hist.columns.size):
                rolling_alpha_i = pm.rolling_alpha(df_hist.iloc[:, i], bmk,
                                                   rolling_period,
                                                   freq).dropna()
                df_view_pred.iloc[i, 0] = np.percentile(rolling_alpha_i, 0.67)
                df_view_pred.iloc[i, 1] = 0.67
            ret_pred = BlackLitterman(df_roll, bmk_roll, rf_roll,
                                      wgt, df_view_pred).ret_bl
            print(ret_pred)
        else:
            ret_pred = None
        return ret_pred

    @staticmethod
    def neg_ir_exp(wgt, ret_exp, df_hist, bmk, rf,
                   freq, period):
        """
        Define the objective function to maximize information ratio based on
        new BL return.
        """
        port_ret_active_exp = wgt.T @ (ret_exp - bmk.mean() *
                                       ut.freq_adj(freq))
        te_hist = pm.PortPerfMetrics(wgt, df_hist, bmk, rf, freq,
                                     period).tracking_error()
        ir_exp = -port_ret_active_exp / te_hist
        return ir_exp

    @staticmethod
    def neg_sr_exp(wgt, ret_exp, df_hist, bmk, rf,
                   freq, period):
        """
        Define the objective function to maximize Sharpe ratio based on
        new BL return.
        """
        port_ret_ex_exp = wgt.T @ (ret_exp - rf.mean() *
                                   ut.freq_adj(freq))
        vol_hist = pm.PortPerfMetrics(wgt, df_hist, bmk, rf, freq,
                                      period).ann_vol()
        sr_exp = -port_ret_ex_exp / vol_hist
        return sr_exp

    @staticmethod
    def port_cvar_mc(wgt, df_sim, xi):
        """
        Monte-Carlo Simulation Min-CVaR objective function.
        (Alcock and Hatherley, 2009; formula 12)

        ---Inputs---
        :param wgt: Series (N x )
        :param df_sim: DataFrame (T x N), simulated panel data
        :param xi: flota, 0.05 / 0.01

        ---Output---
        :return: cvar_exp
        """
        port_sim = df_sim @ wgt
        alpha = -np.percentile(port_sim, xi * 100)      # VaR
        q, _ = df_sim.shape
        bracket = -df_sim @ wgt - alpha
        cvar_exp = alpha + 1/(q * (1-xi)) * sum(bracket * (bracket > 0))
        return cvar_exp

    @staticmethod
    def neg_quad_utility(wgt, ret_exp, cov_hist, risk_aversion):
        U = -(wgt.T @ ret_exp - 0.5 * risk_aversion * wgt.T @ cov_hist @ wgt)
        return U

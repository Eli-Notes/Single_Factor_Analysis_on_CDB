import pandas as pd
import numpy as np
import datetime
from scipy.stats import spearmanr
import statsmodels.api as sm

import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.rcParams['figure.figsize'] = (8, 4)
#plt.style.use("ggplot")

class output_container():
    def __init__(self):
        pass

class Factor(object):


    def __init__(
            self,
            df_conn = None, # TradeDate, AssetID, factor, Price
            price_mode = None, # 'yield_in_pct'
    ):
        # set factor and price
        if (df_conn is not None) and (price_mode is not None):
            self.set_factor_and_price(df_conn, price_mode)
        
        # returns
        self.df_return = None
        self.return_horizon = None
        self.return_mode = None
        # groups
        self.df_groups = None
        self.df_group_factor = None
        self.df_group_return = None
        # ic
        self.ic_series = None
        self.rank_ic_series = None


    def set_factor_and_price( # Done
            self,
            df_conn,
            price_mode,
    ):
        # check name
        cols = ['Trade_Date','Asset_ID','Price','Factor']
        if set(df_conn.columns) != set(cols):
            raise ValueError("Columns should be " + '、'.join(cols) + '.')
        # check type
        df_conn['Trade_Date'] = pd.to_datetime(df_conn['Trade_Date'])
        if df_conn['Asset_ID'].dtype != 'str':
            df_conn['Asset_ID'] = df_conn['Asset_ID'].astype('str')
        if df_conn['Price'].dtype != 'float':
            df_conn['Price'] = df_conn['Price'].astype('float')
        if df_conn['Factor'].dtype != 'float':
            df_conn['Factor'] = df_conn['Factor'].astype('float')
        # sort
        df_conn = df_conn.sort_values(by=['Trade_Date'])
        # save
        self.df_conn = df_conn
        self.df_factor = df_conn.pivot(index='Trade_Date',columns='Asset_ID',values='Factor')
        self.df_price = df_conn.pivot(index='Trade_Date',columns='Asset_ID',values='Price')
        self.price_mode = price_mode
        # return what?


    def summarize_factor( # Done
            self,
            hist_bound = None, # e.g. [0, 100] or None
            hist_bins = 10,
    ):
        des = self.df_factor.describe()
        hist_data = self.df_factor.values.squeeze()
        hist_data = hist_data[~np.isnan(hist_data)]
        if hist_bound is not None:
            hist_data = hist_data[(hist_data>=hist_bound[0]) & (hist_data<=hist_bound[-1])]
        fig, ax = plt.subplots(1, 1)
        ax.hist(hist_data, bins=hist_bins)
        ax.set_ylabel("Frequence")
        ax.set_xlabel("Factor")
        fig.tight_layout()
        return des, fig


    def ic_analysis_all(
            self,
            return_horizon = [0, 5],
            return_mode = 'spread_in_pct',
            min_valid_obs = 5,
    ):
        # para
        self.return_mode = return_mode
        self.return_horizon = return_horizon
        # factor and return
        df_factor = self.df_factor.copy()
        df_return = self.get_return(return_horizon, return_mode)
        # filter by minimum valid obs
        date_index = [idx for idx in df_factor.index \
            if len(df_factor.loc[idx].dropna()) >= min_valid_obs]
        self.df_factor = df_factor.loc[date_index]
        self.df_return = df_return.loc[date_index]
        # ic
        ic_series, rank_ic_series = self.ic_calculator(df_factor, df_return)
        ic_series = ic_series.dropna()
        rank_ic_series = rank_ic_series.dropna()
        df_ic = pd.concat([ic_series, rank_ic_series], axis=1)
        df_ic.columns = ['ic','rank_ic']
        # description of ic
        des_ic = df_ic.describe()
        # plot acf
        fontsize = 10
        fig_ic_acf, ax = plt.subplots(1, 2)
        sm.graphics.tsa.plot_acf(ic_series.values.squeeze(), ax=ax[0], lags=40)
        ax[0].set_title('ACF of IC series', fontsize = fontsize)
        sm.graphics.tsa.plot_acf(rank_ic_series.values.squeeze(), ax=ax[1], lags=40)
        ax[1].set_title('ACF of Rank IC series', fontsize = fontsize)
        fig_ic_acf.tight_layout()
        # plot hist
        fontsize = 10
        bins = 20
        fig_ic_hist, ax = plt.subplots(1, 2)
        ax[0].hist(ic_series.values, bins=bins)
        ax[0].set_title('Histogram of IC', fontsize = fontsize)
        ax[0].grid()
        ax[1].hist(rank_ic_series.values, bins=bins)
        ax[1].set_title('Histogram of Rank IC', fontsize = fontsize)
        ax[1].grid()
        fig_ic_hist.tight_layout()
        # output
        output = output_container()
        output.ic_dataframe = df_ic
        output.ic_description = des_ic
        output.fig_ic_acf = fig_ic_acf
        output.fig_ic_hist = fig_ic_hist
        return output

    def construct_groups(
            self,
            return_horizon = [0, 5],
            return_mode = 'spread_in_pct',
            min_valid_obs = 10,
            group_by = 'quantiles',
            quantile_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            aggregate_factor_by = 'median',
            aggregate_return_by = 'median',
    ):
        # para
        self.return_mode = return_mode
        self.return_horizon = return_horizon
        # generate df_return
        df_factor = self.df_factor.copy()
        df_return = self.get_return(return_horizon, return_mode)
        # filter by minimum valid obs
        date_index = [idx for idx in df_factor.index \
            if len(df_factor.loc[idx].dropna()) >= min_valid_obs]
        self.df_factor = df_factor.loc[date_index]
        self.df_return = df_return.loc[date_index]
        # generate df_groups
        self.df_groups = self.group_factor(group_by, quantile_edges)
        # aggregate
        self.df_group_factor = self.aggregate_factor(aggregate_factor_by)
        self.df_group_return = self.aggregate_return(aggregate_return_by, return_horizon, return_mode)
        # output
        #output = output_container()
        #output.df_group_factor = self.df_group_factor
        #output.df_group_return = self.df_group_return
        #return output


    def plot_group_cum_return(self, title='', fig_size=[15, 5]):
        if self.df_group_return is not None:
            L = self.df_group_return.shape[1]
            plt.rcParams['figure.figsize'] = (fig_size[0], fig_size[1] * L + fig_size[1])
            fig, ax = plt.subplots(L+1, 1)
            col_names = self.df_group_return.columns
            if self.return_mode == 'spread_in_pct':
                cum = self.df_group_return.cumsum()
                self.plot_groups(ax[0], cum, title + "; Cumulative Return")
                for i in range(L):
                    col_name = col_names[i]
                    value = cum - cum.loc[:,[col_name]*len(cum.columns)].values
                    title2 = title + "; All groups' cumulative return subtracted by group " + str(int(col_name))
                    self.plot_groups(ax[i+1], value, title2)
                fig.tight_layout()
            else:
                print('return_mode not defined for this plot.')
            return fig
        else:
            raise ValueError("Group returns are not ready, please fit.")


    def group_factor(self, group_by, quantile_edges):
        """ group factor """
        print('Grouping factor...')
        df_factor = self.df_factor.copy()
        if group_by == 'quantiles':
            df_edges = df_factor.quantile(quantile_edges, axis=1).transpose()
            df_groups = self.group_by_edges(df_factor, df_edges)
        else:
            raise ValueError("Wrong group method, please check.")
        print("Done!")
        return df_groups


    def aggregate_factor(self, aggregate_by):
        """ get combined factor for groups """
        print('Aggregating factor...')
        df_factor = self.df_factor.copy()
        df_groups = self.df_groups.copy()
        df_group_factor = self.aggregate(df_factor, df_groups, aggregate_by)
        print("Done!")
        return df_group_factor


    def aggregate_return(self, aggregate_by, return_horizon, return_mode):
        """ get combined return for groups """
        print("Done!")
        print("Aggregating returns...")
        df_return = self.df_return.copy()
        df_groups = self.df_groups.copy()
        df_group_return = self.aggregate(df_return, df_groups, aggregate_by)
        # replace nan at the left
        def replace_nan(series_old):
            series = series_old.copy()
            last_ele = np.nan
            for idx in reversed(series.index):
                if pd.isnull(series[idx]):
                    series[idx] = last_ele
                else:
                    last_ele = series[idx]
            return series
        df_group_return = df_group_return.apply(replace_nan, axis=1)
        print("Done!")
        return df_group_return


    def get_return(self, return_horizon, return_mode):
        print("Generating returns...")
        price_mode = self.price_mode
        df_price = self.df_price
        price_end = df_price.shift(periods=-return_horizon[-1])
        price_bgn = df_price.shift(periods=-return_horizon[0])
        if price_mode == 'yield_in_pct' and return_mode == 'spread_in_pct':
            df_return = price_end - price_bgn
        else:
            raise ValueError("Mode not defined!")
        print('Done!')
        return df_return


    @staticmethod
    def aggregate(df_values, df_groups, aggregate_by):
        rows = df_values.index
        concat_list = list()
        for row in rows:
            values = np.array([df_values.loc[row], df_groups.loc[row]]).transpose()
            df_temp = pd.DataFrame(values, columns=['value','group']).dropna().groupby('group')
            if aggregate_by == 'mean':
                df_temp = df_temp.mean()
            elif aggregate_by == 'median':
                df_temp = df_temp.median()
            else:
                raise ValueError("Wrong combination method, please check.")
            df_temp = df_temp.rename(columns={'value':row}).transpose()
            concat_list.append(df_temp)
        df_group_aggregation = pd.concat(concat_list, axis=0, join='outer')
        return df_group_aggregation


    @staticmethod
    def group_by_edges(df_values, df_edges):
        """group values given edges"""
        def sub_group_by_edges(series_values): # applied for each row (date)
            def subsub_group_by_edges(value): # applied for each column (asset)
                if np.isnan(value) or (value > edge_last):
                    return np.nan
                else:
                    return (value >= edge_pre).sum()
            date = series_values.name
            series_edges = df_edges.loc[date]
            edge_last = series_edges.iloc[-1]
            edge_pre = series_edges.iloc[:-1]
            series_groups = series_values.apply(subsub_group_by_edges)
            return series_groups
        df_groups = df_values.apply(sub_group_by_edges, axis=1)
        return df_groups


    @staticmethod
    def plot_groups(ax, df_groups, title=''):
        i = 0
        for col in df_groups.columns:
            i += 1
            if i <= 7:
                line = '-'
            elif 7 < i <=14:
                line ='--'
            ax.plot_date(df_groups.index, df_groups[col], line, label=col)
        #ax.legend(loc=0)
        ax.legend(bbox_to_anchor=(1, 1))
        ax.set_title(title)


    @staticmethod
    def ic_calculator(df_group_factor, df_group_return):
        """Calculate ic and rank_ic"""
        print('Calculating IC...')
        # initialization
        ic_series = pd.Series(index=df_group_factor.index, dtype='float')
        rank_ic_series = ic_series.copy()
        # calculate ic and rank_ic.
        for idx in df_group_factor.index:
            # two series
            a1 = df_group_return.loc[idx]
            a2 = df_group_factor.loc[idx]
            # remove any nan
            idx_non_nan = ~(np.logical_or(np.isnan(a1), np.isnan(a2)))
            if np.count_nonzero(idx_non_nan) <= 3:
                continue
            a1 = a1[idx_non_nan]
            a2 = a2[idx_non_nan]
            # get corr
            ic_series.loc[idx] = np.corrcoef(a1, a2)[0][1]
            rank_ic_series.loc[idx] = spearmanr(a1, a2)[0]
        # return
        print('Done!')
        return ic_series, rank_ic_series
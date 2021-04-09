#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd
import numpy as np
cwd = os.getcwd()


def drawdown(ret_series): 
    
    wealth = 1000*(ret_series+1).cumprod()
    prev_peaks = wealth.cummax()
    drawdowns = (wealth - prev_peaks)/prev_peaks
    
    return pd.DataFrame({
        "Wealth" : wealth,
        "Peaks" : prev_peaks,
        "Drawdown" : drawdowns
    })

def annualized_rets(r, periods_per_year):
    comp_growth = (1+r).prod()
    n_periods = r.shape[0]
    
    return comp_growth**(periods_per_year/n_periods)-1

def annualized_vol(r,periods_per_year):
    
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualized_rets(excess_ret,periods_per_year)
    ann_vol = annualized_vol(r, periods_per_year)
    
    return ann_ex_ret/ann_vol

def skewness(r):
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def summary_stats(r, riskfree_rate=0.01, periods=12):
    ann_r = r.aggregate(annualized_rets, periods_per_year=periods)
    ann_vol = r.aggregate(annualized_vol, periods_per_year=periods)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=periods)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })   

slow_ma = 30
fast_ma = 10
threshold = 5


crypto = pd.read_excel(cwd + "\\CryptoPrices.xlsx",parse_dates=True).rename(columns={"XBTUSD BGN Curncy":"btc_price"})[['Date','btc_price']]

crypto_return = (crypto['btc_price'].values/crypto['btc_price'].shift(1)-1).to_frame().set_index(crypto['Date']).dropna()

crypto_price = crypto.set_index('Date')
crypto_price['fast_ma']=np.round(crypto_price['btc_price'].rolling(fast_ma).mean(),2)
crypto_price['slow_ma']=np.round(crypto_price['btc_price'].rolling(slow_ma).mean(),2)
crypto_price['crossover_diff']=crypto_price['fast_ma']-crypto_price['slow_ma']
crypto_price = crypto_price.dropna()

crypto_price.loc[crypto_price['crossover_diff']>threshold, 'signal'] = 1
crypto_price.loc[crypto_price['crossover_diff']<threshold, 'signal'] = 0

backtest_ret = crypto_return.loc[crypto_price.index,:]

signal = crypto_price.loc[:,['signal']]

backtest_perf = (backtest_ret.values*signal.shift(-1)).dropna().rename(columns={'signal':'btc_momentum'})

results_mom = drawdown(backtest_perf.squeeze())
results_buynhold = drawdown(crypto_return.squeeze())

summary_stats(backtest_perf,0.01,261).style.format({"Annualized Return":"{:.2%}",
                     "Annualized Vol":"{:.2%}",
                     "Skewness":"{:.2f}",
                     "Kurtosis":"{:.2f}",
                     "Sharpe Ratio":"{:.2f}",
                     "Max Drawdown":"{:.2%}"
    })


summary_stats(crypto_return,0.01,261).style.format({"Annualized Return":"{:.2%}",
                     "Annualized Vol":"{:.2%}",
                     "Skewness":"{:.2f}",
                     "Kurtosis":"{:.2f}",
                     "Sharpe Ratio":"{:.2f}",
                     "Max Drawdown":"{:.2%}"
    })


results_mom['Wealth'].plot(figsize=(20,10))

results_buynhold['Wealth'].plot(figsize=(20,10))

pric = crypto_price['btc_price'].plot(label='BTC Price',figsize=(20,10))




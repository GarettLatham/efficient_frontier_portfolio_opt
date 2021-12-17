import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import yfinance as yf
import numpy as np
import datetime
     
symbols = ["EDFMT", "BUFTT", "DODGX", "PYSMT", "MIGFT", "NSRIX", "HVVBT", "SGACT", "JANQT", "TMIET", "VGINT", "VGIET", "VGIST", "BACSF", "PTTRX", "VIPIX", "VGITT", "WACIT", "BLIFT", "LNMMT", "LQMMT", "LFMTT", "LBMTT", "LNJTT", "LBMIT", "LGHMT", "LKMMT", "BLPTT", "LZMMT", "PAAIX", "MLSVF", "SPY"]
portfolio = pd.DataFrame(columns=symbols)
portfolio = portfolio.dropna()
#print(portfolio)
month = 4
day = 1


for symbol in symbols:

    try:
        X = yf.download(symbol,'1980-11-18','2020-05-21')
        X.pop('Open')
        X.pop('High')
        X.pop('Low')
        X.pop('Close')
        X.pop('Volume')
        portfolio[symbol] = X["Adj Close"]                
    except Exception as e:
        print(str(e))
        continue

portfolio = portfolio.dropna(axis=1, how='all')
print(portfolio)
print("Symbols")
print(symbols)
portfolio.dropna(inplace=True)
#print(portfolio)
# Calculate expected returns and sample covariance
#mu = expected_returns.mean_historical_return(portfolio)
#mu = expected_returns.capm_return(portfolio)
mu = expected_returns.ema_historical_return(portfolio)
#print(mu)
#S = risk_models.sample_cov(portfolio)
S = risk_models.exp_cov(portfolio)

# Optimise for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
#raw_weights = ef.max_sharpe(risk_free_rate=.0061)
raw_weights = ef.min_volatility()
#raw_weights = ef.max_quadratic_utility()
cleaned_weights = ef.clean_weights()
#ef.save_weights_to_file("weights1.csv")  # saves to file
#print(cleaned_weights)
ef.portfolio_performance(verbose=True,risk_free_rate=.0063)
latest_prices = get_latest_prices(portfolio)

da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=33000)
#allocation, leftover = da.lp_portfolio()
allocation, leftover = da.greedy_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))

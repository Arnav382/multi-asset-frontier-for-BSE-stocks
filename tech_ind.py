import pandas as pd
import numpy as np
from stocktrends import Renko
import matplotlib.pyplot as plt
import pandas as pd
## for ema -> ema(t)=priceToday*q + ema(t-1)*(1-q)
## q=smoothing factor (for span->2/(1+total days), com-> 1/(1+total days)) 
## com more accurate even though formula is different

def MACD(data,fast=12,slow=26,signal=9):
    """computes the MACD of a particular index based on the parameters. Uses exponential moving average
    fast-> no of weeks of the fast signal 
    slow-> no of weeks of the slow signal
    signal-> no of weeks of the baseline signal
    formula-> ema(fast) - ema(slow)
    
    returns the macd and the ema of the baseline signal
    """
    df=data.copy()
    df["ma_fast"]=df["Adj Close"].ewm(span=fast,min_periods=fast).mean()
    df["ma_slow"]=df["Adj Close"].ewm(span=slow,min_periods=slow).mean()
    df["ma_signal"]=df["Adj Close"].ewm(span=signal,min_periods=signal).mean()
    df['macd']=df["ma_fast"]-df["ma_slow"]
    return df.loc[:,["macd","ma_signal"]]

def ATR(data,n=14):
    df=data.copy()
    df["H-L"]=df["High"]-df["Low"]
    df["H-PC"]=df["High"]-df["Adj Close"].shift(1) ## subtracts the adj close of the previous row
    df["L-PC"]=df["Low"]-df["Adj Close"].shift(1) ## subtracts the adj close of the previous row
    df["TR"]=df[["H-L","L-PC","H-PC"]].max(axis=1,skipna=False)
    df["ATR"]=df["TR"].ewm(span=n,min_periods=n).mean()
    return df["ATR"]


def bollinger_bands(data, n=14,k=2):
    """bollinger band consists of three bands
    1. upper band - n day SMA + (S.D*k)
    2. middle band - n day SMA 
    3. lower band - n day SMA - (S.D*k)
    
    data: data of a particular stock"""
    
    df=data.copy()
    df["middle_band"]= df["Adj Close"].rolling(n).mean()
    df["upper_band"] = df["middle_band"] + k*df["Adj Close"].rolling(n).std(ddof=0)
    df["lower_band"] = df["middle_band"] - k*df["Adj Close"].rolling(n).std(ddof=0)
    df["BB_Width"]= df["upper_band"]-df["lower_band"]
    
    return df[["upper_band","middle_band","lower_band","BB_Width"]]


def RSI(data,n=14):
    """computes the Relative Strength Index of a stock for a particular week window
        data: data of a particular stock"""
    df=data.copy()
    df["Change"]=df["Adj Close"]-df["Adj Close"].shift(1)
    df["gain"]=np.where(df["Change"]>=0,df["Change"],0)
    df["loss"]=np.where(df["Change"]<0,df["Change"]*(-1),0)
    df["avgGain"]=df["gain"].ewm(alpha=1/n,min_periods=n).mean()
    df["avgLoss"]=df["loss"].ewm(alpha=1/n,min_periods=n).mean()
    df["RS"]=df["avgGain"]/df["avgLoss"]
    df["RSI"]=100*(1-1/(1+df["RS"]))
    return df["RSI"]


def ADX(data, n=14):
    """ Stands for Average Directional Index(ADI)
        It indicates the strength of the signal( but not buy or sell). 
        The categorisation is as follows
        1. 0-25 -> weak signal
        2. 25-50 -> moderate signal
        3. 50 onwards -> very strong signal(although very unlikely in real life)
    """
    df=data.copy()
    df["ATR"]=ATR(df,n)
    df["upMove"]=df["High"]-df["High"].shift(1)
    df["downMove"]=df["Low"]-df["Low"].shift(1)
    df["dm_plus"]=np.where((df["upMove"]>df["downMove"]) & (df["upMove"]>0),df["upMove"],0)
    df["dm_minus"]=np.where((df["downMove"]>df["upMove"]) & (df["downMove"]>0),df["downMove"],0)
    df["di_plus"]=100*(df["dm_plus"]/df["ATR"]).ewm(span=n,min_periods=n).mean()
    df["di_minus"]=100*(df["dm_minus"]/df["ATR"]).ewm(span=n,min_periods=n).mean()
    df["ADX"]=100 * abs((df["di_plus"]-df["di_minus"])/(df["di_plus"]+df["di_minus"])).ewm(span=n,min_periods=n).mean()
    return df["ADX"]

def renko_df(data,hourly_df):
    """data should be the orignal data fetched from API call without any other indicators or parameters"""
    df=data.copy()
    df.drop("Close",axis=1,inplace=True)
    df.reset_index(inplace=True)
    df.columns=["date","open","high","low","close","volume"] 
    df2=Renko(df)
    df2.brick_size=3*round(ATR(hourly_df,120).iloc[-1],0)
    renko_df = df2.get_ohlc_data()
    return renko_df


def CAGR(data):
    """computes the Cumulative annual growth rate of the given index. 
    The function assumes that data points are at and interval of 1 day.
    """
    df = data.copy()
    total_intervals=df.shape[0]
    df["return"] = df["Adj Close"].pct_change()
    return_per_interval=np.prod(df["return"]+1)**(1/total_intervals)-1
    annual_ret=(1+return_per_day)**252-1
    return annual_ret

def annual_vol(data):
    """returns the annual volatility of an index(es)"""
    df=data.copy()
    df['return']=df["Adj Close"].pct_change()
    return df['return'].std()*np.sqrt(252)


def sharpe_ratio(data,rf=0.03):
    """ average return earned in excess of risk free rate per unit of volatility
    greater than 1 is considered good, more than 2 is very good and greater than 3 indicates an excellent index"""
    df=data.copy()
    sharpe=(CAGR(df)-rf)/annual_vol(df)
    return sharpe

def sortino_ratio(data,rf=0.03):
    """ average return earned in excess of risk free rate per unit of volatility where return is negative"""
    df=data.copy()
    df["return"]=df["Adj Close"].pct_change()
    neg_ret=np.where(df["return"]>0,0,df["return"])
    neg_vol=pd.Series(neg_ret[neg_ret!=0]).std()*np.sqrt(252)
    SORTINO=(CAGR(df)-rf)/neg_vol
    return SORTINO

def max_drawdown(data):
    """
      takes a time series of asset return 
      Computes and returns wealth index, previous peaks and max drawdown
      """
    df=data.copy()
    df['returns']=df["Close"].pct_change()
    df["cumu_return"] = (1+df['returns']).cumprod()
    df["cum_roll_max"]=df["cumu_return"].cummax()
    df["drawdown"] = df["cum_roll_max"]-df["cumu_return"]
    df['drawdown']=df['drawdown']/df['cum_roll_max']
    return df["drawdown"].max()

def calmar_ratio(data):
    "computes the calmar ratio"
    df=data.copy()
    return CAGR(data)/max_drawdown(data)
from turtle import color
from click import style
import pandas as pd 
import quandl
import numpy as np
import scipy.stats
from scipy.optimize import minimize

def drawdown(ret: pd.Series):
      """
      takes a time series of asset return 
      Computes and returns wealth index, previous peaks and max drawdown
      """
      wealth_index = 100*(1+ret).cumprod()
      previous_peaks = wealth_index.cummax()
      drawdown = (-previous_peaks + wealth_index)/previous_peaks
      return pd.DataFrame({"wealth":wealth_index,
                           "previous peak":previous_peaks,
                           "drawdown":drawdown})

def get_data(token,key="uitSwzyGWCxD8Nt2pZGF"):
    """
    gets prices of shares with their designated token and the user key for authentication
    find the token of your desired shares from 
    
    """
    return quandl.get(token)

def compute_daily_return(r):
    """computes the series of daily returns of an index(es) as compared to previous day
    r : series of closing value of a particular index(es)"""
    return r.pct_change()

def skewness(r):
    """
    computes skewness of a particular dataframe or series
    returns a series or integer
    """
    dem_r=r-r.mean()
    sigma=r.std(ddof=0)
    exp=(dem_r**3).mean()
    return exp/sigma**3

def kurtosis(r):
    """
    computes kurtosis of a particular dataframe or series
    returns a series or integer
    """
    dem_r=r-r.mean()
    sigma=r.std(ddof=0)
    exp=(dem_r**4).mean()
    return exp/sigma**4

def is_normal(r,level=0.01):
    """ 
    applies jarque bera test to see if the distribution is normal or not 
    """
    stat,p_value=scipy.stats.jarque_bera(r)
    return p_value>level

def annual_vol(ret):
    """returns the annual volatility of an index(es)"""
    return ret.std()*np.sqrt(252)

def monthly_vol(ret):
    """returns the monthly volatility of an index(es)"""
    return ret.std()*np.sqrt(21)

def daily_vol(ret):
    """returns the daily volatility (σp) of an index(es)"""
    return ret.std()

def annual_ret(ret):
    """returns the monthly return of an index(es)"""
    
    total_days=ret.shape[0]
    return_per_day=np.prod(ret+1)**(1/total_days)-1
    annual_ret=(1+return_per_day)**252-1
    return annual_ret

def sharpe_ratio(ret,rfr=3):
    """ returns the sharpe ratio of an index(es) 
    ret: series of daily returns of an index(es) as compared to previous day"""
    
    rfr=rfr/100
    excess_return=annual_ret(ret)-rfr
    sharpe_ratio=excess_return/annual_vol(ret)
    return sharpe_ratio
    
def var_historic(r,level=5):
    """Computes VaR based on historical data
    returns Value at risk i.e 5 percent(default) chance to lose this percentage in a given month at worst
    """
    if isinstance(r,pd.DataFrame):
        return r.aggregate(var_historic,level=level)
    elif isinstance(r,pd.Series):
        return np.percentile(r,level,axis=0)
    else: raise TypeError("Input must be a series or dataframe")



def cvar_historic(r,level=5):
      """
      Computes CVaR based on historical data
      returns when that 5 percent loss happens the average loss per month is calculated
      """
      if isinstance(r,pd.DataFrame):
        return r.aggregate(cvar_historic,level=level)
      elif isinstance(r,pd.Series):
        is_beyond= r<= - var_historic(r,level=level)
        return -r[is_beyond].mean()
      else: raise TypeError("Input must be a series or dataframe")

def portfolio_return(weights,returns):
    """
    Computes the overall return of the portfolio using the weights
    """
    return weights.T @ returns

def portfolio_vol(weights,covmat):
    """
    Computes the overall volatility of the portfolio using the weights and covmatrix
    """
    return (weights.T @ covmat @ weights)**0.5

def plot_ef2(n_points,er,cov,style="._."):
    if er.shape[0]!=2:
        raise ValueError("can only plot for 2 assets, not {}".format(er.shape[0]))                     
    weights=[np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
    returns=[portfolio_return(w,er) for w in weights]
    vol=[portfolio_vol(w,cov) for w in weights]
    ef=pd.DataFrame({"returns":returns,"risk":vol})
    return ef.plot.scatter(x="risk",y="returns",style=style)

def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x


def optimal_weights(n_points, er, cov):
    """
    Returns a list of weights that represent a grid of n_points on the efficient frontier
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


def gmv(cov):
    """
    weights of the global min vol with a given cov matrix
    """
    n=cov.shape[0]
    return msr(0,np.repeat(1,n),cov) 
def plot_ef(n_points, er, cov,show_cml=False,rf=0,show_ew=False,show_gmv=False):
    """
    Plots the multi-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    ax= ef.plot.line(x="Volatility", y="Returns", style='.-', legend=True)
    if show_gmv:
        w_gmv=gmv(cov)
        r_gmv=portfolio_return(w_gmv,er)
        v_gmv=portfolio_vol(w_gmv,cov)
        ax.plot([v_gmv],[r_gmv],color='midnightblue',markersize=10,marker='o')
    if show_ew:
        n=er.shape[0]
        w_ew=np.repeat(1/n,n)
        r_ew=portfolio_return(w_ew,er)
        v_ew=portfolio_vol(w_ew,cov)
        #Display EW
        ax.plot([v_ew],[r_ew],color="goldenrod",markersize=12,marker="o")
    if show_cml:
        ax.set_xlim(left=0)
        w_msr=msr(rf,er,cov)
        r_msr=portfolio_return(w_msr,er)
        v_msr=portfolio_vol(w_msr,cov)

        # Add CML(Capital Market Line)
        cml_x=[0,v_msr]
        cml_y=[rf,r_msr]
        ax.plot(cml_x,cml_y,color='green',linestyle='dashed')
    return ax

def msr(rf, er, cov):
    """
    returns the max sharpe ratio portfolio line weights with a given 
    risk free rate, expected returns and covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def negative_sharpe_ratio(weights,rf,er,cov):
        r=portfolio_return(weights,er)
        vol=portfolio_vol(weights,cov)
        ans=(r-rf)/vol
        return -1*ans
    weights = minimize(negative_sharpe_ratio, init_guess,
                       args=(rf,er,cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1),
                       bounds=bounds)
    return weights.x
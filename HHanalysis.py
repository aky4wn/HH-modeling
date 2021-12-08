import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, f1_score
import pickle  
from sklearn.metrics import confusion_matrix  
sns.set()
sns.set(font_scale=2.5)

def mse(y, f):
    diffterm = (y.flatten() - f.flatten())**2
    return(np.nanmean(diffterm))

def mad(y, f):
    diffterm = np.abs(y.flatten() - f.flatten())
    return(np.nanmean(diffterm))

def mape(y, f):
    ## Check for 0s in y and don't calculate here
    inds = np.where(y.flatten()!=0)[0]
    diffterm = np.abs(1 - f.flatten()[inds]/y.flatten()[inds])
    return(np.nanmean(diffterm))

def zape(y, f, loss, const = None): ## Calculate ZAPE with given loss function, linear or min
    
    y = y.flatten()
    f = f.flatten()
    if loss == 'linear':
        part0 = const*f[np.where(y == 0)]
    if loss == 'min':
        part0 = np.minimum(1, f[np.where(y == 0)])
    if loss == 'mzape':
        part0 = f[np.where(y == 0)]/(1+f[np.where(y == 0)])   
    diffterm = np.hstack((part0, np.abs(1 - f[np.where(y != 0)]/y[np.where(y != 0)])))
    return(np.nanmean(diffterm))

## gradient descent for mZAPE calculation
## pi0 is probability that y = 0
## normconst = normalizing constant
## G is empirical CDF of p(y)/y
## y is sorted MC samples
## tol is tolerance of convergence
## alpha is learning rate
## mmy is -1 median
def grad_Rf(pi0, normconst, G, y, mmy, tol, alpha):
    f = np.round(mmy, 2) ## initialize
    y = np.sort(y)
     #print(normconst)
    for i in range(200):
        ## eval CDF - y sorted, find index of f in y, then find value of G at that point
        idx = (np.abs(y - f)).argmin() ## find closest value in y to f
        
        grad = pi0/(1+f)**2 + 2/normconst*G[idx] - 1/normconst
        f = np.round(f - alpha*grad, 2)
        ## cant divide by 0
        if f == -1:
            f = -1.01
        diff = np.abs(alpha*grad)
    return(f)



# APE and mZAPE loss when y ~ log T distribution :  (-1) median does not exist 
#   since p(y)/y is not integrable. 
## Find (-1)-median for MAPE optimal forecast
## x = MC forecast samples
#  pi0 = probability of 0 for observed y's
# Results: y = exp(x) ~ LT_k(m,v) 
#       my = median of p(y)
#       mmy = (-1)-median of p(y) approximated via MC 
# Return optimal forecast under MAPE and ZAPE (MAPE, mZAPE)
def fstar_logT(x, pi0):
    # i = np.where(x > 0)[0]
    # y = np.sort(x[i])
    y = np.sort(np.exp(x))
    ## check if all 0s
    if np.sum(x) == 0 or len(y) == 0:
        return(0, 0)
    
    gy = 1.0/y;         # g(y) propto p(y)/y evaluated at MC sample values
    #const = 1/np.sum(gy)
    normconst = 1 / (1/len(x) * np.sum(gy))
    gy = gy/np.sum(gy); Gy=np.cumsum(gy); 
    
    try: 
        mmy = np.min(y[Gy>=0.5]); # MC estimate of (-1)-median of p(y)
    except ValueError:
        mmy = np.median(y)
    
    ## Calculate cutoff value for ZAPE
    ## gradient descent to find f
    tol = 0.01
    alpha = 0.3
    ftil = grad_Rf(pi0, normconst, Gy, y, mmy, tol, alpha)
    cutoff = normconst*pi0/(1+ftil)**2
    ## Check cutoff value
    if cutoff >= 1: ## forecast 0
        fopt = 0
    else: ## forecast from quantile of inverse-CDF
        fopt = ftil
    return(mmy, fopt)



## Calibration plots
## p1 is the forecast probabilities
## y are the observed values
def calc_calib(p1, y):
    bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] 
    
    Npred, x = np.histogram(p1.flatten(), bins = bins)
    Nobs = np.zeros(10)
    Nobs[:-1] = np.array([np.sum(y.flatten()[np.where(np.logical_and(p1.flatten() >=bins[i], 
                                                                        p1.flatten() < bins[i+1]))]) 
                            for i in range(9)])
    Nobs[-1] = np.sum(y.flatten()[np.where(np.logical_and(p1.flatten() >=0.9, 
                                                                    p1.flatten() <= 1))])
    # phat1 = Nobs/Npred
    phat1 = np.where(Npred > 0, Nobs/Npred, 0)
    diff1 = np.where(Nobs > 0, 1.96*np.sqrt(phat1*(1-phat1)/Nobs), 0)
    # diff1 = 1.96*np.sqrt(phat1*(1-phat1)/Nobs)
    return(phat1, diff1)
## Code following https://www.swpc.noaa.gov/sites/default/files/images/u30/Ensemble%20Forecast%20Verification.pdf

## pi0 is probability that y = 0
## normconst = normalizing constant
## G is empirical CDF of p(y)/y
## y is sorted MC samples
## P is empirical CDF of p(y)
## f is forecast value
## constant terms P(1) and 1 are dropped
## mmy is (-1)-median
def R_mZAPE(pi0, normconst, G, y, P, f):
    idx = (np.abs(y - f)).argmin() ## find closest value in y to f
    R = f/(1+f)*pi0 -2*P[idx] + 2*f/normconst*G[idx] - f/normconst
    return(R)

## Grid search from (0, mmy)
def grad_counts(pi0, normconst, G, y, mmy):
    mmy = np.round(mmy).astype(int)
    P = np.cumsum(y/np.cumsum(y))
    ftil = np.arange(0, mmy+1)
    loss_vals = np.array([R_mZAPE(pi0, normconst, G, y, P, ftil[i]) for i in range(mmy+1)])
    ind = np.argmin(loss_vals)
    f = ftil[ind]
    return(f)

####################################################################
# APE and mZAPE loss when y ~ DCMM, use MC samples
# Look at distribution of loss for chosen forecasts, using direct Monte Carlo
# x = MC samples
# Return optimal forecast under MAPE and ZAPE (MAPE, ZAPE)

def fstar_counts(x):
    pi0 = len(np.where(x == 0)[0])/len(x)
    i = np.where(x > 0)[0]
    y = np.sort(x[i])
    ## check if all 0s
    if np.sum(x) == 0:
        return(0,0)
    gy = 1.0/y;         # g(y) propto p(y)/y evaluated at MC sample values
    #const = 1/np.sum(gy)
    normconst = 1 / (1/len(x) * np.sum(gy))
    gy = gy/np.sum(gy); Gy=np.cumsum(gy); 
    
    mmy = np.min(y[Gy>=0.5]); # MC estimate of (-1)-median of p(y)
    
    ## Calculate cutoff value for ZAPE
    
    ## grid search to find f
    ftil = grad_counts(pi0, normconst, Gy, y, mmy)
    cutoff = normconst*pi0/(1+ftil)**2
    
    ## Check cutoff value
    if cutoff >= 1: ## forecast 0
        fopt = 0
    else: ## forecast from quantile of inverse-CDF
        fopt = ftil
    return(mmy, fopt)



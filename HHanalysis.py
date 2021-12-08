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


# ###########################################################################################
# ## p(Return) ##
# ###############################
# ## k is forecast horizon
# ## hhg = household group

# def analyze_pReturn(k, hhg):
#     fname = 'simulated_results/pReturn/HH-' + str(hhg)+ '-k' + str(k) + \
#                                               '-true.npy'    
#     y = np.load(fname)[:, 20:, :]      
#     fname = 'simulated_results/pReturn/HH-' + str(hhg)+ '-k' + str(k) + \
#                                             '-samples.npy'    
#     samples = np.load(fname)
#     ms = np.mean(samples, axis = 1)
#     ms = ms[:, 20:, :]
#     results = np.zeros((3, 3))
#     ## Only do k = 0, 3, 7
#     count = 0
#     for i in [0,3,7]:
#         fpr1, tpr1, _ = roc_curve(y[:, :, i].flatten(), ms[:, :, i].flatten())
#         roc_auc1 = auc(fpr1, tpr1)

#         results[0, count] = np.array([roc_auc1])
        
#         fpr1 = f1_score(y[:, :, i].flatten(), np.round(ms[:, :, i].flatten()))

        
#         results[1, count] = np.array([fpr1])
#         results[2, count] = mse(y[:, :, i].flatten(), ms[:, :, i].flatten())
#         count += 1
    

#     #####################################
#     ## Calibration - MultiStep Ahead 
#     #####################################
    
#     bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] 
#     ## One-Step Ahead
#     phat1, diff1 = calc_calib(ms[:, :, 0].flatten(), y[:, :, 0].flatten())
#     ## 4-Step Ahead
#     phat2, diff2 = calc_calib(ms[:, :, 3].flatten(), y[:, :, 3].flatten())
#     ## 8-Step Ahead
#     phat3, diff3 = calc_calib(ms[:, :, -1].flatten(), y[:, :, -1].flatten())
    
#     _, x = np.histogram(ms[:, :, -1].flatten(), bins = bins)

#     plt.figure()
#     plt.figure(figsize = (10,8))
#     plt.scatter((x-0.05)[1:], phat1, label = 'k=1', s = 100)
#     plt.errorbar((x-0.05)[1:], phat1, yerr = diff1, linewidth = 3)
#     plt.scatter((x-0.05)[1:], phat2, label = 'k=4', color = 'C1', s = 100)
#     plt.errorbar((x-0.05)[1:], phat2, yerr = diff2, color = 'C1', linewidth = 3)
#     plt.scatter((x-0.05)[1:], phat3, label = 'k=8', color = 'C2',s = 100)
#     plt.errorbar((x-0.05)[1:], phat3, yerr = diff3, color = 'C2', linewidth = 3)
#     plt.ylim(0,1)
#     plt.xlim(0,1)
#     plt.plot([0,1],[0,1], '--', color = 'black')
#     plt.xlabel('Predicted Probability')
#     plt.ylabel('Observed Frequency')
#     plt.fill_between(np.array(bins), np.array(bins) - 0.007, np.array(bins) + 0.007, 
#                      color='grey', alpha=.5)
#     plt.legend(loc = 'lower right')
#     plt.tight_layout()
#     fname = 'simulated_results/RESULTS/pReturn/HH-' + str(hhg)+'-k'  + str(k) +  \
#                      '-calibration.png'
#     plt.savefig(fname)
#     plt.close('all')

#     df = pd.DataFrame(results)
#     df.columns = [str(k) for k in [1,4,8]]
#     df.index = ['AUC', 'F1', 'MSE']
#     df.to_csv('simulated_results/RESULTS/pReturn/HH-' + str(hhg)+'-k'  + str(k) + \
#                   '-summary.csv' )
#     print('Done:', hhg)
    
    
# #####################################################    
# k = 8
# analyze_pReturn(k, 1)
# analyze_pReturn(k, 2)
# analyze_pReturn(k, 3)

# #####################################################################################
# #### p(log Total Spend | Return)
# #################################################################################


# def analyze_logTotal(hhg):
#     fname = 'simulated_results/total_spend/HH-' + str(hhg)+ '-k1-true.npy'    
#     y = np.load(fname)[:, 20:]        
#     fname = 'simulated_results/total_spend/HH-' + str(hhg)+\
#                                  '-k1-samples.npy'    
#     samples = np.load(fname)[:, :, 20:]

#     ## Coverage 
#     alpha = np.arange(0.1, 1, 0.05)
#     alpha_low = 0.5 - alpha/2
#     alpha_high = 0.5 + alpha/2
    
#     numhh = y.shape[0]
#     coverage = np.zeros((numhh, len(alpha)))
    
#     weeks = np.zeros(numhh)
#     inds = np.array([np.where(y[i,:] == -10)[0][0] if y[i, -1] == -10 else y.shape[1]
#                                 for i in range(y.shape[0])])
#     for hh in range(numhh):
#         q = inds[hh]
#         for a in range(len(alpha)):
#             lwr = np.quantile(samples[hh,:,:q], alpha_low[a], axis = 0) ## 95% Interval over HH
#             upr = np.quantile(samples[hh,:,:q], alpha_high[a], axis = 0)

#             try:
#                 coverage[hh, a] = np.sum(np.logical_and(y[hh, :q] > lwr, 
#                                                            y[hh, :q] < upr).astype(int))/len(lwr)
#             except RuntimeWarning:
#                 print(hh, a)
#                 print(np.sum(np.logical_and(y[hh, :q] > lwr, y[hh, :q] < upr).astype(int)), 
#                       len(lwr))

#     print(np.any(np.isnan(coverage)))
    
#     color_list = ['C0', 'C1', 'C2']
#     plt.figure()
#     plt.figure(figsize = (10,8))
#     plt.scatter(alpha, np.nanmean(coverage, axis = 0), color = color_list[hhg-1], s = 100)
#     diff = np.nanstd(coverage, axis = 0)
#     plt.errorbar(alpha, np.nanmean(coverage, axis = 0), yerr = diff, color = color_list[hhg-1],
#                 linewidth = 3)
#     plt.ylim(0,1)
#     plt.xlim(0,1)
#     plt.plot([0,1],[0,1], '--', color = 'black')
#     plt.xlabel('Theoretical Coverage')
#     plt.ylabel('Observed Coverage')
#     plt.tight_layout()
#     fname = 'simulated_results/RESULTS/total_spend/HH-' + str(hhg)+ \
#                                 '-k1-coverage.png'
#     plt.savefig(fname)
#     plt.close('all')

#     print('Done:', hhg)
    
    
# #####################################################    
# analyze_logTotal(1)
# analyze_logTotal(2)
# analyze_logTotal(3)

# #####################################################
# ### p(log Total Spend by Category)
# ####################################################

# def load_Cat(hhg, cat, model):
#     k = 1       
#     fname = 'simulated_results/Cat/HH-'+str(hhg)+'-CAT='+str(cat)+ \
#                         '-k'+str(k) + '-m' + str(model) + '-samples.npy'    
#     samples = np.load(fname)
    
#     return(samples)
    
# def analyze_Cat(hhg, cat):
#     fname = 'simulated_results/Cat/HH-'+str(hhg)+'-CAT='+str(cat)+ \
#                     '-k1-true.npy'    
#     y = np.load(fname) 
#     y = y[:, :104]
    
    
#     ## Find indices to forecast over
#     inds = np.array([np.where(y[i,:] == -10)[0][0] if y[i, -1] == -10 else y.shape[1]
#                                 for i in range(y.shape[0])])
#     ## Find indices of HHs to use
#     drop_inds = np.where(inds == 0)[0]
#     hh_inds = np.arange(y.shape[0])
#     if len(drop_inds) > 0:
#         hh_inds = np.delete(hh_inds, drop_inds)
#         inds = np.delete(inds, drop_inds)
#         print(drop_inds, len(hh_inds))

#     numhh = len(hh_inds)

#     ## Exponentiate y
#     y = y[hh_inds, :]
#     y = np.where(y == 0, 0, np.exp(y))
    
#     ## Load in results of modeling
#     fm = list()
#     samples = list()
#     for i in range(1, 4):
#         samples1 = load_Cat(hhg, cat,  i)
#         samples.append(samples1[hh_inds, :, :104])
#         ## Find median for forecast
#         fm.append(np.median(samples1[hh_inds, :, :104], axis = 1))

    
#     numm = len(fm) ## number of models
    
#     ## (1) MAD vs. model
#     opt_mad = np.zeros((numm, numhh))
#     for i in range(numm):
#         for j in range(numhh):
#             ft = np.where(fm[i][j,:inds[j]] == 0, 0, np.exp(fm[i][j,:inds[j]]))
#             opt_mad[i, j] = mad(y[j, :inds[j]], ft)
                     
#     print('MAD Done')
    
#     ## (4) MAPE
    
#     # (2) Approximate (-1) Median of logT for each hh and each time t
#     # Only select values that are non-0
#     mape_list = np.zeros((numm, numhh))
#     zape_list = np.zeros((numm, numhh))
    
#     for i in range(numm):
#         for j in range(numhh):
#             pi0 = np.array([len(np.where(samples[i][j,:,t] == 0)[0]) 
#                                     for t in range(inds[j])])/samples[i].shape[1]
#             f_MAPE, f_mZAPE = np.array([fstar_logT(samples[i][j,:,t], pi0[t]) 
#                                 for t in range(1, inds[j])]).T  
#             mape_list[i, j] = mape(y[j, 1:inds[j]], f_MAPE)
#             zape_list[i, j] = zape(y[j, 1:inds[j]], f_mZAPE, 'mzape')
                
#     ## Subset to appropriate values - MAD
#     results = np.zeros((9, numm))
#     results[0,:] = np.nanmedian(opt_mad, axis = 1)
#     results[1,:] = np.nanquantile(opt_mad, 0.25, axis = 1)
#     results[2,:] = np.nanquantile(opt_mad, 0.75, axis = 1)
#     # Subset to appropriate values - MAPE
#     results[3,:] = np.nanmedian(mape_list, axis = 1)
#     results[4,:] = np.nanquantile(mape_list, 0.25, axis = 1)
#     results[5,:] = np.nanquantile(mape_list, 0.75, axis = 1)
#     # Subset to appropriate values - ZAPE
#     results[6,:] = np.nanmedian(zape_list, axis = 1)
#     results[7,:] = np.nanquantile(zape_list, 0.25, axis = 1)
#     results[8,:] = np.nanquantile(zape_list, 0.75, axis = 1)

#     ## Save Results
#     df = pd.DataFrame(np.round(results, 2))
#     df.columns = ['m' + str(i) for i in np.arange(1, 3 + 1)]
#     df.index = ['MAD Median', 'MAD 25%', 'MAD 75%', 
#                 'MAPE Median', 'MAPE 25%', 'MAPE 75%',
#                 'ZAPE Median', 'ZAPE 25%', 'ZAPE 75%'] 
#     df.to_csv('simulated_results/RESULTS/Cat/HH-'+str(hhg)+'-Cat='+str(cat)+ \
#               '-k1-summary-QUANTILES.csv')    

#     ### Calibration plots
#     ###############################
#     ## (1) Make Calibration Plots
#     ## models to consider [15, 16, 26, 29] -> [14, 15, 25, 28]
       
#     bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] 
#     cat_cov0 = -10*np.ones((numhh, 104))
#     cat_cov1 = -10*np.ones((numhh, 104))
#     cat_cov2 = -10*np.ones((numhh, 104))
#     for j in range(numhh):
#         cat_cov0[j, :inds[j]] = [len(np.where(samples[0][j,:, t] != 0)[0])/samples[0].shape[1]
#                                 for t in range(inds[j])]
#         cat_cov1[j, :inds[j]] = [len(np.where(samples[1][j,:, t] != 0)[0])/samples[1].shape[1]
#                                 for t in range(inds[j])]
#         cat_cov2[j, :inds[j]] = [len(np.where(samples[2][j,:, t] != 0)[0])/samples[2].shape[1]
#                                 for t in range(inds[j])]

#     yacc = y[:,:104].flatten()
#     inds = np.where(yacc != -10) ## Values to end forecasting at == -10
#     yacc = yacc[inds]
#     yacc[np.where(yacc != 0)] = 1
#     cat_cov0 = cat_cov0.flatten()[inds]
#     phat1, diff1 = calc_calib(cat_cov0, yacc.flatten())
#     cat_cov1 = cat_cov1.flatten()[inds]
#     phat2, diff2 = calc_calib(cat_cov1, yacc.flatten())

#     cat_cov2 = cat_cov2.flatten()[inds]
#     phat3, diff3 = calc_calib(cat_cov2, yacc.flatten())

#     _, x = np.histogram(cat_cov2, bins = bins)
#     plt.figure()
#     plt.figure(figsize = (10,8))
#     plt.scatter((x-0.05)[1:], phat1, s = 100, label = 'M1')
#     plt.errorbar((x-0.05)[1:], phat1, linewidth=3, yerr = diff1)
#     plt.scatter((x-0.05)[1:], phat2, s = 100, label = 'M2', color = 'C1')
#     plt.errorbar((x-0.05)[1:], phat2, linewidth=3, yerr = diff2, color = 'C1')
#     plt.scatter((x-0.05)[1:], phat3, s = 100, label = 'M3', color = 'C2')
#     plt.errorbar((x-0.05)[1:], phat3, linewidth=3, yerr = diff2, color = 'C2')
#     plt.ylim(0,1)
#     plt.ylim(0,1)
#     plt.plot([0,1],[0,1], '--', color = 'black')
#     plt.xlabel('Predicted Probability')
#     plt.ylabel('Observed Frequency')
#     plt.fill_between(np.array(bins), np.array(bins) - 0.007, np.array(bins) + 0.007, 
#                         color='grey', alpha=.5)
#     plt.legend(loc = 'lower right')
#     plt.tight_layout()
#     fn = 'simulated_results/RESULTS/Cat/HH-'+str(hhg)+'-Cat='+str(cat)+ \
#               '-calibration1.png'
#     plt.savefig(fn)

# #####################################################    
# for i in [1, 2]:
#     analyze_Cat(1, i) 
#     analyze_Cat(2, i)
#     analyze_Cat(3, i)  

# #####################################################
# ### p(log Total Spend by Sub-Category)
# ####################################################

# def load_SubCat(hhg, sub, model):
#     k = 1       
#     fname = 'simulated_results/Sub_Cat/HH-'+str(hhg)+'-SUB_CAT='+str(sub)+ \
#                         '-k'+str(k) + '-m' + str(model) + '-samples.npy'    
#     samples = np.load(fname)
    
#     return(samples)
    
# def analyze_SubCat(hhg, sub):
#     fname = 'simulated_results/Sub_Cat/HH-'+str(hhg)+'-SUB_CAT='+str(sub)+ \
#                     '-k1-true.npy'    
#     y = np.load(fname) 
#     y = y[:, :104]
    
#     ## Find indices to forecast over
#     inds = np.array([np.where(y[i,:] == -10)[0][0] if y[i, -1] == -10 else y.shape[1]
#                                 for i in range(y.shape[0])])
#     ## Find indices of HHs to use
#     drop_inds = np.where(inds == 0)[0]
#     hh_inds = np.arange(y.shape[0])
#     if len(drop_inds) > 0:
#         hh_inds = np.delete(hh_inds, drop_inds)
#         inds = np.delete(inds, drop_inds)
#         print(drop_inds, len(hh_inds))

#     numhh = len(hh_inds)

#     ## Exponentiate y
#     y = y[hh_inds, :]
#     y = np.where(y == 0, 0, np.exp(y))
    
#     ## Load in results of modeling
#     fm = list()
#     samples = list()
#     for i in range(1, 4):
#         samples1 = load_SubCat(hhg, sub,  i)
#         samples.append(samples1[hh_inds, :, :104])
#         ## Find median for forecast
#         fm.append(np.median(samples1[hh_inds, :, :104], axis = 1))

    
#     numm = len(fm) ## number of models
    
#     ## (1) MAD vs. model
#     numhh = fm[0].shape[0] ## number of HH
#     opt_mad = np.zeros((numm, numhh))
#     for i in range(numm):
#         for j in range(numhh):
#             ft = np.where(fm[i][j,:inds[j]] == 0, 0, np.exp(fm[i][j,:inds[j]]))
#             opt_mad[i, j] = mad(y[j, :inds[j]], ft)
                     
#     print('MAD Done')
    
#     ## (4) MAPE
#     # (2) Approximate (-1) Median of logT for each hh and each time t
#     # Only select values that are non-0
#     mape_list = np.zeros((numm, numhh))
#     zape_list = np.zeros((numm, numhh))
    
#     delVar = 0.9
#     k = 1/(1 - delVar)
#     for i in range(numm):
#         for j in range(numhh):
#             pi0 = np.array([len(np.where(samples[i][j,:,t] == 0)[0]) 
#                                     for t in range(inds[j])])/samples[i].shape[1]
#             f_MAPE, f_mZAPE = np.array([fstar_logT(samples[i][j,:,t], pi0[t]) 
#                                 for t in range(1, inds[j])]).T  
#             mape_list[i, j] = mape(y[j, 1:inds[j]], f_MAPE)
#             zape_list[i, j] = zape(y[j, 1:inds[j]], f_mZAPE, 'mzape')
                
#     ## Subset to appropriate values - MAD
#     results = np.zeros((9, numm))
#     results[0,:] = np.nanmedian(opt_mad, axis = 1)
#     results[1,:] = np.nanquantile(opt_mad, 0.25, axis = 1)
#     results[2,:] = np.nanquantile(opt_mad, 0.75, axis = 1)
#     # Subset to appropriate values - MAPE
#     results[3,:] = np.nanmedian(mape_list, axis = 1)
#     results[4,:] = np.nanquantile(mape_list, 0.25, axis = 1)
#     results[5,:] = np.nanquantile(mape_list, 0.75, axis = 1)
#     # Subset to appropriate values - ZAPE
#     results[6,:] = np.nanmedian(zape_list, axis = 1)
#     results[7,:] = np.nanquantile(zape_list, 0.25, axis = 1)
#     results[8,:] = np.nanquantile(zape_list, 0.75, axis = 1)

#     ## Save Results
#     df = pd.DataFrame(np.round(results, 2))
#     df.columns = ['m' + str(i) for i in np.arange(1, 3 + 1)]
#     df.index = ['MAD Median', 'MAD 25%', 'MAD 75%', 
#                 'MAPE Median', 'MAPE 25%', 'MAPE 75%',
#                 'ZAPE Median', 'ZAPE 25%', 'ZAPE 75%'] 
#     df.to_csv('simulated_results/RESULTS/Sub_Cat/HH-'+str(hhg)+'-Sub_Cat='+ \
#                     str(sub)+ '-k1-summary-QUANTILES.csv')    


# #####################################################    
# for i in ['1A', '1B', '2A', '2B']:
#     analyze_SubCat(1, i) 
#     analyze_SubCat(2, i)
#     analyze_SubCat(3, i)  

########################
## ITEM ################
########################

def analyze_ITEM(hhg, item):

    fname = 'simulated_results/ITEM/HH-'+str(hhg)+ \
                                    '-ITEM='+str(item)+'-k1-true.npy'    
    y = np.load(fname)          
    y = y[:, :104]
    print(y)

    ## Find indices to forecast over
    inds = np.array([np.where(y[i,:] == -10)[0][0] if y[i, -1] == -10 else y.shape[1]
                                for i in range(y.shape[0])])
    ## Find indices of HHs to use
    drop_inds = np.where(inds == 0)[0]
    hh_inds = np.arange(y.shape[0])
    if len(drop_inds) > 0:
        hh_inds = np.delete(hh_inds, drop_inds)
        inds = np.delete(inds, drop_inds)
        print(drop_inds, len(hh_inds))

    numhh = len(hh_inds)
    y = y[hh_inds, :]
    
    ## Load in results of modeling
    fm = list()
    samples = list()
    for i in [1,2,3]:
        fname = 'simulated_results/ITEM/HH-'+str(hhg)+ \
                            '-ITEM='+str(item)+'-k1-m' + \
                            str(i) + '-samples.npy'    
        samples1 = np.load(fname)
        samples.append(samples1[hh_inds, :, :104])
        fm.append(np.median(samples1[hh_inds, :, :104], axis = 1))
    print(fm[0].shape)
    ## Array for saving results of analysis
    numm = len(fm) ## number of models
    
    ### (1) MAD vs. model
    opt_mad = np.zeros((numm, numhh))
    optf_mad_m1 = np.zeros((numhh, 104))
    optf_mad_m2 = np.zeros((numhh, 104))
    optf_mad_m3 = np.zeros((numhh, 104))
    for j in range(numhh):
        opt_mad[0, j] = mad(y[j, :inds[j]], fm[0][j,:inds[j]])
        optf_mad_m1[j, :inds[j]] = fm[0][j,:inds[j]]
        opt_mad[1, j] = mad(y[j, :inds[j]], fm[1][j,:inds[j]])
        optf_mad_m2[j, :inds[j]] = fm[1][j,:inds[j]]
        opt_mad[2, j] = mad(y[j, :inds[j]], fm[2][j,:inds[j]])
        optf_mad_m3[j, :inds[j]] = fm[2][j,:inds[j]]

    print('MAD Done')
    
    # ## (4) MAPE and mZAPE, 
    mape_list = np.zeros((numm, numhh))
    zape_list = np.zeros((numm, numhh))
    optf_mzape_m1 = np.zeros((numhh, 104))
    optf_mzape_m2 = np.zeros((numhh, 104))
    optf_mzape_m3 = np.zeros((numhh, 104))
    
   
    for j in range(numhh):
        f_MAPE, f_mZAPE = np.array([fstar_counts(samples[0][j,:, t])
                            for t in range(inds[j])]).T
        mape_list[0, j] = mape(y[j, :inds[j]], f_MAPE)
        zape_list[0, j] = zape(y[j, :inds[j]], f_mZAPE, 'mzape')
        optf_mzape_m1[j, :inds[j]] = f_mZAPE 

        f_MAPE, f_mZAPE = np.array([fstar_counts(samples[1][j,:, t])
                            for t in range(inds[j])]).T
        mape_list[1, j] = mape(y[j, :inds[j]], f_MAPE)
        zape_list[1, j] = zape(y[j, :inds[j]], f_mZAPE, 'mzape')
        optf_mzape_m2[j, :inds[j]] = f_mZAPE 

        f_MAPE, f_mZAPE = np.array([fstar_counts(samples[2][j,:, t])
                            for t in range(inds[j])]).T
        mape_list[2, j] = mape(y[j, :inds[j]], f_MAPE)
        zape_list[2, j] = zape(y[j, :inds[j]], f_mZAPE, 'mzape')
        optf_mzape_m3[j, :inds[j]] = f_mZAPE 
            
    
    ## Subset to appropriate values - MAD
    results = np.zeros((9, numm))
    results[0,:] = np.nanmedian(opt_mad, axis = 1)
    results[1,:] = np.nanquantile(opt_mad, 0.25, axis = 1)
    results[2,:] = np.nanquantile(opt_mad, 0.75, axis = 1)
    # Subset to appropriate values - MAPE
    results[3,:] = np.nanmedian(mape_list, axis = 1)
    results[4,:] = np.nanquantile(mape_list, 0.25, axis = 1)
    results[5,:] = np.nanquantile(mape_list, 0.75, axis = 1)
    # Subset to appropriate values - ZAPE
    results[6,:] = np.nanmedian(zape_list, axis = 1)
    results[7,:] = np.nanquantile(zape_list, 0.25, axis = 1)
    results[8,:] = np.nanquantile(zape_list, 0.75, axis = 1)

    ## Save Results
    df = pd.DataFrame(np.round(results, 2))
    df.columns = ['m' + str(i) for i in np.arange(1, 3 + 1)]
    df.index = ['MAD Median', 'MAD 25%', 'MAD 75%', 
                'MAPE Median', 'MAPE 25%', 'MAPE 75%',
                'ZAPE Median', 'ZAPE 25%', 'ZAPE 75%'] 
    df.to_csv('simulated_results/RESULTS/ITEM/HH-'+str(hhg)+'-ITEM='+ \
                    str(item)+ '-k1-summary-QUANTILES.csv')  
    if item == 'A':
        ## Save optimal forecasts for each HH
        fn = 'simulated_results/RESULTS/ITEM/HH-'+str(hhg)+'-ITEM='+ \
                        str(item)+ '-k1-MAD-optf'
        np.savez(fn, optf_mad_m1, optf_mad_m2, optf_mad_m3,
                    optf_mzape_m1, optf_mzape_m2, optf_mzape_m3)
        fn = 'simulated_results/RESULTS/ITEM/HH-'+str(hhg)+'-ITEM='+ \
                        str(item)+'-k1-opt-metrics'
        np.savez(fn, opt_mad, zape_list)

    ###############################
    ## (1) Make Calibration Plots
    
    calib1 = -10*np.ones((numhh, 104))
    calib2 = -10*np.ones((numhh, 104))
    for j in range(numhh):
        calib1[j, :inds[j]] = [len(np.where(samples[1][j,:, t] != 0)[0])/samples[1].shape[1]
                                for t in range(inds[j])]
        calib2[j, :inds[j]] = [len(np.where(samples[2][j,:, t] != 0)[0])/samples[2].shape[1]
                                for t in range(inds[j])]

    yacc = y[:,:104].flatten()
    inds = np.where(yacc != -10) ## Values to end forecasting at == -10
    yacc = yacc[inds]
    yacc[np.where(yacc != 0)] = 1
    calib1 = calib1[:,:].flatten()[inds]
    phat1, diff1 = calc_calib(calib1, yacc.flatten())

    calib2 = calib2[:,:].flatten()[inds]
    phat2, diff2 = calc_calib(calib2, yacc.flatten())

       
    bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] 
    
    

    _, x = np.histogram(calib2, bins = bins)
    plt.figure()
    plt.figure(figsize = (10,8))
    plt.scatter((x-0.05)[1:], phat1, label = 'No Discount', s = 100)
    plt.errorbar((x-0.05)[1:], phat1, yerr = diff1, linewidth = 3)
    plt.scatter((x-0.05)[1:], phat2, label = 'Agg. Discount Percent', color = 'C1', s = 100)
    plt.errorbar((x-0.05)[1:], phat2, yerr = diff2, color = 'C1', linewidth = 3)
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.plot([0,1],[0,1], '--', color = 'black')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.fill_between(np.array(bins), np.array(bins) - 0.007, np.array(bins) + 0.007, 
                        color='grey', alpha=.5)
    plt.legend()
    plt.tight_layout()
    fn = 'simulated_results/RESULTS/ITEM/HH-'+str(hhg)+'-ITEM='+ \
                    str(item)+ '-calibration2.png'
    plt.savefig(fn)

##############################
for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
    analyze_ITEM(1, i) 
    analyze_ITEM(2, i)
    analyze_ITEM(3, i)  

#################################################################
## Individual Forecasts - Item Level Modeling
## Plots and Graphic Comparisons 
###########################################################

sns.set()
sns.set(font_scale=2)
  ##### Plotting - ITEM-Level Modeling
def plot_item(hhg, item):
    k = 1
    # Compare individual forecasts - models 
    ## Individual discount since aggregate discount not as impactful
    fn = 'simulated_results/RESULTS/ITEM/HH-'+str(hhg)+'-ITEM='+ \
                    str(item)+ '-k1-MAD-optf.npz'
    f = np.load(fn)
    optf_mad_m1 = f['arr_0'] 
    optf_mad_m2 = f['arr_1']  
    optf_mad_m3 = f['arr_2']  
    optf_mzape_m1 = f['arr_3'] 
    optf_mzape_m2 = f['arr_4'] 
    optf_mzape_m3 = f['arr_5'] 
    fn = 'simulated_results/RESULTS/ITEM/HH-'+str(hhg)+'-ITEM='+ \
                    str(item)+'-k1-opt-metrics.npz'
    f = np.load(fn)
    mad_list = f['arr_0'] 
    zape_list = f['arr_1']

    fname = 'simulated_results/ITEM/HH-'+str(hhg)+ \
                                    '-ITEM='+str(item)+'-k1-m2-m.npy'    
    m2 = np.load(fname)
    fname = 'simulated_results/ITEM/HH-'+str(hhg)+ \
                                    '-ITEM='+str(item)+'-k1-m2-C.npy'    
    C2 = np.load(fname)

    fname = 'simulated_results/ITEM/HH-'+str(hhg)+ \
                                    '-ITEM='+str(item)+'-k1-m1-m.npy'    
    m1 = np.load(fname)
    fname = 'simulated_results/ITEM/HH-'+str(hhg)+ \
                                    '-ITEM='+str(item)+'-k1-m1-C.npy'    
    C1= np.load(fname)


    fname = 'simulated_results/ITEM/HH-'+str(hhg)+ \
                                    '-ITEM='+str(item)+'-k1-true.npy'    
    y = np.load(fname)          
    y = y[:, :104]
    print(y)
    ## Find indices to forecast over
    inds = np.array([np.where(y[i,:] == -10)[0][0] if y[i, -1] == -10 else y.shape[1]
                                for i in range(y.shape[0])])
    ## Find indices of HHs to use
    drop_inds = np.where(inds == 0)[0]
    hh_inds = np.arange(y.shape[0])
    if len(drop_inds) > 0:
        hh_inds = np.delete(hh_inds, drop_inds)
        inds = np.delete(inds, drop_inds)
        print(drop_inds, len(hh_inds))

    numhh = len(hh_inds)
    y = y[hh_inds, :]
    print(numhh)
    print(inds)
    print(hh_inds)
    print(y)
    ## Load in results of modeling
    fname = 'simulated_results/ITEM/HH-'+str(hhg)+ \
                            '-ITEM='+str(item)+'-k1-m2-samples.npy' 
    samples2 = np.load(fname)[hh_inds, :, :]
    fname = 'simulated_results/ITEM/HH-'+str(hhg)+ \
                            '-ITEM='+str(item)+'-k1-m1-samples.npy'
    samples1 = np.load(fname)[hh_inds, :, :]
    
    m1 = m1[hh_inds, :104, :] 
    C1 = C1[hh_inds, :104, :, :]
    m2 = m2[hh_inds, :104, :] 
    C2 = C2[hh_inds, :104, :, :]
    print(m1.shape, m2.shape)
    ul = np.ceil(np.max(mad_list))
    ul = np.min([ul, 10])
    sns.set(font_scale = 2.5)
    plt.figure(figsize = (10,8))
    plt.scatter(mad_list[1, :], mad_list[0, :], color = 'C0', label = 'MAD', s = 100)
    plt.plot([0, ul], [0, ul], '--', color = 'black')
    plt.xlim(0, ul)
    plt.ylim(0, ul)
    plt.xlabel('No Discount')
    plt.ylabel('Individ. Discount Percent')
    plt.tight_layout()
    fn = 'simulated_results/RESULTS/ITEM/HH-'+str(hhg)+'-ITEM='+ \
                    str(item)+ '-k1-m2_v_m1-scatter-MAD.png'
    plt.savefig(fn)

    ul = np.ceil(np.max(zape_list))
    ul = np.min([ul, 5])
    plt.figure(figsize = (10,8))
    plt.scatter(zape_list[1, :], zape_list[0, :], color = 'C1', label = 'ZAPE', s = 100)
    plt.plot([0, ul], [0, ul], '--', color = 'black')
    plt.xlim(0, ul)
    plt.ylim(0, ul)
    plt.xlabel('No Discount')
    plt.ylabel('Individ. Discount Percent')
    # plt.title('ZAPE')
    plt.tight_layout()
    fn = 'simulated_results/RESULTS/ITEM/HH-'+str(hhg)+'-ITEM='+ \
                    str(item)+ '-k1-m2_v_m1-scatter-ZAPE.png'
    plt.savefig(fn)
    plt.close('all')


    # print(np.sort(mad_list[2, :] - mad_list[1, :]))
    # plinds_mad = np.argsort(mad_list[2, :] - mad_list[1, :])[:5]
    # plinds_zape = np.argsort(zape_list[2, :] - mad_list[1, :])[:5]
    # print(plinds_zape)
    sns.set(font_scale = 2)
    # plinds_mad = [0, 1, 2, 2, 4, 5]
    for j in range(20): 
        yplot = y[j, :inds[j]]
        plt.figure()
        plt.figure(figsize = (16,5))
        plt.plot(optf_mad_m2[j, :inds[j]], label = 'No Discount', linewidth=2.5,
                    color = 'C2')
        plt.plot(optf_mad_m1[j, :inds[j]], label = 'Discount Percent', linewidth=2.5,
                    color = 'C0')
        plt.scatter(np.arange(len(yplot)), yplot, color = 'black')
        lwr = np.quantile(samples1[j, :, :inds[j]], 0.05, axis = 0) ## 90% Interval over HH
        upr = np.quantile(samples1[j, :, :inds[j]], 0.95, axis = 0)
        plt.fill_between(np.arange(len(yplot)), lwr, upr, color='C0', 
                            alpha=.5)
        plt.xlabel('Week')
        plt.ylabel('Item Quantity')
        plt.legend()
        plt.tight_layout()
        fn = 'simulated_results/RESULTS/ITEM/HH-'+str(hhg)+'-ITEM='+ \
                    str(item)+'-' + str(j) + '-m1-best-MAD.png'
        plt.savefig(fn)
        plt.close('all')
        
        ## Plot mt and Ct
        T1 = len(np.arange(1,inds[j]))
        BS1 = np.array([np.random.multivariate_normal(m1[j, t, :], 
                                              C1[j, t, :, :], 1000) for t in range(1, inds[j])])
       
        plt.figure(figsize = (16,5))
        plt.plot(np.mean(BS1[:,:,2], axis = 1), label = 'mean', color = 'black', linewidth = 2.5)
        lwr = np.quantile(BS1[:,:,2], 0.05, axis = 1) ## 90% Posterior Credible Interval
        upr = np.quantile(BS1[:,:,2], 0.95, axis = 1)
        plt.fill_between(np.arange(T1), lwr, upr, color='b', alpha=.5)
        plt.axhline(y=0, color='grey', linestyle='--')
        plt.xlabel('Week')
        plt.ylabel(r'$m_t$ - Discount Perc.')
        plt.tight_layout()
        fn = 'simulated_results/RESULTS/ITEM/HH-'+str(hhg)+'-ITEM='+ \
                    str(item)+'-' + str(j) + '-m1-best-mC-MAD-discount.png'
        plt.savefig(fn)
        plt.close('all')
        plt.close('all')

##################################
plot_item(1, 'A')


########################
## DIRECT ITEM #########
########################

def analyze_ITEM_DIRECT(hhg, item):

    fname = 'simulated_results/ITEM/HH-'+str(hhg)+ \
                                    '-ITEM='+str(item)+'-k1-true-DIRECT.npy'    
    y = np.load(fname)          
    y = y[:, :104]


    ## Find indices to forecast over
    inds = np.array([np.where(y[i,:] == -10)[0][0] if y[i, -1] == -10 else y.shape[1]
                                for i in range(y.shape[0])])
    ## Find indices of HHs to use
    drop_inds = np.where(inds == 0)[0]
    hh_inds = np.arange(y.shape[0])
    if len(drop_inds) > 0:
        hh_inds = np.delete(hh_inds, drop_inds)
        inds = np.delete(inds, drop_inds)
        print(drop_inds, len(hh_inds))

    numhh = len(hh_inds)
    y = y[hh_inds, :]
    
    ## Load in results of modeling
    fm = list()
    samples = list()
    for i in [1,2,3, 4]:
        fname = 'simulated_results/ITEM/HH-'+str(hhg)+ \
                            '-ITEM='+str(item)+'-k1-m' + \
                            str(i) + '-samples-DIRECT.npy'    
        samples1 = np.load(fname)
        samples.append(samples1[hh_inds, :, :104])
        fm.append(np.median(samples1[hh_inds, :, :104], axis = 1))
  
    ## Array for saving results of analysis
    numm = len(fm) ## number of models
    
    ### (1) MAD vs. model
    opt_mad = np.zeros((numm, numhh))
    optf_mad = [np.zeros((numhh, 104))]*numm
    # for i in range(21):
    for i in range(numm):
        for j in range(numhh):
            # if inds[j] > 0:
            opt_mad[i, j] = mad(y[j, :inds[j]], fm[i][j,:inds[j]])
            optf_mad[i][j, :inds[j]] = fm[i][j,:inds[j]]
            
    print('MAD Done')
    
    # ## (4) MAPE and mZAPE, 
    mape_list = np.zeros((numm, numhh))
    zape_list = np.zeros((numm, numhh))
    optf_mzape = [np.zeros((numhh, 104))]*numm
    
    for i in range(numm):
        for j in range(numhh):
            f_MAPE, f_mZAPE = np.array([fstar_counts(samples[i][j,:, t])
                                for t in range(inds[j])]).T
            mape_list[i, j] = mape(y[j, :inds[j]], f_MAPE)
            zape_list[i, j] = zape(y[j, :inds[j]], f_mZAPE, 'mzape')
            optf_mzape[i][j, :inds[j]] = f_mZAPE 
            
    
    ## Subset to appropriate values - MAD
    results = np.zeros((9, numm))
    results[0,:] = np.nanmedian(opt_mad, axis = 1)
    results[1,:] = np.nanquantile(opt_mad, 0.25, axis = 1)
    results[2,:] = np.nanquantile(opt_mad, 0.75, axis = 1)
    # Subset to appropriate values - MAPE
    results[3,:] = np.nanmedian(mape_list, axis = 1)
    results[4,:] = np.nanquantile(mape_list, 0.25, axis = 1)
    results[5,:] = np.nanquantile(mape_list, 0.75, axis = 1)
    # Subset to appropriate values - ZAPE
    results[6,:] = np.nanmedian(zape_list, axis = 1)
    results[7,:] = np.nanquantile(zape_list, 0.25, axis = 1)
    results[8,:] = np.nanquantile(zape_list, 0.75, axis = 1)

    ## Save Results
    df = pd.DataFrame(np.round(results, 2))
    df.columns = ['m' + str(i) for i in np.arange(1, 4 + 1)]
    df.index = ['MAD Median', 'MAD 25%', 'MAD 75%', 
                'MAPE Median', 'MAPE 25%', 'MAPE 75%',
                'ZAPE Median', 'ZAPE 25%', 'ZAPE 75%'] 
    df.to_csv('simulated_results/RESULTS/ITEM/HH-'+str(hhg)+'-ITEM='+ \
                    str(item)+ '-k1-summary-QUANTILES-DIRECT.csv')  
    

##############################
for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
    analyze_ITEM_DIRECT(1, i) 
    analyze_ITEM_DIRECT(2, i)
    analyze_ITEM_DIRECT(3, i) 


###############################
### Simultaneous ##############
###############################

def simul_analysis(hhg, item, method):
    k = 1
    fname = 'simulated_results/simul/HH-'+str(hhg)+ \
                                        '-ITEM='+str(item)+'-k'+ str(k) + \
                                        '-true-TRUE-COND.npy'    
    ytl = np.load(fname)[:, :104]         

    fname = 'simulated_results/simul/HH-'+str(hhg)+ \
                                '-ITEM='+str(item)+'-k'+ str(k) + '-' + \
                                '-dates-TRUE-COND'
    with open (fname, 'rb') as fp:
        date_list_tl = pickle.load(fp)
    
    fname = 'simulated_results/simul/HH-'+str(hhg)+ \
                                        '-ITEM='+str(item)+'-k1-' + \
                                        str(method) + '-true.npy'     
    y = np.load(fname)[:, :104] 
    fname = 'simulated_results/simul/HH-'+str(hhg)+ \
                                '-ITEM='+str(item)+'-k'+ str(k) + '-' + \
                                str(method) + '-dates'
    with open (fname, 'rb') as fp:
        date_list = pickle.load(fp)

 
    ## find all dates
    ii = np.argmax(np.array([len(date_list_tl[i]) for i in range(len(date_list_tl))]))
    # print(date_list_tl[ii].values[7:])
    unique_dates = date_list_tl[ii].values[7:-1]
    print(np.sort(np.array([len(date_list_tl[i]) for i in range(len(date_list_tl))]))[::-1])
    

    ## Find where HHs modeled that shouldn't have been
    msimul_mad = -10*np.ones(y.shape)
    msimul_zape = -10*np.ones(y.shape)
    mtrue = -10*np.ones(ytl.shape)


    inds_tl = np.array([np.where(ytl[i,:] == -10)[0][0] if ytl[i, -1] == -10 else ytl.shape[1]
                                    for i in range(ytl.shape[0])])
    ## Find indices of HHs to use
    drop_inds_tl = np.where(inds_tl == 0)[0]
    print(drop_inds_tl)
    hh_inds_tl = np.arange(ytl.shape[0])
    if len(drop_inds_tl) > 0:
        mtrue[drop_inds_tl,:] = 2

    ##################################################
    inds = np.array([np.where(y[i,:] == -10)[0][0] if y[i, -1] == -10 else y.shape[1]
                                    for i in range(y.shape[0])])
    ## Find indices of HHs to use
    drop_inds = np.where(inds == 0)[0]
    print(drop_inds)
    hh_inds = np.arange(y.shape[0])
    if len(drop_inds) > 0:
        msimul_mad[drop_inds,:] = 2
        msimul_zape[drop_inds,:] = 2
    fname = 'simulated_results/simul/HH-'+str(hhg)+ \
                                str(item)+'-k'+ str(k) + '-' + str(method) + '-samples.npy'    
    samples = np.load(fname)[:, :, :104]
    fm = np.median(samples, axis = 1)


    numhh = y.shape[0]
    results2 = np.zeros((9,1))
    
    
    ### (1) MAD vs. model
    opt_mad = np.zeros(numhh)
    optf_mad = -10*np.ones((numhh, 104))
    for j in range(y.shape[0]):
        if inds[j] > 0:
            opt_mad[j] = mad(y[j, :inds[j]], fm[j,:inds[j]])
            optf_mad[j, :inds[j]] = fm[j,:inds[j]]

    print('MAD Done')
    
    results2[0,:] = np.nanmedian(opt_mad[np.where(opt_mad < 25)[0]])
    results2[1,:] = np.nanquantile(opt_mad[np.where(opt_mad < 25)[0]], 0.25)
    results2[2,:] = np.nanquantile(opt_mad[np.where(opt_mad < 25)[0]], 0.75)

    ## (2) MAPE and mZAPE, 
    
    zape_list = np.zeros(numhh)
    mape_list = np.zeros(numhh)
    mape_list = np.zeros(numhh)
    optf_zape = -10*np.ones((numhh, 104))
    for j in range(numhh):
        if inds[j] > 0:
            f_MAPE, f_mZAPE = np.array([fstar_counts(samples[j,:, t])
                                for t in range(inds[j])]).T
            mape_list[j] = mape(y[j, :inds[j]], f_MAPE)
            zape_list[j] = zape(y[j, :inds[j]], f_mZAPE, 'mzape')
            optf_zape[j, :inds[j]] = f_mZAPE 
    

    results2[3,:] = np.nanmedian(mape_list[np.where(mape_list < 5)[0]])
    results2[4,:] = np.nanquantile(mape_list[np.where(mape_list < 5)[0]], 0.25)
    results2[5,:] = np.nanquantile(mape_list[np.where(mape_list < 5)[0]], 0.75)


    results2[6,:] = np.nanmedian(zape_list[np.where(zape_list < 5)[0]])
    results2[7,:] = np.nanquantile(zape_list[np.where(zape_list < 5)[0]], 0.25)
    results2[8,:] = np.nanquantile(zape_list[np.where(zape_list < 5)[0]], 0.75)

    ## Save Results

    df = pd.DataFrame(np.round(results2, 2))
    df.index = ['MAD Median', 'MAD 25%', 'MAD 75%', 
                'MAPE Median', 'MAPE 25%', 'MAPE 75%',
                'ZAPE Median', 'ZAPE 25%', 'ZAPE 75%'] 
    df.to_csv('simulated_results/RESULTS/simul/HH-'+str(hhg)+ \
              '-ITEM='+str(item)+'-k1' + str(method)+ '-k1-summary-QUANTILES.csv')

    print('Done', hhg, method)
#######################
k = 1
item = 'A'

for hhg in [1,2,3]:
    simul_analysis(hhg, item, 'mean')
    simul_analysis(hhg, item, 'median')




###################################
## Calibration - Eval conditioning - MAD / median

def load_data(hhg, cat, sub, item, method):
    fname = 'simulated_results/simul/HH-' + str(hhg)+ '-k' + str(k) + \
                                            '-true-pReturn.npy'    
    y_pret = np.load(fname)          
    fname = 'simulated_results/simul/HH-' + str(hhg)+ '-k' + str(k) + \
                                            '-samples-pReturn.npy'    
    samps_pret = np.load(fname)
    #### log total spend
    fname = 'simulated_results/simul/HH-' + str(hhg)+ '-k' + str(k) + str(method) +\
                                            '-true-logtotal.npy'    
    y_log = np.load(fname)          
    fname = 'simulated_results/simul/HH-' + str(hhg)+ '-k' + str(k) + \
                                            '-' + str(method) + '-samples-logtotal.npy'    
    samps_log = np.load(fname)
    ### CAT 
    fname = 'simulated_results/simul/HH-'+str(hhg)+'-CAT='+str(cat)+'-k'+ \
                                            str(k) +'-' + str(method) + '-true.npy'    
    y_cat = np.load(fname)          
    fname = 'simulated_results/simul/HH-'+str(hhg)+'-CAT='+str(cat)+ \
                        '-k'+str(k) + '-' + str(method) + '-samples.npy'    
    samps_cat = np.load(fname)
    ## SUB-CAT
    fname = 'simulated_results/simul/HH-'+str(hhg)+'-SUB='+str(sub)+ \
                                    '-k'+ str(k) +'-' + str(method)+ '-true.npy'    
    y_sub = np.load(fname)          
    fname = 'simulated_results/simul/HH-'+str(hhg)+'-SUB='+str(sub)+ \
                                '-k'+ str(k) + '-' + str(method) + '-samples.npy'    
    samps_sub = np.load(fname)
    ## ITEM
    fname = 'simulated_results/simul/HH-'+str(hhg)+ \
                                    '-ITEM='+str(item)+'-k'+ str(k) + \
                                    '-' + str(method) + '-true.npy'    
    y_item = np.load(fname)          
    fname = 'simulated_results/simul/HH-'+str(hhg)+ \
                                str(item)+'-k'+ str(k) + '-' + str(method) + '-samples.npy'    
    samps_item = np.load(fname)
    return(y_pret, samps_pret, y_log, samps_log, y_cat, samps_cat,
                    y_sub, samps_sub, y_item, samps_item)


def calibibration_plots(hh_list, hhg, cat, sub, item, df_global, df_cat, df_sub, df_item, 
                                flist_return, flist_cat, flist_sub, method):

    ## Really want to store how much the 0s are missed
    ## Calibration for points that are correctly modeled
    ## Load data
    #samples and y true for all simultaneous levels
    y_pret, samps_pret, y_log, samps_log, y_cat, samps_cat, \
                    y_sub, samps_sub, y_item, samps_item = load_data(hhg, cat, sub, item, method)

    count = 0
    preturn_calib = -10*np.ones((len(hh_list), 4))
    cat_calib = -10*np.ones((len(hh_list), 4))
    sub_calib = -10*np.ones((len(hh_list), 4))

    g2_save = -10*np.ones((len(hh_list), 104))
    g1_save = -10*np.ones((len(hh_list), 104))
    cat_y = -10*np.ones((len(hh_list), 104))
    sub_y = -10*np.ones((len(hh_list), 104))
    
    for i in hh_list:
        ## Select one HH
        df_i = df_item.loc[df_item['HH'] == i].iloc[7:-1,:] ### ITEM level
        df_i1 = df_sub.loc[df_sub['HH'] == i].iloc[7:-1,:] ### SUB level
        df_i2 = df_cat.loc[df_cat['HH'] == i].iloc[7:-1,:] ### CAT level
        df_ig = df_global.loc[df_global['HH'] == i].iloc[7:-1,:] ## Global level
        ########################
        ## p(RETURN)
        ## Where do we predict return?
        ind_true = np.where(df_ig['RETURN']==1 )[0] 
        ## Where does HH actually return
        ind_ret = np.where(np.round(flist_return[count, :]) == 1)[0]
        ## find inds correctly predicted to model
        inds_int = np.where(np.in1d(ind_ret, ind_true))[0]

        ## Which inds_ret are "extra" - predict return but didnt
        inds_ex = np.setdiff1d(ind_ret, ind_true)
        ## Which inds_ret were missed - predict no return but did
        inds_miss = np.setdiff1d(ind_true, ind_ret)

        inds0 = 104 - len(inds_int)- len(inds_ex)- len(inds_miss)
        preturn_calib[count,:] = [len(inds_int), len(inds_ex), len(inds_miss), inds0]
        ###################################################
        ## CAT
        ## Where do we predict return for cat?
        ind_ret = np.where(~np.isin(flist_cat[count,:], [0,-10]))[0]
        ## Where does HH actually return
        ind_true = np.where(df_i2['RETURN']==1 )[0]
        ## find inds correctly predicted
        inds_int = np.where(np.in1d(ind_ret, ind_true))[0]
        
        ## Which inds_ret are "extra" - predict return but didnt
        inds_ex = np.setdiff1d(ind_ret, ind_true)
        ## Which inds_ret were missed - predict no return but did
        inds_miss = np.setdiff1d(ind_true, ind_ret)
        inds0 = 104 - len(inds_int)- len(inds_ex)- len(inds_miss)
        cat_calib[count,:] = [len(inds_int), len(inds_ex), len(inds_miss), inds0]
        ########################################################
        ## SUB
        ## select where return for sub-cat
        ind_ret = np.where(~np.isin(flist_sub[count,:], [0,-10]))[0]
        ind_true = np.where(df_i1['RETURN']==1 )[0]
        ## find inds correctly predicted
        inds_int = np.where(np.in1d(ind_ret, ind_true))[0]
        
        ## Which inds_ret are "extra" - predict return but didnt
        inds_ex = np.setdiff1d(ind_ret, ind_true)
        ## Which inds_ret were missed - predict no return but did
        inds_miss = np.setdiff1d(ind_true, ind_ret)
        inds0 = 104 - len(inds_int)- len(inds_ex)- len(inds_miss)
        sub_calib[count,:] = [len(inds_int), len(inds_ex), len(inds_miss), inds0]


        count += 1

    ## Drop HHs not modeled
    preturn_calib = preturn_calib[np.where(np.sum(preturn_calib, axis = 1))!= -30][0,:,:]
    cat_calib = cat_calib[np.where(np.sum(cat_calib, axis = 1))!= -30][0,:,:]
    sub_calib = sub_calib[np.where(np.sum(sub_calib, axis = 1))!= -30][0,:,:]

    ## Confusion matrix
    results = np.zeros((6, 4))
    results[0,:] = np.sum(preturn_calib, axis = 0)
    results[1,:] = np.round(np.sum(preturn_calib, axis = 0)/np.sum(np.sum(preturn_calib, axis = 0)),2)
    results[2,:] = np.sum(cat_calib, axis = 0)
    results[3,:] = np.round(np.sum(cat_calib, axis = 0)/np.sum(np.sum(cat_calib, axis = 0)),2)
    results[4,:] = np.sum(sub_calib, axis = 0)
    results[5,:] = np.round(np.sum(sub_calib, axis = 0)/np.sum(np.sum(sub_calib, axis = 0)),2)

    df = pd.DataFrame(results)
    df.columns = ['Corr1', 'Extra', 'Miss', 'Corr0']
    df.index = ['pReturn', 'pReturn', 'CAT', 'CAT', 'SUB', 'SUB']
    df.to_csv('simulated_results/RESULTS/simul/HH-'+str(hhg)+ \
              '-ITEM='+str(item)+'-' + str(method)+'-k1-summary-CONFUSION.csv')

    print(preturn_calib.shape)
    ## Normalize counts
    preturn_calib = preturn_calib/np.sum(preturn_calib, axis = 1)[:, np.newaxis]
    cat_calib = cat_calib/np.sum(cat_calib, axis = 1)[:, np.newaxis]
    sub_calib = sub_calib/np.sum(sub_calib, axis = 1)[:, np.newaxis]
    results = np.zeros((12, 3))
    results[:,0] = np.concatenate([np.median(preturn_calib, axis = 0),
                        np.quantile(preturn_calib, 0.25, axis = 0),
                        np.quantile(preturn_calib, 0.75, axis = 0)])
    results[:,1] = np.concatenate([np.median(cat_calib, axis = 0),
                        np.quantile(cat_calib, 0.25, axis = 0),
                        np.quantile(cat_calib, 0.75, axis = 0)])
    results[:,2] = np.concatenate([np.median(sub_calib, axis = 0),
                        np.quantile(sub_calib, 0.25, axis = 0),
                        np.quantile(sub_calib, 0.75, axis = 0)])

    results = np.round(results, 2)

    df = pd.DataFrame(results)
    df.columns = ['Return', 'GRP2', 'GRP1']
    df.index = ['Int-Median', 'Int-25%', 'Int-75%',
                    'Ex-Median', 'Ex-25%', 'Ex-75%',
                    'Miss-Median', 'Miss-25%', 'Miss-75%',
                    'Corr0-Median', 'Corr0-25%', 'Corr0-75%']
    df.to_csv('paper_results/simul/Plots-New/HH-'+str(hhg)+ \
              '-ITEM='+str(item)+'-' + str(method)+'-k1-summary-CALIB.csv')

    ## Calculate calibration 20:
    phat1, diff1 = calc_calib(np.mean(samps_pret[:,:,20:], axis = 1), y_pret[:,20:])

    inds_cat = np.array([np.where(y_cat[i,:] == -10)[0][0] if y_cat[i, -1] == -10 else y_cat.shape[1]
                                for i in range(y_cat.shape[0])])
    cat_cov = -10*np.ones((len(hh_list), 104))
    inds_sub = np.array([np.where(y_sub[i,:] == -10)[0][0] if y_sub[i, -1] == -10 else y_sub.shape[1]
                                for i in range(y_sub.shape[0])])
    sub_cov = -10*np.ones((len(hh_list), 104))
    inds_item = np.array([np.where(y_item[i,:] == -10)[0][0] if y_item[i, -1] == -10 else y_item.shape[1]
                                for i in range(y_item.shape[0])])
    item_cov = -10*np.ones((len(hh_list), 104))
    for j in range(len(hh_list)):
        cat_cov[j, :inds_cat[j]] = [len(np.where(samps_cat[j,:, t] != 0)[0])/samps_cat.shape[1]
                                for t in range(inds_cat[j])]
        sub_cov[j, :inds_sub[j]] = [len(np.where(samps_sub[j,:, t] != 0)[0])/samps_sub.shape[1]
                                for t in range(inds_sub[j])]
        item_cov[j, :inds_item[j]] = [len(np.where(samps_item[j,:, t] != 0)[0])/samps_item.shape[1]
                                for t in range(inds_item[j])]

    print(item_cov[:10, :])
    
    yacc = y_cat[:,:104].flatten()
    inds = np.where(yacc != -10) ## Values to end forecasting at == -10
    yacc = yacc[inds]
    yacc[np.where(yacc != 0)] = 1
    cat_cov = cat_cov[:,:].flatten()[inds]
    phat2, diff2 = calc_calib(cat_cov, yacc.flatten())

    yacc = y_sub[:,:104].flatten()
    inds = np.where(yacc != -10) ## Values to end forecasting at == -10
    yacc = yacc[inds]
    yacc[np.where(yacc != 0)] = 1
    sub_cov = sub_cov[:,:].flatten()[inds]
    phat3, diff3 = calc_calib(sub_cov, yacc.flatten())

    yacc = y_item[:,:104].flatten()
    inds = np.where(yacc != -10) ## Values to end forecasting at == -10
    yacc = yacc[inds]
    yacc[np.where(yacc != 0)] = 1
    item_cov = item_cov[:,:].flatten()[inds]
    phat4, diff4 = calc_calib(item_cov, yacc.flatten())
    
    bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] 
    _, x = np.histogram(item_cov, bins = bins)
    plt.figure()
    plt.figure(figsize = (10,8))
    plt.scatter((x-0.05)[1:], phat1, label = 'Return', s = 100)
    plt.errorbar((x-0.05)[1:], phat1, yerr = diff1, linewidth = 3)
    plt.scatter((x-0.05)[1:], phat2, label = 'Cat.', color = 'C1', s = 100)
    plt.errorbar((x-0.05)[1:], phat2, yerr = diff2, color = 'C1', linewidth = 3)
    plt.scatter((x-0.05)[1:], phat3, label = 'Sub-Cat.', color = 'C2', s = 100)
    plt.errorbar((x-0.05)[1:], phat3, yerr = diff3, color = 'C2', linewidth = 3)
    plt.scatter((x-0.05)[1:], phat4, label = 'Item', color = 'C3', s = 100)
    plt.errorbar((x-0.05)[1:], phat4, yerr = diff4, color = 'C3', linewidth = 3)
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.plot([0,1],[0,1], '--', color = 'black')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.fill_between(np.array(bins), np.array(bins) - 0.007, np.array(bins) + 0.007, 
                        color='grey', alpha=.5)
    plt.legend()
    plt.tight_layout()
    fn = 'simulated_results/RESULTS/simul/HH-'+str(hhg)+ \
              '-ITEM='+str(item)+'-' + str(method)+'-calibration2.png'
    plt.savefig(fn)

        
########################################################
def run_calib(hhg, cat, sub, item, method):
    k = 1 # Forecast 1 step ahead
     
    df_hh1 = pd.read_csv('Data/HH'+str(hhg)+'-DATA.csv')
    ## Group over all items for global summaries
    dfN = df_hh1.groupby(['HH','WEEK']).sum()[['ITEM_QTY', 'TOTAL_SPEND']].reset_index()
    dfH = df_hh1.groupby(['HH','WEEK', 'CAT']).sum()[['ITEM_QTY', 'TOTAL_SPEND']].reset_index()
    dfG = df_hh1.groupby(['HH','WEEK', 'CAT', 'SUB_CAT']).sum()[['ITEM_QTY', 'TOTAL_SPEND']].reset_index()
    dfI = df_hh1
    dfN['RETURN'] = np.where(dfN['ITEM_QTY'] > 0, 1, 0)
    dfG['RETURN'] = np.where(dfG['ITEM_QTY'] > 0, 1, 0)
    dfH['RETURN'] = np.where(dfH['ITEM_QTY'] > 0, 1, 0)
    dfI['RETURN'] = np.where(dfI['ITEM_QTY'] > 0, 1, 0)
    
    # Subset
    hh_list = np.unique(dfN['HH']) ## all HHs

    dfG1 = dfG[dfG['CAT'] == cat]
    dfH1 = dfH[dfH['CAT'] == cat]
    dfI1 = dfI[dfI['CAT'] == cat]

    dfG1 = dfG1[dfG1['SUB_CAT'] == sub]
    dfI1 = dfI1[dfI1['SUB_CAT'] == sub]

    dfI1 = dfI1[dfI1['ITEM'] == item]
    if method == 'mean':
        fname = 'simulated_results/simul/HH-' + str(hhg)+  '-k' + str(k) + \
                                            '-samples-pReturn.npy'   
        samples = np.load(fname)
        flist_return = np.mean(samples, axis = 1)
        fname = 'simulated_results/simul/HH-'+str(hhg)+'-CAT='+str(cat)+'-k'+str(k) + \
                                                '-' + str(method) + '-samples.npy'    
        samples = np.load(fname)
        flist_cat = np.mean(samples, axis = 1)
        fname = 'simulated_results/simul/HH-'+str(hhg)+ \
                                        '-SUB='+str(sub)+'-k'+ str(k) +'-' + \
                                        str(method) + '-samples.npy'  
        samples = np.load(fname)
        flist_sub = np.mean(samples, axis = 1)

    if method == 'median':
        fname = 'simulated_results/simul/HH-' + str(hhg)+  '-k' + str(k) + \
                                            '-samples-pReturn.npy'   
        samples = np.load(fname)
        flist_return = np.median(samples, axis = 1)
        fname = 'simulated_results/simul/HH-'+str(hhg)+'-CAT='+str(cat)+'-k'+str(k) + \
                                                '-' + str(method) + '-samples.npy'    
        samples = np.load(fname)
        flist_cat = np.median(samples, axis = 1)
        fname = 'simulated_results/simul/HH-'+str(hhg)+ \
                                        '-SUB='+str(sub)+'-k'+ str(k) +'-' + \
                                        str(method) + '-samples.npy'  
        samples = np.load(fname)
        flist_sub = np.median(samples, axis = 1)


    calibibration_plots(hh_list, hhg, cat, sub, item, dfN, dfH1, dfG1, dfI1, 
                                flist_return, flist_cat, flist_sub, method)

#####################
k = 1
item = 'A'
sub = '1A'
cat = 1

for hhg in [1,2,3]:
    run_calib(hhg, cat, sub, item, 'mean')
    run_calib(hhg, cat, sub, item, 'median')
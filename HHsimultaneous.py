import numpy as np
import pandas as pd

from pybats.analysis import analysis, analysis_dcmm, analysis_dlmm
from pybats.point_forecast import median
from pybats.define_models import define_dglm, define_dcmm, define_dlmm

def mad(y, f):
    diffterm = np.abs(y.flatten() - f.flatten())
    return(np.mean(diffterm))

def mse(y, f):
    diffterm = (y.flatten() - f.flatten())**2
    return(np.mean(diffterm))

## (1) pReturn
## df = input dataframe
## hhg = household group
## k = forecast horizon
## hh_list = list of households to model
def pReturn(df, hhg, k, hh_list):
    
    disfact_trend = np.array([0.9, 0.93, 0.95, 0.97, 0.99])
    disfact_covar = np.array([0.9, 0.93, 0.95, 0.97, 0.99])
    ###################################################################
    ### P(Return) for HH that do not return every week
    #Forecast every time point 

    ### Forecasting
    forecast_start = 7                                       
    forecast_end = 109 - k
    dind = (forecast_end - forecast_start) + 1

    true_list = np.zeros((len(hh_list), dind))
    samps = np.zeros((len(hh_list), 300, dind))
    
    mse_list = np.zeros((len(hh_list), len(disfact_trend), len(disfact_covar)))
    
    count = 0
    for i in hh_list:
        ## Select model
        df1 = df.loc[df['HH'] == i]
        ### Log TOTAL_SPEND_t-1
        Y = df1['RETURN'].values[1:]
        X = np.where(df1['TOTAL_SPEND'].values[:-1] > 0, 
                      np.log(df1['TOTAL_SPEND'].values[:-1]), 0).reshape(-1,1)
        X = np.round(X - np.mean(X), 2)

        ## Tune discount factor for each HH 
        # prior a0 = 0, R0 = I
        a0 = np.zeros(X.shape[1] + 1)
        R0 = np.eye(X.shape[1] + 1)    
        model_prior = define_dglm(Y, X, a0 = a0, R0 = R0, family='bernoulli', 
                                    n=None, 
                                    ntrend=1, nlf=0, nhol=0,
                                    seasPeriods=[], seasHarmComponents = [])
        for j in range(len(disfact_trend)):
            for l in range(len(disfact_covar)):
                try:
                    mod, samples = analysis(Y, X, family="bernoulli",
                            forecast_start=forecast_start,      # First time step to forecast on
                            forecast_end=forecast_end,          # Final time step to forecast on
                            k = k,                                # Forecast horizon.        
                            rho=1,                 # Random effect extension, 
                            deltrend=disfact_trend[j],          
                            delregn=disfact_covar[l],           
                            nsamps = 25, model_prior = model_prior
                            )
                    forecast = np.mean(np.mean(samples, axis = 0).flatten()) 
                    mse_list[count, j, l] = mse(Y[(forecast_start):forecast_end + 1].flatten(), forecast)
                except ValueError:
                    continue
        ## (2) Now perform modeling
        ## Select optimal discount factors
        deltrend = disfact_trend[np.where(mse_list[count,:, :] == np.min(mse_list[count,:, :]))[0][0]] 
        delregn = disfact_covar[np.where(mse_list[count,:, :] == np.min(mse_list[count,:, :]))[1][0]] 

        print("Modeling Now")
        mod, samples = analysis(Y, X, family="bernoulli",
                    forecast_start=forecast_start,      # First time step to forecast on
                    forecast_end=forecast_end,          # Final time step to forecast on
                    k=k,                                # Forecast horizon. 
                    rho=1,                         # Random effect extension, 
                    deltrend = deltrend,                # Discount factor on the trend  (intercept)
                    delregn = delregn,                   # Discount factor on the regression component
                    nsamps = 300, ## more samples, reduce for number saved
                    model_prior = model_prior            
                    )
        ## Save true values with appropriate lag
        samps[count, :, :] = samples[:,:,0] 
        true_list[count, :] = Y[(forecast_start):forecast_end + 1].flatten() ## i+1 step ahead
    
        count += 1

    fname = 'simulated_results/simul/HH-' + str(hhg)+ '-k' + str(k) + \
                                            '-true-pReturn.npy'    
    np.save(fname, true_list)          
    fname = 'simulated_results/simul/HH-' + str(hhg)+ '-k' + str(k) + \
                                            '-samples-pReturn.npy'    
    np.save(fname, samps)

    print('Done')

#############################################################################
## df = input dataframe
## hhg = household group
## k = forecast horizon
## hh_list = which households to model
## method = type of point forecasts in flist, options = ['mean', 'median']
## flist = forecasts from higher-level model to use, same order as hh_list
## flist is hh x time, each row is mean or median forecast
def plog_total(df, hhg, k, hh_list, flist, method):
    ## Discount factors to tune
    disfact_obs = np.array([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999])
    disfact_trend = np.array([0.95, 0.96, 0.97, 0.98])
    disfact_covar = np.array([0.98, 0.99, 0.995, 0.999])
    ###################################################################
    ### P(log total spend | return) for all HH 
    ### Forecasting - Forecast for all data points after prior initialized with first 6
    ## Since each HH has a different number of weeks that they return
    forecast_start = 7                                         
    forecast_end = 109 
    dind = (forecast_end - forecast_start) + k

    true_list = np.zeros((len(hh_list), dind))
    samp_list = np.zeros((len(hh_list), 300, dind))
    mad_list = np.zeros((len(hh_list), len(disfact_obs), len(disfact_covar), len(disfact_trend)))

    count = 0
    for i in hh_list:
        ##(1) Format Dataframe
        df2 = df.loc[df['HH'] == i]
        ### Indices where return - find using forecasts from p(Return) model
        ## other covars treated as known
        ind_ret = np.where(np.round(flist[count, :]) == 1)[0]
         
        prev_spend = np.array(df2['TOTAL_SPEND'])[ind_ret][:-1]
        diff_weeks = np.diff(ind_ret)       
        df2r = df2.iloc[np.where(np.round(flist[count, :]) == 1)[0]].reindex()
        df2r = df2r.iloc[1:]
        
        ## TOTAL_SPEND at last return
        df2r['LAST_SPEND'] = prev_spend
        
        ## (2) Modeling
        ## Smallest amount spend a week is set to $1.01
        if not np.all(df2r['TOTAL_SPEND'].values > 1):
            ## Set to $1.01 if there is a problem
            df2r['TOTAL_SPEND'].values[np.where(df2r['TOTAL_SPEND'].values <= 1)[0]] = 1.01
            df2r['LAST_SPEND'].values[np.where(df2r['LAST_SPEND'].values <= 1)[0]] = 1.01
        
        if not np.all(df2r['LAST_SPEND'].values > 1):
            df2r['LAST_SPEND'].values[np.where(df2r['LAST_SPEND'].values <= 1)[0]] = 1.01
        
        ### (2) Log TOTAL_SPEND Last Return
        Y = np.log(df2r['TOTAL_SPEND'].values)
        X = np.log(df2r['LAST_SPEND'].values).reshape(-1,1)
        Y = np.round(Y, 2)
        X = np.round(X - np.mean(X), 2)
        ### Forecasting has to change each time since not all HH have same # of weeks when return       
        forecast_end = len(Y) - k
        print(Y.shape, X.shape, forecast_end)
        ## Tune discount factor for each HH
        dind = (forecast_end - forecast_start) + k
        if dind > 10:
            # prior a0 = 0, R0 = I
            a0 = np.zeros(X.shape[1] + 1)
            R0 = np.eye(X.shape[1] + 1)    
            model_prior = define_dglm(Y, X, a0 = a0, R0 = R0, family='normal', 
                                    n=None, 
                                    ntrend=1, nlf=0, nhol=0, s0 = 1, n0 = 1,
                                    seasPeriods=[], seasHarmComponents = [])
            for j in range(len(disfact_obs)):
                for l in range(len(disfact_covar)):
                    for q in range(len(disfact_trend)):
                        
                        mod, samples = analysis(Y, X, family="normal",
                                forecast_start=forecast_start,      # First time step to forecast on
                                forecast_end=forecast_end,          # Final time step to forecast on
                                k=k,                                # Forecast horizon.             
                                #rho=0.5,                 # Random effect extension
                                deltrend=disfact_trend[q],          
                                delregn=disfact_covar[l],
                                delVar = disfact_obs[j],
                                nsamps = 25, model_prior = model_prior
                                )
                        forecast = median(samples) 
                        ## MAD on $ scale, not log $
                        mad_list[count, j, l, q] = mad(np.exp(Y[forecast_start:forecast_end + k]), np.exp(forecast))
                        
                        
            ## Select optimal observation discount factors
            try:
                dfinds = np.where(mad_list[count,:,:,:] == np.min(mad_list[count,:,:,:]))
                delVar = disfact_obs[dfinds[0][0]] 
                delregn = disfact_covar[dfinds[1][0]]
                deltrend = disfact_trend[dfinds[2][0]]
            except IndexError:
                delVar = 0.99
                delregn = 0.98
                deltrend = 0.98
                
                
            print("Modeling Now")
            ## Now run with best discount factor
            mod, samples, model_coef = analysis(Y, X, family="normal",
                        forecast_start=forecast_start,      # First time step to forecast on
                        forecast_end=forecast_end,          # Final time step to forecast on
                        k=k,                                # Forecast horizon.                  
                        rho=0.5,                         # Random effect extension, none
                        deltrend = deltrend,                # Discount factor on the trend  (intercept)
                        delregn = delregn,                   # Discount factor on the regression component
                        delVar = delVar,
                        ret = ['model', 'forecast', 'model_coef'],
                        nsamps = 300, model_prior = model_prior
                        )
            dind = (forecast_end - forecast_start) + k
            samp_list[count, :, :dind] = samples[:,:,0]
            true_list[count, :dind] = Y[forecast_start:forecast_end + k].flatten()
        
        count += 1
        

    print('Modeling Done')
    fname = 'simulated_results/simul/HH-' + str(hhg)+ '-k' + str(k) + str(method) +\
                                            '-true-logtotal.npy'    
    np.save(fname, true_list)          
    fname = 'simulated_results/simul/HH-' + str(hhg)+ '-k' + str(k) + \
                                            '-' + str(method) + '-samples-logtotal.npy'    
    np.save(fname, samp_list)
    
##########################################################
## df_cat = input dataframe - information about cat
## hhg = household group
## cat = specific  category to model
## k = forecast horizon
## hh_list = households to model
## flist_return = forecasts for p(return)
## flist_spend = forecasts for log total spend
## method = mean or median point forecasts in flists
def CAT_modeling(df_cat, hhg, cat, k, hh_list, flist_return, flist_spend, method):
    
    ## Select category
    df_cat = df_cat[df_cat['CAT'] == cat]
    
    ### Forecasting - Forecast for all data points after prior initialized with first 6
    ## Since each HH has a different number of weeks that they return
    forecast_start = 7                                         
    forecast_end = 109 
    dind = (forecast_end - forecast_start) + k
    ## initialize with -10 to select when forecasting ends, 0s occur in data
    true_list = -10*np.ones((len(hh_list), dind)) 
    samp_list = np.zeros((len(hh_list), 300, dind))
    
    ## Forecast all time steps
    count = 0
    for i in hh_list:
        ## Select one HH
        df_i2 = df_cat.loc[df_cat['HH'] == i] ### CAT level
        if df_i2.shape[0] > 0:
            ## Select weeks when HH returns at all - forecast from p(Return)
            df = df_i2.iloc[np.where(np.round(flist_return[count, :]) == 1)[0]].reindex()

            ## Log total spend accounting for 0s - cat level
            df['LOG_TOTAL_SPEND'] = np.where(df['TOTAL_SPEND'] ==0, 0, np.log(df['TOTAL_SPEND']))
            ## Log total spend accounting for 0s - global
            df['LOG_TOTAL_SPEND_GLOBAL'] = flist_spend[count, 
                                                       np.where(np.round(flist_return[count, :]) == 1)[0]]



            ### (3) Log TOTAL_SPEND global/multiscale - this week
            Y = df['LOG_TOTAL_SPEND'].values
            X = df['LOG_TOTAL_SPEND_GLOBAL'].values.reshape(-1,1)
            X = np.round(X - np.mean(X), 2)
            Y = np.round(Y, 2)
            ## Tune Discount Factors - Manually
            ### Forecasting has to change each time since not all HH have same 
            forecast_end = len(Y) - k
            rho = 1
            deltrend_bern = 0.95
            delregn_bern = 0.98
            deltrend_dlm = 0.95
            delregn_dlm = 0.98
            delVar_dlm = 0.9

            dind = (forecast_end - forecast_start) + k
            if dind > 10:
                # prior a0 = 0, R0 = I  
                
                print("Modeling Now")
                if not np.all(X == 0): ## Make sure weeks to actually forecast/model
                    mod, samples = analysis_dlmm(Y, X, 
                                            forecast_start=forecast_start,      # First time step to forecast on
                                            forecast_end=forecast_end,          # Final time step to forecast on
                                            k=k,                                # Forecast horizon. 
                                            prior_length=10,
                                            #rho=rho,                         # Random effect extension, none
                                            deltrend_bern = deltrend_bern,                
                                            delregn_bern = delregn_bern, 
                                            deltrend_dlm = deltrend_dlm, 
                                            delregn_dlm = delregn_dlm,               
                                            delVar_dlm = delVar_dlm,
                                            ret = ['model', 'forecast'],
                                            nsamps = 300)

                    dind = (forecast_end - forecast_start) + k
                    samp_list[count, :, :dind] = samples[:,:,0]
                    true_list[count, :dind] = Y[forecast_start:forecast_end+k].flatten()
            
        count += 1
       
    fname = 'simulated_results/simul/HH-'+str(hhg)+'-CAT='+str(cat)+'-k'+ \
                                            str(k) +'-' + str(method) + '-true.npy'    
    np.save(fname, true_list)          

    fname = 'simulated_results/simul/HH-'+str(hhg)+'-CAT='+str(cat)+ \
                        '-k'+str(k) + '-' + str(method) + '-samples.npy'    
    np.save(fname, samp_list)
    print('Done')
    
########################################################################    
## df_sub = input dataframe - information about sub-cat
## df_global = input dataframe - aggregated info across all categories
## hhg = household group
## sub = specific sub-category to model
## k = forecast horizon
## flist_return - point forecasts from p(Return) modeling
## flist_spend - point forecasts from p(log Total Spend | Return) modeling
## flist_cat - point forecasts from category modeling
## method = mean or median, type of point forecasts in flists
def SUB_modeling(df_sub, hhg, sub, k, hh_list, flist_return, flist_spend, flist_cat, method):
    
    ### Forecasting - Forecast for all data points after prior initialized with first 6
    ## Since each HH has a different number of weeks that they return
    forecast_start = 7                                         
    forecast_end = 109 
    dind = (forecast_end - forecast_start) + k
    ## initialize with -10 to select when forecasting ends, 0s can occur in data
    true_list = -10*np.ones((len(hh_list), dind)) 
    samp_list = np.zeros((len(hh_list), 300, dind))
    ## Forecast all time steps
    count = 0
    for i in hh_list:
        ## Select one HH
        df_i1 = df_sub.loc[df_sub['HH'] == i] ### Sub-Cat level
        ## Subset return for category at all
        if df_i1.shape[0] > 0 and len(np.where(flist_cat[count,:] > 0)[0]) > 0:
            ## Select weeks when HH returns at all for Cat
            ## First select where return at all
            df = df_i1.iloc[np.where(np.round(flist_return[count, :]) == 1)[0]].reindex()
            ## Then select where return for Cat
            idx = np.where(~np.isin(flist_cat[count,:], [0,-10]))[0]
            df = df.iloc[idx[idx < df.shape[0]]].reindex()
            
            ## Log total spend accounting for 0s - Sub-Cat level
            df['LOG_TOTAL_SPEND'] = np.where(df['TOTAL_SPEND'] ==0, 0, np.log(df['TOTAL_SPEND']))
            
            ## Log total spend accounting for 0s - Cat level
            df['LOG_TOTAL_SPEND_CAT'] = 0
            df['LOG_TOTAL_SPEND_CAT'].iloc[idx[idx < df.shape[0]]] = flist_cat[count, idx[idx < df.shape[0]]].flatten()
            

            ### (3) Log TOTAL_SPEND global/multiscale CAT - this week
            Y = df['LOG_TOTAL_SPEND'].values
            X = df['LOG_TOTAL_SPEND_CAT'].values.reshape(-1,1)
            Y = np.round(Y, 2)
            X = np.round(X - np.mean(X), 2)
            ## Tune Discount Factors - Manually
            ### Forecasting has to change each time since not all HH have same # of weeks when return       
            forecast_end = len(Y) - k

            rho = 1
            deltrend_bern = 0.95
            delregn_bern = 0.98
            deltrend_dlm = 0.95
            delregn_dlm = 0.98
            delVar_dlm = 0.98

            dind = (forecast_end - forecast_start) + k
            if dind > 10:
                # prior a0 = 0, R0 = I  
                
                try:
                    print("Modeling Now")
                    ## Now run with best discount factor
                    mod, samples = analysis_dlmm(Y, X, 
                                            forecast_start=forecast_start,      
                                            forecast_end=forecast_end,          
                                            k=k,                                # Forecast horizon. 
                                            prior_length=10,                     
                                            #rho=rho,                         # Random effect extension, none
                                            deltrend_bern = deltrend_bern,                
                                            delregn_bern = delregn_bern, 
                                            deltrend_dlm = deltrend_dlm, 
                                            delregn_dlm = delregn_dlm,               
                                            delVar_dlm = delVar_dlm,
                                            ret = ['model', 'forecast'],
                                            nsamps = 300
                                            )
                    dind = (forecast_end - forecast_start) + k
                    samp_list[count, :, :dind] = samples[:,:,0]
                    true_list[count, :dind] = Y[forecast_start:forecast_end + k].flatten()
                    print(true_list[count-1,:])
                except ValueError:
                    continue

        count += 1
               
    print('Modeling Done')
    fname = 'simulated_results/simul/HH-'+str(hhg)+'-SUB='+str(sub)+ \
                                    '-k'+ str(k) +'-' + str(method)+ '-true.npy'    
    np.save(fname, true_list)          
    fname = 'simulated_results/simul/HH-'+str(hhg)+'-SUB='+str(sub)+ \
                                '-k'+ str(k) + '-' + str(method) + '-samples.npy'    
    np.save(fname, samp_list)
   
#######################################################################    
import pickle
## df_item = input dataframe - information about item
## hhg = household group
## item = specific ITEM to model
## k = forecast horizon
## model = which model to consider in terms of covars
## flist_return - point forecasts from p(Return) modeling
## flist_spend - point forecasts from p(log Total Spend | Return) modeling
## flist_cat - point forecasts from category modeling
## flist_sub - point forecasts from sub-category modeling
## method = mean or median, type of point forecasts in flists
def ITEM_modeling(df_item, hhg, item, k, hh_list, flist_return, flist_spend, flist_cat, flist_sub, method):
    
    ### Forecasting - Forecast for all data points after prior initialized with first 6
    ## Since each HH has a different number of weeks that they return
    forecast_start = 7                                         
    forecast_end = 109 
    dind = (forecast_end - forecast_start) + k
    true_list = -10*np.ones((len(hh_list), dind)) 
    samp_list = np.zeros((len(hh_list), 300, dind))
    date_list = list()
    ## Forecast all time steps
    count = 0
    for i in hh_list:
        ## Select one HH
        df_i = df_item.loc[df_item['HH'] == i] ### ITEM level
        ## Select weeks where return for sub-category
        if df_i.shape[0] > 0 and len(np.where(flist_sub[count,:] > 0)[0]) > 0:
            
            idx = np.where(~np.isin(flist_sub[count,:], [0,-10]))[0]
            df = df_i.iloc[idx[idx < df_i.shape[0]]].reindex()
            df['LOG_TOTAL_SPEND_SUB'] = 0
            df['LOG_TOTAL_SPEND_SUB'].iloc[idx[idx < df.shape[0]]] = flist_sub[count, idx[idx < df.shape[0]]].flatten()
            
            ### (3) Log TOTAL_SPEND - SUB
            Y = df['ITEM_QTY'].values
            X = df['LOG_TOTAL_SPEND_SUB'].values.reshape(-1,1)
            X = np.round(X - np.mean(X), 2)        
            ### Forecasting has to change each time since not all HH have same # of weeks when return       
            forecast_end = len(Y) - k

            rho_bern = 0.75
            rho_pois = 0.75
            deltrend_bern = 0.98
            delregn_bern = 0.95
            deltrend_pois = 0.98
            delregn_pois = 0.95

            dind = (forecast_end - forecast_start) + k
            if dind > 10: ## Select HH with more than 10 points to forecast
                
                try:
                    print("Modeling Now")
                    ## Now run with best discount factor
                    mod, samples, model_coef = analysis_dcmm(Y, X, 
                                            forecast_start=forecast_start,      # First time step to forecast on
                                            forecast_end=forecast_end,          # Final time step to forecast on
                                            k=k,                                # Forecast horizon. 
                                            prior_length=10,                     
                                            rho_bern=rho_bern,                  # Random effect extension,
                                            deltrend_bern = deltrend_bern,                
                                            delregn_bern = delregn_bern, 
                                            deltrend_pois = deltrend_pois, 
                                            delregn_pois = delregn_pois,               
                                            rho_pois = rho_pois,
                                            ret = ['model', 'forecast', 'model_coef'],
                                            nsamps = 300
                                            )
                    dind = (forecast_end - forecast_start) + k
                    samp_list[count, :, :dind] = samples[:,:,0]
                    true_list[count, :dind] = Y[forecast_start:forecast_end + k].flatten()
                    print(true_list[count-1,:])
                    date_list.append(df['WEEK'])
                except ValueError:
                    continue
        count += 1
            
        
    print('Modeling Done')
    fname = 'simulated_results/simul/HH-'+str(hhg)+ \
                                    '-ITEM='+str(item)+'-k'+ str(k) + \
                                    '-' + str(method) + '-true.npy'    
    np.save(fname, true_list)          
    fname = 'simulated_results/simul/HH-'+str(hhg)+ \
                                str(item)+'-k'+ str(k) + '-' + str(method) + '-samples.npy'    
    np.save(fname, samp_list)
    
    fname = 'simulated_results/simul/HH-'+str(hhg)+ \
                                '-ITEM='+str(item)+'-k'+ str(k) + '-' + \
                                str(method) + '-dates'
    with open(fname, 'wb') as fp:
        pickle.dump(date_list, fp)

    print('Done')  
    
#####################################################    
k = 1
cat = 1
sub = '1A'
item = 'A'

def simul_full(hhg, k, hh_list, dfN, dfH, dfG, dfI):
    
    ## (1) p(Return)
    pReturn(dfN, hhg, k, hh_list)
    ## (2) p(log total spend | Return)
    method = 'mean'
    fname = 'simulated_results/simul/HH-' + str(hhg)+  '-k' + str(k) + \
                                            '-samples-pReturn.npy'   
    samples = np.load(fname)
    flist = np.mean(samples, axis = 1)
    plog_total(dfN, hhg, k, hh_list, flist, method)

    method = 'median'
    fname = 'simulated_results/simul/HH-' + str(hhg)+  '-k' + str(k) + \
                                            '-samples-pReturn.npy'   
    samples = np.load(fname)
    flist = np.median(samples, axis = 1)
    plog_total(dfN, hhg, k, hh_list, flist, method)

    # (3) CAT
    method = 'mean'
    fname = 'simulated_results/simul/HH-' + str(hhg)+  '-k' + str(k) + \
                                            '-samples-pReturn.npy'   
    samples = np.load(fname)
    flist_return = np.mean(samples, axis = 1)
    fname = 'simulated_results/simul/HH-' + str(hhg)+ '-k' + str(k) + \
                                                '-' + str(method) + '-samples-logtotal.npy'    
    samples = np.load(fname)
    flist_spend = np.mean(samples, axis = 1)
    CAT_modeling(dfH, hhg, cat, k, hh_list, flist_return, flist_spend, method)

    method = 'median'
    fname = 'simulated_results/simul/HH-' + str(hhg)+  '-k' + str(k) + \
                                            '-samples-pReturn.npy'   
    samples = np.load(fname)
    flist_return = np.median(samples, axis = 1)
    fname = 'simulated_results/simul/HH-' + str(hhg)+'-k' + str(k) + \
                                                '-' + str(method) + '-samples-logtotal.npy'    
    samples = np.load(fname)
    flist_spend = np.median(samples, axis = 1)
    CAT_modeling(dfH, hhg, cat, k, hh_list, flist_return, flist_spend, method)

    ## (4) SUB
    dfG1 = dfG[dfG['CAT'] == cat]
    df = dfG1[dfG1['SUB_CAT'] == sub]

    method = 'mean'
    fname = 'simulated_results/simul/HH-' + str(hhg)+  '-k' + str(k) + \
                                            '-samples-pReturn.npy'   
    samples = np.load(fname)
    flist_return = np.mean(samples, axis = 1)
    fname = 'simulated_results/simul/HH-' + str(hhg)+ '-k' + str(k) + \
                                                '-' + str(method) + '-samples-logtotal.npy'    
    samples = np.load(fname)
    flist_spend = np.mean(samples, axis = 1)
    fname = 'simulated_results/simul/HH-'+str(hhg)+'-CAT='+str(cat)+'-k'+str(k) + \
                                                '-' + str(method) + '-samples.npy'    
    samples = np.load(fname)
    flist_cat = np.mean(samples, axis = 1)
    SUB_modeling(df, hhg,sub, k, hh_list, flist_return, flist_spend, flist_cat, method)

    method = 'median'
    fname = 'simulated_results/simul/HH-' + str(hhg)+  '-k' + str(k) + \
                                            '-samples-pReturn.npy'   
    samples = np.load(fname)
    flist_return = np.median(samples, axis = 1)
    fname = 'simulated_results/simul/HH-' + str(hhg)+'-k' + str(k) + \
                                                '-' + str(method) + '-samples-logtotal.npy'    
    samples = np.load(fname)
    flist_spend = np.median(samples, axis = 1)
    fname = 'simulated_results/simul/HH-'+str(hhg)+'-CAT='+str(cat)+'-k'+str(k) + \
                                                '-' + str(method) + '-samples.npy'    
    samples = np.load(fname)
    flist_cat = np.median(samples, axis = 1)
    SUB_modeling(df, hhg, sub, k, hh_list, flist_return, flist_spend, flist_cat, method)


    ## (5) ITEM
    dfG1 = dfI[dfI['CAT'] == cat]
    df = dfG1[dfG1['SUB_CAT'] == sub]
    df = df[df['ITEM'] == item]

    method = 'mean'
    fname = 'simulated_results/simul/HH-' + str(hhg)+  '-k' + str(k) + \
                                            '-samples-pReturn.npy'   
    samples = np.load(fname)
    flist_return = np.mean(samples, axis = 1)
    fname = 'simulated_results/simul/HH-' + str(hhg)+ '-k' + str(k) + \
                                                '-' + str(method) + '-samples-logtotal.npy'    
    samples = np.load(fname)
    flist_spend = np.mean(samples, axis = 1)
    fname = 'simulated_results/simul/HH-'+str(hhg)+'-CAT='+str(cat)+'-k'+str(k) + \
                                                '-' + str(method) + '-samples.npy'    
    samples = np.load(fname)
    flist_cat = np.mean(samples, axis = 1)
    fname = 'simulated_results/simul/HH-'+str(hhg)+ \
                                        '-SUB='+str(sub)+'-k'+ str(k) +'-' + \
                                        str(method) + '-samples.npy'  
    samples = np.load(fname)
    flist_sub = np.mean(samples, axis = 1)
    ITEM_modeling(df, hhg, item, k, hh_list, flist_return, flist_spend, flist_cat, 
                                        flist_sub, method)

    method = 'median'
    fname = 'simulated_results/simul/HH-' + str(hhg)+  '-k' + str(k) + \
                                            '-samples-pReturn.npy'   
    samples = np.load(fname)
    flist_return = np.median(samples, axis = 1)
    fname = 'simulated_results/simul/HH-' + str(hhg)+'-k' + str(k) + \
                                                '-' + str(method) + '-samples-logtotal.npy'    
    samples = np.load(fname)
    flist_spend = np.median(samples, axis = 1)
    fname = 'simulated_results/simul/HH-'+str(hhg)+'-CAT='+str(cat)+'-k'+str(k) + \
                                                '-' + str(method) + '-samples.npy'    
    samples = np.load(fname)
    flist_cat = np.median(samples, axis = 1)
    fname = 'simulated_results/simul/HH-'+str(hhg)+ \
                                        '-SUB='+str(sub)+'-k'+ str(k) +'-' + \
                                        str(method) + '-samples.npy'  
    samples = np.load(fname)
    flist_sub = np.median(samples, axis = 1)

    ITEM_modeling(df, hhg, item, k, hh_list, flist_return, flist_spend, 
                    flist_cat, flist_sub, method)

#############################
## Save true conditioned Y ##
#############################
def save_truey(df_item, df_sub, hhg, item, k, hh_list):
    
    ### Forecasting - Forecast for all data points after prior initialized with first 6
    ## Since each HH has a different number of weeks that they return
    forecast_start = 7                                         
    forecast_end = 109 
    dind = (forecast_end - forecast_start) + k
    true_list = -10*np.ones((len(hh_list), dind)) 
    date_list = list()
    inds = np.zeros(len(hh_list))
    ## Forecast all time steps
    count = 0
    for i in hh_list:
        ## Select one HH
        df_i = df_item.loc[df_item['HH'] == i] ### ITEM level
        df_i1 = df_sub.loc[df_sub['HH'] == i] ### SUB level
        ## Select weeks when HH returns at all for SUB
        ## Final dataframe for modeling
        print(i, df_i.shape)
        if df_i.shape[0] > 0:
            df = df_i.iloc[np.where(df_i1['RETURN']==1 )[0]].reindex()

            Y = df['ITEM_QTY'].values
            forecast_end = len(Y) - k
            dind = (forecast_end - forecast_start) + k
            if dind > 10:
                date_list.append(df['WEEK'])
                true_list[count, :dind] = Y[forecast_start:forecast_end + k].flatten()   
                inds[count] = i ## HHs actually modeled
        count += 1
    
    print('Modeling Done')
    fname = 'simulated_results/simul/HH-'+str(hhg)+ \
                                    '-ITEM='+str(item)+'-k'+ str(k) + \
                                    '-true-TRUE-COND.npy'    
    np.save(fname, true_list) 

    fname = 'simulated_results/simul/HH-'+str(hhg)+ \
                                    '-ITEM='+str(item)+'-k'+ str(k) + \
                                    '-inds-TRUE-COND.npy'    
    np.save(fname, inds)          
    
    fname = 'simulated_results/simul/HH-'+str(hhg)+ \
                                '-ITEM='+str(item)+'-k'+ str(k) + '-' + \
                                '-dates-TRUE-COND'
    with open(fname, 'wb') as fp:
        pickle.dump(date_list, fp)

    print('Done')  


################################
hhg = 1  
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

hh_list = np.unique(dfN['HH']) ## all HHs

simul_full(hhg, k, hh_list, dfN, dfH, dfG, dfI)
dfG1 = dfG[dfG['CAT'] == cat]
dfI1 = dfI[dfI['CAT'] == cat]
dfG1 = dfG1[dfG1['SUB_CAT'] == sub]
dfI1 = dfI1[dfI1['SUB_CAT'] == sub]
dfI1 = dfI1[dfI1['ITEM'] == item]
save_truey(dfI1, dfG1, hhg, item, k, hh_list)
#############################
hhg = 2
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

hh_list = np.unique(dfN['HH']) ## all HHs

simul_full(hhg, k, hh_list, dfN, dfH, dfG, dfI)
dfG1 = dfG[dfG['CAT'] == cat]
dfI1 = dfI[dfI['CAT'] == cat]
dfG1 = dfG1[dfG1['SUB_CAT'] == sub]
dfI1 = dfI1[dfI1['SUB_CAT'] == sub]
dfI1 = dfI1[dfI1['ITEM'] == item]
save_truey(dfI1, dfG1, hhg, item, k, hh_list)
#######################################
hhg = 3
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

hh_list = np.unique(dfN['HH']) ## all HHs

simul_full(hhg, k, hh_list, dfN, dfH, dfG, dfI)
dfG1 = dfG[dfG['CAT'] == cat]
dfI1 = dfI[dfI['CAT'] == cat]
dfG1 = dfG1[dfG1['SUB_CAT'] == sub]
dfI1 = dfI1[dfI1['SUB_CAT'] == sub]
dfI1 = dfI1[dfI1['ITEM'] == item]
save_truey(dfI1, dfG1, hhg, item, k, hh_list)

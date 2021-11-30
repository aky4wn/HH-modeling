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


####################
## 1. p(Return) ####
####################

# df = input dataframe
# hhg = household group
# k = forecast horizon
def model_pReturn(df, hhg, k):
    ## Tune discount factors
    disfact_trend = np.array([0.9, 0.93, 0.95, 0.97, 0.99])
    disfact_covar = np.array([0.9, 0.93, 0.95, 0.97, 0.99])
    ###################################################################
    ### P(Return) for HH that do not return every week
    # Forecast every time point 


    ## Find which HH do not return every week
    df_r = df.groupby(['HH'])['RETURN'].sum()
    ind = df_r.loc[df_r < 110].index.values
    
    ### Forecasting
    forecast_start = 7                                         
    forecast_end = 109 - k
    dind = (forecast_end - forecast_start) + 1

    true_list = np.zeros((len(ind), dind, k))
    samps = np.zeros((len(ind), 300, dind, k))
    mse_list = np.zeros((len(ind), len(disfact_trend), len(disfact_covar)))
    
    count = 0
    for i in ind:
        ## Select model
        df1 = df.loc[df['HH'] == i]
         ### X =  Log TOTAL_SPEND_t-1
        Y = df1['RETURN'].values[1:]
        X = np.where(df1['TOTAL_SPEND'].values[:-1] > 0, 
                        np.log(df1['TOTAL_SPEND'].values[:-1]), np.log(0.01)).reshape(-1,1)

        ## Tune discount factor for each HH - Tune based on 4 step ahead forecasts (1 month)
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
                            k = 4,                                # Forecast horizon.                     
                            rho= 1,                          # Random effect extension, 
                            deltrend=disfact_trend[j],          
                            delregn=disfact_covar[l],           
                            nsamps = 100,
                            model_prior = model_prior
                            )
                    forecast = np.mean(samples, axis = 0)[:, -1]
                    mse_list[count, j, l] = mse(Y[(forecast_start + 3):forecast_end + 4].flatten(), 
                                    forecast)
                except ValueError:
                    continue

        ## Select optimal discount factors
        deltrend = disfact_trend[np.where(mse_list[count,:, :] == np.min(mse_list[count,:, :]))[0][0]] 
        delregn = disfact_covar[np.where(mse_list[count,:, :] == np.min(mse_list[count,:, :]))[1][0]] 

        ## Now model with best discount factor
        mod, samples = analysis(Y, X, family="bernoulli",
                    forecast_start=forecast_start,      # First time step to forecast on
                    forecast_end=forecast_end,          # Final time step to forecast on
                    k=k,                                # Forecast horizon. 
                    rho=1,                         # Random effect extension, 
                    deltrend = deltrend,                # Discount factor on the trend  (intercept)
                    delregn = delregn,                   # Discount factor on the regression component
                    nsamps = 300,
                    model_prior = model_prior            
                    )
        ## Save true values with appropriate lag
        samps[count, :, :, :] = samples
        for i in range(k):
            true_list[count, :, i] = Y[(forecast_start + i):forecast_end + i+1].flatten() ## i+1 step ahead
    

        count += 1

    fname = 'simulated_results/pReturn/HH-' + str(hhg)+ '-k' + str(k) + \
                                              '-true.npy'    
    np.save(fname, true_list)          
    fname = 'simulated_results/pReturn/HH-' + str(hhg)+ '-k' + str(k) + \
                                            '-samples.npy'    
    np.save(fname, samps)
##########################################
## 2. Log Total Spend | Return - Global ##
##########################################
## df = input dataframe
## hhg = household group
## k = forecast horizon
def model_logTotal(df, hhg, k):
    disfact_obs = np.array([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999])
    disfact_trend = np.array([0.95, 0.96, 0.97, 0.98])
    disfact_covar = np.array([0.98, 0.99, 0.995, 0.999])
    ###################################################################
    ### P(log total spend | return) for all HH 
    
    ind = np.unique(df['HH'])
    ### Forecasting - Forecast for all data points after prior initialized with first 6
    ## Siince each HH has a different number of weeks that they return
    forecast_start = 7                                         
    forecast_end = 109 
    dind = (forecast_end - forecast_start) + k
    
    true_list = -10*np.ones((len(ind), dind))
    samp_list = np.zeros((len(ind), 300, dind))
    mad_list = np.zeros((len(ind), len(disfact_obs), len(disfact_covar), len(disfact_trend)))
    tdist_list = np.zeros((len(ind), 2, dind))
    df_selected = np.zeros((len(ind), 3))
    
    count = 0
    for i in ind:
        ##(1) Format Dataframe
        df2 = df.loc[df['HH'] == i]
        ### Indices where return
        ind_ret = np.where(df2['RETURN']==1 )[0]
        prev_spend = np.array(df2['TOTAL_SPEND'])[ind_ret][:-1]
        
        df2r = df2.loc[df2['RETURN']==1]
        df2r = df2r.iloc[1:]
        ## TOTAL_SPEND at last return
        df2r['LAST_SPEND'] = prev_spend
        
        ## (2) Modeling
        ## Smallest amount spend a week is set to $1.01
        if not np.all(df2r['LAST_SPEND'].values > 1):
            df2r['LAST_SPEND'].values[np.where(df2r['LAST_SPEND'].values <= 1)[0]] = 1.01
        ### X =  Log TOTAL_SPEND Last Return - round 2 decimals, center covars
        Y = np.round(np.log(df2r['TOTAL_SPEND'].values), 2)
        X = np.round(np.log(df2r['LAST_SPEND'].values), 2).reshape(-1,1)
        X = X - np.mean(X)
        
        ### Forecasting has to change each time since not all HH have same # of weeks when return       
        forecast_end = len(Y) - k
        
        ## Tune discount factor for each HH
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
                            # rho=0.5,                 # Random effect extension
                            deltrend=disfact_trend[q],          
                            delregn=disfact_covar[l],
                            delVar = disfact_obs[j],
                            nsamps = 25,
                            s0 = 1, n0 = 1,
                            model_prior = model_prior
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
        print(delVar, delregn, deltrend)
            
        df_selected[count, 0] = delVar ## Selected observation volatility
        df_selected[count, 1] = delregn ## Selected regression df
        df_selected[count, 2] = deltrend ## Selected trend df
        
        print("Modeling Now")
        ## Now run with best discount factor
        mod, samples, model_coef = analysis(Y, X, family="normal",
                    forecast_start=forecast_start,      # First time step to forecast on
                    forecast_end=forecast_end,          # Final time step to forecast on
                    k=k,                                # Forecast horizon. 
#                     prior_length=6,                     # How many data point to use in defining prior
                    # rho=0.5,                         # Random effect extension, none
                    deltrend = deltrend,                # Discount factor on the trend  (intercept)
                    delregn = delregn,                   # Discount factor on the regression component
                    delVar = delVar,
                    ret = ['model', 'forecast', 'model_coef'],
                    nsamps = 300, model_prior = model_prior
                    )
        dind = (forecast_end - forecast_start) + k
        samp_list[count, :, :dind] = samples[:,:,0]
        true_list[count, :dind] = Y[forecast_start:forecast_end + k].flatten()
        ## get m and v for the T distribution
        F = np.hstack((np.ones((X.shape[0], 1)), X))
        ft, qt = map(list,zip(*[mod.get_mean_and_var(F[i,:], model_coef['m'][i,:], 
                                                        model_coef['C'][i,:, :]) for i in range(1,F.shape[0])]))

        tdist_list[count, 0, 1:dind] = np.array(ft)[forecast_start:forecast_end + k]
        if len(np.array(qt).shape) > 1:
            tdist_list[count, 1, 1:dind] = np.array(qt)[forecast_start:forecast_end + k, 0, 0]
        else:
            tdist_list[count, 1, 1:dind] = np.array(qt)[forecast_start:forecast_end + k]
        count += 1
        

    print('Modeling Done')
    fname = 'simulated_results/total_spend/HH-' + str(hhg)+ '-k' + str(k) + '-true.npy'    
    np.save(fname, true_list)          
    fname = 'simulated_results/total_spend/HH-' + str(hhg)+ '-k' + str(k) + \
                                            '-samples.npy'    
    np.save(fname, samp_list)
    fname = 'simulated_results/total_spend/HH-' + str(hhg)+ '-k' + str(k) + \
                                            '-Tparams.npy'    
    np.save(fname, tdist_list)
    fname = 'simulated_results/total_spend/HH-' + str(hhg)+ '-k' + str(k) + \
                                            '-obsvol.npy'    
    np.save(fname, df_selected)



#########################
for hhg in [1,2,3]:
    df_hh1 = pd.read_csv('Data/HH'+str(hhg)+'-DATA.csv')
    ## Group over all items for global summaries
    df1 = df_hh1.groupby(['HH','WEEK']).sum()[['ITEM_QTY', 'TOTAL_SPEND']].reset_index()
    df1['RETURN'] = np.where(df1['ITEM_QTY'] > 0, 1, 0)
    model_pReturn(df1, hhg, 8)
    model_logTotal(df1, hhg, 1)

###################################
## 3. Category Modeling ###########
###################################
## df_cat = input dataframe - information about category
## df_global = input dataframe - aggregated info across all categories
## hhg = household group
## cat = specific category to model
## k = forecast horizon
## model = which model to consider in terms of covars
def model_Cat(df_cat, df_global, hhg, cat, k, model):
    
    ## Select category
    df_cat = df_cat[df_cat['CAT'] == cat]
    ind = np.unique(df_cat['HH']) #All HHs in group
    ### Forecasting - Forecast for all data points after prior initialized with first 6
    ## Since each HH has a different number of weeks that they return
    forecast_start = 7                                         
    forecast_end = 109 
    dind = (forecast_end - forecast_start) + k
    true_list = -10*np.ones((len(ind), dind)) ## initialize with -10 to select when forecasting ends
    samp_list = np.zeros((len(ind), 300, dind))
    tdist_list = np.zeros((len(ind), 2, dind))
    ## Forecast all time steps
    count = 0
    for i in ind:
        ## Select one HH
        df_i2 = df_cat.loc[df_cat['HH'] == i] ### Category level
        df_ig = df_global.loc[df_global['HH'] == i] ## Global level

        ## Select weeks when HH returns at all
        ## Final dataframe for modeling
        df = df_i2.iloc[np.where(df_ig['RETURN']==1 )[0]].reindex()
        dg = df_ig.iloc[np.where(df_ig['RETURN']==1 )[0]].reindex() ## global information
        

        ## Log total spend accounting for 0s - cat level - round 2 decimals
        df['LOG_TOTAL_SPEND'] = np.round(np.where(df['TOTAL_SPEND'] <=0, 0, np.log(df['TOTAL_SPEND'])), 2)
        ## Log total spend accounting for 0s - global
        dg['LOG_TOTAL_SPEND'] = np.round(np.where(dg['TOTAL_SPEND'] <=0, 0, np.log(dg['TOTAL_SPEND'])), 2)
        
        ## Models to select from
        if model == 1: ### (1) Log TOTAL_SPEND Last Return - cat
            Y = df['LOG_TOTAL_SPEND'].values[1:]
            X = df['LOG_TOTAL_SPEND'].values[:-1].reshape(-1,1)
            X = X - np.mean(X) # center covars
            
        if model == 2: ### (2) Log TOTAL_SPEND Last Return - global
            Y = df['LOG_TOTAL_SPEND'].values[1:]
            X = dg['LOG_TOTAL_SPEND'].values[:-1].reshape(-1,1)
            X = X - np.mean(X) # center covars
            
        if model == 3: ### (3) Log TOTAL_SPEND global - this week
            Y = df['LOG_TOTAL_SPEND'].values
            X = dg['LOG_TOTAL_SPEND'].values.reshape(-1,1)
            X = X - np.mean(X) # center covars
            
        
        ## Tune Discount Factors - Manually
        ### Forecasting has to change each time since not all HH have same # of weeks when return       
        forecast_end = len(Y) - k
        
        rho = 1
        deltrend_bern = 0.95
        delregn_bern = 0.98
        deltrend_dlm = 0.95
        delregn_dlm = 0.98
        delVar_dlm = 0.9
            
        if not np.all(X == 0):
            try:
                print("Modeling Now", X.shape)
                
                mod, samples = analysis_dlmm(Y, X, 
                                        forecast_start=forecast_start,      # First time step to forecast on
                                        forecast_end=forecast_end,          # Final time step to forecast on
                                        k=k,                                # Forecast horizon. 
                                        prior_length=10,
                                        # rho=rho,                         # Random effect extension
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
                print(true_list[count, :dind])
                    
         
            except ValueError:
                continue
            count += 1
       
    print('Modeling Done')
    fname = 'simulated_results/Cat/HH-'+str(hhg)+'-CAT='+str(cat)+'-k'+ \
                                            str(k) + '-true.npy'    
    np.save(fname, true_list)          
    fname = 'simulated_results/Cat/HH-'+str(hhg)+'-CAT='+str(cat)+ \
                        '-k'+str(k) + '-m' + str(model) + '-samples.npy'    
    np.save(fname, samp_list)
   
    print('Done')
#########################
for hhg in [1,2,3]:
    for cat in [1,2]:
        df_hh1 = pd.read_csv('Data/HH'+str(hhg)+'-DATA.csv')
        ## Group over all items for global summaries
        df1 = df_hh1.groupby(['HH','WEEK']).sum()[['ITEM_QTY', 'TOTAL_SPEND']].reset_index()
        df1['RETURN'] = np.where(df1['ITEM_QTY'] > 0, 1, 0)
        dfc = df_hh1.groupby(['HH','WEEK', 'CAT']).sum()[['ITEM_QTY', 'TOTAL_SPEND']].reset_index()
        dfc['RETURN'] = np.where(dfc['ITEM_QTY'] > 0, 1, 0)
        model_Cat(dfc, df1, hhg, cat, 1, 1)
        model_Cat(dfc, df1, hhg, cat, 1, 2)
        model_Cat(dfc, df1, hhg, cat, 1, 3)
############################################
### 4. Sub-Category ########################
############################################
## df_sub = input dataframe - information about sub-category
## df_cat = input dataframe - information about category
## hhg = household group
## sub = specific sub-category to model
## k = forecast horizon
## model = which model to consider in terms of covars
def model_SubCat(df_sub, df_cat, hhg, sub, k, model):
    
    ind = np.unique(df_sub['HH'])#All HHs in group
    ### Forecasting 
    ## Since each HH has a different number of weeks that they return
    forecast_start = 7                                         
    forecast_end = 109 
    dind = (forecast_end - forecast_start) + k
    true_list = -10*np.ones((len(ind), dind)) ## initialize with -10 to select when forecasting ends
    samp_list = np.zeros((len(ind), 300, dind))
    tdist_list = np.zeros((len(ind), 2, dind))
    ## Forecast all time steps
    count = 0
    for i in ind:
        ## Select one HH
        df_i1 = df_sub.loc[df_sub['HH'] == i] ### Sub-Cat level
        df_i2 = df_cat.loc[df_cat['HH'] == i] ### Cat level


        ## Select weeks when HH returns at all for category
        ## Final dataframe for modeling
        df = df_i1.iloc[np.where(df_i2['RETURN']==1 )[0]].reindex()
        dh = df_i2.iloc[np.where(df_i2['RETURN']==1 )[0]].reindex() ## Cat info
        print(df.isnull().any().any(), dh.isnull().any().any())
        ## Log total spend accounting for 0s - sub-cat level
        df['LOG_TOTAL_SPEND'] = np.where(df['TOTAL_SPEND'] <=0, 0, np.log(df['TOTAL_SPEND']))
        ## Log total spend accounting for 0s - cat level
        dh['LOG_TOTAL_SPEND'] = np.where(dh['TOTAL_SPEND'] <=0, 0, np.log(dh['TOTAL_SPEND']))
        print(df.isnull().any().any(), dh.isnull().any().any())
        ## Models to select from - covars at CAT level
        if model == 1: ### (1) Log TOTAL_SPEND Last Return - sub-cat
            Y = df['LOG_TOTAL_SPEND'].values[1:]
            X = df['LOG_TOTAL_SPEND'].values[:-1].reshape(-1,1)
            X = np.round(X - np.mean(X), 2)
            
        if model == 2: ### (2) Log TOTAL_SPEND Last Return -  cat
            Y = df['LOG_TOTAL_SPEND'].values[1:]
            X = dh['LOG_TOTAL_SPEND'].values[:-1].reshape(-1,1)
            X = np.round(X - np.mean(X), 2)
            
        if model == 3: ### (3) Log TOTAL_SPEND  cat - this week
            Y = df['LOG_TOTAL_SPEND'].values
            X = dh['LOG_TOTAL_SPEND'].values.reshape(-1,1)
            X = np.round(X - np.mean(X), 2)
               
        ## Tune Discount Factors - Manually
        ### Forecasting has to change each time since not all HH have same # of weeks when return       
        forecast_end = len(Y) - k
        
        ## Option 1:
        rho = 1
        deltrend_bern = 0.95
        delregn_bern = 0.98
        deltrend_dlm = 0.95
        delregn_dlm = 0.98
        delVar_dlm = 0.98
    
        dind = (forecast_end - forecast_start) + k
        if dind > 10:
            print("Modeling Now")
            ## Now run with best discount factor
            try:
                mod, samples = analysis_dlmm(Y, X, 
                                        forecast_start=forecast_start,      # First time step to forecast on
                                        forecast_end=forecast_end,          # Final time step to forecast on
                                        k=k,                                # Forecast horizon. 
                                        prior_length=10,                     
                                        # rho=rho,                         # Random effect extension, none
                                        deltrend_bern = deltrend_bern,                
                                        delregn_bern = delregn_bern, 
                                        deltrend_dlm = deltrend_dlm, 
                                        delregn_dlm = delregn_dlm,               
                                        delVar_dlm = delVar_dlm,
                                        n0_dlm = 1,
                                        s0_dlm = 1,
                                        ret = ['model', 'forecast'],
                                        nsamps = 300)
                dind = (forecast_end - forecast_start) + k
                samp_list[count, :, :dind] = samples[:,:,0]
                true_list[count, :dind] = Y[forecast_start:forecast_end + k].flatten()
                
            except ValueError:
                continue
            
            count += 1
            
        
    print('Modeling Done')
    fname = 'simulated_results/Sub_Cat/HH-'+str(hhg)+ \
                                    '-SUB_CAT='+str(sub)+'-k'+ str(k) + '-true.npy'    
    np.save(fname, true_list)          
    
    fname = 'simulated_results/Sub_Cat/HH-'+str(hhg)+ \
                                '-SUB_CAT='+str(sub)+'-k'+ str(k) + '-m' + str(model) + '-samples.npy'    
    np.save(fname, samp_list)
    
    
    print('Done')
#########################################################
k = 1
sub_list = ['1A', '1B', '2A', '2B']
cat_list = [1, 1, 2, 2]
for hhg in [1,2,3]:
    for i in range(len(sub_list)):
        cat = cat_list[i]
        ug = sub_list[i]
        df_hh1 = pd.read_csv('Data/HH'+str(hhg)+'-DATA.csv')
        ## Group over all items for cat summaries
        dfc = df_hh1.groupby(['HH','WEEK', 'CAT']).sum()[['ITEM_QTY', 'TOTAL_SPEND']].reset_index()
        dfc['RETURN'] = np.where(dfc['ITEM_QTY'] > 0, 1, 0)
        dfc = dfc[dfc['CAT'] == cat]
        ## Sub-cat
        dfs = df_hh1.groupby(['HH','WEEK', 'CAT', 'SUB_CAT']).sum()[['ITEM_QTY', 'TOTAL_SPEND']].reset_index()
        dfs['RETURN'] = np.where(dfs['ITEM_QTY'] > 0, 1, 0)
        dfs = dfs[dfs['SUB_CAT'] == ug]

        model_SubCat(dfs, dfc, hhg, ug, k, 1)
        model_SubCat(dfs, dfc, hhg, ug, k, 2)
        model_SubCat(dfs, dfc, hhg, ug, k, 3)
#################################################
### 5. Item Level ###############################
#################################################
## df_item = input dataframe - information about item
## df_sub = input dataframe - information about sub-category
## hhg = household group
## item = specific ITEM category to model
## k = forecast horizon
## model = which model to consider in terms of covars
## df_discount = aggregate discount information
def model_ITEM(df_item, df_sub, hhg, item, k, model, df_discount = None):
    
    ind = np.unique(df_item['HH'])# All HHs in group
    ### Forecasting - Forecast for all data points after prior initialized with first 6
    ## Since each HH has a different number of weeks that they return
    forecast_start = 7                                         
    forecast_end = 109 
    dind = (forecast_end - forecast_start) + k
    true_list = -10*np.ones((len(ind), dind)) ## initialize with -10 to select when forecasting ends
    samp_list = np.zeros((len(ind), 300, dind))
    m_list = np.zeros((len(ind), 112, 3))
    C_list = np.zeros((len(ind), 112, 3, 3))
    ## Forecast all time steps
    count = 0
    for i in ind:
        ## Select one HH
        df_i = df_item.loc[df_item['HH'] == i] ### ITEM level
        df_i1 = df_sub.loc[df_sub['HH'] == i] ### Sub-Cat level

        ## Select weeks when HH returns at all for Sub-Cat
        ## Final dataframe for modeling
        df = df_i.iloc[np.where(df_i1['RETURN']==1 )[0]].reindex()
        dh1 = df_i1.iloc[np.where(df_i1['RETURN']==1 )[0]].reindex() ## Sub-Cat info
        
        ## Log total spend accounting for 0s - sub-cat level
        dh1['LOG_TOTAL_SPEND'] = np.where(dh1['TOTAL_SPEND'] <=0, 0, np.log(dh1['TOTAL_SPEND']))
    
        ###################################    
        if model == 1: ### (15) Log TOTAL_SPEND SUB_CAT and discount percent reg - ITEM
            Y = df['ITEM_QTY'].values
            X1 = dh1['LOG_TOTAL_SPEND']
            X2 = df['DISCOUNT_PERC']
            # X = np.vstack((dh1['LOG_TOTAL_SPEND'], dh1['DISCOUNT_AMOUNT'])).T
            X1 = np.round(X1 - np.mean(X1), 2)
            X2 = np.round(X1 - np.mean(X2), 2)
            X = np.vstack((X1, X2)).T 

        ##############################################
        
        if model == 2: ### (16) Log TOTAL_SPEND - SUB_CAT
            Y = df['ITEM_QTY'].values
            X = dh1['LOG_TOTAL_SPEND'].values.reshape(-1,1)
            X = np.round(X - np.mean(X), 2)
            
        if model == 3: ## (29) Aggregate discount perc reg information (at item level) like Daniel's
            df_dd = df_discount.iloc[np.where(df_i1['RETURN']==1 )[0]].reindex() ## agg discount info
            Y = df['ITEM_QTY'].values
            ## Log TOTAL_SPEND - SUB_CAT
            X1 = dh1['LOG_TOTAL_SPEND'].values
            X1 = np.round((X1 - np.mean(X1))/np.std(X1), 2)
            ## Aggregate discount information
            X2 = df_dd['DISCOUNT_PERC']
            X2 = np.round((X2 - np.mean(X2))/np.std(X2), 2)
            X = np.vstack((X1, X2)).T

     
        ## Tune Discount Factors 
        ### Forecasting has to change each time since not all HH have same # of weeks when return       
        forecast_end = len(Y) - k

        deltrend_bern = 0.98
        delregn_bern = 0.95
        deltrend_pois = 0.98
        delregn_pois = 0.95
        rho_bern = 0.75
        rho_pois = 0.75

         
        dind = (forecast_end - forecast_start) + k
        if dind > 10: ## Select HH with more than 10 points to forecast
            try:
                # prior a0 = 0, R0 = I  
                a0_pois = np.zeros(X.shape[1] + 1)
                R0_pois = np.eye(X.shape[1] + 1)
                a0_bern = np.zeros(X.shape[1] + 1)
                R0_bern = np.eye(X.shape[1] + 1)
                model_prior = define_dcmm(Y, X, a0_pois = a0_pois, R0_pois = R0_pois,
                                            a0_bern = a0_bern, R0_bern = R0_bern,  
                                            n=None, 
                                            ntrend=1, nlf=0, nhol=0,
                                            seasPeriods=[], seasHarmComponents = [])     

                print("Modeling Now")
                ## Now run with best discount factor
                mod, samples, model_coef = analysis_dcmm(Y, X, 
                                        forecast_start=forecast_start,      # First time step to forecast on
                                        forecast_end=forecast_end,          # Final time step to forecast on
                                        k=k,  # Forecast horizon. 
                                        prior_length=0,                     
                                        rho_bern=rho_bern,                  # Random effect extension,
                                        deltrend_bern = deltrend_bern,                
                                        delregn_bern = delregn_bern, 
                                        deltrend_pois = deltrend_pois, 
                                        delregn_pois = delregn_pois,               
                                        rho_pois = rho_pois,
                                        ret = ['model', 'forecast', 'model_coef'],
                                        nsamps = 300, model_prior = model_prior
                                        )
                dind = (forecast_end - forecast_start) + k
                samp_list[count, :, :dind] = samples[:,:,0]
                true_list[count, :dind] = Y[forecast_start:forecast_end + k].flatten()

                ## get m and v for the T distribution
                F = np.hstack((np.ones((X.shape[0], 1)), X))
                m_list[count, :F.shape[0], :F.shape[1]] = model_coef['m']
                C_list[count, :F.shape[0], :F.shape[1], :F.shape[1]] = model_coef['C'] 
                  
            except ValueError:
                continue
            count += 1
            
        
    print('Modeling Done')
    fname = 'simulated_results/ITEM/HH-'+str(hhg)+ \
                                    '-ITEM='+str(item)+'-k'+ str(k) + '-true.npy'    
    np.save(fname, true_list)          
    
    fname = 'simulated_results/ITEM/HH-'+str(hhg)+ \
                            '-ITEM='+str(item)+'-k'+ str(k) + '-m' + \
                            str(model) + '-samples.npy'    
    np.save(fname, samp_list)
    fname = 'simulated_results/ITEM/HH-'+str(hhg)+ \
                                    '-ITEM='+str(item)+'-k'+ str(k) +'-m' + \
                                    str(model) + '-m.npy'    
    np.save(fname, m_list)
    fname = 'simulated_results/ITEM/HH-'+str(hhg)+ \
                                    '-ITEM='+str(item)+'-k'+ str(k) +'-m' + \
                                    str(model) + '-C.npy'    
    np.save(fname, C_list)
    
    print('Done')
#########################################################
k = 1
sub_list = ['1A', '1A', '1B', '1B', '2A', '2A', '2B', '2B']
item_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
for hhg in [1,2,3]:
    for i in range(len(sub_list)):
        item = item_list[i]
        sub = sub_list[i]
        df_hh1 = pd.read_csv('Data/HH'+str(hhg)+'-DATA.csv')
        ## Sub-cat
        dfs = df_hh1.groupby(['HH','WEEK', 'CAT', 'SUB_CAT']).sum()[['ITEM_QTY', 'TOTAL_SPEND']].reset_index()
        dfs['RETURN'] = np.where(dfs['ITEM_QTY'] > 0, 1, 0)
        dfs = dfs[dfs['SUB_CAT'] == sub]
        dfi = df_hh1[df_hh1['ITEM'] == item]
        df_discount = dfi.groupby(['WEEK']).mean()
        model_ITEM(dfi, dfs, hhg, item, k, 1, df_discount)
        model_ITEM(dfi, dfs, hhg, item, k, 2, df_discount)
        model_ITEM(dfi, dfs, hhg, item, k, 3, df_discount)


# #####################################################
# DIRECT
# ########################
## df_item = input dataframe - information about item
## df_sub = input dataframe - information about SUB_CAT
## hhg = household group
## item = specific ITEM category to model
## k = forecast horizon
## model = which model to consider in terms of covars
def model_Direct(df_item, df_sub, hhg, item, k, model):
    
    ind = np.unique(df_item['HH'])# All HHs in group
    ### Forecasting - Forecast for all data points after prior initialized with first 6
    ## Since each HH has a different number of weeks that they return
    forecast_start = 7                                         
    forecast_end = 109 
    dind = (forecast_end - forecast_start) + k
    true_list = -10*np.ones((len(ind), dind)) ## initialize with -10 to select when forecasting ends
    samp_list = np.zeros((len(ind), 300, dind))

    ## Forecast all time steps
    count = 0
    for i in ind:
        ## Select one HH
        df = df_item.loc[df_item['HH'] == i] ### ITEM level
        dh1 = df_sub.loc[df_sub['HH'] == i] ### Sub-Cat level

        ## Log total spend accounting for 0s - sub-cat level
        dh1['LOG_TOTAL_SPEND'] = np.where(dh1['TOTAL_SPEND'] <=0, 0, np.log(dh1['TOTAL_SPEND']))
        ## Log total spend accounting for 0s - item level
        df['LOG_TOTAL_SPEND'] = np.where(df['TOTAL_SPEND'] <=0, 0, np.log(df['TOTAL_SPEND']))

        
        if model == 1: ### (3) Log TOTAL_SPEND and Discount Amount - SUB_CAT, lagged
            Y = df['ITEM_QTY'].values[1:]
            X = np.vstack((dh1['LOG_TOTAL_SPEND'].values[:-1], dh1['DISCOUNT_PERC'].values[:-1])).T
            X = X - np.mean(X, axis = 0)
            
        if model == 2: ### (4) Log TOTAL_SPEND and Discount Amount - item, lagged
            Y = df['ITEM_QTY'].values[1:]
            X = np.vstack((df['LOG_TOTAL_SPEND'].values[:-1], df['DISCOUNT_PERC'].values[:-1])).T
            X = X - np.mean(X, axis = 0)

        ## Conditioning on information / model decomposition
        if model in [3,4]:
            ## Select weeks when HH returns at all for SUB_CAT
            df = df.iloc[np.where(dh1['RETURN']==1 )[0]].reindex() ## item info
            dh1 = dh1.iloc[np.where(dh1['RETURN']==1 )[0]].reindex() ## sub info
            
            if model == 3: ### (5) Log TOTAL_SPEND - SUB_CAT - FOR COMPARISON LAGGED
                Y = df['ITEM_QTY'].values[1:]
                X = np.vstack((dh1['LOG_TOTAL_SPEND'].values[:-1], 
                            dh1['DISCOUNT_PERC'].values[:-1])).T
                X = X - np.mean(X, axis = 0)

            if model == 4: ### (6) Log TOTAL_SPEND - ITEM - FOR COMPARISON LAGGED
                Y = df['ITEM_QTY'].values[1:]
                X = np.vstack((df['LOG_TOTAL_SPEND'].values[:-1], 
                            df['DISCOUNT_PERC'].values[:-1])).T
                X = X - np.mean(X, axis = 0)
        
        ### Forecasting has to change each time since not all HH have same # of weeks when return       
        forecast_end = len(Y) - k
        
        deltrend_bern = 0.98
        delregn_bern = 0.95
        deltrend_pois = 0.98
        delregn_pois = 0.95
        rho_bern = 0.75
        rho_pois = 0.75
         
        dind = (forecast_end - forecast_start) + k
        if dind > 10: ## Select HH with more than 10 points to forecast
            try:
                # prior a0 = 0, R0 = I  
                a0_pois = np.zeros(X.shape[1] + 1)
                R0_pois = np.eye(X.shape[1] + 1)
                a0_bern = np.zeros(X.shape[1] + 1)
                R0_bern = np.eye(X.shape[1] + 1)
                model_prior = define_dcmm(Y, X, a0_pois = a0_pois, R0_pois = R0_pois,
                                            a0_bern = a0_bern, R0_bern = R0_bern,  
                                            n=None, 
                                            ntrend=1, nlf=0, nhol=0,
                                            seasPeriods=[], seasHarmComponents = []) 
                print("Modeling Now")
                ## Now run with best discount factor
                mod, samples, model_coef = analysis_dcmm(Y, X, 
                                        forecast_start=forecast_start,      # First time step to forecast on
                                        forecast_end=forecast_end,          # Final time step to forecast on
                                        k=k,  # Forecast horizon. 
                                        prior_length=0,                     
                                        rho_bern=rho_bern,                  # Random effect extension,
                                        deltrend_bern = deltrend_bern,                
                                        delregn_bern = delregn_bern, 
                                        deltrend_pois = deltrend_pois, 
                                        delregn_pois = delregn_pois,               
                                        rho_pois = rho_pois,
                                        ret = ['model', 'forecast', 'model_coef'],
                                        nsamps = 300, model_prior = model_prior
                                        )
                dind = (forecast_end - forecast_start) + k
                samp_list[count, :, :dind] = samples[:,:,0]
                true_list[count, :dind] = Y[forecast_start:forecast_end + k].flatten()
                  
            except ValueError:
                continue
            count += 1
            print(true_list[count-1,:])
    print('Modeling Done')
    fname = 'simulated_results/ITEM/HH-'+str(hhg)+ \
                                    '-ITEM='+str(item)+'-k'+ str(k) + '-true-DIRECT.npy'    
    np.save(fname, true_list)          
    
    fname = 'simulated_results/ITEM/HH-'+str(hhg)+ \
                            '-ITEM='+str(item)+'-k'+ str(k) + '-m' + \
                            str(model) + '-samples-DIRECT.npy'    
    np.save(fname, samp_list)
    print('Done')
####################################################
k = 1
sub_list = ['1A', '1A', '1B', '1B', '2A', '2A', '2B', '2B']
item_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
for hhg in [1,2,3]:
    for i in range(len(sub_list)):
        item = item_list[i]
        sub = sub_list[i]
        df_hh1 = pd.read_csv('Data/HH'+str(hhg)+'-DATA.csv')
        ## Sub-cat
        dfs = df_hh1.groupby(['HH','WEEK', 'CAT', 'SUB_CAT']).sum()[['ITEM_QTY', 'TOTAL_SPEND', 'DISCOUNT_PERC']].reset_index()
        dfs['RETURN'] = np.where(dfs['ITEM_QTY'] > 0, 1, 0)
        dfs = dfs[dfs['SUB_CAT'] == sub]
        dfi = df_hh1[df_hh1['ITEM'] == item]
        
        model_Direct(dfi, dfs, hhg, item, k, 1)
        model_Direct(dfi, dfs, hhg, item, k, 2)
        model_Direct(dfi, dfs, hhg, item, k, 3)
        model_Direct(dfi, dfs, hhg, item, k, 4)



'''Reuseable functions for xgboost model selection .

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import xgboost as xgb 
from importlib import reload
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import RandomizedSearchCV

def run_early_stop_rand_hp_opt(param_grid=None, x=None, y=None, x_val=None, 
                               y_val=None, num_trails=1, model_func=None, 
                               es_rounds=5, eval_metric="rmse", 
                               gpu=True, verb=0):
                  
    '''Random search to find optmal hyper parameters
    
    Inputs
    -----
    param_grid: Sklearn parameter grid
    grid of hyper parameters (hps) to search over
    
    x: Array type
    traning input shape = (num_samples, features) 
    
    y: Array type or list
    traning target values shape = num_samples
    
    x_val: Array type
    validaiton input shape = (num_samples, features) 
    
    y_val: Array type or list
    validation target values shape = num_samples
    
    num_trails: Int
    number of parameters to be randomly sampled

    m_func: Function
    that returns xgboost model 
    
    es_rounds: Int
    number of trees created without improvement on validaiton set 
    
    eval_metric: Str
    metric used to evaluate validaiton set
    
    gpu: Bool
    if GPU should be used for traning
    
    verb: Int
    how much detail to give when traning
    
    Returns
    -------
    optmisation results: pd dataframe
    of the search results 
    
    val_losses: list
    validaiton loss of each run
    
    best_model: xgboost model
    model found to have best eval_metric on validaiton set
    
    '''
    
    rng = np.random.default_rng()
    hp_inds = rng.choice(len(param_grid), size=num_trails, replace=False)

    opt_results = {'Smallest val loss': [] , 'Number of trees': [], 'HPs': []}
    losses, val_losses = [], []
    best_loss = None
    for i, ind in enumerate(hp_inds):
        print(f'trail {i+1} of {num_trails}')
        hps = param_grid[ind]
        if gpu:
            hps['tree_method']="gpu_hist"
            
        xgbr = model_func(**hps)
        xgbr.fit(x, y, early_stopping_rounds=es_rounds, 
                 eval_set=[(x_val, y_val)], 
                 eval_metric=eval_metric,
                 verbose=verb)
        
        #store results and model if its better than previos best model
        val_loss = xgbr.evals_result()['validation_0'][eval_metric]
        num_trees = xgbr.best_ntree_limit
        min_loss = min(val_loss)
        opt_results['Smallest val loss'].append(min_loss)
        opt_results['Number of trees'].append(num_trees)
        opt_results['HPs'].append(hps)
        if best_loss == None or min_loss < best_loss:
            best_loss = min_loss
            best_model = xgbr

        #keep if loss plots needed
        val_losses.append(val_loss)
        
    return pd.DataFrame(opt_results), val_losses, best_model 
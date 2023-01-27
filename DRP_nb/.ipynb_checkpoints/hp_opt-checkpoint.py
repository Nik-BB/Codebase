'''Reuseable functions for hyperparameter optimisation

'''
import numpy as np
import pandas as pd
import collections
from DRP_nb import cross_val

def run_random_hp_opt(param_grid, x, y, num_trails, model_func, epochs,
                      k=3, p=1, batch_size=128, cv_type='cblind'):
    '''Random search to find optmal hyper parameters
    
    Inputs
    -----
    param_grid: Sklearn parameter grid
    grid of hyper parameters (hps) to search over
    
    x: List
    where x[0] gives input omics data and x[1] gives input drug data
    drug data array like. omics data pd dataframe with cell lines are 
    indexes and features in cols. 
    
    y: pd series 
    target values
    
    num_trails: Int
    number of parameters to be randomly sampled

    m_func: Function
    that returns keras model
    
    
    epochs: int
    number of epochs to train the model for
    
    k: int
    number of folds to run the cross valdation for
    
    p: int
    number of times to repeats the k fold cross validation 
    
    batch_size: int
    batch size of keras model
    
    Returns
    -------
    optmisation results: pd dataframe
    of the search results 
    '''
    #set the cv implementation 
    if cv_type == 'cblind':
        cv_imp = cross_val.run_cv_cblind
    if cv_type == 'dblind':
        cv_imp = cross_val.run_cv_dblind
 
    rng = np.random.default_rng()
    hp_inds = rng.choice(len(param_grid), size=num_trails, replace=False)

    opt_results = {'Smallest val loss': [], 'SD': [], 'Epoch': [], 'HPs': []}
    losses, val_losses = [], []
    
    for i, ind in enumerate(hp_inds):
        print(i/num_trails)
        hps = param_grid[ind]
        loss, val_loss, *_ = cv_imp(model_func,
                                    x,
                                    y,
                                    hps,
                                    k=k,
                                    p=p,
                                    verbos=0,
                                    epochs=epochs,
                                    batch_size=batch_size)   

        min_loss, sd, epoch = cross_val.best_metric(val_loss)
        opt_results['Smallest val loss'].append(min_loss)
        opt_results['SD'].append(sd)
        opt_results['Epoch'].append(epoch)
        opt_results['HPs'].append(hps)

        #keep if loss plots needed
        losses.append(loss)
        val_losses.append(val_loss)
        
    return pd.DataFrame(opt_results), losses, val_losses 


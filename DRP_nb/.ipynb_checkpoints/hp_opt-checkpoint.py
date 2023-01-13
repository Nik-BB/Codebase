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
    #choose implementation of cv 
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

        min_loss, sd, epoch = best_metric(val_loss)
        opt_results['Smallest val loss'].append(min_loss)
        opt_results['SD'].append(sd)
        opt_results['Epoch'].append(epoch)
        opt_results['HPs'].append(hps)

        #keep if loss plots needed
        losses.append(loss)
        val_losses.append(val_loss)
        
    return pd.DataFrame(opt_results), losses, val_losses 

def cv_metric(metric, func):
    """Finds func of cv across a number of epochs as outputted from run_cv
    
    e.g. for func = np.mean gives mean at each epoch
    """
    metricT = np.array(metric).T
    result = []
    for i in range(len(metricT)):
        result.append(func(metricT[i]))
    return np.array(result)



def best_metric(metric, rounded=True):
    '''Finds the minimum of a metric in the form outputted form run_cv
    '''
    m = cv_metric(metric, np.mean)
    sd = cv_metric(metric, np.std)
    argmin = np.argmin(m)
    #make formatting more readable
    Min_metric = collections.namedtuple('MinMetric', ['mean', 'sd', 'index'])
    if rounded:
        return Min_metric(m[argmin], np.round(sd[argmin], 3), argmin)
    else:
        return Min_metric(m[argmin], sd[argmin], argmin)
        
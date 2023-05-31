'''Reuseable functions for torch cross validation in drug response prediction

CV can be implemented in 3 different ways

Mixed: standard CV there can be overlap between cells or drugs in val and
partial train set
Cancer blind: cell lines in the partial training set can not be in the
validation set
Drug blind: drugs in the partial training set can not be in the
validation
'''
import torch 
import sklearn
import collections
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu" 
device

def run_cv_cblind_torch(m_func, train_data, hp, drug_cl_index, train_loop, 
                        optimiser_fun, epochs=10, k=3, p=1, verbos=1, 
                        random_seed=None, loss_fn=nn.MSELoss(),
                        batch_size=512):
    '''runs cancer blind k fold cv p times for pytorch model
    
    cancer blind means cell lines in the train set are not in the val set
    
    Inputs
    -----
    m_func: Function
    that returns pytorch model.
    
    train_data: Dataset instance
    where __getitem__ returns x_omic x_drug, y the omic drug and target data
    respectively
    
    hp: dict
    hyper parmaters that are to be inputed to m_func.
    
    epochs: int
    number of epochs to train the model for.
    
    k: int
    number of folds to run the cross valdation for.
    
    p: int
    number of times to repeats the k fold cross validation.
    
    verbos: int
    if > 1 print out cv status.
    
    batch_size: int
    batch size of data for traning and validaiton
    
    random_seed: int, or None, deafault=None
    set random seed the same int to get identical splits. 
    
    drug_cl_index: list like
    gives names of drugs and cell lines for each sample in corredct order.
    In format drug_name::cell_line_name
    
    train_loop: function
    pytorch traning loop to train the model
    
    loss_fun: pytorch loss function
    
    optimiser: pytorch optimiser
    
    Returns
    -------
    hist: dic that gives history of model traning
    train_val_cls: cls used for validation
    '''
    if verbos > 0:
        print(f'Traning on {device}')
    
    cv = sklearn.model_selection.RepeatedKFold(
        n_splits=k,
        n_repeats=p,
        random_state=random_seed) 
    #loss = []
    #val_loss = []
    #val_mae = []
    #val_loss_mm = []
    train_val_cls = ([], [])
    hists = []
    
    
    #find a list of jsut cell lines and unique cell lines
    cells = np.array([cdp.split('::')[0] for cdp in drug_cl_index])
    u_cells = np.unique(cells)
    i = 0
    #splits by cell lines then finds all cell line drug pairs assicted with
    #these cell lines
    for train_c_i, val_c_i in cv.split(u_cells):
        if verbos > 0:
            print(i / (k * p))
        p_train_cells, val_cells = u_cells[train_c_i], u_cells[val_c_i]
        train_val_cls[0].append(p_train_cells)
        train_val_cls[1].append(val_cells)
        train_i, val_i = [], []
        #find index of cell lines in val and partial train set
        for ptc, vc in zip(p_train_cells, val_cells):
            train_i.extend(np.where(cells == ptc)[0])
            val_i.extend(np.where(cells == vc)[0])
        #reshuffles inds so in orgian order and not ordered by cl    
        train_i.sort(), val_i.sort()
        assert set(train_i).intersection(val_i) == set()
        i += 1
        
        par_train_dl = Subset(train_data, train_i)
        val_dl = Subset(train_data, val_i)
        par_train_dl = DataLoader(par_train_dl, batch_size=batch_size, 
                                  shuffle=False)
        val_dl = DataLoader(val_dl, batch_size=batch_size, shuffle=False)
        
        model = m_func(hp).to(device)
        optimiser = optimiser_fun(model.parameters(), lr=hp['lr'])
        
        hist = train_loop(
            train_dl=par_train_dl,
            val_dl=val_dl,
            model=model,
            loss_fn=loss_fn,
            optimiser=optimiser,
            epochs=epochs, 
            verb=verbos - 1)
        
        hists.append(hist)
        #loss.append(hist.history['loss'])
        #val_loss.append(hist.history['val_loss'])
       
    return hists, train_val_cls

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
    
    

def run_random_hp_opt(param_grid, train_data, num_trails, model_func, epochs, 
                      k=3, p=1, batch_size=128, cv_type='cblind', verbos=1, 
                      drug_cl_index=None, train_loop=None,
                      loss_fn=None, optimiser_fun=None):
    '''Random search to find optmal hyper parameters for pytorch model
    
    Inputs
    -----
    param_grid: Sklearn parameter grid
    grid of hyper parameters (hps) to search over
    
     
    train_data: Dataset instance
    where train_data can be used as an input to pytorch DataLoader.
    Thus, __getitem__ returns x_omic x_drug, y the omic drug and target data
    respectively
    
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
    
    losses: list of cv losses
    
    val_losses: list of cv val losses
    
    '''
        
    #set the cv implementation 
    if cv_type == 'cblind':
        cv_imp = run_cv_cblind_torch
    if cv_type == 'dblind':
        #add imp
        pass
 
    rng = np.random.default_rng()
    hp_inds = rng.choice(len(param_grid), size=num_trails, replace=False)
    
    opt_results = {'Smallest val loss': [], 'SD': [], 'Epoch': [], 'HPs': []}
    run_hists = []
    
    for i, ind in enumerate(hp_inds):
        if verbos > 0:
            print(f'Fraction of search completed: {i/num_trails:.3}')
        hps = param_grid[ind]
        
        hists, train_val_cls = cv_imp(
            m_func=model_func,
            train_data=train_data,
            hp=hps,
            epochs=epochs,
            k=k,
            p=p,
            verbos=verbos - 1,
            random_seed=None,
            drug_cl_index=drug_cl_index,
            train_loop=train_loop,
            loss_fn=loss_fn, 
            optimiser_fun=optimiser_fun,
            batch_size=batch_size
        )
        
        val_losses = [fold['val_loss'] for fold in hists]
        min_loss, sd, epoch = best_metric(val_losses)
        opt_results['Smallest val loss'].append(min_loss)
        opt_results['SD'].append(sd)
        opt_results['Epoch'].append(epoch)
        opt_results['HPs'].append(hps)

        #keep if loss plots needed
        run_hists.append(hists)
        
    return pd.DataFrame(opt_results), run_hists
        
def plot_cv(train, val, epochs, err=2, skip_epochs=0,
            mm_loss = [], y_lab='Loss', 
            save_name=''):
    '''Func to plot the cross validation loss or metric
    
    Inputs
    ------
    train: list, of length number of cv folds, with each element of the list
    contaning a lists with the train set loss or metric. 
        
    val: same as train but with validation data  
    
    epochs: number of epochs model traiend for
    
    err: 0 1 or 2, defalt=2
    If 0, no error plotted
    If 1, error bars (s.d) plotted
    If 2, contionus error plotted
    
    skip_epochs: int, defalt=0
    number of epochs to skip at the start of plotting 
    
    y_lab: str, defalt=loss
    y label to plot
    
    save_name: str, defalt=''
    file_path\name to save fig. if defalt fig not saved
    '''
    x = range(1, epochs + 1 - skip_epochs) 
    val_mean = cv_metric(val, np.mean)
    train_mean = cv_metric(train, np.mean)
    val_sd = cv_metric(val, np.std)
    train_sd = cv_metric(train, np.std)
    if mm_loss:
        val_mm_mean = np.mean(mm_loss)
        val_mm_mean = np.array([val_mm_mean] * epochs)
        val_mm_sd = np.std(mm_loss)
        val_mm_sd = np.array([val_mm_sd] * epochs)
    
    if err == 1:
        plt.errorbar(
            x, train_mean[skip_epochs: ], yerr= train_sd[skip_epochs: ], 
            label='Train')
        plt.errorbar(
            x, val_mean[skip_epochs: ],  yerr = val_sd[skip_epochs: ], 
            label='Validation')
        plt.fill_between(
            x, train_mean[skip_epochs: ] - train_sd[skip_epochs: ], 
            yfit + train_sd[skip_epochs: ], color='gray', alpha=0.2)
        
        
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(y_lab)
        
    if err == 2:
        plt.plot(x, train_mean[skip_epochs: ], label='Train')
        plt.plot(x, val_mean[skip_epochs: ], label='Validation')
        plt.fill_between(x, train_mean[skip_epochs: ] - train_sd[skip_epochs: ], 
                         train_mean[skip_epochs: ] + train_sd[skip_epochs: ], 
                         color='gray', alpha=0.8)
        plt.fill_between(x, val_mean[skip_epochs: ] - val_sd[skip_epochs: ], 
                 val_mean[skip_epochs: ] + val_sd[skip_epochs: ], 
                 color='gray', alpha=0.8)
        if mm_loss:
            plt.plot(x, val_mm_mean[skip_epochs: ], label='Mean')
            plt.fill_between(x, val_mm_mean[skip_epochs: ] - val_mm_sd[skip_epochs: ],
                             val_mm_mean[skip_epochs: ] + val_mm_sd[skip_epochs: ],
                             alpha=0.6)

        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(y_lab)        
        
    else: 
        plt.plot(x, train_mean[skip_epochs: ], label='Train')
        plt.plot(x, val_mean[skip_epochs: ], label='Validation')
        plt.plot(x, val_mm_mean[skip_epochs: ], label='Mean')
        plt.legend()
        
    if save_name:
        plt.savefig(save_name, dpi=1000)
    
    *_, best_epoch = best_metric(val)
    plt.axvline(1 + best_epoch - skip_epochs)
        
    print(best_metric(val))
        
    plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn

def run_cv(m_func, x, y, hp, epochs=10, k=10, p=1, benchmark=False,
           batch_size=128):
    '''runs k fold cv p times 
    
    Inputs
    -----
    m_func: Function
    that returns keras model
    
    x: List
    where x[0] gives input omics data and x[1] gives input drug data
    omic and drug data are pd dataframes. With cell lines in indexes
    and features in cols. 
    
    y: pd index 
    target values
    
    hp: list
    hyper parmaters that are to be inputed to m_func
    
    epochs: int
    number of epochs to train the model for
    
    k: int
    number of folds to run the cross valdation for
    
    p: int
    number of times to repeats the k fold cross validation 
    
    benchmark: bool
    find the loss for a benchmark mean model to compare with the real model.
    
    batch_size: int
    batch size of keras model
    
    Returns
    -------
    loss, val_loss, val_mae, 
    val_loss_mm: loss of mean model benchmark
    train_val_cls: cls used for validation
    '''
    cv = sklearn.model_selection.RepeatedKFold(n_splits=k, n_repeats=p) 
    loss = []
    val_loss = []
    val_mae = []
    val_loss_mm = []
    train_val_cls = ([], [])
    #x_pp, x_drug = x
    
    i = 0
    for train_i, val_i in cv.split(x[0]):
        print(i / (k * p))
        train_cls = np.unique(x[0].iloc[train_i].index)
        val_cls = np.unique(x[0].iloc[val_i].index)
        train_val_cls[0].append(train_cls)
        train_val_cls[1].append(val_cls)
        i += 1
        
        if benchmark: 
            mm_pre = []

            #find mean model on train set 
            mm = create_mean_model(ic50_df1.loc[train_cls])
            #find prediciton of mm for each drug, repeated for the number of drugs
            val_drugs = np.unique(x[1].iloc[val_i].index)
            for d in val_drugs:
                mm_pre.extend([mm[d]] * int(x[1].iloc[val_i].loc[d].values.sum()))
            loss_mm = mean_squared_error(y.iloc[val_i], mm_pre)
            val_loss_mm.append(loss_mm)
        
        model = m_func(hp)
        hist = model.fit(
            [x[0].iloc[train_i], x[1].iloc[train_i]], 
            y.iloc[train_i],
            validation_data=(
                [x[0].iloc[val_i], x[1].iloc[val_i]], 
                y.iloc[val_i]),
            epochs=epochs, 
            batch_size=batch_size,
            verbose=0)

        loss.append(hist.history['loss'])
        val_loss.append([hist.history['val_loss']])
        val_mae.append(hist.history['val_mae'])
        
    return loss, val_loss, val_mae, val_loss_mm, train_val_cls

#plotting funcs


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
    if rounded:
        return m[argmin], np.round(sd[argmin], 3), argmin
    else:
        return m[argmin], sd[argmin], argmin
        

def plot_cv(test, val, epochs, err=2, skip_epochs=0,
            mm_loss = [], y_lab='Loss', 
            save_name=''):
    '''Func to plot the cross validation loss or metric
    
    Inputs
    ------
    test: list, of length number of cv folds, with each element of the list
    contaning a lists with the test set loss or metric. 
        
    val: same as test but with validation data  
    
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
    test_mean = cv_metric(test, np.mean)
    val_sd = cv_metric(val, np.std)
    test_sd = cv_metric(test, np.std)
    if mm_loss:
        val_mm_mean = np.mean(mm_loss)
        val_mm_mean = np.array([val_mm_mean] * epochs)
        val_mm_sd = np.std(mm_loss)
        val_mm_sd = np.array([val_mm_sd] * epochs)
    
    if err == 1:
        plt.errorbar(
            x, test_mean[skip_epochs: ], yerr= test_sd[skip_epochs: ], 
            label='Test')
        plt.errorbar(
            x, val_mean[skip_epochs: ],  yerr = val_sd[skip_epochs: ], 
            label='Validation')
        plt.fill_between(
            x, test_mean[skip_epochs: ] - test_sd[skip_epochs: ], 
            yfit + test_sd[skip_epochs: ], color='gray', alpha=0.2)
        
        
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(y_lab)
        
    if err == 2:
        plt.plot(x, test_mean[skip_epochs: ], label='Test')
        plt.plot(x, val_mean[skip_epochs: ], label='Validation')
        plt.fill_between(x, test_mean[skip_epochs: ] - test_sd[skip_epochs: ], 
                         test_mean[skip_epochs: ] + test_sd[skip_epochs: ], 
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
        plt.plot(x, test_mean[skip_epochs: ], label='Test')
        plt.plot(x, val_mean[skip_epochs: ], label='Validation')
        plt.plot(x, val_mm_mean[skip_epochs: ], label='Mean')
        plt.legend()
        
    if save_name:
        plt.savefig(save_name, dpi=1000)
        
    print(best_metric(val))
        
    plt.show()
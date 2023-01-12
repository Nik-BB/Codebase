'''Reuseable functions for model selection in drug response prediction


These functions can be extened to other problmes with little to no adjustments
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt
import sklearn
import collections
import DRP_utils.testing as t_nb #my testing module




def run_cv(m_func, x, y, hp, epochs=10, k=10, p=1, benchmark=False,
           batch_size=128, verbos=1, random_seed=None):
    '''runs k fold cv p times 
    
    Inputs
    -----
    m_func: Function
    that returns keras model.
    
    x: List
    where x[0] gives input omics data and x[1] gives input drug data
    omic and drug data are array like. shape (num_samples, ...)
    where ... is depended on the dimensions of the features e.g. for PD 
    datafram, shape = (num_samples, num_features) with cell lines in indexes
    and features in cols. 
    
    y: pd series 
    target values.
    
    hp: list
    hyper parmaters that are to be inputed to m_func.
    
    epochs: int
    number of epochs to train the model for.
    
    k: int
    number of folds to run the cross valdation for.
    
    p: int
    number of times to repeats the k fold cross validation.
    
    benchmark: bool
    find the loss for a benchmark mean model to compare with the real model.
    
    batch_size: int
    batch size of keras model.
    
    verbos: int
    if 1 print out cv status.
    
    random_seed: int, or None, deafault=None
    set random seed the same int to get identical splits. 
    
    Returns
    -------
    loss, val_loss, val_mae, 
    val_loss_mm: loss of mean model benchmark
    train_val_cls: cls used for validation
    '''
    cv = sklearn.model_selection.RepeatedKFold(
        n_splits=k,
        n_repeats=p,
        random_state=random_seed) 
    loss = []
    val_loss = []
    val_mae = []
    val_loss_mm = []
    train_val_cls = ([], [])
    #x_pp, x_drug = x
    
    #check same number of samples
    assert len(x[0]) == len(x[1]) == len(y)
    
    
    i = 0
    for train_i, val_i in cv.split(x[0]):
        if verbos == 1:
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
        if type(x[0]) == pd.DataFrame:
            x0 = x[0].to_numpy()
        else:
            x0 = x[0]
        if type(x[1]) == pd.DataFrame:
            x1 = x[1].to_numpy()
        else:
            x1 = x[1]
        hist = model.fit(
            [x0[train_i], x1[train_i]], 
            y.iloc[train_i],
            validation_data=(
                [x0[val_i], x1[val_i]], 
                y.iloc[val_i]),
            epochs=epochs, 
            batch_size=batch_size,
            verbose=0)

        loss.append(hist.history['loss'])
        val_loss.append(hist.history['val_loss'])
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
    #make formatting more readable
    Min_metric = collections.namedtuple('MinMetric', ['mean', 'sd', 'index'])
    if rounded:
        return Min_metric(m[argmin], np.round(sd[argmin], 3), argmin)
    else:
        return Min_metric(m[argmin], sd[argmin], argmin)
        

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


def run_random_hp_opt(param_grid, x, y, num_trails, model_func, epochs,
                      k=3, p=1, batch_size=128):
    '''Random search to find optmal hyper parameters
    
    Inputs
    -----
    param_grid: Sklearn parameter grid
    grid of hyper parameters (hps) to search over
    
    x: List
    where x[0] gives input omics data and x[1] gives input drug data
    omic and drug data are pd dataframes. With cell lines in indexes
    and features in cols. 
    
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
    
    rng = np.random.default_rng()
    hp_inds = rng.choice(len(param_grid), size=num_trails, replace=False)

    opt_results = {'Smallest val loss': [], 'SD': [], 'Epoch': [], 'HPs': []}
    losses, val_losses = [], []
    for i, ind in enumerate(hp_inds):
        print(i/num_trails)
        hps = param_grid[ind]
        loss, val_loss, *_ = run_cv(model_func,
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

class Keras_tuner_opt:
    '''Optmise the hyperparameter using keras tuner for a given dataset.
    
    Inputs
    ------
    
    x: Array like
    Traning input data
    
    y: Array like
    Testing target data
    
    xval: Array like
    Validation input data
    
    yval: Array like
    Validaiton target data
    
    xtest: Array like
    Testing input data
    
    ytest: Array like
    Testing target data

    m_func: Function
    Returns keras tuner model. 
        
    '''
    
    def __init__(self, x=None, y=None, xval=None, yval=None, xtest=None, 
                 ytest=None, model_function=None):
        self.x = x
        self.y = y 
        self.xval = xval
        self.yval = yval
        self.xtest = xtest
        self.ytest = ytest
        self.model_func = model_function
    
    def kt_opt(self, num_trails=1, patience=5, epochs=100, batch_size=128, 
               direc='my_dir', proj_name='test', ): 
        '''Runs hyper parm opt for keras tuner object.
        
        Input
        -----
        
        num_trails: Int
        number of parameters to be randomly sampled
                    
        patience: Int
        number of epochs to wait with metric improvement before triggering 
        eairly stopping 

        direc: str
        directory to save the keras tuner trails to
        
        epochs: int
        number of epochs to train the model for
        
        batch_size: int
        batch size of keras model
        
        proj_name:
        folder under direc to save the kears tuner trails to
        '''
        self.epochs=epochs
    #create tuner object 
        tuner = kt.RandomSearch(
            hypermodel=self.model_func,
            objective="val_loss",
            max_trials=num_trails,
            executions_per_trial=1,
            overwrite=True,
            directory=direc,
            project_name=proj_name,
        )

        #run search 
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=patience,
            restore_best_weights=True)]
        
        tuner.search(
            self.x, 
            self.y,
            validation_data=(self.xval, self.yval),
            epochs=self.epochs,
            callbacks=callbacks,
            batch_size = batch_size
        )

        self.tuner = tuner 
    
    def find_opt_hp(self, plot=True, verbose=0, patience=10):
        '''Finds the optimal paremters of the optmial model
        
        Input
        -----
        plot: bool
        if True plots loss curves 
        if False no plots
        
        verbose: Int
        controls the verbose of the model fitting
        
        patience: Int
        number of epochs to wait with metric improvement before triggering 
        eairly stopping 
        
        Output
        ------
        loss:
        train loss
        
        val_loss: 
        validaiton loss
        '''
        
        opt_model = self.model_func(self.tuner.get_best_hyperparameters()[0])
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=patience,
            restore_best_weights=True)]
        
        hist = opt_model.fit(
            self.x, 
            self.y, 
            validation_data=(self.xval, self.yval), 
            callbacks=callbacks,
            epochs=self.epochs,
            verbose=verbose
        )
        
        loss = hist.history['loss']
        val_loss = hist.history['val_loss']
        epochs = np.argmin(val_loss)
        self.opt_epochs = epochs
        self.best_hps = self.tuner.get_best_hyperparameters()[0]
        print(f'{self.tuner.get_best_hyperparameters()[0].values} epochs: {epochs}')
        print(min(val_loss))
        if plot: 
            ms_nb.plot_cv(loss, val_loss, len(loss))
            
        return loss, val_loss
            
    def test(self):
        t_nb.plot_heatmap(self.tuner.get_best_models()[0], self.xtest, self.ytest)

        
class Drug_response_opt(Keras_tuner_opt):
    '''Spesfic testing for drug response prediction   
    
    '''
    def test_by_cl(self, cl_to_pair, verb=0):
        mse_r2_rho = {'cl': [], 'mse': [], 'r2': [], 'rho': []}
        for key in test_cl_to_pair.keys():
            pairs = test_cl_to_pair[key]
            xt = [self.xtest[0].loc[pairs], self.xtest[1].loc[pairs]]
            pre = self.tuner.get_best_models()[0].predict(xt)
            true = self.ytest.loc[pairs]
            if verb:
                plt.plot(true, pre)
            mse_r2_rho['cl'].append(key)
            mse_r2_rho['mse'].append(sklearn.metrics.mean_squared_error(true, pre))
            mse_r2_rho['r2'].append(sklearn.metrics.r2_score(true, pre))
            mse_r2_rho['rho'].append(scipy.stats.spearmanr(true, pre)[0])
        return pd.DataFrame(mse_r2_rho)

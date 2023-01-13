'''Reuseable functions for cross validation (CV) in drug response prediction

CV is implemented in 3 different ways

Mixed: standared CV there can be overlap between cells or drugs in val and
partial train set
Cancer blind: cell lines in the partial training set can not be in the
validation set
Drug blind: drugs in the partial training set can not be in the
validation set
'''
import sklearn
import numpy as np
import pandas as pd

def run_cv_cblind(m_func, x, y, hp, epochs=10, k=10, p=1, batch_size=128, 
                  verbos=1, random_seed=None):
    '''runs cancer blind k fold cv p times 
    
    cancer blind means cell lines in the train set are not in the val set
    
    Inputs
    -----
    m_func: Function
    that returns keras model.
    
    x: List
    where x[0] gives input omics data and x[1] gives input drug data
    omic and drug data are  PD 
    dataframs, shape = (num_samples, num_features) with cell lines in indexes
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
    
    #find a list of jsut cell lines and unique cell lines
    cells = np.array([cdp.split('::')[0] for cdp in x[0].index])
    u_cells = np.unique(cells)
    i = 0
    #splits by cell lines then finds all cell line drug pairs assicted with
    #these cell lines
    for train_c_i, val_c_i in cv.split(u_cells):
        if verbos == 1:
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
        
        model = m_func(hp)
        if type(x[1]) == pd.DataFrame:
            x1 = x[1].to_numpy()
        else:
            x1 = x[1]
        if type(x[0]) == pd.DataFrame:
            x0 = x[0].to_numpy()
        else:
            x0 = x[0]
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


def run_cv_dblind(m_func, x, y, hp, epochs=10, k=10, p=1, batch_size=128, 
                 verbos=1, random_seed=None):
    '''runs drug blind k fold cv p times 
    
    cancer blind means cell lines in the train set are not in the val set
    
    Inputs
    -----
    m_func: Function
    that returns keras model.
    
    x: List
    where x[0] gives input omics data and x[1] gives input drug data
    omic and drug data are  PD 
    dataframs, shape = (num_samples, num_features) with cell lines in indexes
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
    
    #find a list of jsut cell lines and unique cell lines
    drugs = np.array([cdp.split('::')[1] for cdp in x[0].index])
    u_drugs = np.unique(drugs)
    i = 0
    for train_d_i, val_d_i in cv.split(u_drugs):
        if verbos == 1:
            print(i / (k * p))
        p_train_drugs, val_drugs = u_drugs[train_d_i], u_drugs[val_d_i]
        train_i, val_i = [], []
        for ptd, vd in zip(p_train_drugs, val_drugs):
            train_i.extend(np.where(drugs == ptd)[0])
            val_i.extend(np.where(drugs == vd)[0])
        #reshuffles inds so not ordered by cell line    
        train_i.sort(), val_i.sort()
        assert set(train_i).intersection(val_i) == set()
        i += 1
        
        model = m_func(hp)
        
        #convert to numpy
        if type(x[1]) == pd.DataFrame:
            x1 = x[1].to_numpy()
        else:
            x1 = x[1]
        if type(x[0]) == pd.DataFrame:
            x0 = x[0].to_numpy()
        else:
            x0 = x[0]
            
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
        
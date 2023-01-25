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
from torch import nn
import numpy as np
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu" 
device

def run_cv_cblind_torch(m_func, train_data, hp, epochs=10, k=3, p=1, verbos=1, 
                        random_seed=None, drug_cl_index=None, train_loop=None,
                        batch_size=512, loss_fn=nn.MSELoss(), 
                        optimiser_fun=None):
    '''runs cancer blind k fold cv p times for pytorch model
    
    cancer blind means cell lines in the train set are not in the val set
    
    Inputs
    -----
    m_func: Function
    that returns pytorch model.
    
    train_data: torch Dataloader instance
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
            epochs=epochs)
        
        hists.append(hist)
        #loss.append(hist.history['loss'])
        #val_loss.append(hist.history['val_loss'])
       
    return hists, train_val_cls
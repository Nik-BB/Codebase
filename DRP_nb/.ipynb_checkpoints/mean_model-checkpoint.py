'''Mean model functions, class to create mean model and func to run cv

'''
import numpy as np
import sklearn
from collections import defaultdict


#make mean model
class MeanModel():
    '''Creates benchmark model that uses mean truth values for prediciton.
    
    For a given drug cell lines pair, d1 c1 in the test set the model predicts
    the mean truth value of all drug cell line pairs in the traning set 
    that include d1.
    
    All drugs need to be in the traning set.
    
    Inputs
    -----
    
    y_train: pd series or dataframe
    traning truth values were index of the df gives cell_line drug 
    seprated by the string :: e.g. for d1 and cl1 index = 'd1::cl1'
    
    all_drugs: list like
    gives all drug names 
    
    Methods
    ------
    predict(y_index): gives models prediciton for cell line drug pairs
    
    replace_nan(re='mean') replace missing values with re
    '''
    
    def __init__(self, y_train, drugs, verb=1):
        model = defaultdict(list)
        #group cls by drugs
        self.verb = verb
        for ind, val in y_train.items():
            cl, d = ind.split('::')
            model[d].append(val)
            
        #take average of all values for a given drug   
        for d in drugs:
            model[d] = np.mean(model[d])
        
        self.model = model
        
    def predict(self, y_index, reformat=True):
        #reformat index to get just drug
        if reformat:
            y_index = [y.split('::')[1] for y in y_index]
            
        return np.array([self.model[y] for y in y_index])
    
    def replace_nan(self, re='mean'):
        #replace nan's with re, deflat re=0
        num_nan = 0
        for k in self.model:
            if np.isnan(self.model[k]):
                num_nan += 1
                if re=='mean':
                    vals = np.array(list(self.model.values()))
                    vals = vals[~np.isnan(vals)]
                    self.model[k] = vals.mean()
                else:
                    self.model[k] = re
        if self.verb > 0:
            print(f'{num_nan} nan values replaced out of {len(self.model)}')


def run_cv_cblind_mm(m_func, y_train, drugs, k=3, p=1, verbos=1, 
                        random_seed=None):
    '''runs cancer blind k fold cv p times for pytorch model
    
    cancer blind means cell lines in the train set are not in the val set
    
    Inputs
    -----
    m_func: Function
    that returns pytorch model.
    
    y_train: pd dataframe
    dataframe of traning data where index gives drug cell line names in 
    fmt drug::cell line
    
    k: int
    number of folds to run the cross valdation for.
    
    p: int
    number of times to repeats the k fold cross validation.
    
    verbos: int
    if > 1 print out cv status.
    
    
    Returns
    -------
    hist: dic that gives history of model traning
    train_val_cls: cls used for validation
    '''
    
    cv = sklearn.model_selection.RepeatedKFold(
        n_splits=k,
        n_repeats=p,
        random_state=random_seed) 
    
    train_val_cls = ([], [])
    hists = {'r2': [], 'mse': []}
    
    
    #find a list of jsut cell lines and unique cell lines
    cells = np.array([cdp.split('::')[0] for cdp in y_train.index])
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
        
        y_pt = y_train.iloc[train_i]
        y_val = y_train.iloc[val_i]
        
        mm = m_func(y_pt, drugs)
        mm.replace_nan()
        mm_pre = mm.predict(y_val.index)
        r2 = r2_score(y_val, mm_pre)
        mse = mean_squared_error(y_val, mm_pre)
        hists['r2'].append(r2)
        hists['mse'].append(mse)
       
    return hists, train_val_cls
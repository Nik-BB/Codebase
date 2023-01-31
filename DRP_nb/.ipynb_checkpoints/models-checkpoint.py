'''Drug response prediction models

'''
import numpy as np
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
    
    def __init__(self, y_train, drugs):
        model = defaultdict(list)
        #group cls by drugs
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
        if verb > 0:
            print(f'{num_nan} nan values replaced out of {len(self.model)}')


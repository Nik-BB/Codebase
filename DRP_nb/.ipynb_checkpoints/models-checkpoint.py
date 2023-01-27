'''Drug response prediction models

'''
import numpy as np
from collections import defaultdict

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
    '''
    
    def __init__(self, y_train, all_drugs):
        model = defaultdict(list)
        #group cls by drugs
        for ind, val in y_train.iteritems():
            cl, d = ind.split('::')
            model[d].append(val)
            
        #take average of all values for a given drug   
        for d in all_drugs:
            model[d] = np.mean(model[d])
        
        self.model = model
        
    def predict(self, y_index, reformat=True):
        
        '''Gives models prediciton for cell line drug pairs
        
        Input
        -----
        y_index : pd series or dataframe
        same format as y_train
        '''
        #reformat index to get just drug
        if reformat:
            y_index = [y.split('::')[1] for y in y_index]
            
        return np.array([self.model[y] for y in y_index])
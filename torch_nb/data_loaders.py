''' Custom dataset loaders for use with torch Dataloader
'''
import numpy as np
from torch.utils.data import Dataset

class DualPdInputs(Dataset):
    '''Creates dataset for use with pytorch data loader for two inputs x1 x2
    
    ---Inputs---
    x1: np array or pd dataframe
    input data 1
    
    x2: np array or pd dataframe
    input data 2
    
    y: np array pd dataframe
    target data
    
    expand_x1_dim: bool, defalt=False
    Add empty dimention to x1 in axis=1 can be used for channel dim in CNN
    e.g. if expand_x1_dim=True and x1.shape=(N, F) --> x1.shape=(N, 1, F)
    
    expand_x2_dim: bool, defalt=False
    equivalent for x2 as expand_x1_dim
    '''
    def __init__(self, x1, x2, y, expand_x1_dim=False, expand_x2_dim=False):
        #convert to numpy arrays
        if type(x1) != np.ndarray:
            x1 = x1.to_numpy(dtype='float32')
        if type(x2) != np.ndarray:
            x2 = x2.to_numpy(dtype='float32')
        if type(y) != np.ndarray:
            y = y.to_numpy(dtype='float32')
        #reshape y to be same shape as torch model output
        if len(y.shape) == 1:  
            y = np.expand_dims(y, -1)
        if expand_x1_dim == True:
            x1 = np.expand_dims(x1, 1)
        if expand_x2_dim == True:
            x2 = np.expand_dims(x2, 1)
            
        self.x1 = x1
        self.x2 = x2
        self.y = y
        
        assert len(self.x1) == len(self.x2)
        assert len(self.y) == len(self.x1)
        
    def __len__(self):
        return len(self.x1)
    
    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]
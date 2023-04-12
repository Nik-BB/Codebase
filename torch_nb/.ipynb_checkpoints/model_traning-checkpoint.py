'''Pytorch traning loops

'''
import torch
from torch import nn
import numpy as np
from scipy.stats import pearsonr
device = "cuda" if torch.cuda.is_available() else "cpu"

def train_loop(train_dl=None, val_dl=None, model=None, loss_fn=nn.MSELoss(), 
               optimiser=None, epochs=10, verb=1):
    '''Torch train loop for arbitrary number of input data types
    
    '''  
    if verb > 0:
        print('')
        print(f"Using {device} device")
        print('')
    
    train_hist = {'train_loss': [], 'val_loss': []}
    for e in range(epochs):
        
        loss_train = 0.0
        model.train()
        for batch, (*x, y) in enumerate(train_dl):
            
            x = [inp.to(device) for inp in x] # Needed for multiple inputs
            y = y.to(device)

            # Prediction error
            pred = model(*x)
            loss = loss_fn(pred, y)

            # Backprop
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            loss_train += loss.item()
        
        train_hist['train_loss'].append(loss_train / len(train_dl))
        
        if val_dl:
            # Find validaiton loss
            val_loss = 0.0
            model.eval()   
            with torch.no_grad():
                for *xv, yv in val_dl:

                    xv = [inp.to(device) for inp in xv]
                    yv = yv.to(device)

                    val_pred = model(*xv)
                    v_loss = loss_fn(val_pred, yv)
                    val_loss += v_loss.item()
            train_hist['val_loss'].append(val_loss / len(val_dl))    
            
        if verb > 0:
            print(f'Epoch {e + 1}\n-----------------')
            ftl = loss_train / len(train_dl)
            if val_dl:
                fvl = val_loss / len(val_dl)
                print(f'Train loss: {ftl:>5f}, Val loss: {fvl:>5f}')
            else:
                print(f'Train loss: {ftl:>5f}')
                
    return train_hist


#early stopping implementation
class EarlyStopping:
    '''Class to implement early stopping
    
     ------inputs------
    model: PyTorch model 
    
    patience: int, defalt=10
    number of epochs the model can not meet the improvement criteria before
    early stopping triggers
    
    delta: int, defalt=0
    How much the loss needs to decrease by to count as an improvement
    e.g. delta=1 means the loss needs to be at least 1 less than previous best loss
    '''
    def __init__(self, model, patience=10, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.count = 0
        self.min_val_loss = np.inf
        self.best_model_dict = None
        self.model = model
        
    def earily_stop(self, val_loss):
        #if loss improves 
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.count = 0 
            self.best_model_dict = self.model.state_dict()
        #if loss does not improved more than delta 
        elif val_loss > (self.min_val_loss + self.delta):
            self.count += 1
            if self.count >= 10: 
                return True
            
        return False #if stopping contions not met
        


#train loop for mutiple dataloaders 
def tl_multi_dls(train_dls=None, y_train=None, val_dls=None, y_val=None, 
                 model=None, loss_fn=nn.MSELoss(), optimiser=None, 
                 epochs=10, verb=1, early_stopping_dict=None):
    '''torch train loop for two data loaders
    
    ------inputs------
    train_dls: iterable 
    contains multiple inputs where each input is a data loader
    
    
    ------returns------
    rain_hist, best_model_dict
    
    if early stopping implemented best model is used to overwrite model
    '''
    train_hist = {'train_loss': [], 'val_loss': []}
    assert type(train_dls) == list
    
    #early stopping
    if early_stopping_dict and val_dls:
        p, d = early_stopping_dict['patience'], early_stopping_dict['delta']
        early_stopper = EarlyStopping(model, patience=p, delta=d)
    
    for e in range(epochs):
        #train_dls_y = train_dls
        #train_dls_y.append(y_train)
        loss_train = 0.0
        model.train()
        for batch, (*x, y) in enumerate(zip(*train_dls, y_train)):
            x = [inp.to(device) for inp in x]
            y = y.to(device)

            # Compute prediction error
            pred = model(*x)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            loss_train += loss.item()
            
        train_hist['train_loss'].append(loss_train / len(train_dls[0]))
        
        if val_dls:
            #val_dls_y = val_dls
            #val_dls_y.append(y_val)
            #find validaiton loss
            val_loss = 0.0
            r2 = 0.0
            mse_val = 0.0 

            model.eval()   
            with torch.no_grad():
                for *xv, yv in zip(*val_dls, y_val):
                    xv = [inp.to(device) for inp in xv]
                    yv = yv.to(device)
                    val_pred = model(*xv)
                    #val_pred = val_pred.reshape(len(val_pred))
                    v_loss = loss_fn(val_pred, yv)
                    val_loss += v_loss.item()
                    yv = yv.cpu().detach().numpy()
                    val_pred = val_pred.cpu().detach().numpy()
            train_hist['val_loss'].append(val_loss / len(val_dls[0]))
            
            #eairly stopping
            if early_stopper:
                if early_stopper.earily_stop(val_loss):
                    print(f'stopping early at epoch {e + 1}')
                    best_model_dict  = early_stopper.best_model_dict
                    model = model.load_state_dict(best_model_dict)
                    model.eval()
                    break
                    
        if verb > 0:
            print(f'Epoch {e + 1}\n-----------------')
            ftl = loss_train / len(train_dls[0])
            if val_dls:
                fvl = val_loss / len(val_dls[0])
                print(f'Train loss: {ftl:>5f}, Val loss: {fvl:>5f}')
                print(pearsonr(yv.reshape(len(yv)), 
                               val_pred.reshape(len(val_pred))))
            else:
                print(f'Train loss: {ftl:>5f}')
    return train_hist, best_model_dict
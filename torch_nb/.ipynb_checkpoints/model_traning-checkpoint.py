'''Pytorch traning loops

'''
import torch
from torch import nn
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


#train loop for mutiple dataloaders 
def tl_multi_dls(train_dls=None, y_train=None, val_dls=None, y_val=None, model=None, loss_fn=nn.MSELoss(), 
               optimiser=None, epochs=10, verb=1):
    '''torch train loop for two data loaders
    
    ------inputs------
    train_dls: iterable 
    contains multiple inputs where each input is a data loader
    
    '''
    train_hist = {'train_loss': [], 'val_loss': []}
    assert type(train_dls) == list
    
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
    return train_hist
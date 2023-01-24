'''Pytorch traning loops

'''
import torch
from torch import nn


def train_loop(train_dl=None, val_dl=None, model=None, loss_fn=nn.MSELoss(), 
               optimiser=None, epochs=10, verb=1):
    '''Torch train loop for arbitrary number of input data types
    
    '''    
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

        if verb > 0:
            print(f'Epoch {e + 1}\n-----------------')
            ftl = loss_train / len(train_dl)
            if val_dl:
                fvl = val_loss / len(val_dl)
                print(f'Train loss: {ftl:>5f}, Val loss: {fvl:>5f}')
            else:
                print(f'Train loss: {ftl:>5f}')
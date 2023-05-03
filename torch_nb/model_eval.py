'''Functions used to evaluate pytorch models

'''

import numpy as np 
import torch

device = "cuda" if torch.cuda.is_available() else "cpu" 
device

def predict(dl1, dl2, model):
    '''Find regresstion predictions of a pytorch model
    
    where the model takes two dataloaderes as inputs
    
    ---inputs---
    dl1: DataLoader
    dl2: DataLoader
    model: torch model that takes two inputs
    
    ---returns---
    array: np array of predictions of the model, shape = len(dl1)
    
    '''
    model.eval()
    preds = []
    with torch.no_grad():
        for xo, xd in zip(dl1, dl2):
            xo, xd = xo.to(device), xd.to(device)
            pred = model(xo, xd)
            pred = pred.cpu().detach().numpy()
            pred = pred.reshape(len(pred))
            preds.extend(pred)
    return np.array(preds)
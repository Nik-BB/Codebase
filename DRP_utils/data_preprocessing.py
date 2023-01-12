'''Reuseable generalisable functions for data preprocessing in DRP.

These functions can be extened to other problmes with little adjustment.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

def keep_overlapping(df1, df2):
    '''Only keeps the cell lines that are in both df1 and df2
    
    Inputs
    ------
    df1: pd dataframe
    has cell lines in the index of the datframe with no duplicates
    
    df2: pd dataframe
    has cell liens in the index of the dtaframe with no duplicates

    Returns
    -------
    x1: pd dataframe
    with only the cell lines that are also in x2
    
    x2: pd dataframe
    with only the cell lines that are also in x1
    '''
    x1 = df1.copy(deep=True)
    x2 = df2.copy(deep=True)
    
    #check for duplicates 
    assert (x1.index.duplicated() == False).all()
    assert (x2.index.duplicated() == False).all()
    
    #cls that are in x1 but not x2 
    drop_x1 = set(x1.index).difference(x2.index)
    x1.drop(index=drop_x1, inplace=True)

    #cls that are in x2 but not x1
    drop_x2 = set(x2.index).difference(x1.index)
    x2.drop(index=drop_x2, inplace=True)
    
    #puts the indices in the same order (only works if all indices are shared)
    x2 = x2.loc[x1.index]
    
    return x1, x2 

def create_all_drugs(x, xd, y, cells):
    '''Create data for all drug and cell line pairs, for use in models.
    
    With cell line data (x) that is not drug spesfic (i.e. the same for 
    all drugs) copies this data for each drug while removing missing values 
    that are contained in y as nan.
    The indexes in the dataframes created agree with each other. 
    E.g. the zeorth index of the dataframes corrisponds to the 
    drug cell line pair given by x.iloc[0], y.iloc[0].
    
    Inputs
    -------
    x: pd dataframe.
    Omic data (i.e. phospo) where the index is the cell lines
    and cols are features.
    
    xd: pd dataframe.
    One hot encoded representation of the drugs.
    
    y: pd datafame.
    Target values (i.e. ic50 values) where the index is 
    the cell lines and cols are the drugs. 
    
    cells: list array or index.
    The cell lines for which the final dataframe should be created for.
    Typically this will be the traning cell lines as testing can be done per 
    drug. It can also be all cells if the test train split has not been done.
    
    Returns
    -------
    x_final: pd dataframe.
    Omics data for all drugs and cell lines
    
    X_drug_final: pd dataframe.
    One hot enocding for all drugs and cell lines
    
    y_final: pd index
    Target values for all drugs and cell lines

    '''
    drug_inds = []
    x_dfs = []
    x_drug_dfs = []
    y_final = []
    
    #only consdier cell lines that are required. 
    y = y.loc[cells]
    x = x.loc[cells]
    x.astype(np.float16)
    for i, d in enumerate(xd.columns):
        #find cell lines without missing truth values
        y_temp = y[d]
        nona_cells = y_temp.index[~np.isnan(y_temp)]
        #finds the index for the start / end of each drug
        ind_high = len(nona_cells) + i
        drug_inds.append((d, i, ind_high))
        i += len(nona_cells)

        #store vals of the cell lines with truth values
        x_pp = x.loc[nona_cells] 
        x_dfs.append(x_pp)
        X_drug = pd.DataFrame([xd[d]] * len(x_pp))
        x_drug_dfs.append(X_drug)
        y_final.append(y_temp.dropna())

    #combine values for all drugs  
    x_final = pd.concat(x_dfs, axis=0)
    X_drug_final = pd.concat(x_drug_dfs, axis=0)
    y_final = pd.concat(y_final, axis=0)
    
    return x_final, X_drug_final, y_final

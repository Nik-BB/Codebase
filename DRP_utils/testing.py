'''Reuseable functions for sorting and plotting models predicitons. 

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import collections
import scipy

def sort_results(pre, true, cls_drug_inds, centered):
    '''Sorts results so that they are drug and cell lines centered.
    
    Allows the resuts to viwed in terms of cell lines or drugs.
    I.e. The peformance of the model for a given cell line or drug 
    in the test set
    
    Inputs
    ------
    pre: array or pd index
    models predicted results
    
    true: array or pd index
    target values
    
    cls_drug_inds: array or pd index
    gives the cell line and drug for the true and pre values.
    Indices need to match pre and true. I.e. cls_drug_inds[0] need to give
    the cell line and drug that corrisonds with true[0] and pre[0] (When 
    inputs are flat arrays)
    Entires need to be in the format 'cellLine.drug'
    
    centered: Int, 0 or 1
    if the sorted results should be drug or cell line centered.
    0 the sorted results are cell line centered
    1 the sorted results are drug centered 
    
    Retruns
    -------
    sorted_results: dict
    keys either drugs or cell lines. Items tuple of predicted , true values 
    where predcited and true are lists. 
    '''
    #check same shape 
    assert pre.shape == true.shape
    
    sorted_results = {}
    
    for pre, true, cl_drug in zip(pre, true, cls_drug_inds):
        cl, drug = cl_drug.split('.')
        if centered == 0:
            key = cl
        if centered == 1:
            key = drug
        if key not in sorted_results.keys():
            sorted_results[key] = ([pre], [true])
        else:
            sorted_results[key][0].append(pre)
            sorted_results[key][1].append(true)
            
    return sorted_results


def plot_heatmap(trained_model, x, y):
    '''Plots heatmap for true vs predicted and prints metrics
    
    Input
    ------
    trained_model: Keras
    A trained model where trained_model.predict gives the
    predicitons of the model.
    
    x: array 
    data compatalbe with input to trained_model s.t
    trained_model(x) gives the prediction for x.
    
    y: array
    target values assocated with x.
    '''
    prediction = trained_model.predict(x)
    prediction = prediction.reshape(len(prediction))
    fig, ax = plt.subplots(figsize=(8, 5))
    pcm = ax.hist2d(prediction, y, bins=75, cmap=plt.cm.jet)
    fig.colorbar(pcm[3])
    plt.show()
    
    #improve formatting
    Scores = collections.namedtuple('Testing', ['R2','MSE'])
    rho = scipy.stats.spearmanr(y, prediction)
    score = Scores(sklearn.metrics.r2_score(y, prediction),
                   sklearn.metrics.mean_squared_error(y, prediction))
    print(score)
    print(rho)

    
def multi_test_run(model_func, hps, epochs, xtrain, ytrain,
                   xtest, ytest, num_runs, batch_size=128):
    '''Builds and tests a keras regresstion model multiple times
    
    Finds the average and SD for the R^2 and MSE metrics across multiple runs  
    using the same raning and testing data, model and hyper parms. This gives
    the spread due to different initalisations / local minimums.
    
    Input
    -----
    model_func: function
    retruns a complied keras model.
    
    hps: dict
    hyper parameters accpected by model_func.
    
    epochs: int
    number of epochs to train the model for.
    
    xtrain: List
    where x[0] gives input traning omics data and x[1] gives input traning
    drug data. Omic and drug data are pd dataframes. With cell lines in 
    indexes and features in cols. 
    
    ytrain: pd series 
    target tranning values.
    
    xtest: List
    where x[0] gives input testing omics data and x[1] gives input testing
    drug data. Omic and drug data are pd dataframes. With cell lines in 
    indexes and features in cols. 
    
    ytest: pd series 
    target testing values.
    
    num_runs: int
    number of times to re-run the model and find metrics.
    
    Returns
    -------
    
    summary_r: pd dataframe
    summary statstics (mean standard devation) of the metrics over the 
    multiple runs of the model.
    
    full results: pd dataframe
    R^2 and MSE for each run of the model. 
    '''
    
    r_dict = {'r2': [], 'mse': []}
    for r in range(num_runs):
        print(np.round(r / num_runs, 3))
        
        model = model_func(hps)
        model.fit(xtrain, ytrain, epochs=epochs,
                  batch_size=batch_size, verbose=0)
        pre = model.predict(xtest)
        pre = pre.reshape(len(pre))
        r2 = sklearn.metrics.r2_score(ytest, pre)
        mse = sklearn.metrics.mean_squared_error(ytest, pre)
        r_dict['r2'].append(r2)
        r_dict['mse'].append(mse)
    summary_r = {'r2_mean': np.mean(r_dict['r2']),
                'r2_sd': np.std(r_dict['r2']),
                'mse_mean': np.mean(r_dict['mse']),
                'mse_sd': np.std(r_dict['mse'])}
    
    return summary_r, pd.DataFrame(r_dict)
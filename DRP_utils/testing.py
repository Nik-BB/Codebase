'''Reuseable functions for sorting and plotting the models prediciton for drug response prediction. 

'''


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
    prediction.reshape(len(prediction))
    fig, ax = plt.subplots(figsize=(8, 5))
    pcm = ax.hist2d(hot_pre, y, bins=75, cmap=plt.cm.jet)
    fig.colorbar(pcm[3])
    plt.show()
    
    #improve formatting
    Scores = namedtuple('Testing', ['R2','MSE'])
    score = Scores(r2_score(prediction, y), 
                   mean_squared_error(prediction, y))
    print(score)
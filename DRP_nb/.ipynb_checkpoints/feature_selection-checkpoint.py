'''Commonly used feature selection methods

'''

import os
import pandas as pd
import numpy as np
#get parent dir that should be codebase 
codebase_path = os.getcwd()

def ltl(phos_features):
    '''FS for landmark target landmark phos data
    
    '''
    landmark_genes = pd.read_csv(
    f'{codebase_path}/downloaded_data_small/landmark_genes_LINCS.txt',sep='\t')
    landmark_genes.index = landmark_genes['Symbol']
    
    #read in targetds data
    path = '/downloaded_data_small/enzsub_relations_phosSite_signor.csv'
    enz_sub = pd.read_csv(
        f'{codebase_path}{path}', index_col=0)
    enz_sub.index = enz_sub['enzyme_genesymbol']
    
    
    phos_genes = np.array([c.split('(')[0] for c in phos_features])
    unique_phos = np.unique(phos_genes)
    overlapping_landmarks = [g for g in landmark_genes.index if g in unique_phos]


    #landmark genes with targets
    lands_with_target = [g for g in landmark_genes.index if g in enz_sub.index]
    land_targets = set(enz_sub.loc[lands_with_target]['substrate_genesymbol'])
    #landmark genes with targets that are in phos data
    overlapping_land_targets = [t for t in land_targets if t in unique_phos]

    targets_also_landmarks = set(overlapping_land_targets).intersection(
        landmark_genes.index) 

    overlap_inds = []
    for gene in targets_also_landmarks:
        overlap_inds.extend(np.where(phos_genes == gene)[0])

    kep_cols = phos_features[overlap_inds]
    
    return kep_cols

''' Functions for spesfic data read ins and preprocessing.

These functions are not generalisable due to the very different data formats
of omics and drug response data downloaed. 
Howerver, the functions are used offten across multiple projects with my DRP. 

To do
-----
Change input norm if dblind or mixed
should norm by drug if dblind should norm by train set if mixed
'''

import numpy as np
import pandas as pd
import sys
import DRP_utils.data_preprocessing as dp_nb
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
#get parent dir that should be codebase 
codebase_path = os.getcwd()
#codebase_path = '/data/home/wpw035/Codebase' #old
sys.path.insert(0, codebase_path) #add path to my codebase models

#read and format target values
def read_targets():
    df_ic50 = pd.read_csv(f'{codebase_path}/downloaded_data_small/GDSC1_ic50.csv')
    frame = {}
    for d in np.unique(df_ic50['CELL_LINE_NAME']):
        cellDf = df_ic50[df_ic50['CELL_LINE_NAME'] == d]
        cellDf.index = cellDf['DRUG_NAME']
        frame[d] = cellDf['LN_IC50']


    def remove_repeats_mean_gdsc1(frame, df_ic50): 
        new_frame = {}
        for cell_line in np.unique(df_ic50['CELL_LINE_NAME']):
            temp_subset = frame[cell_line].groupby(frame[cell_line].index).mean()
            new_frame[cell_line] = temp_subset
        return new_frame  

    new_frame = remove_repeats_mean_gdsc1(frame, df_ic50)
    ic50_df1 = pd.DataFrame(new_frame).T
    
    return ic50_df1

def one_hot_encode(x):
    '''One hot encodes list like input
    '''
    frame = {}
    for i, label in enumerate(x):
        hot_vec = np.zeros(len(x))
        hot_vec[i] = 1
        frame[label] = hot_vec
    encoded_df = pd.DataFrame(frame)
    return encoded_df


def read_phos_druml(norm_type=StandardScaler):
    '''Read phos data from DRUML paper
    DRUML: Drug ranking using machine learning systematically predicts the 
    efficacy of anti-cancer drugs
    
    '''
    phos_path ='/downloaded_data_small/suppData2ppIndexPhospo.csv'
    phos_raw = pd.read_csv(f'{codebase_path}{phos_path}')
    #makes index features 
    phos_raw.index = phos_raw['col.name']
    phos_raw.drop(columns='col.name', inplace=True)
    #formats cell lines in the same way as in target value df. 
    phos_raw.columns = [c.replace('.', '-') for c in phos_raw.columns]
    phos_raw = phos_raw.T
    
    #log transfrom (dont think .replace is needed / doing anything)
    phospho_log = np.log2(phos_raw).replace(-np.inf, 0)
    #norm by cell line standard scale 
    if norm_type:
        scale = norm_type()
        phospho_ls = pd.DataFrame(scale.fit_transform(phospho_log.T), 
                                  columns = phospho_log.index, 
                                  index = phospho_log.columns).T
    return phospho_log


def read_rna_gdsc():
    #read in rna-seq data
    gdsc_path = '/data/home/wpw035/GDSC'
    rna_raw = pd.read_csv(f'{gdsc_path}/downloaded_data/gdsc_expresstion_dat.csv')
    rna_raw.index = rna_raw['GENE_SYMBOLS']
    rna_raw.drop(columns=['GENE_SYMBOLS','GENE_title'], inplace=True)
    cell_names_raw = pd.read_csv(f'{gdsc_path}/downloaded_data/gdsc_cell_names.csv', skiprows=1, skipfooter=1)
    cell_names_raw.drop(index=0, inplace=True)

    #chagne ids to cell names
    id_to_cl = {}
    for _, row in cell_names_raw.iterrows():
        cell_line = row['Sample Name']
        ident = int(row['COSMIC identifier'])
        id_to_cl[ident] = cell_line

    ids = rna_raw.columns
    ids = [int(iden.split('.')[1]) for iden in ids] 

    #ids that are in rna_raw but don't have an assocated cl name 
    #from cell_names_raw (not sure why we have these)
    missing_ids = []
    for iden in ids:
        if iden not in id_to_cl.keys():
            missing_ids.append(iden)
    missing_ids = [f'DATA.{iden}' for iden in missing_ids]      
    rna_raw.drop(columns=missing_ids, inplace=True)

    cell_lines = []
    for iden in ids:
        try:
            cell_lines.append(id_to_cl[iden])
        except KeyError:
            pass
    rna_raw.columns = cell_lines
    rna_raw = rna_raw.T

    #take out duplicated cell line
    rna_raw = rna_raw[~rna_raw.index.duplicated()] 
    #take out nan cols
    rna_raw = rna_raw[rna_raw.columns.dropna()]
    #take out duplciated cols
    rna_raw = rna_raw.T[~rna_raw.columns.duplicated()].T

    #check for duplications and missing value in cols and index
    assert sum(rna_raw.index.duplicated()) == 0
    assert sum(rna_raw.columns.duplicated()) == 0
    assert sum(rna_raw.index.isna()) == 0
    assert sum(rna_raw.columns.isna()) == 0
    
    return rna_raw


def read_prot_map():
    ''' Read data from Pan-cancer proteomic map of 949 human cell lines paper
    
    '''
    #read in protmics data
    uniprot_ids = pd.read_csv(
        f'{codebase_path}/downloaded_data_small/Proteinomics_large.tsv',
        sep = '\t', skipfooter = 951).columns[2 :] #keep ids of col names

    prot_raw = pd.read_csv(
        f'{codebase_path}/downloaded_data_small/Proteinomics_large.tsv',
        sep = '\t', header=1
    )
    prot_raw.drop(index=0, inplace=True)

    prot_raw.index = prot_raw['symbol']
    prot_raw.drop(columns=['symbol', 'Unnamed: 1'], inplace=True)
    
    #replace missing protomics values
    p_miss = prot_raw.isna().sum().sum() / (len(prot_raw) * len(prot_raw.columns))
    print(f'% of missing prot values {p_miss}')
    #close to 40% of the dataframe is missing valuse
    #replace nan with zero (as done in the paper)
    prot = prot_raw.replace(np.nan, 0)

    #check for duplications and missing value in cols and index
    assert sum(prot.index.duplicated()) == 0
    assert sum(prot.columns.duplicated()) == 0
    assert sum(prot.index.isna()) == 0
    assert sum(prot.columns.isna()) == 0
    
    return prot

def read_smiles(max_smile_len=90, pad_character='!', 
                smile_path=f'{codebase_path}/downloaded_data_small'\
                '/drugs_to_smiles'):
    '''Read and procsses smiles to all be the same length
    
    defalt max_smile_len=90 deterimed from hist of all smile lengths:
    
    Only 1 smlie is longer than 133 in length.
    Thus,  could cut off at this length and pad the rest of the smiles
    But only 11 smiles longer than 90. so will instead to cut off at 90. 
    Menas 43 less padding for majorty of smiles.
    
    '''
    
    drugs_to_smiles_raw = pd.read_csv(smile_path, index_col=0)
    
    #checks 
    smile_len = [len(smile) for smile in drugs_to_smiles_raw['smiles']]
    smile_len = np.array(smile_len)
    num_short = len(smile_len[smile_len > max_smile_len])
    mean_pad = np.mean(smile_len[smile_len < max_smile_len])
    print(f'Num smiles that will be shortened {num_short}')
    print(f'Mean number of padding characters needed {mean_pad}')
    
    #do padding and shortening
    new_smiles = []
    for smile in drugs_to_smiles_raw['smiles']:
    #subset smiles that are too long
        if len(smile) > max_smile_len:
            new_smile = smile[: max_smile_len]
        #add padding to shorter smiles
        elif len(smile) < max_smile_len:
            new_smile = smile + '!' * (max_smile_len - len(smile))
        new_smiles.append(new_smile)

    drugs_to_smiles = pd.DataFrame(
        new_smiles, index=drugs_to_smiles_raw.index, columns=['smiles'])
    #check above is done correctly. 
    for smile in drugs_to_smiles['smiles']:
        assert len(smile) == max_smile_len
    
    return drugs_to_smiles

def create_drug_to_hot_smile(max_smile_len=188):
    '''creates dict to map drug name to one hot rep of smiles
    
    '''
    
    dts = read_smiles(max_smile_len=max_smile_len) 

    #creates smiles one-hot encoding
    label_encoder = LabelEncoder()
    int_encoded = label_encoder.fit_transform(list(''.join(dts['smiles'])))
    int_encoded = int_encoded.reshape(len(dts), max_smile_len)

    one_hot_smiles  = np.zeros(
        shape = (len(dts), max_smile_len,max(int_encoded.flatten()) + 1))
    for i, sample in enumerate(int_encoded):
        for j, char_numb in enumerate(sample):
            one_hot_smiles[i, j, char_numb] = 1

    #dict to map drug to one hot enconded vec
    drug_to_hot_smile = {}
    for i, drug in enumerate(dts.index):
        drug_to_hot_smile[drug] = one_hot_smiles[i]


    # some quick checks to see if one hot is done right
    assert (one_hot_smiles[0][0] == one_hot_smiles[1][0]).all() == True
    assert (one_hot_smiles[-1][0] == one_hot_smiles[1][0]).all() == True
    assert (one_hot_smiles[0][2] == one_hot_smiles[1][1]).all() == True
    assert (one_hot_smiles[1][2] == one_hot_smiles[1][1]).all() != True
    assert (one_hot_smiles[0][5] == one_hot_smiles[2][3]).all() == True
    
    return drug_to_hot_smile

def create_paris_to_graphs(
    smile_path=f'{codebase_path}/downloaded_data_small/drugs_to_smiles'):
    '''find dict to map durg names to graphs rep
    
    '''
    
    # graph data
    with open(f'{home_path}{path}edges_features.pkl', 'rb') as f:
        edge_indices, features = pickle.load(f)
    dts = pd.read_csv(smile_path, index_col=0)
    drugs_to_graphs = {}
    for d, ft, edg_inds in zip(dts.index, features, edge_indices):
        edg_inds = torch.tensor(edg_inds).t().contiguous()
        ft = torch.tensor(ft.astype(np.float32))
        drugs_to_graphs[d] = tgd.Data(x=ft, edge_index=edg_inds)
    return drugs_to_graphs


    


class DrpInputData():
    '''Class to load in and preprocess input data for DRP
    
    '''
    def __init__(self, omic_types: list=[], drug_rep: str='marker', 
                 target: str='gdsc1_ic50', max_smile_len: int=188):
        #--- to do: add genomics---- 
        #cheks
        supported_omcis_dtypes = ['phos', 'prot', 'rna']
        supported_drug_dtypes = ['marker', 'smile', 'mol_graph']
        if drug_rep not in supported_drug_dtypes:
            raise Exception(
                f'{drug_rep} not supported drug \
                rep as to be one of the {supported_drug_dtypes} ')
            
        for omic in omic_types:
            if omic not in supported_omcis_dtypes:
                raise Exception(
                    f"{omic} not in supported dtypes,\
                    omic has to be one of {supported_omcis_dtypes}")
        if not omic_types:
            raise Exception("add input omics types e.g. phos, prot, rna")
            
        self.omic_types = omic_types    
        self.all_omics = {}
        if target == 'gdsc1_ic50':
            self.y_df = read_targets() 
        if 'phos' in omic_types:
            self.phos = read_phos_druml()
            self.all_omics['phos'] = self.phos
        if 'prot' in omic_types:
            self.prot = read_prot_map()
            self.all_omics['prot'] = self.prot
        if 'rna' in omic_types:
            self.rna = read_rna_gdsc()
            self.all_omics['rna'] = self.rna 
        if drug_rep == 'marker':
            self.all_drugs = self.y_df.columns 
        if drug_rep == 'smile':
            self.dths = create_drug_to_hot_smile(max_smile_len=max_smile_len)
            self.all_drugs = list(self.dths.keys())
        if drug_rep == 'mol_graph':
            self.dtg = create_paris_to_graphs()
            self.dtg = list(self.dths.keys())
            
            
        #---- To Do: add other drug reps ----
        #need to remove drugs without smiels strings from all_drugs when done 
        self.marker_drugs = one_hot_encode(self.all_drugs)
        
    def remove_disjoint(self):
        '''Remove cell lines that don't overlap between datasets

        '''
       
        omic_dfs = list(self.all_omics.values())
        overlap_cls = [idx for idx in omic_dfs[0].index 
                       if idx in self.y_df.index]
        for df in omic_dfs[1: ]:
            overlap_cls = [idx for idx in df.index if idx in overlap_cls]
            
        if 'phos' in self.omic_types:
            self.phos = self.phos.loc[overlap_cls]
            self.all_omics['phos'] = self.phos
        if 'prot' in self.omic_types:
            self.prot = self.prot.loc[overlap_cls]
            self.all_omics['prot'] = self.prot
        if 'rna' in self.omic_types:
            self.rna = self.rna.loc[overlap_cls]
            self.all_omics['rna'] = self.rna 
         
        
        #check same cls in each omics df
        for df1 in self.all_omics.values():
            for df2 in self.all_omics.values():
                assert (df1.index == df2.index).all()
                
    #should do in diff / class 
    def tt_split(self):
        pass
        #can use the same tt split method for each omics df


#-------older funcs used in previos work ----------           
def read_inputs_phos_ect():
    '''Reads input data for overlap of phos, rna and prot 
    
    returns prot, rna, phospho_ls, ic50_df1, one_hot_cls, one_hot_drugs
    '''
    
    ic50_df1 = read_targets()
    _all_drugs = ic50_df1.columns
    
    #drug one hot encoding
    frame = {}
    for i,d in enumerate(_all_drugs):
        hot_vec = np.zeros(len(_all_drugs))
        hot_vec[i] = 1
        frame[d] = hot_vec
    one_hot_drugs = pd.DataFrame(frame)

    #read in protmics data
    uniprot_ids = pd.read_csv(
        f'{codebase_path}/downloaded_data_small/Proteinomics_large.tsv',
        sep = '\t', skipfooter = 951).columns[2 :] #keep ids of col names

    prot_raw = pd.read_csv(
        f'{codebase_path}/downloaded_data_small/Proteinomics_large.tsv',
        sep = '\t', header=1
    )
    prot_raw.drop(index=0, inplace=True)

    prot_raw.index = prot_raw['symbol']
    prot_raw.drop(columns=['symbol', 'Unnamed: 1'], inplace=True)
    
    #replace missing protomics values
    p_miss = prot_raw.isna().sum().sum() / (len(prot_raw) * len(prot_raw.columns))
    print(f'% of missing prot values {p_miss}')
    #close to 40% of the dataframe is missing valuse
    #replace nan with zero (as done in the paper)
    prot = prot_raw.replace(np.nan, 0)

    #check for duplications and missing value in cols and index
    assert sum(prot.index.duplicated()) == 0
    assert sum(prot.columns.duplicated()) == 0
    assert sum(prot.index.isna()) == 0
    assert sum(prot.columns.isna()) == 0

    #only keep interserciton of cl's and truth values
    ic50_df1, prot = dp_nb.keep_overlapping(ic50_df1, prot)
    print(f'num non overlapping prot and target cls: {len(prot_raw) - len(prot)}')

    #read in rna-seq data
    gdsc_path = '/data/home/wpw035/GDSC'
    rna_raw = pd.read_csv(f'{gdsc_path}/downloaded_data/gdsc_expresstion_dat.csv')
    rna_raw.index = rna_raw['GENE_SYMBOLS']
    rna_raw.drop(columns=['GENE_SYMBOLS','GENE_title'], inplace=True)
    cell_names_raw = pd.read_csv(f'{gdsc_path}/downloaded_data/gdsc_cell_names.csv', skiprows=1, skipfooter=1)
    cell_names_raw.drop(index=0, inplace=True)

    #chagne ids to cell names
    id_to_cl = {}
    for _, row in cell_names_raw.iterrows():
        cell_line = row['Sample Name']
        ident = int(row['COSMIC identifier'])
        id_to_cl[ident] = cell_line

    ids = rna_raw.columns
    ids = [int(iden.split('.')[1]) for iden in ids] 

    #ids that are in rna_raw but don't have an assocated cl name 
    #from cell_names_raw (not sure why we have these)
    missing_ids = []
    for iden in ids:
        if iden not in id_to_cl.keys():
            missing_ids.append(iden)
    missing_ids = [f'DATA.{iden}' for iden in missing_ids]      
    rna_raw.drop(columns=missing_ids, inplace=True)

    cell_lines = []
    for iden in ids:
        try:
            cell_lines.append(id_to_cl[iden])
        except KeyError:
            pass
    rna_raw.columns = cell_lines
    rna_raw = rna_raw.T

    #take out duplicated cell line
    rna_raw = rna_raw[~rna_raw.index.duplicated()] 
    #take out nan cols
    rna_raw = rna_raw[rna_raw.columns.dropna()]
    #take out duplciated cols
    rna_raw = rna_raw.T[~rna_raw.columns.duplicated()].T

    #check for duplications and missing value in cols and index
    assert sum(rna_raw.index.duplicated()) == 0
    assert sum(rna_raw.columns.duplicated()) == 0
    assert sum(rna_raw.index.isna()) == 0
    assert sum(rna_raw.columns.isna()) == 0
    
    #read and format phos data
    phos_path ='/downloaded_data_small/suppData2ppIndexPhospo.csv'
    phos_raw = pd.read_csv(f'{codebase_path}{phos_path}')
    #makes index features 
    phos_raw.index = phos_raw['col.name']
    phos_raw.drop(columns='col.name', inplace=True)
    #formats cell lines in the same way as in target value df. 
    phos_raw.columns = [c.replace('.', '-') for c in phos_raw.columns]
    phos_raw = phos_raw.T
    #only keep overlapping truth values and phos values
    phos_raw, ic50_df1 = dp_nb.keep_overlapping(phos_raw, ic50_df1)
    phos_raw.shape, ic50_df1.shape
    
    #log transfrom (dont think .replace is needed / doing anything)
    phospho_log = np.log2(phos_raw).replace(-np.inf, 0)
    #norm by cell line standard scale 
    scale = StandardScaler()
    phospho_ls = pd.DataFrame(scale.fit_transform(phospho_log.T),
                           columns = phospho_log.index,
                           index = phospho_log.columns).T


    
    
    #only keep intersect of rna and prot cl's and phos
    phospho_ls, rna = dp_nb.keep_overlapping(phospho_ls, rna_raw)
    phospho_ls.shape, rna.shape

    prot, rna = dp_nb.keep_overlapping(prot, rna)
    phospho_ls, rna = dp_nb.keep_overlapping(phospho_ls, rna)
    _all_cls = prot.index
    print(f'num non overlapping cls: {len(rna_raw) - len(rna)}')
    del rna_raw
    
    #one hot input data (instead of omic)
    one_hot_cls = []
    for i, cl in enumerate(_all_cls):
        hot_cl = np.zeros(len(_all_cls))
        hot_cl[i] = 1
        one_hot_cls.append(hot_cl)
    one_hot_cls = pd.DataFrame(one_hot_cls)  
    one_hot_cls.index = _all_cls

    
    return prot, rna, phospho_ls, ic50_df1, one_hot_cls, one_hot_drugs


def read_inputs_rna_prot():
    '''Reads input data for overlap of rna and prot
    
    returns  prot, rna, ic50_df1, one_hot_cls, one_hot_drugs
    '''
    
    
    ic50_df1 = read_targets()
    _all_drugs = ic50_df1.columns
    
    #drug one hot encoding
    frame = {}
    for i,d in enumerate(_all_drugs):
        hot_vec = np.zeros(len(_all_drugs))
        hot_vec[i] = 1
        frame[d] = hot_vec
    one_hot_drugs = pd.DataFrame(frame)

    #read in protmics data
    uniprot_ids = pd.read_csv(
        f'{codebase_path}/downloaded_data_small/Proteinomics_large.tsv',
        sep = '\t', skipfooter = 951).columns[2 :] #keep ids of col names

    prot_raw = pd.read_csv(
        f'{codebase_path}/downloaded_data_small/Proteinomics_large.tsv',
        sep = '\t', header=1
    )
    prot_raw.drop(index=0, inplace=True)

    prot_raw.index = prot_raw['symbol']
    prot_raw.drop(columns=['symbol', 'Unnamed: 1'], inplace=True)
    
    #replace missing protomics values
    p_miss = prot_raw.isna().sum().sum() / (len(prot_raw) * len(prot_raw.columns))
    print(f'% of missing prot values {p_miss}')
    #close to 40% of the dataframe is missing valuse
    #replace nan with zero (as done in the paper)
    prot = prot_raw.replace(np.nan, 0)

    #check for duplications and missing value in cols and index
    assert sum(prot.index.duplicated()) == 0
    assert sum(prot.columns.duplicated()) == 0
    assert sum(prot.index.isna()) == 0
    assert sum(prot.columns.isna()) == 0

    #only keep interserciton of cl's and truth values
    ic50_df1, prot = dp_nb.keep_overlapping(ic50_df1, prot)
    print(f'num non overlapping prot and target cls: {len(prot_raw) - len(prot)}')

    #read in rna-seq data
    gdsc_path = '/data/home/wpw035/GDSC'
    rna_raw = pd.read_csv(f'{gdsc_path}/downloaded_data/gdsc_expresstion_dat.csv')
    rna_raw.index = rna_raw['GENE_SYMBOLS']
    rna_raw.drop(columns=['GENE_SYMBOLS','GENE_title'], inplace=True)
    cell_names_raw = pd.read_csv(f'{gdsc_path}/downloaded_data/gdsc_cell_names.csv', skiprows=1, skipfooter=1)
    cell_names_raw.drop(index=0, inplace=True)

    #chagne ids to cell names
    id_to_cl = {}
    for _, row in cell_names_raw.iterrows():
        cell_line = row['Sample Name']
        ident = int(row['COSMIC identifier'])
        id_to_cl[ident] = cell_line

    ids = rna_raw.columns
    ids = [int(iden.split('.')[1]) for iden in ids] 

    #ids that are in rna_raw but don't have an assocated cl name 
    #from cell_names_raw (not sure why we have these)
    missing_ids = []
    for iden in ids:
        if iden not in id_to_cl.keys():
            missing_ids.append(iden)
    missing_ids = [f'DATA.{iden}' for iden in missing_ids]      
    rna_raw.drop(columns=missing_ids, inplace=True)

    cell_lines = []
    for iden in ids:
        try:
            cell_lines.append(id_to_cl[iden])
        except KeyError:
            pass
    rna_raw.columns = cell_lines
    rna_raw = rna_raw.T

    #take out duplicated cell line
    rna_raw = rna_raw[~rna_raw.index.duplicated()] 
    #take out nan cols
    rna_raw = rna_raw[rna_raw.columns.dropna()]
    #take out duplciated cols
    rna_raw = rna_raw.T[~rna_raw.columns.duplicated()].T

    #check for duplications and missing value in cols and index
    assert sum(rna_raw.index.duplicated()) == 0
    assert sum(rna_raw.columns.duplicated()) == 0
    assert sum(rna_raw.index.isna()) == 0
    assert sum(rna_raw.columns.isna()) == 0
    
    #only keep intersect of rna and prot cl's
    prot, rna = dp_nb.keep_overlapping(prot, rna_raw)
    _all_cls = prot.index
    print(f'num non overlapping rna prot and target cls: {len(rna_raw) - len(rna)}')
    prot.shape, rna.shape
    
    #one hot input data (instead of omic)
    one_hot_cls = []
    for i, cl in enumerate(_all_cls):
        hot_cl = np.zeros(len(_all_cls))
        hot_cl[i] = 1
        one_hot_cls.append(hot_cl)
    one_hot_cls = pd.DataFrame(one_hot_cls)  
    one_hot_cls.index = _all_cls

    
    return prot, rna, ic50_df1, one_hot_cls, one_hot_drugs


def read_smiles(max_smile_len=90, pad_character='!', 
                smile_path=f'{codebase_path}/downloaded_data_small'\
                '/drugs_to_smiles'):
    '''Read and procsses smiles to all be the same length
    
    defalt max_smile_len=90 deterimed from hist of all smile lengths:
    
    Only 1 smlie is longer than 133 in length.
    Thus,  could cut off at this length and pad the rest of the smiles
    But only 11 smiles longer than 90. so will instead to cut off at 90. 
    Menas 43 less padding for majorty of smiles.
    
    '''
    
    drugs_to_smiles_raw = pd.read_csv(smile_path, index_col=0)
    
    #checks 
    smile_len = [len(smile) for smile in drugs_to_smiles_raw['smiles']]
    smile_len = np.array(smile_len)
    num_short = len(smile_len[smile_len > max_smile_len])
    mean_pad = np.mean(smile_len[smile_len < max_smile_len])
    print(f'Num smiles that will be shortened {num_short}')
    print(f'Mean number of padding characters needed {mean_pad}')
    
    #do padding and shortening
    new_smiles = []
    for smile in drugs_to_smiles_raw['smiles']:
    #subset smiles that are too long
        if len(smile) > max_smile_len:
            new_smile = smile[: max_smile_len]
        #add padding to shorter smiles
        elif len(smile) < max_smile_len:
            new_smile = smile + '!' * (max_smile_len - len(smile))
        new_smiles.append(new_smile)

    drugs_to_smiles = pd.DataFrame(
        new_smiles, index=drugs_to_smiles_raw.index, columns=['smiles'])
    #check above is done correctly. 
    for smile in drugs_to_smiles['smiles']:
        assert len(smile) == max_smile_len
    
    return drugs_to_smiles


def one_hot_encode_smiles():
    #To add
    #should be applied sepreately for train and test
    #but fine for cell-blind teste as all drugs in train set.
    #think should add to DRP utils rather than this.
    pass 
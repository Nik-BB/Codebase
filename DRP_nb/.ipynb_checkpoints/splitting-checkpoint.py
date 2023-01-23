''' Functions for train test and validation splitting for DRP

cancer blind split implemented here. Cancer blind Means cell lines 
do not overlap between the train test and val sets. 
'''

import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split(seed, _all_cls, _all_drugs, all_targets, train_size=0.8, 
          split_type='cblind'):
    '''Train test split for cancer or drug blind testing (no val set)
    
    Cancer blind testing means cell lines do not overlap between
    the train test and val sets. 
    Drug blind testing means drugs do not overlap between
    the train test and val sets. 
    '''
    #input cheks
    if type(_all_drugs) == type(_all_cls) != pd.Index:
        print('_all_drugs and _all_cls need to be PD indxes')

    #cancer blind splitting
    if split_type == 'cblind': 
        train_cls, test_cls = train_test_split(_all_cls, train_size=train_size, 
                                               random_state=seed)

        assert len(set(train_cls).intersection(test_cls)) == 0

        frac_train_cl = len(train_cls) / len(_all_cls)
        frac_test_cl = len(test_cls) / len(_all_cls)

        print('Fraction of cls in sets, relative to all cls'\
              'before mising values are removed')          
        print(f'train fraction {frac_train_cl}, test fraction {frac_test_cl}')
        print('------')


        #add in the drugs to each cell line. 
        def create_cl_drug_pair(cells):
            all_pairs = []
            for drug in _all_drugs:
                pairs = cells + '::' + drug
                #only keep pair if there is a truth value for it
                for pair in pairs:
                    if pair in all_targets:
                        all_pairs.append(pair)


            return np.array(all_pairs)

        train_pairs = create_cl_drug_pair(train_cls)
        test_pairs = create_cl_drug_pair(test_cls)
        
    #drug blind splitting    
    if split_type == 'dblind':
        train_ds, test_ds = train_test_split(_all_drugs, train_size=train_size, 
                                           random_state=seed)

        assert len(set(train_ds).intersection(test_ds)) == 0

        frac_train_ds = len(train_ds) / len(_all_drugs)
        frac_test_ds = len(test_ds) / len(_all_drugs)

        print('Fraction of drugs in sets, relative to all drugs'\
              'before mising values are removed')          
        print(f'train fraction {frac_train_ds}, test fraction {frac_test_ds}')
        print('------')

        #add in the drugs to each cell line. 
        def create_cl_drug_pair(drugs):
            all_pairs = []
            for cell in _all_cls:
                pairs = cell + '::' + drugs
                #only keep pair if there is a truth value for it
                for pair in pairs:
                    if pair in all_targets:
                        all_pairs.append(pair)


            return np.array(all_pairs)

        train_pairs = create_cl_drug_pair(train_ds)
        test_pairs = create_cl_drug_pair(test_ds)


    train_pairs = sklearn.utils.shuffle(train_pairs, random_state=seed)
    test_pairs = sklearn.utils.shuffle(test_pairs, random_state=seed)
          

    assert len(set(train_pairs).intersection(test_pairs)) == 0

    num_all_examples = len(_all_cls) * len(_all_drugs)
    frac_train_pairs = len(train_pairs) / num_all_examples
    frac_test_pairs = len(test_pairs) / num_all_examples
          
    print('Fraction of cls in sets, relative to all cl drug pairs, after '\
          'mising values are removed')
    print(f'train fraction {frac_train_pairs}, test fraction '\
          f'{frac_test_pairs}')

    #checking split works as expected.  
 
    if split_type == 'cblind':   
    #create mapping of cls to cl drug pairs for test train and val set. 
        def create_cl_to_pair_dict(cells):
            '''Maps a cell line to all cell line drug pairs with truth values

            '''
            dic = {}
            for cell in cells:
                dic[cell] = []
                for drug in _all_drugs:
                    pair = cell + '::' + drug
                    #filters out pairs without truth values
                    if pair in all_targets:
                        dic[cell].append(pair)
            return dic

        train_cl_to_pair = create_cl_to_pair_dict(train_cls)
        test_cl_to_pair = create_cl_to_pair_dict(test_cls)
        #check right number of cls 
        assert len(train_cl_to_pair) == len(train_cls)
        assert len(test_cl_to_pair) == len(test_cls)

    if split_type == 'dblind':
        def create_cl_to_pair_dict(drugs):
            '''Maps a drug to all cell line drug pairs with truth values

            '''
            dic = {}
            for drug in drugs:
                dic[drug] = []
                for cell in _all_cls:
                    pair = cell + '::' + drug
                    #filters out pairs without truth values
                    if pair in all_targets:
                        dic[drug].append(pair)
            return dic
        
        train_cl_to_pair = create_cl_to_pair_dict(train_ds)
        test_cl_to_pair = create_cl_to_pair_dict(test_ds)
        #check right number of drugs 
        assert len(train_cl_to_pair) == len(train_ds)
        assert len(test_cl_to_pair) == len(test_ds)
    
    # more checks 
    #unpack dict check no overlap and correct number 
    def flatten(l):
        return [item for sublist in l for item in sublist]

    train_flat = flatten(train_cl_to_pair.values())
    test_flat = flatten(test_cl_to_pair.values())

    assert len(train_flat) == len(train_pairs)
    assert len(test_flat) == len(test_pairs)
    assert len(set(train_flat).intersection(test_flat)) == 0
    
    return train_pairs, test_pairs







def split_val(seed, _all_cls, _all_drugs, all_targets, train_size=0.8, 
          val_size=0.5):
    '''Train test val split for cancer blind testing with checks
    
    Cancer blind testing means cell lines do not overlap between
    the train test and val sets. 
    '''
    train_cls, test_cls = train_test_split(_all_cls, train_size=train_size, 
                                           random_state=seed)
    test_cls, val_cls = train_test_split(test_cls, test_size=val_size,
                                         random_state=seed)

    assert len(set(train_cls).intersection(test_cls)) == 0
    assert len(set(val_cls).intersection(test_cls)) == 0
    assert len(set(val_cls).intersection(train_cls)) == 0
    
    frac_train_cl = len(train_cls) / len(_all_cls)
    frac_test_cl = len(test_cls) / len(_all_cls)
    frac_val_cl = len(val_cls) / len(_all_cls)

    print('Fraction of cls in sets, relative to all cls'\
          'before mising values are removed')
    print(f'train fraction {frac_train_cl}, test fraction {frac_test_cl},'\
          f'validaiton fraciton {frac_val_cl}')
    print('------')

    
    #add in the drugs to each cell line. 
    def create_cl_drug_pair(cells):
        all_pairs = []
        for drug in _all_drugs:
            pairs = cells + '::' + drug
            #only keep pair if there is a truth value for it
            for pair in pairs:
                if pair in all_targets:
                    all_pairs.append(pair)


        return np.array(all_pairs)

    train_pairs = create_cl_drug_pair(train_cls)
    test_pairs = create_cl_drug_pair(test_cls)
    val_pairs = create_cl_drug_pair(val_cls)

    train_pairs = sklearn.utils.shuffle(train_pairs, random_state=seed)
    test_pairs = sklearn.utils.shuffle(test_pairs, random_state=seed)
    val_pairs = sklearn.utils.shuffle(val_pairs, random_state=seed)

    assert len(set(train_pairs).intersection(test_pairs)) == 0
    assert len(set(val_pairs).intersection(train_pairs)) == 0
    assert len(set(val_pairs).intersection(train_pairs)) == 0

    num_all_examples = len(_all_cls) * len(_all_drugs)
    frac_train_pairs = len(train_pairs) / num_all_examples
    frac_test_pairs = len(test_pairs) / num_all_examples
    frac_val_pairs = len(val_pairs) / num_all_examples
    print('Fraction of cls in sets, relative to all cl drug pairs, after'\
          'mising values are removed')
    print(f'train fraction {frac_train_pairs}, test fraction'\
          f'{frac_test_pairs}, validaiton fraciton {frac_val_pairs}')

    #checking split works as expected.  
    #create mapping of cls to cl drug pairs for test train and val set. 
    def create_cl_to_pair_dict(cells):
        '''Maps a cell line to all cell line drug pairs with truth values
        
        '''
        dic = {}
        for cell in cells:
            dic[cell] = []
            for drug in _all_drugs:
                pair = cell + '::' + drug
                #filters out pairs without truth values
                if pair in all_targets:
                    dic[cell].append(pair)
        return dic

    train_cl_to_pair = create_cl_to_pair_dict(train_cls)
    test_cl_to_pair = create_cl_to_pair_dict(test_cls)
    val_cl_to_pair = create_cl_to_pair_dict(val_cls)

    #checks 
    #check right number of cls 
    assert len(train_cl_to_pair) == len(train_cls)
    assert len(test_cl_to_pair) == len(test_cls)
    assert len(val_cl_to_pair) == len(val_cls)

    #unpack dict check no overlap and correct number 
    def flatten(l):
        return [item for sublist in l for item in sublist]

    train_flat = flatten(train_cl_to_pair.values())
    test_flat = flatten(test_cl_to_pair.values())
    val_flat = flatten(val_cl_to_pair.values())


    assert len(train_flat) == len(train_pairs)
    assert len(test_flat) == len(test_pairs)
    assert len(val_flat) == len(val_pairs)
    assert len(set(train_flat).intersection(test_flat)) == 0
    assert len(set(train_flat).intersection(val_flat)) == 0
    assert len(set(test_flat).intersection(val_flat)) == 0
    
    return train_pairs, test_pairs, val_pairs
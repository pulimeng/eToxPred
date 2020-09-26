import argparse

import json
import pandas as pd

from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem

import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump

rdBase.DisableLog('rdApp.error')

def myargs():
    parser = argparse.ArgumentParser()                                              
    parser.add_argument('--datafile', required=True, 
                        help='training data filename')
    parser.add_argument('--paramfile', required=True,
                        help='parameters for random search filename')
    parser.add_argument('--outputfile', required=False, default='./best_model',
                        help='output file to save the best model')
    parser.add_argument('--iter', required=True, type=int,
                        help='number of iterations to search for')
    parser.add_argument('--scorer', type=str, default='balanced_accuracy',
                        help='Scoring function to use for the parameter search. Suggest using one of balanced_accuracy, f1, roc_auc')
    args = parser.parse_args()
    return args

def load_data(filename):
    """
    Loading data from .smi file. And generating Morgan's fingerprints and labels 
    for the smiles data.
    Input: filename -> path to the .smi file in the format of SmilesString\tCompoundName\tLabel
    Output: two arrays X -> fingerprints, y -> labels
    """
    df = pd.read_csv(filename, sep='\t', names=['smiles', 'name', 'toxicity'])
    smiles_list = df['smiles'].tolist()
    mols = [Chem.MolFromSmiles(x, sanitize=False) for x in smiles_list]
    labels = df['toxicity'].tolist()
    X = []
    y = []
    cnt = 0
    for mol, label in zip(mols, labels):
        if mol is None:
            print('Error encountered parsing SMILES {}'.format(smiles_list[cnt]))
            continue
        mol = Chem.AddHs(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fp_string = fp.ToBitString()
        tmpX = np.array(list(fp_string),dtype=float)
        X.append(tmpX)
        y.append(label)
        cnt += 1
    X = np.array(X)
    y = np.array(y)
    return X, y
    
def eToxPred_RandParamSearch(opt):
    """
    Randomized parameter search for the Xtree classifier using 5-fold cross-validation
    Input: opt
    Output: best model
    """
    print('...loading and processing data')
    X, y = load_data(opt.datafile)
    et = ExtraTreesClassifier()
    with open(opt.paramfile) as f:
        params = json.load(f)
    print('...using {} as metric and starting the search'.format(opt.scorer))
    clf = RandomizedSearchCV(et, param_distributions=params, n_iter=opt.iter, cv=5,
                             scoring=opt.scorer,
                             random_state=12345, n_jobs=-1, verbose=1)
    search = clf.fit(X, y)
    print('...searching done!')
    print('Best parameter set:')
    print(search.best_params_)
    print('Best {}:'.format(opt.scorer))
    print('{:.4f}'.format(search.best_score_))
    print('...saving the best model to {}'.format(opt.outputfile+'.joblib'))
    best_model = search.best_estimator_
    dump(best_model, opt.outputfile+'.joblib')
    
if __name__ == "__main__":
    opt = myargs()
    eToxPred_RandParamSearch(opt)

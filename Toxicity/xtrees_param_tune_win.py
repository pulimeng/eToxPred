# -*- coding: utf-8 -*-
"""
Grid search for the best parameters (min_leaf,max_depth,min_split) of the ET,
based on the ROC curve and the MCC.
@author: Limeng Pu
"""

from __future__ import division

import pickle
import numpy as np
from itertools import cycle

import argparse

import pybel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.externals import joblib

def myargs():
    parser = argparse.ArgumentParser()                                              
    parser.add_argument('--input', '-i', required = True, help = 'input filename')
    args = parser.parse_args()
    return args

# load the .mat format data
def load_data(dataset):
    fps = []
    toxs = []
    for mol in pybel.readfile('smi',dataset):
        mol.addh()
        temp_smi = mol.write('smi')
        temp_smi.replce('\n','')
        temp_list = temp_smi.split('\t')
        smiles_str = temp_list[0]
        smiles = pybel.readstring('smi',smiles_str)
        y = int(temp_list[2])
        fp_bits = smiles.calcfp().bits
        fp_string = bits2string(fp_bits)
        X = np.array(list(fp_string),dtype=float)
        fps.append(X)
        toxs.append(y)
    my_x = np.asarray(fps)
    my_y = np.asarray(toxs)
    X, Xt, y, yt = train_test_split(my_x, my_y, random_state=233)
    return X, Xt, y, yt

# set the parameters' range for the search
def setgrid():
    mtry = np.linspace(10, 100, 91) # max_features
    mtry = mtry.astype('uint8')
    ml = np.linspace(3, 20, 18) # min_leaf
    ml = ml.astype('uint8')
    ms = np.linspace(5, 20, 16) # min_split
    ms = ms.astype('uint8')
    grid = (mtry,ml,ms)
    return grid

# predict the toxicity using different threshold of the Tox-score
def predict_th(probs, th):
    pred = np.zeros([len(probs), len(th)])
    for i in range(len(probs)):
        for k in range(len(th)):
            temp_th = th[k]
            if probs[i][1] > temp_th:
                pred[i][k] = 1
            else:
                pred[i][k] = 0
    return pred

# compute the MCC
def compute_mcc(true, pred):
    mcc = np.zeros([pred.shape[1], ])
    for k in range(pred.shape[1]):
        temp_pred = pred[:, k]
        temp_mcc = matthews_corrcoef(true, temp_pred)
        mcc[k] = temp_mcc
    return mcc

# convert index to subscript
def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[1])
    rows = rows.astype('uint8')
    cols = (ind.astype('int') % array_shape[1])
    cols = cols.astype('uint8')
    return (rows, cols)

def update(old_mcc, mcc, mtry, ml, ms):
    if mcc > old_mcc:
        best_mcc = mcc
        best_mtry = mtry
        best_ml = ml
        best_ms = ms
        return best_mcc, best_mtry, best_ml, best_ms
    else:
        return old_mcc, mtry, ml, ms
    
def start_tuning(path, grid):
    X, Xt, y, yt = load_data(path)
    cv = StratifiedKFold(n_splits = 5)
    # create evaluation matrices
    evaluation_list = []
    mtry = grid[0]
    ml = grid[1]
    ms = grid[2]
    mccs = np.zeros((len(ml), len(ms)))
    current_mcc = 0
    for item in mtry:
        temp_mtry = item
        print('Current mtry is: '+str(temp_mtry))
        for j in range(len(ml)):
            for k in range(len(ms)):
                temp_ml = ml[j]
                temp_ms = ms[k]
                ext = ExtraTreesClassifier(n_estimators=500, max_depth=None,
                                   min_samples_leaf=temp_ml, max_features=temp_mtry,
                                   min_samples_split=temp_ms)
                mcc_tr = np.zeros([101, ])
                ths = np.linspace(0, 1, 101)
                colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue'])
                i = 0
                for (train, test), color in zip(cv.split(X, y), colors):
                    probas_tr = ext.fit(X[train], y[train]).predict_proba(X[test])
                    pred = predict_th(probas_tr, ths)
                    pred = pred.astype('uint8')
                    true = y[test]
                    mcc_tr += compute_mcc(true,pred)
                    i += 1
                mean_mcc_tr = mcc_tr/6
                idx = list(mean_mcc_tr).index(max(mean_mcc_tr))
                probas_ext = ext.fit(X, y).predict_proba(Xt)
                pred = predict_th(probas_ext, [ths[idx]])
                pred = pred.astype('uint8')
                mcc_ext = matthews_corrcoef(yt, pred)
                mccs[j, k] = mcc_ext
        metrics = mccs
        max_mcc = mccs.max()
        print('The maximum MCC is: '+str(max_mcc))
        max_mcc_idx = mccs.argmax()
        r, c = ind2sub(mccs.shape, max_mcc_idx)
        best_ml = ml[r]
        best_ms = ms[c]
        print('The best ml is: '+str(best_ml))
        print('The best ms is: '+str(best_ms))
        evaluation_list.append(metrics)
        best_mcc, best_mtry, best_ml, best_ms = update(current_mcc, max_mcc, temp_mtry, best_ml, best_ms)
        current_mcc = best_mcc
    return best_mcc, best_mtry, best_ml, best_ms
    
    
def save_best_model(path,mtry,ml,ms):
    with open(path, 'rb') as train_file:
        train_data = pickle.load(train_file)
        X = train_data[0]
        y = train_data[1]
    print('Getting the best model.')
    ext = ExtraTreesClassifier(n_estimators=500, max_depth=None,
                                   min_samples_leaf=ml, max_features=mtry,
                                   min_samples_split=ms)
    ext.fit(X,y)
    joblib.dump(ext,'best_tox_model.pkl')

if __name__ == "__main__":
    args = myargs()
    grid = setgrid()
    best_mcc, best_mtry, best_ml, best_ms = start_tuning(args.input ,grid)
    save_best_model(args.input,best_mtry, best_ml, best_ms)
© 2018 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
API
Training
Shop
Blog
About
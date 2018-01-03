# -*- coding: utf-8 -*-
"""
Grid search for the best parameters (min_leaf,max_depth,min_split) of the ET,
based on the ROC curve and the MCC.

@author: Limeng Pu
"""

from __future__ import division

import scipy.io as sio
import numpy as np
from itertools import cycle

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# load the .mat format data
content = sio.loadmat('toxicity_train.mat')
y = content['y']
y = np.ravel(y)
X = content['X']
content = sio.loadmat('toxicity_test.mat')
yt = content['y']
yt = np.ravel(yt)
Xt = content['X']

# set the parameters' range for the search
mtry = np.linspace(10, 100, 91) # max_features
mtry = mtry.astype('uint8')
ml = np.linspace(3, 20, 18) # min_leaf
ml = ml.astype('uint8')
ms = np.linspace(5, 20, 16) # min_split
ms = ms.astype('uint8')

# create evaluation matrices
evaluation_list = []
tprs = np.zeros((len(ml), len(ms)))
fprs = np.zeros((len(ml), len(ms)))
mccs = np.zeros((len(ml), len(ms)))

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

# 5-fold cross-validation
cv = StratifiedKFold(n_splits = 5)

for i in range(len(mtry)):
    temp_mtry = mtry[i]
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
            cnf = confusion_matrix(yt, pred)
            mcc_ext = matthews_corrcoef(yt, pred)
            TP1 = cnf[1][1]
            FP1 = cnf[0][1]
            FN1 = cnf[1][0]
            TN1 = cnf[0][0]
            fpr1 = float(FP1/(FP1+TN1))
            tpr1 = float(TP1/(TP1+FN1))
            mccs[j, k] = mcc_ext
            tprs[j, k] = tpr1
            fprs[j, k] = fpr1
    metrics = [mccs, tprs, fprs]
    min_mcc = mccs.min()
    print('The minimum MCC is: '+str(min_mcc))
    min_mcc_idx = mccs.argmin()
    r, c = ind2sub(mccs.shape, min_mcc_idx)
    best_ml = ml[r]
    best_ms = ms[c]
    print('The best ml is: '+str(best_ml))
    print('The best ms is: '+str(best_ms))
    evaluation_list.append(metrics)
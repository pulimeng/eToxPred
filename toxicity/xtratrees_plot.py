
# -*- coding: utf-8 -*-
"""
Extratree using scikit learn on balanced dataset 1015/1015 1024 bit

@author: Limeng Pu
"""

from __future__ import division

import scipy.io as sio
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from scipy import interp
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, confusion_matrix, accuracy_score
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib

##### Load training data
content = sio.loadmat('toxicity_over.mat')
y = content['y']
y = np.ravel(y)
X = content['X']

cv = StratifiedKFold(n_splits = 6)

md = 70
ml = 5
ms = 19

ext = ExtraTreesClassifier(n_estimators = 100, max_depth = md,
                           min_samples_leaf = ml, max_features = 'log2',
                           min_samples_split = ms)
                         
print("Max depth %i" % md)
print("Min samples leaf %i" % ml)
print("Min samples split %i" % ms)

scores = cross_val_score(ext, X, y, cv = 10)
print("Average cross-validation score: %f" % np.mean(scores))

##### ROC curve for the training data
fig = plt.figure()
mean_tpr_tr = 0.0
mean_fpr_tr = np.linspace(0, 1, 100)    
colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
lw = 2
i = 0
for (train, test), color in zip(cv.split(X, y), colors):
    probas_tr = ext.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_tr[:, 1])
    mean_tpr_tr += interp(mean_fpr_tr, fpr, tpr)
    mean_tpr_tr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    i += 1
mean_tpr_tr /= cv.get_n_splits(X, y)
mean_tpr_tr[-1] = 1.0
mean_auc_tr = auc(mean_fpr_tr, mean_tpr_tr)
plt.plot(mean_fpr_tr, mean_tpr_tr, color='g', linestyle='--',
         label='Training with 6-fold cv (area = %0.2f)' % mean_auc_tr, lw=lw, linewidth = 2)
         
##### MCC for the training data
def predict_th(probs,th):
    pred = np.zeros([len(probs),len(th)])
    for i in range(len(probs)):
        for k in range(len(th)):
            temp_th = th[k]
            if probs[i][1] > temp_th:
                pred[i][k] = 1
            else:
                pred[i][k] = 0
    return pred
    
def compute_mcc(true,pred):
    mcc = np.zeros([pred.shape[1],])
    for k in range(pred.shape[1]):
        temp_pred = pred[:,k]
        temp_mcc = matthews_corrcoef(true,temp_pred)
        mcc[k] = temp_mcc
    return mcc

mcc_tr = np.zeros([101,])
ths = np.linspace(0,1,101)
colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
i = 0
cv = StratifiedKFold(n_splits = 6)

for (train, test), color in zip(cv.split(X, y), colors):
    probas_tr = ext.fit(X[train], y[train]).predict_proba(X[test])
    pred = predict_th(probas_tr,ths)
    pred = pred.astype('uint8')
    true = y[test]
    mcc_tr += compute_mcc(true,pred)
    i += 1
    
mean_mcc_tr = mcc_tr/6
idx = list(mean_mcc_tr).index(max(mean_mcc_tr))

##### Load test data
content = sio.loadmat('tox4test.mat')
yt = content['y']
yt = np.ravel(yt)
Xt = content['X']

probas_ext = ext.fit(X,y).predict_proba(Xt)
fprs_ext, tprs_ext, thresholds = roc_curve(yt, probas_ext[:, 1])
roc_auc_ext = auc(fprs_ext, tprs_ext)

##### ROC curve for the test data
plt.plot(fprs_ext, tprs_ext, color='r', linestyle='-',
         label='Test (area = %0.2f)' % roc_auc_ext, linewidth = 2)

##### Random Guess
plt.plot([0, 1], [0, 1], linestyle=':', color='k',
         label='Random guess', linewidth = 2)
         
##### MCC for the test data
probas_ext = ext.fit(X,y).predict_proba(Xt)
pred = predict_th(probas_ext,[ths[idx]])
pred = pred.astype('uint8')

acc = accuracy_score(yt, pred)
cnf = confusion_matrix(yt,pred)
mcc_ext = compute_mcc(yt,pred)

TP1 = cnf[1][1]
FP1 = cnf[0][1]
FN1 = cnf[1][0]
TN1 = cnf[0][0]

fpr1 = float(FP1/(FP1+TN1))
tpr1 = float(TP1/(TP1+FN1))

print("True positive: %f, false positive: %f" % (tpr1, fpr1))
print("Test dataset MCC: %f" % mcc_ext)
print("Test dataset accuracy: %f" % acc)

##### ROC figure
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic of the proposed model\n on test datasets')
plt.title('Extra Tree')
plt.legend(loc="lower right",numpoints = 1)
plt.show()

##### MCC figure
fig = plt.figure(dpi = 1000)
ax = fig.gca()
line1, = ax.plot(ths, mean_mcc_tr, '-', linewidth=2,
                 label='Training')
max1 = ax.plot(ths[idx],mcc_ext, 'r*',linewidth = 2, markersize = 10, label = 'Testing')
                 
ax.legend(loc='lower right')
plt.xlabel('Decision probability')
plt.ylabel('MCC')
plt.show()

joblib.dump(ext, 'ETmodel4.pkl') 
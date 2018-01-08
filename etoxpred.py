#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
eToxPred: 
Read SMILES data and estimate the SAscorea and Tox-score

@author: Limeng Pu
"""

from __future__ import division

import sys

# add folder contains the modules to the path
sys.path.insert(0,'SAscore') # input the path to the "SAscore" folder here
sys.path.insert(0,'toxicity') # input the path to the "toxicity" folder here

import argparse

import pybel

import pickle
import theano
import numpy as np
from theano.gof import graph
from sa_dbn import DBN

from sklearn.externals import joblib

def argdet():
    print(len(sys.argv))
    if len(sys.argv) == 1:
        print('Need input file!')
        exit()
    if len(sys.argv) == 3:
        print('No output file will be produced!')
        args = myargs()
        return args
    if len(sys.argv) == 5:
        print('Output file is produced!')
        args = myargs()
        return args
    else:
        print('Cannot recognize the inputs!')
        exit()

def myargs():
    parser = argparse.ArgumentParser()                                              
    parser.add_argument('--input', '-i', required = True, help = 'input filename')
    parser.add_argument('--output', '-o', required = False, help = 'output filename')
    args = parser.parse_args()
    return args

def write2file(filename, predicted_values, proba):
    sa_filename = filename + '_sa.txt'
    with open(sa_filename, 'w') as sa_output_file:
        np.savetxt(sa_filename, predicted_values, fmt = '%1.4f')
    sa_output_file.close()
    tox_filename = filename + '_tox.txt'
    with open(tox_filename, 'w') as tox_output_file:
        np.savetxt(tox_filename, proba, fmt = '%1.4f')
    tox_output_file.close()
        
# load the data from a .sdf file
def bits2string(x):
    fp_string = '0'*1024
    fp_list = list(fp_string)
    for item in x:
        fp_list[item-1] = '1'
    fp_string=''.join(reversed(fp_list)) #reverse bit order to match openbabel output
    return fp_string

def load_data(filename = 'fda_approved_nr.sdf'):
    fps = []
    for mol in pybel.readfile('sdf', 'fda_approved_nr.sdf'):
        mol.addh()
        temp_smiles = mol.data['SMILES_CANONICAL']
        smiles = pybel.readstring('smi',temp_smiles)
        fp_bits = smiles.calcfp().bits
        fp_string = bits2string(fp_bits)
        # convert the data from string to a numpy matrix
        X = np.array(list(fp_string),dtype=float)
        fps.append(X)
    fps = np.asarray(fps)
    return fps

# define the prediction function
def predict(X_test, sa_model = 'sa_trained_model.pkl', tox_model = 'tox_trained_model.pkl'):
    # load the saved model
    with open(sa_model, 'rb') as in_strm:
        regressor = pickle.load(in_strm)
    in_strm.close()
    y_pred = regressor.linearLayer.y_pred
    # find the input to theano graph
    inputs = graph.inputs([y_pred])
    # select only x
    inputs = [item for item in inputs if item.name == 'x']
    # compile a predictor function
    predict_model = theano.function(
        inputs=inputs,
        outputs=y_pred)
    X_test = X_test.astype(np.float32)
    predicted_values = predict_model(X_test)
    # the SAscore here is between 0 and 1 to suit the range of the activation function
    # the following line converts the output to between 1 and 10
    predicted_values = np.asarray(predicted_values*10)
    predicted_values = np.reshape(predicted_values,(len(predicted_values),1))
    xtree = joblib.load(tox_model)
    proba = xtree.predict_proba(X_test)[:,1]
    print('Prediction done!')
    return predicted_values,proba

if __name__ == "__main__":
    args = argdet()
    X = load_data(args.input)
    predicted_values,proba = predict(X,'SA_trained_model_cpu.pkl','Tox_trained_model.pkl') # if cuda is not installed, use the trained_model_cpu
    write2file(args.output, predicted_values, proba)
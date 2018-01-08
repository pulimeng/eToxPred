# -*- coding: utf-8 -*-
"""
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
    if len(sys.argv) == 1:
        print('Need input file!')
        exit()
    if len(sys.argv) == 2:
        print('No output file will be produced!')
        args = myargs()
        return args
    if len(sys.argv) == 3:
        print('Output file is produced!')
        args = myargs()
        return args

def myargs():
    parser = argparse.ArgumentParser()                                              
    parser.add_argument('infile', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    args = parser.parse_args()
    return args

def write2file(filename):
    if filename == '<stdout>':
        return
    else:
        output_string = 'Predicted SAscore is: ' + str(predicted_values)+'\nTox-score is: '+ str(proba) + '\n'
        with open(filename, 'w') as output_file:
            output_file.write(output_string)

# load the data from a .sdf file
def bits2string(x):
    fp_string = '0'*1024
    fp_list = list(fp_string)
    for k in range(len(x)):
        idx = x[k]-1
        fp_list[idx] = '1'
    fp_string = ''.join(fp_list)
    return fp_string

def load_data(filename = 'fda_approved_nr.sdf'):
    #for mol in pybel.readfile('sdf', 'fda_approved_nr.sdf'):
    mols = pybel.readfile('sdf', filename)
    mol = mols.next()
    mol.addh()
    tempid = mol.data['MOLID']
    print(str(tempid))
    temp_smiles = mol.data['SMILES_CANONICAL']
    #temp_smiles = 'OC[C@@H](C(=O)N[C@H](C(=O)N[C@@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N1CCC[C@H]1C(=O)NNC(=O)N)CCCN=C(N)N)CC(C)C)COC(C)(C)C)Cc1ccc(cc1)O)NC(=O)[C@H](Cc1c[nH]c2c1cccc2)NC(=O)[C@@H](NC(=O)[C@@H]1CCC(=O)N1)Cc1[nH]cnc1'
    smiles = pybel.readstring('smi',temp_smiles)
    fp_calc = smiles.calcfp()
    fp_bits = fp_calc.bits
    fp_sdf = mol.data['FINGERPRINT']
    #fp_string = bits2string[fp_bits]
    #print(fp_string == fp_sdf)
    #fp_sdf = '0000010000000011000000000000111000000001000000000111101000001000001000000001000000000001000000001100010110000001000011100001000000000100000001000001001011000010000000000010000000000000000100000000001000000001001000000010000000100000000000000000000000011110000000000110000000001001100000000000000000001011000011101100101000101000001100001000000000011100010100000000001110111000000000000001111000000000100010000001001000000000011110000010100000001001101000010001101000010000110010000000000000001100000000000000110000010000001000000000100000000000000011100000000000000000000000000000010010110000000001000001100001000000000000100000001000000001111000000000000000001101001000000011100000100100100001100010100001110011100000001100000010000011100100110101000000000000000110000000000000100100001111000000000110000000000000000000001000000000010010001001001100000000000000000001000000000000000000010000000100001101000000000000001000000100000000000100111000000000000010110110000000000100010000100000000000100000000000011010011000101000'
    return fp_sdf

# define the prediction function
def predict(X_test, sa_model = 'sa_trained_model.pkl', tox_model = 'tox_trained_model.pkl'):
    # convert the data from string to a numpy matrix
    X = np.array(list(X_test),dtype=float)
    X = np.reshape(X,(1,1024))
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
    X = X.astype(np.float32)
    predicted_values = predict_model(X)
    # the SAscore here is between 0 and 1 to suit the range of the activation function
    # the following line converts the output to between 1 and 10
    predicted_values = np.asscalar(predicted_values*10)
    xtree = joblib.load(tox_model)
    proba = xtree.predict_proba(X)[0][1]
    return predicted_values,proba

if __name__ == "__main__":
    args = argdet()
    X = load_data(args.infile.name)
    predicted_values,proba = predict(X,'SA_trained_model_cpu.pkl','Tox_trained_model.pkl') # if cuda is not installed, use the trained_model_cpu
    print('Predicted SAscore is: ' + str(predicted_values))
    print('Tox-score is: '+str(proba))
    write2file(args.outfile.name)
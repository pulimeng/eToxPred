# eToxPred
eToxPred is a tool to reliably estimate the toxicity and synthetic accessibility of small organic compounds.

This README file is written by Limeng PU. 

If you find this tool is useful to you, please cite this paper:

Limeng Pu, Misagh Naderi, Tairan Liu, Hsiao-Chun Wu, Supratik Mukhopadhyay, and Michal Brylinski. "eToxPred: A Machine Learning-Based Approach to Estimate the Toxicity of Drug Candidates."

# Prerequisites:
1. Python 2.7+ or Python 3.5+
2. numpy 1.8.2 or higher
3. scipy 0.13.3 or higher
4. scikit-learn 0.18.1 or higher
5. Openbabel 2.3.1
6. (Optional) CUDA 8.0 or higher


# Usage:

The software package contains 2 parts:
1. SAscore prediction (in the folder SAscore)
2. Toxicity prediction (in the folder toxicity)

===================================================

SAscore prediction:
1. Install theano package according to http://deeplearning.net/software/theano/
2. Run sa_dbn.py to train the DBN, the trained model is saved in the folder
3. predict.py is used to predict SAscore on a single compound. predict_all.py is used to predict SAscore on an entire dataset.

===================================================

Toxicity prediction
1. The data is in the .mat format, which is a MATLAB format. I made it this way for easy visiualization in the early stage and they are relatively smaller compared to .pkl. You can read it if you have scipy installed. The toxicity_over.mat is the oversampled dataset for training while tox4test.mat is the testing dataset.
2. The model parameters are determined in the xtrees_param_tune.py file.
3. The plots are generated in the xtratrees_plot.py.

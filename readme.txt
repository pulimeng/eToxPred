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
1. The data is in the .mat format, which is a MATLAB format. I made it this way for easy visiualization in the early stage and they are relatively
smaller compared to .pkl. You can read it if you have scipy installed. The toxicity_over.mat is the oversampled dataset for training while tox4test.mat is the testing dataset.
2. The model parameters are determined in the xtratrees_param_tune.py file.
3. The plots are generated in the xtratrees_plot.py.
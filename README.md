# eToxPred
eToxPred is a tool to reliably estimate the toxicity and calculate the synthetic accessibility of small organic compounds. **This is a newer implementation. The libraries used have been updated and the Deep Belief Network (DBN) for the SA score prediction has been replaced by the exact SA score calculation. For older implementation please refer to the folder /stale.**

This README file is written by Limeng PU. 

If you find this tool is useful to you, please cite this paper:

Pu, L., Naderi, M., Liu, T. et al. eToxPred: a machine learning-based approach to estimate the toxicity of drug candidates. BMC Pharmacol Toxicol 20, 2 (2019). https://doi.org/10.1186/s40360-018-0282-6

# Prerequisites:
1. Python 3.7.*
2. Pandas 1.0 or higher
3. scikit-learn 2.3.*
4. rdkit 2020.03.1

# Usage:

The software package contains 2 parts:
1. SAscore calculation
2. Toxicity prediction

To use the trained models for predictinos:
1. Download and extract the package.
2. Run the eToxPred by `python etoxpred_predict.py --datafile tcm600_nr.smi --modelfile etoxpred_best_model.joblib --outputfile results.csv`
  - `--datafile` specifies the input .smi file which stores the SMILES data.
  - `--modelfile` specifies the location of the trained model.
  - `--outputfile` specifies the output file to store the predicted SAscores and Tox-scores. If this term is not provided, the code will save the output to `./results.csv`.
3. The trianed toxicity model is provided as `etoxpred_best_model.tar.gz`. Please untar before use. For those who wonders, the best parameter setup is `n_estimators` 550, `min_samples_split` 16, `min_samples_leaf` 3, and `max_features` 10.

To use the package to train your own models:
1. Prepare the training dataset. The dataset contains three parts: the smiles, the name of the compound, and the label. The label is 0 or 1, where 0 means safe and 1 means toxic. The dataset has to be stored in a .smi file, where each field is separated by a tab, in the format:
 [SmilesString\tID\tLabel].
2. Train the ET for toxicity prediction. The code provided performs a randomized parameter search. It will return the best result (depending on chosen metric), parameters (.json format), and the model (.joblib format). Run `etoxpred_train.py` in by `python etoxpred_train.py --datafile your_training_set.smi --paramfile params.json --outputfile best_model --iter 3 --scorer balanced_accuracy`.
  - `--datafile` specifies the path to your training datset with the aforementioned format.
  - `--paramfile` specifies the parameter file contains the parameters and the range/distribution of them that you want to search for during the training. An example file is provided, namely `param.json`. More parameters can be added according to https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html.
    - `--outputfile` specifies the output file to store the best model. If this term is not provided, the code will save the output to `best_model.joblib`.
    - `--iter` is the number of iterations to run the randomized search.
    - `--scorer` is the metric to evaluate the performance of each run. It defaultly uses `balanced_accuracy`. recommanded metrics include `accuracy`, `balanced_accuracy`, `f1`, and `roc_auc`.

# Datasets:

An example test dataset that can be used for prediction (in the .smi format) is provided in `tcm600_nr.smi`. The ready to use dataset for ET training is provided in `trainig_set.smi`. Much larger dataset for training can be found at `https://osf.io/m4ah5/`. The general format is SmilesString\tID\tToxicity. The results of the testing set TCM6000_NR are also provied in `tcm_results.csv`.

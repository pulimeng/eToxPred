# eToxPred
eToxPred is a tool to reliably estimate the toxicity and synthetic accessibility of small organic compounds. **This is a newer implementation. The libraries used have been updated and the Deep Belief Network (DBN) for the SA score prediction has been replaced by the exact SA score calculation. For older implementation please refer to the folder /stale.**

This README file is written by Limeng PU. 

If you find this tool is useful to you, please cite this paper:

Limeng Pu, Misagh Naderi, Tairan Liu, Hsiao-Chun Wu, Supratik Mukhopadhyay, and Michal Brylinski. "eToxPred: A Machine Learning-Based Approach to Estimate the Toxicity of Drug Candidates."

# Prerequisites:
1. Python 3.7.*
2. scikit-learn 2.3.*
3. rdkit 2020.03.1

# Usage:

The software package contains 2 parts:
1. SAscore calculation (in the folder SAscore)
2. Toxicity prediction (in the folder toxicity)

To use the trained models for predictinos:
1. Download and extract the package. Make sure `etoxpred.py` and the other two folders (SAscore and toxicity) are in the same folder. Otherwise you have to chagne the path in the `etoxpred.py` (line 13 and 14).
2. Run the eToxPred by `python etoxpred.py -i tcm600_nr.smi -o output`
  - the first input argument `-i` specifies the input .smi file which stores the SMILES data.
  - the second input argument `-o` specifies the output file to store the predicted SAscores and Tox-scores. Note that no file extension is needed since the program will produce two files `output_sa.txt` and `output_tox.txt` to store the ID and predicted values respectively.
3. The corresponding trianed models are in SAscore and toxicity folders respectively. The `trained_model_gpu.pkl` can be used when CUDA is installed and properly configured.

To use the package to train your own models:
1. Prepare the training dataset. The dataset contains two parts: the fingerprints and the label. The label can be the binary class labels for toxicity prediction or the SAscores. The dataset has to be stored in a .smi file in the format:
 [SMILES string\tID\tLabel].
2. Train the DBN for SAscore prediction. Run the `sa_dbn.py` in the SAscore folder by `python sa_dbn.py -i your_training_set.smi`
  - The input arguement is the path to your training datset. The data has to be in the format:
  - The data will be randomly split into training, testing, and validation sets (60%/20%/20%).
  - The parameters of the DBN can be changed in `sa_dbn.py` at line 471.
    - `finetune_lr is` the learning rate used in finetune stage. Default is 0.2.
    - `pretrainig_epochs` is the epochs employed in the pretraining stage. Default is 20.
    - `k` is the number of Gibbs steps in CD/PCD. Default is 1.
    - `training_epochs` is the maxical number of iterations ot run the optimizer. Default is 1000
    - `batch_size` is the the size of a minibatch. Default is 50.
  - The best trained model will be saved as `best_sa_model.pkl`, which can be used for prediction later. Note that the model trained with GPU can only be used with GPU prediction.
3. Train the ET for toxicity prediction. Select the best parameters automatically. Run `xtrees_param_tune.py` in the toxicity folder by `python xtrees_param_tube.py -i your_training_set.txt`.
  - The input arguement is the path to your training datset.
  - The input data should contain both toxic and non-toxic instances. Otherwise, the code will produce error since the model predicts everything to be toxic or non-toxic.
  - The parameters to be tuned are:
    - `min_samples_leaf`: The minimum number of samples required to be at a leaf node.
    - `max_features`: The number of features to consider when looking for the best split.
    - `min_samples_split`: The minimum number of samples required to split an internal node.
  - The tuning range can be set in the `setgrid()` function in `xtrees_param_tune.py`.
  - The best set of parameters will be printed and the model will be saved as `best_tox_model.pkl`. Note that this step might take a long time. Progress will be printed in between.

# Datasets:

An example test dataset that can be used for prediction (in the .smi format) is provided in `tcm600_nr.smi`. The ready to used dataset for ET and DBN training can be found at `https://osf.io/m4ah5/`. The data is in text format. The general format is SMILES string\tID\tSAscore/Toxicity. The results of our experiments in terms of SAscores and Tox-scores are also provied in `sa_results.txt` and `tox_results.txt`. Both ID and SAscore/Tox-score is included in the aforementioned files.

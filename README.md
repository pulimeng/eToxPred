# eToxPred
eToxPred is a tool to reliably estimate the toxicity and synthetic accessibility of small organic compounds.

This README file is written by Limeng PU. 

If you find this tool is useful to you, please cite this paper:

Limeng Pu, Misagh Naderi, Tairan Liu, Hsiao-Chun Wu, Supratik Mukhopadhyay, and Michal Brylinski. "eToxPred: A Machine Learning-Based Approach to Estimate the Toxicity of Drug Candidates."

# Prerequisites:
1. Python 2.7+ or Python 3.5+
2. numpy 1.8.2 or higher
3. scipy 0.13.3 or higher
4. scikit-learn 0.18.1 (higher version can produce error due to the model is trained using this version)
5. Openbabel 2.3.1
6. (Optional) CUDA 8.0 or higher


# Usage:

The software package contains 2 parts:
1. SAscore prediction (in the folder SAscore)
2. Toxicity prediction (in the folder toxicity)

To use the trained models for predictinos:
1. Download and extract the package. Make sure etoxpred.py and the other two folders (SAscore and toxicity) are in the same folder. Otherwise you have to chagne the path in the etoxpred.py (line 13 and 14).
2. Run the eToxPred by `python etoxpred.py -i fda_approved_nr.sdf -o output`
  - the first input argument `-i` specifies the input .sdf file which stores the SMILES data.
  - the second input argument `-o` specifies the output file to store the predicted SAscores and Tox-scores. Note that no file extension is needed since the program will produce two files `output_sa.txt` and `output_tox.txt` to store the predicted values respectively.
3. The corresponding trianed models are in SAscore and toxicity folders respectively. The `trained_model_gpu.pkl` can be used when CUDA is installed and properly configured.

To use the package to train your own models:
1. Prepare the training dataset. The dataset contains two parts: the fingerprints and the label. The label can be the binary class labels for toxicity prediction or the SAscores. The dataset has to be stored in a .pkl file, which is a serialized Python structure used to store many objects.
2. Train the DBN for SAscore prediction. Run the sa_dbn.py in the SAscore folder by `python sa_dbn.py -i your_training_set.pkl`
  - The input arguement is the path to your training datset.
  - The data will be shuffled automatically and split into training, testing, and validation sets (60%/20%/20%).
  - The parameters of the DBN can be changed in `sa_dbn.py` at line 471.
    - `finetune_lr is` the learning rate used in finetune stage. Default is 0.2.
    - `pretrainig_epochs` is the epochs employed in the pretraining stage. Default is 20.
    - `k` is the number of Gibbs steps in CD/PCD. Default is 1.
    - `training_epochs` is the maxical number of iterations ot run the optimizer. Default is 1000
    - `batch_size` is the the size of a minibatch. Default is 50.
  - The best trained model will be saved as best_trained_model.pkl, which can be used for prediction later. Note that the model trained with GPU can only be used with GPU prediction.
3. Run the 

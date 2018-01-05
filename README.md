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

To use the package:
1. Download and extract the package. Make sure etoxpred.py and the other two folders (SAscore and toxicity) are in the same folder. Otherwise you have to chagne the path in the etoxpred.py (line 13 and 14).

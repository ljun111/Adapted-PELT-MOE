cfgs: hyperparameters for different public datasets
data: notebooks for preprocessing data and data files for different datasets
Encoder: model code for different experts  
metric: code for evaluation metrics  
models: model output files  
Online_PELT_MOE_separated: code for the multi-expert APM model  
Plot: plotting code 

online_pelt_moe_separated.py: the training code for the APM model  
results_online_pelt_moe_separated.py: the evaluation file for the APM model


To run the code, you first need to download the relevant public dataset files into the specified subfolders within the data folder. Then execute the corresponding notebook code in the data folder to get .npy file, and finally run the online_pelt_moe_separated.py.

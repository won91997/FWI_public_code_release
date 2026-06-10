# -*- coding: utf-8 -*-

from param_config import *                                                      # Get the dataset name
import os

###################################################
####                 PATHS                    #####
###################################################

main_dir        = 'D:'
out_dir         = 'D:'

data_dir        = main_dir + 'data/'                                            # The path of dataset
results_dir     = out_dir + 'results/'                                         # Output path of run results (not model information)
models_dir      = out_dir + 'models/'                                          # The path where the model will be stored at the end of the run

###################################################
####              DYNAMIC PATHS               #####
###################################################

temp_results_dir= results_dir + '{}results/'.format(dataset_name)               # Generate results storage paths for specific dataset
temp_models_dir = models_dir  + '{}Model/'.format(dataset_name)                 # Generate model   storage paths for specific dataset
#data_dir        = data_dir    + '{}/'.format(dataset_name)                      # Generate data    storage paths for specific dataset
data_dir        = main_dir

if os.path.exists(temp_results_dir) and os.path.exists(temp_models_dir):
    results_dir = temp_results_dir
    models_dir  = temp_models_dir
else:
    os.makedirs(temp_results_dir)
    os.makedirs(temp_models_dir)
    results_dir = temp_results_dir
    models_dir  = temp_models_dir


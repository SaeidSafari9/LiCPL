# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 12:03:41 2023

@author: Saeed
"""

import numpy as np
import PL_utils
import os
import pickle
import PL_unwTools
from joblib import Parallel, delayed

with open("config.txt", 'r') as file:
    for line in file:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('#'):
            continue
        if stripped_line.startswith('root_path ='):
            root_path = stripped_line.split('=')[1].strip().rstrip('/\\')
        if stripped_line.startswith('avg_coh ='):
            avg_coh = eval(stripped_line.split('=')[1].strip().rstrip('/\\'))
        if stripped_line.startswith('GoF ='):
            GoF_threshhold = eval(stripped_line.split('=')[1].strip().rstrip('/\\'))
        if stripped_line.startswith('network_config ='):
            network_config = eval(stripped_line.split('=')[1].strip().rstrip('/\\'))
with open(os.path.join(root_path, '00_Data/parameters.pkl'), 'rb') as f:
    nRows, nCols, nSLCs, sorted_dates, available_combos, date_list = pickle.load(f)
geoc_path = os.path.join(root_path, "GEOC")
#%% Find a sample tif for gdal
import os
def find_first_tif(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".tif"):
                return os.path.join(root, file)
    return None 

sampleFile_path = find_first_tif(geoc_path)
#%%
mean_coh = PL_utils.get_avgCoh(root_path) # Get Average Coherence Map
date_indices = {date: idx+1 for idx, date in enumerate(sorted_dates)}
range_indices = {range_: (date_indices[range_.split('_')[0]], date_indices[range_.split('_')[1]]) for range_ in date_list}

def filter_dates(input_dict, threshold):
    filtered_dict = {key: value for key, value in input_dict.items() if abs(value[0] - value[1]) <= threshold}
    return filtered_dict

if network_config == 2:
    range_indices = filter_dates(range_indices, 3)
#%%
LP_path = root_path + "/02_LP/"
methods = os.listdir(LP_path)

cohs_path = os.path.join(root_path, '00_Data/cohs_stacked.npy')
cohs_stacked = np.load(cohs_path)

for method in methods:    
    method_path = os.path.join(LP_path , method)
    method_name = method
    if method_name!="EMI" and method_name!="EVD":
        continue
    
    print(f"Preparing {method} for unwrapping.", flush=True)
    GoodnessOfFit_map = np.load(root_path + f"/Results/{method_name}/GoF.npy")
    GoF_Mask = GoodnessOfFit_map < GoF_threshhold
    mask_file = mean_coh < avg_coh
    mask_file = np.isnan(mean_coh)
    mask_file = np.logical_or(mask_file, GoF_Mask)
    Parallel(n_jobs=10)(delayed(PL_unwTools.Create_UnwrapExportsRedundant)(i,j,root_path,method_name,sorted_dates,mask_file,sampleFile_path,cohs_stacked,date_list) for range_, (i, j) in range_indices.items()) 
    
#%% Clean Output Folder
import shutil
# Remove all folders except GEOC and 03_LicsarExp
for item in os.listdir(root_path):
    item_path = os.path.join(root_path, item)
    if os.path.isdir(item_path) and item not in ['GEOC', '03_LicsarExp','batch_LiCSBAS.sh']:
        shutil.rmtree(item_path)

# Move EMI and EVD folders from 03_LicsarExp to the root directory
for folder in ['EMI', 'EVD']:
    src_folder = os.path.join(root_path, '03_LicsarExp', folder)
    if os.path.exists(src_folder):
        shutil.move(src_folder, root_path)

# Remove 03_LicsarExp folder
shutil.rmtree(os.path.join(root_path, '03_LicsarExp'))

for folder in ['EMI', 'EVD']:
    target_geoc_path = os.path.join(root_path, folder, 'GEOC')
    os.makedirs(target_geoc_path, exist_ok=True)
    for file in os.listdir(geoc_path):
        if file.endswith('.tif'):
            shutil.copy(os.path.join(geoc_path, file), target_geoc_path)
            
batch_file = os.path.join(root_path, 'batch_LiCSBAS.sh')
if os.path.exists(batch_file):
    for folder in ['EMI', 'EVD']:
        shutil.copy(batch_file, os.path.join(root_path, folder))
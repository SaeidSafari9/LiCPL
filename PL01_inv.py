# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 21:48:29 2022

@author: Saeed
"""

import numpy as np
import time
import itertools
import PL_utils
import PL_algo as PL
import os
import pickle
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

with open("config.txt", 'r') as file:
    for line in file:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('#'):
            continue
        if stripped_line.startswith('root_path ='):
            root_path = stripped_line.split('=')[1].strip().rstrip('/\\')
        if stripped_line.startswith('avg_coh ='):
            avg_coh = eval(stripped_line.split('=')[1].strip().rstrip('/\\'))
            
#%%
patches_path = os.path.join(root_path, '01_Patches') 
patches_path = patches_path.replace('\\', '/')           
files = os.listdir(os.path.join(patches_path,"scm"))
nPatches = sum(1 for file in files if file.endswith('.npy'))
with open(os.path.join(root_path, '00_Data/parameters.pkl'), 'rb') as f:
    nRows, nCols, nSLCs, sorted_dates, available_combos, date_list = pickle.load(f)

PL_coh_threshold = 0

with open(os.path.join(root_path, '00_Data/dates_list.pkl'), 'rb') as f:
    date_list = pickle.load(f)
#%%

def getPL(scm):
    
    wasNan = 1
    if not np.array_equal(np.angle(scm), np.zeros((nSLCs,nSLCs))):
       
        LP_EMI_PBF, LP_EVD_PBF, nRemoved_PBF, EMIbadPix, EVDbadPix = PL.getPBFilteredPL_Connected(scm.copy(),PL_coh_threshold)
        return  LP_EMI_PBF, LP_EVD_PBF,nRemoved_PBF, EMIbadPix, EVDbadPix, 0
    else:
        return np.zeros((scm.shape[0])), np.zeros((scm.shape[0])),0, 0, 0, wasNan
    

for row_tile in range(1,nPatches+1):
    scm_patch = np.load(patches_path + "/scm/scm_patch_" + str(row_tile) + ".npy")

    Linked_Phases_EMI = np.zeros((scm_patch.shape[0], scm_patch.shape[1], nSLCs),dtype=np.complex64)
    Linked_Phases_EVD = np.zeros((scm_patch.shape[0], scm_patch.shape[1], nSLCs),dtype=np.complex64)
    Linked_Phases_nRemoved = np.zeros((scm_patch.shape[0], scm_patch.shape[1]),dtype=np.int16)
    Linked_Phases_EMIbadPix = np.zeros((scm_patch.shape[0], scm_patch.shape[1]),dtype=np.int16)
    Linked_Phases_EVDbadPix = np.zeros((scm_patch.shape[0], scm_patch.shape[1]),dtype=np.int16)
    DataAvailabilityMap = np.zeros((scm_patch.shape[0], scm_patch.shape[1]),dtype=np.int16)
    dataSize = scm_patch.shape[0]* scm_patch.shape[1]
    nLoop = 0
    current_percent = 0
    
    start = time.time() 
    results = Parallel(n_jobs=10)(delayed(getPL)(scm_patch[i,j,:,:]) for i in range(scm_patch.shape[0]) for j in range(scm_patch.shape[1]))
    
    for idx, (i, j) in enumerate(((i, j) for i in range(scm_patch.shape[0]) for j in range(scm_patch.shape[1]))):
        
        Linked_Phases_EMI[i, j, :], Linked_Phases_EVD[i, j, :],Linked_Phases_nRemoved[i, j],Linked_Phases_EMIbadPix[i, j],Linked_Phases_EVDbadPix[i, j], DataAvailabilityMap[i, j] = results[idx] 
    
    file_path_name_EVD = patches_path + "/LP/EVD/LP_patch_" + str(row_tile) 
    PL_utils.save_file(file_path_name_EVD, Linked_Phases_EVD)
      
    file_path_name_EMI = patches_path + "/LP/EMI/LP_patch_" + str(row_tile) 
    PL_utils.save_file(file_path_name_EMI, Linked_Phases_EMI)
             
    runtime = time.time() - start
    minutes = int(runtime // 60)
    seconds = int(runtime % 60)

    print(f"Time spent for patch {row_tile}: {minutes} minutes and {seconds} seconds.", flush=True)
#%%   This part will find every kind of patch for every method or map file if there is any
import os
import re
import numpy as np
from collections import defaultdict

def extract_type_and_number(filename):
    match = re.search(r'_(\d+)\.npy', filename)
    return (filename.split('_')[0], int(match.group(1))) if match else (None, float('inf'))

def process_method_folder(input_folder, method_folder, output_base_folder):
    type_files = defaultdict(list)
    method_path = os.path.join(input_folder, method_folder)
    
    for root, _, files in os.walk(method_path):
        for f in files:
            if f.endswith('.npy'):
                patch_type, number = extract_type_and_number(f)
                if patch_type:
                    type_files[patch_type].append((os.path.join(root, f), number))

    for patch_type, files in type_files.items():
        arrays = [np.load(f[0]) for f in sorted(files, key=lambda x: x[1])]
        
        # Ensure all arrays have the same shape except for the concatenation axis (axis 0)
        shapes = [arr.shape for arr in arrays]
        if not all(shape[1:] == shapes[0][1:] for shape in shapes):
            raise ValueError(f"Arrays for type {patch_type} in method {method_folder} have mismatched shapes: {shapes}")
        
        concatenated_array = np.concatenate(arrays, axis=0)
        
        # Determine the save path and save the concatenated file
        save_dir = os.path.join(output_base_folder, method_folder)
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, f'{patch_type}.npy')
        
        np.save(output_path, concatenated_array)
        
        if patch_type == 'LP':
            for i in range(nSLCs):
                
                file_path_name = root_path + "/02_LP/" + method_folder + "/LP_epoch_" + str(i+1) + ".npy"
                np.save(file_path_name, concatenated_array[:,:,i])

# Specify the input and output base folders
input_folder = os.path.join(root_path, "01_Patches/LP/")
output_base_folder = os.path.join(root_path, "02_LP/")
PL_utils.Create_Missing_Folder(output_base_folder)
# Process each method folder
for method_folder in os.listdir(input_folder):
    method_path = os.path.join(input_folder, method_folder)
    if os.path.isdir(method_path):
        process_method_folder(input_folder, method_folder, output_base_folder)

#%% Find where the images were not all nan
ints_path = os.path.join(root_path, '00_Data/ints_stacked.npy')
Obs_diff = np.load(ints_path)
LP_combos = available_combos

lp_patches_path = root_path + "/01_Patches/LP"
methods = os.listdir(root_path + "/02_LP")
for method in methods:
    if method != 'SP' and method.split('0')[0] != 'nRemoved' :
        LP_path = root_path + "/02_LP/" + method + "/"
        
        if os.path.isfile(LP_path + 'LP.npy'):
        
            print('Calculating Goodness of Fit for',method, flush=True)
            temporal_coherence = np.zeros((nRows, nCols))
            
            LP = np.angle(np.load(LP_path + "LP.npy"))
            LP -= np.atleast_3d(LP[:,:,0])
            LP_diff = -(LP.reshape(-1,nSLCs)[:,LP_combos[:,1]] - LP.reshape(-1,nSLCs)[:,LP_combos[:,0]])
            LP_diff = np.angle(np.exp(1j* LP_diff))
            LP_diff = LP_diff.reshape(nRows, nCols,LP_combos.shape[0])
            
            nAvailableTS = np.isnan(Obs_diff)
            nAvailableTS = np.nansum(~nAvailableTS,axis=2)
            temp_coh_patch = np.nansum(np.cos((Obs_diff - LP_diff)), axis=2) / (nAvailableTS)
            temporal_coherence = temp_coh_patch
                
            PL_utils.save_file(root_path + f"/Results/{method}/GoF" , temporal_coherence)
            
        else:
            print('Skipping Goodness of Fit for',method,'.', flush=True)   
        

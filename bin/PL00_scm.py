# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:04:46 2024

@author: Saeed
"""
import numpy as np
import pickle
import os
from datetime import datetime
import PL_utils

current_directory = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(current_directory, '..', 'config.txt')

with open(config_path, 'r') as file:
    for line in file:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('#'):
            continue
        if stripped_line.startswith('root_path ='):
            root_path = stripped_line.split('=')[1].strip().rstrip('/\\')
        if stripped_line.startswith('patches_nRows ='):
            patches_nRows = eval(stripped_line.split('=')[1].strip().rstrip('/\\'))
 
geoc_folder = os.path.join(root_path, "GEOC")

#%%
from osgeo import gdal

def get_sorted_date_pairs(folder):
    date_pairs = []
    for entry in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, entry)):
            dates = entry.split('_')
            date_pairs.append((dates[0], dates[1]))
    return sorted(date_pairs, key=lambda x: (x[0], x[1]))

def get_unique_sorted_dates(date_pairs):
    unique_dates = sorted(set([date for pair in date_pairs for date in pair]))
    return unique_dates

def stack_files(folder, date_list, file_extension):
    stacked_list = []
    for date_pair in date_list:
        folder_path = os.path.join(folder, f"{date_pair[0]}_{date_pair[1]}")
        file_path = os.path.join(folder_path, f"{date_pair[0]}_{date_pair[1]}.{file_extension}")
        
        ds = gdal.Open(file_path)
        if ds is None:
            raise FileNotFoundError(f"File {file_path} not found.")
        
        array = ds.ReadAsArray()
        stacked_list.append(array)
        ds = None  # Close the dataset
    
    return np.stack(stacked_list, axis=-1)

# Main process
date_pairs = get_sorted_date_pairs(geoc_folder)
date_list = [f"{pair[0]}_{pair[1]}" for pair in date_pairs]

# Get the sorted unique dates and calculate nSLCs
sorted_dates = get_unique_sorted_dates(date_pairs)
nSLCs = len(sorted_dates)

# Stack the images for .geo.diff_pha.tif and .geo.cc.tif files
ints_stacked = stack_files(geoc_folder, date_pairs, 'geo.diff_pha.tif')
cohs_stacked = stack_files(geoc_folder, date_pairs, 'geo.cc.tif')

# Get the number of rows and columns
nRows, nCols = cohs_stacked.shape[:2]

# Print the results
print(f"nRows: {nRows}, nCols: {nCols}, nSLCs: {nSLCs}", flush=True)
print(f"number of Interferograms: {ints_stacked.shape[2]}", flush=True)
#%%

def get_combinations(masked_dates):
    # Split the dates to get individual dates
    single_dates = []
    for date_range in masked_dates:
        start_date, end_date = date_range.split('_')
        single_dates.extend([start_date, end_date])
    
    # Get unique and sorted dates
    sorted_dates = sorted(set(single_dates))
    nSLCs = len(sorted_dates)

    # Create a dictionary to map dates to indices
    date_to_index = {date: idx for idx, date in enumerate(sorted_dates)}

    # Find available combinations
    available_combos = []
    for date_range in masked_dates:
        start_date, end_date = date_range.split('_')
        combo = [date_to_index[start_date], date_to_index[end_date]]
        available_combos.append(combo)

    available_combos = np.array(available_combos)

    return sorted_dates, nSLCs, available_combos

# Assuming masked_dates is already defined

sorted_dates, nSLCs, available_combos = get_combinations(date_list)

#%%
ints_path = os.path.join(root_path, '00_Data/ints_stacked.npy')
cohs_path = os.path.join(root_path, '00_Data/cohs_stacked.npy')
PL_utils.save_file(ints_path, ints_stacked)
PL_utils.save_file(cohs_path, cohs_stacked)
with open(os.path.join(root_path, '00_Data/dates_list.pkl'), 'wb') as f:
    pickle.dump(date_list, f)
    
variables = (nRows, nCols, nSLCs, sorted_dates, available_combos, date_list)
# Save variables to a file
with open(os.path.join(root_path, '00_Data/parameters.pkl'), 'wb') as f:
    pickle.dump(variables, f)

#%% Create elements needed for SCM
patches_path = os.path.join(root_path, '01_Patches')
nPatches = int(np.ceil(nRows/patches_nRows))
last_patch_nRows = nRows - (patches_nRows*nPatches - patches_nRows)

    
for patch in range(1,nPatches+1):
    
    if patches_nRows*patch > nRows:
        cohs_scm = np.zeros((last_patch_nRows, nCols ,nSLCs,nSLCs), dtype=np.int16)
        ints_scm = np.zeros((last_patch_nRows, nCols ,nSLCs,nSLCs), dtype=np.float32)
        
        cohs_stacked_patch = cohs_stacked[(patch-1)*patches_nRows:,:,:]
        ints_stacked_patch = ints_stacked[(patch-1)*patches_nRows:,:,:]
        
    else:
        cohs_scm = np.zeros((patches_nRows, nCols ,nSLCs,nSLCs), dtype=np.int16)
        ints_scm = np.zeros((patches_nRows, nCols ,nSLCs,nSLCs), dtype=np.float32)
        
        cohs_stacked_patch = cohs_stacked[(patch-1)*patches_nRows:(patch)*patches_nRows,:,:]
        ints_stacked_patch = ints_stacked[(patch-1)*patches_nRows:(patch)*patches_nRows,:,:]
        
    for i in range(nSLCs):
        cohs_scm[:,:,i,i] = 255
        ints_scm[:,:,i,i] = 0
        for j in range(i+1,nSLCs):
            int_date = (sorted_dates[i]+'_'+sorted_dates[j])
            if int_date in date_list:
                index = date_list.index(int_date)

                ints_scm[:,:,i,j] = ints_stacked_patch[:,:,index]
                ints_scm[:,:,j,i] = -ints_stacked_patch[:,:,index]
                
                
                cohs_scm[:,:,i,j] = cohs_stacked_patch[:,:,index]
                cohs_scm[:,:,j,i] = cohs_stacked_patch[:,:,index]
                
            else:
                continue
            
    
    PL_utils.save_file(os.path.join(patches_path,f'diff/diff_patch_{patch}'), ints_scm)
    PL_utils.save_file(os.path.join(patches_path,f'coh/coh_patch_{patch}'), cohs_scm)
    print('Patch',patch,"Created.", flush=True)

scm_dateDiff = np.zeros((nSLCs,nSLCs), dtype=np.int16)
for i in range(nSLCs):
    for j in range(i+1,nSLCs):
            int_date = (sorted_dates[i]+'_'+sorted_dates[j])
            if int_date in date_list:
                
                index = date_list.index(int_date)
                date_str1, date_str2 = date_list[index].split('_')
                date1 = datetime.strptime(date_str1, '%Y%m%d')
                date2 = datetime.strptime(date_str2, '%Y%m%d')
                diff = abs((date2 - date1).days)
                scm_dateDiff[i,j] = diff
                scm_dateDiff[j,i] = diff
                
            else:
                continue

PL_utils.save_file(os.path.join(root_path, '00_Data/scm_dateDifferences'), scm_dateDiff)

#%% Create scm patches

del ints_stacked,cohs_stacked,cohs_scm,ints_scm
for i in range(1, nPatches + 1):
    coh_file_path = os.path.join(patches_path, "coh", f"coh_patch_{i}.npy")
    diff_file_path = os.path.join(patches_path, "diff", f"diff_patch_{i}.npy")
    
    coh_patch = np.load(coh_file_path)
    diff_patch = np.load(diff_file_path)
    
    coh_patch = np.nan_to_num(coh_patch, nan=0)
    diff_patch = np.nan_to_num(diff_patch, nan=0)
    
    scm_patch = coh_patch * np.exp( 1j * diff_patch)
    PL_utils.save_file(os.path.join(patches_path,f'scm/scm_patch_{i}'), scm_patch)
    os.remove(coh_file_path)
    os.remove(diff_file_path)
    print('SCM Patch',i,"Created.", flush=True)

# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 20:13:30 2024

@author: Saeed
"""

import numpy as np
import os
import PL_utils

def vectorized_median_filter(img):
    # Create shifted versions of the image
    shifts = [
        (0, 0),
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1)
    ]
    
    rows, cols = img.shape
    stack = np.zeros((rows, cols, len(shifts)), dtype=img.dtype)
    
    for k, (i_shift, j_shift) in enumerate(shifts):
        shifted_img = np.roll(np.roll(img, i_shift, axis=0), j_shift, axis=1)
        if i_shift > 0:
            shifted_img[:i_shift, :] = 0
        elif i_shift < 0:
            shifted_img[i_shift:, :] = 0
        if j_shift > 0:
            shifted_img[:, :j_shift] = 0
        elif j_shift < 0:
            shifted_img[:, j_shift:] = 0
        stack[:, :, k] = shifted_img
    
    # Compute the median across the stack
    median_filtered = np.median(stack, axis=2)
    
    return median_filtered

def Create_Unique_Coherence(image1,image2,coh_window_size):
    nRows, nCols = image1.shape
    np.seterr(divide='ignore', invalid='ignore')
    coherency_image = np.zeros_like(image1,dtype="float32")
    buffer = int((coh_window_size-1)/2)
    
    buffered_image1 = np.zeros((image2.shape[0]+coh_window_size-1, image2.shape[1]+coh_window_size-1),dtype="complex64")
    buffered_image1[buffer:-buffer,buffer:-buffer] = image1
    buffered_image2 = np.zeros((image2.shape[0]+coh_window_size-1, image2.shape[1]+coh_window_size-1),dtype="complex64")
    buffered_image2[buffer:-buffer,buffer:-buffer] = image2
    
    neighbourhood_image1 = np.zeros((image1.shape[0],image1.shape[1],coh_window_size**2),dtype="complex64")
    neighbourhood_image2 = np.zeros((image1.shape[0],image1.shape[1],coh_window_size**2),dtype="complex64")

    for i in range(buffer,nRows+1):
        for j in range(buffer,nCols+1):
            image1_vec = buffered_image1[i-buffer:i+buffer+1,j-buffer:j+buffer+1].flatten()
            # image1_vec = np.atleast_2d(image1_vec[~np.isnan(image1_vec)])
            image2_vec = buffered_image2[i-buffer:i+buffer+1,j-buffer:j+buffer+1].flatten()
            # image2_vec = np.atleast_2d(image2_vec[~np.isnan(image2_vec)])
            neighbourhood_image1[i-buffer,j-buffer,:] = image1_vec
            neighbourhood_image2[i-buffer,j-buffer,:] = image2_vec
            
            
    numerator = np.sum((np.multiply(neighbourhood_image1 , np.conjugate(neighbourhood_image2))) , axis=2)
    denominator1 = np.sqrt(np.sum((np.multiply(neighbourhood_image1 , np.conjugate(neighbourhood_image1))), axis=2))
    denominator2 = np.sqrt(np.sum((np.multiply(neighbourhood_image2 , np.conjugate(neighbourhood_image2))), axis=2))
    with np.errstate(over='ignore'):
        coherency_image = np.abs(numerator / (denominator1*denominator2))
    coherency_image = np.nan_to_num(coherency_image, nan=0)
    coherency_image[coherency_image > 1] = 1
    return coherency_image

def Create_UnwrapExports(diff_image,coherency_image,interferogram_date,root_path,method_name,sampleFile_path):
    PL_utils.Create_Missing_Folder(root_path + f'/03_LicsarExp/{method_name}/GEOC/' + interferogram_date)
    
    PL_utils.Create_GDAL_Raster(diff_image, sampleFile_path, root_path + f'/03_LicsarExp/{method_name}/GEOC/' + interferogram_date + '/' + interferogram_date + '.geo.diff_pha.tif')
    PL_utils.Create_GDAL_Raster(coherency_image, sampleFile_path, root_path + f'/03_LicsarExp/{method_name}/GEOC/' + interferogram_date + '/' + interferogram_date + '.geo.cc.tif')

def Create_UnwrapExportsRedundant(i,j,root_path,method_name,sorted_dates,mask_file,sampleFile_path,cohs_stacked,date_list):
    
    method_path = os.path.join(root_path, "02_LP", method_name)
    image1 = np.load(method_path + "/LP_epoch_"+ str(i) +".npy")
    file_path = method_path + "/LP_epoch_"+ str(j) +".npy"
    try:
        image2 = np.load(method_path + "/LP_epoch_"+ str(j) +".npy")
        
        
        int_date = sorted_dates[i-1] + "_" + sorted_dates[j-1]
        
        date_mask = [date == int_date for date in date_list]
        org_coh = (cohs_stacked[:,:,date_mask]).reshape(cohs_stacked.shape[0:2])
        nan_mask = np.isnan(org_coh)
        zero_mask = org_coh == 0
        noData_mask = (nan_mask | zero_mask)
        
        
        # image1 = np.exp(1j*vectorized_median_filter(np.angle(image1)))
        # image2 = np.exp(1j*vectorized_median_filter(np.angle(image2)))
        
        # image1[noData_mask] = 0
        # image2[noData_mask] = 0
        
        
        coh_img = Create_Unique_Coherence(image2,image1,3)*255
        coherency_image = coh_img.copy()
        coherency_image = np.nan_to_num(coherency_image, nan=0)
        coherency_image[mask_file] = 0
        diff_image = -(np.angle(image2) - np.angle(image1))
        diff_image = np.nan_to_num(diff_image, nan=0)
        diff_image = np.angle(np.exp(1j * diff_image))
        diff_image[mask_file] = 0
        # diff_image = vectorized_median_filter(diff_image)
        
        diff_image[noData_mask] = 0
        coherency_image[noData_mask] = 0
        
        Create_UnwrapExports(diff_image,coherency_image,int_date,root_path,method_name,sampleFile_path)
    except FileNotFoundError:
        print(f"File {file_path} does not exist. Skipping.")
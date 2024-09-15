# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 18:02:48 2023

@author: Saeed
"""

import os
# import tkinter as tk
# from tkinter import filedialog
from os.path import dirname
import numpy as np
from osgeo import gdal
gdal.PushErrorHandler('CPLQuietErrorHandler')
def Create_Missing_Folder(path):
    if not os.path.exists( path ):
        os.makedirs( path )

from matplotlib.colors import ListedColormap

def read_cmap_from_txt(root_path):
    filename = root_path + 'cmaps/roma.txt'
    with open(filename, 'r') as file:
        lines = file.readlines()
    cmap_data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:  # Assuming RGB format
            r, g, b = map(float, parts)
            cmap_data.append([r , g , b])
    cmap_data = cmap_data[::-1]# Normalize to [0, 1] range
    custom_cmap = ListedColormap(np.array(cmap_data), name='roma_r')
    return custom_cmap

        

def list_folders(path):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    return folders

def list_files(path):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return files


def save_file(file_path, file):
    Create_Missing_Folder(dirname(file_path))
    np.save(file_path,file)

def get_avgCoh(root_path):
    
    mean_coh_path = os.path.join(root_path, 'Results/Mean Coherency/mean_coh.npy')
    
    if not os.path.exists(mean_coh_path):
        
        print("Average coherency not found, calculating...")
        cohs_path = os.path.join(root_path, '00_Data/cohs_stacked.npy')
        
        all_cohs = np.load(cohs_path).astype(np.float32)
        # all_cohs = np.nan_to_num(all_cohs, nan=0)
        all_cohs[all_cohs == 0] = np.nan
        mean_coh = np.nanmean(all_cohs,axis=2)
        save_file(mean_coh_path, mean_coh)
        del all_cohs
        
    else:
        mean_coh = np.load(mean_coh_path)
        
    
    return np.float32(mean_coh)
    
def Create_GDAL_Raster(input_img, sample_path, target_path):
    
    raster_ds = gdal.Open(sample_path)
    sample_img = raster_ds.GetRasterBand(1).ReadAsArray()
    gt = raster_ds.GetGeoTransform()
    proj = raster_ds.GetProjection()
    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    outds = driver.Create(target_path, xsize=sample_img.shape[1],ysize=sample_img.shape[0], bands = 1, eType = gdal.GDT_Float32)
    outds.SetGeoTransform(gt)
    outds.SetProjection(proj)
    outband1 = outds.GetRasterBand(1)
    outband1.WriteArray(input_img)
    outband1.FlushCache()    
    outband1 = None
    outds = None
    
def find_referencePoint(cum,mask_file):
    nRows, nCols, nImages = cum.shape
    sumsq_cum_wrt_med = np.zeros((nRows, nCols), dtype=np.float32)
    for i in range(cum.shape[2]):
        a = cum[:,:,i]
        a[mask_file] = np.nan
        sumsq_cum_wrt_med = sumsq_cum_wrt_med + (a-np.nanmedian(a))**2
    rms_cum_wrt_med = np.sqrt(sumsq_cum_wrt_med/nImages)
    min_rms = np.nanmin(rms_cum_wrt_med)
    refy1s, refx1s = np.where(rms_cum_wrt_med==min_rms)
    refy1s, refx1s = refy1s[0], refx1s[0]
    return refy1s, refx1s
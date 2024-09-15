# -*- coding: utf-8 -*-
"""
Created on Tue May  2 21:45:48 2023

@author: Saeed
"""

import numpy as np
import os
import PL_utils
from os import listdir
import scipy.sparse as sparse
from scipy.sparse import linalg
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.linalg import inv as sp_inv

def mask_unconnected(matrix, band=5):

    # Step 1: Calculate row_sum for upper triangle
    upper_true_mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    row_sum = np.cumsum(upper_true_mask, axis=1)

    # Step 3: Calculate row_cumsum for the original matrix (upper triangle part)
    upper_triangle_mask = np.triu(matrix != 0, k=1)
    row_cumsum = np.cumsum(upper_triangle_mask, axis=1)

    # Step 4: Calculate col_cumsum for the original matrix (lower triangle part)
    # lower_triangle_mask = np.tril(matrix != 0, k=-1)
    # col_cumsum = np.cumsum(lower_triangle_mask, axis=0).T

    flipped_vertically = np.flipud(upper_triangle_mask)
    flipped_horizontally = np.fliplr(flipped_vertically)
    a = np.cumsum(flipped_horizontally, axis=0)
    col_cumsum = np.fliplr(np.flipud(a))

    flipped_vertically = np.flipud(upper_true_mask)
    flipped_horizontally = np.fliplr(flipped_vertically)
    a = np.cumsum(flipped_horizontally, axis=0)
    col_sum = np.fliplr(np.flipud(a))
    # Step 5: Compare row_cumsum and col_cumsum with row_sum and col_sum
    row_connected = np.triu((row_cumsum == row_sum), k=1)
    col_connected = np.triu((col_cumsum == col_sum), k=1)
    
    
    # connectivity_mask = row_connected
    connectivity_mask = row_connected | col_connected
    band_mask = np.triu((np.abs(np.indices(matrix.shape)[0] - np.indices(matrix.shape)[1]) <= band), k=1)
    
    retain_mask = connectivity_mask | (band_mask & (matrix != 0))

    # Create the final binary matrix
    pruned_upper_triangle = retain_mask.astype(int)
    
    # Step 6: Combine the connectivity masks and create the final binary matrix
    binary_matrix = pruned_upper_triangle + pruned_upper_triangle.T
    np.fill_diagonal(binary_matrix, 1)

    return binary_matrix.astype(bool)

def get_network_mask(C, threshold):
    C_u = np.triu(C)
    # Invert the matrix and replace infs with zero
    with np.errstate(divide='ignore'):
        inverted_C = np.where(C_u != 0, 1.0 / C_u, 0)
    
    # Convert to sparse matrix format
    sparse_matrix = sparse.csr_matrix(inverted_C)
    
    # Create a minimum spanning tree to ensure connectivity
    mst = minimum_spanning_tree(sparse_matrix)
    
    # Create a mask from the MST
    mst_matrix = mst.toarray()
    mask = mst_matrix > 0
    
    # Reflect the mask to make it symmetric
    mask = mask | mask.T
    
    # Apply the mask to retain necessary edges
    new_matrix = np.where(mask, C, 0)
    
    final_mask = C >= threshold
    combined_mask = final_mask | (new_matrix > 0)
    # final_matrix = np.where(combined_mask, C, 0)
    return mask, combined_mask

# Function to check if a matrix is positive definite
def is_positive_definite(matrix):
    try:
        # Try to compute the Cholesky decomposition
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

# Function to iteratively regularize a matrix until it is positive definite
def regularize_matrix(matrix, epsilon=1e-8, max_iter=100, scale_factor=10):
    regularized_matrix = matrix.copy()
    iter_count = 0
    while not is_positive_definite(regularized_matrix):
        regularized_matrix += epsilon * np.eye(matrix.shape[0])
        epsilon *= scale_factor  # Increase epsilon multiplicatively
        iter_count += 1
        if iter_count > max_iter:
            raise ValueError("Unable to regularize the matrix to make it positive definite.")
    return regularized_matrix

# Function to invert a matrix with regularization if needed
def invert_matrix(matrix):
    if is_positive_definite(matrix):
        return sp_inv(matrix)
    else:
        # print("Matrix is not positive definite. Regularizing it...")
        regularized_matrix = regularize_matrix(matrix)
        return sp_inv(regularized_matrix)

def EMI(scm):
    try:
        # Calculating inverse of abolute coherence matrix
        SigCoh_inv = invert_matrix(np.abs(scm))
        
        # Hadamard product of SCM and its absolute coherene value and EVD of their result
        eigenValues, eigenVectors = np.linalg.eigh(np.multiply(SigCoh_inv,scm))
        
        # Getting linked phases by selecting eigen vector of lowest eigen value
        pl_EMI = (eigenVectors[:,0])
        # pl_EMI = np.angle(eigenVectors[:,0])
        
        # Setting first phase as master
        # pl_EMI -= pl_EMI[0]
        return pl_EMI
    
    except np.linalg.LinAlgError as e:
        print("Singular matrix encountered. Returning a zeros array...")
        return np.zeros_like(scm.shape[1])

def EVD(scm):
    eigenValues, eigenVectors = np.linalg.eigh(scm)
    pl_EVD = eigenVectors[:,-1] 
    # pl_EVD -= pl_EVD[0]   
    # pl_EVD = np.angle(pl_EVD)
    return pl_EVD


def sparseEVD(scm):
    retries=10
    n = scm.shape[0]
    initial_vector = np.random.rand(n)
    for i in range(retries):
        try:
            eigenValues, eigenVectors = linalg.eigs(sparse.csr_matrix(scm, dtype=complex), 1,which='LM', v0=initial_vector)
            initial = eigenVectors.copy()
            eigenVectors *= np.abs(eigenVectors[0, 0])/eigenVectors[0, 0]
            return eigenVectors.flatten(), initial,0
        except Exception as e:
            initial_vector = np.random.rand(n)
            
    try:
        eigenVectors
    except NameError:
        return np.zeros_like(scm.shape[1]),np.zeros_like(scm.shape[1]),1
    

def sparseEMI(scm,initial_vector):
    retries=10
    n = scm.shape[0]
    EVD_result = initial_vector.copy()
    try:
        SigCoh_inv = invert_matrix(np.abs(scm))
        # SigCoh_inv = invert_SpareMatrix(np.abs(scm))
        M = np.multiply(SigCoh_inv,scm)
        # initial_vector = linalg.eigs(sparse.csr_matrix(scm, dtype=complex), 1,which='LM')[1]
        for i in range(retries):
            try:
                eigenValues, eigenVectors = linalg.eigs(sparse.csr_matrix(M, dtype=complex), 1,which='SM', v0=initial_vector)
                eigenVectors *= np.abs(eigenVectors[0, 0])/eigenVectors[0, 0]
                return eigenVectors.flatten(),0
                break
            except Exception as e:
                initial_vector = np.random.rand(n)
        try:
            eigenVectors
        except NameError:
            eigenVectors = EVD_result.copy()
            # eigenVectors *= np.abs(eigenVectors[0, 0])/eigenVectors[0, 0]
            return eigenVectors.flatten(),2
    except np.linalg.LinAlgError as e:
        print("Singular matrix encountered. Returning a zeros array...")
        return np.zeros_like(scm.shape[1]),1

def getBandPL(scm,threshhold,bw):
    band_matrix = np.zeros((scm.shape[0],scm.shape[1]), dtype=np.complex128)
    row_indices, col_indices = np.indices(scm.shape)
    band_indices = np.abs(row_indices - col_indices) <= bw
    band_matrix[band_indices] = scm[band_indices]
    PL_EMI, PL_EVD, nRemoved, EMIbadPix, EVDbadPix = getPBFilteredPL(band_matrix,threshhold)
    return PL_EMI, PL_EVD, nRemoved, EMIbadPix, EVDbadPix

def getBandMatrix(matrix,bw):
    band_matrix = np.zeros((matrix.shape[0],matrix.shape[1]), dtype=matrix.dtype)
    row_indices, col_indices = np.indices(matrix.shape)
    band_indices = np.abs(row_indices - col_indices) <= bw
    band_matrix[band_indices] = matrix[band_indices]
    return band_matrix

# Pixel Based Filtered EMI and EVD
def getPBFilteredPL(C, threshold):
    
    
    band_scm = getBandMatrix(C.copy(),3)
    
    mst_mask, scm_filter = get_network_mask(np.abs(band_scm), threshold)
    
    threshold_mask = np.abs(C) >= threshold
    combined_mask = threshold_mask | (mst_mask)
    
    
    np.fill_diagonal(mst_mask, True)


    # Count non-zero cells in the upper triangle after filtering
    # mst_scm = C.copy()
    # mst_scm[~mst_mask] = 0
    # initial = sparseEVD(mst_scm)[0]
    # Apply the filter to the matrix
    C[~combined_mask] = 0
    final_non_zero_count = np.count_nonzero(np.triu(C, 1))
    PL_EVD, initial, EVDbadPix = sparseEVD(C)
    PL_EMI, EMIbadPix = sparseEMI(C,initial)
    
    return PL_EMI, PL_EVD, final_non_zero_count, EMIbadPix, EVDbadPix

def invert_SpareMatrix(matrix):
    if is_positive_definite(matrix):
        matrix = sparse.csc_matrix(matrix)
        # matrix_inv = np.linalg.pinv(matrix.toarray())
        matrix_inv = linalg.inv(matrix)
        matrix_inv_dense = matrix_inv.toarray()
        return matrix_inv_dense
    else:
        matrix = regularize_matrix(matrix)
        matrix = sparse.csc_matrix(matrix)
        matrix_inv = linalg.inv(matrix)
        matrix_inv_dense = matrix_inv.toarray()
        return matrix_inv_dense
    
# Pixel Based Filtered EMI and EVD
def getPBFilteredPL_Connected(C, threshold):
    
    
    band_scm = getBandMatrix(C.copy(),5)
    
    mst_mask, scm_filter = get_network_mask(np.abs(band_scm), threshold)
    
    threshold_mask = np.abs(C) > threshold
    combined_mask = threshold_mask | (mst_mask)
    
    
    np.fill_diagonal(mst_mask, True)

    combined_mask = mask_unconnected(combined_mask)
    # Count non-zero cells in the upper triangle after filtering
    # mst_scm = C.copy()
    # mst_scm[~mst_mask] = 0
    # initial = sparseEVD(mst_scm)[0]
    # Apply the filter to the matrix
    C[~combined_mask] = 0
    final_non_zero_count = np.count_nonzero(np.triu(C, 1))
    PL_EVD, initial, EVDbadPix = sparseEVD(C)
    PL_EMI, EMIbadPix = sparseEMI(C,initial)
    
    return PL_EMI, PL_EVD, final_non_zero_count, EMIbadPix, EVDbadPix
import numpy as np
from typing import Union, Tuple, Dict
import os
from src.hpvs import Bipolar_HPV

def hamming_distance(vector_1:Bipolar_HPV, vector_2:Bipolar_HPV)->float:
    """Calculate normalize Hamming distance between 2 hpv vectors"""
    assert vector_1.shape() == vector_2.shape(), f"Found {vector_1.shape()} and {vector_2.shape()}"
    return vector_1.binding(vector_2).count_element(value=-1)/vector_1.shape()

def get_data_size(dataset_hpv: dict):
    return sum([v.shape[0] for v in dataset_hpv.values()])

def get_sim_matrix_by_hamming(repr_hpvs: dict):
    """
    Arg:
        repr_hpvs: key: int, value: np.ndarray
    """
    
    class_list = list(repr_hpvs.keys())
    num_classes = len(class_list)
    
    simm_matrix = np.ones(shape= (num_classes, num_classes), dtype= np.float16)
    
    for ith, each_class in enumerate(class_list):
        for other_class in range(each_class, num_classes):
            
            current_distance = 1 - hamming_distance(vector_1 = repr_hpvs[each_class], vector_2 = repr_hpvs[other_class])
            # assign to sim matrix
            simm_matrix[each_class, other_class] = current_distance
            simm_matrix[other_class, each_class] = current_distance
            
    return simm_matrix


def get_sparse_sim_matrix_by_hamming(repr_hpvs: dict):
    """
    Arg:
        repr_hpvs: key: int, value: np.ndarray
    """
    
    class_list = list(repr_hpvs.keys())
    num_classes = len(class_list)
    
    simm_matrix = np.ones(shape= (num_classes, num_classes), dtype= np.float16)
    
    for ith, each_class in enumerate(class_list):
        curr_row_distacnes = []
        for other_class in range(each_class, num_classes):
            
            current_distance = 1 -hamming_distance(vector_1 = repr_hpvs[each_class], vector_2= repr_hpvs[other_class])
            
            # assign to sim matrix
            simm_matrix[each_class, other_class] = current_distance
            simm_matrix[other_class, each_class] = current_distance
            
    return simm_matrix

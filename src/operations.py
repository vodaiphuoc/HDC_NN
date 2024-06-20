import numpy as np
from typing import List
from copy import deepcopy
    
def permute(hpv_vector: np.ndarray, times_shift: int) -> np.ndarray:
    # assert hpv_vector.ndim != 2, "each vector has two dims"
    # assert hpv_vector.shape[0] > 0, "first dim not greater than zero"
    return np.roll(hpv_vector, shift = times_shift, axis= 0)

def bunding_of_3(hpv_1: np.ndarray, hpv_2: np.ndarray, hpv_3: np.ndarray, is_bipolar: bool = True):
    return np.where(hpv_1 + hpv_2 + hpv_3 > 0, 1, -1) if is_bipolar else np.where(hpv_1 + hpv_2 + hpv_3 > 0, 1, 0)

def bundling(stacked_hpvs: np.ndarray):
    """
    Bunding from stack of hpvs (type: np.ndarray) with shape (N, D_im)
    """
    counter = np.sum(stacked_hpvs, axis = 0)
    out_hpvs = np.zeros_like(counter)
    for ith, element in enumerate(counter):
        if element > 0:
            out_hpvs[ith] = 1
        elif element < 0:
            out_hpvs[ith] = -1
        else:
            out_hpvs[ith] = np.random.choice(a = [1,-1], size = None, p = [0.5,0.5])
    
    assert len(np.where(out_hpvs == 0)[0]) == 0
    return out_hpvs, counter.astype(np.int32)
    
def binding(vector_1: np.ndarray, vector_2: np.ndarray) -> np.ndarray:
    """Perform XOR element-wise multiplication between 2 vectors only"""
    return np.multiply(vector_1, vector_2).astype(np.int8)

def multi_binding(hpv_vectors_list: List[np.ndarray]) -> np.ndarray:
    """Binding operation of many hyperdimensional vectors"""
    temp_xor_result = deepcopy(hpv_vectors_list[0])
    for hpv in hpv_vectors_list[1:]:
        temp_xor_result = binding(temp_xor_result, hpv)
    return temp_xor_result

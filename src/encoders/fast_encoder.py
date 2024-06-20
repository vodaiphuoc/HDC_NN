"""implement encoder with itertools"""

import itertools
import numpy as np
from copy import deepcopy
from typing import List, Tuple

# operation utilities
def binding(vector_1: np.ndarray, vector_2: np.ndarray) -> np.ndarray:
    """Perform XOR element-wise multiplication between 2 vectors only"""
    return np.multiply(vector_1, vector_2).astype(np.int8)

def multi_binding(hpv_vectors_list: List[np.ndarray]) -> np.ndarray:
    """Binding operation of many hyperdimensional vectors"""
    temp_xor_result = deepcopy(hpv_vectors_list[0])
    for hpv in hpv_vectors_list[1:]:
        temp_xor_result = binding(temp_xor_result, hpv)
    return temp_xor_result

# parameters
N = 4
H = 3
W = 7

pattern = np.array([1 if i<N else 0 for i in range(0, H*W)])
selectors_iter = [itertools.compress(itertools.product(range(H),range(W)),np.roll(pattern, i).tolist()) for i in range(H*W-N+1)]

num_channels = 3


# methods in class
def map2high_dim(tuple_pos: Tuple[int], img: np.ndarray, times_shift: int):
    channel_vector = img[tuple_pos[0], tuple_pos[1]]
    #channel_value = channel_vector[0]
    return np.roll(channel_vector, shift = times_shift)

def process_n_gram(n_gram_iter: iter, img_iter: iter):
    """Binding all element in N-gram together"""
    seq_iter = range(N)
    list_hpvs = list(map(map2high_dim, n_gram_iter, img_iter, seq_iter))
    return multi_binding(list_hpvs)

def encode_an_image():
    image = np.random.randint(low= 0, high= 2, size= (H,W,3))
    print("image: ", image)
    img_iter = [ itertools.repeat(image) for i in range(H*W-N+1)]
    return list(map(process_n_gram, selectors_iter, img_iter))

print(encode_an_image())

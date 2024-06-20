"""Make item memory of color
color_item_memory:
{
    "R": 
    {
        0: 
        {
            "start": int,
            "end:: int,
            "hpv": np.ndarray
        },
        1: 
        {
            "start": int,
            "end:: int,
            "hpv": np.ndarray
        }
        ...    
    },
    "G":
    {
        0: 
        {
            "start": int,
            "end:: int,
            "hpv": np.ndarray
        },
        1: 
        {
            "start": int,
            "end:: int,
            "hpv": np.ndarray
        }
        ...
    },
    "B": 
    ...
}

"""
import numpy as np
from typing import Dict, List, Union
from copy import deepcopy

class Item_memory(object):
    def _flip_for_binary(self,level_hpv: np.ndarray, v_indices: np.ndarray):
        """Replace element in the level_hpv with target indices"""
        out_hpv = level_hpv.copy()
        replace_element = out_hpv[v_indices]
        
        if v_indices.shape[0] == 1:
            replace_element = [0] if replace_element[0] == 1 else [1]
        else:
            # find index of "1" elements and replace by "0"
            replace_element = np.where(replace_element > 0, 0, 1)
        
        # put flipped element into 'v_indices' in the level hpv
        np.put(out_hpv, v_indices, replace_element, mode='raise')
        return out_hpv
        
    def _flip_for_bipolar(self,level_hpv: np.ndarray, v_indices: np.ndarray):
        """Replace element in the level_hpv with target indices"""
        out_hpv = level_hpv.copy()
        replace_element = out_hpv[v_indices]
        if v_indices.shape  == 1:
            replace_element = [-1] if replace_element[0] == 1 else [1]    
        else:
            # find index of "1" elements and replace by "-1"
            replace_element = np.where(replace_element > 0, -1, 1)
        
        # put flipped element into 'v_indices' in the level hpv
        np.put(out_hpv, v_indices, replace_element, mode='raise')
        return out_hpv
    

class RGB_item_memory(Item_memory):
    def __init__(self, 
                 M_levels:int, 
                 D_dim:int, 
                 bipolar: bool = True
                 ) -> None:
        super().__init__()
        self.M_levels = M_levels
        # number of bits to flip
        self.b:int = D_dim//(2*(self.M_levels -1))
        self.D = D_dim
        
        self.bipolar = bipolar
        if self.bipolar:
            self.elements = [-1,1]
            self.main_dtype = np.int8
            self._flip = self._flip_for_bipolar
        else:
            self.elements = [0,1]
            self.main_dtype = np.uint8
            self._flip = self._flip_for_binary
        
        self.generator_hpv = np.random.default_rng(800)
        self.gen_fip = np.random.default_rng(300)
    
    
        self.item_memory = {0: {}, 1:{}, 2:{}}
        self.step_value = 255//self.M_levels
        for i_th, value in enumerate(range(0, self.step_value*self.M_levels, self.step_value)):
            for channel in [0,1,2]:
                self.item_memory[channel][i_th] = {}
                self.item_memory[channel][i_th]["start"] = value
                if i_th == self.M_levels -1:
                    self.item_memory[channel][i_th]["end"] = 255
                else:
                    self.item_memory[channel][i_th]["end"] = value + self.step_value
    
    def _make_rest_level_hpvs(self, level_hpvs: np.ndarray)->np.ndarray:
        """Fill the rest of other level flip bits 
        level_hpvs of current feature expect to have shape of (M,D)
        """
        level_hpvs = deepcopy(level_hpvs)
        v_indices_list = self.gen_fip.choice(self.D, size=(self.M_levels-1,self.b), replace=False, p=None)
        # loop over M levels
        for _each_level in range(1,level_hpvs.shape[0]):
            # print(v_indices_list[_each_level-1,:])
            level_hpvs[_each_level,:] = self._flip(level_hpvs[_each_level-1,:], v_indices_list[_each_level-1,:])
        
        return level_hpvs
    
    def generate(self)->Dict[str,Dict[int, Dict[str,Union[int, np.ndarray]]]]:
        """Generate imtem memory for R,G,B channel"""
        lv_hpvs = np.empty(shape = (3,self.M_levels,self.D), dtype= self.main_dtype)
        
        # first level is random
        lv_hpvs[:,0,:] = self.generator_hpv.choice(self.elements, size=(3,self.D), replace=True, p=[0.5,0.5], axis=0, shuffle=True).astype(self.main_dtype)        
        # loop over each R,G,B axis
        for channel_ith in range(lv_hpvs.shape[0]):
            lv_hpvs[channel_ith,:,:] = self._make_rest_level_hpvs(lv_hpvs[channel_ith,:,:])
    
        # pack to item memory format
        for channel_ids, arr in enumerate(lv_hpvs):
            for level, sub_arr in enumerate(arr):
                self.item_memory[channel_ids][level]["hpv"] = sub_arr    
        
        return self.item_memory
    
class Position_item_memory(object):
    """For positional item memory, M level for each dimensiona equal to H/W"""
    def __init__(self, 
                 H: int, 
                 W:int, 
                 D_dim:int,
                 bipolar: bool = True
                 ) -> None:
        self.D = D_dim
        self.H_standard = H
        self.W_standard = W
        
        self.bipolar = bipolar
        if self.bipolar:
            self.elements = [-1,1]
            self.main_dtype = np.int8
        else:
            self.elements = [0,1]
            self.main_dtype = np.uint8
        
        self.generator_hpv = np.random.default_rng(900)
        self.item_memory = {"H": {}, "W":{}}
    
    def generate(self)->Dict[str,Dict[int, Dict[str,Union[int, np.ndarray]]]]:
        """Generate imtem memory for each pixel in H, W dimension"""
        H_lv_hpvs = self.generator_hpv.choice(a = self.elements, size = (self.H_standard,self.D), replace= True, p = [0.5, 0.5])
        W_lv_hpvs = self.generator_hpv.choice(a = self.elements, size = (self.W_standard,self.D), replace= True, p = [0.5, 0.5])
        
        # pack to item memory format
        # for H dimension
        for each_pixel_position, H_arr in enumerate(H_lv_hpvs):
            self.item_memory["H"][each_pixel_position] = H_arr    
        
        # for W dimension
        for each_pixel_position, W_arr in enumerate(W_lv_hpvs):
            self.item_memory["W"][each_pixel_position] = W_arr
        
        return self.item_memory
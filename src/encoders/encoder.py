"""
Main implement encoder
"""
from src.encoders.item_memory import RGB_item_memory, Position_item_memory
from src.operations import multi_binding
from src.operations import permute
from src.operations import bunding

from copy import deepcopy
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.datasets import Flowers102
from torchvision.datasets import MNIST
from PIL.Image import Image
from tqdm import tqdm
import jdata as jd
#import deeplake
#from deeplake.util.iterable_ordered_dict import IterableOrderedDict
from typing import Tuple
import os
from typing import List, Callable, Literal
import itertools

class Dataset_handling(object):
    """ Implement static methods for handling target dataset"""
    @staticmethod
    def _torch_dataset_handling(element: Tuple[Image,int])->Tuple[np.ndarray,int]:
        return (np.array(element[0]), element[1])
    
    # @staticmethod
    # def _nabirds_handling(element: IterableOrderedDict) -> Tuple[np.ndarray, int]:
    #     return (element["images"].squeeze().numpy(), element["labels"].squeeze().item())
    
    @staticmethod
    def dataset_encode(dataset_name: str):
        """
        Support datasets:
        - Cifar 10
        - Oxford Flowers 102
        - NABirds  (dont use for now)
            https://openaccess.thecvf.com/content_cvpr_2015/papers/Horn_Building_a_Bird_2015_CVPR_paper.pdf

        
        Format of dataset contains hpvs, class index: 0, 1, ... K
        {
            "0" : List[np.ndarray (shape: (1,D_im))]
            "1" : List[np.ndarray]
            ...
        }
        format for serialization
        {
            "0" : np.ndarray shape (N_0, D_dim)
            "1" : np.ndarray shape (N_1, D_dim)
            ...
        }
        with N_0, N_1, ... are the number of data in class 0, 1, ...
        """
        if dataset_name == "cifar10":
            num_classes = 10
            handler = Dataset_Encoder._torch_dataset_handling
            
            train_path = '.dataset/cifar10/train'
            train_set = CIFAR10(root = "src/data/"+ train_path, train = True, download = True)
        
            test_path = '.dataset/cifar10/test'
            test_set = CIFAR10(root = "src/data/"+ test_path, train = False, download = True)
    
        elif dataset_name == "mnist":
            num_classes = 10
            handler = Dataset_Encoder._torch_dataset_handling
            
            train_path = '.dataset/mnist/train'
            train_set = MNIST(root = "src/data/"+ train_path, train= True, download = True)
        
            test_path = '.dataset/mnist/test'
            test_set = MNIST(root = "src/data/"+ test_path, train= False, download = True)
    
        elif dataset_name == "flowers102":
            num_classes = 102
            handler = Dataset_Encoder._torch_dataset_handling
            
            train_path = '.dataset/flowers102/train'
            train_set = Flowers102(root = "src/data/"+ train_path, split= "train", download = True)
        
            test_path = '.dataset/flowers102/test'
            test_set = Flowers102(root = "src/data/"+ test_path, split= "test", download = True)

        
        # (dont use for now)
        # elif dataset_name == "nabirds":
        #     num_classes = 1011
        #     handler = Dataset_Encoder._nabirds_handling
        #     if is_train:
        #         data_path = "src/data/.dataset/nabirds/train"
        #     else:
        #         data_path = "src/data/.dataset/nabirds/test"
            
        #     ds = deeplake.load(path= data_path)
        #     target_set = ds.pytorch(decode_method= {"images":"numpy", "labels": "numpy"})    
            
        else:
            raise "support datasets: cifar10, flowers102, nabirds"
        
        return num_classes, train_set, test_set, handler

class Dataset_Encoder(Dataset_handling):
    """Perform encoder for all the dataset"""
    def __init__(self,
                    dataset_name: str,
                    num_channels: int = 3,
                    H_standard:int = 32,
                    W_standard:int = 32,
                    M_levels: int = 35,
                    D_dim: int = 10000,
                    is_bipolar: bool = True,
                    use_binding_of_purmute: bool = True
                    ) -> None:
        super(Dataset_Encoder).__init__()
        
        RGB = RGB_item_memory(M_levels= M_levels, D_dim= D_dim,bipolar= is_bipolar)
        self.color_item_memory = RGB.generate()
        self.step_level = RGB.step_value
        self.final_key = list(self.color_item_memory[0].keys())[-1]
        
        self.H_standard = H_standard
        self.W_standard = W_standard
        
        self.num_channels = num_channels
        if self.num_channels > 1:
            self._mapping2high_dim: Callable = self._3channels_mapping2high_dim
        else:
            self._mapping2high_dim: Callable = self._channel_mapping2high_dim
        
        self.D_dim = D_dim
        if is_bipolar:
            self.main_type = np.int8
        else:
            self.main_type = np.uint8
        
        # assign encoding method
        if use_binding_of_purmute:
            self.encode_image: Callable = self.encode_image_by_N_grams
        else:
            self.pos_item_memory = Position_item_memory(H=H_standard, W= W_standard, D_dim= D_dim, bipolar= is_bipolar).generate()
            self.encode_image: Callable = self.encode_image_by_key_value
        
        # config dataset
        self.num_classes, self.train_set, self.test_set, self.handler = self.dataset_encode(dataset_name = dataset_name)
    

    def set_dataset(self, is_train: bool):
        if is_train:
            self.target_set = self.train_set
            print("using training set")
        elif not is_train:
            self.target_set = self.test_set
            print("using test set")
        else:
            self.target_set = None
    
    """ method implementation"""
    def _channel_mapping2high_dim(self, channel_vector: np.ndarray):
        """
        Args: input_vector shape (1)
        """
        target_key = [each_key for each_key, dict_vale in self.color_item_memory[0].items() if channel_vector >= dict_vale["start"] and channel_vector < dict_vale["end"] ]
        main_key = target_key[0] if len(target_key) != 0 else self.final_key
        return self.color_item_memory[0][main_key]["hpv"]
    
    def _3channels_mapping2high_dim(self, channel_vector: np.ndarray):
        """
        Args: 
            input_vector: input image
            tuple_pos: yields from iter
        Return:
            a hpv represent for a RGB pixel
        """
        
        # define output
        output_hpv = []
        
        channel_value = channel_vector[0]
        target_key = [each_key for each_key, dict_vale in self.color_item_memory[0].items() if channel_value >= dict_vale["start"] and channel_value < dict_vale["end"] ]
        
        first_key = target_key[0] if len(target_key) != 0 else self.final_key
        output_hpv.append(self.color_item_memory[0][first_key]["hpv"])
        start_point_value = self.color_item_memory[0][first_key]["start"]
        
        for ith in range(1,self.num_channels):
            current_pixel = channel_vector[ith]
            if current_pixel > start_point_value:
                curr_key = (current_pixel - start_point_value)//self.step_level + first_key
            else:
                curr_key = -(start_point_value - current_pixel)//self.step_level + first_key 
            
            if curr_key > self.final_key:
                curr_key = self.final_key
            
            output_hpv.append(self.color_item_memory[ith][curr_key]["hpv"])
        
        return multi_binding(output_hpv)
    
    def _map_with_permute(self, tuple_pos: Tuple[int], input_image: np.ndarray):
        """Mapping function only used by "encode_image_by_N_grams" method """
        flatten_position = tuple_pos[0]*self.W_standard + tuple_pos[1] + 1
        
        channel_vector = input_image[tuple_pos[0],tuple_pos[1]]
        out_hpv = self._mapping2high_dim(channel_vector)
        return permute(hpv_vector= out_hpv,times_shift = flatten_position)

    def encode_image_by_N_grams(self, input_image: np.ndarray):
        """
        Args:
            input_img: np.ndarray shape (H,W,3) for RGB image, (H,W) for grayscale image
        """
        pos_iter = itertools.product(range(self.H_standard), range(self.W_standard))
        img_iter = itertools.repeat(input_image)
        
        final_hpv_for_binding = list(map(self._map_with_permute, pos_iter, img_iter))
        return multi_binding(final_hpv_for_binding)
    
    def _key_value_mapping(self, tuple_pos: Tuple[int], input_image: np.ndarray):
        """Mapping function only used by "encode_image_by_key_value" method """
        H_hpv = self.pos_item_memory["H"][tuple_pos[0]]
        W_hpv = self.pos_item_memory["W"][tuple_pos[1]]
        
        channel_vector = input_image[tuple_pos[0],tuple_pos[1]]
        out_hpv = self._mapping2high_dim(channel_vector)
        return multi_binding([out_hpv, H_hpv, W_hpv])
    
    def encode_image_by_key_value(self, input_image: np.ndarray):
        """
        Args:
            input_img: np.ndarray shape (H,W,3) for RGB image, (H,W) for grayscale image
        """
        pos_iter = itertools.product(range(self.H_standard), range(self.W_standard))
        img_iter = itertools.repeat(input_image)
        
        list_hpvs_for_bundling = list(map(self._key_value_mapping, pos_iter, img_iter))        
        return bunding(np.stack(list_hpvs_for_bundling, axis= 0))
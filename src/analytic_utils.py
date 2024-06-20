import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import List
from src.utils import hamming_distance

def pca_visualize(config:dict, 
                  data: np.ndarray, 
                  labels: np.ndarray, 
                  model_gens: np.ndarray = None, 
                  model_labels: np.ndarray = None
                  ) -> None:
    # native PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    
    
    plt.figure(figsize=(12, 6))

    # Đồ thị cho kết quả từ PCA
    plt.subplot(1, 2, 2)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c = labels, cmap='viridis')
    plt.title('PCA Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    if model_gens is not None and model_labels is not None:
        model_result = pca.fit_transform(model_gens)
        # Đồ thị cho kết quả từ mô hình
        plt.subplot(1, 2, 1)
        plt.scatter(model_result[:,0], model_result[:,1], c = model_labels, cmap='viridis')
        plt.title('Model generates')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig(os.getcwd()+f"/plots/pca_ver_{config['version']}.png")

    return None

def hamming_box_plot(config:dict,
                     class_represent_hpvs: np.ndarray, 
                     test_hpvs: np.ndarray, 
                     test_labels: np.ndarray,
                     generated_hpvs: np.ndarray = None,
                     generated_hpvs_labels: np.ndarray = None,
                     save_dir:str = "plots/hamm_box_plot"
                     )-> None:
    """Create multi box plot to visualize hamming distance from
    each hpv in 'hpvs' set to its class represent hpv 
    the input hpvs can be from test data or generated from model
    Args:
        class_represent_hpvs: (dtype: np.ndarray[np.int8]) (shape: [num_classes, D])
        test_hpvs: (dtype: np.ndarray[np.int8]) (shape: [S, D])
        test_labels: (dtype: np.ndarray[np.int8]) (shape: [S,])
        generated_hpvs: (dtype: np.ndarray[np.int8]) (shape: [S, D])
        generated_hpvs_labels: (dtype: np.ndarray[np.int8]) (shape: [S,])
    """
    assert class_represent_hpvs.dtype == np.int8, "class represent hpv not in int8 dtype"
    assert test_hpvs.dtype == np.int8, "test hpv not in int8 dtype"
    
    if generated_hpvs is not None:
        assert generated_hpvs.dtype == np.int8, "generated hpv not in int8 dtype"
        assert test_hpvs.shape[0] == generated_hpvs.shape[0], f"number of test hpvs and generated hpvs are not the same, \
            found {test_hpvs.shape[0]} and {generated_hpvs.shape[0]}"
    
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(save_dir+"/ver_"+config["version"]):
        save_dir = save_dir+"/ver_"+config["version"]
        os.makedirs(save_dir)
    
    num_classes = class_represent_hpvs.shape[0]
    for current_class_current_plot in range(num_classes):
        
        current_class_test_hpvs = test_hpvs[np.where(test_labels == current_class_current_plot)[0]]
        master_test_distance:List[List[int]] = []
        
        current_class_generated_hpvs = None
        master_gen_distance = None
        if generated_hpvs is not None:
            current_class_generated_hpvs = generated_hpvs[np.where(generated_hpvs_labels == current_class_current_plot)[0]]
            master_gen_distance:List[List[int]] = []
        
        
        for curr_class in range(num_classes):
            current_class_rep_hpv = class_represent_hpvs[curr_class]
            test_hpvs_distance = []
            for each_test_hpv in current_class_test_hpvs:
                distance = hamming_distance(vector_1= each_test_hpv, vector_2= current_class_rep_hpv)
                test_hpvs_distance.append(distance)
            
            master_test_distance.append(test_hpvs_distance)
            
            if generated_hpvs is not None:
                gen_hpvs_distance = []
                for each_gen_hpv in current_class_generated_hpvs:
                    distance = hamming_distance(vector_1= each_gen_hpv, vector_2= current_class_rep_hpv)
                    gen_hpvs_distance.append(distance)
                
                master_gen_distance.append(gen_hpvs_distance)
        
                
        assert len(master_test_distance) == num_classes, f"num classes is not equal {num_classes}"
        #assert len(master_gen_distance) == num_classes, f"num classes in gen is not equal {num_classes}"
        
        # make plot
        fig, ax = plt.subplots(figsize=(9, 4))
        
        # manage positions
        if generated_hpvs is None:
            test_positions = [y+1 for y in range(0,num_classes,1)]
            
            # each box plot created base on each column
            master_test_distance = np.array(master_test_distance).T
            ax.boxplot(master_test_distance, 
                       positions= test_positions,
                       patch_artist= True, 
                       boxprops=dict(facecolor="C0"))
            ax.set_xticks([y+1 for y in range(num_classes)],labels=[f"class {cls}" for cls in range(num_classes)])
            
        else:
            test_positions = [3*y+1 for y in range(0,num_classes)]
            
            # each box plot created base on each column
            master_test_distance = np.array(master_test_distance).T
            test_box = ax.boxplot(master_test_distance, 
                                positions= test_positions, 
                                manage_ticks= False,
                                patch_artist= True,
                                boxprops=dict(facecolor="C0")
                                )
            
            gen_positions = [3*y+2 for y in range(0,num_classes)]
            master_gen_distance = np.array(master_gen_distance).T
            gen_box = ax.boxplot(master_test_distance, 
                                 positions= gen_positions,
                                 patch_artist= True, 
                                 manage_ticks= False,
                                 boxprops=dict(facecolor="C3")
                                 )
            ax.legend([test_box["boxes"][0], gen_box["boxes"][0]], ["before training", "after training"])        
            ax.set_xticks([3*y+1.5 for y in range(num_classes)],labels=[f"class {cls}" for cls in range(num_classes)])
        
        
        ax.yaxis.grid(True)
        ax.set_xlabel('Class representation vectors')
        ax.set_ylabel('Hamming distance')
        ax.set_title(f"Hamming distance distribution of data in class {current_class_current_plot}")
        plt.savefig(os.getcwd()+ "/"+ save_dir + f"/hamming_distance_{current_class_current_plot}.png")
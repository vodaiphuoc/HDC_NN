"""Baseline model with bundling method"""

import jdata as jd
import numpy as np
from src.hpvs import Bipolar_HPV
from src.utils import get_sim_matrix_by_hamming, hamming_distance
from typing import Tuple, Dict

class Baseline_HDC(object):
    def __init__(self,  config: dict) -> None:
        self.config = config
        # deserializa hpv dataset
        self.num_classes = config["num_classes"]
        # construct dictionary that hold hpv of each class
        self.class_represent_hpvs = {class_ith: Bipolar_HPV(D = config["hpv_dim"]) for class_ith in range(self.num_classes)}

        self.class_masks = {class_ith: {"pos": Bipolar_HPV(D = self.config["hpv_dim"], name= f"pos_{class_ith}"), 
                                        "neg": Bipolar_HPV(D = self.config["hpv_dim"], name= f"neg_{class_ith}")}  for class_ith in range(self.num_classes)}
    
    def training_baseline(self, train_dataset_hpv: Dict[str, np.ndarray])-> None:
        print("Baseline training")
        for class_id, stacked_hpvs in train_dataset_hpv.items():
            # print("class id: ", class_id)
            for ith, current_hpv in enumerate(stacked_hpvs):    
                self.class_represent_hpvs[int(class_id)].add(Bipolar_HPV(input_sequence = current_hpv))
                
            self.class_represent_hpvs[int(class_id)].accumulate()

        print("baseline similarity matrix: \n", get_sim_matrix_by_hamming(self.class_represent_hpvs))

    def evaluate_on_test(self, test_dataset_hpv: Dict[str, np.ndarray])-> None:
        print("Evaluate on test set")
        acc = 0
        number_predicts = 0
        for class_id, stacked_hpvs in test_dataset_hpv.items():
            for ith, current_hpv in enumerate(stacked_hpvs):
                number_predicts += 1
                result = [100]*self.num_classes
                # loop over all class repr hpvs
                for class_label, repr_hpv in self.class_represent_hpvs.items():
                    result[class_label] = hamming_distance(vector_1 = repr_hpv, vector_2 = Bipolar_HPV(input_sequence= current_hpv))
                min_distance_class = np.argmin(result)
                if min_distance_class == int(class_id):
                    acc += 1
                else:
                    continue

        print(f"Accuracy: {acc/number_predicts}")

    def mask_from_pos_neg(self, 
                          pos_hpv: Bipolar_HPV,
                          neg_hpv: Bipolar_HPV
                          ) -> Bipolar_HPV:
        list_mask = [-1 if pos_element == -1 and neg_element == 1 else 1 for pos_element, neg_element in zip(pos_hpv.get(), neg_hpv.get())]
        list_mask = np.array(list_mask, dtype= np.int8)
        return Bipolar_HPV(input_sequence= list_mask)

    def retraining_by_mask(self, input_dataset_hpv: Dict[str, np.ndarray])-> None:
        # self.class_represent_hpvs = {class_ith: Bipolar_HPV(D = self.config["hpv_dim"]) for class_ith in range(self.num_classes)}

        for class_id, stacked_hpvs in input_dataset_hpv.items():
            for ith, current_hpv in enumerate(stacked_hpvs):
                current_bipolar_hpv = Bipolar_HPV(input_sequence = current_hpv)

                # do normal hamming classification logic
                result = [100]*self.num_classes
                for class_label, repr_hpv in self.class_represent_hpvs.items():
                    result[class_label] = hamming_distance(vector_1 = repr_hpv, vector_2 = current_bipolar_hpv)
                min_distance_class = np.argmin(result)

                # anchor branch
                if min_distance_class == int(class_id):
                    # self.class_represent_hpvs[int(class_id)].add(current_bipolar_hpv)
                    continue

                # positive and negative branch
                else:
                    xor_result_for_current_class = self.class_represent_hpvs[int(class_id)].binding(input_hpv = current_bipolar_hpv)
                    # no flip for positive class
                    self.class_masks[int(class_id)]["pos"].add(xor_result_for_current_class)

                    # flip in case negative sample of other class
                    # xor_result.flip()
                    for each_class in range(self.num_classes):
                        if each_class != int(class_id):
                            
                            xor_result_for_miss_clf_class = self.class_represent_hpvs[each_class].binding(input_hpv = current_bipolar_hpv)
                            self.class_masks[each_class]["neg"].add(xor_result_for_miss_clf_class)

        # accumulate after adding all data
        for class_label, mask_hpv_dict in self.class_masks.items():
            # self.class_represent_hpvs[class_label].accumulate()
            mask_hpv_dict["pos"].accumulate()
            mask_hpv_dict["neg"].accumulate()

            # find dimension both contain -1 in two hpvs
            mask = self.mask_from_pos_neg(pos_hpv= mask_hpv_dict["pos"], neg_hpv= mask_hpv_dict["neg"])

            print(class_label, mask_hpv_dict["pos"].count_element(value= -1),mask_hpv_dict["neg"].count_element(value= -1))
            print(mask.count_element(-1))
            self.class_represent_hpvs[class_label].flip_by_mask(mask = mask)

        print("Retraining similarity matrix: \n", get_sim_matrix_by_hamming(self.class_represent_hpvs))
import jdata as jd
import numpy as np
import json
from src.models import Baseline_HDC
# from src.hpvs import Bipolar_HPV

if __name__ == "__main__":

    with open("cfg.json", "r") as f:
        config = json.load(f)

    train_dataset_hpv = jd.load(f'save_hpvs/{config["dataset"]}/{config["version"]}/train_dataset_hpvs.json')
    test_dataset_hpv = jd.load(f'save_hpvs/{config["dataset"]}/{config["version"]}/test_dataset_hpvs.json')

    model = Baseline_HDC(config= config)
    model.training_baseline(train_dataset_hpv= train_dataset_hpv)
    model.evaluate_on_test(test_dataset_hpv=test_dataset_hpv)

    model.retraining_by_mask(input_dataset_hpv= train_dataset_hpv)
    model.evaluate_on_test(test_dataset_hpv=test_dataset_hpv)
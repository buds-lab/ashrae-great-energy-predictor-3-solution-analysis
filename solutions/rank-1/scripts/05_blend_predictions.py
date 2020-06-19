import os
import glob
import numpy as np 
import pandas as pd 
from functools import partial
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from ashrae.blenders import load_preds, GeneralizedMeanBlender
from ashrae.utils import OUTPUT_PATH, load_data, rmsle, timer


MODEL_LIST = [
    f"{OUTPUT_PATH}/lgb-split_meter-no_normalization.npy",
    f"{OUTPUT_PATH}/lgb-split_meter-target_normalization.npy",
    f"{OUTPUT_PATH}/lgb-split_primary_use-no_normalization.npy",
    f"{OUTPUT_PATH}/lgb-split_primary_use-target_normalization.npy",
    f"{OUTPUT_PATH}/lgb-split_site-no_normalization.npy",
    f"{OUTPUT_PATH}/lgb-split_site-target_normalization.npy",
    f"{OUTPUT_PATH}/cb-split_meter-no_normalization.npy",
    f"{OUTPUT_PATH}/cb-split_meter-target_normalization.npy",
    f"{OUTPUT_PATH}/cb-split_primary_use-no_normalization.npy",
    f"{OUTPUT_PATH}/cb-split_primary_use-target_normalization.npy",
    f"{OUTPUT_PATH}/cb-split_site-no_normalization.npy",
    f"{OUTPUT_PATH}/cb-split_site-target_normalization.npy",
    f"{OUTPUT_PATH}/mlp-split_meter-no_normalization.npy",
    f"{OUTPUT_PATH}/submission_cleanup.csv",
    f"{OUTPUT_PATH}/submission_kfold.csv",
    f"{OUTPUT_PATH}/submission_meter.csv",
]

if __name__ == "__main__":
    """
    python scripts/05_blend_predictions.py
    """           

    # load test data
    with timer("load test data"):
        test = load_data("test_clean")
        leak = load_data("is_leak")
        target = leak["meter_reading"].values

    # load predictions
    with timer("load predictions"):
        preds_matrix = [np.load(x) for x in MODEL_LIST if ".npy" in x]
        replace_inds = (test.site_id == 0) & (test.meter == 0)

        if len([x for x in MODEL_LIST if ".csv" in x]) > 0:
            preds_matrix += [pd.read_csv(x).meter_reading.values for x in MODEL_LIST if ".csv" in x]

        preds_matrix = np.vstack(preds_matrix).T
        preds_matrix[preds_matrix < 0] = 0

    #  blend predictions
    with timer("blend predictions"):
        gmb = GeneralizedMeanBlender()
        gmb.p = 0.11375872112626925
        gmb.c = 0.99817730007820798
        gmb.weights = [0.01782498, 0.03520153, 0.03286305, 0.00718961,
                       0.01797213, 0.0004982 , 0.14172883, 0.12587602,
                       0.08538773, 0.09482115, 0.09476288, 0.10101228,
                       0.15306998, 0.03358389, 0.00719679, 0.05101097]
        test_preds = 0.99576627605010293*np.expm1(gmb.transform(np.log1p(preds_matrix)))

    # create submission            
    with timer("create submission"):            
        subm = load_data("sample_submission")
        subm["meter_reading"] = test_preds
        subm.loc[subm.meter_reading < 0, "meter_reading"] = 0
        subm.loc[~np.isnan(target), "meter_reading"] = target[~np.isnan(target)]

    # save data
    with timer("save data"):                
        subm.to_csv(f"{OUTPUT_PATH}/final_submission.csv", index=False)

import os
import glob
import numpy as np 
import pandas as pd 
from functools import partial
from sklearn.metrics import mean_squared_error
from ashrae.blenders import load_preds, GeneralizedMeanBlender
from ashrae.utils import OUTPUT_PATH, load_data, rmsle, timer


MODEL_LIST = [
    f"{OUTPUT_PATH}/cb-split_meter-no_normalization.npy",
    f"{OUTPUT_PATH}/cb-split_meter-target_normalization.npy",
    f"{OUTPUT_PATH}/cb-split_primary_use-no_normalization.npy",
    f"{OUTPUT_PATH}/cb-split_primary_use-target_normalization.npy",
    f"{OUTPUT_PATH}/cb-split_site-no_normalization.npy",
    f"{OUTPUT_PATH}/cb-split_site-target_normalization.npy",
    f"{OUTPUT_PATH}/lgb-split_meter-no_normalization.npy",
    f"{OUTPUT_PATH}/lgb-split_meter-target_normalization.npy",
    f"{OUTPUT_PATH}/lgb-split_primary_use-no_normalization.npy",
    f"{OUTPUT_PATH}/lgb-split_primary_use-target_normalization.npy",
    f"{OUTPUT_PATH}/lgb-split_site-no_normalization.npy",
    f"{OUTPUT_PATH}/lgb-split_site-target_normalization.npy",
    f"{OUTPUT_PATH}/mlp-split_meter-no_normalization.npy",
    f"{OUTPUT_PATH}/mlp-split_meter-target_normalization.npy",
    f"{OUTPUT_PATH}/submission_cleanup.csv",
    f"{OUTPUT_PATH}/submission_kfold.csv",
    f"{OUTPUT_PATH}/submission_meter.csv",
]


if __name__ == "__main__":
    """
    python scripts/05_blend_predictions.py
    """           

    # load test and leak
    with timer("load test and leak"):
        test = load_data("test_clean")
        leak = load_data("is_leak")
        target = leak["meter_reading"].values

    # load predictions
    with timer("load predictions"):
        preds_matrix = [np.load(x) for x in MODEL_LIST if ".npy" in x]
        if len([x for x in MODEL_LIST if ".csv" in x]) > 0:
            preds_matrix += [pd.read_csv(x).meter_reading.values for x in MODEL_LIST if ".csv" in x]
        preds_matrix = np.vstack(preds_matrix).T
        preds_matrix[preds_matrix < 0] = 0

    # initialize data
    with timer("initialize data"):    
        X_train = preds_matrix[~np.isnan(target)]
        y_train = target[~np.isnan(target)]

    # correct site 0
    with timer("correct site 0"):        
        correction_indices = (test.site_id[~np.isnan(target)]==0) & (test.meter[~np.isnan(target)]==0)
        X_train[correction_indices] *= 0.2931
        y_train[correction_indices] *= 0.2931

    #  optimize weights
    with timer("optimize weights"):
        test_preds = np.zeros(len(test))
        for m in range(4):
            meter_indices = np.where(test[~np.isnan(target)].meter == m)[0]
            gmb = GeneralizedMeanBlender(p_range=(0,1))
            gmb.fit(np.log1p(X_train[meter_indices]),
                    np.log1p(y_train[meter_indices]),
                    n_trials=100)

            test_indices = np.where(test.meter == m)[0]
            test_preds[test_indices] = np.expm1(gmb.transform(np.log1p(preds_matrix[test_indices])))

    # create submission            
    with timer("create submission"):            
        subm = load_data("sample_submission")
        subm["meter_reading"] = test_preds
        subm.loc[subm.meter_reading < 0, "meter_reading"] = 0
        subm.loc[~np.isnan(target), "meter_reading"] = target[~np.isnan(target)]

    # save data
    with timer("save data"):                
        subm.to_csv(f"{OUTPUT_PATH}/final_submission.csv", index=False)

import os
import glob
import numpy as np 
import pandas as pd 
from functools import partial
from sklearn.metrics import mean_squared_error
from ashrae.blenders import load_preds, GeneralizedMeanBlender
from ashrae.utils import OUTPUT_PATH, load_data, rmsle


if __name__ == "__main__":
    """
    python scripts/05_blend_predictions.py
    """           

    # load test and leak
    test = load_data("test_clean")
    leak = load_data("is_leak")
    target = leak["meter_reading"].values

    # load predictions
    preds_matrix = [np.load(x) for x in glob.glob(f"{OUTPUT_PATH}/*.npy")]
    if len(glob.glob(f"{OUTPUT_PATH}/*.csv")) > 0:
        preds_matrix += [pd.csv(x).meter_reading.values for x in glob.glob(f"{OUTPUT_PATH}/*.csv")]
    preds_matrix = np.vstack(preds_matrix).T
    preds_matrix[preds_matrix < 0] = 0

    # initialize data
    X_train = preds_matrix[~np.isnan(target)]
    y_train = target[~np.isnan(target)]

    # correct site 0
    correction_indices = (test.site_id[~np.isnan(target)]==0) & (test.meter[~np.isnan(target)]==0)
    X_train[correction_indices] *= 0.2931
    y_train[correction_indices] *= 0.2931

    #  optimize weights
    test_preds = np.zeros(len(test))
    for m in range(4):
        meter_indices = np.where(test[~np.isnan(target)].meter == m)[0]
        gmb = GeneralizedMeanBlender(p_range=(-1,1))
        gmb.fit(np.log1p(X_train[meter_indices]),
                np.log1p(y_train[meter_indices]),
                n_trials=100)
        
        test_indices = np.where(test.meter == m)[0]
        test_preds[test_indices] = np.expm1(gmb.transform(np.log1p(preds_matrix[test_indices])))

    # create submission
    subm = load_data("sample_submission")
    subm["meter_reading"] = test_preds
    subm.loc[subm.meter_reading < 0, "meter_reading"] = 0

    subm.loc[~np.isnan(target), "meter_reading"] = target[~np.isnan(target)]

    # save data
    subm.to_csv(f"{OUTPUT_PATH}/final_submission.csv", index=False)

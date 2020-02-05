import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

import numpy as np 
import pandas as pd
from ashrae.encoders import FastLabelEncoder
from ashrae.utils import MODEL_PATH, DATA_PATH, load_data, timer

FEATURES = [
    # building meta features
    "square_feet", "year_built", "floor_count",
    
    # cat cols
    "building_id", "site_id", "primary_use", 
    "hour", "weekday", "weekday_hour",
    "building_weekday_hour", "building_weekday",
    "building_hour", 
    
    # raw weather features
    "air_temperature", "cloud_coverage", "dew_temperature",
    "precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed",
    
    # derivative weather features
    "air_temperature_mean_lag7", "air_temperature_max_lag7",
    "air_temperature_min_lag7", "air_temperature_std_lag7",
    "air_temperature_mean_lag73", "air_temperature_max_lag73",
    "air_temperature_min_lag73", "air_temperature_std_lag73",
    
    # time features
    "hour_x", "hour_y", "weekday_x", "weekday_y", "is_holiday",
    
    # target encoding features
    "gte_meter_building_id_hour", "gte_meter_building_id_weekday",
]

CAT_COLS = [
    "building_id", "site_id", "primary_use", 
    "hour", "weekday", "weekday_hour",
    "building_weekday_hour", "building_weekday",
    "building_hour", 
]

NUM_COLS = [x for x in FEATURES if x not in CAT_COLS]


if __name__ == "__main__":
    """
    python scripts/02_preprocess_nn_data.py
    """           

    # meter split target normalization
    with timer("Loading data"):
        train, test = load_data("clean")
        
    with timer("Normalize Target"):
        target_encode_cols = [x for x in train.columns if "gte" in x]
        train[target_encode_cols] = train[target_encode_cols]/np.log1p(train[["square_feet"]].values)
        test[target_encode_cols] = test[target_encode_cols]/np.log1p(test[["square_feet"]].values)
        train["target"] = np.log1p(train["meter_reading"])/np.log1p(train["square_feet"])  
        
    with timer("Standardize Numeric Features"):    
        for m in range(4):
            train_indices = train.meter == m
            test_indices = test.meter == m   
            
            X = np.concatenate([
                train.loc[train_indices, NUM_COLS].values,
                test.loc[test_indices, NUM_COLS].values
            ])
            mu = X.mean(0)
            sig = X.std(0)
            
            train.loc[train_indices, NUM_COLS] = (train.loc[train_indices, NUM_COLS] - mu)/sig
            test.loc[test_indices, NUM_COLS] = (test.loc[test_indices, NUM_COLS] - mu)/sig    
                
    with timer("Encode Categorical Features"):    
        for m in range(4):
            train_indices = train.meter == m
            test_indices = test.meter == m    
            for col in CAT_COLS:

                x = np.concatenate([train.loc[train_indices, col], test.loc[test_indices, col]])
                encoder = FastLabelEncoder()
                encoder.fit(x)

                train.loc[train_indices, col] = encoder.transform(train.loc[train_indices, col])
                test.loc[test_indices, col] = encoder.transform(test.loc[test_indices, col])

    with timer("Save Data"):
        train.to_pickle(f"{DATA_PATH}/preprocessed/train_nn_target_normalized_meter.pkl")
        test.to_pickle(f"{DATA_PATH}/preprocessed/test_nn_target_normalized_meter.pkl")    


    # meter split no normalization
    with timer("Loading data"):
        train, test = load_data("clean")
        train["target"] = np.log1p(train["meter_reading"])

    with timer("Standardize Numeric Features"):    
        for m in range(4):
            train_indices = train.meter == m
            test_indices = test.meter == m   
            
            X = np.concatenate([
                train.loc[train_indices, NUM_COLS].values,
                test.loc[test_indices, NUM_COLS].values
            ])
            mu = X.mean(0)
            sig = X.std(0)
            
            train.loc[train_indices, NUM_COLS] = (train.loc[train_indices, NUM_COLS] - mu)/sig
            test.loc[test_indices, NUM_COLS] = (test.loc[test_indices, NUM_COLS] - mu)/sig

    with timer("Encode Categorical Features"):    
        for m in range(4):
            train_indices = train.meter == m
            test_indices = test.meter == m    
            for col in CAT_COLS:

                x = np.concatenate([train.loc[train_indices, col], test.loc[test_indices, col]])
                encoder = FastLabelEncoder()
                encoder.fit(x)

                train.loc[train_indices, col] = encoder.transform(train.loc[train_indices, col])
                test.loc[test_indices, col] = encoder.transform(test.loc[test_indices, col])

    with timer("Save Data"):
        train.to_pickle(f"{DATA_PATH}/preprocessed/train_nn_meter.pkl")
        test.to_pickle(f"{DATA_PATH}/preprocessed/test_nn_meter.pkl")        


    # meter site target normalization
    with timer("Loading data"):
        train, test = load_data("clean")
        train["target"] = np.log1p(train["meter_reading"])

    with timer("Normalize Target"):
        target_encode_cols = [x for x in train.columns if "gte" in x]
        train[target_encode_cols] = train[target_encode_cols]/np.log1p(train[["square_feet"]].values)
        test[target_encode_cols] = test[target_encode_cols]/np.log1p(test[["square_feet"]].values)
        train["target"] = np.log1p(train["meter_reading"])/np.log1p(train["square_feet"])  

    with timer("Standardize Numeric Features"):    
        for s in range(16):
            train_indices = train.site_id == s
            test_indices = test.site_id == s 
            
            X = np.concatenate([
                train.loc[train_indices, NUM_COLS].values,
                test.loc[test_indices, NUM_COLS].values
            ])
            mu = X.mean(0)
            sig = X.std(0)
            
            train.loc[train_indices, NUM_COLS] = (train.loc[train_indices, NUM_COLS] - mu)/sig
            test.loc[test_indices, NUM_COLS] = (test.loc[test_indices, NUM_COLS] - mu)/sig  

    with timer("Encode Categorical Features"):    
        for s in range(16):
            train_indices = train.site_id == s
            test_indices = test.site_id == s
            for col in CAT_COLS:

                x = np.concatenate([train.loc[train_indices, col], test.loc[test_indices, col]])
                encoder = FastLabelEncoder()
                encoder.fit(x)

                train.loc[train_indices, col] = encoder.transform(train.loc[train_indices, col])
                test.loc[test_indices, col] = encoder.transform(test.loc[test_indices, col])

    with timer("Save Data"):
        train.to_pickle(f"{DATA_PATH}/preprocessed/train_nn_target_normalized_site.pkl")
        test.to_pickle(f"{DATA_PATH}/preprocessed/test_nn_target_normalized_site.pkl")        


    # meter site no normalization
    with timer("Loading data"):
        train, test = load_data("clean")

    with timer("Standardize Numeric Features"):    
        for s in range(16):
            train_indices = train.site_id == s
            test_indices = test.site_id == s 
            
            X = np.concatenate([
                train.loc[train_indices, NUM_COLS].values,
                test.loc[test_indices, NUM_COLS].values
            ])
            mu = X.mean(0)
            sig = X.std(0)
            
            train.loc[train_indices, NUM_COLS] = (train.loc[train_indices, NUM_COLS] - mu)/sig
            test.loc[test_indices, NUM_COLS] = (test.loc[test_indices, NUM_COLS] - mu)/sig

    with timer("Encode Categorical Features"):    
        for s in range(16):
            train_indices = train.site_id == s
            test_indices = test.site_id == s
            for col in CAT_COLS:

                x = np.concatenate([train.loc[train_indices, col], test.loc[test_indices, col]])
                encoder = FastLabelEncoder()
                encoder.fit(x)

                train.loc[train_indices, col] = encoder.transform(train.loc[train_indices, col])
                test.loc[test_indices, col] = encoder.transform(test.loc[test_indices, col])

    with timer("Save Data"):
        train.to_pickle(f"{DATA_PATH}/preprocessed/train_nn_site.pkl")
        test.to_pickle(f"{DATA_PATH}/preprocessed/test_nn_site.pkl")        
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

import argparse
import glob
import keras
import numpy as np 
import pandas as pd 

from ashrae.utils import (
    MODEL_PATH,  OUTPUT_PATH, timer, make_dir, rmsle,
    load_data, get_validation_months,
)

parser = argparse.ArgumentParser(description="")

parser.add_argument("--normalize_target", action="store_true", 
    help="If True then normalize the meter_reading by dividing by log1p(square_feet).")

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

def get_inputs(df):
    inputs = {col: np.array(df[col]) for col in CAT_COLS}
    inputs["numerical_inputs"]  = df[NUM_COLS].values    
    return inputs, df.target.values

def predict_mlp(df, save_name):
    model = keras.models.load_model(save_name)
    return model.predict(get_inputs(df)[0], batch_size=1024)            


if __name__ == "__main__":
    """
    python scripts/04_predict_mlp_meter.py --normalize_target
    python scripts/04_predict_mlp_meter.py
    """
    
    args = parser.parse_args()

    with timer("Loading data"):
        if args.normalize_target:
            test = load_data("test_nn_target_normalized_meter")
            test_square_feet = load_data("test_clean")["square_feet"].values
        else:
            test = load_data("test_nn_meter")
        test["target"] = -1

    with timer("Predicting"):            
        test_preds = np.zeros(len(test))

        for m in range(4):        
            print(m)
            # get base file name
            model_name = f"mlp-split_meter"
            make_dir(f"{MODEL_PATH}/{model_name}")

            # create sub model path
            if args.normalize_target:
                sub_model_path = f"{MODEL_PATH}/mlp-split_meter/target_normalization/meter_{m}"
            else:
                sub_model_path = f"{MODEL_PATH}/mlp-split_meter/no_normalization/meter_{m}"

            # remove indices not in this meter
            X = test.loc[test.meter == m, FEATURES + ["target"]]
            print(f"split meter {m}: test size {len(X)}")

            # load models
            model_list = glob.glob(f"{sub_model_path}/*")
            
            # predict 
            msg = f'Predicting for meter {m} - models# {len(model_list)}, test# {len(X)}'
            with timer(msg):
                # predict    
                assert len(model_list) != 0, "No models to load"

                if len(model_list) == 1:
                    preds = predict_mlp(X, model_list[0])
                else:
                    preds = np.mean([predict_mlp(X, model_name) for model_name in model_list], 0)                           
                test_preds[test.meter == m] = preds[:,0]

        # invert target transformation    
        if args.normalize_target:
            test_preds *= np.log1p(test_square_feet)

        test_preds = np.expm1(test_preds)

        # correct site 0
        test_preds[(test.site_id == 0) & (test.meter == 0)] *= 3.4118
        test_preds[test_preds < 0 ] = 0

    # save data
    if args.normalize_target:
        np.save(f"{OUTPUT_PATH}/mlp-split_meter-target_normalization", test_preds)
    else:
        np.save(f"{OUTPUT_PATH}/mlp-split_meter-no_normalization", test_preds)
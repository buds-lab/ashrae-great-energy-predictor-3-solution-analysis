import os
import argparse
import numpy as np 
import lightgbm as lgb
from datetime import datetime
from ashrae.utils import (
    MODEL_PATH, DATA_PATH, Logger, timer, make_dir,
    rmsle, load_data, get_validation_months,
)


parser = argparse.ArgumentParser(description="")

parser.add_argument("--overwrite", action="store_true", 
    help="If True then overwrite existing files")

parser.add_argument("--normalize_target", action="store_true", 
    help="If True then normalize the meter_reading by dividing by log1p(square_feet).")

parser.add_argument("--n_leaves", type=int, default=500,
    help="Number of leaves in each tree")

parser.add_argument("--lr", type=float, default=0.03,
    help="Learning rate.")

parser.add_argument("--feature_fraction", type=float, default=0.7,
    help="Fraction of features to select for each trees.")

parser.add_argument("--subsample", type=float, default=0.4,
    help="Fraction of rows to use when fitting trees.")


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
    "air_temperature_mean_lag7", "air_temperature_std_lag7",
    "air_temperature_mean_lag73", "air_temperature_std_lag73",
     
    # time features
    "weekday_x", "weekday_y", "is_holiday",
    
    # target encoding features
    "gte_meter_building_id_hour", "gte_meter_building_id_weekday",
]

CAT_COLS = [
    "building_id", "site_id", "primary_use", 
    "hour", "weekday", "weekday_hour",
    "building_weekday_hour", "building_weekday",
    "building_hour", 
]

DROP_COLS = [
    # time columns
    "year", "timestamp", "hour_x", "hour_y", 
    
    # weather extremum
    "air_temperature_min_lag7", "air_temperature_max_lag7",
    "air_temperature_min_lag73", "air_temperature_max_lag73",    
    
    # first-order gte
    "gte_hour", "gte_weekday", "gte_month", "gte_building_id",
    "gte_meter", "gte_meter_hour", "gte_primary_use", "gte_site_id", 
    
    # second-order gte
    "gte_meter_weekday", "gte_meter_month", "gte_meter_building_id",
    "gte_meter_primary_use", "gte_meter_site_id",  
    
    # month columns
    "month_x", "month_y", "building_month", #"month", 
    "gte_meter_building_id_month"
]


if __name__ == "__main__":
    """
    python scripts/03_train_lgb_meter.py --normalize_target
    python scripts/03_train_lgb_meter.py 
    """
    
    args = parser.parse_args()
    
    with timer("Loading data"):
        train = load_data("train_clean")
        train.drop(DROP_COLS, axis=1, inplace=True)
        train = train.loc[train.is_bad_meter_reading==0].reset_index(drop=True)

    with timer("Preprocesing"):
        for x in CAT_COLS:
            train[x] = train[x].astype("category")

        if args.normalize_target:
            target_encode_cols = [x for x in train.columns if "gte" in x]
            train[target_encode_cols] = train[target_encode_cols]/np.log1p(train[["square_feet"]].values)
            train["target"] = np.log1p(train["meter_reading"])/np.log1p(train["square_feet"])  
        else:
            train["target"] = np.log1p(train["meter_reading"])

    # get base file name
    model_name = f"lgb-split_meter"
    make_dir(f"{MODEL_PATH}/{model_name}")
    
    with timer("Training"):        
        # for seed in range(3): #@Matt, difference seed adds very littler diversity
        for seed in [0]:
            for n_months in [1,2,3,4,5,6]:
            #for n_months in [1,4,5,6]: #@Matt, n_months=3 brings optimal tradeoff between single model performance and diversity for the ensemble
                # validation_months_list = get_validation_months(n_months) #@Matt, fixed the bug -> hard-coded n_months
                validation_months_list = get_validation_months(n_months)                

                for fold_, validation_months in enumerate(validation_months_list):    
                    for m in range(4):    

                        # create sub model path
                        if args.normalize_target:
                            sub_model_path = f"{MODEL_PATH}/{model_name}/target_normalization/meter_{m}"
                            make_dir(sub_model_path)
                        else:
                            sub_model_path = f"{MODEL_PATH}/{model_name}/no_normalization/meter_{m}"
                            make_dir(sub_model_path)

                        # create model version
                        model_version = "_".join([
                            str(args.n_leaves), str(args.lr),
                            str(args.feature_fraction), str(args.subsample),
                            str(seed), str(n_months), str(fold_), 
                        ])    

                        # check if we can skip this model
                        full_sub_model_name = f"{sub_model_path}/{model_version}.txt"
                        if os.path.exists(full_sub_model_name):
                            if not args.overwrite:
                                print(f'{datetime.now()} -  {full_sub_model_name} already exists! Skipping...')
                                continue

                        # get this months indices
                        trn_idx = np.where(np.isin(train.month, validation_months, invert=True))[0]
                        val_idx = np.where(np.isin(train.month, validation_months, invert=False))[0]
                        #print(f"split meter: train size {len(trn_idx)} val size {len(val_idx)}")

                        # remove indices not in this meter
                        trn_idx = np.intersect1d(trn_idx, np.where(train.meter == m)[0])
                        val_idx = np.intersect1d(val_idx, np.where(train.meter == m)[0])
                        #print(f"split meter: train size {len(trn_idx)} val size {len(val_idx)}")

                        # initialize model
                        model = lgb.LGBMRegressor(random_state=seed+9999*args.normalize_target,
                                                  n_estimators=9999,
                                                  learning_rate=args.lr,
                                                  feature_fraction=args.feature_fraction,
                                                  subsample=args.subsample,
                                                  num_leaves=args.n_leaves,
                                                  metric="rmse", 
                                                  silent=False)

                        # fit model
                        msg = f'Training {full_sub_model_name} - train# {len(trn_idx)} val# {len(val_idx)}'
                        #print(f'{datetime.now()} - Training {full_sub_model_name} - train# {len(trn_idx)} val# {len(val_idx)}')
                        with timer(msg):
                            model.fit(train.loc[trn_idx, FEATURES], train.loc[trn_idx, "target"],
                                      eval_set=[(train.loc[val_idx, FEATURES], train.loc[val_idx, "target"])],
                                      early_stopping_rounds=50,
                                      verbose=50)

                        model.booster_.save_model(full_sub_model_name)
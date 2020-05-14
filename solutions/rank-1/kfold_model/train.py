#!/usr/bin/env python
# coding: utf-8
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import datetime
import gc

from utils import fill_weather_dataset, reduce_mem_usage, features_engineering, add_sg, timer


DATA_PATH       = "../input/"
PROCESSED_PATH  = "../processed/"
OUTPUT_PATH     = "../output/"
MODEL_PATH      = '../models/'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true', help='debug mode')
args = parser.parse_args()


def train(debug=True):
    nround = 1000
    if debug:
        nround = 10
        print ('debug mode, nround=', nround)

    black_day = 10
    
    with timer("Preprocessing"):    
        train_df = pd.read_csv(DATA_PATH + 'train.csv')

        # Remove outliers
        train_df = train_df [ train_df['building_id'] != 1099 ]
        train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')

        building_df = pd.read_csv(DATA_PATH + 'building_metadata.csv')
        weather_df = pd.read_csv(DATA_PATH + 'weather_train.csv')


        # remove continuous zero 
        train_df_black = pd.read_feather(PROCESSED_PATH + 'train_black.feather')
        train_df = train_df[train_df_black.black_count < 24*black_day]

        del train_df_black
        gc.collect()


        # site -0 correction
        site_0_bids = building_df[building_df.site_id == 0].building_id.unique()
        train_df.loc[(train_df.building_id.isin(site_0_bids)) & (train_df.meter==0), 'meter_reading'] = train_df[(train_df.building_id.isin(site_0_bids)) & (train_df.meter==0)]['meter_reading'] * 0.2931


        # ## Fill Weather Information
        weather_df = fill_weather_dataset(weather_df)
        add_sg(weather_df)


        # ## Memory Reduction
        train_df = reduce_mem_usage(train_df,use_float16=True)
        building_df = reduce_mem_usage(building_df,use_float16=True)
        weather_df = reduce_mem_usage(weather_df,use_float16=True)


        # ## Merge Data
        # 
        # We need to add building and weather information into training dataset.

        train_df = train_df.merge(building_df, left_on='building_id',right_on='building_id',how='left')
        train_df = train_df.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
        del weather_df
        gc.collect()


    # ## Features Engineering
    with timer("Feature engineering"):
        train_df = features_engineering(train_df)

    # ## Features & Target Variables

    target = np.log1p(train_df["meter_reading"])
    features = train_df.drop('meter_reading', axis = 1)
    del train_df
    gc.collect()

    # ##  KFOLD LIGHTGBM Model

    categorical_features = ["building_id", "site_id", "meter", "primary_use", "is_holiday", "weekend"]
    params = {
        "objective": "regression",
        "boosting": "gbdt",
        "num_leaves": 1280,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "reg_lambda": 2,
        "metric": "rmse",
    }

    kf = KFold(n_splits=3)
    models = []
    
    with timer("Training"):    
        for train_index,test_index in kf.split(features):
            train_features = features.loc[train_index]
            train_target = target.loc[train_index]

            test_features = features.loc[test_index]
            test_target = target.loc[test_index]

            d_training = lgb.Dataset(train_features, label=train_target,categorical_feature=categorical_features, free_raw_data=False)
            d_test = lgb.Dataset(test_features, label=test_target,categorical_feature=categorical_features, free_raw_data=False)

            msg = f'Training - train# {len(train_index)} val# {len(test_index)}'
            with timer(msg):
                model = lgb.train(params, train_set=d_training, num_boost_round=nround, valid_sets=[d_training,d_test], verbose_eval=25, early_stopping_rounds=50)

            models.append(model)
            del train_features, train_target, test_features, test_target, d_training, d_test
            gc.collect()


    del features, target
    gc.collect()

    if not debug:
        import pickle
        with open(MODEL_PATH + 'kfold_model.pickle', mode='wb') as f:
            pickle.dump([models], f)

    del models


if __name__ == '__main__':
    debug = args.debug
    print (debug)
    train(debug)
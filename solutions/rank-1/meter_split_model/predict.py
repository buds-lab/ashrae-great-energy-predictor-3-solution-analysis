#!/usr/bin/env python
# coding: utf-8

# this kernel was based https://www.kaggle.com/corochann/ashrae-training-lgbm-by-meter-type

import argparse
import gc
import os
from pathlib import Path
import random
import sys

from tqdm import tqdm_notebook as tqdm
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import KFold
import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from utils import reduce_mem_usage, add_holiyday, correct_localtime, add_lag_feature, add_sg, preprocess, timer

from pathlib import Path
import pickle

# dirs
LEAK_DIR = Path('../input')
DATA_DIR = Path('../processed')

OUTPUT_DIR = Path('../output')
MODEL_DIR  = Path('../models')

parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true', help='debug mode')
args = parser.parse_args()


category_cols = ['building_id', 'site_id', 'primary_use', 'IsHoliday']  # , 'meter'
feature_cols = ['square_feet', 'year_built'] + [
    'hour', 'weekend',
    'day', # 'month' ,
#    'dayofweek',
#    'building_median'
    ] + [
    'air_temperature', 'cloud_coverage',
    'dew_temperature', 'precip_depth_1_hr',
    'sea_level_pressure',
#'wind_direction', 'wind_speed',
    'air_temperature_mean_lag72',
    'air_temperature_max_lag72', 'air_temperature_min_lag72',
    'air_temperature_std_lag72', 'cloud_coverage_mean_lag72',
    'dew_temperature_mean_lag72', 'precip_depth_1_hr_mean_lag72',
    'sea_level_pressure_mean_lag72',
#'wind_direction_mean_lag72',
    'wind_speed_mean_lag72', 
    'air_temperature_mean_lag3',
    'air_temperature_max_lag3',
    'air_temperature_min_lag3', 'cloud_coverage_mean_lag3',
    'dew_temperature_mean_lag3',
    'precip_depth_1_hr_mean_lag3',
    'sea_level_pressure_mean_lag3',
#    'wind_direction_mean_lag3', 'wind_speed_mean_lag3',
#    'floor_area',
    'year_cnt', 'bid_cnt',
    'dew_smooth', 'air_smooth',
    'dew_diff', 'air_diff',
    'dew_diff2', 'air_diff2',
]


def create_X(test_df, building_meta_df, weather_test_df, target_meter):
    target_test_df = test_df[test_df['meter'] == target_meter]
    target_test_df = target_test_df.merge(building_meta_df, on='building_id', how='left')
    target_test_df = target_test_df.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')
    #X_test = target_test_df[feature_cols + category_cols + ['month']]
    X_test = target_test_df[feature_cols + category_cols ]
    return X_test


def pred(X_test, models, batch_size=1000000):
    iterations = (X_test.shape[0] + batch_size -1) // batch_size
    print('iterations', iterations)

    y_test_pred_total = np.zeros(X_test.shape[0])
    for i, (mindex, model) in enumerate(models):
        print(f'predicting {i}-th model')
        for k in tqdm(range(iterations)):
            y_pred_test = model.predict(X_test[k*batch_size:(k+1)*batch_size], num_iteration=model.best_iteration)
            y_test_pred_total[k*batch_size:(k+1)*batch_size] += y_pred_test

    y_test_pred_total /= len(models)
    return y_test_pred_total


def predict(deubg=True):

    # replace leak before submission
    replace_leak = True 

    # some tuning parameters of models
    black_day = 10 # threshold of removing continuos zero values

    # # Prediction on test data

    #with open(model_dir/'meter_split.pickle', mode='rb') as f:
    with open(MODEL_DIR/'meter_split_model.pickle', mode='rb') as f:
        [models0, models1, models2, models3, bid_map] = pickle.load(f)


    with timer("Preprocessing"):        
        # categorize primary_use column to reduce memory on merge...
        building_meta_df = pd.read_feather(DATA_DIR/'building_metadata.feather')

        primary_use_list = building_meta_df['primary_use'].unique()
        primary_use_dict = {key: value for value, key in enumerate(primary_use_list)} 
        print('primary_use_dict: ', primary_use_dict)
        building_meta_df['primary_use'] = building_meta_df['primary_use'].map(primary_use_dict)

        year_map = building_meta_df.year_built.value_counts()
        building_meta_df['year_cnt'] = building_meta_df.year_built.map(year_map)

        building_meta_df = reduce_mem_usage(building_meta_df, use_float16=True)

        gc.collect()


        print('loading...')
        test_df = pd.read_feather(DATA_DIR/'test.feather')
        weather_test_df = pd.read_feather(DATA_DIR/'weather_test.feather')

        weather_test_df = weather_test_df.drop_duplicates(['timestamp', 'site_id'])

        correct_localtime(weather_test_df)
        add_holiyday(weather_test_df)

        print('preprocessing building...')
        test_df['date'] = test_df['timestamp'].dt.date
        preprocess(test_df)

        print('preprocessing weather...')
        weather_test_df = weather_test_df.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))
        weather_test_df.groupby('site_id').apply(lambda group: group.isna().sum())

        add_sg(weather_test_df)
        
    with timer("Feature engineering"):        

        add_lag_feature(weather_test_df, window=3)
        add_lag_feature(weather_test_df, window=72)

        test_df['bid_cnt'] = test_df.building_id.map(bid_map)

    print('reduce mem usage...')
    test_df = reduce_mem_usage(test_df, use_float16=True)
    weather_test_df = reduce_mem_usage(weather_test_df, use_float16=True)

    gc.collect()
    print (test_df.shape)


    sample_submission = pd.read_feather(os.path.join(DATA_DIR, 'sample_submission.feather'))
    sample_submission = reduce_mem_usage(sample_submission)

    # meter 0
    
    X_test = create_X(test_df, building_meta_df, weather_test_df, target_meter=0)
    gc.collect()
    X_test.info()

    with timer("Predicting meter# 0"):
        y_test0 = pred(X_test, models0)

    #sns.distplot(y_test0)

    print(X_test.shape, y_test0.shape)

    del X_test
    gc.collect()

    # meter 1

    X_test = create_X(test_df, building_meta_df, weather_test_df, target_meter=1)
    gc.collect()

    with timer("Predicting meter# 1"):
        y_test1 = pred(X_test, models1)
    #sns.distplot(y_test1)

    print(X_test.shape, y_test1.shape)

    del X_test
    gc.collect()

    # meter 2
    X_test = create_X(test_df, building_meta_df, weather_test_df, target_meter=2)
    gc.collect()

    with timer("Predicting meter# 2"):
        y_test2 = pred(X_test, models2)
    #sns.distplot(y_test2)

    print(X_test.shape, y_test2.shape)
    del X_test

    gc.collect()

    # meter 3

    X_test = create_X(test_df, building_meta_df, weather_test_df, target_meter=3)
    gc.collect()

    with timer("Predicting meter# 3"):
        y_test3 = pred(X_test, models3)

    #sns.distplot(y_test3)
    print(X_test.shape, y_test3.shape)
    del X_test
    gc.collect()

    # check
    print(sample_submission.loc[test_df['meter'] == 0, 'meter_reading'].shape,np.expm1(y_test0).shape)
    print(sample_submission.loc[test_df['meter'] == 1, 'meter_reading'].shape,np.expm1(y_test1).shape)
    print(sample_submission.loc[test_df['meter'] == 2, 'meter_reading'].shape,np.expm1(y_test2).shape)
    print(sample_submission.loc[test_df['meter'] == 3, 'meter_reading'].shape,np.expm1(y_test3).shape)

    sample_submission.loc[test_df['meter'] == 0, 'meter_reading'] = np.expm1(y_test0)
    sample_submission.loc[test_df['meter'] == 1, 'meter_reading'] = np.expm1(y_test1)
    sample_submission.loc[test_df['meter'] == 2, 'meter_reading'] = np.expm1(y_test2)
    sample_submission.loc[test_df['meter'] == 3, 'meter_reading'] = np.expm1(y_test3)


    # # site-0 correction 

    # https://www.kaggle.com/c/ashrae-energy-prediction/discussion/119261#latest-684102
    site_0_bids = building_meta_df[building_meta_df.site_id == 0].building_id.unique()
    sample_submission.loc[(test_df.building_id.isin(site_0_bids)) & (test_df.meter==0), 'meter_reading'] = sample_submission[(test_df.building_id.isin(site_0_bids)) & (test_df.meter==0)]['meter_reading'] * 3.4118


    if not debug:
        sample_submission.to_csv(OUTPUT_DIR/'submission_meter.csv', index=False, float_format='%.4f')

    #np.log1p(sample_submission['meter_reading']).hist(bins=100)

    # # replace leak data
    
    with timer("Post-processing"):

        if replace_leak:
            leak_df = pd.read_feather(LEAK_DIR/'leak.feather')

            print(leak_df.duplicated().sum())
            print(leak_df.meter.value_counts())

            leak_df.fillna(0, inplace=True)
            leak_df = leak_df[(leak_df.timestamp.dt.year > 2016) & (leak_df.timestamp.dt.year < 2019)]
            leak_df.loc[leak_df.meter_reading < 0, 'meter_reading'] = 0 # remove large negative values
            leak_df = leak_df[leak_df.building_id!=245]

            sample_submission.loc[sample_submission.meter_reading < 0, 'meter_reading'] = 0

            test_df['pred'] = sample_submission.meter_reading

            leak_df = leak_df.merge(test_df[['building_id', 'meter', 'timestamp', 'pred', 'row_id']], left_on = ['building_id', 'meter', 'timestamp'], right_on = ['building_id', 'meter', 'timestamp'], how = "left")
            leak_df = leak_df.merge(building_meta_df[['building_id', 'site_id']], on='building_id', how='left')

        if replace_leak:
            leak_df.site_id.unique()

        if replace_leak:
            leak_df['pred_l1p'] = np.log1p(leak_df.pred)
            leak_df['meter_reading_l1p'] = np.log1p(leak_df.meter_reading)

            sns.distplot(leak_df.pred_l1p)
            sns.distplot(leak_df.meter_reading_l1p)

            leak_score = np.sqrt(mean_squared_error(leak_df.pred_l1p, leak_df.meter_reading_l1p))


        if replace_leak:
            leak_df = leak_df[['meter_reading', 'row_id']].set_index('row_id').dropna()
            sample_submission.loc[leak_df.index, 'meter_reading'] = leak_df['meter_reading']


    if not debug and replace_leak:
        sample_submission.to_csv(OUTPUT_DIR/'submission_replaced_meter.csv', index=False, float_format='%.4f')


    # # Scores
    #LV score= 0.9743280741946935
    print('LV score=', leak_score)


if __name__ == '__main__':
    debug = args.debug
    print ('debug=', debug)
    predict(debug)

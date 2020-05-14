#!/usr/bin/env python
# coding: utf-8
# this kernel was based on https://www.kaggle.com/corochann/ashrae-training-lgbm-by-meter-type

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

from utils import reduce_mem_usage, add_holiyday, correct_localtime, add_lag_feature, add_sg, timer

from pathlib import Path

DATA_DIR = Path('../processed')
MODEL_DIR = Path('../models')

parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true', help='debug mode')
args = parser.parse_args()


def train(debug=False):

    # some tuning parameters of models
    black_day = 10 # threshold of removing continuos zero values
    num_rounds = 1000 # num_rounds for lgbm
    folds = 3 

    if debug:
        num_rounds = 5

    # # Fast data loading
    with timer("Preprocessing"):

        train_df = pd.read_feather(DATA_DIR/'train.feather')
        train_df_black = pd.read_feather(DATA_DIR/'train_black.feather')
        weather_train_df = pd.read_feather(DATA_DIR/'weather_train.feather')
        building_meta_df = pd.read_feather(DATA_DIR/'building_metadata.feather')


        # # remove bad buildings

        train_df = train_df[ train_df['building_id'] != 1099 ]

        # # Timezone Correction
        # change wheather meta data localtime to UTC 

        correct_localtime(weather_train_df)
        add_holiyday(weather_train_df)

        weather_train_df.head()


        # # Remove continuous zero meter
        train_df = train_df[train_df_black.black_count < 24*black_day]


        del train_df_black
        gc.collect()


        # ## Removing bad data on site_id 0
        # 
        # As you can see above, this data looks weired until May 20. It is reported in [this discussion](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/113054#656588) by @barnwellguy that **All electricity meter is 0 until May 20 for site_id == 0**. I will remove these data from training data.
        # 
        # It corresponds to `building_id <= 104`.

        train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
        train_df = train_df.query('not (building_id == 954 & meter_reading == 0)')
        train_df = train_df.query('not (building_id == 1221 & meter_reading == 0)')

        train_df = train_df.reset_index()
        gc.collect()


        # # Site-0 Correction

        # https://www.kaggle.com/c/ashrae-energy-prediction/discussion/119261#latest-684102
        site_0_bids = building_meta_df[building_meta_df.site_id == 0].building_id.unique()
        print (len(site_0_bids), len(train_df[train_df.building_id.isin(site_0_bids)].building_id.unique()))
        #train_df[train_df.building_id.isin(site_0_bids)].head()


        train_df.loc[(train_df.building_id.isin(site_0_bids)) & (train_df.meter==0), 'meter_reading'] = train_df[(train_df.building_id.isin(site_0_bids)) & (train_df.meter==0) ]['meter_reading'] * 0.2931

        #train_df[train_df.building_id.isin(site_0_bids)].head()

        # # Data preprocessing
        # 
        # Now, Let's try building GBDT (Gradient Boost Decision Tree) model to predict `meter_reading_log1p`.
        # I will try using LightGBM in this notebook.
        train_df['date'] = train_df['timestamp'].dt.date
        train_df['meter_reading_log1p'] = np.log1p(train_df['meter_reading'])


    # # Add time feature
    # Some features introduced in https://www.kaggle.com/ryches/simple-lgbm-solution by @ryches
    # 
    # Features that are likely predictive:
    # 
    # #### Weather
    # 
    # - time of day
    # - holiday
    # - weekend
    # - cloud_coverage + lags
    # - dew_temperature + lags
    # - precip_depth + lags
    # - sea_level_pressure + lags
    # - wind_direction + lags
    # - wind_speed + lags
    # 
    # However we should be careful of putting time feature, since we have only 1 year data in training,
    # including `date` makes overfiting to training data.

    def preprocess(df):
        df["hour"] = df["timestamp"].dt.hour
        df["day"] = df["timestamp"].dt.day
        df["weekend"] = df["timestamp"].dt.weekday
        df["month"] = df["timestamp"].dt.month
        df["dayofweek"] = df["timestamp"].dt.dayofweek
        
    with timer("Feature engineering"):

        preprocess(train_df)


        # # Fill Nan value in weather dataframe by interpolation
        # 
        # 
        # weather data has a lot of NaNs!!
        # 
        # ![](http://)I tried to fill these values by **interpolating** data.

        weather_train_df = weather_train_df.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))

        # Seems number of nan has reduced by `interpolate` but some property has never appear in specific `site_id`, and nan remains for these features.

        # ## lags
        # 
        # Adding some lag feature

        add_lag_feature(weather_train_df, window=3)
        add_lag_feature(weather_train_df, window=72)

        # # count encoding

        year_map = building_meta_df.year_built.value_counts()
        building_meta_df['year_cnt'] = building_meta_df.year_built.map(year_map)

        bid_map = train_df.building_id.value_counts()
        train_df['bid_cnt'] = train_df.building_id.map(bid_map)

        # categorize primary_use column to reduce memory on merge...
        primary_use_list = building_meta_df['primary_use'].unique()
        primary_use_dict = {key: value for value, key in enumerate(primary_use_list)} 
        print('primary_use_dict: ', primary_use_dict)
        building_meta_df['primary_use'] = building_meta_df['primary_use'].map(primary_use_dict)

    gc.collect()

    train_df = reduce_mem_usage(train_df, use_float16=True)
    building_meta_df = reduce_mem_usage(building_meta_df, use_float16=True)
    weather_train_df = reduce_mem_usage(weather_train_df, use_float16=True)


    # # Savitzky–Golay Filter for Weather data

    add_sg(weather_train_df)


    # # Train model

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


    def create_X_y(train_df, target_meter):
        target_train_df = train_df[train_df['meter'] == target_meter]
        target_train_df = target_train_df.merge(building_meta_df, on='building_id', how='left')
        target_train_df = target_train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')

        X_train = target_train_df[feature_cols + category_cols]
        y_train = target_train_df['meter_reading_log1p'].values

        del target_train_df
        return X_train, y_train


    def fit_lgbm(train, val, devices=(-1,), seed=None, cat_features=None, num_rounds=1500, lr=0.1, bf=0.1):
        """Train Light GBM model"""
        X_train, y_train = train
        X_valid, y_valid = val
        metric = 'l2'
        params = {'num_leaves': 31,
                  'objective': 'regression',
    #               'max_depth': -1,
                  'learning_rate': lr,
                  "boosting": "gbdt",
                  "bagging_freq": 5,
                  "bagging_fraction": bf,
                  "feature_fraction": 0.9,
                  "metric": metric,
    #               "verbosity": -1,
    #               'reg_alpha': 0.1,
    #               'reg_lambda': 0.3
                  }
        device = devices[0]
        if device == -1:
            # use cpu
            pass
        else:
            # use gpu
            print(f'using gpu device_id {device}...')
            params.update({'device': 'gpu', 'gpu_device_id': device})

        params['seed'] = seed

        early_stop = 20
        verbose_eval = 20

        d_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
        d_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_features)
        watchlist = [d_train, d_valid]

        print('training LGB:')
        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop)

        # predictions
        y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)

        print('best_score', model.best_score)
        log = {'train/mae': model.best_score['training']['l2'],
               'valid/mae': model.best_score['valid_1']['l2']}
        return model, y_pred_valid, log


    from sklearn.model_selection import GroupKFold

    seed = 666
    shuffle = False
    kf = GroupKFold(n_splits=folds)


    # # Train model by each meter type

    def get_groups(df, meter):
        if folds == 12:
            return df[df.meter==meter].month -1
        elif folds == 6:
            return (df[df.meter==meter].month -1) // 2
        elif folds == 3:
            return (df[df.meter==meter].month -1) // 4


    # ## model for meter 0

    oof_total = 0
    target_meter = 0    
    
    with timer("Training meter #" + str(target_meter)):
    
        X_train, y_train = create_X_y(train_df, target_meter=target_meter)
        y_valid_pred_total = np.zeros(X_train.shape[0])
        gc.collect()
        print('target_meter', target_meter, X_train.shape)
        X_train.info()

        cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]
        print('cat_features', cat_features)

        models0 = []

        #for train_idx, valid_idx in kf.split(X_train, y_train):

        for train_idx, valid_idx in kf.split(X_train, y_train, groups=get_groups(train_df, target_meter)):    
            train_data = X_train.iloc[train_idx,:], y_train[train_idx]
            valid_data = X_train.iloc[valid_idx,:], y_train[valid_idx]

            mindex = train_df[train_df.meter==target_meter].iloc[valid_idx,:].month.unique()
            print (mindex)
            
            msg = f'Training - train# {len(train_idx)} val# {len(train_idx)}'
            with timer(msg):
        #     model, y_pred_valid, log = fit_cb(train_data, valid_data, cat_features=cat_features, devices=[0,])
                model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols,
                                                    num_rounds=num_rounds, lr=0.05, bf=0.7)
            y_valid_pred_total[valid_idx] = y_pred_valid
            models0.append([mindex, model])
            gc.collect()
            if debug:
                break

    oof0 = mean_squared_error(y_train, y_valid_pred_total)

    sns.distplot(y_train)
    sns.distplot(y_valid_pred_total)

    oof_total += oof0 * len(y_train)

    del X_train, y_train
    gc.collect()


    def plot_feature_importance(model):
        importance_df = pd.DataFrame(model[1].feature_importance(),
                                     index=feature_cols + category_cols,
                                     columns=['importance']).sort_values('importance')
        fig, ax = plt.subplots(figsize=(8, 8))
        importance_df.plot.barh(ax=ax)
        fig.show()


    # ## model for meter 1

    target_meter = 1
    with timer("Training meter #" + str(target_meter)):
        X_train, y_train = create_X_y(train_df, target_meter=target_meter)
        y_valid_pred_total = np.zeros(X_train.shape[0])
        gc.collect()
        print('target_meter', target_meter, X_train.shape)

        cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]
        print('cat_features', cat_features)

        models1 = []

        for train_idx, valid_idx in kf.split(X_train, y_train, groups=get_groups(train_df, target_meter)):    
        #for train_idx, valid_idx in kf.split(X_train, y_train):
            train_data = X_train.iloc[train_idx,:], y_train[train_idx]
            valid_data = X_train.iloc[valid_idx,:], y_train[valid_idx]

            mindex = train_df[train_df.meter==target_meter].iloc[valid_idx,:].month.unique()

            #print('train', len(train_idx), 'valid', len(valid_idx))
        #     model, y_pred_valid, log = fit_cb(train_data, valid_data, cat_features=cat_features, devices=[0,])
            msg = f'Training - train# {len(train_idx)} val# {len(valid_idx)}'
            with timer(msg):        
                model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols, num_rounds=num_rounds,
                                                   lr=0.05, bf=0.5)
            y_valid_pred_total[valid_idx] = y_pred_valid
            models1.append([mindex, model])
            gc.collect()
            if debug:
                break

    oof1 = mean_squared_error(y_train, y_valid_pred_total)

    sns.distplot(y_train)
    sns.distplot(y_valid_pred_total)

    oof_total += oof1 * len(y_train)

    del X_train, y_train
    gc.collect()


    # ## model for meter 2

    target_meter = 2
    with timer("Training meter #" + str(target_meter)):
        X_train, y_train = create_X_y(train_df, target_meter=target_meter)
        y_valid_pred_total = np.zeros(X_train.shape[0])

        gc.collect()
        print('target_meter', target_meter, X_train.shape)

        cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]
        print('cat_features', cat_features)

        models2 = []


        for train_idx, valid_idx in kf.split(X_train, y_train, groups=get_groups(train_df, target_meter)):
        #for train_idx, valid_idx in kf.split(X_train, y_train):
            train_data = X_train.iloc[train_idx,:], y_train[train_idx]
            valid_data = X_train.iloc[valid_idx,:], y_train[valid_idx]

            mindex = train_df[train_df.meter==target_meter].iloc[valid_idx,:].month.unique()
            #print('train', len(train_idx), 'valid', len(valid_idx))
        #     model, y_pred_valid, log = fit_cb(train_data, valid_data, cat_features=cat_features, devices=[0,])
            msg = f'Training - train# {len(train_idx)} val# {len(valid_idx)}'
            with timer(msg):    
                model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols,
                                                    num_rounds=num_rounds, lr=0.05, bf=0.8)
            y_valid_pred_total[valid_idx] = y_pred_valid
            models2.append([mindex, model])
            gc.collect()
            if debug:
                break

    oof2 = mean_squared_error(y_train, y_valid_pred_total)

    oof_total += oof2 * len(y_train)

    sns.distplot(y_train)
    sns.distplot(y_valid_pred_total)

    del X_train, y_train
    gc.collect()


    # ## model for meter 3

    target_meter = 3
    with timer("Training meter #" + str(target_meter)):
        X_train, y_train = create_X_y(train_df, target_meter=target_meter)
        y_valid_pred_total = np.zeros(X_train.shape[0])

        gc.collect()
        print('target_meter', target_meter, X_train.shape)

        cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]
        print('cat_features', cat_features)

        models3 = []

        for train_idx, valid_idx in kf.split(X_train, y_train, groups=get_groups(train_df, target_meter)):    
        #for train_idx, valid_idx in kf.split(X_train, y_train):
            train_data = X_train.iloc[train_idx,:], y_train[train_idx]
            valid_data = X_train.iloc[valid_idx,:], y_train[valid_idx]

            mindex = train_df[train_df.meter==target_meter].iloc[valid_idx,:].month.unique()
            #print('train', len(train_idx), 'valid', len(valid_idx))
        #     model, y_pred_valid, log = fit_cb(train_data, valid_data, cat_features=cat_features, devices=[0,])
            msg = f'Training - train# {len(train_idx)} val# {len(valid_idx)}'
            with timer(msg):        
                model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols, num_rounds=num_rounds,
                                                   lr=0.03, bf=0.9)
            y_valid_pred_total[valid_idx] = y_pred_valid
            models3.append([mindex, model])
            gc.collect()
            if debug:
                break

    oof3 = mean_squared_error(y_train, y_valid_pred_total)

    oof_total += oof3 * len(y_train)

    oof_total = oof_total / len(train_df)

    sns.distplot(y_train)
    sns.distplot(y_valid_pred_total)

    del X_train, y_train
    gc.collect()


    # # OOF MSE

    print('oof0=', np.sqrt(oof0))
    print('oof1=', np.sqrt(oof1))
    print('oof2=', np.sqrt(oof2))
    print('oof3=', np.sqrt(oof3))
    print('total=', np.sqrt(oof_total))

    # # Save Models
    if not debug:
        import pickle
        with open(MODEL_DIR/'meter_split_model.pickle', mode='wb') as f:
            pickle.dump([models0, models1, models2, models3, bid_map], f)


if __name__ == '__main__':
    debug = args.debug
    print ('debug=', debug)

    train(debug)

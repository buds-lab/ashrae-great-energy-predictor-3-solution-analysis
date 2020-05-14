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
import pickle

from utils import fill_weather_dataset, reduce_mem_usage, features_engineering, add_sg, timer


DATA_PATH      = "../input/"
PROCESSED_PATH = "../processed/"
OUTPUT_PATH    = "../output/"
MODEL_PATH     = '../models/'


parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true', help='debug mode')
args = parser.parse_args()


def predict(debug=True):
    # # Load Models

    with open(MODEL_PATH + 'kfold_model.pickle', mode='rb') as f:
        [models] = pickle.load(f)
        
        
    with timer("Preprocessing"):

        # ## Load Test Data

        building_df = pd.read_csv(DATA_PATH + 'building_metadata.csv')
        building_df = reduce_mem_usage(building_df,use_float16=True)
        site_0_bids = building_df[building_df.site_id == 0].building_id.unique()

        test_df = pd.read_csv(DATA_PATH + 'test.csv')
        row_ids = test_df["row_id"]
        test_df.drop("row_id", axis=1, inplace=True)
        test_df = reduce_mem_usage(test_df)


        # ## Merge Building Data
        test_df = test_df.merge(building_df,left_on='building_id',right_on='building_id',how='left')
        del building_df
        gc.collect()


        # ## Fill Weather Information

        weather_df = pd.read_csv(DATA_PATH + 'weather_test.csv')
        weather_df = fill_weather_dataset(weather_df)
        add_sg(weather_df)
        weather_df = reduce_mem_usage(weather_df)


        # ## Merge Weather Data

        test_df = test_df.merge(weather_df,how='left',on=['timestamp','site_id'])
        del weather_df
        gc.collect()


    # ## Features Engineering
    with timer("Feature engineering"):
        test_df = features_engineering(test_df)
        test_df = reduce_mem_usage(test_df)

    #test_df.head(20)

    # ## Prediction
    with timer("Predicting"):
        results = []
        for model in models:
            with timer("Predicting #"):
                if  len(results) == 0:
                    results = np.expm1(model.predict(test_df, num_iteration=model.best_iteration)) / len(models)
                else:
                    results += np.expm1(model.predict(test_df, num_iteration=model.best_iteration)) / len(models)
            del model
            gc.collect()


    # ## Submission

    sample_submission = pd.DataFrame({"row_id": row_ids, "meter_reading": np.clip(results, 0, a_max=None)})
    del row_ids,results
    gc.collect()

    #site-0 correction
    sample_submission.loc[(test_df.building_id.isin(site_0_bids)) & (test_df.meter==0), 'meter_reading'] = sample_submission[(test_df.building_id.isin(site_0_bids)) & (test_df.meter==0)]['meter_reading'] * 3.4118

    del test_df, models
    gc.collect()


    if not debug:
        sample_submission.to_csv(OUTPUT_PATH + "submission_kfold.csv", index=False, float_format='%.4f')

    with timer("Post-processing"):

        leak_df = pd.read_feather(DATA_PATH + 'leak.feather')

        leak_df.fillna(0, inplace=True)
        leak_df = leak_df[(leak_df.timestamp.dt.year > 2016) & (leak_df.timestamp.dt.year < 2019)]
        leak_df.loc[leak_df.meter_reading < 0, 'meter_reading'] = 0 # remove large negative values
        leak_df = leak_df[leak_df.building_id!=245]

        test_df = pd.read_feather(PROCESSED_PATH  + 'test.feather')
        building_meta_df = pd.read_feather(PROCESSED_PATH  + 'building_metadata.feather')
        test_df['timestamp'] = pd.to_datetime(test_df.timestamp)

        sample_submission.loc[sample_submission.meter_reading < 0, 'meter_reading'] = 0

        test_df['pred'] = sample_submission.meter_reading

        leak_df = leak_df.merge(test_df[['building_id', 'meter', 'timestamp', 'pred', 'row_id']], left_on = ['building_id', 'meter', 'timestamp'], right_on = ['building_id', 'meter', 'timestamp'], how = "left")
        leak_df = leak_df.merge(building_meta_df[['building_id', 'site_id']], on='building_id', how='left')

        import matplotlib.pyplot as plt
        import seaborn as sns

        from sklearn.metrics import mean_squared_error

        leak_df['pred_l1p'] = np.log1p(leak_df.pred)
        leak_df['meter_reading_l1p'] = np.log1p(leak_df.meter_reading)

        #sns.distplot(leak_df.pred_l1p)
        #sns.distplot(leak_df.meter_reading_l1p)

        leak_score = np.sqrt(mean_squared_error(leak_df.pred_l1p, leak_df.meter_reading_l1p))


    # # LV score

    print('total score=', leak_score)


    leak_df = leak_df[['meter_reading', 'row_id']].set_index('row_id').dropna()
    sample_submission.loc[leak_df.index, 'meter_reading'] = leak_df['meter_reading']
    if not debug:
        sample_submission.to_csv(OUTPUT_PATH + 'submission_replaced_kfold.csv', index=False, float_format='%.4f')

    sample_submission.head(20)

    #np.log1p(sample_submission['meter_reading']).hist(bins=100)


if __name__ == '__main__':
    debug = args.debug
    print ('debug=', debug)
    predict(debug)

#!/usr/bin/env python
# coding: utf-8
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import pickle
from utils import *


INPUT_DIR      = '../input/'
PROCESSED_PATH = "../processed/"
OUTPUT_DIR     = '../output/'
MODEL_PATH     = '../models/'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true', help='debug mode')
args = parser.parse_args()

drops= ["timestamp", 'wind_direction', 'wind_speed']


def predict(debug=True):
    
    with open(MODEL_PATH + 'cleanup_model.pickle', mode='rb') as f:
        [model] = pickle.load(f)
        
    with timer("Preprocesing"):
        X = combined_test_data()
        X = compress_dataframe(add_time_features(X))
        X = X.drop(columns=drops)  # Raw timestamp doesn't help when prediction
        
    with timer("Training"):
        predictions = pd.DataFrame({
            "row_id": X.index,
            "meter_reading": np.clip(np.expm1(model.predict(X)), 0, None)
        })
        predictions.loc[(X.site_id == 0) & (X.meter==0), 'meter_reading'] =  predictions.loc[(X.site_id == 0) & (X.meter==0), 'meter_reading'] * 3.4118

    del X

    # Finally, write the predictions out for submission. After that, it's Miller Time (tm).

    if not debug:
        predictions.to_csv(OUTPUT_DIR + "submission_cleanup.csv", index=False, float_format="%.4f")
        
        
    with timer("Post-procesing"):
        # # LB Score
        leak_df = pd.read_feather(INPUT_DIR + 'leak.feather')

        leak_df.fillna(0, inplace=True)
        leak_df = leak_df[(leak_df.timestamp.dt.year > 2016) & (leak_df.timestamp.dt.year < 2019)]
        leak_df.loc[leak_df.meter_reading < 0, 'meter_reading'] = 0 # remove large negative values
        leak_df = leak_df[leak_df.building_id!=245]

        test_df = pd.read_feather(PROCESSED_PATH + 'test.feather')
        building_meta_df = pd.read_feather(PROCESSED_PATH + 'building_metadata.feather')
        test_df['timestamp'] = pd.to_datetime(test_df.timestamp)

        test_df['pred'] = predictions.meter_reading
        leak_df = leak_df.merge(test_df[['building_id', 'meter', 'timestamp', 'pred', 'row_id']], left_on = ['building_id', 'meter', 'timestamp'], right_on = ['building_id', 'meter', 'timestamp'], how = "left")
        leak_df = leak_df.merge(building_meta_df[['building_id', 'site_id']], on='building_id', how='left')

        leak_df['pred_l1p'] = np.log1p(leak_df.pred)
        leak_df['meter_reading_l1p'] = np.log1p(leak_df.meter_reading)

        sns.distplot(leak_df.pred_l1p)
        sns.distplot(leak_df.meter_reading_l1p)

        leak_score = np.sqrt(mean_squared_error(leak_df.pred_l1p, leak_df.meter_reading_l1p))

        leak_df = leak_df[['meter_reading', 'row_id']].set_index('row_id').dropna()
        predictions.loc[leak_df.index, 'meter_reading'] = leak_df['meter_reading']
        if not debug:
             predictions.to_csv(OUTPUT_DIR + 'submission_replaced_cleanup.csv', index=False, float_format='%.4f')
                
        print('total score=', leak_score)

if __name__ == '__main__':
    debug = args.debug
    print ('debug=', debug)
    predict(debug)
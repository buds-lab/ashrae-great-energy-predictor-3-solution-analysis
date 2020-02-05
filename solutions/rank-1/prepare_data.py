#!/usr/bin/env python
# coding: utf-8

# based on public kernel https://www.kaggle.com/corochann/ashrae-feather-format-for-fast-loading

import os
import random
import gc

import tqdm
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def prepare(root, output):
    train_df = pd.read_csv(os.path.join(root, 'train.csv'))
    test_df = pd.read_csv(os.path.join(root, 'test.csv'))
    building_meta_df = pd.read_csv(os.path.join(root, 'building_metadata.csv'))
    sample_submission = pd.read_csv(os.path.join(root, 'sample_submission.csv'))
    weather_train_df = pd.read_csv(os.path.join(root, 'weather_train.csv'))
    weather_test_df = pd.read_csv(os.path.join(root, 'weather_test.csv'))

    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    weather_train_df['timestamp'] = pd.to_datetime(weather_train_df['timestamp'])
    weather_test_df['timestamp'] = pd.to_datetime(weather_test_df['timestamp'])

    # # Save data in feather format
    train_df.to_feather(os.path.join(output,'train.feather'))
    test_df.to_feather(os.path.join(output,'test.feather'))
    weather_train_df.to_feather(os.path.join(output,'weather_train.feather'))
    weather_test_df.to_feather(os.path.join(output,'weather_test.feather'))
    building_meta_df.to_feather(os.path.join(output,'building_metadata.feather'))
    sample_submission.to_feather(os.path.join(output,'sample_submission.feather'))

    # # Read data in feather format
    train_df = pd.read_feather(os.path.join(output, 'train.feather'))
    weather_train_df = pd.read_feather(os.path.join(output, 'weather_train.feather'))
    test_df = pd.read_feather(os.path.join(output, 'test.feather'))
    weather_test_df = pd.read_feather(os.path.join(output, 'weather_test.feather'))
    building_meta_df = pd.read_feather(os.path.join(output, 'building_metadata.feather'))
    sample_submission = pd.read_feather(os.path.join(output, 'sample_submission.feather'))


    # # Count zero streak
    train_df = train_df.merge(building_meta_df, on='building_id', how='left')
    train_df = train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')

    train_df['black_count']=0

    for bid in train_df.building_id.unique():
        df = train_df[train_df.building_id==bid]
        for meter in df.meter.unique():
            dfm = df[df.meter == meter]
            b = (dfm.meter_reading == 0).astype(int)
            train_df.loc[(train_df.building_id==bid) & (train_df.meter == meter), 'black_count'] = b.groupby((~b.astype(bool)).cumsum()).cumsum()

    #train_df[train_df.building_id == 0].meter_reading.plot()
    #train_df[train_df.building_id == 0].black_count.plot()


    train_df.to_feather(os.path.join(output, 'train_black.feather'))


if __name__ == '__main__':    
    root = 'input'
    output = 'processed'
    prepare(root, output)

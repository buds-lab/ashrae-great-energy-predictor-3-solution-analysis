#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import _pickle as cPickle
from copy import deepcopy
from datetime import date, datetime, timedelta
import lightgbm as lgb
import pickle
from scipy import stats
import time
from tqdm import tqdm

import os 
code_path = os.path.dirname(os.path.abspath(__file__))

metadata = pd.read_csv(f'{code_path}/../input/building_metadata.csv')
weather_train = pd.read_csv(f'{code_path}/../input/weather_train.csv')

train = pd.read_csv(f'{code_path}/../input/train.csv')
train = train.merge(metadata,on='building_id',how = 'left')
train = train.merge(weather_train,on=['site_id','timestamp'],how='left')

leaked_df = pd.read_feather(f'{code_path}/../external_data/leak.feather')
leaked_df['meter'] = leaked_df['meter'].astype(int)
leaked_df = leaked_df.rename(columns={'meter_reading':'leaked_meter_reading'})

dic = train[['building_id','site_id']]
dic = dic.set_index('building_id').to_dict()
dic = dic['site_id']

leaked_df = leaked_df[leaked_df['leaked_meter_reading'].notnull()]
leaked_df['site_id'] = leaked_df['building_id'].map(dic)

leaked_df = leaked_df.rename(columns={'leaked_meter_reading': 'meter_reading'})

# 覚え書き
# 連続で同じ値を取るやつを除去
# ただし、同じ値を取るやつが最小値だった場合は除去しない(電気データの場合、最小値=休みの日とかの可能性があるため)

del_list = list()

for building_id in range(1449):
    leaked_df_gb = leaked_df[leaked_df['building_id'] == building_id].groupby("meter")

    for meter, tmp_df in leaked_df_gb:
#         print("building_id: {}, meter: {}".format(building_id, meter))
        data = tmp_df['meter_reading'].values
#         splited_value = np.split(data, np.where((data[1:] != data[:-1]) | (data[1:] == min(data)))[0] + 1)
#         splited_date = np.split(tmp_df.timestamp.values, np.where((data[1:] != data[:-1]) | (data[1:] == min(data)))[0] + 1)
        splited_idx = np.split(tmp_df.index.values, np.where((data[1:] != data[:-1]) | (data[1:] == min(data)))[0] + 1)
        for i, x in enumerate(splited_idx):
            if len(x) > 24:
#                 print("length: {},\t{}-{},\tvalue: {}".format(len(x), x[0], x[-1], splited_value[i][0]))
                del_list.extend(x[1:])
                
                
#         print()

del tmp_df, leaked_df_gb

del_list_new = leaked_df.loc[del_list].index#query('timestamp < 20160901').index

leaked_df = leaked_df.drop(del_list_new).reset_index(drop=True)

with open(f'{code_path}/../prepare_data/leak_data_drop_bad_rows.pkl', 'wb') as f:
    pickle.dump(leaked_df,f)

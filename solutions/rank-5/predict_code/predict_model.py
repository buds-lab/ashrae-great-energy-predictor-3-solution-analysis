#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

import _pickle as cPickle
import argparse
from copy import deepcopy
import japanize_matplotlib
import lightgbm as lgb
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error
import time
from tqdm import tqdm

import os 

code_path = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('seed', type=int)
arg('iteration_mul', type=float)
arg('train_file', type=str)
arg('test_file', type=str)
arg('--learning_rate', type=float, default=0.05)
arg('--num_leaves', type=int, default=31)
arg('--n_estimators', type=int, default=500)
args = parser.parse_args()#args=['1', '0.5','train_fe.ftr', 'test_fe.ftr'])

# print(args)

train_fe = pd.read_feather(f'{code_path}/../prepare_data/{args.train_file}')
test_fe = pd.read_feather(f'{code_path}/../prepare_data/{args.test_file}')

target_fe = train_fe['meter_reading']
train_fe = train_fe.drop('meter_reading', axis=1)

X_train = train_fe.query('20160115 <= timestamp < 20160601 & site_id != 0')
X_valid = train_fe.query('20160901 <= timestamp < 20170101 & site_id != 0')
X_test = test_fe

y_train = target_fe.loc[X_train.index]
y_valid = target_fe.loc[X_valid.index]
# y_train = np.log1p(y_train)
# y_valid = np.log1p(y_valid)

X_train = X_train.drop('timestamp', axis=1)
X_valid = X_valid.drop('timestamp', axis=1)
X_test = X_test.drop('timestamp', axis=1)

# print(X_train.shape)

def meter_predict(meter, model, X_test, best_iteration, iteration_mul=1.5):
    X_test_m = X_test.query('meter == {}'.format(meter)).drop('meter', axis=1)
    g = X_test_m.groupby('building_id')
    
    y_pred = []
    for building_id in tqdm(sorted(X_test_m['building_id'].unique())):
        X_building = g.get_group(building_id)
        y_pred.append(pd.Series(model.predict(X_building, n_jobs=4,num_iteration=min(models_all[meter].n_estimators, int(best_iteration[meter][building_id]*iteration_mul))), index=X_building.index))
        
    return pd.concat(y_pred).sort_index()

# load model
load_name = '{}/../model/model_use_{}_seed{}_leave{}_lr{}_tree{}.pkl'.format(code_path, args.train_file.replace('.ftr', ''),args.seed, args.num_leaves, str(args.learning_rate).replace('.', ''), args.n_estimators)
with open(load_name, 'rb') as f:
    models = pickle.load(f)

# with open(f'{code_path}/../model/model_5_95_hokan_cleaning_50000tree_seed{}.pkl'.format(args.seed), 'wb') as f:
#     pickle.dump(models, f)

# 各building, meter毎の最良のiteration数
best_iteration = dict()
for meter in [0,1,2,3]:
    best_iteration[meter] = dict()
#     for i in range(1448):
#         best_iteration[meter][i] = 200
    for i in tqdm(sorted(X_valid.query('meter == {}'.format(meter))['building_id'].unique())):
        best_iteration[meter][i] = max(20, np.argmin(np.array(models[meter].evals_result_[i]['rmse'])) + 1)
#         best_iteration[meter][i] = np.argmin(np.array(models[meter].evals_result_[i]['rmse'])) + 1

del_list = [list(), list(), list(), list()]
for meter in [0,1,2,3]:
    for buildingID, itr in best_iteration[meter].items():
        if itr<=20:
            del_list[meter].append(buildingID)
        if itr<=100:
            best_iteration[meter][buildingID] = 100
#         if itr>=int(models[0].n_estimators * 0.98):
#             best_iteration[meter][buildingID] = models[0].n_estimatorss

for meter in [0,1,2,3]:
    for i in range(1448):
        if i not in best_iteration[meter]:
            best_iteration[meter][i] = 200

#load model
load_name = '{}/../model/model_all_use_{}_seed{}_leave{}_lr{}_tree{}.pkl'.format(code_path, args.train_file.replace('.ftr', ''),args.seed, args.num_leaves, str(args.learning_rate).replace('.', ''), args.n_estimators)
with open(load_name, 'rb') as f:
    models_all = pickle.load(f)

# meter type毎のtestの予測    
preds = list()
for i in tqdm([3,2,1,0]):
    preds.append(meter_predict(i, models_all[i], X_test, best_iteration, iteration_mul=args.iteration_mul))

y_preds = pd.concat(preds).sort_index()

# lgb.plot_importance(models_all[0], importance_type='gain', figsize=(10,20))
# lgb.plot_importance(models_all[0], importance_type='split', figsize=(10,20))

submission = pd.read_csv(f'{code_path}/../input/sample_submission.csv')
submission['meter_reading'] = (np.expm1(y_preds))
submission.loc[submission['meter_reading']<0, 'meter_reading'] = 0

save_name = '{}/../output/use_{}_seed{}_leave{}_lr{}_tree{}_mul{}.csv'.format(code_path, args.train_file.replace('.ftr', ''), args.seed, args.num_leaves, str(args.learning_rate).replace('.', ''), args.n_estimators, str(args.iteration_mul).replace('.', ''))
submission.to_csv(save_name, index=False)

submission.head()



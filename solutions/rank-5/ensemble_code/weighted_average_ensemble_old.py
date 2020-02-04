#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

from copy import deepcopy
from functools import partial
import matplotlib.pyplot as plt
import optuna
import pickle
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import os 
code_path = os.path.dirname(os.path.abspath(__file__))

# leaked_df = pd.read_csv(f'{code_path}/../input/leaked_data_all.csv', parse_dates=['timestamp'])
with open(f'{code_path}\\..\\prepare_data\\leak_data_drop_bad_rows.pkl', 'rb') as f:
    leaked_df = pickle.load(f).rename(columns={'meter_reading': 'leaked_meter_reading'})

# leaked_df = pd.read_feather(f'{code_path}/../input/leak_data.feather').rename(columns={'meter_reading': 'leaked_meter_reading'})
leaked_df = leaked_df[['building_id','meter','timestamp', 'leaked_meter_reading']]
leaked_df = leaked_df.query('timestamp>=20170101')

building_meta = pd.read_csv(f"{code_path}\\..\\input\\building_metadata.csv")

leaked_df = leaked_df.merge(building_meta[['building_id', 'site_id']], on='building_id', how='left')

leaked_df = leaked_df.query('~(meter==0 & site_id==0)')
# leaked_df = leaked_df.query('site_id==[2,4,15]')
# leaked_df = leaked_df.query('105<=building_id<=564 | 656<=building_id')

test = pd.read_csv(f"{code_path}\\..\\input\\test.csv", parse_dates=['timestamp'])

i = 1

for mul in tqdm(['05', '10', '15']):
    submission_s1 = pd.read_csv(f'{code_path}\\..\\output\\use_train_fe_seed1_leave31_lr005_tree500_mul{mul}.csv')
#     submission_s2 = pd.read_csv(f'{code_path}/../output/use_train_fe_seed2_leave31_lr005_tree500_mul{mul}.csv')
#     submission_s3 = pd.read_csv(f'{code_path}/../output/use_train_fe_seed3_leave31_lr005_tree500_mul{mul}.csv')
#     test[f'pred{i}'] = (submission_s1['meter_reading'] + submission_s2['meter_reading'] + submission_s3['meter_reading']) / 3
    test[f'pred{i}'] = submission_s1['meter_reading']
    i += 1
# del submission_s1, submission_s2, submission_s3

# for name in ['fe2_lgbm', 'submission_tomioka', 'submission_half_and_half', 'submission_distill', 'submission_TE_50000tree_seed1_mul075']:
for name in ['submission_half_and_half', 'submission_simple_data_cleanup']:#, 'use_train_fe_seed1_leave15_lr001_tree20000_mul05']:#, 'fe2_lgbm']:
    print(i, end=' ')
    test[f'pred{i}'] = pd.read_csv(f'{code_path}\\..\\external_data\\{name}.csv')['meter_reading']
    i += 1

test[f'pred{i}'] = np.exp(1) - 1
i += 1

test = test.merge(leaked_df, on=['building_id', 'meter', 'timestamp'], how='left')
N = test.columns.str.startswith('pred').sum()
print(N)

test_sub = test.copy()
test = test[~test['leaked_meter_reading'].isnull()]
test2017 = test.query('timestamp<20180101')
test2018 = test.query('20180101<=timestamp')

def preproceeding(submission, N):
    submission.loc[:,'pred1':'leaked_meter_reading'] = np.log1p(submission.loc[:,'pred1':'leaked_meter_reading'])
    g = submission.groupby('meter')
    sub_sub = [dict(), dict(), dict(), dict()]
    leak_sub = [dict(), dict(), dict(), dict()]
    leak_leak = [0,0,0,0]
    
    for meter in [3,2,1,0]:
        for i in tqdm(range(1,N+1)):
            leak_sub[meter][i] = sum(-2 * g.get_group(meter)['leaked_meter_reading'] * g.get_group(meter)[f'pred{i}'])
            for j in range(1,N+1):
                if i > j: 
                    sub_sub[meter][(i,j)] = sub_sub[meter][(j,i)]
                else:
                    sub_sub[meter][(i,j)] = sum(g.get_group(meter)[f'pred{i}'] * g.get_group(meter)[f'pred{j}'])
        
        leak_leak[meter] = (sum(g.get_group(meter)['leaked_meter_reading'] ** 2))
    
    return sub_sub, leak_sub, leak_leak

def optimization(meter, sub_sub, leak_sub, leak_leak, length, W):
    
#     global count_itr
#     if count_itr%1000 == 0: print(count_itr, end=' ')
#     count_itr += 1
    
    loss_total = 0

    for i, a in enumerate(W, 1):
        for j, b in enumerate(W, 1):
            loss_total += a * b * sub_sub[meter][(i, j)]

    for i, a in enumerate(W, 1):
        loss_total += leak_sub[meter][i] * a

    loss_total += leak_leak[meter]
    
    return np.sqrt(loss_total / length)

def make_ensemble_weight(focus_df, N):

    sub_sub, leak_sub, leak_leak = preproceeding(focus_df.copy(), N)


    np.random.seed(1)
    score = [list(), list(), list(), list()]
    weight = [list(), list(), list(), list()]

    for meter in [0,1,2,3]:
        f = partial(optimization, meter, sub_sub, leak_sub, leak_leak, len(focus_df.query(f'meter=={meter}')))
        for i in tqdm(range(1000000)):
            W = np.random.rand(N)

            to_zero = np.arange(N)
            np.random.shuffle(to_zero)

            W[to_zero[:np.random.randint(N)]] = 0
            W /= W.sum()
            W *= np.random.rand() * 0.3 + 0.8
            score[meter].append(f(W))
            weight[meter].append(W)

        score[meter] = np.array(score[meter])
        weight[meter] = np.array(weight[meter])
    
    return weight, score

weight2017, score2017 = make_ensemble_weight(test2017, N)

weight2018, score2018 = make_ensemble_weight(test2018, N)

for meter in [0,1,2,3]:
#     for i in range(N):
    print(weight2017[meter][score2017[meter].argmin()])
    print()

# for meter in [0,1,2,3]:
#     print(score2017[meter].min())
#     print(weight2017[meter][score2017[meter].argmin()].sum())
#     print()

for meter in [0,1,2,3]:
#     for i in range(N):
    print(weight2018[meter][score2018[meter].argmin()])
    print()

# for meter in [0,1,2,3]:
#     print(score2018[meter].min())
#     print(weight2018[meter][score2018[meter].argmin()].sum())
#     print()

def new_pred(test, weight, score, N):
    pred_new = list()
    for meter in [0,1,2,3]:
        test_m = test.query(f'meter=={meter}')
        ensemble_m = sum([np.log1p(test_m[f'pred{i+1}']) * weight[meter][score[meter].argmin()][i] for i in range(N)])
        pred_new.append(ensemble_m)

    pred_new = pd.concat(pred_new).sort_index()
    return np.expm1(pred_new)

ensembled_pred2017 = new_pred(test_sub.query('timestamp<20180101'), weight2017, score2017, N)
ensembled_pred2018 = new_pred(test_sub.query('20180101<=timestamp'), weight2018, score2018, N)

ensembled_pred = pd.concat([ensembled_pred2017, ensembled_pred2018], axis=0).sort_index()

print(np.sqrt(mean_squared_error(np.log1p(test2018['leaked_meter_reading']), np.log1p(ensembled_pred.loc[test2018.index]))))
print(np.sqrt(mean_squared_error(np.log1p(test2017['leaked_meter_reading']), np.log1p(ensembled_pred.loc[test2017.index]))))
print(np.sqrt(mean_squared_error(np.log1p(test['leaked_meter_reading']), np.log1p(ensembled_pred.loc[test.index]))))

new_submission = pd.read_csv(f'{code_path}\\..\\input\\submission.csv')

new_submission['meter_reading'] = ensembled_pred.values

new_submission.to_csv(f'{code_path}\\..\\output\\submission_my_leak_validation.csv', index=False)

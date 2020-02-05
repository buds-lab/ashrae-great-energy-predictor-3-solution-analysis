import gc
import os
from pathlib import Path
import random
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import optuna
from functools import partial

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

def reduce_mem_usage(df, use_float16=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            # skip datetime type or categorical type
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


DATA_PATH='processed'
OUTPUT_PATH='output'

# prediction file to use ensemble
# do not select after leak replacement data
submission_list = [
    'submission_cleanup',
    'submission_kfold',
    'submission_meter'
]


root = Path(DATA_PATH)

train_df = pd.read_feather(root/'train.feather')
test_df = pd.read_feather(root/'test.feather')
building_meta_df = pd.read_feather(root/'building_metadata.feather')


leak_df = pd.read_feather('input/leak.feather')
#leak_df = pd.read_feather('input/leak_null.feather') # drop missing value version.

# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
# Modified to support timestamp type, categorical type
# Modified to add option to use float16 or not. feather format does not support float16.

# load leakdata
leak_df.fillna(0, inplace=True)
leak_df = leak_df[(leak_df.timestamp.dt.year > 2016) & (leak_df.timestamp.dt.year < 2019)]
leak_df.loc[leak_df.meter_reading < 0, 'meter_reading'] = 0 # remove large negative values
leak_df = leak_df[leak_df.building_id!=245]

# load all submissions
for i,f in enumerate(submission_list):
    x = pd.read_csv(f'{OUTPUT_PATH}/{f}.csv', index_col=0).meter_reading
    x[x < 0] = 0
    test_df[f'pred{i}'] = x

del  x
gc.collect()

test_df = reduce_mem_usage(test_df)
leak_df = reduce_mem_usage(leak_df)

leak_df = leak_df.merge(test_df[['building_id', 'meter', 'timestamp', *[f"pred{i}" for i in range(len(submission_list))], 'row_id']], left_on = ['building_id', 'meter', 'timestamp'], right_on = ['building_id', 'meter', 'timestamp'], how = "left")
leak_df = leak_df.merge(building_meta_df[['building_id', 'site_id']], on='building_id', how='left')

leak_df['meter_reading_l1p'] = np.log1p(leak_df.meter_reading)

for i in range(len(submission_list)):
    leak_df[f'pred{i}_l1p'] = np.log1p(leak_df[f"pred{i}"])

for i in range(len(submission_list)):
    leak_score = np.sqrt(mean_squared_error(leak_df[f"pred{i}_l1p"], leak_df.meter_reading_l1p))
    print (f'score{i}=', leak_score)   


leak_df['mean_l1p_pred'] = np.log1p(np.mean(leak_df[[f"pred{i}" for i in range(len(submission_list))]].values, axis=1))
leak_score = np.sqrt(mean_squared_error(leak_df.mean_l1p_pred, leak_df.meter_reading_l1p))
print ('mean score=', leak_score)


class GeneralizedMeanBlender():
    """Combines multiple predictions using generalized mean"""
    def __init__(self, p_range=(0,2)):
        """"""
        self.p_range = p_range
        self.p = None
        self.weights = None
        
        
    def _objective(self, trial, X, y):
                    
        # create hyperparameters
        p = trial.suggest_uniform(f"p", *self.p_range)
        weights = [
            trial.suggest_uniform(f"w{i}", 0, 1)
            for i in range(X.shape[1])
        ]

        # blend predictions
        blend_preds, total_weight = 0, 0
        if p <= 0:
            for j,w in enumerate(weights):
                blend_preds += w*np.log1p(X[:,j])
                total_weight += w
            blend_preds = np.expm1(blend_preds/total_weight)
        else:
            for j,w in enumerate(weights):
                blend_preds += w*X[:,j]**p
                total_weight += w
            blend_preds = (blend_preds/total_weight)**(1/p)
            
        # calculate mean squared error
        return np.sqrt(mean_squared_error(y, blend_preds))

    def fit(self, X, y, n_trials=10): 
        # optimize objective
        obj = partial(self._objective, X=X, y=y)
        study = optuna.create_study()
        study.optimize(obj, n_trials=n_trials)
        # extract best weights
        if self.p is None:
            self.p = [v for k,v in study.best_params.items() if "p" in k][0]
        self.weights = np.array([v for k,v in study.best_params.items() if "w" in k])
        self.weights /= self.weights.sum()

    def transform(self, X): 
        assert self.weights is not None and self.p is not None,\
        "Must call fit method before transform"
        if self.p == 0:
            return np.expm1(np.dot(np.log1p(X), self.weights))
        else:
            return np.dot(X**self.p, self.weights)**(1/self.p)
    
    def fit_transform(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X)

X = leak_df[[f"pred{i}_l1p" for i in range(len(submission_list))]].values
y = leak_df["meter_reading_l1p"].values

gmb = GeneralizedMeanBlender()
gmb.fit(X, y, n_trials=25)


print ('gmb socre=', np.sqrt(mean_squared_error(gmb.transform(X), leak_df.meter_reading_l1p)))

# gmb per meter
gmbs = []
for m in range(4):
    X = leak_df[leak_df.meter==m][[f"pred{i}_l1p" for i in range(len(submission_list))]].values
    y = leak_df[leak_df.meter==m]["meter_reading_l1p"].values

    gmb = GeneralizedMeanBlender()
    gmb.fit(X, y, n_trials=25)
    gmbs.append(gmb)


Xx = []
yy = []
for m in range(4):
    X = leak_df[leak_df.meter==m][[f"pred{i}_l1p" for i in range(len(submission_list))]].values
    y = leak_df[leak_df.meter==m]["meter_reading_l1p"].values
    Xx.append(gmbs[m].transform(X))
    yy.append(y)
    print (m, np.sqrt(mean_squared_error(gmbs[m].transform(X), y)))

#print ('normal gmb', gmb_score)
print ('each gmb total', np.sqrt(mean_squared_error(np.hstack(Xx), np.hstack(yy))))

X_test = test_df[[f"pred{i}" for i in range(len(submission_list))]].values
sample_submission = pd.read_feather(os.path.join(root, 'sample_submission.feather'))

for m in range(4):
    sample_submission.loc[test_df.meter==m, 'meter_reading'] = np.expm1(gmbs[m].transform(np.log1p(X_test[test_df.meter==m])))

sample_submission.loc[sample_submission.meter_reading < 0, 'meter_reading'] = 0

sample_submission.to_csv('output/submission_raw.csv', index=False, float_format='%.4f')

leak_df = leak_df[['meter_reading', 'row_id']].set_index('row_id').dropna()
sample_submission.loc[leak_df.index, 'meter_reading'] = leak_df['meter_reading']

sample_submission.to_csv('output/submission_all_leak.csv', index=False, float_format='%.4f')

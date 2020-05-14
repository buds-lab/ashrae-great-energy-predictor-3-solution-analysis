import os
import time
import json
import string
import random
import itertools
from datetime import datetime
import numpy as np
import pandas as pd
from numba import jit
from sklearn.metrics import mean_squared_error
from contextlib import contextmanager, redirect_stdout
import matplotlib.pyplot as plt

N_TRAIN = 20216100
N_TEST  = 41697600

# load file paths
settings = json.load(open("./settings.json"))
OUTPUT_PATH = settings["OUTPUT_PATH"]
MODEL_PATH = settings["MODEL_PATH"]
DATA_PATH = settings["DATA_PATH"]

PRIMARY_USE_GROUPINGS = [
    ["Education"],
    ["Lodging/residential"],
    ["Office"],
    ["Entertainment/public assembly"],
    ["Public services"],
    ["Other", "Retail", "Parking", "Warehouse/storage",
    "Food sales and service", "Religious worship", "Utility", "Technology/science", 
    "Healthcare", "Manufacturing/industrial", "Services",]
]

def take_first(x): return x.values[0]
def take_last(x): return x.values[-1]

@contextmanager
def timer(name):
    print(f'{datetime.now()} - [{name}] ...')
    t0 = time.time()
    yield
    print(f'{datetime.now()} - [{name}] done in {time.time() - t0:.0f} s\n')
    

def make_dir(dir_name):
    """Create a directory if it doesn"t already exist"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

        
class Logger(object):
    """Save a string line(s) to a file."""
    
    def __init__(self, file_path, mode="w", verbose=False):
        self.file_path = file_path
        self.verbose = verbose
        open(file_path, mode=mode)
        
    def append(self, line, print_line=None):
        if print_line or self.verbose:
            print(line)
        with open(self.file_path, "a") as f:
            with redirect_stdout(f):
                print(line)        

        
@jit(nopython=True)
def find_zero_streaks(x):
    n = len(x)
    streaks = np.zeros(n)
    
    if x[0] == 0:
        streaks[0] = 1
        
    for i in range(1,n):
        if x[i] == 0:
            streaks[i] = streaks[i-1] + 1
    return streaks

def find_zero_streaks_wrapper(x):
    return find_zero_streaks(x.values)


@jit(nopython=True) 
def find_constant_values(x, min_constant_values=6):
    i = 0
    j = i + 1
    n = len(x)
    ignore_values = np.zeros(n)
    while j < n:
        if x[i] == x[j]:
            k = j+1
            while k < n and x[i] == x[k]:
                k += 1        
            if k-1-i > min_constant_values:
                ignore_values[i+1:k] = 1
            i = k
        else:
            i += 1
        j = i + 1    
    return ignore_values==1


def rmsle(x,y):
    x = np.log1p(x)
    y = np.log1p(y)
    return np.sqrt(mean_squared_error(x, y))


def plot_feature_importance(model, feature_cols):
    importance_df = pd.DataFrame(
        model.feature_importance(),
        index=feature_cols,
        columns=['importance']).sort_values('importance')
    fig, ax = plt.subplots(figsize=(8, 8))
    importance_df.plot.barh(ax=ax)
    fig.show()


def get_validation_months(n_months):
    validation_months_list = [np.arange(i+1,i+2+n_months-1)
                              for shift in range(n_months)
                              for i in range(shift,12+shift, n_months)]
    validation_months_list = [(x-1) % 12 + 1 for x in validation_months_list]
    return validation_months_list
    

def reduce_mem_usage(df, skip_cols=[], verbose=False):
    """ Reduce memory usage in a pandas dataframe

    Based on this great kernel:
    https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
    """
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in np.setdiff1d(df.columns, skip_cols):
        if df[col].dtype != object:  # Exclude strings            

            # print column type
            if verbose:
                print("******************************")
                print("Column: ",col)
                print("dtype before: ",df[col].dtype)            

            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            if verbose:
                print("min for this col: ",mn)
                print("max for this col: ",mx)

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True            

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
            
            if verbose:
                print("dtype after: ",df[col].dtype)
                print("******************************")

    # Print final result
    if verbose:
        print("___MEMORY USAGE AFTER COMPLETION:___")
        mem_usg = df.memory_usage().sum() / 1024**2 
        print("Memory usage is: ",mem_usg," MB")
        print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist


def load_data(data_name):
    """Loads and formats data"""
    
    # raw
    if data_name == "train":
        return pd.read_csv(f"{DATA_PATH}/train.csv")

    if data_name == "test":
        return pd.read_csv(f"{DATA_PATH}/test.csv") 

    if data_name == "input":
        return load_data("train"), load_data("test")

    # clean
    if data_name == "train_clean":
        return pd.read_pickle(f"{DATA_PATH}/preprocessed/train_clean.pkl")

    if data_name == "test_clean":
        return pd.read_pickle(f"{DATA_PATH}/preprocessed/test_clean.pkl")        

    if data_name == "clean":
        return load_data("train_clean"), load_data("test_clean")

    # nn meter
    if data_name == "train_nn_meter":
        return pd.read_pickle(f"{DATA_PATH}/preprocessed/train_nn_meter.pkl")

    if data_name == "test_nn_meter":
        return pd.read_pickle(f"{DATA_PATH}/preprocessed/test_nn_meter.pkl")        

    if data_name == "nn_meter":
        return load_data("train_nn_meter"), load_data("test_nn_meter")        

    # nn target normalized meter
    if data_name == "train_nn_target_normalized_meter":
        return pd.read_pickle(f"{DATA_PATH}/preprocessed/train_nn_target_normalized_meter.pkl")

    if data_name == "test_nn_target_normalized_meter":
        return pd.read_pickle(f"{DATA_PATH}/preprocessed/test_nn_target_normalized_meter.pkl")        

    if data_name == "nn_target_normalized_meter":
        return load_data("train_nn_target_normalized_meter"), load_data("test_nn_target_normalized_meter")

    # nn site
    if data_name == "train_nn_site":
        return pd.read_pickle(f"{DATA_PATH}/preprocessed/train_nn_site.pkl")

    if data_name == "test_nn_site":
        return pd.read_pickle(f"{DATA_PATH}/preprocessed/test_nn_site.pkl")        

    if data_name == "nn_site":
        return load_data("train_nn_site"), load_data("test_nn_site")        

    # nn target normalized site
    if data_name == "train_nn_target_normalized_site":
        return pd.read_pickle(f"{DATA_PATH}/preprocessed/train_nn_target_normalized_site.pkl")

    if data_name == "test_nn_target_normalized_site":
        return pd.read_pickle(f"{DATA_PATH}/preprocessed/test_nn_target_normalized_site.pkl")        

    if data_name == "nn_target_normalized_site":
        return load_data("train_nn_target_normalized_site"), load_data("test_nn_target_normalized_site")

    # debug 1000
    if data_name == "train_clean_debug_1000":
        return pd.read_pickle(f"{DATA_PATH}/preprocessed/train_clean_debug_1000.pkl")

    if data_name == "test_clean_debug_1000":
        return pd.read_pickle(f"{DATA_PATH}/preprocessed/test_clean_debug_1000.pkl")        

    if data_name == "clean_debug_1000":
        return load_data("train_clean_debug_1000"), load_data("test_clean_debug_1000")

    if data_name == "leak_debug_1000":
        return pd.read_pickle(f"{DATA_PATH}/preprocessed/leak_debug_1000.pkl")

    # debug 10000
    if data_name == "train_clean_debug_10000":
        return pd.read_pickle(f"{DATA_PATH}/preprocessed/train_clean_debug_10000.pkl")

    if data_name == "test_clean_debug_10000":
        return pd.read_pickle(f"{DATA_PATH}/preprocessed/test_clean_debug_10000.pkl")        

    if data_name == "clean_debug_10000":
        return load_data("train_clean_debug_10000"), load_data("test_clean_debug_10000")

    if data_name == "leak_debug_10000":
        return pd.read_pickle(f"{DATA_PATH}/preprocessed/leak_debug_10000.pkl")        

    # raw weather
    if data_name == "train_weather":
        return pd.read_csv(f"{DATA_PATH}/weather_train.csv")

    if data_name == "test_weather":
        return pd.read_csv(f"{DATA_PATH}/weather_test.csv")        

    if data_name == "weather":
        return load_data("train_weather"), load_data("test_weather")

    # leak
    if data_name == "leak":
        return pd.read_feather(f"{DATA_PATH}/leak.feather")

    # leak
    if data_name == "is_leak":
        return pd.read_feather(f"{DATA_PATH}/is_leak.feather")        
        
    # rows to drop
    if data_name == "bad_meter_readings":
        return pd.read_csv(f"{DATA_PATH}/bad_meter_readings.csv")

    # meta
    if data_name == "meta":
        return pd.read_csv(f"{DATA_PATH}/building_metadata.csv")

    # submissions
    if data_name == "sample_submission":
        return pd.read_csv(f"{DATA_PATH}/sample_submission.csv")

    # meta
    if data_name == "best_submission":
        return pd.read_csv(f"{DATA_PATH}/submissions/final_average_top4.csv")


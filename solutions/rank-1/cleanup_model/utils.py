import pandas as pd
import numpy as np
import os
import warnings
import time

from datetime import datetime
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_squared_log_error
from scipy.signal import savgol_filter as sg
from contextlib import contextmanager, redirect_stdout

pd.set_option("max_columns", 500)

@contextmanager
def timer(name):
    print(f'{datetime.now()} - [{name}] ...')
    t0 = time.time()
    yield
    print(f'{datetime.now()} - [{name}] done in {time.time() - t0:.0f} s\n')

def add_sg(df):
    w = 11
    p = 2
    for si in df.site_id.unique():
        index = df.site_id == si
        df.loc[index, 'air_smooth'] = sg(df[index].air_temperature, w, p)
        df.loc[index, 'dew_smooth'] = sg(df[index].dew_temperature, w, p)
        
        df.loc[index, 'air_diff'] = sg(df[index].air_temperature, w, p, 1)
        df.loc[index, 'dew_diff'] = sg(df[index].dew_temperature, w, p, 1)
        
        df.loc[index, 'air_diff2'] = sg(df[index].air_temperature, w, p, 2)
        df.loc[index, 'dew_diff2'] = sg(df[index].dew_temperature, w, p, 2)

        
def make_is_bad_zero(Xy_subset, min_interval=48, summer_start=3000, summer_end=7500):
    """Helper routine for 'find_bad_zeros'.
    
    This operates upon a single dataframe produced by 'groupby'. We expect an 
    additional column 'meter_id' which is a duplicate of 'meter' because groupby 
    eliminates the original one."""
    meter = Xy_subset.meter_id.iloc[0]
    is_zero = Xy_subset.meter_reading == 0
    if meter == 0:
        # Electrical meters should never be zero. Keep all zero-readings in this table so that
        # they will all be dropped in the train set.
        return is_zero

    transitions = (is_zero != is_zero.shift(1))
    all_sequence_ids = transitions.cumsum()
    ids = all_sequence_ids[is_zero].rename("ids")
    if meter in [2, 3]:
        # It's normal for steam and hotwater to be turned off during the summer
        keep = set(ids[(Xy_subset.timestamp < summer_start) |
                       (Xy_subset.timestamp > summer_end)].unique())
        is_bad = ids.isin(keep) & (ids.map(ids.value_counts()) >= min_interval)
    elif meter == 1:
        time_ids = ids.to_frame().join(Xy_subset.timestamp).set_index("timestamp").ids
        is_bad = ids.map(ids.value_counts()) >= min_interval

        # Cold water may be turned off during the winter
        jan_id = time_ids.get(0, False)
        dec_id = time_ids.get(8283, False)
        if (jan_id and dec_id and jan_id == time_ids.get(500, False) and
                dec_id == time_ids.get(8783, False)):
            is_bad = is_bad & (~(ids.isin(set([jan_id, dec_id]))))
    else:
        raise Exception(f"Unexpected meter type: {meter}")

    result = is_zero.copy()
    result.update(is_bad)
    return result


def find_bad_zeros(X, y):
    """Returns an Index object containing only the rows which should be deleted."""
    Xy = X.assign(meter_reading=y, meter_id=X.meter)
    is_bad_zero = Xy.groupby(["building_id", "meter"]).apply(make_is_bad_zero)
    return is_bad_zero[is_bad_zero].index.droplevel([0, 1])


def find_bad_sitezero(X):
    """Returns indices of bad rows from the early days of Site 0 (UCF)."""
    return X[(X.timestamp < 3378) & (X.site_id == 0) & (X.meter == 0)].index

#find_bad_building1099 identifies the most absurdly high readings from building 1099. These are orders of magnitude higher than all data, and have been emperically seen in LB probes to be harmful outliers.
#
def find_bad_building1099(X, y):
    """Returns indices of bad rows (with absurdly high readings) from building 1099."""
    return X[(X.building_id == 1099) & (X.meter == 2) & (y > 3e4)].index


def find_bad_rows(X, y):
    return find_bad_zeros(X, y).union(find_bad_sitezero(X)).union(find_bad_building1099(X, y))


def input_file(file):
    path = f"../input/{file}"
    if not os.path.exists(path): return path + ".gz"
    return path

def compress_dataframe(df):
    result = df.copy()
    for col in result.columns:
        col_data = result[col]
        dn = col_data.dtype.name
        if dn == "object":
            result[col] = pd.to_numeric(col_data.astype("category").cat.codes, downcast="integer")
        elif dn == "bool":
            result[col] = col_data.astype("int8")
        elif dn.startswith("int") or (col_data.round() == col_data).all():
            result[col] = pd.to_numeric(col_data, downcast="integer")
        else:
            result[col] = pd.to_numeric(col_data, downcast='float')
    return result

def read_train():
    df = pd.read_csv(input_file("train.csv"), parse_dates=["timestamp"])
    df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
    return compress_dataframe(df)


def read_building_metadata():
    return compress_dataframe(pd.read_csv(
        input_file("building_metadata.csv")).fillna(-1)).set_index("building_id")

site_GMT_offsets = [-5, 0, -7, -5, -8, 0, -5, -5, -5, -6, -7, -5, 0, -6, -5, -5]


def read_weather_train(fix_timestamps=True, interpolate_na=True, add_na_indicators=True):
    df = pd.read_csv(input_file("weather_train.csv"), parse_dates=["timestamp"])
    df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
    if fix_timestamps:
        GMT_offset_map = {site: offset for site, offset in enumerate(site_GMT_offsets)}
        df.timestamp = df.timestamp + df.site_id.map(GMT_offset_map)
    if interpolate_na:
        site_dfs = []
        for site_id in df.site_id.unique():
            # Make sure that we include all possible hours so that we can interpolate evenly
            site_df = df[df.site_id == site_id].set_index("timestamp").reindex(range(8784))
            site_df.site_id = site_id
            for col in [c for c in site_df.columns if c != "site_id"]:
                if add_na_indicators: site_df[f"had_{col}"] = ~site_df[col].isna()
                site_df[col] = site_df[col].interpolate(limit_direction='both', method='linear')
                # Some sites are completely missing some columns, so use this fallback
                site_df[col] = site_df[col].fillna(df[col].median())
            site_dfs.append(site_df)
        df = pd.concat(site_dfs).reset_index()  # make timestamp back into a regular column
    elif add_na_indicators:
        for col in df.columns:
            if df[col].isna().any(): df[f"had_{col}"] = ~df[col].isna()

    df = compress_dataframe(df)
    add_sg(df)
    
    return compress_dataframe(df).set_index(["site_id", "timestamp"])


def combined_train_data(fix_timestamps=True, interpolate_na=True, add_na_indicators=True):
    Xy = compress_dataframe(read_train().join(read_building_metadata(), on="building_id").join(
        read_weather_train(fix_timestamps, interpolate_na, add_na_indicators),
        on=["site_id", "timestamp"]).fillna(-1))
    
    # site -0 correction
    Xy.loc[(Xy.site_id == 0) & (Xy.meter==0), 'meter_reading'] =  Xy.loc[(Xy.site_id == 0) & (Xy.meter==0), 'meter_reading'] * 0.2931
    return Xy.drop(columns=["meter_reading"]), Xy.meter_reading


def add_time_features(X):
    return X.assign(tm_day_of_week=((X.timestamp // 24) % 7), tm_hour_of_day=(X.timestamp % 24))


class CatSplitRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model, col):
        self.model = model
        self.col = col

    def fit(self, X, y):
        self.fitted = {}
        importances = []
        for val in X[self.col].unique():
            X1 = X[X[self.col] == val].drop(columns=[self.col])
            self.fitted[val] = clone(self.model).fit(X1, y.reindex_like(X1))
            importances.append(self.fitted[val].feature_importances_)
            del X1
        fi = np.average(importances, axis=0)
        col_index = list(X.columns).index(self.col)
        self.feature_importances_ = [*fi[:col_index], 0, *fi[col_index:]]
        return self

    def predict(self, X):
        result = np.zeros(len(X))
        for val in X[self.col].unique():
            ix = np.nonzero((X[self.col] == val).values)
            predictions = self.fitted[val].predict(X.iloc[ix].drop(columns=[self.col]))
            result[ix] = predictions
        return result

categorical_columns = [
    "building_id", "meter", "site_id", "primary_use", "had_air_temperature", "had_cloud_coverage",
    "had_dew_temperature", "had_precip_depth_1_hr", "had_sea_level_pressure", "had_wind_direction",
    "had_wind_speed", "tm_day_of_week", "tm_hour_of_day"
]


class LGBMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, categorical_feature=None, **params):
        self.model = LGBMRegressor(**params)
        self.categorical_feature = categorical_feature

    def fit(self, X, y):
        with warnings.catch_warnings():
            cats = None if self.categorical_feature is None else list(
                X.columns.intersection(self.categorical_feature))
            warnings.filterwarnings("ignore",
                                    "categorical_feature in Dataset is overridden".lower())
            self.model.fit(X, y, **({} if cats is None else {"categorical_feature": cats}))
            self.feature_importances_ = self.model.feature_importances_
            return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {**self.model.get_params(deep), "categorical_feature": self.categorical_feature}

    def set_params(self, **params):
        ctf = params.pop("categorical_feature", None)
        if ctf is not None: self.categorical_feature = ctf
        self.model.set_params(params)


def read_test():
    df = pd.read_csv(input_file("test.csv"), parse_dates=["timestamp"])
    df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
    return compress_dataframe(df).set_index("row_id")


def read_weather_test(fix_timestamps=True, interpolate_na=True, add_na_indicators=True):
    df = pd.read_csv(input_file("weather_test.csv"), parse_dates=["timestamp"])
    df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
    if fix_timestamps:
        GMT_offset_map = {site: offset for site, offset in enumerate(site_GMT_offsets)}
        df.timestamp = df.timestamp + df.site_id.map(GMT_offset_map)
    if interpolate_na:
        site_dfs = []
        for site_id in df.site_id.unique():
            # Make sure that we include all possible hours so that we can interpolate evenly
            site_df = df[df.site_id == site_id].set_index("timestamp").reindex(range(8784, 26304))
            site_df.site_id = site_id
            for col in [c for c in site_df.columns if c != "site_id"]:
                if add_na_indicators: site_df[f"had_{col}"] = ~site_df[col].isna()
                site_df[col] = site_df[col].interpolate(limit_direction='both', method='linear')
                # Some sites are completely missing some columns, so use this fallback
                site_df[col] = site_df[col].fillna(df[col].median())
            site_dfs.append(site_df)
        df = pd.concat(site_dfs).reset_index()  # make timestamp back into a regular column
    elif add_na_indicators:
        for col in df.columns:
            if df[col].isna().any(): df[f"had_{col}"] = ~df[col].isna()
    df = compress_dataframe(df)
    add_sg(df)
    return compress_dataframe(df).set_index(["site_id", "timestamp"])


def combined_test_data(fix_timestamps=True, interpolate_na=True, add_na_indicators=True):
    X = compress_dataframe(read_test().join(read_building_metadata(), on="building_id").join(
        read_weather_test(fix_timestamps, interpolate_na, add_na_indicators),
        on=["site_id", "timestamp"]).fillna(-1))
    return X


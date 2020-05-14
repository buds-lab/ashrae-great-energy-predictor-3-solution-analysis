import gc
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from ashrae.utils import DATA_PATH, timer, load_data, reduce_mem_usage
from ashrae.encoders import GaussianTargetEncoder


# define groupings and corresponding priors
groups_and_priors = {
    
    # singe encodings
    ("hour",):        None,
    ("weekday",):     None,
    ("month",):       None,
    ("building_id",): None,
    ("primary_use",): None,
    ("site_id",):     None,    
    ("meter",):       None,
    
    # second-order interactions
    ("meter", "hour"):        ["gte_meter", "gte_hour"],
    ("meter", "weekday"):     ["gte_meter", "gte_weekday"],
    ("meter", "month"):       ["gte_meter", "gte_month"],
    ("meter", "building_id"): ["gte_meter", "gte_building_id"],
    ("meter", "primary_use"): ["gte_meter", "gte_primary_use"],
    ("meter", "site_id"):     ["gte_meter", "gte_site_id"],
        
    # higher-order interactions with building_id
    ("meter", "building_id", "hour"):    ["gte_meter_building_id", "gte_meter_hour"],
    ("meter", "building_id", "weekday"): ["gte_meter_building_id", "gte_meter_weekday"],
    ("meter", "building_id", "month"):   ["gte_meter_building_id", "gte_meter_month"],
    
}


def process_timestamp(df): 
    df.timestamp = pd.to_datetime(df.timestamp)
    df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600            


def process_weather(df, dataset, fix_timestamps=True, interpolate_na=True, add_na_indicators=True):
    if fix_timestamps:
        site_GMT_offsets = [-5, 0, -7, -5, -8, 0, -5, -5, -5, -6, -7, -5, 0, -6, -5, -5]
        GMT_offset_map = {site: offset for site, offset in enumerate(site_GMT_offsets)}
        df.timestamp = df.timestamp + df.site_id.map(GMT_offset_map)

    if interpolate_na:
        site_dfs = []
        for site_id in df.site_id.unique():
            # Make sure that we include all possible hours so that we can interpolate evenly
            if dataset == "train":
                site_df = df[df.site_id == site_id].set_index("timestamp").reindex(range(8784))
            elif dataset == "test":
                site_df = df[df.site_id == site_id].set_index("timestamp").reindex(range(8784, 26304))
            else: 
                raise ValueError(f"dataset={dataset} not recognized")
            site_df.site_id = site_id
            for col in [c for c in site_df.columns if c != "site_id"]:
                if add_na_indicators: site_df[f"had_{col}"] = ~site_df[col].isna()
                site_df[col] = site_df[col].interpolate(limit_direction='both', method='spline', order=3,)
                # Some sites are completely missing some columns, so use this fallback
                site_df[col] = site_df[col].fillna(df[col].median())
            site_dfs.append(site_df)
        df = pd.concat(site_dfs).reset_index()  # make timestamp back into a regular column

    if add_na_indicators:
        for col in df.columns:
            if df[col].isna().any(): df[f"had_{col}"] = ~df[col].isna()

    return df.fillna(-1) # .set_index(["site_id", "timestamp"])


def add_lag_feature(df, window=3, group_cols="site_id", lag_cols=["air_temperature"]):
    rolled = df.groupby(group_cols)[lag_cols].rolling(window=window, min_periods=0, center=True)
    lag_mean = rolled.mean().reset_index().astype(np.float16)
    lag_max = rolled.quantile(0.95).reset_index().astype(np.float16)
    lag_min = rolled.quantile(0.05).reset_index().astype(np.float16)
    lag_std = rolled.std().reset_index().astype(np.float16)
    
    for col in lag_cols:
        df[f"{col}_mean_lag{window}"] = lag_mean[col]
        df[f"{col}_max_lag{window}"] = lag_max[col]
        df[f"{col}_min_lag{window}"] = lag_min[col]
        df[f"{col}_std_lag{window}"] = lag_std[col]
        
        
def add_features(df):
    # time features
    df["hour"] = df.ts.dt.hour
    df["weekday"] = df.ts.dt.weekday
    df["month"] = df.ts.dt.month
    df["year"] = df.ts.dt.year    
    
    # time interactions
    df["weekday_hour"] = df.weekday.astype(str) + "-" + df.hour.astype(str)
    
    # apply cyclic encoding of periodic features
    df["hour_x"] = np.cos(2*np.pi*df.timestamp/24)
    df["hour_y"] = np.sin(2*np.pi*df.timestamp/24)
    
    df["month_x"] = np.cos(2*np.pi*df.timestamp/(30.4*24))
    df["month_y"] = np.sin(2*np.pi*df.timestamp/(30.4*24))
    
    df["weekday_x"] = np.cos(2*np.pi*df.timestamp/(7*24))
    df["weekday_y"] = np.sin(2*np.pi*df.timestamp/(7*24))
            
    # meta data features
    df["year_built"] = df["year_built"]-1900
    
    # bulding_id interactions
    bm_ = df.building_id.astype(str) + "-" + df.meter.astype(str) + "-" 
    df["building_weekday_hour"] = bm_ + df.weekday_hour
    df["building_weekday"]      = bm_ + df.weekday.astype(str)
    df["building_month"]        = bm_ + df.month.astype(str)
    df["building_hour"]         = bm_ + df.hour.astype(str)    
    df["building_meter"]        = bm_

    # get holidays
    dates_range = pd.date_range(start="2015-12-31", end="2019-01-01")
    us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())    
    df["is_holiday"] = (df.ts.dt.date.astype("datetime64").isin(us_holidays)).astype(np.int8)   
    
if __name__ == "__main__":

    with timer("Loading data"):
        train, test = load_data("input")
        building_meta = load_data("meta")
        train_weather, test_weather = load_data("weather")
 
    with timer("Process timestamp"):
        train["ts"] = pd.to_datetime(train.timestamp)
        test["ts"] = pd.to_datetime(test.timestamp)
        process_timestamp(train)
        process_timestamp(test)
        process_timestamp(train_weather)
        process_timestamp(test_weather)

    with timer("Process weather"):
        process_weather(train_weather, "train")
        process_weather(test_weather, "test")
        
        for window_size in [7, 73]:
            add_lag_feature(train_weather, window=window_size)
            add_lag_feature(test_weather, window=window_size)

    with timer("Combine data"):
        train = pd.merge(train, building_meta, "left", "building_id")
        train = pd.merge(train, train_weather, "left", ["site_id", "timestamp"])

        test = pd.merge(test, building_meta, "left", "building_id")
        test = pd.merge(test, test_weather, "left", ["site_id", "timestamp"])    
    
    with timer("Flag bad meter readings"):
        is_bad_meter_reading = load_data("bad_meter_readings").values
        train["is_bad_meter_reading"] = is_bad_meter_reading

    with timer("Correct site 0 meter reading"):
        train.loc[(train.site_id == 0) & (train.meter==0), "meter_reading"] *= 0.2931

    with timer("Add base features to train"):
        add_features(train)
    
    with timer("Add base features to test"):
        add_features(test)
    
    with timer("Free up memory"):
        del train_weather, test_weather, building_meta
        gc.collect()
        
    with timer("Reduce memory usage"):
        train, _ = reduce_mem_usage(train, skip_cols=['ts', 'timestamp'], verbose=False)
        test, _ = reduce_mem_usage(test, skip_cols=['ts', 'timestamp'], verbose=False)
        

    with timer("Add target encoding features - train"):
        train["target"] = np.log1p(train.meter_reading)
        test["target"] = np.mean(train["target"])
        
        features = []
        good_train = train[train.is_bad_meter_reading.values==0].copy()
        good_train_ = good_train.copy()
        for group_cols, prior_cols in groups_and_priors.items():
            print(group_cols)
            features.append(f"gte_{'_'.join(group_cols)}")
            gte = GaussianTargetEncoder(list(group_cols), "target", prior_cols)
            good_train[features[-1]] = gte.fit_transform(good_train)
            train[features[-1]] = gte.transform(train)

    with timer("Save as pickle - train"):
        train.drop(["ts", "target"], 1, inplace=True)
        train, _ = reduce_mem_usage(train, skip_cols=['ts', 'timestamp'], verbose=False)
        train.to_pickle(f"{DATA_PATH}/preprocessed/train_clean.pkl")
                            
    with timer("Free up memory"):
        del train, good_train, gte
        gc.collect()

    with timer("Add target encoding features - test"):    
        features = []
        good_train = good_train_
        for group_cols, prior_cols in groups_and_priors.items():
            print(group_cols)
            features.append(f"gte_{'_'.join(group_cols)}")
            gte = GaussianTargetEncoder(list(group_cols), "target", prior_cols)
            good_train[features[-1]] = gte.fit_transform(good_train)
            test[features[-1]] = gte.transform(test)
            
    with timer("Free up memory"):
        del good_train, good_train_, gte
        gc.collect()
            
    with timer("Save as pickle - test"):            
        test.drop(["ts", "target"], 1, inplace=True)
        test, _ = reduce_mem_usage(test, skip_cols=['ts', 'timestamp'], verbose=False)
        gc.collect()
        test.to_pickle(f"{DATA_PATH}/preprocessed/test_clean.pkl")                           
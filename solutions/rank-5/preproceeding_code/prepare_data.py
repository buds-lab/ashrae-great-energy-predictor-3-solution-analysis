#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import argparse
from copy import deepcopy
from datetime import date, datetime, timedelta
from tqdm import tqdm
import os 

code_path = os.path.dirname(os.path.abspath(__file__))

# args
parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--train_file', type=str, default='train.csv')
arg('--test_file', type=str, default='test.csv')
arg('--weather_train_file', type=str, default='weather_train.csv')
arg('--weather_test_file', type=str, default='weather_test.csv')
args = parser.parse_args()
print(args)


# read data

train = pd.read_csv(f'{code_path}/../input/{args.train_file}', parse_dates=['timestamp'])
test = pd.read_csv(f'{code_path}/../input/{args.test_file}', parse_dates=['timestamp'])
building_meta = pd.read_csv(f'{code_path}/../input/building_metadata.csv')
weather_train = pd.read_csv(f'{code_path}/../input/{args.weather_train_file}', parse_dates=['timestamp'])
weather_test = pd.read_csv(f'{code_path}/../input/{args.weather_test_file}', parse_dates=['timestamp'])


# drop bad rows

del_list = list()

for building_id in range(1449):
    train_gb = train[train['building_id'] == building_id].groupby("meter")

    for meter, tmp_df in train_gb:
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

del tmp_df, train_gb


def idx_to_drop(df):
    drop_cols = []
    electric_zero = df[(df['meter']==0)&(df['meter_reading']==0)].index.values.tolist()
    drop_cols.extend(electric_zero)
    not_summer = df[(df['timestamp'].dt.month!=7)&(df['timestamp'].dt.month!=8)]
    not_summer['cumsum'] = not_summer.groupby(['building_id','meter'])['meter_reading'].cumsum()
    not_summer['shifted'] = not_summer.groupby(['building_id','meter'])['cumsum'].shift(48)
    not_summer['difference'] = not_summer['cumsum']-not_summer['shifted']
    steam_zero = not_summer[(not_summer['difference']==0) & (not_summer['meter']==2)].index.values.tolist()
    hotwater_zero = not_summer[(not_summer['difference']==0) & (not_summer['meter']==3)].index.values.tolist()
    drop_cols.extend(steam_zero)
    drop_cols.extend(hotwater_zero)
    del not_summer
    not_winter = train[(df['timestamp'].dt.month!=12)&(df['timestamp'].dt.month!=1)]
    not_winter['cumsum'] = not_winter.groupby(['building_id','meter'])['meter_reading'].cumsum()
    not_winter['shifted'] = not_winter.groupby(['building_id','meter'])['cumsum'].shift(48)
    not_winter['difference'] = not_winter['cumsum']-not_winter['shifted']
    chilled_zero = not_winter[(not_winter['difference']==0) & (not_winter['meter']==1)].index.values.tolist()
    drop_cols.extend(chilled_zero)
    return drop_cols

del_list.extend(idx_to_drop(train))



del_list_new = train.loc[del_list].index#query('timestamp < 20160901').index

train = train.drop(del_list_new)

train = train.query('(not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")) & (not (meter==0 & meter_reading==0))')



train['meter_reading'] = np.log1p(train['meter_reading'])
train = train.reset_index(drop=True)



# Correct time
weather = pd.concat([weather_train, weather_test], axis=0).reset_index(drop=True)


country = ['UnitedStates', 'England', 'UnitedStates', 'UnitedStates', 'UnitedStates',
           'England', 'UnitedStates', 'Canada', 'UnitedStates', 'UnitedStates',
           'UnitedStates', 'Canada', 'Ireland', 'UnitedStates', 'UnitedStates', 'UnitedStates']

city = ['Jacksonville', 'London', 'Phoenix', 'Philadelphia', 'San Francisco',
       'Loughborough', 'Philadelphia', 'Montreal', 'Jacksonville', 'San Antonio',
       'Las Vegas', 'Montreal', 'Dublin', 'Minneapolis', 'Philadelphia', 'Pittsburgh']

UTC_offset = [-4, 0, -7, -4, -9, 0, -4, -4, -4, -5, -7, -4, 0, -5, -4, -4]

location_data = pd.DataFrame(np.array([country, city, UTC_offset]).T, index=range(16), columns=['country', 'city', 'UTC_offset'])


for idx in location_data.index:
    weather.loc[weather['site_id']==idx, 'timestamp'] += timedelta(hours=int(location_data.loc[idx, 'UTC_offset']))



# complement weather data
def fill_weather_dataset(weather_df):
    
    # Find Missing Dates
    time_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.strptime(weather_df['timestamp'].min(),time_format)
    end_date = datetime.strptime(weather_df['timestamp'].max(),time_format)
    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
    hours_list = [(end_date - timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]

    missing_hours = []
    for site_id in range(16):
        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])
        new_rows = pd.DataFrame(np.setdiff1d(hours_list,site_hours),columns=['timestamp'])
        new_rows['site_id'] = site_id
        weather_df = pd.concat([weather_df,new_rows])

        weather_df = weather_df.reset_index(drop=True)           

    # Add new Features
    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])
    weather_df["day"] = weather_df["datetime"].dt.day
    weather_df["week"] = weather_df["datetime"].dt.week
    weather_df["month"] = weather_df["datetime"].dt.month
    
    # Reset Index for Fast Update
    weather_df = weather_df.set_index(['site_id','day','month'])

    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])
    weather_df.update(air_temperature_filler,overwrite=False)

    # Step 1
    cloud_coverage_filler = weather_df.groupby(['site_id','day','month'])['cloud_coverage'].mean()
    # Step 2
    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"])

    weather_df.update(cloud_coverage_filler,overwrite=False)

    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])
    weather_df.update(due_temperature_filler,overwrite=False)

    # Step 1
    sea_level_filler = weather_df.groupby(['site_id','day','month'])['sea_level_pressure'].mean()
    # Step 2
    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'),columns=['sea_level_pressure'])

    weather_df.update(sea_level_filler,overwrite=False)

    wind_direction_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])
    weather_df.update(wind_direction_filler,overwrite=False)

    wind_speed_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_speed'].mean(),columns=['wind_speed'])
    weather_df.update(wind_speed_filler,overwrite=False)

    # Step 1
    precip_depth_filler = weather_df.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()
    # Step 2
    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'),columns=['precip_depth_1_hr'])

    weather_df.update(precip_depth_filler,overwrite=False)

    weather_df = weather_df.reset_index()
    weather_df = weather_df.drop(['datetime','day','week','month'],axis=1)
        
    return weather_df

weather['timestamp'] = weather['timestamp'].astype(str)
weather = fill_weather_dataset(weather)
weather['timestamp'] = pd.to_datetime(weather['timestamp'])




# holiday imformation

import holidays

en_holidays = holidays.England()
ir_holidays = holidays.Ireland()
ca_holidays = holidays.Canada()
us_holidays = holidays.UnitedStates()

en_idx = weather.query('site_id == 1 or site_id == 5').index
ir_idx = weather.query('site_id == 12').index
ca_idx = weather.query('site_id == 7 or site_id == 11').index
us_idx = weather.query('site_id == 0 or site_id == 2 or site_id == 3 or site_id == 4 or site_id == 6 or site_id == 8 or site_id == 9 or site_id == 10 or site_id == 13 or site_id == 14 or site_id == 15').index

weather['IsHoliday'] = 0
weather.loc[en_idx, 'IsHoliday'] = weather.loc[en_idx, 'timestamp'].apply(lambda x: en_holidays.get(x, default=0))
weather.loc[ir_idx, 'IsHoliday'] = weather.loc[ir_idx, 'timestamp'].apply(lambda x: ir_holidays.get(x, default=0))
weather.loc[ca_idx, 'IsHoliday'] = weather.loc[ca_idx, 'timestamp'].apply(lambda x: ca_holidays.get(x, default=0))
weather.loc[us_idx, 'IsHoliday'] = weather.loc[us_idx, 'timestamp'].apply(lambda x: us_holidays.get(x, default=0))

holiday_idx = weather['IsHoliday'] != 0
weather.loc[holiday_idx, 'IsHoliday'] = 1
weather['IsHoliday'] = weather['IsHoliday'].astype(np.uint8)




target = train['meter_reading'].values
# train = train.drop('meter_reading', axis=1)
row_id = test['row_id']
test = test.drop('row_id', axis=1)

df = pd.concat([train.drop('meter_reading', axis=1), test], axis=0).reset_index(drop=True)
df = df.merge(building_meta, on='building_id', how='left')

df = df.merge(weather, on=['site_id', 'timestamp'], how='left')

df['day'] = df['timestamp'].dt.day #// 3
df['hour'] = df['timestamp'].dt.hour
df['weekday'] = df['timestamp'].dt.weekday

train = df.iloc[:len(target)].copy().reset_index(drop=True)
train['meter_reading'] = target#.values
test = df.iloc[len(target):].copy().reset_index(drop=True)



df['is_day_off_or_holiday'] = (df['weekday'] >= 5) | df['IsHoliday']


# target_encoding 1
def make_fraction(col1, col2):
    col2_frac = train.groupby([col1, col2])[['meter_reading']].median()
    col2_frac_idx = col2_frac.index
    col2_frac_sum = col2_frac.groupby(col1).sum().rename(columns = {'meter_reading':'sum'})
    col2_frac = col2_frac.merge(col2_frac_sum, on = col1, how='left')
    col2_frac.index = col2_frac_idx
    col2_frac['frac_{}_{}'.format(col1, col2)] = col2_frac['meter_reading'] / col2_frac['sum']
    col2_frac = col2_frac[['frac_{}_{}'.format(col1, col2)]]
    return col2_frac



building_weekday_frac = make_fraction('building_id', 'weekday')
building_hour_frac = make_fraction('building_id', 'hour')
building_day_frac = make_fraction('building_id', 'day')

primary_weekday_frac = make_fraction('primary_use', 'weekday')
primary_hour_frac = make_fraction('primary_use', 'hour')
primary_day_frac = make_fraction('primary_use', 'day')


df = df.merge(building_weekday_frac, on=['building_id', 'weekday'], how='left')
df = df.merge(building_hour_frac, on=['building_id', 'hour'], how='left')
df = df.merge(building_day_frac, on=['building_id', 'day'], how='left')

df = df.merge(primary_weekday_frac, on=['primary_use', 'weekday'], how='left')
df = df.merge(primary_hour_frac, on=['primary_use', 'hour'], how='left')
df = df.merge(primary_day_frac, on=['primary_use', 'day'], how='left')


# target encoding 2
# 95 percentile for each building_id
building_meter_95 = train.groupby(['building_id', 'meter'])['meter_reading'].apply(lambda arr: np.percentile(arr, 95)).rename('building_meter_95')
df = df.merge(building_meter_95, on=['building_id', 'meter'], how='left')

# 5 percentile for each building_id
building_meter_5 = train.groupby(['building_id', 'meter'])['meter_reading'].apply(lambda arr: np.percentile(arr, 5)).rename('building_meter_5')
df = df.merge(building_meter_5, on=['building_id', 'meter'], how='left')


### transform part of features into categorical feature
is_categorical = ['meter', 'building_id', 'site_id', 'primary_use', 'hour', 'day', 'weekday']
df[is_categorical] = df[is_categorical].astype('category')
df['year_built_cat'] = df['year_built'].astype('category')



drop_columns = []#, 'hour', 'day', 'weekday']
drop_df = df[drop_columns]
df = df.drop(drop_columns, axis=1)


train_fe = df.iloc[:len(target)].copy().reset_index(drop=True)
train_fe['meter_reading'] = target#.values
test_fe = df.iloc[len(target):].copy().reset_index(drop=True)

# save features
train_fe.to_feather(f'{code_path}/../prepare_data/train_fe.ftr')
test_fe.to_feather(f'{code_path}/../prepare_data/test_fe.ftr')


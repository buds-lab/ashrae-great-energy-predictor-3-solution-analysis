#!/usr/bin/env python
# coding: utf-8
import argparse
from utils import *

params={
    'n_estimators':100,
    'num_leaves':700, #31
    'learning_rate':0.1,
}

INPUT_DIR = '../input/'
OUTPUT_DIR = '../output/'
MODEL_PATH  = '../models/'

drops= ["timestamp", 'wind_direction', 'wind_speed']

parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true', help='debug mode')
args = parser.parse_args()

def train(debug=True):
    
    with timer("Preprocesing"):
        X, y = combined_train_data()

        bad_rows = find_bad_rows(X, y)
        #pd.Series(bad_rows.sort_values()).to_csv("rows_to_drop.csv", header=False, index=False)

        X = X.drop(index=bad_rows)
        y = y.reindex_like(X)

        # Additional preprocessing
        X = compress_dataframe(add_time_features(X))
        X = X.drop(columns=drops)  # Raw timestamp doesn't help when prediction
        y = np.log1p(y)

#    X.head()

    with timer("Training"):
        model = CatSplitRegressor(
            LGBMWrapper(**params, random_state=0, n_jobs=-1, categorical_feature=categorical_columns), "meter")
        model.fit(X, y)
        
    del X, y

    if not debug:
        import pickle
        with open(MODEL_PATH + 'cleanup_model.pickle', mode='wb') as f:
            pickle.dump([model], f)
        del model


if __name__ == '__main__':
    debug = args.debug
    print ('debug=', debug)
    train(debug)
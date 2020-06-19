import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

import argparse
import keras
import numpy as np 
import pandas as pd 

from ashrae.utils import (
    MODEL_PATH,  timer, make_dir, rmsle,
    load_data, get_validation_months,
)

parser = argparse.ArgumentParser(description="")

parser.add_argument("--overwrite", action="store_true", 
    help="If True then overwrite existing files")

parser.add_argument("--normalize_target", action="store_true", 
    help="If True then normalize the meter_reading by dividing by log1p(square_feet).")

FEATURES = [
    # building meta features
    "square_feet", "year_built", "floor_count",
    
    # cat cols
    "building_id", "site_id", "primary_use", 
    "hour", "weekday", "weekday_hour",
    "building_weekday_hour", "building_weekday",
    "building_hour", 
    
    # raw weather features
    "air_temperature", "cloud_coverage", "dew_temperature",
    "precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed",
    
    # derivative weather features
    "air_temperature_mean_lag7", "air_temperature_max_lag7",
    "air_temperature_min_lag7", "air_temperature_std_lag7",
    "air_temperature_mean_lag73", "air_temperature_max_lag73",
    "air_temperature_min_lag73", "air_temperature_std_lag73",
    
    # time features
    "hour_x", "hour_y", "weekday_x", "weekday_y", "is_holiday",
    
    # target encoding features
    "gte_meter_building_id_hour", "gte_meter_building_id_weekday",
]

CAT_COLS = [
    "building_id", "site_id", "primary_use", 
    "hour", "weekday", "weekday_hour",
    "building_weekday_hour", "building_weekday",
    "building_hour", 
]

NUM_COLS = [x for x in FEATURES if x not in CAT_COLS]


def get_inputs(df):
    inputs = {col: np.array(df[col]) for col in CAT_COLS}
    inputs["numerical_inputs"]  = df[NUM_COLS].values    
    return inputs, df.target.values

def train_mlp(
    train,
    valid,
    cat_counts, 
    save_name,
    lr=1e-3,
    lr_decay=1e-4,
    batch_size=512,
    epochs=25,
    emb_l2_reg=1e-3,
    emb_dim=1,
    n_dense_max=256,
    n_dense_min=32,
    n_layers=3,
    dropout=0.5):
            
    #-------------------------
    with timer("Create  model"):        
        
        # inputs
        num_inputs = keras.layers.Input(shape=(len(NUM_COLS),), name="numerical_inputs")
        cat_inputs = [keras.layers.Input(shape=(1,), name=x) for x in CAT_COLS]

        # embedding
        emb_inputs = []
        for x,i in zip(cat_counts, cat_inputs):
            emb = keras.layers.Embedding(
                input_dim=cat_counts[x],
                output_dim=emb_dim,
                embeddings_regularizer=keras.regularizers.l2(emb_l2_reg))
            emb = keras.layers.Flatten()(emb(i))
            emb_inputs.append(emb)

        # mlp
        inputs = keras.layers.Concatenate(name="general_features")([num_inputs, *emb_inputs])
        for i in range(n_layers):
            n_dense = int(max((0.5**i)*n_dense_max, n_dense_min))
            inputs = keras.layers.Dense(n_dense, activation="relu")(inputs)
            inputs = keras.layers.Dropout(dropout)(inputs)    
            inputs = keras.layers.BatchNormalization()(inputs)

        # output
        outputs = keras.layers.Dense(1, activation=None, name="outputs")(inputs)
        model = keras.models.Model(
            inputs = [num_inputs, *cat_inputs],
            outputs = outputs
        )

        # compile
        model.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.Adam(lr=lr, decay=lr_decay)
        )
        
        model.summary()
        
    #-------------------------
    msg = f'Training {save_name} - train# {train.shape} val# {valid.shape}'
    with timer(msg):
        model.fit(
            *get_inputs(train), 
            batch_size=batch_size,
            epochs=epochs,
            validation_data=get_inputs(valid),
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=2,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    save_name, # f"{MODEL_PATH}/model_oof.hdf5"
                    save_best_only=True,
                    verbose=1,
                    monitor='val_loss',
                    mode='min'
                )
            ]
        )
    return

if __name__ == "__main__":
    """
    python scripts/03_train_mlp_meter.py --normalize_target
    python scripts/03_train_mlp_meter.py
    """
    
    args = parser.parse_args()
    
    with timer("Loading data"):
        train = load_data("train_nn_meter")
        train = train.loc[train.is_bad_meter_reading==0].reset_index(drop=True)
        train.loc[(train.meter == 0) & (train.site_id == 0), "meter_reading"] *= 0.2931
        if args.normalize_target:
            square_feet = load_data("train_clean")["square_feet"]
            train["target"] = np.log1p(train["target"]/square_feet)
        else:
            train["target"] = np.log1p(train["target"])
        
    with timer("Preprocesing"):
        meter_cat_counts = train.groupby(["meter"])[CAT_COLS].agg(lambda x: len(np.unique(x)))

    # get base file name
    model_name = f"mlp-split_meter"
    make_dir(f"{MODEL_PATH}/{model_name}")

    with timer("Training"):
        for seed in [0]:
            for n_months in [1,2,3,4]:
                validation_months_list = get_validation_months(n_months)

                for fold_, validation_months in enumerate(validation_months_list):    
                    for m in range(4):    

                        # create sub model path
                        if args.normalize_target:
                            sub_model_path = f"{MODEL_PATH}/{model_name}/target_normalization/meter_{m}"
                            make_dir(sub_model_path)
                        else:
                            sub_model_path = f"{MODEL_PATH}/{model_name}/no_normalization/meter_{m}"
                            make_dir(sub_model_path)

                        # create model version
                        model_version = "_".join([
                            str(seed), str(n_months), str(fold_), 
                        ])    

                        # check if we can skip this model
                        full_sub_model_name = f"{sub_model_path}/{model_version}.h5"
                        if os.path.exists(full_sub_model_name):
                            if not args.overwrite:
                                break

                        # get this months indices
                        trn_idx = np.where(np.isin(train.month, validation_months, invert=True))[0]
                        val_idx = np.where(np.isin(train.month, validation_months, invert=False))[0]
                        #rint(f"split meter: train size {len(trn_idx)} val size {len(val_idx)}")

                        # remove indices not in this meter
                        trn_idx = np.intersect1d(trn_idx, np.where(train.meter == m)[0])
                        val_idx = np.intersect1d(val_idx, np.where(train.meter == m)[0])
                        #rint(f"split meter: train size {len(trn_idx)} val size {len(val_idx)}")

                        # fit model
                        train_mlp(
                            train = train.loc[trn_idx, FEATURES+["target"]],
                            valid = train.loc[val_idx, FEATURES+["target"]],
                            cat_counts = dict(meter_cat_counts.loc[m]),
                            save_name = full_sub_model_name
                        )
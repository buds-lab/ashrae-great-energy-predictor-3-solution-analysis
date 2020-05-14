#!/bin/bash

# Isamu ---------------
# train 3 models

# meter_model
cd meter_split_model
python train.py &> train.log
cd ..

# kfold model
cd kfold_model
python train.py &> train.log
cd ..

# cleanup model
cd cleanup_model
python train.py &> train.log
cd ..

# Matt ---------------
python scripts/03_train_cb_meter.py --normalize_target
python scripts/03_train_cb_meter.py

python scripts/03_train_cb_primary_use.py --normalize_target
python scripts/03_train_cb_primary_use.py

python scripts/03_train_cb_site.py --normalize_target
python scripts/03_train_cb_site.py

python scripts/03_train_lgb_meter.py --normalize_target
python scripts/03_train_lgb_meter.py

python scripts/03_train_lgb_primary_use.py --normalize_target
python scripts/03_train_lgb_primary_use.py

python scripts/03_train_lgb_site.py --normalize_target
python scripts/03_train_lgb_site.py

python scripts/03_train_mlp_meter.py --normalize_target
python scripts/03_train_mlp_meter.py
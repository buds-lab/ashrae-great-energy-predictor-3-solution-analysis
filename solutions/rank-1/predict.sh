#!/bin/bash
# run 3 models

# Isamu ---------------
# meter_model
cd meter_split_model
python predict.py &> predict.log
cd ..

# kfold model
cd kfold_model
python predict.py &> predict.log
cd ..

# cleanup model
cd cleanup_model
python predict.py &> predict.log
cd ..

# Matt ---------------
python scripts/04_predict_cb_meter.py --normalize_target
python scripts/04_predict_cb_meter.py

python scripts/04_predict_cb_primary_use.py --normalize_target
python scripts/04_predict_cb_primary_use.py

python scripts/04_predict_cb_site.py --normalize_target
python scripts/04_predict_cb_site.py

python scripts/04_predict_lgb_meter.py --normalize_target
python scripts/04_predict_lgb_meter.py

python scripts/04_predict_lgb_primary_use.py --normalize_target
python scripts/04_predict_lgb_primary_use.py

python scripts/04_predict_lgb_site.py --normalize_target
python scripts/04_predict_lgb_site.py

python scripts/04_predict_mlp_meter.py --normalize_target
python scripts/04_predict_mlp_meter.py
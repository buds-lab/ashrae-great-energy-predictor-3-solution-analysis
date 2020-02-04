#!/usr/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)

for mul in 0.5 1 1.5 ; do echo $mul ; done | xargs -P6 -I{} $python $SCRIPT_DIR/predict_model.py 1 {} train_fe.ftr test_fe.ftr

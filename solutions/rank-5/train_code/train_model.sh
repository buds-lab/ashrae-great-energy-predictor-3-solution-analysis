#!/usr/bin/bash 

SCRIPT_DIR=$(cd $(dirname $0); pwd)

python $SCRIPT_DIR/train_model.py 1

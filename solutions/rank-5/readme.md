Hello!

Below you can find a outline of how to reproduce my solution for the ASHRAE competition.
If you run into any trouble with the setup/code or have any questions please contact me at sano.tatsuya.sw@alumni.tsukuba.ac.jp

### ARCHIVE CONTENTS
 - model                     : model binaries used in generating solution
 - output                   : model predictions
 - train_code                  : code to rebuild models from scratch
 - predict_code                : code to generate predictions from model binaries
 - ensemble_code               : code to ensemble predictions
 - preproceeding_code          : code to preproceeding
 - prepare_data                : preproceeding data
 - external_data               : leak_data(leak.feather, got from https://www.kaggle.com/yamsam/ashrae-leak-data-station) and other competitor's submission file. We used other competitor's submit files from
   1. https://www.kaggle.com/purist1024/ashrae-simple-data-cleanup-lb-1-08-no-leaks  
   2. https://www.kaggle.com/rohanrao/ashrae-half-and-half
and these submit file (`submission.csv`) renamed to `submission_simple_data_cleanup.csv` and `submission_half_and_half.csv` respectively.

### HARDWARE 
The following specs were used to create the original solution
 - Ubuntu 16.04.5 LTS
 - Intel Xeon Gold 6126 @ 2.60GHz x2(12Core/24Thread, Skylake)
 - DDR4-2666 DIMM ECC REG 32GB x12 = 384GB

### SOFTWARE 
 - Python 3.6.8
 
 Python packages are detailed separately in `requirements.txt`

### DATA SETUP 

We assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed. 
Below are the shell commands used in each step, as run from the top level directory

```
mkdir input
cd input
kaggle competitions download -c ashrae-energy-prediction
unzip ashrae-energy-prediction.zip
rm ashrae-energy-prediction.zip
cd ../
```

### DATA PROCESSING

The train/predict code will also call this script if it has not already been run on the relevant data.

```
python ./preproceeding_code/prepare_data.py
python ./ensemble_code/leak_data_drop_bad_rows.py
python ./preproceeding_code/prepare_data_simplified.py
```


### MODEL BUILD 

There are three options to produce the solution.
 1) very fast prediction
    a) runs in about 5 minutes
    b) uses binary model files
 2) ordinary prediction
    a) run in about 20~25 minutes(about 8 minutes for prediction, and about 15minutes for ensemble)
    b) uses binary model files
 3) retrain models
    a) run in about 10 minutes
    b) trains all models from scratch
    c) follow this with (2) to produce entire solution from scratch

Shell command to run each build is below

 1) very fast prediction (overwrites output/use_train_fe_seed1_leave31_lr005_tree500_mul05.csv)
 
    `python ./predict_code/predict_model.py 1 0.5 train_fe.ftr test_fe.ftr`

 2) ordinary prediction (overwrites predictions in output directory)
 
    `sh predict_code/predict_model.sh`
    `python ./ensemble_code/weighted_average_ensemble.py`
 
    (after run command, it generate submission.csv and this file achieve 1.047 Public and 1.236 Private)

 3) retrain models (overwrites models in model directory)
 
    `sh ./train_code/train_model.sh`

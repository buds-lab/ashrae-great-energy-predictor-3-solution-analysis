# ashrae-energy-prediction3
1st place solution (only my part) in kaggle competition ashrae-eneger-prediction3

## Preparation

### Data
Download all data from https://www.kaggle.com/c/ashrae-energy-prediction/data and unzip and move all csv files into 'input' directory.

Alternatively, run this script
```
./init.sh
```

For faster inference times download the models and output from our [shared google drive](https://drive.google.com/drive/folders/1E0Ua1zoJ8fGppSAnS8_fFviZDoyn_EOG).

### Install the local ashrae Python package
Be sure to add the path of the `ashrae` directory your `~/.bashrc` file. This library contains useful helper functions for this competition. 

### Leak Data
Download leak data from https://www.kaggle.com/yamsam/ashrae-leak-data-station/output and move it into 'input' direcotry.
Download leak data (drop missing value version) from https://www.kaggle.com/yamsam/ashrae-leak-data-station-drop-null/output and move it into 'input' direcotry.


### Preprocess
Run ```python prepare_data.py``` to make feather format data and preprocesed data into 'processed' directory. 

## Models

### cleanup_model
This model was based on the public kernel(https://www.kaggle.com/purist1024/ashrae-simple-data-cleanup-lb-1-08-no-leaks).
I added some features and tuned hyper parameter parameters using local validation(leak).
This kernel output following files 
 * 'output/submission_cleanup.csv'
 * 'output/submission_replaced_cleanup.csv'(replaced by leak data)

### kfold_model
This model was based on the public kernel(https://www.kaggle.com/aitude/ashrae-kfold-lightgbm-without-leak-1-08). 
I added some features and data cleaning and tuned hyper parameters using local validation(leak used).
This kernel output following files 
 * 'output/submission_kfold.csv'
 * 'output/submission_replaced_kfold.csv'(replaced by leak data)

### meter_split_model
This is based on the public kernel(https://www.kaggle.com/corochann/ashrae-training-lgbm-by-meter-type).
I added some features and data cleaning and tuned hyper parameters using local validation(leak used).
This kernel output following files 
 * 'output/submission_meter.csv'
 * 'output/submission_replaced_meter.csv'(replaced by leak)

## Setup
I used kaggle kernel to run my code during the competition so I recomend you to use kaggle docker to run my code.
https://github.com/Kaggle/docker-python

## How to use

### train models
Trains and saves the models. 
```
./train.sh
```

### predict result
Makes predictions with the trained models
```
./predict.sh
```

### ensemble
Takes the predictions in output and ensembles them into the final submission.
```
./ensemble.sh
```

This the a general purpose ensembling script that you can use if you have other predictions for emsembling, for example predictions from public kernel, move prediction file to 'output' directory and add filename into submission_list in ensemble.py

```python emsemble.py``` 

This script output following files 
 * 'output/submission_raw.csv'
 * 'output/submission_all_leak.csv'(replaced by leak)

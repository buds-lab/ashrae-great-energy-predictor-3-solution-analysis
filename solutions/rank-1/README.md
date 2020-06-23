This first rank solution scored 1.231 on the private leaderboard and 0.938 (rank 14) on the public leaderboard. The technical detail of this solution can be found in this [Kaggle post](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/124709). We describe here the code structure and step-by-step instructions on how to reproduce this solution from scratch.

## Directory structure
 
 - `ashrae` - ashrae Python package directory with utility functions
 - `cleanup_model` - contains code for cleanup_model
 - `ensemble.py` - model ensembling script
 - `ensemble.sh` - shell script to invoke the ensembling script
 - `ensemble_meter.py` - model ensembling script
 - `init.sh` - shell script to download the competition data and to pre-process it
 - `input` - contains input and pre-processed datasets
 - `kfold_model` - contains code for kfold_model 
 - `meter_split_model` - contains code for meter_split_model
 - `models` - contains all trained model files
 - `output` - contains all model output files
 - `predict.sh` - shell script to invoke all model prediction scripts
 - `prepare_data.py` - data pre-processing code
 - `processed` - contains all pre-processed datasets
 - `requirements.txt` - list of python packages
 - `scripts` - contains train and predict scripts
 - `settings.json` - settings file 
 - `train.sh ` - shell script to invoke all model training scripts
 
 
## System configuration and setup
The following are the hardware and software specifications of the system on which this solution was reproduced.

### Hardware 
The hardware specifications of the system are:
  - Cloud Computing Service: AWS EC2 
  - Instance Type: [g4dn.4xlarge (16 vCPUs, 64 GB RAM, and 600 GB disk)](https://aws.amazon.com/ec2/instance-types/g4/)
  - AMI: Deep Learning AMI (Ubuntu 18.04) Version 27.0 - ami-0a7789c77135a5f8a

The details on how to launch a Deep Learning AMI can be found [here](https://aws.amazon.com/getting-started/hands-on/get-started-dlami/). After launching, we connected to the instance using `PuTTY`. The details can be found [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html).

### Software
The Deep Learning AMI comes with pre-installed Anaconda-based Python environments and GPU software for developing deep learning models. We used the `tensorflow_p36` environment to reproduce this solution. 

#### 1. Activating the `tensorflow_p36` environment

We will start by activating the `tensorflow_p36` environment.

```
ubuntu@ip-172-31-29-254:~$ source activate tensorflow_p36
WARNING: First activation might take some time (1+ min).
Installing TensorFlow optimized for your Amazon EC2 instance......
Env where framework will be re-installed: tensorflow_p36
Instance g4dn.4xlarge is identified as a GPU instance, removing tensorflow-serving-cpu
Installation complete.
(tensorflow_p36) ubuntu@ip-172-31-29-254:~$
```

Next, we will check the `Python` and `pip` versions.

```
(tensorflow_p36) ubuntu@ip-172-31-29-254:~$ python --version
Python 3.6.6 :: Anaconda, Inc.
(tensorflow_p36) ubuntu@ip-172-31-29-254:~$ pip --version
pip 19.3.1 from /home/ubuntu/anaconda3/envs/tensorflow_p36/lib/python3.6/site-packages/pip (python 3.6)
```

Next, let's also check the GPU details using the `nvidia-smi` command.

```
(tensorflow_p36) ubuntu@ip-172-31-29-254:~$ nvidia-smi
Mon Apr 25 16:24:09 2020
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 435.21       Driver Version: 435.21       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   65C    P0    30W /  70W |      0MiB / 15109MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

We can see that this AWS EC2 instance has a `Tesla T4` GPU with 16 GB memory. 

#### 2. Installation of Kaggle API
The [Kaggle API](https://github.com/Kaggle/kaggle-api) is a python package that enables accessing competition datasets using a command-line interface. This is required for us to download the original competition dataset and the output files from third-party kernels. 

We install the Kaggle API using `pip`

```
(tensorflow_p36) ubuntu@ip-172-31-29-254:~$ pip install kaggle
```

Let's make sure it works properly and verify the version.
```
(tensorflow_p36) ubuntu@ip-172-31-29-254:~$ kaggle --version
Kaggle API 1.5.6
```
After the installation, we need to set up the API credentials file `kaggle.json`. The details can be found [here](https://github.com/Kaggle/kaggle-api).



## Reproducing the solution

The following are the steps to reproducing this solution:
 - Download the solution and set up the input datasets
 - Data pre-processing and feature engineering
 - Model training
 - Model prediction
 - Model ensembling  

We explain these steps and the required commands in detail below. 

#### 1. Download the solution and input datasets

We clone the GitHub repository using the `git` command.

```
(tensorflow_p36) ubuntu@ip-172-31-29-254:~$ pwd
/home/ubuntu
(tensorflow_p36) ubuntu@ip-172-31-29-254:~$ git clone git://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis.git
```

The contents of `rank-1` directory should be looking like this.
```
(tensorflow_p36) ubuntu@ip-172-31-29-254:~$ cd ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-1
(tensorflow_p36) ubuntu@ip-172-31-29-254:~/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-1$ ls -l
total 96
-rw-rw-r-- 1 ubuntu ubuntu  3046 Apr 25 16:48 README.md
drwxrwxr-x 2 ubuntu ubuntu  4096 Apr 25 16:48 ashrae
drwxrwxr-x 2 ubuntu ubuntu  4096 Apr 25 16:48 cleanup_model
-rw-rw-r-- 1 ubuntu ubuntu  7680 Apr 25 16:48 ensemble.py
-rw-rw-r-- 1 ubuntu ubuntu    62 Apr 25 16:48 ensemble.sh
-rw-rw-r-- 1 ubuntu ubuntu  8219 Apr 25 16:48 ensemble_meter.py
-rw-rw-r-- 1 ubuntu ubuntu   143 Apr 25 16:48 init.sh
drwxrwxr-x 2 ubuntu ubuntu  4096 Apr 25 16:48 input
drwxrwxr-x 2 ubuntu ubuntu  4096 Apr 25 16:48 kfold_model
drwxrwxr-x 2 ubuntu ubuntu  4096 Apr 25 16:48 meter_split_model
drwxrwxr-x 2 ubuntu ubuntu  4096 Apr 25 16:48 models
drwxrwxr-x 2 ubuntu ubuntu  4096 Apr 25 16:48 output
-rw-rw-r-- 1 ubuntu ubuntu   893 Apr 25 16:48 predict.sh
-rw-rw-r-- 1 ubuntu ubuntu  2931 Apr 25 16:48 prepare_data.py
drwxrwxr-x 2 ubuntu ubuntu  4096 Apr 25 16:48 processed
-rw-rw-r-- 1 ubuntu ubuntu 11414 Apr 25 16:48 requirements.txt
drwxrwxr-x 2 ubuntu ubuntu  4096 Apr 25 16:48 scripts
-rw-rw-r-- 1 ubuntu ubuntu    94 Apr 25 16:48 settings.json
-rw-rw-r-- 1 ubuntu ubuntu   956 Apr 25 16:48 train.sh

```

Next, we download the Kaggle competition dataset using the `kaggle` command and extract them into the `input` directory.

```
(tensorflow_p36) $ pwd
/home/ubuntu/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-1/
(tensorflow_p36) $ mkdir input
(tensorflow_p36) $ cd input
(tensorflow_p36) $ kaggle competitions download -c ashrae-energy-prediction
(tensorflow_p36) $ unzip ashrae-energy-prediction.zip
(tensorflow_p36) $ ls -l
total 2937920
-rw-rw-r-- 1 ubuntu ubuntu  397104766 Apr 21 18:58 ashrae-energy-prediction.zip
-rw-rw-r-- 1 ubuntu ubuntu      45527 Oct 10  2019 building_metadata.csv
-rw-rw-r-- 1 ubuntu ubuntu  447562511 Oct 10  2019 sample_submission.csv
-rw-rw-r-- 1 ubuntu ubuntu 1462461085 Oct 10  2019 test.csv
-rw-rw-r-- 1 ubuntu ubuntu  678616640 Oct 10  2019 train.csv
-rw-rw-r-- 1 ubuntu ubuntu   14787908 Oct 10  2019 weather_test.csv
-rw-rw-r-- 1 ubuntu ubuntu    7450075 Oct 10  2019 weather_train.csv
(tensorflow_p36) $ cd ../

```

Alternatively, we can run this script `./scripts/01_get_data.sh`.


Next, we download leaked datasets from [this](https://www.kaggle.com/yamsam/ashrae-leak-data-station/output) and [this](https://www.kaggle.com/yamsam/ashrae-leak-data-station-drop-null/output) kernels and stored them into the `input` directory using the `kaggle` command.


```
(tensorflow_p36) $ cd input
(tensorflow_p36) $ pwd
/home/ubuntu/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-2/
(tensorflow_p36) $ kaggle kernels output yamsam/ashrae-leak-data-station
(tensorflow_p36) $ kaggle kernels output yamsam/ashrae-leak-data-station-drop-null

```

This creates `leak.feather` and `leak_null.feather` files into the `input` directory.


Finally, we need to add the local `ashrae` library path to `PYTHONPATH` environment variable. The `ashrae` library contains useful helper functions. 
```
(tensorflow_p36) ubuntu@ip-172-31-29-254:~$ export PYTHONPATH=/home/ubuntu/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-1
```

#### 2. Data pre-processing and feature engineering

In this step, data cleaning and feature engineering are done. We need to execute the below three scripts.

   1. `python prepare_data.py` - This script creates feather format files and pre-processed data into the `processed` directory. These files are used by the `meter_split_model`, `kfold_model`, and `cleanup_model`. This script ran for 9 minutes.
   
   2. `python scripts/02_preprocess_data.py` - This script creates the cleaned train and test dataset for the CatBoost and LightGBM models within the `input/preprocesed` directory. This script ran for 34 minutes.
   
   3. `python scripts/02_preprocess_nn_data.py` - This script creates the cleaned train and test dataset for the MLP models within the `input/preprocesed` directory. This script ran for 91 minutes.


#### 3. Model training

This solution consists of 17 models. There are separate python script used to train these models.

   1. `meter_split_model` - This model is based on [this public kernel](https://www.kaggle.com/corochann/ashrae-training-lgbm-by-meter-type). It trains four LightGBM models, one each per meter, 
and creates `meter_split_model.pickle` within the `models` directory. It ran for 7 minutes.

   ```
   (tensorflow_p36) $ pwd
   /home/ubuntu/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-1/   
   (tensorflow_p36) $ cd meter_split_model
   (tensorflow_p36) $ python train.py
   (tensorflow_p36) $ cd ..
   ``` 
   
   2. `kfold_model` - This is a 3-fold LightGBM model using the entire training set. It creates `kfold_model.pickle` within the `models` directory.  This script ran for 8 minutes.
   ```
   (tensorflow_p36) $ cd kfold_model
   (tensorflow_p36) $ python train.py
   (tensorflow_p36) $ cd ..
   ```

   3. `cleanup_model` - This is also a LightGBM model trained on cleaned training set with different categorical features. 
It creates `cleanup_model.pickle` within the `models` directory. This script ran for 4 minutes.

   ```
   (tensorflow_p36) $ cd cleanup_model
   (tensorflow_p36) $ python train.py
   (tensorflow_p36) $ cd ..
   ```
   

##### 3.1 CatBoost models   

   Next, separate CatBoost models are trained on a different subset of data, split by meter, primary use, and site. A 12-fold cross-validation method was used (9 consecutive months as training and the remaining 3 months as validation sets). Further two separate models are trained: one without using the normalized target variable and another one with the normalized target variable. 
   
   We need to execute the below commands one-by-one from the `rank-1` directory.    
   
   4. `python scripts/03_train_cb_meter.py --normalize_target` - This script trains 48 models (4 meters x 12-folds) on different training subsets using normalized target variable. It creates 48 individual model files within the `models/cb-split_meter/target_normalization/` directory. This script ran for 18 hours and 53 minutes. 
   
   5. `python scripts/03_train_cb_meter.py` - This script trains 48 models (4 meters x 12-folds) on different training subsets. It creates 48 individual model files within the `models/cb-split_meter/no_normalization/` directory. This script ran for 20 hours and 15 minutes. 
   
   6. `python scripts/03_train_cb_primary_use.py --normalize_target` - This script trains 72 models (6 primary use x 12-folds) on different training subsets using normalized target variable. It creates 72 individual model files within the `models/cb-split_primary_use/target_normalization/` directory. This script ran for 17 hours and 29 minutes. 
   
   7. `python scripts/03_train_cb_primary_use.py` - This script trains 72 models (6 primary use x 12-folds) on different training subsets with using normalized target variable. It creates 72 individual model files within the `models/cb-split_primary_use/no_normalization/` directory. This script ran for 17 hours and 44 minutes. 
   
   8. `python scripts/03_train_cb_site.py --normalize_target` - This script trains 192 models (16 sites x 12-folds) on different training subsets using normalized target variable. It creates 72 individual model files within the `models/cb-split_site/target_normalization/` directory. This script ran for 12 hours and 33 minutes. 
   
   9. `python scripts/03_train_cb_site.py` - This script trains 192 models (16 sites x 12-folds) on different training subsets. It creates 192 individual model files within the `models/cb-split_site/no_normalization/` directory. This script ran for 12 hours and 42 minutes. 
   
##### 3.2 LightGBM models 

   Next, separate LightGBM models are trained on a different subset of data, split by meter, primary use, and site. A 12-fold cross-validation method was used (9 consecutive months as training and the remaining 3 months as validation sets). Further two separate models are trained: one without using the normalized target variable and another one with the normalized target variable. 
   
   10. `python scripts/03_train_lgb_meter.py --normalize_target` - This script trains 48 models (4 meters x 12-folds) on different training subsets using normalized target variable. It creates 48 individual model files within the `models/lgb-split_meter/target_normalization/` directory. This script ran for 82 minutes. 
   
   11. `python scripts/03_train_lgb_meter.py` - This script trains 48 models (4 meters x 12-folds) on different training subsets. It creates 48 individual model files within the `models/lgb-split_meter/no_normalization/` directory. This script ran for 117 minutes. 
   
   12. `python scripts/03_train_lgb_primary_use.py --normalize_target` - This script trains 72 models (6 primary use x 12-folds) on different training subsets using normalized target variable. It creates 72 individual model files within the `models/lgb-split_primary_use/target_normalization/` directory. This script ran for 129 minutes. 
   
   13. `python scripts/03_train_lgb_primary_use.py` - This script trains 72 models (6 primary use x 12-folds) on different training subsets. It creates 72 individual model files within the `models/lgb-split_primary_use/no_normalization/` directory. This script ran for 95 minutes. 
   
   14. `python scripts/03_train_lgb_site.py --normalize_target` - This script trains 192 models (16 sites x 12-folds) on different training subsets using normalized target variable. It creates 72 individual model files within the `models/lgb-split_site/target_normalization/` directory. This script ran for 125 minutes. 
   
   15. `python scripts/03_train_lgb_site.py` - This script trains 192 models (16 sites x 12-folds) on different training subsets. It creates 192 individual model files within the `models/lgb-split_site/no_normalization/` directory. This script ran for 153 minutes. 
   
##### 3.3 Multilayer Perceptron (MLP) models 

   Next, separate MLP models are trained on a different subset of data split by meter type. A 12-fold cross-validation method was used (9 consecutive months as training and the remaining 3 months as validation sets). Further two separate models are trained: one without using the normalized target variable and another one with the normalized target variable. 
   
   16. `python scripts/03_train_mlp_meter.py --normalize_target` - This script trains 48 models (4 meters x 12-folds) on different training subsets using normalized target variable. It creates 48 individual model files within the `models/mlp-split_meter/target_normalization/` directory. This script ran for 6 hours and 19 minutes. 
   
   17. `python scripts/03_train_mlp_meter.py` - This script trains 48 models (4 meters x 12-folds) on different training subsets. It creates 48 individual model files within the `models/mlp-split_meter/no_normalization/` directory. This script ran for 6 hours and 8 minutes.    
   
Alternatively, we can run `train.sh` that invokes all model training scripts one-by-one. 
   
#### 4. Model prediction
Next, we need to execute the following scripts to make predictions using the trained models.

   1. `meter_split_model` - This script makes predictions using the `meter_split_model` and creates `submission_meter.csv` and `submission_replaced_meter.csv` within the `output` directory. This script ran for 10 minutes. 

   ```
   (tensorflow_p36) $ pwd
   /home/ubuntu/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-1/   
   (tensorflow_p36) $ cd meter_split_model
   (tensorflow_p36) $ python predict.py
   (tensorflow_p36) $ cd ..
   ```
      
   2. `kfold_model` - This script makes predictions using the kfold_model and creates `submission_kfold.csv` and `submission_replaced_kfold.csv` within the `output` directory. This script ran for 16 minutes. 

   ```
   (tensorflow_p36) $ cd kfold_model
   (tensorflow_p36) $ python predict.py
   (tensorflow_p36) $ cd ..
   ```

   3. `cleanup_model` - This makes predictions using the kfold_model and creates `submission_cleanup.csv` and `submission_replaced_cleanup.csv` within the `output` directory. This script ran for 8 minutes. 

   ```
   (tensorflow_p36) $ cd cleanup_model
   (tensorflow_p36) $ python train.py
   (tensorflow_p36) $ cd ..
   ```
   
##### 4.1 CatBoost models   

   Next, we execute the below commands to make predictions using the previously trained CatBoost models. 
   
   4. `python scripts/04_predict_cb_meter.py --normalize_target` - This script creates `cb-split_meter-target_normalization.npy` within the `output` directory. It ran for 49 minutes.   
   5. `python scripts/04_predict_cb_meter.py` - This script creates `cb-split_meter-no_normalization.npy` within the `output` directory. It ran for 48 minutes.

   6. `python scripts/04_predict_cb_primary_use.py --normalize_target` - This script creates `cb-split_primary_use-target_normalization.npy` within the `output` directory. It ran for 49 minutes.
   
   7. `python scripts/04_predict_cb_primary_use.py` - This script creates `cb-split_primary_use-no_normalization.npy` within the `output` directory. It ran for 49 minutes.

   8. `python scripts/04_predict_cb_site.py --normalize_target` - This script creates `cb-split_site-target_normalization.npy` within the `output` directory. It ran for 33 minutes.
   
   9. `python scripts/04_predict_cb_site.py` - This script creates `cb-split_site-no_normalization.npy` within the `output` directory. It ran for 31 minutes.

   
##### 4.2 LightGBM models 

   Next, we execute the below commands to make predictions using the previously trained LightGBM models. 
   
   10. `python scripts/04_predict_lgb_meter.py --normalize_target` - This script creates `lgb-split_meter-target_normalization.npy` within the `output` directory. It ran for 45 minutes.   
   
   11. `python scripts/04_predict_lgb_meter.py` - This script creates `lgb-split_meter-no_normalization.npy` within the `output` directory. It ran for 45 minutes.

   12. `python scripts/04_predict_lgb_primary_use.py --normalize_target` - This script creates `lgb-split_primary_use-target_normalization.npy` within the `output` directory. It ran for 67 minutes.
   
   13. `python scripts/04_predict_lgb_primary_use.py` - This script creates `lgb-split_primary_use-no_normalization.npy` within the `output` directory. It ran for 71 minutes.

   14. `python scripts/04_predict_lgb_site.py --normalize_target` - This script creates `lgb-split_site-target_normalization.npy` within the `output` directory. It ran for 40 minutes.
   
   15. `python scripts/04_predict_lgb_site.py` - This script creates `lgb-split_site-no_normalization.npy` within the `output` directory. It ran for 41 minutes.
   
##### 4.3 Multilayer Perceptron (MLP) models 
   Next, we execute the below commands to make predictions using the previously trained MLP models. 
   
   16. `python scripts/04_predict_mlp_meter.py --normalize_target` - This script creates `mlp-split_meter-target_normalization.npy` within the `output` directory. It ran for 53 minutes.   
   17. `python scripts/04_predict_mlp_meter.py` - This script creates `mlp-split_meter-no_normalization.npy` within the `output` directory. It ran for 53 minutes.
   
Alternatively, we can run `predict.sh` that invokes all model prediction scripts one-by-one.    


#### 5. Model ensembling
The final model ensembling step determines the weights for the individual prediction files and combines them into a final submission file. 

   1. `python scripts/05_blend_predictions.py` - This scripts ensembles the predictions from all models and creates the final submission file `final_submission.csv` within the `output` directory. It ran for 36 minutes. There is another script `scripts/05_optimize_blend_predictions.py` that finds model weights programmatically.

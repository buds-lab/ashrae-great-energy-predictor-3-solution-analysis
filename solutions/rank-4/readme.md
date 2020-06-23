The fourth rank solution scored 1.235 on the private leaderboard and 0.936 (rank 44) on the public leaderboard. The technical detail of this solution can be found in [this Kaggle post](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/124788). We describe here the code structure and step-by-step instructions on how to reproduce this solution from scratch.

## Directory structure

This solution consists of the following list of directories and files.

 - `input` - Contains the input dataset (the original competition dataset).
 - `output` - Contains the output files created by each notebook and the final submission file.
 - `exception_label` - IPython notebooks for generating exception labels or outliers.
 - `model1` - IPython notebooks for the first model (2 folds XGBoost) training and prediction.
 - `model2` - IPython notebooks for the second model (5 folds XGBoost) training and prediction.
 - `model3` - IPython notebooks for the third model (3 folds LightGBM) training and prediction.
 - `blend_of_model` - IPython notebooks for model ensembling and for creating the final submission file.

## System configuration

The following are the hardware and software specifications of the system on which this solution was reproduced from scratch.

### Hardware 
The hardware specifications of the system are:
  - Cloud Computing Service: AWS EC2 
  - Instance Type: [g4dn.4xlarge (16 vCPUs, 64 GB RAM, and 120 GB disk)](https://aws.amazon.com/ec2/instance-types/g4/)
  - AMI: Deep Learning AMI (Ubuntu 18.04) Version 27.0 - ami-0a7789c77135a5f8a

The details on how to launch a Deep Learning AMI can be found [here](https://aws.amazon.com/getting-started/hands-on/get-started-dlami/). After launching, we connected to the instance using `PuTTY`. The details can be found [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html).

### Software
The Deep Learning AMI comes with pre-installed Anaconda-based Python environments and GPU software for developing deep learning models. We used the `tensorflow_p36` environment to reproduce this solution. 

#### 1. Activating the `tensorflow_p36` environment

We will start by activating the `tensorflow_p36` environment.

   ```
   ubuntu@ip-172-31-38-204:~$ source activate tensorflow_p36
   WARNING: First activation might take some time (1+ min).
   Installing TensorFlow optimized for your Amazon EC2 instance......
   Env where framework will be re-installed: tensorflow_p36
   Instance g4dn.4xlarge is identified as a GPU instance, removing tensorflow-serving-cpu
   Installation complete.
   (tensorflow_p36) ubuntu@ip-172-31-38-204:~$
   ```

Next, we will check the `Python` and `pip` versions.

   ```
   (tensorflow_p36) ubuntu@ip-172-31-38-204:~$ python --version
   Python 3.6.6 :: Anaconda, Inc.
   (tensorflow_p36) ubuntu@ip-172-31-38-204:~$ pip --version
   pip 19.3.1 from /home/ubuntu/anaconda3/envs/tensorflow_p36/lib/python3.6/site-packages/pip (python 3.6)
   ```

Next, let's also check the GPU details using the `nvidia-smi` command.

   ```
   (tensorflow_p36) ubuntu@ip-172-31-38-204:~$ nvidia-smi
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

#### 2. Set up a Jupyter Notebook Server
As this solution was implemented using Jupyter Notebooks, we need to set up a Jupyter Notebook Server on our instance. The details can be found [here](https://docs.aws.amazon.com/dlami/latest/devguide/setup-jupyter.html).

#### 3. Installation of Kaggle API
The [Kaggle API](https://github.com/Kaggle/kaggle-api) is a python package that enables accessing competition datasets using a command-line interface. This is required for us to download the original competition dataset and the output files from third-party kernels. 

We install the Kaggle API using `pip`

   ```
   (tensorflow_p36) ubuntu@ip-172-31-38-204:~$ pip install kaggle
   ```

Let's make sure it works properly and verify the version.
   ```
   (tensorflow_p36) ubuntu@ip-172-31-38-204:~$ kaggle --version
   Kaggle API 1.5.6
   ```
After the installation, we need to set up the API credentials file `kaggle.json`. The details can be found [here](https://github.com/Kaggle/kaggle-api).


## Reproducing the solution

The reproduction of this solution involves three steps:

  - Download the solution and set up the input datasets.
  - Data pre-processing (generating exception labels or outliers).
  - Model development and prediction.
  - Model ensembling.

We explain these steps and the required commands in detail below.

**Important**: *We have to execute one notebook at a time and shut down the current notebook before starting the next one to reuse the main memory*.

### 1. Download the solution and set up the input datasets

We clone the GitHub repository using the git command.


   ```
   (tensorflow_p36) ubuntu@ip-172-31-38-204:~$ pwd
   /home/ubuntu
   (tensorflow_p36) ubuntu@ip-172-31-38-204:~$ git clone git://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis.git
   ```

The contents of `rank-1` directory should be looking like this.

   ```
   (tensorflow_p36) ubuntu@ip-172-31-38-204:~/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-4$ ls -l
   total 30764
   -rw-rw-r--  1 ubuntu ubuntu   235013 May 20 13:48 'MODEL SUMMARY.docx'
   drwxrwxr-x  2 ubuntu ubuntu     4096 May 20 13:48  blend_of_model
   drwxrwxr-x  2 ubuntu ubuntu     4096 May 20 13:48  exception_label
   drwxrwxr-x  2 ubuntu ubuntu     4096 May 20 14:11  input
   drwxrwxr-x  2 ubuntu ubuntu     4096 May 20 13:48  model1
   drwxrwxr-x  2 ubuntu ubuntu     4096 May 20 13:48  model2
   drwxrwxr-x  2 ubuntu ubuntu     4096 May 20 13:48  model3
   drwxrwxr-x 31 ubuntu ubuntu     4096 May 20 13:48  output
   -rw-rw-r--  1 ubuntu ubuntu     1218 May 20 13:48  readme.md
   -rw-rw-r--  1 ubuntu ubuntu     5190 May 20 13:48  requirements.txt
   -rw-rw-r--  1 ubuntu ubuntu 31221401 May 20 13:48 'solution explaining.mp4'
   
   ```

Next, we download the Kaggle competition dataset using the kaggle command and extract them into the `input/ashrae-energy-prediction/` directory.

```
(tensorflow_p36) $ pwd
/home/ubuntu/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-1/
(tensorflow_p36) $ mkdir input/ashrae-energy-prediction
(tensorflow_p36) $ cd input/ashrae-energy-prediction
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
(tensorflow_p36) $ cd ../../

```

Next, we download the leaked dataset from [this kernel](https://www.kaggle.com/yamsam/ashrae-leak-data-station/output) and store them into the `output/ashrae-leak-data-station` directory using the `kaggle` command.


   ```
   (tensorflow_p36) $ cd output/ashrae-leak-data-station
   (tensorflow_p36) $ kaggle kernels output yamsam/ashrae-leak-data-station
   
   ```

This command downloads `leak.feather` file into `output/ashrae-leak-data-station` directory.


### 2. Data pre-processing

After setting up the input dataset, the next step is to pre-process the dataset to prepare the exception labels or outliers which are excluded from the model development and prediction process. 

We need to execute 6 IPython notebooks in the following order to generate the final exception label file `train_exception.pkl`. This file is required by other notebooks.
    
   1. `ashrae-eda-2.ipynb` - Exploratory Data Analysis (EDA) notebook that assigns exception labels to the training set after manually visualizing the data points. This notebook will run for 3 minutes.

   2. `ashrae-test-exception-label.ipynb` - This notebook generates exception label file `test_exception.pkl` for the test dataset. This notebook will run for 2 minutes.   
   
   3. `ashrae-eda-exception-label1.ipynb` - This notebook updates the exception labels generated by the previous notebook. This notebook will run for 16 minutes.

   4. `fork-of-ashrae-eda-exception-label2.ipynb` - This notebook updates the exception labels for different time slots that were manually identified as outliers. This notebook will run for 4 minutes.

   5. `ashrae-eda-exception-label5.ipynb` - This notebook also updates the exception labels for different time slots that were manually identified as outliers. This notebook will run for 11 minutes.

   6. `fork-of-ashrae-eda-exception-label5.ipynb` - This notebook also updates the exception labels for different time slots that we manually identified as outliers. This is the final notebook of pre-processing that creates the final exception label file named `train_exception.pkl`. This notebook will run for 1 minute.


### 3. Model development
There are three models involved in this solution. They are implemented independently and their predictions are combined to make the final submission file.

#### 3.1 First model - 2-fold XGBoost

The first model is a 2-fold XGBoost. We need to execute the following notebooks sequentially to generate the final prediction file of this model. 

  1. `as-data-minification.ipynb` - This notebook reads the original competition dataset files from `input/ashrae-energy-prediction/`, minimizes the memory usage, and saves them as `.pkl` files within the `./output/as-data-minification/` directory. This notebook ran for 2 minutes.

  2. `fork-of-as-2kfold-model6-train-df.ipynb` - This notebook extracts the features and stores the training dataset file `train_df.pkl` in to the `output/fork-of-as-2kfold-model6-train-df` directory. This notebook ran for 3 minutes.

  3. `fork-of-as-2kfold-model6-test-df.ipynb` - This notebook extracts the features and stores the testing dataset file `test_df.pkl` in to the `output/fork-of-as-2kfold-model6-test-df` directory. This notebook ran for 2 minutes.

  4. `fork-of-as-2kfold-model6-xgb-fr7d12-fold0.ipynb` - This notebook trains an XGBoost model using the first fold training set and stores the binary model into `output/fork-of-as-2kfold-model6-xgb-fr7d12-fold0/xgb_kfold_1.bin`. This notebook ran for 7 minutes. 

  5. `fork-of-as-2kfold-model6-xgb-fr7d12-fold1.ipynb` - This notebook trains an XGBoost model using the second fold training set and stores the binary model into `output/fork-of-as-2kfold-model6-xgb-fr7d12-fold1/xgb_kfold_2.bin`. This notebook also ran for 7 minutes. 

  6. `fork-of-as-2kfold-model6-xgb-fr7d12-pred.ipynb` - This notebook makes the 2-fold XGBoost model predictions using previously created model binary finals and saves the final prediction file into `output/fork-of-as-2kfold-model6-xgb-fr7d12-pred/submission.csv`. This notebook ran for 3 minutes. 


#### 3.2 Second model - 5 folds XGBoost per meter type

The second model is a 5-fold XGBoost trained by meter type. We need to execute the following notebooks sequentially to generate the final prediction file.

  1. `ashrae-feather-format-for-fast-loading.ipynb` - This notebook prepares feather format dataset files for faster loading. It reads the original raw datasets from `input/ashrae-energy-prediction/` and saves them as `.feather` files within the `./output/ashrae-feather-format-for-fast-loading/` directory. This notebook ran for 2 minutes.

  2. `as-meter2-no-1099-xgb-meter0-fold0.ipynb` - This notebook extracts the features, trains an XGBoost model using the first fold training set from meter 0, and stores the model prediction into `output/as-meter2-no-1099-xgb-meter0-fold0/submission.csv`. This notebook ran for 11 minutes. 

  3. `as-meter2-no-1099-xgb-meter0-fold1.ipynb` - This notebook extracts the features, trains an XGBoost model using the second fold training set from meter 0, and stores the model prediction into `output/as-meter2-no-1099-xgb-meter0-fold1/submission.csv`. This notebook ran for 11 minutes. 

  4. `as-meter2-no-1099-xgb-meter0-fold2.ipynb` - This notebook extracts the features, trains an XGBoost model using the third fold training set from meter 0, and stores the model prediction into `output/as-meter2-no-1099-xgb-meter0-fold2/submission.csv`. This notebook ran for 18 minutes. 

  5. `as-meter2-no-1099-xgb-meter0-fold3.ipynb` - This notebook extracts the features, trains an XGBoost model using the third fold training set from meter 0, and stores the model prediction into `output/as-meter2-no-1099-xgb-meter0-fold3/submission.csv`. This notebook ran for 11 minutes. 

  6. `as-meter2-no-1099-xgb-meter0-fold4.ipynb` - This notebook extracts the features, trains an XGBoost model using the third fold training set from meter 0, and stores the model prediction into `output/as-meter2-no-1099-xgb-meter0-fold4/submission.csv`. This notebook ran for 9 minutes. 

  7. `as-meter2-no-1099-xgb-meter0.ipynb` - This notebook combines the prediction from all individual fivefold models and creates a single submission file for meter 0, `output/as-meter2-no-1099-xgb-meter0/submission.csv`, after accounting for exception labels. This notebook ran for 4 minutes.

  8. `as-meter2-no-1099-xgb-meter1.ipynb` - This notebook extracts the features, trains a 5-fold XGBoost model using the training set from meter 1, and stores the model prediction into `output/as-meter2-no-1099-xgb-meter1/submission.csv`. This notebook ran for 21 minutes. 

  9. `as-meter2-no-1099-xgb-meter2.ipynb`- This notebook extracts the features, trains a 5-fold XGBoost model using the training set from meter 2, and stores the model prediction into `output/as-meter2-no-1099-xgb-meter2/submission.csv`. This notebook ran for 11 minutes. 

  10. `as-meter2-no-1099-xgb-meter3.ipynb`- This notebook extracts the features, trains a 5-fold XGBoost model using the training set from meter 3, and stores the model prediction into `output/as-meter2-no-1099-xgb-meter3/submission.csv`. This notebook ran for 7 minutes. 

  11. `as-meter2-no-1099-xgb.ipynb` - This notebook combines all predictions from meter 0, 1, 2, and 3 and creates a single prediction file into `output/as-meter2-no-1099-xgb/submission.csv`. This notebook ran for 3 minutes.

#### 3.3 Third model - 3 folds LightGBM

The third model is a 3-fold LightGBM based on [this public kernel](https://www.kaggle.com/aitude/ashrae-kfold-lightgbm-without-leak-1-08). We need to execute the below notebook to generate the final prediction file from this model.

  1. `ashrae-kfold-lightgbm-without-leak-1-08.ipynb` - This notebook is a fork from [this public kernel](https://www.kaggle.com/aitude/ashrae-kfold-lightgbm-without-leak-1-08) that uses a 3-fold lightGBM model without using the leaked dataset. This notebook ran for 16 minutes.

### 4. Model ensembling

The individual prediction files from the above three models are combined to generate the final submission file. We need to execute the following notebook to generate the final submission file.

  1. `as-leakage-replace-new.ipynb` - This notebook calculates the weights for each model using the leaked dataset and stores the combined final submission file into `output/final_submission.csv`. This notebook ran for 6 minutes.

In summary, the entire solution took about 3 hours and 10 minutes to complete. It requires manual intervention to start and stop each IPython notebook as per the above steps.
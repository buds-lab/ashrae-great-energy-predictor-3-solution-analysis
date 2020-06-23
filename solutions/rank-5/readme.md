The fifth rank solution scored 1.237 on the private leaderboard and 0.940 (rank 23) on the public leaderboard. The technical detail of this solution can be found in [this Kaggle post](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/127086). We describe here the code structure and step-by-step instructions on how to reproduce this solution from scratch.

## Directory structure
This solution consists of the following list of directories and files. 

 - `model` - trained model binary files
 - `output` - model predictions and the final submission file
 - `train_code` - code to train models from scratch
 - `predict_code` - code to generate predictions from model binaries
 - `ensemble_code` - code to ensemble the predictions
 - `preproceeding_code` - code to pre-process the data
 - `prepare_data` - pre-processed data files
 - `external_data` - external files required by this solution such as leak data
 - `requirements.txt` - python package dependencies
 - `SETTINGS.json` - a json configuration file 

## System configuration and setup
The following are the hardware and software specifications of the system on which this solution was reproduced from scratch. 

### Hardware 
The hardware specifications of the system are:
  - Cloud Computing Services: AWS EC2 
  - Instance Type: r5.2xlarge (8 vCPUs, 64 GB memory, and 40 GB EBS volume)
  - AMI: Ubuntu Server 18.04 LTS (HVM), SSD Volume Type - ami-09a4a9ce71ff3f20b (64-bit x86) 
  - Python: Python 3.6.9

Note that the competitor had used a server with below hardware specifications.
 - Intel Xeon Gold 6126 @ 2.60GHz x2(12Core/24Thread, Skylake)
 - DDR4-2666 DIMM ECC REG 32GB x 12 = 384GB
 - Ubuntu 16.04.5 LTS
 - Python: Python 3.6.8

### Software
This solution was originally developed using Python 3.6.8. So we will set up a Python 3 virtual environment with the required python packages. The list of packages used can be found in `requirements.txt`.

After launching the AWS instance, we connected to the instance using `PuTTY`. The details can be found [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html).

We will start by installing the required tools such as `pip3` package manager and `git`. 

```
$ sudo add-apt-repository universe
$ sudo apt update
$ sudo apt-get install python3-pip
$ sudo apt-get install git
```

#### 1. Download the solution

Next, we check out the solution code from GitHub using the `git` command and see the contents.

```
$ pwd
/home/redhat
$ git clone git://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis.git
```

The contents of `rank-5` directory should be looking like this.

```
$ cd ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-5
$ pwd
/home/ubuntu/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-5
$ ls -l
total 612
-rw-rw-r-- 1 ubuntu ubuntu 462118 Apr 20 10:32 ASHRAE_fifth_place_solution.pdf
-rw-rw-r-- 1 ubuntu ubuntu 106078 Apr 20 10:32 ModelSummary.docx
-rw-rw-r-- 1 ubuntu ubuntu    206 Apr 20 10:32 SETTINGS.json
-rw-rw-r-- 1 ubuntu ubuntu    163 Apr 20 10:32 directory_structure.txt
drwxrwxr-x 2 ubuntu ubuntu   4096 Apr 20 10:32 ensemble_code
drwxrwxr-x 2 ubuntu ubuntu   4096 Apr 20 18:31 external_data
drwxrwxr-x 2 ubuntu ubuntu   4096 Apr 20 19:11 input
drwxrwxr-x 2 ubuntu ubuntu   4096 Apr 20 20:13 model
drwxrwxr-x 2 ubuntu ubuntu   4096 Apr 20 20:50 output
drwxrwxr-x 2 ubuntu ubuntu   4096 Apr 20 20:14 predict_code
drwxrwxr-x 2 ubuntu ubuntu   4096 Apr 20 19:42 prepare_data
drwxrwxr-x 2 ubuntu ubuntu   4096 Apr 20 18:45 preproceeding_code
-rw-rw-r-- 1 ubuntu ubuntu   3165 Apr 20 10:32 readme.md
-rw-rw-r-- 1 ubuntu ubuntu   2395 Apr 20 10:32 requirements.txt
drwxrwxr-x 2 ubuntu ubuntu   4096 Apr 20 20:11 train_code
```

#### 2. Installing Kaggle API
The [Kaggle API](https://github.com/Kaggle/kaggle-api) is a python package that enables accessing competition datasets using a command-line interface. This is required for us to download the original competition dataset and the output files from third-party kernels. 

We install the Kaggle API using `pip3`

```
$ pip3 install kaggle
```

Let's make sure it works properly and also verify the version
```
$ kaggle --version
Kaggle API 1.5.6
```
After the installation, we also need to set up the API credentials before using it. You need to have a Kaggle account and the details can be found [here](https://github.com/Kaggle/kaggle-api).

#### 3. Setting up a python virtual environment.

Next, we need to set up a python virtual environment. This is required to keep all python packages within an isolated environment so that our changes will not affect the existing python setup in the system and also to avoid version conflicts. 

We install the `virtualenv` package using `pip3`

```
# pip3 install virtualenv
```

Next, we create a new python virtual environment based on Python 3.6.8 within the solution #5 code directory.

```
$ pwd
/home/redhat/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-5

$ virtualenv --python=/usr/bin/python3.6 py36
```

This will take a while to set up and copy the python libraries to the new virtual environment named `py36`. Next, we need to activate this environment using the `source` command as below.

```
$ source py36/bin/activate
(py36) $
(py36) $ python --version
Python 3.6.9
```

#### 4. Installing the required python packages
The list of packages required to reproduce this solution is given in a separate `requirements.txt` file. 

We use `pip3` to install all dependencies listed in the `requirements.txt`

```
(py36) $ pip3 install -r requirements.txt
```
This installation step will take several minutes to complete as it needs to download and install over 100 python packages into the newly created virtual environment. Note that we have removed some of the unnecessary packages from the original `requirements.txt` provided by the competitor. 


## Reproducing the solution

The reproduction of this solution involves four steps:
 - Setting up the input datasets
 - Data pre-processing and feature engineering
 - Model training and prediction
 - Ensembling  

We explain these steps and the required commands in detail below. 

### Setting up the input datasets

We need to set up two datasets before proceeding with the next steps: 1) the original competition dataset, and 2) the external datasets. We use the *Kaggle API**'s command-line interface to download and set up both of these datasets. 

1. **Competition dataset**

This is the [original dataset provided by the Kaggle](https://www.kaggle.com/c/ashrae-energy-prediction/data). It consists of train, test, weather, building metadata, and a sample submission file.

We execute the following commands sequentially to download and extract the original competition dataset into the `input` data directory.

```
(py36) $ pwd
/home/redhat/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-5
(py36) $ mkdir input
(py36) $ cd input
(py36) $ kaggle competitions download -c ashrae-energy-prediction
(py36) $ unzip ashrae-energy-prediction.zip
(py36) $ rm ashrae-energy-prediction.zip
(py36) $ ls -l
total 2549756
-rw-rw-r-- 1 ubuntu ubuntu      45527 Oct 10  2019 building_metadata.csv
-rw-rw-r-- 1 ubuntu ubuntu  447562511 Oct 10  2019 sample_submission.csv
-rw-rw-r-- 1 ubuntu ubuntu 1462461085 Oct 10  2019 test.csv
-rw-rw-r-- 1 ubuntu ubuntu  678616640 Oct 10  2019 train.csv
-rw-rw-r-- 1 ubuntu ubuntu   14787908 Oct 10  2019 weather_test.csv
-rw-rw-r-- 1 ubuntu ubuntu    7450075 Oct 10  2019 weather_train.csv
(py36) $ cd ../
```

2. **External datasets**

In addition to the original competition dataset, there are three external datasets used in this solution -- leaked dataset and the final submission files from two public Kaggle kernels. We need to download and save these files into the 
`external_data` directory. Once again we use the Kaggle API's command-line interface which is very convenient to download the output files from Kaggle kernels.

We download and store the leaked data file named `leak.feather` from [this kernel](https://www.kaggle.com/yamsam/ashrae-leak-data-station) into the `external_data` directory using below commands.

```
(py36) $ pwd
/home/redhat/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-5
(py36) $ mkdir external_data
(py36) $ cd external_data
(py36) $ kaggle kernels output yamsam/ashrae-leak-data-station
Output file downloaded to /home/redhat/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-5/external_data/leak.feather
Kernel log downloaded to /home/redhat/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-5/external_data/ashrae-leak-data-station.log 
```

Next, we download the final submission files from two public kernels ([this](https://www.kaggle.com/purist1024/ashrae-simple-data-cleanup-lb-1-08-no-leaks) and [this](https://www.kaggle.com/rohanrao/ashrae-half-and-half)) and store them into the `external_data` directory using below commands.

```
(py36) $ kaggle kernels output purist1024/ashrae-simple-data-cleanup-lb-1-08-no-leaks
Output file downloaded to /home/redhat/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-5/external_data/rows_to_drop.csv
Output file downloaded to /home/redhat/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-5/external_data/submission.csv
Kernel log downloaded to /home/redhat/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-5/external_data/ashrae-simple-data-cleanup-lb-1-08-no-leaks.log 
(py36) $ mv submission.csv submission_simple_data_cleanup.csv

(py36) $ kaggle kernels output rohanrao/ashrae-half-and-half
Output file downloaded to /home/redhat/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-5/external_data/submission.csv
Kernel log downloaded to /home/redhat/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-5/external_data/ashrae-half-and-half.log 
(py36) $ mv submission.csv submission_half_and_half.csv
```

After setting up all external datasets, the contents of `external_data` should be looking like this

```
(py36) $ ls -l
total 2332608
-rw-rw-r-- 1 ubuntu ubuntu       1613 Apr 20 18:28 ashrae-half-and-half.log
-rw-rw-r-- 1 ubuntu ubuntu       3295 Apr 20 18:25 ashrae-leak-data-station.log
-rw-rw-r-- 1 ubuntu ubuntu        668 Apr 20 18:31 ashrae-simple-data-cleanup-lb-1-08-no-leaks.log
-rw-rw-r-- 1 ubuntu ubuntu  537604472 Apr 20 18:25 leak.feather
-rw-rw-r-- 1 ubuntu ubuntu    8086989 Apr 20 18:30 rows_to_drop.csv
-rw-rw-r-- 1 ubuntu ubuntu 1129137613 Apr 20 18:28 submission_half_and_half.csv
-rw-rw-r-- 1 ubuntu ubuntu  713725080 Apr 20 18:31 submission_simple_data_cleanup.csv
```

### Data pre-processing and feature engineering

After setting up all input datasets, the next step is to preprocess the input files and to extract the features from them. The preprocessed files will be stored in the `prepare_data` directory. 

1. **Data cleaning and feature engineering**

The python script named `./preproceeding_code/prepare_data.py` performs data preprocessing and creates features. This involves removing outliers, correcting timezones and filling missing values in the weather data file, and the creation of feature variables both in train and test datasets. The technical details about these features can be found in the [competitor's original post in the Kaggle discussion board](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/127086).

```
(py36) $ mkdir prepare_data 
(py36) $ python ./preproceeding_code/prepare_data.py
```
This script took about 9 minutes to complete. It creates two files `train_fe.ftr` and 'test_fe.ftr' within the `prepare_data` directory. 


2. **Cleaning leaked data**

Next, we also need to clean up the leaked data file `leak.feather` by removing some bad entries. This is done by executing the below command. 

```
(py36) $ python ./ensemble_code/leak_data_drop_bad_rows.py
```
This script took about 2 minutes to complete. It creates the revised leaked data file 'leak_data_drop_bad_rows.pkl' within the `prepare_data` directory. 

3. **Creation of simplified train and test datasets**

Finally, we need to execute the below script that creates a simplified version, a minimal set of important features, of the train and test datasets, named 'train_fe_simplified.ftr' and 'test_fe_simplified.ftr', within the `prepare_data` directory.

```
(py36) $ python ./preproceeding_code/prepare_data_simplified.py
```

This script took less than a minute to complete. Note that these simplified train and test datasets were never used in the model development step. As per the competitor, these files can possibly be used to create simple models, with less than 10 features, that would perform reasonably good. But we haven't yet verified this. 


### Model development
The model development step involves model training and model prediction.

1. **Model training**

This solution trains two sets of models, one using a partial training dataset to identify the model parameters (num boosting rounds) and another one using the whole training set. Each set consists of four models one per meter type. Again, the technical details about these models can be found in the [competitor's original post in the Kaggle discussion board](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/127086).

The shell script named `./train_code/train_model.sh` trains two models. We execute this script using the command below. 

```
(py36) $ mkdir model
(py36) $ sh train_code/train_model.sh
```

This shell script executes the python script from the command line as `train_code/train_model.py 1`, where the argument 1 is the random seed value.

This model training process took about 13 minutes to complete. It creates two sets of model binary files, named `model_all_use_train_fe_seed1_leave31_lr005_tree500.pkl` and `model_use_train_fe_seed1_leave31_lr005_tree500.pkl` within  the `model` directory. 

2. **Model prediction**

The next step is predicting the final model output using the previously trained models. The python script named `./predict_code/predict_model.py` makes the final predictions. We need to run this script with different parameters as below.

```
(py36) $ mkdir output
(py36) $ python ./predict_code/predict_model.py 1 0.5 train_fe.ftr test_fe.ftr`
(py36) $ python ./predict_code/predict_model.py 1 1.0 train_fe.ftr test_fe.ftr`
(py36) $ python ./predict_code/predict_model.py 1 1.5 train_fe.ftr test_fe.ftr`
```
Each model prediction step took about 6-8 minutes with a total of 20 minutes to make all model predictions. These scripts will create three output files, named `use_train_fe_seed1_leave31_lr005_tree500_mul05.csv`, `use_train_fe_seed1_leave31_lr005_tree500_mul10.csv`, and `use_train_fe_seed1_leave31_lr005_tree500_mul15.csv` respectively, within the `output` data directory.

**Note:** *There is a shell script file named `./predict_code/predict_model.sh` which can be used to invoke all three predictions. But this script got failed with `Permission denied` error even after running this script with elevated rights, using `sudo`. We didn't investigate this error.*

### Model ensembling

The final step in this reproduction process is combing the three model prediction files into a single final submission file. The python script named `./ensemble_code/weighted_average_ensemble.py` performs this task. 

```
(py36) $ python ./ensemble_code/weighted_average_ensemble.py
```
This script took about 16 minutes to complete. It creates the final submission file named `submission.csv` within the `output` data directory.

In summary, the entire solution would take about 1 hour to complete after setting up the required software tools and python packages.

## Issues and fixes
Here is the list of issues and their possible fixes in addition to those mentioned above.

1. *ModuleNotFoundError: No module named `tkinter`*
This error will arise when `matplotlib` was imported to the code and when we run the code on the command line. As no visualization is required, we can comment out the lines that imported `matplotlib`.

\
The second rank solution scored 1.232 on the private leaderboard and 0.937 (rank 12) on the public leaderboard. The technical detail of this solution can be found in [this Kaggle post](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/123481). We describe here the code structure and step-by-step instructions on how to reproduce this solution from scratch.

## Directory structure
 - `code` - all IPython notebooks
 - `input` - input datasets and leaked datasets
 - `sub` - Model output files and the final submission file

## System configuration and setup
The following are the hardware and software specifications of the system on which this solution was reproduced.

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
ubuntu@ip-172-31-29-207:~$ source activate tensorflow_p36
WARNING: First activation might take some time (1+ min).
Installing TensorFlow optimized for your Amazon EC2 instance......
Env where framework will be re-installed: tensorflow_p36
Instance g4dn.4xlarge is identified as a GPU instance, removing tensorflow-serving-cpu
Installation complete.
(tensorflow_p36) ubuntu@ip-172-31-29-207:~$
```

Next, we will check the `Python` and `pip` versions.

```
(tensorflow_p36) ubuntu@ip-172-31-29-207:~$ python --version
Python 3.6.6 :: Anaconda, Inc.
(tensorflow_p36) ubuntu@ip-172-31-29-207:~$ pip --version
pip 19.3.1 from /home/ubuntu/anaconda3/envs/tensorflow_p36/lib/python3.6/site-packages/pip (python 3.6)
```

Next, let's also check the GPU details using the `nvidia-smi` command.

```
(tensorflow_p36) ubuntu@ip-172-31-29-207:~$ nvidia-smi
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
(tensorflow_p36) ubuntu@ip-172-31-29-207:~$ pip install kaggle
```

Let's make sure it works properly and verify the version.
```
(tensorflow_p36) ubuntu@ip-172-31-29-207:~$ kaggle --version
Kaggle API 1.5.6
```
After the installation, we need to set up the API credentials file `kaggle.json`. The details can be found [here](https://github.com/Kaggle/kaggle-api).

## Reproducing the solution

The following are the steps to reproducing this solution:
 - Download the solution and set up the input datasets
 - Model training and prediction
 - Ensembling  

We explain these steps and the required commands in detail below. 

#### 1. Download the solution and input datasets

We clone the solutions from the GitHub repository using the `git` command.

```
(tensorflow_p36) ubuntu@ip-172-31-29-207:~$ pwd
/home/ubuntu
(tensorflow_p36) ubuntu@ip-172-31-29-207:~$ git clone git://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis.git
```

The contents of `rank-2` directory should be looking like this.
```
(tensorflow_p36) ubuntu@ip-172-31-29-207:~/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-2$ ls -l -R
.:
total 300
-rw-rw-r-- 1 ubuntu ubuntu 287867 Apr 27 13:00 'ASHRAE - Great Energy Predictor III solution.pdf'
-rw-rw-r-- 1 ubuntu ubuntu   1416 Apr 27 13:00  README.md
drwxrwxr-x 2 ubuntu ubuntu   4096 Apr 27 13:00  code
drwxrwxr-x 2 ubuntu ubuntu   4096 Apr 27 13:00  input
drwxrwxr-x 2 ubuntu ubuntu   4096 Apr 27 13:00  sub

./code:
total 9192
-rw-rw-r-- 1 ubuntu ubuntu 2397315 Apr 27 13:00 cb.ipynb
-rw-rw-r-- 1 ubuntu ubuntu    7839 Apr 27 13:00 ensemble.ipynb
-rw-rw-r-- 1 ubuntu ubuntu  104603 Apr 27 13:00 ffnn-site-10.ipynb
-rw-rw-r-- 1 ubuntu ubuntu  108023 Apr 27 13:00 ffnn-site-11.ipynb
-rw-rw-r-- 1 ubuntu ubuntu   99975 Apr 27 13:00 ffnn-site-12.ipynb
-rw-rw-r-- 1 ubuntu ubuntu   96591 Apr 27 13:00 ffnn-site-13.ipynb
-rw-rw-r-- 1 ubuntu ubuntu  109770 Apr 27 13:00 ffnn-site-3.ipynb
-rw-rw-r-- 1 ubuntu ubuntu  101642 Apr 27 13:00 ffnn-site-5.ipynb
-rw-rw-r-- 1 ubuntu ubuntu   98456 Apr 27 13:00 ffnn-site-6.ipynb
-rw-rw-r-- 1 ubuntu ubuntu   96521 Apr 27 13:00 ffnn-site-7.ipynb
-rw-rw-r-- 1 ubuntu ubuntu  131400 Apr 27 13:00 ffnn-site-8.ipynb
-rw-rw-r-- 1 ubuntu ubuntu  115936 Apr 27 13:00 ffnn-site-9.ipynb
-rw-rw-r-- 1 ubuntu ubuntu    2567 Apr 27 13:00 ffnn-sites-all.ipynb
-rw-rw-r-- 1 ubuntu ubuntu 2584003 Apr 27 13:00 lgb.ipynb
-rw-rw-r-- 1 ubuntu ubuntu  143241 Apr 27 13:00 pubv1.ipynb
-rw-rw-r-- 1 ubuntu ubuntu   24406 Apr 27 13:00 pubv2.ipynb
-rw-rw-r-- 1 ubuntu ubuntu   47375 Apr 27 13:00 pubv3.ipynb
-rw-rw-r-- 1 ubuntu ubuntu   45949 Apr 27 13:00 pubv4.ipynb
-rw-rw-r-- 1 ubuntu ubuntu 3061203 Apr 27 13:00 xgb.ipynb

./input:
total 0

./sub:
total 0

```

Next, we download the Kaggle competition dataset using the `kaggle` command and extract them into the `input/ashrae-energy-prediction` directory.

```
(tensorflow_p36) $ pwd
/home/ubuntu/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-2/
(tensorflow_p36) $ mkdir input/ashrae-energy-prediction
(tensorflow_p36) $ cd input/ashrae-energy-prediction
(tensorflow_p36) $ kaggle competitions download -c ashrae-energy-prediction
(tensorflow_p36) $ unzip ashrae-energy-prediction.zip
(tensorflow_p36) $ ls -l
total 2937920
-rw-rw-r-- 1 ubuntu ubuntu  397104766 Apr 21 18:58 ashrae-energy-prediction.zip
-rw-rw-r-- 1 ubuntu ubuntu      45527 Oct 10  2019 building_metadata.csv
-rw-rw-r-- 1 ubuntu ubuntu  447562511 Oct 10  2019 sample_submission.csv
-rw-rw-r-- 1 ubuntu ubuntu     366922 Nov 26 08:27 site0-building4-orig.csv
-rw-rw-r-- 1 ubuntu ubuntu 1462461085 Oct 10  2019 test.csv
-rw-rw-r-- 1 ubuntu ubuntu  678616640 Oct 10  2019 train.csv
-rw-rw-r-- 1 ubuntu ubuntu   14787908 Oct 10  2019 weather_test.csv
-rw-rw-r-- 1 ubuntu ubuntu    7450075 Oct 10  2019 weather_train.csv
(tensorflow_p36) $ cd ../../

```

Next, we download the leaked datasets using the `kaggle` command and extract them into the `input/leakdata` directory.

```
(tensorflow_p36) $ pwd
/home/ubuntu/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-2/
(tensorflow_p36) $ mkdir input/leakdata
(tensorflow_p36) $ cd input/leakdata
(tensorflow_p36) $ kaggle datasets download berserker408/ashare-leakdata
(tensorflow_p36) $ unzip ashare-leakdata.zip
(tensorflow_p36) $ ls -l
total 1002284
-rw-rw-r-- 1 ubuntu ubuntu 161934837 Apr 27 16:46 ashare-leakdata.zip
-rw-rw-r-- 1 ubuntu ubuntu 794181625 Jan 13 04:19 leak.csv
-rw-rw-r-- 1 ubuntu ubuntu  70219989 Jan 13 04:20 site4_leak_rrr_v1.csv
(tensorflow_p36) $ cd ../../

```

#### 2. Model training and prediction
This solution consists of 8 models including 4 third-party models. They can be found within the `code` directory. 
The workflow of these models, including data preprocessing, feature engineering, model training and prediction, is there within the IPython notebook itself. We open and execute them one-by-by to get their individual prediction files.

   1. `lgb.ipynb` - This notebook performs data preprocessing and feature engineering and creates the train and test datasets, `train.pkl` and `test.pkl`, within the `code` directory. These files are used by other notebooks. Next, this notebook trains 3-fold LightGBM models, makes predictions, and creates the submission file `lgb_3wise.csv` within the `sub` directory. This notebook ran for 104 minutes.

   2. `xgb.ipynb` - This notebook trains 3-fold XGBoost models and creates the prediction file `xgb_3wise.csv` within the `sub` directory. This notebook ran for 19 hours and 17 minutes.

   3. `cb.ipynb` - This notebook trains 3-fold Catboost models and creates the prediction file `cb_3wise.csv` within the `sub` directory. This notebook ran for 8 hours and 41 minutes.

   4. Next, we need to execute the following four notebooks which are based on public Kaggle kernels.
      - `pubv1.ipynb` - This notebook trains 3-fold LightGBM models using minimal number of features and creates the prediction file `submission_1.08_v1.csv` within the `sub` directory. This notebook ran for 16 minutes. 

      - `pubv2.ipynb` - This notebook trains LightGBM models and creates the prediction file `submission_1.08_v2` within the `sub` directory. This notebook also creates the `rows_to_drop.csv` within the `input/rows-do-drop/` directory. This notebook ran for 7 minutes. 

      - `pubv3.ipynb` - This notebook also trains 3-fold LightGBM models and creates the prediction file `submission_1.08_v3` within the `sub` directory. This notebook ran for 44 minutes. 

      - `pubv4.ipynb` - This notebook trains 2-fold LightGBM models using a minimal number of features and creates the prediction file `submission_1.10_v1` within the `sub` directory. This notebook ran for 16 minutes. 

   4. Next, we need to execute the following notebooks that train Feed-Forward Neural Network (FFNN) models and make predictions for specific sites for meter id 0 only. 
      - `ffnn-site-3.ipynb` - This notebook trains 4-fold FFNN models for site #3 and creates the prediction file `ffnn_pred_site_3.csv` within the `sub` directory. This notebook ran for 37 minutes. 

      - `ffnn-site-5.ipynb` - This notebook trains 4-fold FFNN models for site #5 and creates the prediction file `ffnn_pred_site_5.csv` within the `sub` directory. This notebook ran for 17 minutes. 

      - `ffnn-site-6.ipynb` - This notebook trains 4-fold FFNN models for site #6 and creates the prediction file `ffnn_pred_site_6.csv` within the `sub` directory. This notebook ran for 8 minutes. 

      - `ffnn-site-7.ipynb` - This notebook trains 4-fold FFNN models for site #7 and creates the prediction file `ffnn_pred_site_7.csv` within the `sub` directory. This notebook ran for 6 minutes. 

      - `ffnn-site-8.ipynb` - This notebook trains 4-fold FFNN models for site #8 and creates the prediction file `ffnn_pred_site_8.csv` within the `sub` directory. This notebook ran for 14 minutes. 

      - `ffnn-site-9.ipynb` - This notebook trains 4-fold FFNN models for site #9 and creates the prediction file `ffnn_pred_site_9.csv` within the `sub` directory. This notebook ran for 27 minutes. 

      - `ffnn-site-10.ipynb` - This notebook trains 4-fold FFNN models for site #10 and creates the prediction file `ffnn_pred_site_10.csv` within the `sub` directory. This notebook ran for 7 minutes. 

      - `ffnn-site-11.ipynb` - This notebook trains 4-fold FFNN models for site #11 and creates the prediction file `ffnn_pred_site_11.csv` within the `sub` directory. This notebook ran for 3 minutes. 

      - `ffnn-site-12.ipynb` - This notebook trains 4-fold FFNN models for site #12 and creates the prediction file `ffnn_pred_site_12.csv` within the `sub` directory. This notebook ran for 8 minutes. 

      - `ffnn-site-13.ipynb` - This notebook trains 4-fold FFNN models for site #13 and creates the prediction file `ffnn_pred_site_13.csv` within the `sub` directory. This notebook ran for 17 minutes. 

      - `ffnn-sites-all.ipynb` - This notebook combines the predictions from the above notebooks and creates the combined prediction file `ffnn.csv` within the `sub` directory. This notebook ran for 1 minute. 

#### 3. Model ensembling
The final model ensembling step determines the weights for the above eight individual prediction files and combines them into a single final submission file. 

   1. `ensemble.ipynb` - This notebook creates the final submission file `final.csv` within the `sub` directory. This notebook ran for 7 minutes.
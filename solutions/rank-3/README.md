This third rank solution scored 1.234 on the private leaderboard and 0.946 (rank 48) on the public leaderboard. The technical detail of this solution can be found in [this Kaggle post](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/124984). We describe here the code structure and step-by-step instructions on how to reproduce this solution from scratch.


## Directory structure
 - `Catboost on GPU` - Catboost model code
 - `Keras_NN_weights` - Keras model files
 - `*.ipynb` - 9 IPython notebooks

## System configuration and setup
The following are the hardware and software specifications of the system on which this solution was reproduced.

### Hardware 
The hardware specifications of the system are:

**For Catboost:**
  - Cloud Computing Service: AWS EC2 
  - Instance Type: [p3.2xlarge (8 vCPUs, 61 GB RAM, and 120 GB disk)](https://aws.amazon.com/ec2/instance-types/p3/)
  - AMI: Deep Learning AMI (Ubuntu 18.04) Version 27.0 - ami-0a7789c77135a5f8a
  
**For all other models:**  
  - Cloud Computing Service: AWS EC2 
  - Instance Type: [m5a.4xlarge (16 vCPUs, 64 GB RAM, and 120 GB disk)](https://aws.amazon.com/ec2/instance-types/m5/)
  - AMI: Deep Learning AMI (Ubuntu 18.04) Version 27.0 - ami-0a7789c77135a5f8a

The details on how to launch a Deep Learning AMI can be found [here](https://aws.amazon.com/getting-started/hands-on/get-started-dlami/). After launching, we connected to the instance using `PuTTY`. The details can be found [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html).

### Software
The Deep Learning AMI comes with pre-installed Anaconda-based Python environments and GPU software for developing deep learning models. We used the `tensorflow_p36` environment to reproduce this solution. 

#### 1. Activating the `tensorflow_p36` environment

We will start by activating the `tensorflow_p36` environment.

```
ubuntu@ip-172-31-45-3:~$ source activate tensorflow_p36
WARNING: First activation might take some time (1+ min).
Installing TensorFlow optimized for your Amazon EC2 instance......
Env where framework will be re-installed: tensorflow_p36
Instance g4dn.4xlarge is identified as a GPU instance, removing tensorflow-serving-cpu
Installation complete.
(tensorflow_p36) ubuntu@ip-172-31-45-3:~$
```

Next, we will check the `Python` and `pip` versions.

```
(tensorflow_p36) ubuntu@ip-172-31-45-3:~$ python --version
Python 3.6.6 :: Anaconda, Inc.
(tensorflow_p36) ubuntu@ip-172-31-45-3:~$ pip --version
pip 19.3.1 from /home/ubuntu/anaconda3/envs/tensorflow_p36/lib/python3.6/site-packages/pip (python 3.6)
```

Next, let's also check the GPU details using the `nvidia-smi` command.

```
(tensorflow_p36) ubuntu@ip-172-31-45-3:~$ nvidia-smi
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
(tensorflow_p36) ubuntu@ip-172-31-45-3:~$ pip install kaggle
```

Let's make sure it works properly and verify the version.
```
(tensorflow_p36) ubuntu@ip-172-31-45-3:~$ kaggle --version
Kaggle API 1.5.6
```
After the installation, we need to set up the API credentials file `kaggle.json`. The details can be found [here](https://github.com/Kaggle/kaggle-api).

## Reproducing the solution

The following are the steps to reproducing this solution:
 - Download the solution and set up the input datasets
 - Data preprocessing and feature engineering
 - Model training and prediction
 - Ensembling  

We explain these steps and the required commands in detail below. 

#### 1. Download the solution and input datasets

We clone the solutions from the GitHub repository using the `git` command.

```
(tensorflow_p36) ubuntu@ip-172-31-45-3:~$ pwd
/home/ubuntu
(tensorflow_p36) ubuntu@ip-172-31-45-3:~$ git clone git://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis.git
```

The contents of `rank-3` directory should be looking like this.
```
(tensorflow_p36) ubuntu@ip-172-31-45-3:~/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-3$ ls -l
total 1992
drwxrwxr-x 2 ubuntu ubuntu   4096 Apr 21 09:22 'Catboost on GPU'
drwxrwxr-x 2 ubuntu ubuntu   4096 Apr 21 09:22  Keras_NN_weights
-rw-rw-r-- 1 ubuntu ubuntu   1932 Apr 21 09:22  README.md
-rw-rw-r-- 1 ubuntu ubuntu  33106 Apr 21 09:22  generate_datasets.ipynb
-rw-rw-r-- 1 ubuntu ubuntu  22273 Apr 21 09:22  generate_leak_data.ipynb
-rw-rw-r-- 1 ubuntu ubuntu 653677 Apr 21 09:22  level1--submission_multimeter003--lightgbm.ipynb
-rw-rw-r-- 1 ubuntu ubuntu 615189 Apr 21 09:22  level1--submission_multimeter004_nobuild--lightgbm.ipynb
-rw-rw-r-- 1 ubuntu ubuntu  56679 Apr 21 09:22  level1--submission_nn001--DenseNN.ipynb
-rw-rw-r-- 1 ubuntu ubuntu  52934 Apr 21 09:22  level1--submission_nn007lofo--CNN.ipynb
-rw-rw-r-- 1 ubuntu ubuntu  53169 Apr 21 09:22  level1--submission_whatsyourcv3_0052_trncl--lightgbm.ipynb
-rw-rw-r-- 1 ubuntu ubuntu 205923 Apr 21 09:22  level1--submission_withoutleak001--lightgbm.ipynb
-rw-rw-r-- 1 ubuntu ubuntu 187011 Apr 21 09:22  level2--ensembling_model.ipynb
-rw-rw-r-- 1 ubuntu ubuntu 121963 Apr 21 09:22  model_summary.pdf
-rw-rw-r-- 1 ubuntu ubuntu   5285 Apr 21 09:22  requirement.txt

```

Next, we download the Kaggle competition dataset using the `kaggle` command and extract them into the `rank-3` directory.

```
(tensorflow_p36) $ pwd
/home/ubuntu/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-3/
(tensorflow_p36) $ kaggle competitions download -c ashrae-energy-prediction
(tensorflow_p36) $ unzip ashrae-energy-prediction.zip
```

Next, we download the leaked datasets using the `kaggle` command and save them into the `./leaked` directory.
There are leaked datasets from the following five sites: 
  - [site 0]{https://www.kaggle.com/yamsam/new-ucf-starter-kernel}
  - [site 1]{https://www.kaggle.com/mpware/ucl-data-leakage-episode-2}
  - [site 2]{https://www.kaggle.com/pdnartreb/asu-buildings-energy-consumption}
  - [site 4]{https://www.kaggle.com/serengil/ucb-data-leakage-site-4-81-buildings/output}
  - [site 15]{https://www.kaggle.com/pp2file/ashrae-site15-cornell/output}

  
```
(tensorflow_p36) $ pwd
/home/ubuntu/ashrae-great-energy-predictor-3-solution-analysis/solutions/rank-3/
(tensorflow_p36) $ mkdir leaked
(tensorflow_p36) $ cd leaked
(tensorflow_p36) $ kaggle kernels output yamsam/new-ucf-starter-kernel 
(tensorflow_p36) $ kaggle kernels output mpware/ucl-data-leakage-episode-2
(tensorflow_p36) $ kaggle datasets download pdnartreb/asu-buildings-energy-consumption -f asu_2016-2018.csv
(tensorflow_p36) $ kaggle kernels output serengil/ucb-data-leakage-site-4-81-buildings
(tensorflow_p36) $ kaggle kernels output pp2file/ashrae-site15-cornell 
(tensorflow_p36) $ ls -l
total 1699612
-rw-rw-r-- 1 ubuntu ubuntu      9903 Apr 21 13:45 ashrae-site15-cornell.log
-rw-rw-r-- 1 ubuntu ubuntu  50278620 Apr 21 13:26 asu_2016-2018.csv.zip
-rwxrwxrwx 1 ubuntu ubuntu       741 Apr 21 13:42 leaked.sh
-rw-rw-r-- 1 ubuntu ubuntu      9871 Apr 21 13:44 new-ucf-starter-kernel.log
-rw-rw-r-- 1 ubuntu ubuntu  42087597 Apr 21 13:44 site1.pkl
-rw-rw-r-- 1 ubuntu ubuntu 159488445 Apr 21 13:45 site15_leakage.csv
-rw-rw-r-- 1 ubuntu ubuntu  64559540 Apr 21 13:45 site4.csv
-rw-rw-r-- 1 ubuntu ubuntu 712110120 Apr 21 13:43 submission.csv
-rw-rw-r-- 1 ubuntu ubuntu 711807680 Apr 21 13:44 submission_ucf_replaced.csv
-rw-rw-r-- 1 ubuntu ubuntu      5186 Apr 21 13:45 ucb-data-leakage-site-4-81-buildings.log
-rw-rw-r-- 1 ubuntu ubuntu      3571 Apr 21 13:44 ucl-data-leakage-episode-2.log

```

#### 2. Data preprocessing and feature engineering
Next, we open and execute the below two notebooks that performs data cleaning and preprocessing.

   1. `generate_datasets.ipynb` - This notebook performs data cleaning and feature engineering. It creates `train_cleanup_001.feather`, `train_simple_cleanup.feather` and `test_simple_cleanup.feather` files. These files are required in other notebooks. This notebook ran for five hours and 2 minutes.
   
   2. `generate_leak_data.ipynb` - This notebook combines all leaked data and creates `leak012345_001.feather` file. This notebook ran for 3 minutes.
   

#### 3. Model training and prediction

This solution consists of 4 models: LightGBM, Dense NN, CNN, and CatBoost. 

The workflow of these models, including data preprocessing, feature engineering, model training and prediction, is there within the IPython notebook itself. We open and execute them one-by-by to get their individual prediction files.

   1. `level1--submission_multimeter003--lightgbm.ipynb` - This notebook is based on [this public kernel](https://www.kaggle.com/aitude/ashrae-kfold-lightgbm-without-leak-1-08). It trains 3-fold LightGBM models, one per meter type, and creates the model prediction file `submission_multimeter003.csv.gz`. This notebook ran for 36 minutes.
  
   2. `level1--submission_multimeter004_nobuild--lightgbm.ipynb` - This notebook is also based on [this public kernel](https://www.kaggle.com/aitude/ashrae-kfold-lightgbm-without-leak-1-08). It trains 3-fold LightGBM models, one each meter type, using minimal number of features, and creates the model prediction file `submission_multimeter004_nobuild.csv.gz`. This notebook ran for 48 minutes.

   3. `level1--submission_nn001--DenseNN.ipynb` - This notebook is based on [this public kernel](https://www.kaggle.com/isaienkov/keras-nn-with-embeddings-for-cat-features-1-15). It trains 2-fold Dense Neural Network models and creates the model prediction file `submission_nn001.csv`. This notebook ran for 8 hours and 29 minutes.

   4. `level1--submission_nn007lofo--CNN.ipynb` - This notebook trains 2-fold CNN models and creates the model prediction file `submission_nn007lofo.csv.gz`. This notebook ran for 12 hours and 38 minutes.

   5. `level1--submission_whatsyourcv3_0052_trncl--lightgbm.ipynb` - This notebook is based on [this public kernel](https://www.kaggle.com/kimtaegwan/what-s-your-cv-method). It trains 3-fold LightGBM models and creates the model prediction file `submission_whatsyourcv3_0052_trncl.csv.gz`. This notebook ran for 1 hour and 13 minutes.
   
   6. `level1--submission_withoutleak001--lightgbm.ipynb` - This notebook trains 3-fold LightGBM models, without using the leadked data, and creates the model prediction file `submission_withoutleak001.csv.gz`. This notebook ran for 33 minutes.
   
##### 3.1. CatBoost models
We need to run CatBoost notebooks on a different AWS EC2 instance that has a GPU. 

   7. `level1--catboost002--Catboost.ipynb` - This notebook is based on [this public kernel](https://www.kaggle.com/ragnar123/another-1-08-lb-no-leak). It trains 3-fold CatBoost models and creates the model prediction file `submission_catboost002.csv.gz`. We need to move this file to first AWS instance. This notebook ran for 1 hour and 13 minutes.

#### 4. Model ensembling
The model ensembling step determines the weights for the above models and combines their predictions into a single final submission file. 

In addition to the above models, this solution uses prediction files from the following two public kernels:
   - `fe2_lgbm.csv` - downloaded from https://www.kaggle.com/ragnar123/another-1-08-lb-no-leak (commit #1)
   - `histgradboost113.csv` -  downloaded the output file `fe2_lgbm.csv` from https://www.kaggle.com/tunguz/ashrae-histgradientboosting (commit #7) and renamed it to `histgradboost113.csv`.
   

   1. `level2--ensembling_model.ipynb` - This notebook creates the final submission file `submission_ensemble_3rd_place.csv.gz`. This notebook ran for 23 minutes.

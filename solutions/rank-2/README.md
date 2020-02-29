# ASHRAE - Great Energy Predictor III - Second ranked solution

### Harward Requirements:
12 GB GPU (recommended)

### Python 3 packages:
- numpy==1.13.1
- pandas==0.19.2
- tensorflow==1.14.0
- scikit-learn==0.22 
- lightgbm==2.2.3
- xgboost==0.90
- catboost==0.20.1  
- keras==2.2.5

### Instructions
Please run with the following order:
 1) lgb.ipynb
 2) xgb.ipynb
 3) cb.ipynb
 4) pubv1/v2/v3/v4.ipynb
 5) ffnn-site-3/5/6/7/8/9/10/11/12/13.ipynb
 6) ffnn-sites-all.ipynb
 7) ensemble.ipynb
 
Then you can get the final.csv, which scored 0.937(public LB)/1.232(private LB). It may take four days to run them all. One thing still need to metion is that you may need to change data path and the GPU device ID.

In addition, we need to show gratitude the follow public kernels, which have small weights in our final ensemble results.
https://www.kaggle.com/aitude/ashrae-kfold-lightgbm-without-leak-1-08 (pubv1.ipynb)
https://www.kaggle.com/purist1024/ashrae-simple-data-cleanup-lb-1-08-no-leaks (pubv2.ipynb)
https://www.kaggle.com/ragnar123/another-1-08-lb-no-leak (pubv3.ipynb)

### External data:
Our external data can be downloaded from https://www.kaggle.com/berserker408/ashare-leakdata, and put into the input/leakdata folder. The external data are obtained from the public kernels and sources mentioned in the external data thread on Kaggle. They only comprise of the leak data of sites 0, 1, 2, 4 and 15.



# ASHRAE Great Energy Predictor III (GEPIII) - Top 5 Winning Solutions Explained

This repository contains the code and documentation of top-5 winning solutions from the ASHRAE - Great Energy Predictor III cometition. It also contains comparative analysis of these solutions with respect to their characteristics such as workflow, computation time, and score distributation with respect to meter type, site, and primary space usage, etc.

A full overview of the GEPIII competition can be [found online](https://www.tandfonline.com/doi/full/10.1080/23744731.2020.1795514)

To cite this analysis:

Clayton Miller, Pandarasamy Arjunan, Anjukan Kathirgamanathan, Chun Fu, Jonathan Roth, June Young Park, Chris Balbach, Krishnan Gowri, Zoltan Nagy, Anthony D. Fontanini & Jeff Haberl (2020) The ASHRAE Great Energy Predictor III competition: Overview and results, Science and Technology for the Built Environment, DOI: 10.1080/23744731.2020.1795514

### First rank solution
 - [Code](../../tree/master/solutions/rank-1/)
 - [Sulution summary (Kaggle discussion board)](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/124709)
 
### Second rank solution
 - [Code](../../tree/master/solutions/rank-2/)
 - [Solution summary (PDF)](../../tree/master/solutions/rank-2/ASHRAE%20-%20Great%20Energy%20Predictor%20III%20solution.pdf)
 - [Sulution summay (Kaggle discussion board)](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/123481)
 
### Third rank solution
 - [Code](../../tree/master/solutions/rank-3/)
 - [Solution summary (PDF)](../../tree/master/solutions/rank-3/model_summary.pdf)
 - [Sulution summay (Kaggle discussion board)](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/124984)
  
### Fourth rank solution
 - [Code](../../tree/master/solutions/rank-4/)
 - [Solution summary (.DOCX)](../../tree/master/solutions/rank-4/MODEL%20SUMMARY.docx)
 - [Sulution summay (Kaggle discussion board)](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/124788)
 
### Fifth rank solution
 - [Code](../../tree/master/solutions/rank-5/)
 - [Solution summary (.DOCX)](../../tree/master/solutions/rank-5/ModelSummary.docx)
 - [Solution summary presentation (PDF)](../../tree/master/solutions/rank-5/ASHRAE_fifth_place_solution.pdf)
 - [Sulution summay (Kaggle discussion board)](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/127086)
 

## Comparison
|   Final Rank | Team Name             |   Final Private Leaderboard Score | Preprocessing Strategy                                                     | Features Strategy Overview                                                                                                           | Modeling Strategy Overview                                                                                          | Post-Processing strategy                                               |
|-------------:|:----------------------|----------------------------------:|:---------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------|
|            1 | Isamu & Matt          |                             1.231 | Removed anomalies in meter data and imputed missing values in weather data | 28 features, Extensively focused on feature engineering and selected                                                                 | LightGBM, CatBoost, and MLP models trained on different subsets of the training and public data                     | Ensembled the model predictions using weighted generalized mean.       |
|            2 | cHa0s                 |                             1.232 | Visual analytics and manual inspection                                     | Raw energy meter data, temporal features,  building metadata, simple statistical features of weather data.                           | XGBoost, LightGBM, Catboost, and Feed-forward Neural Network models trained on different subset of the training set | Weighted mean. (different weights were used for different meter types) |
|            3 | eagle4                |                             1.234 | Eliminated 0s in the same period in the same site                          | nan                                                                                                                                  | Keras CNN, LightGBM and Catboost                                                                                    | nan                                                                    |
|            4 | 不用leakage上分太难了 |                             1.235 | Not available                                                              | 23 features including raw data, aggregate, weather lag features, and target encoding. Features are selected using sub-training sets. | XGBoost (2-fold, 5-fold) and Light GBM (3-fold)                                                                     | Ensembled three models. Weights were determined using the leaked data. |
|            5 | mma                   |                             1.237 | Dropped long streaks of constant values and zero target values.            | Target encoding using percentile and proportion and used the weather data temporal features                                          | LightGBM in two steps -- identify model parameters on a subset and then train on the whole set for each building.   | Weighted average.                                                      |

### Comparison of execution time

| Solution | Preprocessing | Feature engineering | Training | Prediction | Ensembling | Total (minutes) |
|----------|--------------:|--------------------:|---------:|-----------:|-----------:|----------------:|
| Rank 1   |             9 |                 128 |     7440 |        708 |         35 |            8320 |
| Rank 2   |            36 |                  24 |     1850 |         94 |          7 |            2011 |
| Rank 3   |           178 |                  12 |      501 |        100 |         14 |             805 |
| Rank 4   |            40 |                   7 |       85 |         46 |          6 |             184 |
| Rank 5   |             3 |                   9 |       13 |         20 |         16 |              61 |

Note: all solutions were reproduced on AWS EC2 (g4dn.4xlarge) using Deep Learning AMI.

## Links
1. [Top 5 winning solutions - code and docs (original submission by the winners)](https://www.dropbox.com/sh/73iryui7t0w74ik/AAAY-yF87A2zrLdqHv11vFlsa?dl=0)
2. [Top 5 winning solutions - explainer videos](https://www.dropbox.com/sh/tmnhkmy33vs3uya/AACVU-CcwyqGwApEvhNmSH4Qa?dl=0)

# ASHRAE Great Energy Predictor III (GEPIII) - Top 5 Winning Solutions Explained

This repository contains the code and documentation of top-5 winning solutions from the [ASHRAE - Great Energy Predictor III cometition](https://www.kaggle.com/c/ashrae-energy-prediction) that was held in late 2019 on the Kaggle platform. It also contains comparative analysis of these solutions with respect to their characteristics such as workflow, computation time, and score distributation with respect to meter type, site, and primary space usage, etc.

An video overview of the competition can be found from the [ASHRAE 2020 Online Conference](https://youtu.be/xqtBVy5cZgA) by Clayton Miller from the [BUDS Lab at the National University of Singapore](http://budslab.org/)

## Full Overview
A full overview of the GEPIII competition can be [found in a Science and Technology for the Built Environment Journal](https://www.tandfonline.com/doi/full/10.1080/23744731.2020.1795514) - [Preprint found on arXiv](https://arxiv.org/abs/2007.06933)

To cite this competition or analysis:

Clayton Miller, Pandarasamy Arjunan, Anjukan Kathirgamanathan, Chun Fu, Jonathan Roth, June Young Park, Chris Balbach, Krishnan Gowri, Zoltan Nagy, Anthony D. Fontanini & Jeff Haberl (2020) The ASHRAE Great Energy Predictor III competition: Overview and results, Science and Technology for the Built Environment, DOI: 10.1080/23744731.2020.1795514

The data set from the competition is now opened as the [Building Data Genome 2 project](https://github.com/buds-lab/building-data-genome-project-2) that is outlined in [a paper submitted to the journal Scientific Data](https://arxiv.org/abs/2006.02273)

## Detailed Reproduction of Solutions Overview
Instructions to fully reproduce each solution are [found in the wiki for this repository and other details found below](https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis/wiki).

The raw data data for the [top 5 winning solutions - code and docs (original submission by the winners)](https://www.dropbox.com/sh/73iryui7t0w74ik/AAAY-yF87A2zrLdqHv11vFlsa?dl=0)

## Explanatory Overview Videos from the Winners
The [top five winning solutions can be understood through a series of explainer videos hosted here, including extended presentations at the ASHRAE 2020 Online Conferece in June 2020](https://www.dropbox.com/sh/tmnhkmy33vs3uya/AACVU-CcwyqGwApEvhNmSH4Qa?dl=0). **Potential users of these solutions should note that each winner gave advice on the solution complexity vs. accuracy.** These videos are also listed below individually for each solution.

## Solutions Overview Details
### First Ranked Solution
 - [Code](../../tree/master/solutions/rank-1/)
 - [Solution Summary (Kaggle discussion board)](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/124709)
 - [Overview Video of Solution by team member Matt Motoki](https://youtu.be/ZVX9EbHnH0E)
 - [ASHRAE Annual Meeting Solution Overvew Presentation by team member Matt Motoki](https://youtu.be/fKgNKTAn26M)
 
### Second Ranked Solution
 - [Code](../../tree/master/solutions/rank-2/)
 - [Solution summary (PDF)](../../tree/master/solutions/rank-2/ASHRAE%20-%20Great%20Energy%20Predictor%20III%20solution.pdf)
 - [Solution Summary (Kaggle discussion board)](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/123481)
 - [Overview Video of Solution by team member Rohan Rao](https://youtu.be/Zfhb4c4mB44)
 - [ASHRAE Annual Meeting Solution Overvew Presentation by team member Rohan Rao](https://youtu.be/EhC9CCqMxkM)
 
### Third Ranked Solution
 - [Code](../../tree/master/solutions/rank-3/)
 - [Solution summary (PDF)](../../tree/master/solutions/rank-3/model_summary.pdf)
 - [Solution Summary (Kaggle discussion board)](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/124984)
 - [Overview Video of Solution by team member Xavier Capdepon](https://youtu.be/vc2JOpAVDUA)
 - [ASHRAE Annual Meeting Solution Overvew Presentation by Xavier Capdepon](https://youtu.be/aqOmV37Htp0)
  
### Fourth Ranked Solution
 - [Code](../../tree/master/solutions/rank-4/)
 - [Solution summary (.DOCX)](../../tree/master/solutions/rank-4/MODEL%20SUMMARY.docx)
 - [Solution Summary (Kaggle discussion board)](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/124788)
 - [Overview Video of Solution by Jun Yang](https://youtu.be/m4SigmQ9xhs)
 
### Fifth Ranked Solution
 - [Code](../../tree/master/solutions/rank-5/)
 - [Solution summary (.DOCX)](../../tree/master/solutions/rank-5/ModelSummary.docx)
 - [Solution summary presentation (PDF)](../../tree/master/solutions/rank-5/ASHRAE_fifth_place_solution.pdf)
 - [Solution Summary (Kaggle discussion board)](https://www.kaggle.com/c/ashrae-energy-prediction/discussion/127086)
 - [Overview Video of Solution by team member Yuta Kobayashi](https://youtu.be/2WG1Z4eiL7w)

## Solutions High Level Comparisons
|   Final Rank | Team              |   Final Private Leaderboard Score | Preprocessing Strategy                                                     | Features Strategy Overview                                                                                                           | Modeling Strategy Overview                                                                                          | Post-Processing strategy                                               |
|-------------:|:----------------------|----------------------------------:|:---------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------|
|            1 | Matthew Motoki and Isamu Yamashita (Isamu and Matt)          |                             1.231 | Removed anomalies in meter data and imputed missing values in weather data | 28 features, Extensively focused on feature engineering and selected                                                                 | LightGBM, CatBoost, and MLP models trained on different subsets of the training and public data                     | Ensembled the model predictions using weighted generalized mean.       |
|            2 | Rohan Rao, Anton Isakin, Yangguang Zang, and Oleg Knaub (cHa0s)                 |                             1.232 | Visual analytics and manual inspection                                     | Raw energy meter data, temporal features,  building metadata, simple statistical features of weather data.                           | XGBoost, LightGBM, Catboost, and Feed-forward Neural Network models trained on different subset of the training set | Weighted mean. (different weights were used for different meter types) |
|            3 | Xavier Capdepon (eagle4)                |                             1.234 | Eliminated 0s in the same period in the same site                          | 21 features including raw data, weather, and various meta data                                                                                                                                | Keras CNN, LightGBM and Catboost                                                                                    | Weighted average                                                                   |
|            4 | Jun Yang  (不用leakage上分太难了) |                             1.235 | Deleted outliers during the training phase                                                             | 23 features including raw data, aggregate, weather lag features, and target encoding. Features are selected using sub-training sets. | XGBoost (2-fold, 5-fold) and Light GBM (3-fold)                                                                     | Ensembled three models. Weights were determined using the leaked data. |
|            5 | Tatsuya Sano, Minoru Tomioka, and Yuta Kobayashi (mma)                   |                             1.237 | Dropped long streaks of constant values and zero target values.            | Target encoding using percentile and proportion and used the weather data temporal features                                          | LightGBM in two steps -- identify model parameters on a subset and then train on the whole set for each building.   | Weighted average.                                                      |

### Execution Time Comparison

| Solution | Preprocessing | Feature engineering | Training | Prediction | Ensembling | Total (minutes) |
|----------|--------------:|--------------------:|---------:|-----------:|-----------:|----------------:|
| Rank 1   |             9 |                 128 |     7440 |        708 |         35 |            8320 |
| Rank 2   |            36 |                  24 |     1850 |         94 |          7 |            2011 |
| Rank 3   |           178 |                  12 |      501 |        100 |         14 |             805 |
| Rank 4   |            40 |                   7 |       85 |         46 |          6 |             184 |
| Rank 5   |             3 |                   9 |       13 |         20 |         16 |              61 |

Note: all solutions were reproduced on AWS EC2 (g4dn.4xlarge) using Deep Learning AMI.



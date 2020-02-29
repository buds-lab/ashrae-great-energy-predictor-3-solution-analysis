# ASHRAE - Great Energy Predictor III - Fourth ranked solution

Hello!

Below you can find a outline of how to reproduce my solution for the <ASHRAE - Great Energy Predictor III> competition.
If you run into any trouble with the setup/code or have any questions please contact me at <1136337803@qq.com>

## CONTENTS
  - code
  - MODEL SUMMARY.docx
  - solution explaining.mp4

### HARDWARE: (The following specs were used to create the original solution)
Kaggle kernel hardware

### SOFTWARE:
Kaggle kernel software

## OPERATING INSTRUCTIONS:
All the code is done in the kaggle kernel and has been shared with the competition host. At the same time, I also downloaded these codes and models from the Kaggle kernel and packed them into dsdd.zip for upload. It is recommended that the competition host reproduce the solution directly in the shared kaggle kernel. 
Here is the relevant code description:

Note: All code should run in kaggle kernel and All kernels should import the <ASHRAE - Great Energy Predictor III> competition dataset.

## exception_label:
  "fork-of-ashrae-eda-exception-label5" is the kernel that generates the final exception label file "train_exception.pkl". 
  The kernel "fork-of-ashrae-eda-exception-label5" and its generated files need to be imported into the kernel of model1 and model2.

## model #1:(a 2 folds XGBoost model)
  1. "as-data-minification" performs a preliminary conversion of the data and shrinks the memory, which should be run first.
  2. "fork-of-as-2kfold-model6-train-df" and "fork-of-as-2kfold-model6-test-df" generate train and test sets for model2 respectively. Later kernels need to import these two kernels and the files they generate.
  3. Turn on the GPU of the kaggle kernel and run "fork-of-as-2kfold-model6-xgb-fr7d12-fold0" and "fork-of-as-2kfold-model6-xgb-fr7d12-fold1" to generate two model files "xgb_kfold_1.bin" and "xgb_kfold_2.bin". Because the "tree_method" parameter is set to 'gpu_hist', the generated model is random, but the model effect is basically the same. Later kernels need to import these two kernels and their generated model files.
  4. Run "fork-of-as-2kfold-model6-xgb-fr7d12-pred" to generate predictions.

## model #2:(a 5 folds XGBoost model trained by meter type)
  1. "ashrae-feather-format-for-fast-loading" is a public kernel that can be directly imported into the later kernels.
  2. Turn on the GPU of the kaggle kernel and run "as-meter2-no-1099-xgb-meter0-fold0","as-meter2-no-1099-xgb-meter0-fold1","as-meter2-no-1099-xgb-meter0-fold2","as-meter2-no-1099-xgb-meter0-fold3","as-meter2-no-1099-xgb-meter0-fold4","as-meter2-no-1099-xgb-meter1","as-meter2-no-1099-xgb-meter2","as-meter2-no-1099-xgb-meter3" will generate sub-models and predictions for each meter type. Because the "tree_method" parameter is set to 'gpu_hist', the generated model is random, but the model effect is basically the same.
  3. "as-meter2-no-1099-xgb-meter0" needs to import "as-meter2-no-1099-xgb-meter0-fold0", "as-meter2-no-1099-xgb-meter0-fold1", "as-meter2-no-1099-xgb-meter0-fold2", "as-meter2-no-1099-xgb-meter0-fold3", "as-meter2-no-1099-xgb-meter0-fold4" and their prediction files to generate prediction of meter0.
  4. "as-meter2-no-1099-xgb" needs to import "as-meter2-no-1099-xgb-meter0", "as-meter2-no-1099-xgb-meter1", "as-meter2-no-1099-xgb-meter2", "as-meter2-no-1099-xgb-meter3" and their prediction files to generate the final prediction.
  5. The kernel in the "new_meter_model" folder is a modified version of the above kernels. The code for exporting the model is added and the models are downloaded to this folder.

## model #3:(a 3 folds LightGBM model shared by a public kernel)
  "ashrae-kfold-lightgbm-without-leak-1-08" is a public kernel that can be imported directly into the later kernel.

## blend of model:
  "as-leakage-replace-new" needs to import "fork-of-as-2kfold-model6-xgb-fr7d12-pred", "as-meter2-no-1099-xgb", "ashrae-kfold-lightgbm-without-leak-1-08" and some public kernels. Run "as-leakage-replace-new" to generate the final submission.If you use "as-model-summary-leakage-replace" to generate the final submission, you need to additionally import the "as-model-summary-meter-xgb" kernel in the "new_meter_model" folder.

import os
import datetime
import gc
import logging
from pathlib import Path

from icecream import ic
import pandas as pd

# import matplotlib.pyplot as plt
# from optuna.integration import lightgbm as lgbm



from func import load_datasets, pred_lgbm


LOG = "log/"
LOG_file = LOG + f"pred_logger.log"
if LOG_file not in os.listdir(LOG):
    Path(LOG_file).touch()

formatter = '%(asctime)s : %(message)s'

logging.basicConfig(filename=LOG_file, 
    level=logging.DEBUG, 
    format=formatter)



FEATURES = "."
INPUT = "input/"

feats = list(set([file.split("_")[0] for file in os.listdir("./features")]))
train, test = load_datasets(feats, FEATURES)
ic(train.shape)
ic(test.shape)

submission = pd.read_csv(INPUT + "sample_submission.csv")

del_col = ["price", ]
X = train.drop(del_col, axis=1)
y = train["price"]

del train
gc.collect();

# optunaで1晩かけて出したやつ
params = {'objective': 'regression',
 'metric': 'rmse',
 'force_col_wise': True,
 'feature_pre_filter': False,
 'lambda_l1': 0.3707461932636412,
 'lambda_l2': 1.33931355357755e-07,
 'num_leaves': 256,
 'feature_fraction': 0.8,
 'bagging_fraction': 0.8569397682401525,
 'bagging_freq': 5,
 'min_child_samples': 20,
 'num_iterations': 1000,
 'early_stopping_round': 20} 

N_splits = 5

preds = pred_lgbm(X, y, test, N_splits, params)

OUTPUT = "output/"
submission["取引価格（総額）_log"] = preds

# 提出ファイル名を現在時刻から付ける
dt_now = datetime.datetime.now()
save_name = dt_now.strftime('%Y_%m_%d_%H_%M')

submission.to_csv(OUTPUT + f"{save_name}.csv", index=False)
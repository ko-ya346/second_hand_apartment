import argparse
import inspect
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from icecream import ic

import lightgbm as lgbm

from sklearn.metrics import mean_absolute_error 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

from abc import ABCMeta, abstractmethod
from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir) / f'{self.name}_train.pkl'
        self.test_path = Path(self.dir) / f'{self.name}_test.pkl'
    
    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        ic(self.train.shape)
        ic(self.test.shape)

        return self
    
    @abstractmethod
    def create_features(self):
        raise NotImplementedError
    
    def save(self):
        self.train.to_pickle(str(self.train_path))
        self.test.to_pickle(str(self.test_path))

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing files')
    return parser.parse_args()


def get_features(namespace):
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()

def generate_features(namespace, overwrite):
    for f in get_features(namespace):
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save()

def label_encoder(train, test, col):
    le = LabelEncoder()
    data = pd.concat([train, test])
    data[col] = le.fit_transform(data[col])
    
    tmp_train = data.iloc[:train.shape[0]]
    tmp_test = data.iloc[train.shape[0]:]
    return tmp_train, tmp_test


def kfold_cv(X, y, n_splits=5, random_state=0): 
    folds = KFold(n_splits=n_splits, 
    random_state=random_state, 
    shuffle=True) 
    return list(folds.split(X, y))

def load_datasets(feats, dir="."):
    dfs = [pd.read_pickle(f'{dir}/features/{f}_train.pkl') for f in feats]
    train = pd.concat(dfs, axis=1)
    dfs = [pd.read_pickle(f'{dir}/features/{f}_test.pkl') for f in feats]
    test = pd.concat(dfs, axis=1)
    return train, test

def pred_lgbm(X, y, test, N_splits, params):
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test))

    importance = pd.DataFrame() 
    evals_result = {} 

    for i, (tr_idx, va_idx) in enumerate(kfold_cv(X, y, n_splits=N_splits, random_state=0)): 
        print(f'\nFold {i + 1}') 
        # 学習用データ、評価用データに分ける 
        tr_x, va_x = X.iloc[tr_idx], X.iloc[va_idx] 
        tr_y, va_y = y.iloc[tr_idx], y.iloc[va_idx] 
        # 学習に使用するdataframeをlgbmに渡す 
        lgbm_train= lgbm.Dataset(tr_x, tr_y) 
        lgbm_valid = lgbm.Dataset(va_x, va_y) 

        # 学習 
        model = lgbm.train( 
            params=params,  
            train_set=lgbm_train, 
            valid_sets=[lgbm_train, lgbm_valid], 
            valid_names=["training", "valid"], 
            verbose_eval=100, 
            evals_result=evals_result 
            ) 

        va_pred = model.predict(va_x) 
        oof_preds[va_idx] = va_pred 
        test_preds += model.predict(test) / N_splits
        # val_score = model.best_score["valid"]["l2"] 

        # imp_df = pd.DataFrame({ 
        #     "feature": model.feature_name(), 
        #     "gain": model.feature_importance(importance_type="gain"), 
        #     "fold": i+1 
        # }) 
        # importance = pd.concat([importance, imp_df], axis=0) 

    logging.info("%s", "mae:", mean_absolute_error(y, oof_preds))
    return test_preds
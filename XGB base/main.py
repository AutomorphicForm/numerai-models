# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:55:39 2020

"""


import pandas as pd
import xgboost as xgb
import numpy as np


def main():
    
    #load the data
    train_set = pd.read_csv('https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz')
    features = train_set.columns[train_set.columns.str.startswith('feature')]
    xtrain = train_set[features]
    
    
    #A very basic and straightforward model, low risk of overfitting.
    model1 = xgb.XGBRegressor(n_estimators = 100, max_depth = 6, learning_rate = 0.3, tree_method = 'gpu_hist')
    model2 = xgb.XGBRegressor(n_estimators = 350, max_depth = 2, learning_rate = 0.5, tree_method = 'gpu_hist')
    
    
    
    
    model1.fit(xtrain, train_set['target_kazutsugi'])
    model2.fit(xtrain, train_set['target_kazutsugi'])
    
    
    
    model1.save_model('xgb1.model')
    model2.save_model('xgb2.model')
    
    
if __name__ == '__main__':
    main()

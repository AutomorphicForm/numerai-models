# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:55:39 2020

@author: vikto
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
    model1 = xgb.XGBRegressor(n_estimators = 100, max_depth = 3, learning_rate = 0.5, tree_method = 'gpu_hist')
    
    #Cross validation is used to select the next two values of n_estimators.
    model2 = xgb.XGBRegressor(n_estimators = 150, max_depth = 3, learning_rate = 0.5, tree_method = 'gpu_hist')
    model3 = xgb.XGBRegressor(n_estimators = 750, max_depth = 2, learning_rate = 0.5, colsample_bytree = 0.1, tree_method = 'gpu_hist')
    
    
    
    
    model1.fit(xtrain, train_set['target_kazutsugi'])
    model2.fit(xtrain, train_set['target_kazutsugi'])
    model3.fit(xtrain, train_set['target_kazutsugi'])
    
    
    
    model1.save_model('xgb1.model')
    model2.save_model('xgb2.model')
    model3.save_model('xgb3.model')
    
    
if __name__ == '__main__':
    main()
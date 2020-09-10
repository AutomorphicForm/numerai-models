# -*- coding: utf-8 -*-

import pandas as pd
import xgboost as xgb
import numpy as np
from joblib import dump, load


def iterated_xgb(dataset, num_models = 3, n_estimators = 90, max_depth = 3, bag_percent=0.5):
    """
    Create an XGB model and add boost the model by adding trees and training
    it on randomly sampled eras.
    """
    estimator_list = [n_estimators for i in range(num_models)]
    
    model = xgb.XGBRegressor(n_estimators = 60, max_depth = max_depth, tree_method = 'gpu_hist')
    features = dataset.columns[dataset.columns.str.startswith('feature')]
    X = dataset[features]
    y = dataset['target_kazutsugi']
    model.fit(X, y)
    for i in range(num_models):
        eras_sample = np.random.choice(dataset.era.unique(), int(bag_percent * dataset.era.unique().shape[0]))
        current_data = dataset[dataset.era.isin(eras_sample)]
        X = current_data[features]
        y = current_data['target_kazutsugi']
        model.n_estimators += estimator_list[i]
        
        booster = model.get_booster()
        model.fit(X, y, xgb_model = booster)
    return(model)



def ensemble_xgb(dataset, num_models = 3, n_estimators = 90, max_depth = 3, bag_percent = 0.5):
    """
    Create a list of uncorrelated XGBoost trees through bagging by era.
    For use with ensemble_predictor.
    """
    estimator_list = [n_estimators for i in range(num_models)]
    depth_list = [max_depth for i in range(num_models)]
    features = dataset.columns[dataset.columns.str.startswith('feature')]
    X = dataset[features]
    y = dataset['target_kazutsugi']
    base_model = xgb.XGBRegressor(n_estimators = 300, max_depth = 2, tree_method = 'gpu_hist')
    base_model.fit(X, y)
    model_ensemble = [base_model]
    for i in range(num_models):
        eras_sample = np.random.choice(dataset.era.unique(), int(bag_percent * dataset.era.unique().shape[0]))
        current_data = dataset[dataset.era.isin(eras_sample)]
        X = current_data[features]
        y = current_data['target_kazutsugi']
        model = xgb.XGBRegressor(tree_method = 'gpu_hist', n_estimators = estimator_list[i], max_depth = depth_list[i])
        model.fit(X,y)
        
        model_ensemble.append(model)

    return(model_ensemble)


class ensemble_predictor:
    def __init__(self, model_list):
        self.models = model_list
        
        
    def predict(self, X):
        
        predictions = self.models[0].predict(X)        
        for i in range(1, len(self.models)):
            predictions = predictions + self.models[i].predict(X)
        prediction = predictions / len(self.models)
        return prediction
        
    def fit(self, dataset,num_models,  n_estimators, max_depth):
        bag_percent = 0.5
        estimator_list = [n_estimators for i in range(num_models)]
        depth_list = [max_depth for i in range(num_models)]
        features = dataset.columns[dataset.columns.str.startswith('feature')]
        X = dataset[features]
        y = dataset['target_kazutsugi']
        base_model = xgb.XGBRegressor(n_estimators = 300, max_depth = 2, tree_method = 'gpu_hist')
        base_model.fit(X, y)
        self.models.append(base_model)
        for i in range(num_models):
            eras_sample = np.random.choice(dataset.era.unique(), int(bag_percent * dataset.era.unique().shape[0]))
            current_data = dataset[dataset.era.isin(eras_sample)]
            X = current_data[features]
            y = current_data['target_kazutsugi']
            model = xgb.XGBRegressor(tree_method = 'gpu_hist', n_estimators = estimator_list[i], max_depth = depth_list[i])
            model.fit(X,y)
        
            self.models.append(model)
        
def load_data():
    dataset1 = pd.read_csv('https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz')
    features = dataset1.columns[dataset1.columns.str.startswith('feature')]
    return(dataset1, features)
         

def main():
    dataset1 = pd.read_csv('https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz')
    features = dataset1.columns[dataset1.columns.str.startswith('feature')]
    
    model_list = ensemble_xgb(dataset1, num_models=7, n_estimators=[50 for i in range(7)], max_depth=[3 for i in range(7)])
    model_class = ensemble_predictor(model_list)
    
    live_data = pd.read_csv("https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_tournament_data.csv.xz")
    live_features = live_data[features]
    predictions = model_class.predict(live_features)
    predictions.to_csv('predictions.csv')
    
    
if __name__ == '__main__':
     main()
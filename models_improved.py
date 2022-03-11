import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
from xgboost import XGBClassifier


class ACAClassifier:
    def __init__(self):

        self.soft_voting_clf = None
        self.fitted = False

    def fit(self,
            data_x: pd.DataFrame,
            data_y: np.array):

        rf = RandomForestClassifier()
        svc = SVC(probability=True)
        xgb = XGBClassifier(eval_metric='auc', use_label_encoder=False)  # Changed logloss to auc relatively to docs
        nn = MLPClassifier()
        
        # SVC GRID SEARCHING
        print('SVC')
        svc_param_grid = {'degree': np.arange(1, 10),
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

        svc_grid_search = RandomizedSearchCV(svc, svc_param_grid, cv=2, scoring='balanced_accuracy')
        svc_grid_search.fit(data_x, data_y)
        svc = SVC(degree=svc_grid_search.best_params_['degree'], gamma=svc_grid_search.best_params_['gamma'], probability=True)

        # RF GRID SEARCHING
        print('RF')
        rf_param_grid = {'n_estimators': np.arange(10, 500, step=20),
                         'max_features': ['auto', 'log2']}

        rf_grid_search = RandomizedSearchCV(rf, rf_param_grid, cv=2, scoring='balanced_accuracy')

        rf_grid_search.fit(data_x, data_y)

        rf = RandomForestClassifier(n_estimators=rf_grid_search.best_params_['n_estimators'], max_features=rf_grid_search.best_params_['max_features'])

        # XGB GRID SEARCH
        print('XGB')
        xgb_param_grid = {'n_estimators': np.arange(100, 1000, step=100),
                          'learning_rate': [0.001, 0.01, 0.1]}

        xgb_grid_search = RandomizedSearchCV(xgb, xgb_param_grid, cv=2, scoring='balanced_accuracy')
        xgb_grid_search.fit(data_x, data_y)
        
        xgb = XGBClassifier(n_estimators=xgb_grid_search.best_params_['n_estimators'], learning_rate=xgb_grid_search.best_params_['learning_rate'],
                            eval_metric='auc', use_label_encoder=False)

        # NN GRID SEARCH
        print('NN')
        max_dim = data_x.shape[1]
        nn_param_grid = {'hidden_layer_sizes': [(i, j) for i in np.arange(10, max_dim, step=int(max_dim/10))
                                                for j in np.arange(1, 4)],
                         'max_iter': np.arange(1000, 2000, step=100),
                         'alpha': 10.0 ** -np.arange(1, 10)}

        nn_grid_search = RandomizedSearchCV(nn, nn_param_grid, cv=2, scoring='balanced_accuracy')
        nn_grid_search.fit(data_x, data_y)
        nn = MLPClassifier(hidden_layer_sizes=nn_grid_search.best_params_['hidden_layer_sizes'], max_iter=nn_grid_search.best_params_['max_iter'],
                            alpha=nn_grid_search.best_params_['alpha'])

        self.soft_voting_clf = VotingClassifier(estimators=[('rf', rf), ('svc', svc), ('xgb', xgb), ('nn', nn)],
                                                voting='hard')

        self.soft_voting_clf.fit(data_x, data_y)
        self.fitted = True
        
        with open('parameters.txt', 'a') as file:
            file.write(f'N_Sample:{len(data_y)}; svc:{svc_grid_search.best_params_}; rf:{rf_grid_search.best_params_}; xgb:{xgb_grid_search.best_params_}; nn:{nn_grid_search.best_params_}\n')

    def predict(self, x):

        if self.fitted:
            return self.soft_voting_clf.predict(x)
        else:
            raise Exception('Need to fit model before making predictions')






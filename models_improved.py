import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
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

        # RF GRID SEARCHING
        rf_param_grid = {'n_estimators': np.arange(10, 1000, step=10),
                         'max_features': ['auto', 'log2']}

        rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=10, scoring='balanced_accuracy')

        rf_grid_search.fit(data_x, data_x)

        rf = RandomForestClassifier(rf_grid_search.best_params_)

        # SVC GRID SEARCHING
        svc_param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                          'degree': np.arange(1, 10),
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                          'probability': True}

        svc_grid_search = GridSearchCV(svc, svc_param_grid, cv=10, scoring='balanced_accuracy')
        svc_grid_search.fit(data_x, data_y)
        svc = SVC(svc_grid_search.best_params_)

        # XGB GRID SEARCH

        xgb_param_grid = {'n_estimators': np.arange(100, 2000, step=100),
                          'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]}

        xgb_grid_search = GridSearchCV(xgb, xgb_param_grid, cv=10, scoring='balanced_accuracy')
        xgb_grid_search.fit(data_x, data_y)
        xgb = XGBClassifier(xgb_grid_search.best_params_)

        # NN GRID SEARCH

        max_dim = data_x.shape[1]
        nn_param_grid = {'hidden_layer_sizes': [(i, j) for i in np.arange(1, max_dim, step=int(max_dim/10))
                                                for j in np.arange(1, 4)],
                         'activation': ['identity', 'logistic', 'tanh', 'relu'],
                         'solver': ['lbfgs', 'sgd', 'adam'],
                         'max_iter': np.arange(1000, 2000, step=100),
                         'alpha': 10.0 ** -np.arange(1, 10)}

        nn_grid_search = GridSearchCV(nn, nn_param_grid, cv=10, scoring='balanced_accuracy')
        nn = MLPClassifier(nn_grid_search.best_params_)

        self.soft_voting_clf = VotingClassifier(estimators=[('rf', rf), ('svc', svc), ('xgb', xgb), ('nn', nn)],
                                                voting='soft')

        self.soft_voting_clf.fit(data_x, data_y)
        self.fitted = True

    def predict(self, x):

        if self.fitted:
            return self.soft_voting_clf.predict(x)
        else:
            raise Exception('Need to fit model before making predictions')






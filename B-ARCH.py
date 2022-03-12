import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
from sentence_transformers import SentenceTransformer
import xgboost as xgb

# Testing other classifiers for ensemble:
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

startTime = time.time()

data_path_local = '/Users/nizarmichaud/PycharmProjects/ACA_Public/joe_dutch_clean.xlsx'
data_path_aws = '/home/ec2-user/environment/ACA_Public/joe_dutch_clean.xlsx'

output_path_local = '/Users/nizarmichaud/PycharmProjects/ACA_Public/scores/predictive_data_'
output_path_aws = '/home/ec2-user/environment/ACA_Public/predictive_data_'

env = 'aws'
if env == 'local':
    path = data_path_local
    out_path = output_path_local
else:
    path = data_path_aws
    out_path = output_path_aws

base_df = pd.read_excel(path)
passive = base_df['passive'].to_numpy().astype(int)
proactive = base_df['proactive'].to_numpy().astype(int)
sentences = base_df['text'].to_numpy()


train_set, test_set = train_test_split(sentences, test_size=0.2, random_state=42)
passive_train_set, passive_test_set = train_test_split(passive, test_size=0.2, random_state=42)
proactive_train_set, proactive_test_set = train_test_split(proactive, test_size=0.2, random_state=42)

# Goal here is to have a df like this example:
# models*; accuracy; balanced accuracy; p0_1**; p1_1**; p0_2; p_0x; p_1x; p_0n; p_1n
# RF-sbert0; 0.76; 0.74; x; x; x; x; x; x; x
# NN-sbert0; x; x; x; x; x; x; x; x; x; x; x
#
# * (type and on which sbert was trained)
# ** prediction 1 on test set for class 0 and for class 1
# Then build an algorithm to know which combination and which type of vote we need to have the best model
# Test on half of test set; validation on other half

# Models considered:
# Random Forest; MLP (search and without search); SVC(search and without search);
# XGBoost; GaussianNB; Logistic Reg; KNeighbors


class SVCSearch(BaseEstimator, ClassifierMixin):
    def __init__(self):

        self.fitted = False
        self.svc = None

    def fit(self, x, y):

        svc = SVC(probability=True)

        svc_param_grid = {'degree': np.arange(1, 10),
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

        svc_grid_search = RandomizedSearchCV(svc, svc_param_grid, cv=5, scoring='balanced_accuracy')
        svc_grid_search.fit(x, y)
        self.svc = SVC(degree=svc_grid_search.best_params_['degree'], gamma=svc_grid_search.best_params_['gamma'],
                       probability=True)

        self.svc.fit(x, y)
        self.fitted = True

    def predict(self, x):

        if self.fitted:
            return self.svc.predict(x)
        else:
            raise Exception('Need to fit model before making predictions')

    def predict_proba(self, x):

        if self.fitted:
            return self.svc.predict_proba(x)
        else:
            raise Exception('Need to fit model before making predictions')


class MLPSearch(BaseEstimator, ClassifierMixin):
    def __init__(self):

        self.fitted = False
        self.nn = None

    def fit(self, x, y):

        nn = MLPClassifier()
        max_dim = x.shape[1]
        nn_param_grid = {'hidden_layer_sizes': [(i, j) for i in np.arange(10, max_dim, step=int(max_dim / 10))
                                                for j in np.arange(1, 4)],
                         'max_iter': np.arange(1000, 2000, step=100),
                         'alpha': 10.0 ** -np.arange(1, 10)}

        nn_grid_search = RandomizedSearchCV(nn, nn_param_grid, cv=5, scoring='balanced_accuracy')
        nn_grid_search.fit(x, y)
        self.nn = MLPClassifier(hidden_layer_sizes=nn_grid_search.best_params_['hidden_layer_sizes'],
                                max_iter=nn_grid_search.best_params_['max_iter'],
                                alpha=nn_grid_search.best_params_['alpha'])

        self.nn.fit(x, y)
        self.fitted = True

    def predict(self, x):

        if self.fitted:
            return self.nn.predict(x)
        else:
            raise Exception('Need to fit model before making predictions')

    def predict_proba(self, x):

        if self.fitted:
            return self.nn.predict_proba(x)
        else:
            raise Exception('Need to fit model before making predictions')


# Note there is some way bigger gtr-t5 models with better general accuracy
sbert_models = ['distiluse-base-multilingual-cased-v1']#,
                #'distiluse-base-multilingual-cased-v2',
                #'paraphrase-multilingual-MiniLM-L12-v2',
                #'paraphrase-multilingual-mpnet-base-v2',
                #'all-mpnet-base-v2',
                #'multi-qa-mpnet-base-dot-v1',
                #"gtr-t5-large",
                #"multi-qa-mpnet-base-cos-v1"]

models = {'xgboost': xgb.XGBClassifier(eval_metric='auc', use_label_encoder=False),
          'rf': RandomForestClassifier(),
          'mlp': MLPClassifier(),
          'mlps': MLPSearch(),
          'svc': SVC(probability=False),
          'svcs': SVCSearch(),
          'gnb': GaussianNB(),
          'lr': LogisticRegression(),
          'kn': KNeighborsClassifier()}

every_models = {f'{model_label}-{sb_label}': (sb_label, model)
                for model_label, model in models.items()
                for sb_label in sbert_models}


def text_to_embeddings(x, sbert_label):
    sbert_model = SentenceTransformer(sbert_label)
    return sbert_model.encode(x)

print('embeddings train')
embeddings_train = {sb_label: text_to_embeddings(train_set, sb_label) for sb_label in sbert_models}
print('embeddings test')
embeddings_test = {sb_label: text_to_embeddings(test_set, sb_label) for sb_label in sbert_models}


models_df = []
acc_df = []
b_acc_df = []
proba_df = pd.DataFrame()
true_val_df = proactive_test_set

for model_label, mdlsbert in every_models.items():
    print(model_label)
    sb_label = mdlsbert[0]
    mdl = mdlsbert[1]
    mdl.fit(embeddings_train[sb_label], proactive_train_set)

    prediction = mdl.predict(embeddings_test[sb_label])
    acc = accuracy_score(proactive_test_set, prediction)
    b_acc = balanced_accuracy_score(proactive_test_set, prediction)
    proba = mdl.predict_proba(embeddings_test[sb_label])

    proba_data = []
    proba_columns = []
    for i in range(proba.shape[0]):
        proba_data.append(proba[i][0])
        proba_data.append(proba[i][1])
        proba_columns.append(f'p{i}_0')
        proba_columns.append(f'p{i}_1')

    proba_df_model = pd.DataFrame.from_dict({model_label: proba_data}, orient='index', columns=proba_columns)

    models_df.append(model_label)
    acc_df.append(acc)
    b_acc_df.append(b_acc)
    proba_df = pd.concat([proba_df, proba_df_model])

    break

df = pd.DataFrame({'model': models_df, 'accuracy': acc_df, 'balanced_accuracy': b_acc_df})
df_true_values = pd.DataFrame({'true_values': true_val_df})

proba_df.to_excel('/home/ec2-user/environment/ACA_Public/prob_for_crossmodeleval.xlsx')
df.to_excel('/home/ec2-user/environment/ACA_Public/cross-models_evaluation.xlsx')
df_true_values.to_excel('/home/ec2-user/environment/ACA_Public/true_values_for_crossmodeleval.xlsx')

print(f'Process took:{time.time() - startTime}')

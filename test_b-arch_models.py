from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pandas as pd
import numpy as np

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


def text_to_embeddings(x, sbert_label):
    sbert_model = SentenceTransformer(sbert_label)
    return sbert_model.encode(x)


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


class SBERTCLF(BaseEstimator, ClassifierMixin):
    def __init__(self, model, sbert_model):

        self.fitted = False
        self.sbert_label = sbert_model
        self.model = model

    def fit(self, x, y):

        x_embeddings = text_to_embeddings(x, self.sbert_label)

        self.model.fit(x_embeddings, y)
        self.fitted = True

    def predict(self, x):

        x_embeddings = text_to_embeddings(x, self.sbert_label)

        return self.model.predict(x_embeddings)


models = ['mlp-paraphrase-multilingual-MiniLM-L12-v2',
          'mlp-all-mpnet-base-v2',
          'mlps-distiluse-base-multilingual-cased-v1',
          'gnb-gtr-t5-large']

mlp_paraphrase = SBERTCLF(MLPClassifier(), 'paraphrase-multilingual-MiniLM-L12-v2')

mlp_mpnet = SBERTCLF(MLPClassifier(), 'all-mpnet-base-v2')

mlps_distiluse = SBERTCLF(MLPSearch(), 'distiluse-base-multilingual-cased-v1')

gnb_gtr = SBERTCLF(GaussianNB(), 'gtr-t5-large')

VC = VotingClassifier(estimators=[('mlp-paraphrase', mlp_paraphrase),
                                  ('mlp-mpnet', mlp_mpnet),
                                  ('mlps-distiluse', mlps_distiluse),
                                  ('gnb-gtr', gnb_gtr)],
                      voting='hard')


VC.fit(train_set, proactive_test_set)
predictions = VC.predict(test_set)

b_acc = balanced_accuracy_score(predictions, proactive_test_set)
acc = accuracy_score(predictions, proactive_test_set)

print(f'Balanced accuracy: {b_acc}')
print(f'Accuracy: {acc}')

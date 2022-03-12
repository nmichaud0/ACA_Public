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
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

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


class ACAClf(BaseEstimator, ClassifierMixin):
    def __init__(self, sbert: str):

        self.soft_voting_clf = None
        self.fitted = False
        self.svc, self.rf, self.xgbclf, self.nn = None, None, None, None
        self.sbert_label = sbert
        self.sbert = sbert

        # Testing other classifiers:
        self.nb, self.lr, self.kn, self.gbc = None, None, None, None

    def fit(self,
            x_strings: list,
            y: np.array):
        
        self.sbert = SentenceTransformer(self.sbert_label)

        x = self.text_to_embeddings(x_strings)

        self.rf = RandomForestClassifier()
        svc = SVC(probability=True)
        self.xgbclf = xgb.XGBClassifier(eval_metric='auc', use_label_encoder=False)
        nn = MLPClassifier()

        # testing
        self.nb = GaussianNB()
        self.lr = LogisticRegression()
        self.kn = KNeighborsClassifier()
        self.gbc = GradientBoostingClassifier()

        # SVC Searching
        print('SVC searhing', self.sbert_label)
        svc_param_grid = {'degree': np.arange(1, 10),
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

        svc_grid_search = RandomizedSearchCV(svc, svc_param_grid, cv=5, scoring='balanced_accuracy')
        svc_grid_search.fit(x, y)
        self.svc = SVC(degree=svc_grid_search.best_params_['degree'], gamma=svc_grid_search.best_params_['gamma'],
                       probability=True)
        
        self.svc = SVC(probability=True)

        # NN Searching
        print('NN searching', self.sbert_label)
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
        
        self.nn = MLPClassifier((100, 2))
        
        print('Soft Voting fitting')
        self.soft_voting_clf = VotingClassifier(estimators=[('rf', self.rf), ('svc', self.svc), ('xgb', self.xgbclf),
                                                            ('nn', self.nn),  # testing
                                                            ('nb', self.nb), ('lr', self.lr),
                                                            ('kn', self.kn), ('gbc', self.gbc)],
                                                voting='soft')

        self.soft_voting_clf.fit(x, y)
        self.fitted = True

    def predict(self, xs):

        x = self.text_to_embeddings(xs)

        if self.fitted:
            return self.soft_voting_clf.predict(x)
        else:
            raise Exception('Need to fit model before making predictions')

    def predict_proba(self, xs):

        x = self.text_to_embeddings(xs)

        if self.fitted:
            return self.soft_voting_clf.predict_proba(x)
        else:
            raise Exception('Need to fit model before making predictions')

    def predict_for_each(self, xs):

        x = self.text_to_embeddings(xs)

        if not self.fitted:
            raise Exception('Need to fit model before making predictions')

        return [est.predict(x) for est in self.soft_voting_clf.estimators_]

    def predict_proba_for_each(self, xs):

        x = self.text_to_embeddings(xs)

        if not self.fitted:
            raise Exception('Need to fit model before making predictions')

        return [est.predict_proba(x) for est in self.soft_voting_clf.estimators_]

    def text_to_embeddings(self, text: list):
        
        print('Embeddings encoding')
        
        return self.sbert.encode(text)


# NOT USED
class MyVotingCLF:
    def __init__(self, trained_estimators: list):

        self.estimators = trained_estimators

    def predict(self, x):

        predictions = []

        for est in self.estimators:
            sbert_model = SentenceTransformer(est['sbert'])
            embeddings = sbert_model.encode(x)
            predictions.append(est['aca'].predict_proba(embeddings))

        print(predictions)

        # TODO: check predictions and make the classification
        #  also check if I want to weight some models...


class MetaCLF:
    def __init__(self, models: list):

        self.soft_voting_clf = None
        self.fitted = False
        self.models = models

    def fit(self, sentences_local: np.array, y: np.array):

        estimators = []

        for index, model in enumerate(self.models):

            new_aca = ACAClf(model)
            estimators.append((f'aca_{index}', new_aca))

        self.soft_voting_clf = VotingClassifier(estimators=estimators, voting='soft')
        self.soft_voting_clf.fit(sentences_local, y)
        self.fitted = True

    def predict(self, x):

        if self.fitted:
            return self.soft_voting_clf.predict(x)
        else:
            raise Exception('Need to fit model before making prediction')


sbert_models = ['distiluse-base-multilingual-cased-v1',
                'distiluse-base-multilingual-cased-v2',
                'paraphrase-multilingual-MiniLM-L12-v2',
                'paraphrase-multilingual-mpnet-base-v2',
                'all-mpnet-base-v2',
                'multi-qa-mpnet-base-dot-v1']

# training_sizes = np.arange(100, len(proactive_train_set), step=20)
training_sizes = [1200]


def train_and_predict(models, title):

    scores = []
    balanced_scores = []
    features = []
    sizes = []
    models_df = []

    for model in models:
        print(model.sbert_label)
        for fn, feature in enumerate(['proactive']):
            for size in training_sizes:
                print(f'Feature number:{fn + 1}/2; Size: {size}/{max(training_sizes)}')

                if feature == 'passive':
                    model.fit(train_set[:size], passive_train_set[:size])
                    true_val = passive_test_set[:size]
                else:
                    model.fit(train_set[:size], proactive_train_set[:size])
                    true_val = proactive_test_set[:size]

                prediction = model.predict(test_set[:size])
                b_acc = balanced_accuracy_score(true_val, prediction)
                score = accuracy_score(true_val, prediction)
                balanced_scores.append(b_acc)
                scores.append(score)
                features.append(feature)
                sizes.append(size)
                models_df.append(model.sbert_label)
                
        pd.DataFrame.to_excel(pd.DataFrame({'scores': scores, 'balanced_scores': balanced_scores, 'features': features, 'sizes': sizes, 'model': models_df}), f'{out_path}{title}{model.sbert_label}.xlsx')

    predictive_data = pd.DataFrame({'scores': scores,
                                    'balanced_scores': balanced_scores,
                                    'features': features,
                                    'sizes': sizes,
                                    'model': models_df})

    pd.DataFrame.to_excel(predictive_data, f'{out_path}{title}.xlsx')

    colors = {'passive': 'blue', 'proactive': 'green'}
    symbols = ['o', 'P', '*', 'D', 'x', 'p']
    markers = {i: j for i, j in zip(sbert_models, symbols)}
    plt.scatter(predictive_data['model'].tolist(), predictive_data['balanced_scores'].tolist(),
                c=predictive_data['features'].map(colors))
    plt.title(title)
    plt.savefig(f'/home/ec2-user/environment/ACA_Public/{title}.png')
    plt.show()


ACA_Models = [ACAClf(i) for i in sbert_models]

train_and_predict(ACA_Models, title='ACA model depending on sberts Nsample=1200')
print(f'Process took: {time.time() - startTime}')

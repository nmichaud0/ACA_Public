import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
from sentence_transformers import SentenceTransformer
import xgboost as xgb

startTime = time.time()

data_path_local = '/Users/nizarmichaud/PycharmProjects/ACA_Public/joe_dutch_clean.xlsx'
data_path_aws = '/home/ec2-user/environment/ACA_Public/joe_dutch_clean.xlsx'

output_path_local = '/Users/nizarmichaud/PycharmProjects/ACA_Public/scores/predictive_data_'
output_path_aws = '/home/ec2-user/environment/ACA_Public/predictive_data_'

env = 'local'
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


class ACAClf:
    def __init__(self):

        self.soft_voting_clf = None
        self.fitted = False
        self.svc, self.rf, self.xgbclf, self.nn = None, None, None, None

    def fit(self,
            x: np.array,
            y: np.array):

        self.rf = RandomForestClassifier()
        svc = SVC(probability=True)
        self.xgbclf = xgb.XGBClassifier(eval_metric='auc', use_label_encoder=False)
        nn = MLPClassifier()

        # SVC Searching
        svc_param_grid = {'degree': np.arange(1, 10),
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

        svc_grid_search = RandomizedSearchCV(svc, svc_param_grid, cv=5, scoring='balanced_accuracy')
        svc_grid_search.fit(x, y)
        self.svc = SVC(degree=svc_grid_search.best_params_['degree'], gamma=svc_grid_search.best_params_['gamma'],
                       probability=True)

        # NN Searching

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

        self.soft_voting_clf = VotingClassifier(estimators=[('rf', self.rf), ('svc', self.svc), ('xgb', self.xgbclf),
                                                            ('nn', self.nn)],
                                                voting='soft')

        self.soft_voting_clf.fit(x, y)
        self.fitted = True

    def predict(self, x):

        if self.fitted:
            return self.soft_voting_clf.predict(x)
        else:
            raise Exception('Need to fit model before making predictions')

    def predict_proba(self, x):

        if self.fitted:
            return self.soft_voting_clf.predict_proba(x)
        else:
            raise Exception('Need to fit model before making predictions')

    def predict_for_each(self, x):

        if not self.fitted:
            raise Exception('Need to fit model before making predictions')

        return [est.predict(x) for est in self.soft_voting_clf.estimators_]

    def predict_proba_for_each(self, x):

        if not self.fitted:
            raise Exception('Need to fit model before making predictions')

        return [est.predict_proba(x) for est in self.soft_voting_clf.estimators_]


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
        #  also check if I want to weight some models


class MetaCLF:
    def __init__(self, models: list):

        self.soft_voting_clf = None
        self.fitted = False
        self.models = models

    def fit(self, sentences_local: np.array, y: np.array):

        estimators = []

        for index, model in enumerate(self.models):

            sbert_model = SentenceTransformer(model)
            embeddings = sbert_model.encode(sentences_local)

            new_aca = ACAClf()
            new_aca.fit(embeddings, y)

            estimators.append({'sbert': model, 'aca': new_aca})

        self.soft_voting_clf = MyVotingCLF(estimators)
        self.fitted = True

    def predict(self, x):

        if self.fitted:
            self.soft_voting_clf.predict(x)
        else:
            raise Exception('Need to fit model before making prediction')


sbert_models = ['distiluse-base-multilingual-cased-v1',
                'distiluse-base-multilingual-cased-v2',
                'paraphrase-multilingual-MiniLM-L12-v2',
                'paraphrase-multilingual-mpnet-base-v2']

# training_sizes = np.arange(100, len(proactive_train_set), step=20)
training_sizes = [100]
sbert_models = [sbert_models[0]]


def train_and_predict(model, title):

    scores = []
    balanced_scores = []
    features = []
    sizes = []

    for fn, feature in enumerate(['passive', 'proactive']):
        for size in training_sizes:
            print(f'Feature number:{fn + 1}/2; Size: {size}/{max(training_sizes)}')

            if feature == 'passive':
                model.fit(train_set[:size], passive_train_set[:size])
                true_val = passive_test_set[:size]
            else:
                model.fit(train_set[:size], proactive_train_set[:size])
                true_val = proactive_test_set[:size]

            prediction = model.predict(test_set)
            b_acc = balanced_accuracy_score(true_val, prediction)
            balanced_scores.append(b_acc)
            print(b_acc)
            features.append(feature)
            sizes.append(size)

        predictive_data = pd.DataFrame({'scores': scores,
                                        'balanced_scores': balanced_scores,
                                        'features': features,
                                        'sizes': sizes})

        pd.DataFrame.to_excel(predictive_data, f'{out_path}{title}.xlsx')

        colors = {'passive': 'blue', 'proactive': 'green'}
        plt.scatter(predictive_data['sizes'].tolist(), predictive_data['balanced_scores'].tolist(),
                    c=predictive_data['features'].map(colors))
        plt.title(title)
        plt.savefig(f'/home/ec2-user/environment/ACA_Public/{title}.png')
        plt.show()


MetaACA = MetaCLF(sbert_models)
train_and_predict(MetaACA, title='Meta ACA')
print(f'Process took: {time.time() - startTime}')

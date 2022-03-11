import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import xgboost as xgb
# import umap
from models_improved import ACAClassifier

local_paths = '/Users/nizarmichaud/PycharmProjects/ACA_Public/joe_dutch_clean.xlsx', \
              '/Users/nizarmichaud/PycharmProjects/ACA_Public/sequences.xlsx',\
              '/Users/nizarmichaud/PycharmProjects/ACA_Public/scores/predictive_data_',\
              '/Users/nizarmichaud/PycharmProjects/ACA_Public/embeddings_all-mpnet-base-v2.xlsx',\
              '/Users/nizarmichaud/PycharmProjects/ACA_Public/multilingual-embeddings.xlsx'

aws_paths = '/home/ec2-user/environment/ACA_Public/joe_dutch_clean.xlsx',\
            '/home/ec2-user/environment/ACA_Public/sequences.xlsx',\
            '/home/ec2-user/environment/ACA_Public/predictive_data_',\
            '/home/ec2-user/environment/ACA_Public/embeddings_all-mpnet-base-v2.xlsx',\
            '/home/ec2-user/environment/ACA_Public/multilingual-embeddings.xlsx'

env = 'aws'
token_transfo = 'transformers', 'multilingual'

if env == 'local':
    paths = local_paths
else:
    paths = aws_paths

if token_transfo[0] == 'token':
    train_test_data_path = paths[0]
elif token_transfo[1] == 'multilingual':
    train_test_data_path = paths[4]
else:
    train_test_data_path = paths[3]

startTime = time.time()

# Loading Data

sequences_df = pd.read_excel(train_test_data_path)
sequences_df = sequences_df.drop(columns=['Unnamed: 0'])

"""reducer = umap.UMAP(n_components=20, metric='euclidean')

sequences_df = pd.DataFrame(reducer.fit_transform(sequences_df))

"""
base_df = pd.read_excel(paths[0])
passive = base_df['passive'].to_numpy().astype(int)
proactive = base_df['proactive'].to_numpy().astype(int)

train_set, test_set = train_test_split(sequences_df, test_size=0.2, random_state=42)
passive_train_set, passive_test_set = train_test_split(passive, test_size=0.2, random_state=42)
proactive_train_set, proactive_test_set = train_test_split(proactive, test_size=0.2, random_state=42)

# models

training_sizes = np.arange(100, len(proactive_train_set), step=100)


def train_and_predict(model, title):

    scores = []
    balanced_scores = []
    features = []
    sizes = []

    for fn, feature in enumerate(['passive', 'proactive']):
        for size in training_sizes:
            print(f'Feature number:{fn+1}/2; Size: {size}/{max(training_sizes)}')

            if feature == 'passive':
                model.fit(train_set.iloc[:size], passive_train_set[:size])
                true_val = passive_test_set
            else:
                model.fit(train_set.iloc[:size], proactive_train_set[:size])
                true_val = proactive_test_set

            prediction = model.predict(test_set)
            scores.append(accuracy_score(true_val, prediction))
            balanced_scores.append(balanced_accuracy_score(true_val, prediction))
            features.append(feature)
            sizes.append(size)

    predictive_data = pd.DataFrame({'scores': scores,
                                    'balanced_scores': balanced_scores,
                                    'features': features,
                                    'sizes': sizes})
                                    
    pd.DataFrame.to_excel(predictive_data, f'{paths[2]}{title}.xlsx')

    colors = {'passive': 'blue', 'proactive': 'green'}
    plt.scatter(predictive_data['sizes'].tolist(), predictive_data['balanced_scores'].tolist(),
                c=predictive_data['features'].map(colors))
    plt.savefig('/home/ec2-user/environment/ACA_Public/trans_multi_aca_classifier.png')
    plt.title(title)
    plt.show()


# train_and_predict(xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False),
#                  title='Transformers – XGBoost')

# train_and_predict(RandomForestClassifier(), title='Transformers multilingual – RF')

"""
rf = RandomForestClassifier()
svc = SVC(probability=True)
xgb = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
nn = MLPClassifier((50, 20))

voting_clf = VotingClassifier(estimators=[('rf', rf), ('svc', svc), ('xgb', xgb), ('nn', nn)], voting='soft')
"""

train_and_predict(ACAClassifier(), title='Transformers multilingual – ACA Classifier')
print(f'Process took: {time.time() - startTime}')

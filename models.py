import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import xgboost as xgb

startTime = time.time()

# Loading Data

base_df = pd.read_excel('/home/ec2-user/environment/ACA_Public/joe_dutch_clean.xlsx')

sequences_df = pd.read_excel('/home/ec2-user/environment/ACA_Public/sequences.xlsx')

sequences_df = sequences_df.drop(columns=['Unnamed: 0'])

passive = base_df['passive'].to_numpy().astype(int)
proactive = base_df['proactive'].to_numpy().astype(int)

train_set, test_set = train_test_split(sequences_df, test_size=0.2, random_state=42)
passive_train_set, passive_test_set = train_test_split(passive, test_size=0.2, random_state=42)
proactive_train_set, proactive_test_set = train_test_split(proactive, test_size=0.2, random_state=42)

# models

training_sizes = np.arange(10, len(proactive_train_set), step=10)


def train_and_predict(model):

    scores = []
    balanced_scores = []
    features = []
    sizes = []

    for fn, feature in enumerate(['passive', 'proactive']):
        for size in training_sizes:
            print(f'Feature number:{fn}/2; Size: {size}/{max(training_sizes)}')

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
                                    
    pd.DataFrame.to_excel(predictive_data, '/home/ec2-user/environment/ACA_Public/predictive_data_xgboost.xlsx')

    colors = {'passive': 'blue', 'proactive': 'green'}
    plt.scatter(predictive_data['sizes'].tolist(), predictive_data['balanced_scores'].tolist(),
                c=predictive_data['features'].map(colors))
    plt.title(f'{str(model.__class__()).replace("()", "")}')
    plt.savefig('/home/ec2-user/environment/ACA_Public/xgb_fig.png')
    plt.show()


train_and_predict(xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False))
print(f'Process took: {time.time() - startTime}')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

curr_dir = os.getcwd()

data_path = os.path.realpath('joe_dutch_categorized_excel.xlsx')

df = pd.read_excel(data_path)

features = ['passive', 'proactive', 'destructive', 'somewhat']
features_of_interest = ['passive', 'proactive']

# df should be of len 1548: OK

# SkLearn preprocessing

cat_encoder = LabelBinarizer()
df_cat = df['category']
df_cat_1hot = cat_encoder.fit_transform(df_cat)
Onehot_df = pd.DataFrame(df_cat_1hot)
df = pd.concat([df, Onehot_df], axis=1)

df.rename(columns={0: "destructive",
                    1: 'passive',
                    2: 'proactive',
                    3: 'somewhat'}, inplace=True)

# Dropping low-represented categories: destructive & somewhat
df = df.drop(df[df.category == 'destructive'].index)
df = df.drop(df[df.category == 'somewhat'].index)
df = df.drop(columns=['destructive', 'somewhat'])

# Tokenizing text:

tokenizer = Tokenizer(num_words=None, oov_token='<OOV>')

sentences = df['text'].tolist()

tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

sentences_len = [len(i) for i in sequences]
max_len = max(sentences_len)

sequences_new = pad_sequences(sequences, maxlen=max_len, padding='post')

columns_df_w_sequences = [f'W{i}' for i in list(range(len(sequences_new[0])))]

sequences_df = pd.DataFrame(sequences_new, columns=columns_df_w_sequences)

df = pd.concat([df, sequences_df], axis=1)

import pandas as pd
from sentence_transformers import SentenceTransformer

df = '/home/ec2-user/environment/ACA_Public/joe_dutch_clean.xlsx'

sentences = df['text'].tolist()

#model

model = SentenceTransformer('all-mpnet-base-v2')

embeddings = model.encode(sentences)

print(type(embeddings))

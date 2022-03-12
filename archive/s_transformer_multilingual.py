import pandas as pd
from sentence_transformers import SentenceTransformer

path = '/home/ec2-user/environment/ACA_Public/joe_dutch_clean.xlsx'

df = pd.read_excel(path)

sentences = df['text'].tolist()

#model

model = SentenceTransformer('distiluse-base-multilingual-cased')

embeddings = model.encode(sentences)

embeddings_df = pd.DataFrame(data=embeddings)

pd.DataFrame.to_excel(embeddings_df, '/home/ec2-user/environment/ACA_Public/multilingual-embeddings.xlsx')

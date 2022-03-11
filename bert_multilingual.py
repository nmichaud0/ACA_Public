import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import time

startTime = time.time()

model_name = "bert-base-multilingual-cased"

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

df = pd.read_excel('/Users/nizarmichaud/PycharmProjects/ACA_Public/joe_dutch_clean.xlsx')
data = df['text'].tolist()

embeddings = np.array([])

for i in data:
    inputs = tokenizer(i, return_tensors='pt')
    outputs = model(**inputs)

    embeddings = np.append(embeddings, np.asarray(outputs[1]))

pd.DataFrame(embeddings).to_excel('bert_multilingual_embeddings.xlsx')

print('Process took:', time.time()-startTime)

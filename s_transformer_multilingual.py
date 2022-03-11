import pandas as pd
from transformers import BertTokenizer, BertForTokenClassification, BertModel, BertForSequenceClassification

path = '/home/ec2-user/environment/ACA_Public/joe_dutch_clean.xlsx'

df = pd.read_excel(path)

sentences = df['text'].tolist()

# model

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained("bert-base-multilingual-cased")

data_tokenized = tokenizer.tokenize(sentences)

encoded_input = tokenizer(sentences, return_tensors='pt', padding=True)

data_input = tokenizer.tokenize(sentences)

embeddings = model(**encoded_input).last_hidden_state.detach().numpy()[0]

embeddings_df = pd.DataFrame(data=embeddings)

pd.DataFrame.to_excel(embeddings_df, '/home/ec2-user/environment/ACA_Public/BERT-multilingual.xlsx')

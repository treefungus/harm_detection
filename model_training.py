import pandas as pd
import numpy as np
import joblib

import streamlit as st

data = pd.read_csv('sources/train.csv')

for harm in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    print(harm + ':')
    print(data.comment_text[data[harm] == 1].values[42] + '\n')

freq_harms = {}
for harm in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    freq_harms[harm] = len(data[data[harm] == 1])

freq_isharm = {}
vanilla_mask = (data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] == 0).all(axis=1)
freq_isharm['vanilla'] = len(data[vanilla_mask])
freq_isharm['all_harm'] = len(data) - len(data[vanilla_mask])

data['vanilla_mask'] = vanilla_mask.astype(int)

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

def clean_words(tokens):
	stops = stopwords.words('english')
	clean_tokens = [token.lower().strip() for token in tokens if token.lower() not in stops and token.isalnum()]

	return clean_tokens

#new_len = int(np.round(len(data)*0.1))
new_len = int(np.round(len(data)))
data_sample = data.iloc[0:new_len].copy()

data_sample['tokens'] = data_sample['comment_text'].apply(lambda x: word_tokenize(x))
data_sample['tokens'] = data_sample['tokens'].apply(lambda x: clean_words(x))

harm_corpus = []
vanilla_corpus = []

for comment in data_sample['tokens'][data['vanilla_mask'] != 1]:
    for token in comment:
        harm_corpus.append(token)

for comment in data_sample['tokens'][data['vanilla_mask'] == 1]:
    for token in comment:
        vanilla_corpus.append(token)
        
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

X = data_sample['tokens'].apply(lambda tokens: ' '.join(tokens))
y = data_sample['vanilla_mask']

X_train, X_valid, y_train, y_valid = train_test_split(X, y)

vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_valid_vectors = vectorizer.transform(X_valid)

model = LogisticRegression()

model.fit(X_train_vectors, y_train)

joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

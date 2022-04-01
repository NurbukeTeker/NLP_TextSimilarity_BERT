# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:39:01 2022

@author: nurbuketeker
"""
from sentence2meanpool import sentence2meanpooled,list2CSV
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
import nltk
nltk.download('punkt')

import pandas as pd

col_names = ["index","text","intent"]
rev = pd.read_csv(r"domain_train.csv",names=col_names, header=None)
del rev["index"]

rev.head()

df_atis = rev[rev["intent"] == "1"]
df_banking = rev[rev["intent"] == "2"]
df_acid = rev[rev["intent"] == "3"]
df_clinc = rev[rev["intent"] == "4"]


# https://towardsdatascience.com/bert-for-measuring-text-similarity-eec91c6bf9e1
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer('bert-base-nli-mean-tokens')

sentences_atis= list(df_atis["text"])
# sentence_embeddings_atis = model.encode(sentences_atis)

sentences_banking= list(df_banking["text"])
# sentence_embeddings_banking = model.encode(sentences_banking)

sentences_acid= list(df_acid["text"])
# sentence_embeddings_acid = model.encode(sentences_acid)

sentences_clinc= list(df_clinc["text"])
# sentence_embeddings_clinc = model.encode(sentences_clinc)

similiarity_atis_banking = []

for sentence1 in sentences_atis:
    for sentence2 in sentences_banking:                
        mean_pooled = sentence2meanpooled(sentence1)
        mean_pooled2 = sentence2meanpooled(sentence2)
        similarity = cosine_similarity(
            mean_pooled,
            mean_pooled2
        )
        if similarity[0][0] > 0.50 :
            similiarity_atis_banking.append([sentence1,sentence2,similarity[0][0]])
        
list2CSV(similiarity_atis_banking, "similiarity_atis_banking.csv")       
        

similiarity_atis_clinc = []

for sentence1 in sentences_atis:
    for sentence2 in sentences_clinc:                
        mean_pooled = sentence2meanpooled(sentence1)
        mean_pooled2 = sentence2meanpooled(sentence2)
        similarity = cosine_similarity(
            mean_pooled,
            mean_pooled2
        )
        if similarity[0][0] > 0.50 :
            similiarity_atis_clinc.append([sentence1,sentence2,similarity[0][0]])
       
list2CSV(similiarity_atis_clinc, "similiarity_atis_clinc.csv")       



similiarity_atis_acid = []

for sentence1 in sentences_atis:
    for sentence2 in sentences_clinc:                
        mean_pooled = sentence2meanpooled(sentence1)
        mean_pooled2 = sentence2meanpooled(sentence2)
        similarity = cosine_similarity(
            mean_pooled,
            mean_pooled2
        )
        if similarity[0][0] > 0.50 :
            similiarity_atis_acid.append([sentence1,sentence2,similarity[0][0]])
       
list2CSV(similiarity_atis_acid, "similiarity_atis_acid.csv")       

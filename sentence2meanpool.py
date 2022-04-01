# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 10:18:36 2022

@author: nurbuketeker
"""
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')


sentence1 =    "Three years later, the coffin was still full of Jello."
sentence2 =   "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go."


def sentence2meanpooled(sentence):
# initialize dictionary to store tokenized sentences
    tokens = {'input_ids': [], 'attention_mask': []}
    
    
        # encode each sentence and append to dictionary
    new_tokens = tokenizer.encode_plus(sentence, max_length=128,
                                           truncation=True, padding='max_length',
                                           return_tensors='pt')
    tokens['input_ids'].append(new_tokens['input_ids'][0])
    tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    # reformat list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    
    
    outputs = model(**tokens)
    outputs.keys()
    
    embeddings = outputs.last_hidden_state
    
    attention_mask = tokens['attention_mask']
    attention_mask.shape
    
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    
    masked_embeddings = embeddings * mask
    
    summed = torch.sum(masked_embeddings, 1)
    
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    
    mean_pooled = summed / summed_mask
    
    # convert from PyTorch tensor to numpy array
    mean_pooled = mean_pooled.detach().numpy()
    return mean_pooled
    



# from sklearn.metrics.pairwise import cosine_similarity

# mean_pooled = sentence2meanpooled(sentence1)
# mean_pooled2 = sentence2meanpooled(sentence2)

# calculate
# similiarity[0][0] = cosine_similarity(
#     mean_pooled,
#     mean_pooled2
# )


import numpy as np
  
def list2CSV(rows , filename):
           
    np.savetxt(filename, 
               rows,
               delimiter =", ", 
               fmt ='% s')

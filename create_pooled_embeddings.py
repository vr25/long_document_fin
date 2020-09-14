import gc
import numpy
import copy
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import torch
torch.cuda.empty_cache()
import os
import pandas as pd
import sys
import re
import numpy as np
import time
from transformers import RobertaModel, RobertaConfig
from transformers import LongformerTokenizer, LongformerModel, BertTokenizer, RobertaTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, AdamW, BertConfig, RobertaConfig
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer, AutoConfig
import datetime
import random
from sklearn.metrics import matthews_corrcoef

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

start = time.time()

d = 768

np_sum = np.empty((0,d))
np_max = np.empty((0,d))
np_mean = np.empty((0,d))


df = pd.read_csv("new_all_2436_mda_roa.csv")

# Get the lists of sentences and their labels.
sentences = df['mda']
#np_hist = np.asarray(df['prev_roa'].tolist())
#np_roa = np.asarray(df['roa'].tolist())

max_length = 4096

tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')

sentences = [tokenizer.tokenize(t) for t in sentences]

tokenized_text = []

for t in sentences:
    tokenized_text.append([t[i:i+max_length] for i in range(0, len(t), max_length)])


# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

for t in tokenized_text:

    input_ids_list = []
    attention_masks_list = []

    for i in range(len(t)):
        encoded_dict = tokenizer.encode_plus(
        t[i],                      # Sentence to encode.
        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
        max_length = 4096,           # Pad & truncate all sentences.
        pad_to_max_length = True,
        return_attention_mask = True,   # Construct attn. masks.
        return_tensors = 'pt',     # Return pytorch tensors.
        )
    
        # Add the encoded sentence to the list.    
        input_ids_list.append(encoded_dict['input_ids'])
                                                                
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks_list.append(encoded_dict['attention_mask'])

        
    input_ids.append(input_ids_list)
    attention_masks.append(attention_masks_list)


model = AutoModel.from_pretrained('allenai/longformer-base-4096') 

    
# For each batch of training data...
for i in range(len(sentences)):

    input_ids_ = torch.stack(input_ids[i], dim=0)
    attention_mask_ = torch.stack(attention_masks[i], dim=0)
           
    d = TensorDataset(input_ids_, attention_mask_) 

    cls_vec = []

    for v, b in enumerate(d):

        _, last_cls_ = model(input_ids=b[0], attention_mask=b[1])
        #last_cls_ = outputs[0][:,0,:]

        print("LAST_CLS: ", last_cls_.shape)
        
        cls_vec.append(last_cls_)

    cls_vec = torch.stack(cls_vec, dim=0)
    
    cls_vec_sum = torch.sum(cls_vec, dim=0)
    cls_vec_sum = cls_vec_sum.detach().numpy()

    cls_vec_max = torch.max(cls_vec, dim=0)
    cls_vec_max = cls_vec_max[0].detach().numpy()

    cls_vec_mean = torch.mean(cls_vec, dim=0)
    cls_vec_mean = cls_vec_mean.detach().numpy()

    print("LAST_CLS: ", cls_vec_mean.shape)

    np_sum = np.append(np_sum, cls_vec_sum, axis=0)
    np_max = np.append(np_max, cls_vec_max, axis=0)
    np_mean = np.append(np_mean, cls_vec_mean, axis=0)


with open('lf_4096_np_sum.npy', 'wb') as f1:
    np.save(f1, np_sum)

with open('lf_4096_np_max.npy', 'wb') as f2:
    np.save(f2, np_max)

with open('lf_4096_np_mean.npy', 'wb') as f3:
    np.save(f3, np_mean)

'''
with open('np_hist.npy', 'wb') as f4:
    np.save(f4, np_hist)

with open('np_roa.npy', 'wb') as f5:
    np.save(f5, np_roa)
'''

print("Total execution time: ", time.time() - start)

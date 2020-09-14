import os
import random
import sys
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import transformers
from transformers import AutoModel, AutoTokenizer 
from transformers import AdamW
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
start = time.time()

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


class BERT_Arch(nn.Module):

    def __init__(self, bert):

        super(BERT_Arch, self).__init__()

        self.bert = bert 
      
        # dropout layer
        self.dropout = nn.Dropout(0.5)
      
        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768,512)
      
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(513,1)

        #softmax activation function
        #self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask, hist):

        #pass the inputs to the model  
        _, cls_hs = self.bert(sent_id, attention_mask=mask)
      
        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        #print("hist: ", hist.shape)

        #print("Before x: ", x.shape)

        x = torch.cat((x, hist.unsqueeze(1)), dim=1)

        x = self.dropout(x)

        #print("After x: ", x.shape)

        # output layer
        x = self.fc2(x)
      
        # apply softmax activation
        #x = self.softmax(x)

        return x


# function to train the model
def train():

    model.train()

    total_loss, total_accuracy = 0, 0
  
    # empty list to save model predictions
    total_preds=[]
  
    # iterate over batches
    for step,batch in enumerate(train_dataloader):
    
        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]
 
        sent_id, mask, hist, labels = batch

        # clear previously calculated gradients 
        model.zero_grad()        

        # get model predictions for the current batch
        preds = model(sent_id, mask, hist)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    #returns the loss and predictions
    return avg_loss, total_preds



# function for evaluating the model
def evaluate():

    print("\nEvaluating...")
  
    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0
  
    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step,batch in enumerate(val_dataloader):
    
        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
      
            # Calculate elapsed time in minutes.
            #elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, hist, labels = batch

        # deactivate autograd
        with torch.no_grad():
      
            # model predictions
            preds = model(sent_id, mask, hist)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds,labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader) 

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds


def test(ckpt_model):

    # empty list to save the model predictions
    total_preds = []

    for step,batch in enumerate(test_dataloader):

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, hist, labels = batch

        with torch.no_grad():
            preds = ckpt_model(sent_id, mask, hist)
            preds = preds.detach().cpu().numpy()
            # append the model predictions
            total_preds.append(preds)


    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return total_preds

# specify GPU
device = torch.device("cuda")

'''
df = pd.read_csv("spamdata_v2.csv", index_col=False)
df = df[:2436]
df['label'] = np.random.uniform(0, 1, len(df))
df['hist'] = np.random.rand(len(df),1)
print(df.head(5))
'''

df = pd.read_csv("new_all_2436_mda_roa.csv", index_col=False)
#df2_ = pd.read_csv("roa_data_2006_2005_nonscaled.csv", index_col=False)
#df = pd.merge(df1_, df2_, on='cik_year')
#df = df[:20]
#print(df.head(1))
#print(df.columns)
#sys.exit(0)

# split train dataset into train, validation and test sets
train_text, temp_text, train_hist, temp_hist, train_labels, temp_labels = train_test_split(df['mda'], df['prev_roa'], df['roa'], 
                                                                    random_state=2018, 
                                                                    test_size=0.3) 


val_text, test_text, val_hist, test_hist, val_labels, test_labels = train_test_split(temp_text, temp_hist, temp_labels, 
                                                                random_state=2018, 
                                                                test_size=0.5) 

# import BERT-base pretrained model
bert = AutoModel.from_pretrained('allenai/longformer-base-4096') #bert-base-uncased')

# Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096') #bert-base-uncased')

'''
# sample data
text = ["this is a bert model tutorial", "we will fine-tune a bert model"]
for t in text:
    marked_text = "[CLS] " + t + " [SEP]"
    print("BERT tokens: ", tokenizer.tokenize(marked_text))

# encode text
sent_id = tokenizer.batch_encode_plus(text, add_special_tokens=True, padding=True)

# output
print(sent_id)
'''

# get length of all the messages in the train set
seq_len = [len(i.split()) for i in train_text]

# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = 4096,
    pad_to_max_length=True,
    truncation=True
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = 4096,
    pad_to_max_length=True,
    truncation=True
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = 4096,
    pad_to_max_length=True,
    truncation=True
)

## convert lists to tensors

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_hist = torch.tensor(train_hist.tolist())
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_hist = torch.tensor(val_hist.tolist())
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_hist = torch.tensor(test_hist.tolist())
test_y = torch.tensor(test_labels.tolist())

#define a batch size
batch_size = 1

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_hist, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_hist, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

# wrap tensors
test_data = TensorDataset(test_seq, test_mask, test_hist, test_y)

# sampler for sampling the data during training
test_sampler = SequentialSampler(test_data)

# dataLoader for validation set
test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size=batch_size)

# freeze all the parameters
for param in bert.parameters():
    param.requires_grad = True #False

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)

# define the optimizer
optimizer = AdamW(model.parameters(),
                  lr = 1e-2)          # learning rate

# define the loss function
cross_entropy  = nn.MSELoss()  #CrossEntropyLoss()

# number of training epochs
epochs = 5

# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

#for each epoch
for epoch in range(epochs):

    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
    #train model
    train_loss, _ = train()
    
    #evaluate model
    valid_loss, _ = evaluate()
    
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss

        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), 'saved_epoch5_doc_long_4096_weights.pt')
    
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')



#load weights of best model
path = 'saved_epoch5_doc_long_4096_weights.pt'
model.load_state_dict(torch.load(path))

ckpt_model = model

# get predictions for test data
preds = np.asarray(test(ckpt_model))

'''
with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device), test_hist.to(device))
    preds = preds.detach().cpu().numpy()
'''

test_y = test_y.numpy()

mse = mean_squared_error(test_y, preds)

test_error = pd.DataFrame()
test_error['test_y'] = test_y.tolist()
test_error['preds'] = [p[0] for p in preds.tolist()]
test_error.to_csv("error_epoch5_doc_long_4096.csv", index=False)

print("mse: ", mse)

mse_file = open("mse_epoch5_doc_bert_4096.txt", "w")
mse_file.write(str(mse))
mse_file.close()

print("Total executon time: ", time.time() - start)

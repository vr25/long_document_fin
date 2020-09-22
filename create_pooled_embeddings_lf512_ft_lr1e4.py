from pytorch_memlab import MemReporter
import gc
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
from transformers import LongformerTokenizer, AutoModel, AutoTokenizer 
from transformers import AdamW
from torch.cuda.amp import autocast
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
start = time.time()

torch.cuda.empty_cache()

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
        self.dropout = nn.Dropout(0.1)
      
        # relu activation function
        self.relu =  nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(769,1) #512)
      
        # dense layer 2 (Output layer)
        #self.fc2 = nn.Linear(513,1)

    #define the forward pass
    def forward(self, sent_id, mask, hist):

        torch.cuda.empty_cache()

        d = 768
        cls_vec = torch.zeros((1,d), requires_grad=True).to(device)
        chunk_num = len(sent_id)
        print("# chunks: ", chunk_num)

        for i in range(len(sent_id)):

            print("chunk id: ", i)

            ip_id = torch.tensor(sent_id[i]).unsqueeze(0).to(device)
            attn_mask = torch.tensor(mask[i]).unsqueeze(0).to(device)

            #pass the inputs to the model  
            _, cls_hs = self.bert(input_ids=ip_id, attention_mask=attn_mask)

            #cls_vec.append(cls_hs)
            cls_vec = torch.add(cls_vec, cls_hs)
            print("Adding: ")
            
            #print("cls_vec on adding: ", cls_vec) 
            
            #.append(cls_hs.cpu().data.numpy())
            
            #print("cls_vec: ", cls_vec.shape, type(cls_vec))

            del cls_hs
            gc.collect()
            torch.cuda.empty_cache()

        #cls_vec = torch.stack(cls_vec, dim=0)           
        #cls_vec_sum = torch.mean(cls_vec, dim=0)

        cls_vec = torch.div(cls_vec, chunk_num)
        #cls_vec = cls_vec.squeeze(0)
        #print("cls_vec: ", type(cls_vec))
        #sys.exit(0)
        
        #cls_vec = np.vstack(cls_vec)
        #cls_vec = np.mean(cls_vec, axis=0)
        #cls_vec = torch.tensor(cls_vec).unsqueeze(0).to(device)

        #x = self.fc1(cls_vec)

        #del cls_vec
        #torch.cuda.empty_cache()

        #x = self.relu(x)

        #x = self.dropout(x)

        #print("cls_vec: ", cls_vec.shape)
        #print("x: ", x.shape)
        #sys.exit(0)

        x = cls_vec
        hist = hist.unsqueeze(0).unsqueeze(0)

        del cls_vec
        torch.cuda.empty_cache()

        print("x shape: ", x.shape)
        print("hist shape: ", hist, hist.shape)

        x = torch.cat((x, hist), dim=1)

        print("After concat hist: x shape: ", x.shape)

        x = self.dropout(x)

        # output layer
        y = self.fc1(x)

        del x
        torch.cuda.empty_cache()

        #print("final o/p: ", y.shape)

        gc.collect()

        return y


# function to train the model
def train():

    model.train()

    #torch.cuda.empty_cache()

    total_loss, total_accuracy = 0, 0
  
    # empty list to save model predictions
    total_preds = []
  
    # iterate over list of documents
    for i in range(len(train_seq)):

        print("Train doc id: ", i)

        sent_id = train_seq[i]
        mask = train_mask[i]
        hist = train_hist[i] 
        labels = train_y[i].unsqueeze(0).unsqueeze(0)

        # clear previously calculated gradients 
        model.zero_grad()        

        with autocast():
            # get model predictions for the current batch
            preds = model(sent_id, mask, hist)

            #print('inside train: ', preds.shape)

            # compute the loss between actual and predicted values
            loss = mse_loss(preds, labels)

            # model predictions are stored on GPU. So, push it to CPU
            preds = preds.detach().cpu().numpy()

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_seq)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    '''
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            del obj

    torch.cuda.empty_cache()
    '''

    gc.collect()
    torch.cuda.empty_cache()

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

    # iterate over list of documents
    for i in range(len(val_seq)):

        print("val doc id: ", i)

        sent_id = val_seq[i]
        mask = val_mask[i]
        hist = val_hist[i]
        labels = val_y[i].unsqueeze(0).unsqueeze(0)

        # deactivate autograd
        with torch.no_grad():
      
            with autocast():
            # model predictions
                preds = model(sent_id, mask, hist)

                # compute the validation loss between actual and predicted values
                loss = mse_loss(preds,labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_seq) 

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds


def test():

    # empty list to save the model predictions
    total_preds = []

    for i in range(len(test_seq)):

        print('test doc id: ', i)

        sent_id = test_seq[i]
        mask = test_mask[i]
        hist = test_hist[i]
        labels = test_y[i].unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            with autocast():
                preds = model(sent_id, mask, hist)
            preds = preds.detach().cpu().numpy()
            # append the model predictions
            total_preds.append(preds)


    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return total_preds

# specify GPU
device = torch.device("cuda")

df = pd.read_csv("new_all_2436_mda_roa.csv", index_col=False)
df = df[:10]

max_length = 510 #append two [CLS] and [SEP] tokens to make 512

# split train dataset into train, validation and test sets
train_text, temp_text, train_hist, temp_hist, train_labels, temp_labels = train_test_split(df['mda'], df['prev_roa'], df['roa'], 
                                                                    random_state=2018, 
                                                                    test_size=0.3) 


val_text, test_text, val_hist, test_hist, val_labels, test_labels = train_test_split(temp_text, temp_hist, temp_labels, 
                                                                random_state=2018, 
                                                                test_size=0.5) 
bert_path = os.getcwd()

# import BERT-base pretrained model
bert = AutoModel.from_pretrained(bert_path + "/" + 'longformer-base-4096/') #bert-base-uncased')

# Load the BERT tokenizer
tokenizer = LongformerTokenizer.from_pretrained(bert_path + "/" + 'longformer-base-4096/') #bert_path + "/" + 'longformer-base-4096/') #bert-base-uncased')

#TRAIN
# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    add_special_tokens=False
)

#Extract input ids
train_seq_ = tokens_train['input_ids']
#Split each document into 510 tokens
train_seq = [[train_seq_[j][i:i + max_length] for i in range(0, len(train_seq_[j]), max_length)] for j in range(len(train_seq_))]
#Add [CLS], [SEP] and [PAD] tokens
train_seq = [[[0] + train_seq[j][i] + [2] if len(train_seq[j][i]) == max_length else train_seq[j][i] + [1] * (max_length+2-len(train_seq[j][i])) for i in range(len(train_seq[j]))] for j in range(len(train_seq))]

'''
#print all sublists of documents lengths
print("train_seq")
for i in range(len(train_seq)):
    for j in range(len(train_seq[i])):
        print("i j", i, j, len(train_seq[i][j]))

#print(train_seq[5][0], len(train_seq[5][0]))
#print("# docs: ", len(train_seq))
'''

#Extract attention masks
train_mask_ = tokens_train['attention_mask']
#Split each document into 510 tokens
train_mask = [[train_mask_[j][i:i + max_length] for i in range(0, len(train_mask_[j]), max_length)] for j in range(len(train_mask_))]
#Add [1] for attention and [0] for [PAD]
train_mask = [[[1] + train_mask[j][i] + [1] if len(train_mask[j][i]) == max_length else train_mask[j][i] + [0] * (max_length+2-len(train_mask[j][i])) for i in range(len(train_mask[j]))] for j in range(len(train_mask))]

'''
#print all sublists of documents lengths
print("train_mask")
for i in range(len(train_mask)):
    for j in range(len(train_mask[i])):
        print("i j", i, j, len(train_mask[i][j]))

sys.exit(0)
#train_dataloader = zip(train_seq, train_mask, train_hist, train_y)
'''


#VALIDATION
# tokenize and encode sequences in the val set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    add_special_tokens=False
)

#Extract input ids
val_seq_ = tokens_val['input_ids']
#Split each document into 510 tokens
val_seq = [[val_seq_[j][i:i + max_length] for i in range(0, len(val_seq_[j]), max_length)] for j in range(len(val_seq_))]
#Add [CLS], [SEP] and [PAD] tokens
val_seq = [[[0] + val_seq[j][i] + [2] if len(val_seq[j][i]) == max_length else val_seq[j][i] + [1] * (max_length+2-len(val_seq[j][i])) for i in range(len(val_seq[j]))] for j in range(len(val_seq))]


#Extract attention masks
val_mask_ = tokens_val['attention_mask']
#Split each document into 510 tokens
val_mask = [[val_mask_[j][i:i + max_length] for i in range(0, len(val_mask_[j]), max_length)] for j in range(len(val_mask_))]
#Add [1] for attention and [0] for [PAD]
val_mask = [[[1] + val_mask[j][i] + [1] if len(val_mask[j][i]) == max_length else val_mask[j][i] + [0] * (max_length+2-len(val_mask[j][i])) for i in range(len(val_mask[j]))] for j in range(len(val_mask))]


#TEST
# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    add_special_tokens=False
)

#Extract input ids
test_seq_ = tokens_test['input_ids']
#Split each document into 510 tokens
test_seq = [[test_seq_[j][i:i + max_length] for i in range(0, len(test_seq_[j]), max_length)] for j in range(len(test_seq_))]
#Add [CLS], [SEP] and [PAD] tokens
test_seq = [[[0] + test_seq[j][i] + [2] if len(test_seq[j][i]) == max_length else test_seq[j][i] + [1] * (max_length+2-len(test_seq[j][i])) for i in range(len(test_seq[j]))] for j in range(len(test_seq))]


#Extract attention masks
test_mask_ = tokens_test['attention_mask']
#Split each document into 510 tokens
test_mask = [[test_mask_[j][i:i + max_length] for i in range(0, len(test_mask_[j]), max_length)] for j in range(len(test_mask_))]
#Add [1] for attention and [0] for [PAD]
test_mask = [[[1] + test_mask[j][i] + [1] if len(test_mask[j][i]) == max_length else test_mask[j][i] + [0] * (max_length+2-len(test_mask[j][i])) for i in range(len(test_mask[j]))] for j in range(len(test_mask))]

train_hist = torch.tensor(train_hist.tolist()).to(device)
train_y = torch.tensor(train_labels.tolist()).to(device)

val_hist = torch.tensor(val_hist.tolist()).to(device)
val_y = torch.tensor(val_labels.tolist()).to(device)

test_hist = torch.tensor(test_hist.tolist()).to(device)
test_y = torch.tensor(test_labels.tolist()).to(device)

# freeze all the parameters
for param in bert.parameters():
    param.requires_grad = True

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)

# verbose shows how storage is shared across multiple Tensors
reporter = MemReporter(model)
reporter.report(verbose=True)

# define the optimizer
optimizer = AdamW(model.parameters(),
                  lr = 1e-3)          # learning rate

# define the loss function
mse_loss  = nn.MSELoss()  

# number of training epochs
epochs = 2

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
        torch.save(model_to_save.state_dict(), 'saved_weights_dp1_lr1e3_lf512_chunk_ft.pt')
    
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')



del model
torch.cuda.empty_cache()

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)

#load weights of best model
path = 'saved_weights_dp1_lr1e3_lf512_chunk_ft.pt'
model.load_state_dict(torch.load(path))

# get predictions for test data
preds = np.asarray(test())

test_y = test_y.cpu().data.numpy()

mse = mean_squared_error(test_y, preds)

test_error = pd.DataFrame()
test_error['test_y'] = test_y.tolist()
test_error['preds'] = [p[0] for p in preds.tolist()]
test_error.to_csv("error_dp1_lr1e3_lf512_chunk_ft.csv", index=False)

print("mse: ", mse)

mse_file = open("mse_dp1_lr1e3_lf512_chunk_ft.txt", "w")
mse_file.write(str(mse))
mse_file.close()

print("Total execution time: ", time.time() - start)

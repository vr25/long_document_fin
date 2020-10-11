import optuna
import random
import numpy as np
import os
import sys
import csv
import pandas as pd
import time
import torch
import torch.nn as nn
from transformers import AdamW
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import mean_squared_error

csv.field_size_limit(sys.maxsize) 

start = time.time()

torch.cuda.empty_cache()

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

class FT_Arch(nn.Module):

    def __init__(self):
        super(FT_Arch, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, ip):

        x = self.fc1(ip)
        x = self.relu(x)
        x = self.dropout(x)
        #x = torch.cat((x, hist), dim=1)
        x = self.dropout(x)
        y = self.fc2(x)

        return y


def train(model, device, train_dataloader, optimizer):

    model.train()

    total_loss = 0
          
    # iterate over list of documents
    for i, b in enumerate(train_dataloader):

        ip = b[0].to(device)
        label = b[1].to(device)

        # clear previously calculated gradients 
        model.zero_grad()    
        
        mse_loss = nn.MSELoss()     

        with autocast():
            # get model predictions for the current batch
            preds = model(ip)

            # compute the loss between actual and predicted values
            loss = mse_loss(preds, label)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    #returns the loss
    return avg_loss


# function for evaluating the model
def evaluate(model, device, val_dataloader):

    print("\nEvaluating...")
      
    # deactivate dropout layers
    model.eval()

    total_loss = 0

    mse_loss  = nn.MSELoss()
                    
    # iterate over list of documents
    for i, b in enumerate(val_dataloader):

        ip = b[0].to(device)
        label = b[1].to(device)

        # deactivate autograd
        with torch.no_grad():
                                                                
            with autocast():
                # model predictions
                preds = model(ip)

                # compute the validation loss between actual and predicted values
                loss = mse_loss(preds,label)

            total_loss = total_loss + loss.item()

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader) 

    return avg_loss


def test(model, device, test_dataloader):

    # empty list to save the model predictions
    total_preds = []

    for i, b in enumerate(test_dataloader):

        ip = b[0].to(device)
        label = b[1].to(device)

        with torch.no_grad():
            with autocast():
                preds = model(ip)
            preds = preds.detach().cpu().numpy()
            # append the model predictions
            total_preds.append(preds)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return total_preds


def chunk_train(trial):
    X = torch.Tensor(np.load("np_mean.npy")) 
    y = torch.Tensor(np.load("np_roa.npy"))

    X_train = X[:1705]
    X_val = X[1705:1705+243]
    X_test = X[1705+243:]

    y_train = y[:1705]
    y_val = y[1705:1705+243]
    y_test = y[1705+243:]

    '''
    X_train = torch.Tensor("train_vol_lf512_np_mean.npy")
    y_train = torch.Tensor(np.load("train_vol_np_logvol_plus.npy"))

    X_val = torch.Tensor(np.load("val_vol_lf512_np_mean.npy"))  
    y_val = torch.Tensor(np.load("val_vol_np_logvol_plus.npy"))  

    X_test = torch.Tensor(np.load("test_vol_lf512_np_mean.npy"))
    y_test = torch.Tensor(np.load("test_vol_np_logvol_plus.npy")) 

    X_train = X_train[:10]
    y_train = y_train[:10]

    X_val = X_val[:2]
    y_val = y_val[:2]

    X_test = X_test[:4]
    y_test = y_test[:4]
    '''


    train_data = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_data, batch_size=1)

    val_data = TensorDataset(X_val, y_val)    
    val_dataloader = DataLoader(val_data, batch_size=1)

    test_data = TensorDataset(X_test, y_test) 
    test_dataloader = DataLoader(test_data, batch_size=1)

    # specify GPU
    device = torch.device("cuda")

    model = FT_Arch()

    # push the model to GPU
    model = model.to(device)

    lr = trial.suggest_loguniform('lr', 1e-6, 1e-1)
    #momentum = trial.suggest_uniform('momentum', 0.4, 0.99)

    # define the optimizer
    optimizer = AdamW(model.parameters(),
                  lr = lr)
     #             momentum = momentum)          # learning rate

    # define the loss function
    mse_loss  = nn.MSELoss()  

    # number of training epochs
    epochs = 20

    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses=[]
    valid_losses=[]

    #for each epoch
    for epoch in range(epochs):

        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
            
        #train model
        train_loss = train(model, device, train_dataloader, optimizer)
                        
        #evaluate model
        valid_loss = evaluate(model, device, val_dataloader)
                                    
        #save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), 'saved_weights_lf512_chunk.pt')
                                                                            
        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
                                                                                            
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

    del model
    torch.cuda.empty_cache()

    model = FT_Arch()

    # push the model to GPU
    model = model.to(device)

    #load weights of best model
    path = 'saved_weights_lf512_chunk.pt'
    model.load_state_dict(torch.load(path))

    # get predictions for test data
    preds = np.asarray(test(model, device, test_dataloader))

    y_test = y_test.cpu().data.numpy()

    mse = mean_squared_error(y_test, preds)

    test_error = pd.DataFrame()
    test_error['test_y'] = y_test.tolist()
    test_error['preds'] = [p[0] for p in preds.tolist()]
    test_error.to_csv("error_lf512_chunk.csv", index=False)

    print("mse: ", mse)

    return mse

if __name__ == '__main__':
    
    study = optuna.create_study() 
    study.optimize(chunk_train, n_trials=20)  
    
    print("Best trial:")
    trial = study.best_trial

    print("--Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("Total execution time: ", time.time() - start)

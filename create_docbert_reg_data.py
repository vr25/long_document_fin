import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("new_all_2436_mda_roa.csv", index_col=False)
#df = df[:10]

# split train dataset into train, validation and test sets
train_text, temp_text, train_labels, temp_labels = train_test_split(df['mda'], df['roa'], 
                                                                    random_state=2018, 
                                                                    test_size=0.3) 


val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
                                                                random_state=2018, 
                                                                test_size=0.5) 

print("val_text: ", type(val_text))

df_train = pd.DataFrame()
df_val = pd.DataFrame()
df_test = pd.DataFrame()

df_train['roa'] = train_labels
df_train['mda'] = train_text

df_val['roa'] = val_labels
df_val['mda'] = val_text

df_test['roa'] = test_labels
df_test['mda'] = test_text

df_train.to_csv('train.tsv', index=False, header=False, sep='	')
df_val.to_csv('dev.tsv', index=False, header=False, sep='	')
df_test.to_csv('test.tsv', index=False, header=False, sep='	')
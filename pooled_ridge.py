import torch
import os
import sys
import csv
import numpy as np 
import pandas as pd 
from sklearn.metrics import mean_squared_error
import re
import numpy as np
import time
from sklearn.model_selection import cross_val_score
from multiscorer import MultiScorer
from sklearn.kernel_ridge import KernelRidge

csv.field_size_limit(sys.maxsize)

start = time.time()

np_mean = np.load('lf_4096_np_sum.npy')
np_hist = np.load('np_hist.npy')
np_roa = np.load('np_roa.npy')

print("mean/hist/roa shapes: ", np_mean.shape, np_hist.shape, np_roa.shape)

np_mean = np.append(np_mean, np_hist.reshape(-1,1), axis=1)

#Normalize feature vector
'''X_vect = np_mean

#Normalize X_vect matrix for 'sum'
xmax, xmin = X_vect.max(), X_vect.min()
X_vect = (X_vect - xmin)/(xmax - xmin)
'''

scorer = MultiScorer({'mse' : (mean_squared_error, {})})

#KernelRidgeRegression model - default degree=3
model = KernelRidge(kernel='poly', alpha=0.1, gamma=0.1)

# Perform 10-fold cross validation
scores = cross_val_score(model, np_mean, np_roa, cv=10, scoring=scorer)
results = scorer.get_results()

final_scores = []

for metric_name in results.keys():
    average_score = np.average(results[metric_name])
    print('%s : %f' % (metric_name, average_score))
    final_scores.append(average_score)

print("Total execution time: ", time.time() - start)
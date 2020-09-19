import pandas as pd
import csv
import numpy as np
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from multiscorer import MultiScorer
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.kernel_ridge import KernelRidge


def compute(X_vect,y):

    model = LinearRegression().fit(X_vect, y) 

    scorer = MultiScorer({'mse' : (mean_squared_error, {})})

    scores = cross_val_score(model, X_vect, y, scoring=scorer)
    results = scorer.get_results()

    final_scores = []

    for metric_name in results.keys():
        average_score = np.average(results[metric_name])
        final_scores.append(average_score)
    
    print(final_scores[0]) 

df = pd.read_csv('new_all_2436_mda_roa.csv', usecols=['roa', 'prev_roa']) 

X = np.array(df['prev_roa']).reshape(-1,1)

y = np.array(df['roa'])

compute(X,y)

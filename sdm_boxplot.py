import pandas as pd
import csv
import numpy as np
from matplotlib import pyplot as plt
import sys

def data_stats(v):
	mean_ = v.mean()
	std_ = v.std()
	min_ = v.min()
	max_ = v.max()

	return mean_, std_, min_, max_

df1 = pd.read_csv('roa_data_nonscaled.csv') 
#print(df1.head(10))
print("len df1: ", len(df1))
data = []
d1 = df1['roa'].tolist()
d2 = df1['prev_roa'].tolist()
data = d1 + d2
print("data: ", len(data))

x = pd.Series(data)
print("Before len(x): ", len(x))
x = x[x.between(x.quantile(.05), x.quantile(.95))] 
print("After len(x): ", len(x))
#sys.exit(0)

# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)
xticklabels = ['ROA']

# Create the boxplot
bp = ax.boxplot(data)
ax.set_xticklabels(xticklabels)
plt.xticks(rotation=45, ha="right", fontsize=15)
plt.yticks(fontsize=15)
plt.title('min-max scaled distribution of ROA', fontsize=15)
# Save the figure
fig.savefig('sdm_boxplot_roa.png', bbox_inches='tight')
plt.show()
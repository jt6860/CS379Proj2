# -*- coding: utf-8 -*-
"""
Created on Tue Jan 9 13:22:41 2024

@author: John Torres
Course: CS379 - Machine Learning
Project 1a: Used Car Price Prediction
Supervised Algorithm: Decision Tree
"""

# Import pandas for data ingestion and modification.
import pandas as pd
import matplotlib.pyplot as plt

# Read from source CSV
data = pd.read_csv('bank-additional-full.csv', sep = ';')

# Split into categorical and numerical columns.
cat_data_columns = [column_name for column_name in data if data[column_name].dtype == 'O']
num_data_columns = list(set(data.columns) - set(cat_data_columns))

print(data[cat_data_columns])

print(data[num_data_columns])
print(data[num_data_columns].corr())

for col in data[cat_data_columns]:
    counts = data[col].value_counts().sort_index()
    if len(counts) > 10:
      fig = plt.figure(figsize=(30, 10))
    else:
      fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    counts.plot.bar(ax = ax, color='steelblue')
    ax.set_title(col + ' counts')
    ax.set_xlabel(col) 
    ax.set_ylabel("Frequency")
plt.show()
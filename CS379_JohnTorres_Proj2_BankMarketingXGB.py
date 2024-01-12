# -*- coding: utf-8 -*-
"""
Created on Tue Jan 9 13:22:41 2024

@author: John Torres
Course: CS379 - Machine Learning
Project 2: Bank Marketing Dataset Predictions with Gradient Boosted Tree
Supervised Algorithm: Gradient Boosted Tree
"""

# Import pandas for data ingestion and modification.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

# Read from source CSV.
data = pd.read_csv('bank-additional-full.csv', sep = ';')

# Drop irrelevant columns.
data.drop(['day_of_week', 'contact', 'month'], axis=1, inplace = True)

# Change column used for prediction from yes/no to 1/0.
data['y'] = data['y'].apply(lambda x: 0 if x=='no' else (1 if x=='yes' else -1))

# Separate the predicting column ('y') from the remainder of the dataset.
X = data.drop('y', axis=1)
Y = data['y']

# Change categorical values to indicator values.
X  = pd.get_dummies(X, drop_first = True)

# Scale the data.
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into test and train datasets.
x_train, x_test, y_train, y_test = train_test_split(X, Y)

# Instantiate Extreme Gradient Boosting Classifier model and fit with training data.
xgb = XGBClassifier()
xgb.fit(x_train,y_train)

# Make predictions using the trained classifier.
xgb_y_test_pred = xgb.predict(x_test)
xgb_y_train_pred = xgb.predict(x_train)

# Compare predictions to actuals and get accuracy score.
xgb_accuracy_test = accuracy_score(y_test, xgb_y_test_pred)
xgb_accuracy_train =  accuracy_score(y_train, xgb_y_train_pred)

# Print Train and Test accuracy scores.        
print('XGB Train accuracy is:', xgb_accuracy_train )
print('XGB Test accuracy is:', xgb_accuracy_test )
print()

# Compare predictions to actuals and get f1-score, precision, and recall scores.
xgb_f1 = f1_score(y_test, xgb_y_test_pred)
xgb_precision = precision_score(y_test, xgb_y_test_pred)
xgb_recall = recall_score(y_test, xgb_y_test_pred) 

# Print f1-score, precision, and recall scores.
print("XGB F score is:", xgb_f1 )
print("XGB Precision is:", xgb_precision)
print("XGB Recall is:", xgb_recall)

# Create confusion matrix with predictions and actuals.
xgb_confusionMatrix = confusion_matrix(y_test, xgb_y_test_pred)

# Print Confusion Matrix, Accuracy Score, and Classification Report.
print('Confusion Matrix:')
print(confusion_matrix(y_test, xgb_y_test_pred))
print('Accuracy Score:', accuracy_score(y_test, xgb_y_test_pred))
print('Report:')
print(classification_report(y_test, xgb_y_test_pred, zero_division=0))

# Extract classes for use in Confusion Matrix display.
xgb_classes = xgb.classes_

# Create and show Confusion Matrix display.    
xgb_cm_display = ConfusionMatrixDisplay(confusion_matrix=xgb_confusionMatrix, display_labels=xgb_classes)
xgb_cm_display.plot(include_values=True, xticks_rotation='vertical')
plt.show()
# -*- coding: utf-8 -*-

# Logistic Regression- Dataset: BikeShare 
# Import libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression

sns.set(style='darkgrid')
#Set directory
os.chdir(r"C:\Users\HP\Downloads\data science\data mining\project")

#Read data into dataframe
data_bikes=pd.read_csv("bikes_hour.csv")

# View data in dataset 
data_bikes.info()
data_bikes.head()

# Data Analysis
#Checking null values in data
data_bikes.isnull().sum()

# Barchart 
fig=sns.barplot(x='season', y='cnt', data=data_bikes)
fig.set(xlabel='Season', ylabel='Total bikes rented')
plt.title("Season wise bikes rented")
plt.show()

# Scatter plot
plt.scatter(x = data_bikes['casual'] + data_bikes['registered'], y = data_bikes['cnt'])
plt.title("Comparision of casual & Registered users vs. Count")
plt.show()

# Correlation heatmap
data_correlation = data_bikes[['temp', 'atemp', 'casual', 'registered', 'hum', 'windspeed', 'cnt']].corr()
mask = np.array(data_correlation)
mask[np.tril_indices_from(mask)] = False
fig = plt.subplots(figsize=(20,10))
sns.heatmap(data_correlation, mask=mask, vmax=1, square=True, annot=True)

# description of count variable
data_bikes.iloc[:,-1].describe()

# Feature Engineering
data_bikes['High'] = data_bikes.cnt.map(lambda x: 1 if x>190 else 0)

# Selecting Predictors and target variables
X = data_bikes.iloc[:,2:-3]
y =data_bikes['High']

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# Applying Logistic Regression- Training or Model Fitting
model = LogisticRegression()
log_fit=model.fit(X_train, y_train)

log_fit.score(X,y)
print(log_fit.coef_,log_fit.intercept_)
y_pred =log_fit.predict(X_test)

#Evaluation Metrics
from sklearn.metrics import accuracy_score, confusion_matrix
# Confusion Matrix
confusion_matrix(y_test,y_pred)

# Accuracy Score
accuracy_score(y_test,y_pred)


# ROC Curve and AUC

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_pred, y_test)
roc_auc = auc(fpr, tpr)
print("ROC-AUC:",roc_auc)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# k-fold

accuracies = cross_val_score(estimator = log_fit, X = X_train,y = y_train, cv = 10, scoring = 'accuracy')
accuracy = accuracies.mean()
print('r2 = {}'.format(accuracy))
accuracies


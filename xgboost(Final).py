# XGBoost

# Install xgboost following the instructions on this link: http://xgboost.readthedocs.io/en/latest/build.html#

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import shap

# Importing the dataset
dataset = pd.read_csv('XGB-edit.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#Feature importance
print(classifier.feature_importances_)
plt.bar(range(len(classifier.feature_importances_)), classifier.feature_importances_)
plt.show()

xgb.plot_importance(classifier)
plt.show()


#Visualizing the tree
from xgboost import plot_tree
plot_tree(classifier)
plt.show()
plot_tree(classifier, num_trees = 0, rankdir='LR')
plt.show()
fig, ax = plt.subplots(figsize=(30, 10))
xgb.plot_tree(classifier, num_trees=10, ax=ax, rankdir = 'LR')
plt.show()

#Shap
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
shap_values()
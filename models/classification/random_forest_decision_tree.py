# import os
# import sys
# import time
# import pprint
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import plotly.express as px
# import seaborn as sns
import shap

import lightgbm as lgb

# from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import svm
# from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import plot_confusion_matrix, precision_recall_fscore_support
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, roc_auc_score, precision_score
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import feature_descriptions

# dataset from Kaggle, has no NaN or missing values
# https://www.kaggle.com/sakshigoyal7/credit-card-customers
data_raw = pd.read_csv('data/BankChurners.csv')
# Columns to be dropped (client id and Naive_Bayes_Classifiers)
dataset = data_raw.copy()
unused_columns = ['CLIENTNUM',
                'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']
dataset = dataset.drop(unused_columns,axis=1)

dataset[dataset.select_dtypes('object').columns] = dataset.select_dtypes('object').apply(lambda x: x.astype('category'))

# Attrition_Flag is our dependent variable,
# convert to 0 for existing and 1 for attrited customer
codes = {'Existing Customer':0, 'Attrited Customer':1}
dataset['Attrition_Flag'] = dataset['Attrition_Flag'].map(codes)
# print(dataset.info())
y = dataset['Attrition_Flag']
X = dataset.drop('Attrition_Flag',axis=1)
#print(y.shape, X.shape)

# a complete piece of borrowed code
####
# https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python 
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 

features_to_encode = X.select_dtypes('category').columns.to_list()
for feature in features_to_encode:
    X = encode_and_bind(X, feature)
    
# X.info()
####

# A stratified split preserves the ratio of 1 and 0 between the splits
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=555, test_size=0.2, shuffle= True, stratify = y) # 555 is lucky :)

# I see some code repeating below, make it DRY, please
rfc = RandomForestClassifier(random_state=555)
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)

#joblib.dump(rfc, 'models/rfc.joblib')

# shap_values = shap.TreeExplainer(rfc).shap_values(X_train)
# shap.summary_plot(shap_values, X_train, plot_type="bar")

print(f'rfc score is {round(rfc.score(X_test, y_test), 3)}')
rfc_prfs = precision_recall_fscore_support(y_test, y_pred_rfc, average='binary')
print(f'rfc stats are:\nPrecision: {round(rfc_prfs[0], 3)}\nRecall: {round(rfc_prfs[1], 3)}\nFscore: {round(rfc_prfs[2], 3)}')


explainer = ClassifierExplainer(rfc, X_test, y_test, 
                               
                               descriptions=feature_descriptions,
                               labels=['Existing Customer', 'Attrited Customer']) # cats=['Sex', 'Deck', 'Embarked'],
# Warning: calculating shap interaction values can be slow! Pass shap_interaction=False to remove interactions tab.
ExplainerDashboard(explainer, title="babushka churn prediction model dashboard", shap_interaction=False, simple=True).run() # simple=True, shap_interaction=False


"""
dtree = tree.DecisionTreeClassifier()
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)

print(f'dtree score is {round(dtree.score(X_test, y_test), 3)}')
dtree_prfs = precision_recall_fscore_support(y_test, y_pred_dtree, average='binary')
print(f'dtree stats are:\nPrecision: {round(dtree_prfs[0], 3)}\nRecall: {round(dtree_prfs[1], 3)}\nFscore: {round(dtree_prfs[2], 3)}')

# check the truth table
disp = plot_confusion_matrix(rfc, X_test, y_test, display_labels=rfc.classes_, cmap=plt.cm.Blues, normalize=None)
disp.ax_.set_title('random forest classifier')

disp1 = plot_confusion_matrix(dtree, X_test, y_test, display_labels=dtree.classes_, cmap=plt.cm.Blues, normalize=None)
disp1.ax_.set_title('decision tree classifier')

plt.show()
"""

# SVC is not running properly yet
'''
svmc = svm.SVC()
svmc.fit(X_train, y_train)
y_pred_svmc = svmc.predict(X_test)

print(f'support vector machine score is {round(svmc.score(X_test, y_test), 3)}')
svmc_prfs = precision_recall_fscore_support(y_test, y_pred_svmc, average='binary')
print(f'dtree stats are:\nPrecision: {round(svmc_prfs[0], 3)}\nRecall: {round(svmc_prfs[1], 3)}\nFscore: {round(svmc_prfs[2], 3)}')


disp2 = plot_confusion_matrix(svmc, X_test, y_test, display_labels=svmc.classes_, cmap=plt.cm.Blues, normalize=None)
disp2.ax_.set_title('support vector machines classifier')
'''

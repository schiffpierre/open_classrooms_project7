# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

#sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer 
from sklearn.metrics import fbeta_score

# SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from collections import Counter

import sys
sys.path.insert(0,"../data")
sys.path.insert(0,"../models")

import pickle

#Lime
import lime
import lime.lime_tabular

#LGBM
import lightgbm as lgb

from xgboost import XGBClassifier

# Set up environment
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 300)
plt.rcParams['figure.figsize'] = [14, 6]

# Training data
app_train = pd.read_csv('../data/application_train.csv')
print('Training data shape: ', app_train.shape)
app_train.head()


# Testing data features
app_test = pd.read_csv('../data/application_test.csv')
print('Testing data shape: ', app_test.shape)
app_test.head()

# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in app_train:
    if app_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(app_train[col])
            # Transform both training and testing data
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)

# one-hot encoding of categorical variables
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)


# Create train labels
train_labels = app_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
# When we do the align, we must make sure to set axis = 1 to align the dataframes based on the columns and not on the rows!
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

# Add the target back in
app_train['TARGET'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)


# Dealing with issues in the Employment column
# Create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

print('There are %d anomalies in the test data out of %d entries' % (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))


# Adding domain-specific features
app_train_domain = app_train.copy()
app_test_domain = app_test.copy()

app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']

# Adding domain-specific features
app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']

# Drop all records with more than 7 children
app_train_domain = app_train_domain[app_train_domain['CNT_CHILDREN'] <=7]

app_train_domain.to_csv('../data/unscaled_data.csv')

# Creating small unscaled dataset for use in dashboard
unscaled_data_columns = ['SK_ID_CURR', 'TARGET', 'AMT_CREDIT']
unscaled_data = app_train_domain[unscaled_data_columns]
unscaled_data.to_csv('../data/unscaled_data_small.csv')

# Creating sample for use in dashboard
app_train_domain_sample = app_train_domain.sample(int(len(app_train_domain)*0.1))

app_train_domain_sample.to_csv('')


# ## Building Models and prepping data

# Create train labels
train_labels = app_train_domain['TARGET']
train = app_train_domain.copy()

# Drop the IDs and record them for later
train_ids = train['SK_ID_CURR']
train.drop(columns = ['SK_ID_CURR'], inplace = True)

# Feature names
features = list(train.columns)

# Median imputation of missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(train)

# Transform both training and testing data
train = imputer.transform(train)

# Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)


# Recreate a data frame from the training dataset
df_final = pd.DataFrame(train, columns = features)
df_final['SK_ID_CURR'] = train_ids

# Create data sample for use in dashboard
df_final_sample = df_final.sample(n = int(len(df_final)*0.1))

# Drop target column and convert to array for training
train = df_final.drop(columns = ['TARGET', 'SK_ID_CURR'])
train = np.asarray(train)

Counter(train_labels)

# Define a dataframe to store the results of our different models
results = pd.DataFrame(columns = ['model','score', 'train_score', 'test_score'])
# Create F-Beta scorer
f_beta = make_scorer(fbeta_score, beta=2)
# Define the two scoring techniques for our models
scoring = {'F-Beta': f_beta,
           'ROC AUC': 'roc_auc'}

X_train, X_test, y_train, y_test = train_test_split(train, train_labels, test_size=0.3, random_state=42)


# ### Logistic Regression

# #### NO SMOTE

# Define model
model = LogisticRegression(random_state = 42)

# Fit model
log_reg = model.fit(X_train, y_train)

# Define model
model = LogisticRegression(random_state = 42)

# Define pipeline
over = SMOTE(sampling_strategy=0.3)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('over', over), ('under', under), ('model', model)]
pipeline = Pipeline(steps=steps)

# Parameter grid
parameters = {'model__penalty': ['l1', 'l2'],
              'model__C': [10,1,0.1]
             }

# Hyperparameter Tuning
log_reg_search = GridSearchCV(estimator = pipeline, 
                              param_grid = parameters,
                              cv = 4, 
                              verbose = 3,
                              n_jobs = -1, 
                              scoring = f_beta
                             )

# Fit model
log_reg_tuned = log_reg_search.fit(X_train, y_train)
# Serialize model
pickle.dump(log_reg_tuned, open('../models/LRModel2.obj', 'wb'))


# ### XG Boost
# Define model
xgb = XGBClassifier(objective='binary:logistic', random_state = 42)

# Define pipeline
over = SMOTE(sampling_strategy=0.3)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('over', over), ('under', under), ('model', xgb)]
pipeline = Pipeline(steps=steps)

# Parameter grid
parameters = {'model__max_depth': [4,5,6],
              'model__min_child_weight': [1,3]
             }

# Hyperparameter Tuning
xgb_search = GridSearchCV(estimator = pipeline, 
                              param_grid = parameters,
                              cv = 4, 
                              verbose = 3,
                              n_jobs = -1, 
                              scoring = f_beta
                             )

#Variable 1
xgbtuned = xgb_search.fit(X_train, y_train)
# Serialize model
pickle.dump(xgbtuned, open('../models/XGBModel3.obj', 'wb'))


# ### Light Gradient Boosting
# Define model
lgbm = lgb.LGBMClassifier(objective = 'binary', random_state = 42)

# Define pipeline
over = SMOTE(sampling_strategy=0.3)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('over', over), ('under', under), ('model', lgbm)]
pipeline = Pipeline(steps=steps)

# Parameter grid
parameters = {'model__is_unbalance': [True, False],
              'model__boosting_type': ['gbdt', 'goss']
             }

# Hyperparameter Tuning
lgbm_search = GridSearchCV(estimator = pipeline, 
                              param_grid = parameters,
                              cv = 4, 
                              verbose = 3,
                              n_jobs = -1, 
                              scoring = f_beta
                             )

# Fit model
lgbm_tuned = lgbm_search.fit(X_train, y_train)
# Serialize model
pickle.dump(lgbm_tuned, open('../models/LGBModel2.obj', 'wb'))


# #### Random Forest
# Define model
rfc = RandomForestClassifier(random_state = 42)

# Define pipeline
over = SMOTE(sampling_strategy=0.3)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('over', over), ('under', under), ('model', rfc)]
pipeline = Pipeline(steps=steps)

# Parameter grid
parameters = {'model__min_samples_split': [2,4]
             }

# Hyperparameter Tuning
rfc_search = GridSearchCV(estimator = pipeline, 
                              param_grid = parameters,
                              cv = 4, 
                              verbose = 3,
                              #n_jobs =-1, 
                              scoring = f_beta
                             )

#Variable 1
rfc_tuned = rfc_search.fit(X_train, y_train)
# Serialize model
pickle.dump(rfc_tuned, open('../models/RFCModel2.obj', 'wb'))
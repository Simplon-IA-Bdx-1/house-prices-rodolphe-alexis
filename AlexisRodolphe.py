# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:49:37 2019
@authors: Rodolphe & Alexis 
"""

# Import des modules données

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot
from scipy.stats import skew
from scipy.stats.stats import pearsonr

## Keras Model

from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l1
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Datasets

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Concat train/test en all_data

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))


# Taille figure 

matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)


# Application d'une fonction log sur Price (+1 pour eviter les nan/zéro)

prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()

# Remplace log.Price sur le dataset all_data

train["SalePrice"] = np.log1p(train["SalePrice"])

# Sortir le nombre de catégorielles du all_data

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Pour normaliser les "skewed features"

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index


# all_data ajout des skewed_feats

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# get_dummies pour avoir les colonnes catégorielles.
all_data = pd.get_dummies(all_data)

# S'occuper des NA du all_data par la moyenne du tableau

all_data = all_data.fillna(all_data.mean())

# Création du X_train, X_test et y

X_train = all_data[:train.shape[1]]
X_test = all_data[train.shape[1]:]
y = train.SalePrice

# Import XGBoost

import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y)

xgb_preds = np.expm1(model_xgb.predict(X_test))
lasso_preds = np.expm1(model_lasso.predict(X_test))

predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})
predictions.plot(x = "xgb", y = "lasso", kind = "scatter")

preds = 0.7*lasso_preds + 0.3*xgb_preds

solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution_csv = solution.to_csv("ridge_sol.csv", index = False)

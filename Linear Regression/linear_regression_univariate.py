import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# sklearn installed in the venv env 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
#import the linear model
from sklearn.linear_model import LinearRegression
# load_boston is deprecated => ignore teh warnings 
import warnings
from sklearn.metrics import mean_squared_error


with warnings.catch_warnings():
  warnings.filterwarnings("ignore")
  boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target

"""
### get the data manually (load_boston is deprecated)
# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]

### statistic summary of the data ---------- ###
print('------Data description:\n', boston.describe().round(2))

### visualize the data --------------------- ###
boston.hist(column='LSTAT', bins=20)
plt.show()

### correlation ---------------------------- ###
corr_matrix = boston.corr().round(2)
print('------Correlation matrix:\n', corr_matrix)

### Feature selection ---------------------- ###
## RM & MEDV r positively correlated (RM-up => MEDV-up)
boston.plot(kind = 'scatter', x = 'RM', y = 'MEDV', figsize = (8,6))
# plt.savefig('figures/RM_MEDV.png')
## LSTAT & MEDV r negatively correlated (LSTAT-down => MEDV-up)
boston.plot(kind = 'scatter', x = 'LSTAT', y = 'MEDV', figsize = (8,6))
# plt.savefig('figures/LSTAT_MEDV.png')
plt.show()
#"""

### building model ------------------------- ###
## MEDV = b + m * RM
## In scikit-learn, models require a two-dimensional feature matrix (X, 2darray or a pandas DataFrame) and a one-dimensional target array (Y).
X = boston[['RM']] # pandas dataframe
Y = boston['MEDV'] # pandas series

### instanciating the model -----------------###
model = LinearRegression()

## we split the data into 70% train - 30% test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=1)

## we fit the model (train the model)
model.fit(X_train, Y_train)

print('------Intercept :\n', model.intercept_.round(2))  # -30.57
print('------Coef :\n', model.coef_.round(2))       # [8.46]
## the fitted model is : MEDV = -30.57 + 8.46 * RM

### predictions ---------------------------- ###
with warnings.catch_warnings():
  warnings.filterwarnings('ignore')

  # predict for one value (median)
  new_RM = np.array([6.5]).reshape(-1,1) # has to be 2d
  print('------Prediction for [[6.5]] :\n', model.predict(new_RM)) # same as : (model.intercept_ + model.coef_*6.5)

  # predict for the entire test data
  y_test_predicted = model.predict(X_test)
  print('------y_test_predict shape is same as observed :\n', y_test_predicted.shape)
  print('------type of y_test_predict :\n', type(y_test_predicted))
  
#""" -- visualizing the prediction/data
plt.scatter(X_test, Y_test, label='testing data')
# plt.plot(X_test, y_test_predicted, label='prediction', linewidth=3)   # gives error
plt.xlabel('RM'); plt.ylabel('MEDV')
plt.legend(loc='upper left')
# plt.savefig("figures/prediction_to_data.png")
plt.show()
#"""

### Residuals (margin of error between observed and predicted value)
residuals = Y_test - y_test_predicted

#""" -- plot residuals
plt.scatter(X_test, residuals)
# plot a horizontal line at y = 0
plt.hlines(y = 0, xmin = X_test.min(), xmax=X_test.max(), color='k', linestyle='--')
plt.xlim((4, 9))
plt.xlabel('RM'); plt.ylabel('residuals')
plt.savefig("Linear Regression/figures/model_prediction_residuals.png")
plt.show()
#"""

### R-Squared (between 0 & 100%)
print('------R-Squared:\n ', model.score(X_test, Y_test))


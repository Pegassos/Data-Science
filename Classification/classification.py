import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, plot_confusion_matrix

# iris = pd.read_csv('https://sololearn.com/uploads/files/iris.csv')
iris = pd.read_csv('Classification/data/iris.csv')
iris.drop('id', axis=1, inplace=True)

### ------------------------------------------ ###
""" -- Data Description
print(iris.describe())
iris.hist()
# plt.savefig('Classification/figures/iris_attr.png')
plt.show()
#"""
### ------------------------------------------ ###
""" -- Data Visualisation
# dict mapping species to int code
inv_name_dict = {
  'iris-setosa': 0,
  'iris-versicolor': 1,
  'iris-virginica': 2
}

# int color code 0, 1, 2
colors = [inv_name_dict[item] for item in iris['species']]

# scatter plot sepals -----------------
scatter = plt.scatter(iris['sepal_len'], iris['sepal_wd'], c = colors)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.legend(
  handles = scatter.legend_elements()[0],
  labels = inv_name_dict.keys(),
)
#plt.savefig('Classification/figures/scatter_sepals.png')
plt.show()

# scatter plot petals -----------------
scatter2 = plt.scatter(iris['petal_len'], iris['petal_wd'],c = colors)
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.legend(
  handles= scatter2.legend_elements()[0],
  labels = inv_name_dict.keys()
)
# plt.savefig("Classification/figures/scatter_petals.png")
plt.show()

pd.plotting.scatter_matrix(iris)
# plt.savefig('Classification/figures/all_graphs.png')
plt.show()

#""" 
### ------------------------------------------ ###
""" -- Data Preparation / Prediction
X = iris[['petal_len', 'petal_wd']]
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
# startify to make sure test & train sets have the same amount of the 3 species 
print(y_train.value_counts())
print(y_test.value_counts())

# instanciate the model
knn = KNeighborsClassifier(n_neighbors=5)   # default is 5
knn.fit(X_train, y_train)

pred = knn.predict(X_test)
# returns: ['iris-virginica' 'iris-setosa' 'iris-setosa' 'iris-versicolor' 'iris-versicolor' ...]

# instead, we can predict the probabilty of a flower being each one of the available 
y_pred_prob = knn.predict_proba(X_test)
print('----------prediction prob:\n', y_pred_prob[10:12]) # soft prediction
print('----------prediction:\n', pred[10:12])             # hard prediction
#"""
### ------------------------------------------ ###
""" -- Model Evaluation knn(5) (accuracy, confusion_matrix)
X = iris[['petal_len', 'petal_wd']]
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Accuracy --------------------------
# # Accuracy: compare the model result with the actual observed data
# print('y_pred == y_test sum: ', (y_pred==y_test.values).sum())  # 44
# print('y_test size: ', y_test.size)                             # 45
# # => the classifier made 1 error (mistake)
print('Accuracy: ', (y_pred==y_test.values).sum() / y_test.size)
# # accuracy is same as score
# print('Score: ', knn.score(X_test, y_test))

# Confusion matrix: describe the performance of a classification model -------------------
# print('--------Confusion matrix:\n', confusion_matrix(y_test, y_pred))
# plot_confusion_matrix(knn, X_test, y_test, cmap=plt.cm.Blues)     # deprecated
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
# plt.savefig('Classification/figures/confusion_matrix.png')
plt.show()
#"""
### ------------------------------------------ ###
""" -- Model Evaluation knn(3) K-Fold Cross Validation 
## instead of train/test sets we divide the data into ksubsets and train the model on k-1 and test for one set
# this way we have more accuracy and a variety of different sets to train the model
from sklearn.model_selection import cross_val_score

X = iris[['petal_len', 'petal_wd']]
y = iris['species']
knn_cv = KNeighborsClassifier(n_neighbors=3)

# train model with 5-fold cv
cv_scores = cross_val_score(knn_cv, X, y, cv=5)

# print each cv score (accuracy) 
print('-------------cv score (accuracy):\n', cv_scores)
print('-------------cv score mean:\n', cv_scores.mean())    # 3nn model based on 5-fold cv accuracy = 95%
#"""
### ------------------------------------------ ###
#""" -- Tuning the hyperparameter k in knn + Predictions
## Grid Search-------------
from sklearn.model_selection import GridSearchCV

knn2 = KNeighborsClassifier()

# create a dict of all values we want to test for n_neighbors
param_grid = { 'n_neighbors': np.arange(2, 10) }

# use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)

# fit model to data
X = iris[['petal_len', 'petal_wd']]
y = iris['species']
knn_gscv.fit(X, y)

print('---------top performing n_neighbors value:\n', knn_gscv.best_params_)
print('---------accuracy of the model with best params:\n', knn_gscv.best_score_.round(4) * 100, '%')
## grid search improved the model accuracy by 1%

# build the final model
knn_final = KNeighborsClassifier(n_neighbors=knn_gscv.best_params_['n_neighbors'])
knn_final.fit(X, y)

y_pred = knn_final.predict(X)
print('\n---------Final model accuracy:\n', (knn_final.score(X, y) * 100).round(2), '%')

## Prediction --------------------------------------------------------
new_data = np.array([3.76, 1.20]).reshape(1,-1)
print('\n---------Predicxtion for [3.76, 1.2] petals len, wd:\n', knn_final.predict(new_data))

## Prediction for data with close values (same petal width) ----------
new_data = np.array([[3.76, 1.2], [5.25, 1.2], [1.58, 1.2]])
print('\n---------Predicxtion for iris with same petals wd:\n', knn_final.predict(new_data))
print('---------Predicted probabilities:\n', knn_final.predict_proba(new_data))
#"""
### ------------------------------------------ ###
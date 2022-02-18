import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_wine  # 13 features
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = load_wine()
wine = pd.DataFrame(data.data, columns=data.feature_names)

### ------------------------------------------ ###
""" -- Data Description & Visualisation
## Description
print('shape: ', wine.shape)
print('columns: ', wine.columns)
print('------description (7 first features):')
print(wine.iloc[:, :7].describe())

## Visualisation
scatter_matrix(wine.iloc[:,[0,5]])
plt.savefig('Clustering/figures/alcohol__total_phenols.png')
plt.show()
#"""
### ------------------------------------------ ###
""" -- Preprocessing: Standarization
X = wine[['alcohol', 'total_phenols']]

## standarize the data: z = (x - mean) / std  (center the data arround (0,0))
# instanciate the scaler
scale = StandardScaler()
# compute the mean and std to be used later for scaling
scale.fit(X)
# StandardScaler(copy=True, with_mean=True, with_std=True)
mean = scale.mean_  # [alcohol_mean, total_phenols_mean]
std = scale.scale_  # [alcohol_std, total_phenols_std]

# fit data & transform it
X_scaled = scale.transform(X)

## Visualize data with scaled
plt.scatter(X_scaled[:,0], X_scaled[:,1], label='Scaled')
plt.scatter(X.iloc[:,0], X.iloc[:,1], label='original')

plt.title('Original / Scaled (normalization) data')
plt.xlabel('alcohol')
plt.ylabel("total phenols")
plt.legend()
plt.savefig('Clustering/figures/original__scaled_data.png')
plt.show()
#"""
### ------------------------------------------ ###
#""" -- K-Means
X = wine[['alcohol', 'total_phenols']] 

## Model ---------------------------------
scale = StandardScaler()
scale.fit(X)
X_scaled = scale.transform(X)

# instantiate the model (with 3 clusters)
kmeans = KMeans(n_clusters=3)

# fit the model
kmeans.fit(X_scaled)

# make predictions
y_pred = kmeans.predict(X_scaled)

print('-------Prediction:\n', y_pred)
print('-------Coordinates of the clusters:\n', kmeans.cluster_centers_)

## Visualize the results --------------------
# plot the scaled data
plt.scatter(
  X_scaled[:,0], 
  X_scaled[:,1], 
  c=y_pred
)

# identify the centroids
plt.scatter(
  kmeans.cluster_centers_[:, 0,], 
  kmeans.cluster_centers_[:, 1,], 
  marker = '*',
  s = 250,
  c = [0, 1, 2],
  edgecolors = 'k'
)

plt.title('k-means (k = 3)')
plt.xlabel('alcohol')
plt.ylabel("total phenols")
plt.savefig('Clustering/figures/k-means_k3.png')
plt.show()

## Predict ------------------------------
# to predict a wine: first std the data then predict
X_new = np.array([[13, 2.5]])   # alcohol:13 - total phenols:2.5

# standarize new data
X_new_scaled = scale.transform(X_new)
print('------ Standarized new data:\n', X_new_scaled)  # [[-0.00076337  0.32829793]]

# prediction
print('------ Prediction for new data:', kmeans.predict(X_new_scaled))

#"""
### ------------------------------------------ ###
#"""
#"""
### ------------------------------------------ ###
#"""
#"""
### ------------------------------------------ ###
#"""
#"""
### ------------------------------------------ ###
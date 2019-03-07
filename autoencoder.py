
import scipy.io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

file_name = 'mat_files/output_not_original_30.csv_tf-idf-l2.mat'
mat = scipy.io.loadmat(file_name)
X=mat['X'].todense()


#PCA
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

pca = PCA()
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
variance=np.asarray(pca.explained_variance_ratio_)
sv=np.asarray(pca.singular_values_)
np.savetxt("variance.csv",variance, delimiter=",")
np.savetxt("singular_values.csv",sv, delimiter=",")

# AUTOENCODER

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

np.random.shuffle(X)

x_train = X[:4000]
x_test = X[4000:]

model = Sequential()
model.add(Dense(units=3000, activation='relu', input_dim=4431))
model.add(Dense(units=2000, activation='relu'))
model.add(Dense(units=1500, activation='linear', name='bottleneck'))
model.add(Dense(units=2000, activation='relu'))
model.add(Dense(units=3000, activation='relu'))
model.add(Dense(units=4431, activation='linear'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x_train, x_train, epochs=300, batch_size=16)

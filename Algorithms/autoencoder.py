
import scipy.io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

file_name = '../dataset/mat_files/output_not_original_30.csv_tf-idf-l2.mat'
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

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers

np.random.shuffle(X)

x_train = X[:4000]
x_test = X[4000:]

encoding_dim = 2000

input_x = Input(shape=(4431,))

encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_x)
decoded = Dense(4431, activation="sigmoid")(encoded)

autoencoder = Model(input_x, decoded)

encoder = Model(input_x, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


autoencoder.fit(x_train, x_train, epochs=500, batch_size=32, shuffle=True, validation_data=(x_test,x_test))



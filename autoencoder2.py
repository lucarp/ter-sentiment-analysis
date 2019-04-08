import scipy.io
import sys
import numpy as np

# -------------------------
# HYPER - PARAMETERS

activation = 'relu'
#loss = 'sparse_categorical_crossentropy'
loss = 'categorical_crossentropy'
#loss = 'binary_crossentropy'

# -------------------------

file_name = sys.argv[1]
mat = scipy.io.loadmat(file_name)
X=mat['X'].todense()

# AUTOENCODER

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers

np.random.shuffle(X)

x_train = X[:4000]
x_test = X[4000:]

input_dim = x_train.shape[1]

encoding_dim = 5

input_x = Input(shape=(input_dim,))

encoded = Dense(2048, activation=activation, activity_regularizer=None)(input_x)
encoded = Dense(512, activation=activation, activity_regularizer=None)(encoded)
encoded = Dense(encoding_dim, activation=activation, activity_regularizer=None)(encoded)

decoded = Dense(512, activation=activation, activity_regularizer=None)(encoded)
decoded = Dense(2048, activation=activation, activity_regularizer=None)(decoded)
decoded = Dense(input_dim, activation="softmax", activity_regularizer=None)(decoded)

autoencoder = Model(input_x, decoded)

autoencoder.compile(optimizer='adam', loss=loss)

autoencoder.fit(x_train, x_train, epochs=500, batch_size=32, shuffle=True, validation_data=(x_test,x_test))



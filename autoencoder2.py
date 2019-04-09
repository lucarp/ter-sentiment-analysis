import scipy.io
import sys
import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Dropout
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers

# -------------------------
# HYPER - PARAMETERS

activation = 'relu'
#loss_function = 'sparse_categorical_crossentropy'
#loss_function = 'categorical_crossentropy'
#loss_function = 'binary_crossentropy'
loss_function = 'mean_squared_error'
epoch = 1 #1000
learning_rate = 0.00025

test_ratio = 1/5

l2_beta = 0.00001
dropout = 0.
useBatchNorm = False

# -------------------------

if l2_beta is not 0.0:
	regularizer = regularizers.l2(l2_beta)
else:
	regularizer = None

np.random.seed(0)

# Load Data

file_name = sys.argv[1]
mat = scipy.io.loadmat(file_name)
X=mat['X'].todense()
len_dataset = X.shape[0]

X_ind = np.arange(len_dataset)
np.random.shuffle(X_ind)
X = X[X_ind]

k = int(sys.argv[2])
maxK = int(1/test_ratio)
k = min(max(k, 0), maxK-1)

test_size = int(len_dataset * test_ratio)		
if test_size != 0:
	b_split = int(len_dataset * test_ratio * k)
	e_split = int((b_split + len_dataset * test_ratio) % len_dataset)
	x_test = X[b_split:e_split]
	x_test_ind = X_ind[b_split:e_split]
	x_train = np.concatenate([X[0:b_split], X[e_split:]])

# AUTOENCODER
input_dim = x_train.shape[1]

encoding_dim = 5

def full_layer(input_layer, output_dim, activation, regularizer, useBatchNorm, dropout=0.0):
	layer = Dense(output_dim, activity_regularizer=regularizer)(input_layer)
	if useBatchNorm:
		layer = BatchNormalization()(layer)
	layer = Activation(activation)(layer)
	layer = Dropout(dropout)(layer)
	return layer

input_x = Input(shape=(input_dim,))
#encoded = Dense(2048, activation=activation, activity_regularizer=regularizer)(input_x)
encoded = full_layer(input_x, 2048, activation, regularizer, useBatchNorm, dropout)
#encoded = Dense(512, activation=activation, activity_regularizer=regularizer)(encoded)
encoded = full_layer(encoded, 512, activation, regularizer, useBatchNorm, dropout)
#encoded = Dense(encoding_dim, activation=activation, activity_regularizer=regularizer)(encoded)
encoded = full_layer(encoded, encoding_dim, activation, regularizer, useBatchNorm, dropout)

#decoded = Dense(512, activation=activation, activity_regularizer=regularizer)(encoded)
decoded = full_layer(encoded, 512, activation, regularizer, useBatchNorm, dropout)
#decoded = Dense(2048, activation=activation, activity_regularizer=regularizer)(decoded)
decoded = full_layer(decoded, 2048, activation, regularizer, useBatchNorm, dropout)
#decoded = Dense(input_dim, activation="linear", activity_regularizer=regularizer)(decoded)
decoded = full_layer(decoded, input_dim, "linear", regularizer, useBatchNorm)

autoencoder = Model(input_x, decoded)

optimizer = optimizers.Adam(lr=learning_rate)

autoencoder.compile(optimizer=optimizer, loss=loss_function)

encoder = Model(input_x, encoded)

autoencoder.fit(x_train, x_train, epochs=epoch, batch_size=32, shuffle=True, validation_data=(x_test,x_test))

code = encoder.predict(x_test)

print(code)
print(code.shape)

pd.DataFrame(code).to_csv("autoencoder_"+str(k)+"_code.csv")
pd.DataFrame(x_test_ind).to_csv("autoencoder_"+str(k)+"_ind.csv")

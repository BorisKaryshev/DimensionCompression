import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import JsonToMatrix as jsm
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import (BatchNormalization, Dense, Dropout, Flatten, Input,
                          Lambda, Reshape)
from tensorflow import keras

#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#x_train = x_train/255
#x_test = x_test/255

#_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
#_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

hidden_dim = 2
batch_sz = 30

def dropout_and_batch(x):
    return Dropout(0.3)(BatchNormalization()(x))

x_train = jsm.JsonToAr("input.txt")
x_train = np.reshape(x_train, (len(x_train), len(x_train[0]), 1, 1))

input_img = Input((len(x_train[0]), 1, 1))

x = Flatten()(input_img)
x = Dense(256, activation='relu')(input_img)
x = dropout_and_batch(x)
x = Dense(128, activation='relu')(x)
x = dropout_and_batch(x)

z_mean = Dense(hidden_dim)(x)
z_log_var = Dense(hidden_dim)(x)

def noiser(args):
  global z_mean, z_log_var
  z_mean, z_log_var = args
  N = K.random_normal(shape=(batch_sz, hidden_dim), mean=0., stddev=1.0)
  return K.exp(z_log_var / 2) * N + z_mean

h = Lambda(noiser, output_shape=(hidden_dim,))([z_mean, z_log_var])

input_dec = Input(shape=(hidden_dim,))
d = Dense(128, activation='relu')(input_dec)
d = Dense(256, activation='relu')(d)
d = Dense(len(x_train[0]), activation='sigmoid')(d)
decoded = Reshape((len(x_train[0]),))(d)


encoder = keras.Model(input_img, h, name="encoder")
decoder = keras.Model(input_dec, decoded, name="decoder")

model = keras.Model(input_img, decoder(encoder(input_img)), name='autoencoder')

def vae_loss(x, y):
    x = K.reshape(x, shape=(batch_sz, len(x_train[0])))
    y = K.reshape(y, shape=(batch_sz, len(x_train[0])))
    loss = K.sum(K.square(x-y),axis=-1)
    kl_loss = -0.5 *K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return loss + kl_loss


model.compile(optimizer='adam', loss=vae_loss)

model.fit(x_train, x_train,
          epochs=500,
          batch_size=batch_sz
) 

his = encoder.predict(x_train[:], batch_size=batch_sz)
plt.scatter(his[:, 0], his[:, 1])#, c = y_train[:])

plt.show()

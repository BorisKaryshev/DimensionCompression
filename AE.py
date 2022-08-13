import numpy as np

def compute(input: np.ndarray, dims: int = 2, num_of_iterations: int = 500) -> np.ndarray:
  import keras.backend as K
  from keras.layers import (BatchNormalization, Dense, Dropout, Flatten, Input,
                            Lambda, Reshape)
  from tensorflow import keras
  from tensorflow.python.framework.ops import disable_eager_execution
  
  disable_eager_execution()
  hidden_dim = dims
  batch_sz = 1

  def dropout_and_batch(x):
      return Dropout(0.2)(BatchNormalization()(x))

  x_train = input
  x_train = np.reshape(x_train, (len(x_train), len(x_train[0])))

  if(len(x_train) > 20):
    batch_sz = int(len(x_train)/20)
  
  while(len(x_train) % batch_sz != 0):
    batch_sz -= 1

  input_img = Input((len(x_train[0])))

  x = Flatten()(input_img)
  x = Dense(len(x_train[0])*2, activation='relu')(input_img)
  x = dropout_and_batch(x)
  x = Dense(len(x_train[0]), activation='relu')(input_img)
  x = dropout_and_batch(x)
  x = Dense(len(x_train[0])/2, activation='relu')(x)
  x = dropout_and_batch(x)

  z_mean = Dense(hidden_dim, activation='linear')(x)
  z_log_var = Dense(hidden_dim, activation='linear')(x)

  def noiser(args):
    global z_mean, z_log_var
    z_mean, z_log_var = args
    N = K.random_normal(shape=(batch_sz, hidden_dim), mean=0., stddev=1.0)
    return K.exp(z_log_var / 2) * N + z_mean


  h = Lambda(noiser, output_shape=(hidden_dim,))([z_mean, z_log_var])

  input_dec = Input(shape=(hidden_dim,))
  d = Dense(len(x_train[0])/2, activation='relu')(input_dec)
  d = dropout_and_batch(d)
  d = Dense(len(x_train[0]), activation='relu')(d)
  d = dropout_and_batch(d)
  d = Dense(len(x_train[0])*2, activation='relu')(d)
  d = dropout_and_batch(d)
  d = Dense(len(x_train[0]), activation='linear')(d)
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

  model.compile(optimizer='adam', loss=vae_loss)#, experimental_run_tf_function=False)

  model.fit(x_train, x_train,
            epochs=num_of_iterations,
            batch_size=batch_sz,
            shuffle = True
  )
  return encoder.predict(x_train[:], batch_size=batch_sz)
from random import random
from tabnanny import verbose
import umap
import numpy as np
import time



def umap_compute(input: np.ndarray, dims: int = 2, n_of_neighbours: int = 15, num_of_iterations: int = 500) -> np.ndarray:
    model = umap.UMAP(n_components=dims,
                      n_neighbors=n_of_neighbours, 
                      n_epochs=num_of_iterations,
                      n_jobs=4            
            )
    out = model.fit_transform(input)
    return out


def tSNE_compute(input: np.ndarray, dims: int = 2, pereplexity: float = 15, number_of_iterations: int = 500) -> np.ndarray:
    from sklearn.manifold import TSNE
    embed = TSNE(
        n_components=dims,
        perplexity=pereplexity,
        early_exaggeration=100,
        learning_rate=200,
        n_iter=number_of_iterations,
        n_iter_without_progress=300,
        min_grad_norm=0.00000001,
        metric='euclidean',
        init='random',
        verbose=0,
        random_state=int(time.time()),
        method='barnes_hut',
        angle=0.5,
        n_jobs=4,
    )

    output = embed.fit_transform(input)

    return output


def VAE_compute(input: np.ndarray, dims: int = 2, num_of_iterations: int = 500) -> np.ndarray:
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

    if (len(x_train) > 20):
        batch_sz = int(len(x_train)/20)

    while (len(x_train) % batch_sz != 0):
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

    model = keras.Model(input_img, decoder(
        encoder(input_img)), name='autoencoder')

    def vae_loss(x, y):
        x = K.reshape(x, shape=(batch_sz, len(x_train[0])))
        y = K.reshape(y, shape=(batch_sz, len(x_train[0])))
        loss = K.sum(K.square(x-y), axis=-1)
        kl_loss = -0.5 * K.sum(1 + z_log_var -
                               K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return loss + kl_loss

    model.compile(optimizer='adam', loss=vae_loss)

    model.fit(x_train, x_train,
              epochs=num_of_iterations,
              batch_size=batch_sz,
              shuffle=True,
              verbose=0
              )
    return encoder.predict(x_train[:], batch_size=batch_sz)


def AE_compute(input: np.ndarray, dims: int = 2, num_of_iterations: int = 500) -> np.ndarray:

    import keras.backend as K
    from keras.layers import (Dense, Flatten, Input,
                              Reshape)
    from tensorflow import keras
    from tensorflow.python.framework.ops import disable_eager_execution

    disable_eager_execution()
    hidden_dim = dims
    batch_sz = 1

    x_train = input
    x_train = np.reshape(x_train, (len(x_train), len(x_train[0])))

    if (len(x_train) > 20):
        batch_sz = int(len(x_train)/20)

    while (len(x_train) % batch_sz != 0):
        batch_sz -= 1

    input_img = Input((len(x_train[0])))

    x = Flatten()(input_img)
    x = Dense(len(x_train[0])*2, activation='relu')(input_img)
    x = Dense(len(x_train[0]), activation='relu')(x)
    x = Dense(len(x_train[0]), activation='relu')(x)
    x = Dense(len(x_train[0]), activation='relu')(x)
    h = Dense(hidden_dim, activation='linear')(x)

    input_dec = Input(shape=(hidden_dim,))
    d = Dense(len(x_train[0]), activation='relu')(input_dec)
    d = Dense(len(x_train[0]), activation='relu')(d)
    d = Dense(len(x_train[0]), activation='relu')(d)
    d = Dense(len(x_train[0])*2, activation='relu')(d)
    d = Dense(len(x_train[0]), activation='linear')(d)
    decoded = Reshape((len(x_train[0]),))(d)

    encoder = keras.Model(input_img, h, name="encoder")
    decoder = keras.Model(input_dec, decoded, name="decoder")

    model = keras.Model(input_img, decoder(
        encoder(input_img)), name='autoencoder')

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, x_train,
              epochs=num_of_iterations,
              batch_size=batch_sz,
              shuffle=True,
              verbose=0
              )

    return encoder.predict(x_train[:], batch_size=batch_sz)

def pca_compute(input: np.ndarray, dims: int = 2, num_of_iterations: int = 500) -> np.ndarray:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=dims,
              iterated_power=num_of_iterations,
              random_state=int(time.time())
             )
    return pca.fit_transform(input)

def kmapper_compute(input: np.ndarray) -> np.ndarray:
    import kmapper as km
    import sklearn
    mapper = km.KeplerMapper(verbose=2)
    projected_data = mapper.fit_transform(input, projection=sklearn.manifold.TSNE())
    return projected_data

def NMF_compute(input: np.ndarray, dims: int = 2, num_of_iterations: int = 500) -> np.ndarray:
    from sklearn.decomposition import NMF
    model = NMF(n_components=dims, 
                init='random', 
                random_state=int(time.time()),
                beta_loss='frobenius', #or 'kullback-leibler'
                max_iter=num_of_iterations
                )
    return model.fit_transform(input)

def kernelPCA_compute(input: np.ndarray, dims: int = 2, num_of_iterations: int = 500) -> np.ndarray:
    from sklearn.decomposition import KernelPCA
    model = KernelPCA(n_components=dims,
                      max_iter=num_of_iterations,
                      kernel='linear', #or 'poly', ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’
                      )
    return model.fit_transform(input)

def grad_boost_test(x: np.ndarray, y: np.ndarray, num_of_iterations: int = 100, max_depth: int = 3):
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=num_of_iterations, 
                                     learning_rate=1.0, 
                                     max_depth=max_depth, 
                                     random_state=int(time.time())
         )
    n = int(len(x) * 0.8)
    x_train = x[:n]
    y_train = y[:n]
    x_test = x[n:]
    y_test = y[n:]
    clf.fit(x_train, y_train)
    return (clf.predict_proba(x), clf.score(x_test, y_test)) 

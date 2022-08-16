from typing import Tuple
import numpy as np
import lib
import JsonToMatrix as jsm
import matplotlib.pyplot as plt

def load_mnist(size:int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[:size]
    y_train = y_train[:size]
    x_train = np.reshape(x_train, (len(x_train), 28*28))
    return (x_train, y_train)

def load_iris() -> Tuple[np.ndarray, np.ndarray]:
    from sklearn import datasets
    iris = datasets.load_iris()
    return (iris.data, iris.target)

def load_wine() -> Tuple[np.ndarray, np.ndarray]:
    from sklearn import datasets
    wine = datasets.load_wine()
    return (wine.data, wine.target)

def make() -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import make_classification
    return make_classification(n_features=50, 
                               n_redundant=30, 
                               n_repeated=10, 
                               n_informative=5, 
                               n_clusters_per_class=1, 
                               n_classes=2, 
                               n_samples=60000, 
                               class_sep=3
                              )

def draw_with_kmapper(input: np.ndarray, labels: np.ndarray):
    import kmapper as km
    import sklearn

    mapper = km.KeplerMapper(verbose=2)
    projected_data = lib.kmapper_compute(input)
    graph = mapper.map(
        projected_data,
        clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),
        cover=km.Cover(35, 0.4),
    )

    km.draw_matplotlib(graph, layout="spring")
    plt.show()
    exit()


from sklearn.datasets import make_blobs
#(x, y) = make_blobs(n_features=1000, centers=8, n_samples=800)
#(x, y) = make()
(x, y) = load_mnist()

draw_with_kmapper(x, y)
#out = lib.AE_compute(x, num_of_iterations=100, dims=2)
#out = lib.VAE_compute(x, num_of_iterations=100, dims=2)
#out = lib.tSNE_compute(x, pereplexity=50, dims=2, number_of_iterations=500)
#out = lib.umap_compute(x, n_of_neighbours=30, num_of_iterations=100)
#out = lib.pca_compute(x, num_of_iterations=500)
out = lib.kmapper_compute(x)


scatter = plt.scatter(out[:,0], out[:,1], c = y[:])
ma = np.max(y)
mi = np.min(y)

plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(mi, ma+2)-0.5).set_ticks(np.arange(mi, ma+1))

plt.show()
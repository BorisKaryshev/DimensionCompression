import numpy as np
import AE
import tSNE 
import UMAP
import JsonToMatrix as jsm
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:10000]
y_train = y_train[:len(x_train)]
x = x_train#jsm.JsonToAr("input.json")
x = np.reshape(x_train, (len(x_train), 28*28))

out = AE.compute(x, num_of_iterations=500, dims=2)
#out = tSNE.compute(x, pereplexity=30, dims=2)
#out = UMAP.compute(x, n_of_neighbours=20, num_of_iterations=100)

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
scatter = plt.scatter(out[:,0], out[:,1], c = y_train[:])
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
#plt.legend(handles=scatter.legend_elements(), labels=y_train)
plt.show()
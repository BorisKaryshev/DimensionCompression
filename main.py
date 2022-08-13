import numpy as np
import AE
import tSNE 
import JsonToMatrix as jsm
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()#jsm.JsonToAr("input.json")
x_train = x_train[:10000]
y_train = y_train[:len(x_train)]
x = np.reshape(x_train, (len(x_train), 28*28))

out = AE.compute(x, num_of_iterations=100, dims=2)

#out = tSNE.compute(x, pereplexity=20, dims=3)

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
plt.scatter(out[:,0], out[:,1], c = y_train[:])
plt.show()
from colorsys import yiq_to_rgb
from typing import Tuple
import numpy as np
import lib
import JsonToMatrix as jsm
import matplotlib.pyplot as plt
from tabulate import tabulate
import test

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
                               n_classes=8, 
                               n_samples=1000, 
                               class_sep=1.5
                              )

def draw_with_kmapper(input: np.ndarray):
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

def load_titanic() -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    import seaborn as sns
    from pandas import DataFrame
    trd = pd.read_csv('./datasets/titanic.csv')
    trd.isnull().sum()
    trd.Embarked.fillna(trd.Embarked.mode()[0], inplace = True)
    trd.Cabin = trd.Cabin.fillna('NA')
    trd.Age = trd.Age.fillna(30)
    from sklearn import preprocessing
    trd['Sex'] = preprocessing.LabelEncoder().fit_transform(trd['Sex'])
    pd.get_dummies(trd.Embarked, prefix="Emb", drop_first = True)
    #trd.drop(['Pclass', 'Fare','Cabin', 'Fare_Category','Name','Salutation', 'Deck', 'Ticket','Embarked', 'Age_Range', 'SibSp', 'Parch', 'Age'], axis=1, inplace=True)
    trd.drop(['Name','Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    x_train = trd
    y_train = DataFrame(trd['Survived'])
    y_train = DataFrame.to_numpy(y_train)
    x_train.drop(['Survived', 'PassengerId'], axis=1, inplace=True)
    x_train = x_train.dropna()
    x_train = DataFrame.to_numpy(x_train)
    y_train = np.reshape(y_train, (len(y_train),))

    return (x_train, y_train)
    
def load_scores() -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    from pandas import DataFrame
    out = pd.read_csv('./datasets/scores.csv')

    x_train = out
    y_train = DataFrame(out['Student Placed'])
    y_train = DataFrame.to_numpy(y_train)
    x_train.drop(['Student Placed'], axis=1, inplace=True)
    x_train = x_train.dropna()
    x_train = DataFrame.to_numpy(x_train)
    y_train = np.reshape(y_train, (len(y_train),))
    
    return (x_train, y_train)

def load_water_potability() -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    from pandas import DataFrame
    out = pd.read_csv('./datasets/water_potability.csv')
    out.ph = out.ph.fillna(0)
    out.Sulfate = out.Sulfate.fillna(0)
    out.Trihalomethanes = out.Trihalomethanes.fillna(0)

    x_train = out
    y_train = DataFrame(out['Potability'])
    y_train = DataFrame.to_numpy(y_train)
    x_train.drop(['Potability'], axis=1, inplace=True)
    x_train = x_train.dropna()
    x_train = DataFrame.to_numpy(x_train)
    y_train = np.reshape(y_train, (len(y_train),))
    
    return (x_train, y_train)

from sklearn.datasets import make_blobs
#(x, y) = make_blobs(n_features=1000, centers=8, n_samples=800)
#(x, y) = make()

#out = lib.kmapper_compute(x)
#out = lib.NMF_compute(x, num_of_iterations=1000)
#add_to_table(table, x, out, y, name="AE")

table = [["Dimentional reduction method", 
         "Dataset", 
         "Original score",
         "Projection score",
         "Difference: in %",
         "Additional information"]]

x, y = load_wine()
table = test.test_dataset(x, y, table, "Wine", stat_iter=3, depth=3, gr_boost_iter=20)

x, y = load_iris()
table = test.test_dataset(x, y, table, "Iris", stat_iter=3, depth=3, gr_boost_iter=20)

x, y = load_scores()
table = test.test_dataset(x, y, table, "Scores", stat_iter=3, depth=3, gr_boost_iter=20)

x, y = load_water_potability()
table = test.test_dataset(x, y, table, "Water Quality", stat_iter=3, depth=3, gr_boost_iter=20)

x, y = load_titanic()
table = test.test_dataset(x, y, table, "Titanic", stat_iter=3, depth=3, gr_boost_iter=20)

#x, y = make()
#table = test.test_dataset(x, y, table, "Artificial", stat_iter=3, depth=3, gr_boost_iter=20)

x, y = load_mnist(1000)
table = test.test_dataset(x, y, table, "Mnist", stat_iter=3, depth=3, gr_boost_iter=20)

file = open("output.txt", mode='w')
file.write(tabulate(table, headers='firstrow', tablefmt='grid'))

exit()
out = lib.AE_compute(x, num_of_iterations=100)

scatter = plt.scatter(out[:,0], out[:,1], c = y[:])
ma = np.max(y)
mi = np.min(y)

plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(mi, ma+2)-0.5).set_ticks(np.arange(mi, ma+1))

plt.show()
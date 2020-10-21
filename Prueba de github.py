#Libro ML Redes neuronales
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
x = iris.data[:,(2,3)] #longitud de petalo, grosor petalo
y = (iris.target == 0).astype(np.int) #Iris setosa?

per_clf = Perceptron()
per_clf.fit(x, y)

y_pred = per_clf.predict([[2,0.5]])


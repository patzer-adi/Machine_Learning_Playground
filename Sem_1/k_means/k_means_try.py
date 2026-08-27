from sklearn.datasets import load_iris
import numpy as np


#k-means functions 

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

import numpy as np
from collections import Counter


def euclidean_distance(x, x_train):
    return np.sqrt(np.sum((x - x_train) ** 2))


class knn:
    def __init__(self, k=3):
        self.X_train = None
        self.y_train = None
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        kn_indexes = np.argsort(distances)[:5]
        nearest_n = [self.y_train[idx] for idx in kn_indexes]
        nn = Counter(nearest_n).most_common(1)[0][0]
        return nn


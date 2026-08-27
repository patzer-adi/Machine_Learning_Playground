"""Simple K-Nearest Neighbors classifier/regressor written plainly.

This version uses straightforward loops and comments so it's easy to follow.
"""
import numpy as np


class KNN:
    def __init__(self, k=3, task="classification"):
        """k: number of neighbors; task: 'classification' or 'regression'"""
        self.k = int(k)
        self.task = task
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.X_train = X
        self.y_train = np.asarray(y)
        return self

    def _euclidean_distance(self, a, b):
        """Compute Euclidean distance between two 1-D arrays."""
        return np.sqrt(np.sum((a - b) ** 2))

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        predictions = []
        for x in X:
            # compute distance from x to every training example
            distances = []
            for xt in self.X_train:
                distances.append(self._euclidean_distance(x, xt))

            # get indices of the k smallest distances
            idx_sorted = np.argsort(distances)[: self.k]
            neighbor_labels = self.y_train[idx_sorted]

            if self.task == "classification":
                # majority vote (simple loop to be clear)
                counts = {}
                for lab in neighbor_labels:
                    counts[int(lab)] = counts.get(int(lab), 0) + 1
                # pick label with max count
                best = max(counts.items(), key=lambda t: t[1])[0]
                predictions.append(best)
            else:
                # regression: average of neighbor values
                predictions.append(float(np.mean(neighbor_labels)))

        return np.array(predictions)

    def score(self, X, y):
        preds = self.predict(X)
        if self.task == "classification":
            return float(np.mean(preds == y))
        else:
            from .utils import mse

            return mse(y, preds)

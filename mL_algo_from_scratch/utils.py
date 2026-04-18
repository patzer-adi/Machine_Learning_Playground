"""Tiny helpers used across the examples.

This is written in a simple style so it's easy to read for students.
"""
import numpy as np


def add_intercept(X):
    """Make X two-dimensional (if needed) and add a column of 1s on the left.

    X: array-like, shape (n,) or (n, d)
    returns: array shape (n, d+1)
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


def mse(y_true, y_pred):
    """Mean squared error between true and predicted values."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))

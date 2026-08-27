"""Linear Regression (simple, educational implementation).

Supports closed-form solution (normal equation) and a simple gradient descent.
This file is written plainly with comments to be easy to understand.
"""
import numpy as np
from .utils import add_intercept, mse
from .gradient_descent import GradientDescent


class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.theta = None  # parameters (including intercept if used)

    def _prepare(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            return add_intercept(X)
        return X

    def fit_closed_form(self, X, y, l2=0.0):
        """Fit parameters using normal equation (closed-form).

        l2: ridge regularization strength (simple)
        """
        Xp = self._prepare(X)
        y = np.asarray(y).reshape(-1, 1)
        d = Xp.shape[1]
        I = np.eye(d)
        if self.fit_intercept:
            I[0, 0] = 0.0  # do not regularize bias term
        A = Xp.T @ Xp + l2 * I
        self.theta = np.linalg.pinv(A) @ Xp.T @ y
        self.theta = self.theta.ravel()
        return self

    def _grad(self, theta, X, y):
        """Gradient of MSE loss w.r.t. theta. X already has intercept if needed."""
        m = X.shape[0]
        preds = X @ theta
        grad = (2.0 / m) * (X.T @ (preds - y))
        return grad

    def fit_gradient_descent(self, X, y, lr=0.01, epochs=1000, tol=1e-8, verbose=False):
        """Fit using simple gradient descent (uses GradientDescent helper)."""
        Xp = self._prepare(X)
        y = np.asarray(y).reshape(-1)
        theta0 = np.zeros(Xp.shape[1])
        gd = GradientDescent(lr=lr, epochs=epochs, tol=tol, verbose=verbose)

        def grad_fn(t, Xloc, yloc):
            return self._grad(t, Xloc, yloc)

        theta, history = gd.fit(Xp, y, grad_fn, theta0)
        self.theta = theta
        return self

    def predict(self, X):
        Xp = self._prepare(X)
        if self.theta is None:
            raise ValueError("Model is not fitted yet")
        return Xp @ self.theta

    def score(self, X, y):
        preds = self.predict(X)
        return mse(np.asarray(y), preds)

import numpy as np
from models.optimizers import StochasticGradientDescent, Adam


class LinearRegression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias    # add an extra bias weight
        self.w = None

    def cost_fn(self, X, y, w):
        cost = 0.5 * np.sum((y - np.dot(X, w))**2)
        return cost

    def gradient(self, X, y, w):

        y_hat = np.dot(X, w)
        grad = np.dot(X.transpose(), y_hat - y) / X.shape[0]

        return grad

    # 2.1 Analytic linear regression
    def analytic_fit(self, X, y):
        # w = (A^T.A)^{-1}.A^T.y
        XTX = np.dot(X.transpose(), X)
        XTX_inv = np.linalg.inv(XTX)
        XTX_inv_XT = np.dot(XTX_inv, X.transpose())
        w = np.dot(XTX_inv_XT, y)
        return w

    def fit(self, X, y, analytic_fit=False, learning_rate=.1, epsilon=1e-4, max_iters=1e5, verbose=True, batch_size=1):

        A = np.copy(X)
        if self.add_bias:
            A = np.concatenate([np.ones((A.shape[0], 1)), A], axis=1)

        if analytic_fit:
            self.w = self.analytic_fit(A, y)
        else:
            self.w = np.zeros((A.shape[1], y.shape[1]))
            optimizer = StochasticGradientDescent(learning_rate=learning_rate, epsilon=epsilon, max_iters=max_iters,
                                                  verbose=verbose, batch_size=batch_size)
            self.w = optimizer.run(self.gradient, A, y, self.w)

        return self

    def predict(self, X):
        assert self.w is not None
        A = np.copy(X)
        if self.add_bias:
            A = np.concatenate([np.ones((A.shape[0], 1)), A], axis=1)

        return np.dot(A, self.w)


def softmax(X):
    """Implement softmax to avoid overflow"""
    eps = 1e-8
    return np.exp(X - np.max(X, axis=1, keepdims=True)) / (np.sum(np.exp(X-np.max(X, axis=1, keepdims=True)), axis=1,
                                                                  keepdims=True) + eps)


class LogisticRegression:

    def __init__(self, add_bias=True):
        self.add_bias = add_bias

    def cost_fn(self, X, y, w):
        """Compute the negative log likelihood for multi-class classification, given by the equation in slide 37 of:
        https://www.cs.mcgill.ca/~isabeau/COMP551/F23/slides/4-logisticregression.pdf"""
        z = np.dot(X, w)
        zbar = np.max(z, axis=1, keepdims=True)

        cost = - np.trace(np.dot(y, z.transpose())) + np.sum(
            zbar + np.log(np.sum(np.exp(z - zbar), axis=1, keepdims=True)))
        return cost

    def gradient(self, X, y, w):
        """
        Compute the gradient of the negative log likelihood, according to slide 39 of:
        https://www.cs.mcgill.ca/~isabeau/COMP551/F23/slides/4-logisticregression.pdf
        """
        y_hat = softmax(np.dot(X, w))
        gradient = np.dot(X.transpose(), y_hat - y) / y.shape[0]
        return gradient

    def fit(self, X, y, learning_rate=.1, epsilon=1e-4, max_iters=1e5, batch_size=1, verbose=True):

        A = np.copy(X)
        if self.add_bias:
            A = np.concatenate([np.ones((A.shape[0], 1)), A], axis=1)

        self.w = np.zeros((A.shape[1], y.shape[1]))
        optimizer = Adam(learning_rate=learning_rate, epsilon=epsilon, max_iters=max_iters,
                                              verbose=verbose, batch_size=batch_size, beta_1=0.9, beta_2=0.9)
        self.w = optimizer.run(self.gradient, A, y, self.w)

        return self

    def predict(self, X):
        A = np.copy(X)
        if self.add_bias:
            A = np.concatenate([np.ones((A.shape[0], 1)), A], axis=1)

        y_preds = softmax(np.dot(A, self.w))

        return y_preds

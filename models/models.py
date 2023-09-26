import numpy as np
import pandas as pd


# 2.1 Analytic linear regression

class AnalyticLinearRegression():
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        self.w = None

    def fit(self, X, y):
        A = np.copy(X)
        if self.add_bias:
            A = np.concatenate([np.ones((A.shape[0], 1)), A], axis=1)

        # w = (A^T.A)^{-1}.A^T.y
        ATA = np.dot(A.transpose(), A)
        ATA_inv = np.linalg.inv(ATA)
        ATA_inv_AT = np.dot(ATA_inv, A.transpose())
        self.w = np.dot(ATA_inv_AT,y)

    def predict(self, X):
        assert self.w is not None
        A = np.copy(X)
        if self.add_bias:
            A = np.concatenate([np.ones((A.shape[0], 1)), A], axis=1)

        return np.dot(A, self.w)

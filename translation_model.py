import numpy as np
from itertools import chain


class TranslationModel:
    @classmethod
    def fit(cls, X):
        dx = np.mean(X[:, 0] - X[:, 2])
        dy = np.mean(X[:, 1] - X[:, 3])
        return dx, dy

    @classmethod
    def get_error(cls, X, model):
        return np.mean(np.abs(X[:, :2] - X[:, 2:] - np.array(model)), 1)

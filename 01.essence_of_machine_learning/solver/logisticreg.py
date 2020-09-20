import numpy as np
from scipy import linalg

THRESHMIN = 1e-10


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self, tol=0.001, max_iter=3, random_seed=0):
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_seed)
        self.w_ = None

    def fit(self, X, y):
        self.w_ = self.random_state.randn(X.shape[1] + 1)
        Xtil = np.c_[np.ones(X.shape[0]), X]
        diff = np.inf
        w_prev = self.w_
        iter = 0
        while diff > self.tol and iter < self.max_iter:
            yhat = sigmoid(np.dot(Xtil, self.w_))
            
            # 1e-10〜無限大の範囲に収める。
            # 対角行列の成分が0になると計算できないため。
            r = np.clip(yhat * (1 - yhat),
                        THRESHMIN, np.inf)
            
            XR = Xtil.T * r
            XRX = np.dot(Xtil.T * r, Xtil)
            w_prev = self.w_
            b = np.dot(XR, np.dot(Xtil, self.w_) -
                       1 / r * (yhat - y))
            self.w_ = linalg.solve(XRX, b)
            w_abs = np.sqrt(self.w_ ** 2).mean()
            diff = abs(w_prev - self.w_).mean() / w_abs
            iter += 1
            print("{0:4d}: Diff={1:e}, Abs={2:e}".format(iter, diff, w_abs))

    def predict(self, X):
        Xtil = np.c_[np.ones(X.shape[0]), X]
        yhat = sigmoid(np.dot(Xtil, self.w_))
        return np.where(yhat > .5, 1, 0)


import numpy as np
from operator import itemgetter

def rel_error(a, b):
    magnitude = 0.5 * (abs(a) + abs(b))
    error = abs(a-b) / magnitude
    return error

class SVCHard:
    def fit(self, X, y, selections=None):
        a = np.zeros(X.shape[0])
        ay = 0
        ayx = np.zeros(X.shape[1])
        yx = y.reshape(-1, 1)*X
        indices = np.arange(X.shape[0])
        while True:
            ydf = y*(1-np.dot(yx, ayx.T))
            iydf = np.c_[indices, ydf]
            i = int(min(iydf[(y < 0) | (a > 0)],
                        key=itemgetter(1))[0])
            j = int(max(iydf[(y > 0) | (a > 0)],
                        key=itemgetter(1))[0])
            if ydf[i] >= ydf[j]:
                break
            ay2 = ay - y[i]*a[i] - y[j]*a[j]
            ayx2 = ayx - y[i]*a[i]*X[i, :] - y[j]*a[j]*X[j, :]
            ai = ((1-y[i]*y[j]
                   + y[i]*np.dot(X[i, :] - X[j, :],
                                 X[j, :]*ay2 - ayx2))
                  / ((X[i] - X[j])**2).sum())
            if ai < 0:
                ai = 0
            aj = (-ai * y[i] - ay2) * y[j]
            if aj < 0:
                aj = 0
                ai = (-aj*y[j] - ay2)*y[i]
            ay += y[i]*(ai - a[i]) + y[j]*(aj - a[j])
            ayx += y[i]*(ai - a[i])*X[i, :] + y[j]*(aj - a[j])*X[j, :]
            
            #print(ai, a[i])
            print("ai={0:.6f}, a[i]={1:.6f}".format(ai, a[i]))

            # 浮動小数点の一致比較は収束条件に出来ないので
            # 相対誤差で判定する。
            #if ai == a[i]:
            if rel_error(ai,a[i]) < 1.0e-6:
                break
            a[i] = ai
            a[j] = aj
            
        self.a_ = a
        ind = a != 0.
        self.w_ = ((a[ind] * y[ind]).reshape(-1, 1)
                   * X[ind, :]).sum(axis=0)
        self.w0_ = (y[ind]
                    - np.dot(X[ind, :], self.w_)).sum() / ind.sum()

    def predict(self, X):
        return (self.w0_ + np.dot(X, self.w_))
        #return np.sign(self.w0_ + np.dot(X, self.w_))

class SVCSoft:
    def __init__(self, C=1.):
        self.C = C

    def fit(self, X, y, selections=None):
        a = np.zeros(X.shape[0])
        ay = 0
        ayx = np.zeros(X.shape[1])
        yx = y.reshape(-1, 1)*X
        indices = np.arange(X.shape[0])
        while True:
            ydf = y*(1-np.dot(yx, ayx.T))
            iydf = np.c_[indices, ydf]
            i = int(min(iydf[((a > 0) & (y > 0)) |
                             ((a < self.C) & (y < 0))],
                        key=itemgetter(1))[0])
            j = int(max(iydf[((a > 0) & (y < 0)) |
                             ((a < self.C) & (y > 0))],
                        key=itemgetter(1))[0])
            if ydf[i] >= ydf[j]:
                break
            ay2 = ay - y[i]*a[i] - y[j]*a[j]
            ayx2 = ayx - y[i]*a[i]*X[i, :] - y[j]*a[j]*X[j, :]
            ai = ((1-y[i]*y[j]
                   + y[i]*np.dot(X[i, :] - X[j, :],
                                 X[j, :]*ay2 - ayx2))
                  / ((X[i] - X[j])**2).sum())
            if ai < 0:
                ai = 0
            elif ai > self.C:
                ai = self.C
            aj = (-ai * y[i] - ay2) * y[j]
            if aj < 0:
                aj = 0
                ai = (-aj*y[j]-ay2)*y[i]
            elif aj > self.C:
                aj = self.C
                ai = (-aj*y[j]-ay2)*y[i]
            ay += y[i]*(ai - a[i]) + y[j]*(aj - a[j])
            ayx += y[i]*(ai - a[i])*X[i, :] + y[j]*(aj - a[j])*X[j, :]
            
            #if ai == a[i]:
            if rel_error(ai,a[i]) < 1.0e-6:
                break
            a[i] = ai
            a[j] = aj
        self.a_ = a
        ind = a != 0.
        self.w_ = ((a[ind] * y[ind]).reshape(-1, 1)
                   * X[ind, :]).sum(axis=0)
        self.w0_ = (y[ind]
                    - np.dot(X[ind, :], self.w_)).sum() / ind.sum()

    def predict(self, X):
        return np.sign(self.w0_ + np.dot(X, self.w_))

class RBFKernel:
    def __init__(self, X, sigma):
        self.sigma2 = sigma**2
        self.X = X
        self.values_ = np.empty((X.shape[0], X.shape[0]))

    def value(self, i, j):
        return np.exp(-((self.X[i, :] - self.X[j, :])**2).sum()
                      / (2*self.sigma2))

    def eval(self, Z, s):
        return np.exp(-((self.X[s, np.newaxis, :]
                         - Z[np.newaxis, :, :])**2).sum(axis=2)
                      / (2*self.sigma2))


class SVCKernel:
    def __init__(self, C=1., sigma=1., max_iter=10000):
        self.C = C
        self.sigma = sigma
        self.max_iter = max_iter

    def fit(self, X, y, selections=None):
        a = np.zeros(X.shape[0])
        ay = 0
        kernel = RBFKernel(X, self.sigma)
        indices = np.arange(X.shape[0])
        for _ in range(self.max_iter):
            s = a != 0.
            ydf = y * (1 - y*np.dot(a[s]*y[s],
                                    kernel.eval(X, s)).T)
            iydf = np.c_[indices, ydf]
            i = int(min(iydf[((a > 0) & (y > 0)) |
                             ((a < self.C) & (y < 0))],
                        key=itemgetter(1))[0])
            j = int(max(iydf[((a > 0) & (y < 0)) |
                             ((a < self.C) & (y > 0))],
                        key=itemgetter(1))[0])
            if ydf[i] >= ydf[j]:
                break
            ay2 = ay - y[i]*a[i] - y[j]*a[j]
            kii = kernel.value(i, i)
            kij = kernel.value(i, j)
            kjj = kernel.value(j, j)
            s = a != 0.
            s[i] = False
            s[j] = False
            kxi = kernel.eval(X[i, :].reshape(1, -1), s).ravel()
            kxj = kernel.eval(X[j, :].reshape(1, -1), s).ravel()
            ai = ((1 - y[i]*y[j]
                   + y[i]*((kij - kjj)*ay2
                           - (a[s]*y[s]*(kxi-kxj)).sum()))
                  / (kii + kjj - 2*kij))
            if ai < 0:
                ai = 0
            elif ai > self.C:
                ai = self.C
            aj = (-ai*y[i] - ay2)*y[j]
            if aj < 0:
                aj = 0
                ai = (-aj*y[j] - ay2)*y[i]
            elif aj > self.C:
                aj = self.C
                ai = (-aj*y[j] - ay2)*y[i]
            ay += y[i] * (ai-a[i]) + y[j] * (aj-a[j])
            if ai == a[i]:
                break
            a[i] = ai
            a[j] = aj
        self.a_ = a
        self.y_ = y
        self.kernel_ = kernel
        s = a != 0.
        self.w0_ = (y[s]
                    - np.dot(a[s]*y[s],
                             kernel.eval(X[s], s))).sum() / s.sum()
        with open("svm.log", "w") as fp:
            print(a, file=fp)

    def predict(self, X):
        s = self.a_ != 0.
        return np.sign(self.w0_
                       + np.dot(self.a_[s]*self.y_[s],
                                self.kernel_.eval(X, s)))


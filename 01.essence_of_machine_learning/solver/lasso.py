import numpy as np

# x > 0のとき、0以上の値を返し、
# x < 0のとき、0以下の値を返す。
# y > |x|のとき、0を返す。
def soft_thresholding(x, y):
    return np.sign(x) * max(abs(x) - y, 0)


class Lasso:
    # tol: トレランス(収束判定のための許容度)
    # max_iter: 最大繰り返し回数
    def __init__(self, lambda_, tol=0.0001, max_iter=1000):
        self.lambda_ = lambda_
        self.tol = tol
        self.max_iter = max_iter
        self.w_ = None

    # t: 正解データ
    def fit(self, X, t):
        n, d = X.shape
        self.w_ = np.zeros(d + 1)
        avgl1 = 0.
        for _ in range(self.max_iter):
            avgl1_prev = avgl1
            self._update(n, d, X, t)

            # avgl1: |w|/dの変化量がtol以下になるまで繰り返す
            avgl1 = np.abs(self.w_).sum() / self.w_.shape[0]
            if abs(avgl1 - avgl1_prev) <= self.tol:
                break

    def _update(self, n, d, X, t):
        # w0の計算は例外のため、先に求める
        self.w_[0] = (t - np.dot(X, self.w_[1:])).sum() / n
        w0vec = np.ones(n) * self.w_[0]

        # 1次元ずつ偏微分係数が0になるようなxを求め、
        # そのxを次の初期値とする。
        for k in range(d):
            ww = self.w_[1:]
            ww[k] = 0 # w_k=0とおいて、w_kについて解く。
            q = np.dot(t - w0vec - np.dot(X, ww), X[:, k])
            r = np.dot(X[:, k], X[:, k])
            self.w_[k + 1] = soft_thresholding(q / r, self.lambda_)

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)
        Xtil = np.c_[np.ones(X.shape[0]), X]
        return np.dot(Xtil, self.w_)


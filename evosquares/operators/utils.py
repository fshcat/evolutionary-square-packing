import numpy as np

def sort_closest(X, R):
    p, n_vars = X.shape
    n = n_vars // 3

    x = X[:, :n]
    y = X[:, n:n*2]
    t = X[:, n*2:]

    dists = (R[:, :n] - x)**2 + (R[:, n:n*2] - y)**2

    sort_inds = np.argsort(dists, axis=1)
    x_sorted = np.take_along_axis(x, sort_inds, axis=1)
    y_sorted = np.take_along_axis(y, sort_inds, axis=1)
    t_sorted = np.take_along_axis(t, sort_inds, axis=1)

    return x_sorted, y_sorted, t_sorted

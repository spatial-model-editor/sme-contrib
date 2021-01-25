import numpy as np
import multiprocessing


def hessian_f(dat):
    (typ, i, j, f, x0, rel_eps) = dat
    if typ == 0:
        x = np.array(x0, dtype=np.float64)
        return f(x)
    if typ == 1:
        x = np.array(x0, dtype=np.float64)
        dx_i = rel_eps * x[i]
        x[i] = x[i] + dx_i
        return f(x)
    x = np.array(x0, dtype=np.float64)
    dx_i = rel_eps * x[i]
    dx_j = rel_eps * x[j]
    x[i] = x[i] + dx_i
    x[j] = x[j] + dx_j
    return f(x)


# Numerical approx to hessian of f at x0
# requires N^2 + N + 1 evaluations of f
# https://en.wikipedia.org/wiki/Finite_difference#Multivariate_finite_differences
def hessian(f, x0, rel_eps=1e-2):
    pool = multiprocessing.Pool()
    n = len(x0)
    idxs = np.zeros((n, n), dtype=np.int)
    dat = []
    dat.append((0, 0, 0, f, x0, rel_eps))  # f_(0,0)
    for i in range(n):
        dat.append((1, i, 0, f, x0, +rel_eps))  # f_(i+,0)
    for i in range(n):
        dat.append((1, i, 0, f, x0, -rel_eps))  # f_(i-,0)
    for i in range(n):
        for j in range(0, i):
            dat.append((2, i, j, f, x0, +rel_eps))  # f_(i+,j+)
    k = 0
    for i in range(n):
        for j in range(0, i):
            dat.append((2, i, j, f, x0, -rel_eps))  # f_(i-,j-)
            idxs[(i, j)] = k
            k = k + 1
    ff = pool.map(hessian_f, dat)
    pool.close()
    pool.join()
    h = np.zeros((n, n))
    # diagonal elements
    for i in range(n):
        h[(i, i)] = ff[i + 1] - 2 * ff[0] + ff[n + i + 1]
        h[(i, i)] /= rel_eps * rel_eps * x0[i] * x0[i]
    # off-diagonal elements
    for i in range(n):
        for j in range(0, i):
            idx = idxs[(i, j)]
            h[(i, j)] = (
                ff[idx + 2 * n + 1]
                - ff[i + 1]
                + ff[0]
                - ff[j + 1]
                + ff[0]
                - ff[n + i + 1]
                + ff[idx + 2 * n + 1 + (n * (n - 1) // 2)]
                - ff[n + j + 1]
            )
            h[(i, j)] /= 2.0 * rel_eps * rel_eps * x0[i] * x0[j]
            h[(j, i)] = h[(i, j)]
    return h

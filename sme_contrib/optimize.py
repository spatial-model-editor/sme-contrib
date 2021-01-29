import numpy as np
import multiprocessing
import pyswarms as ps
from functools import partial


def _hessian_f(f, x0, i, j, rel_eps):
    if i is None:
        x = np.array(x0, dtype=np.float64)
        return f(x)
    if j is None:
        x = np.array(x0, dtype=np.float64)
        x[i] = (1.0 + rel_eps) * x[i]
        return f(x)
    x = np.array(x0, dtype=np.float64)
    x[i] = (1.0 + rel_eps) * x[i]
    x[j] = (1.0 + rel_eps) * x[j]
    return f(x)


# Numerical approx to hessian of f at x0
# requires N^2 + N + 1 evaluations of f
# https://en.wikipedia.org/wiki/Finite_difference#Multivariate_finite_differences
def hessian(f, x0, rel_eps=1e-2):
    n = len(x0)
    # make list of arguments for each f call required
    args = []
    # f([.., x_i, .., x_j, ..])
    args.append((f, x0, None, None, rel_eps))
    # f([.., x_i+dx_i, .., x_j, ..])
    for i in range(n):
        args.append((f, x0, i, None, +rel_eps))
    # f([.., x_i-dx_i, .., x_j, ..])
    for i in range(n):
        args.append((f, x0, i, None, -rel_eps))
    # f([.., x_i+dx_i, .., x_j+dx_j, ..])
    for i in range(n):
        for j in range(0, i):
            args.append((f, x0, i, j, +rel_eps))
    # f([.., x_i-dx_i, .., x_j-dx_j, ..])
    for i in range(n):
        for j in range(0, i):
            args.append((f, x0, i, j, -rel_eps))
    # call f with each set of args
    pool = multiprocessing.Pool()
    ff = pool.starmap(_hessian_f, args)
    pool.close()
    pool.join()
    # construct hessian elements from these values
    h = np.zeros((n, n))
    # diagonal elements
    for i in range(n):
        h[(i, i)] = ff[i + 1] - ff[0] + ff[n + i + 1] - ff[0]
        h[(i, i)] /= rel_eps * rel_eps * x0[i] * x0[i]
    # off-diagonal elements
    offset_ij = 2 * n + 1
    n_ij = (n * (n - 1)) // 2
    index_ij = 0
    for i in range(n):
        for j in range(0, i):
            h[(i, j)] = (
                ff[0]
                - ff[1 + i]
                + ff[0]
                - ff[1 + j]
                + ff[offset_ij + index_ij]
                - ff[1 + n + i]
                + ff[offset_ij + n_ij + index_ij]
                - ff[1 + n + j]
            )
            h[(i, j)] /= 2.0 * rel_eps * rel_eps * x0[i] * x0[j]
            h[(j, i)] = h[(i, j)]
            index_ij += 1
    return h


# rescale `x` such that the maximum element equals `new_max_element`
def rescale(x, new_max_element):
    old_max_element = np.amax(x)
    return np.multiply(x, new_max_element / old_max_element)


# sum of squares difference between every element of x and y
def abs_diff(x, y):
    return 0.5 * np.sum(np.power(x - y, 2))


# calculate function f for each x in xs
def _f_eval(xs, f):
    pool = multiprocessing.Pool(processes=12)
    norms = pool.map(f, xs)
    pool.close()
    pool.join()
    return np.array(norms)


# minimize objective function using particle swarm
def minimize(f, lowerbounds, upperbounds, particles=20, iterations=20):
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    optimizer = ps.single.GlobalBestPSO(
        particles,
        dimensions=len(lowerbounds),
        options=options,
        bounds=(np.array(lowerbounds), np.array(upperbounds)),
    )
    ps_cost, ps_res = optimizer.optimize(_f_eval, iters=iterations, f=f)
    return ps_cost, ps_res

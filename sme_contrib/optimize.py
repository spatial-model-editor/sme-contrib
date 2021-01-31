"""Optimization, fitting and analysis"""


import numpy as np
import multiprocessing
import pyswarms as ps
from functools import partial
from os import cpu_count


def _hessian_f(f, x0, i, j, rel_eps):
    x = np.array(x0, dtype=np.float64)
    if i is None:
        return f(x)
    if j is None:
        x[i] = (1.0 + rel_eps) * x[i]
        return f(x)
    x[i] = (1.0 + rel_eps) * x[i]
    x[j] = (1.0 + rel_eps) * x[j]
    return f(x)


def hessian(f, x0, rel_eps=1e-2, processes=None):
    """Approximate Hessian of function ``f`` at point ``x0``

    Uses a `finite difference`_ approximation where the step size used
    for each element ``i`` of ``x0`` is ``rel_eps * x[i]``.

    Requires :math:`N^2 + N + 1` evalulations of ``f``, where :math:`N`
    is the number of elements of ``x0``

    The evaluations of ``f`` are done in parallel, so ``f`` must be a
    thread-safe function that can safely be called from multiple threads
    at the same time.

    Note:
        This choice of step size allows the different elements of x0 to have
        vastly different scales without causing numerical instabilities,
        but it will fail if an element of ``x0`` is equal to 0.

    Args:
        f: The function to evaluate, it should be callable as f(x0) and return a scalar
        x0: The point at which to evaluate the function, a flot or list of floats.
        rel_eps: The relative step size to use
        processes: The number of processes to use (uses number of cpu cores as default)

    Returns:
        np.array: The Hessian as a 2d numpy array of floats

    .. _finite difference:
        https://en.wikipedia.org/wiki/Finite_difference#Multivariate_finite_differences

    """
    if processes == None:
        processes = cpu_count()
    n = len(x0)
    # make list of arguments for each f call required
    args = []
    # f(x)
    args.append((f, x0, None, None, rel_eps))
    # f([.., x_i+dx_i, ..])
    for i in range(n):
        args.append((f, x0, i, None, +rel_eps))
    # f([.., x_i-dx_i, ..])
    for i in range(n):
        args.append((f, x0, i, None, -rel_eps))
    # f([.., x_j+dx_j, .., x_i+dx_i, ..])
    for i in range(n):
        for j in range(0, i):
            args.append((f, x0, i, j, +rel_eps))
    # f([.., x_j-dx_j, .., x_i-dx_i, ..])
    for i in range(n):
        for j in range(0, i):
            args.append((f, x0, i, j, -rel_eps))
    # call f with each set of args
    pool = multiprocessing.Pool(processes)
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


# calculate function f for each x in xs
def _minimize_f(xs, f, processes):
    pool = multiprocessing.Pool(processes=processes)
    norms = pool.map(f, xs)
    pool.close()
    pool.join()
    return np.array(norms)


# minimize objective function using particle swarm
def minimize(f, lowerbounds, upperbounds, particles=20, iterations=20, processes=None):
    """Minimize function ``f`` using particle swarm

    The function ``f`` should take an array or list of parameters ``x``, and return a
    value: parameters will be found using particle swarm that minimize this value.

    Each parameter should have a specified lower and upper bound.

    The evaluations of ``f`` are done in parallel, so ``f`` must be a
    thread-safe function that can safely be called from multiple threads
    at the same time. The evaluations are parallelized over the particles
    for each iteration, so for good performance the number of particles should
    be larger than the number of processes.

    Note:
        This choice of step size allows the different elements of x0 to have
        vastly different scales without causing numerical instabilities,
        but it will fail if an element of ``x0`` is equal to 0.

    Args:
        f: The function to evaluate, it should be callable as f(x) and return a scalar
        lowerbounds: The lower bound for each element of x.
        upperbounds: The upper bound for each element of x.
        particles: The number of particles to use in the swarm
        iterations: The number of iterations to do
        processes: The number of processes to use (if not set uses all available cpu cores)

    Returns:
        np.array: The Hessian as a 2d numpy array of floats

    .. _finite difference:
        https://en.wikipedia.org/wiki/Finite_difference#Multivariate_finite_differences

    """
    if processes == None:
        processes = cpu_count()
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    optimizer = ps.single.GlobalBestPSO(
        particles,
        dimensions=len(lowerbounds),
        options=options,
        bounds=(np.array(lowerbounds), np.array(upperbounds)),
    )
    ps_cost, ps_res = optimizer.optimize(
        _minimize_f, iters=iterations, f=f, processes=processes
    )
    return ps_cost, ps_res


def rescale(x, new_max_element):
    """Rescale an array

    Args:
        x(numpy.array): The array to rescale
        new_max_element(float): The desired new maximum element value

    Returns:
        np.array: The rescaled array
    """
    old_max_element = np.amax(x)
    scale_factor = new_max_element / old_max_element
    return np.multiply(x, scale_factor)


def abs_diff(x, y):
    """Absolute difference between two arrays

    :math:`\\frac{1}{2} \\sum_i  (x_i - y_i)^2`

    Args:
        x(numpy.array): The first array
        y(numpy.array): The second array

    Returns:
        float: absolute difference between the two arrays
    """
    return 0.5 * np.sum(np.power(x - y, 2))

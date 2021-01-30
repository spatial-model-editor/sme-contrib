import numpy as np
import multiprocessing
import pyswarms as ps
from functools import partial
from os import cpu_count

# calculate function f for each x in xs
def _f_eval(xs, f, processes):
    pool = multiprocessing.Pool(processes=processes)
    norms = pool.map(f, xs)
    pool.close()
    pool.join()
    return np.array(norms)


# minimize objective function using particle swarm
def minimize(
    f, lowerbounds, upperbounds, particles=20, iterations=20, processes=cpu_count()
):
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
        processes: The number of processes to use (uses number of cpu cores as default)

    Returns:
        np.array: The Hessian as a 2d numpy array of floats

    .. _finite difference:
        https://en.wikipedia.org/wiki/Finite_difference#Multivariate_finite_differences

    """
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    optimizer = ps.single.GlobalBestPSO(
        particles,
        dimensions=len(lowerbounds),
        options=options,
        bounds=(np.array(lowerbounds), np.array(upperbounds)),
    )
    ps_cost, ps_res = optimizer.optimize(
        _f_eval, iters=iterations, f=f, processes=processes
    )
    return ps_cost, ps_res

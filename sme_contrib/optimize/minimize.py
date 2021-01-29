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

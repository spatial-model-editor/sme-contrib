import sme_contrib.optimize as opt
import numdifftools
import numpy as np

# analytic test function for minimize with minimum at x = [0.375, -0.9]
def f(x):
    return 1.23 + (x[0] - 0.375) ** 2 + (x[1] + 0.9) ** 4


def test_minimize() -> None:
    cost, res = opt.minimize(f, [-5.0, -5.0], [5.0, 5.0], particles=24, iterations=100)
    assert np.abs(res[0] - 0.375) < 0.100
    assert np.abs(res[1] + 0.900) < 0.100

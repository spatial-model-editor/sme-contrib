import sme_contrib.optimize as opt
import numdifftools
import numpy as np

# analytic test function for hessian
def r(x):
    return (
        (1.123 - x[0]) ** 2
        + 105124.124 * (x[1] - x[0] ** 2) ** 2
        + 0.123 * (x[0] ** 2) * (x[1] ** 3) * (x[2] ** 5)
    )


def test_hessian() -> None:
    # Simple test of the hessian approximation using an analytic function of three vars
    r0 = [1.86545, 1.123, 0.789]
    # get reference solution using numdifftools
    h0 = numdifftools.Hessian(r)(r0)
    norm = np.sqrt(np.mean(np.power(h0, 2)))
    # compare to hessian() with default step size:
    h1 = opt.hessian(r, r0)
    assert np.sqrt(np.mean(np.power(h1 - h0, 2))) / norm < 1e-4
    # compare to hessian() with smaller step size:
    h2 = opt.hessian(r, r0, 1e-5)
    assert np.sqrt(np.mean(np.power(h2 - h0, 2))) / norm < 1e-6


# analytic test function for minimize with minimum at x = [0.375, -0.9]
def f(x):
    return 1.23 + (x[0] - 0.375) ** 2 + (x[1] + 0.9) ** 4


def test_minimize() -> None:
    cost, res = opt.minimize(f, [-5.0, -5.0], [5.0, 5.0], particles=24, iterations=100)
    assert np.abs(res[0] - 0.375) < 0.100
    assert np.abs(res[1] + 0.900) < 0.100


def test_rescale() -> None:
    a = np.array([1.0, 3.0, 5.0])
    a_rescaled = opt.rescale(a, 3.4)
    factor = 3.4 / 5.0
    assert len(a_rescaled) == 3
    assert np.abs(a_rescaled[0] - factor) < 1e-13
    assert np.abs(a_rescaled[1] - 3.0 * factor) < 1e-13
    assert np.abs(a_rescaled[2] - 5.0 * factor) < 1e-13


def test_abs_diff() -> None:
    a = np.array([1.0, 3.0, 5.0])
    b = np.array([2.0, 3.1, 5.0])
    assert np.abs(opt.abs_diff(a, b) - 0.505) < 1e-13

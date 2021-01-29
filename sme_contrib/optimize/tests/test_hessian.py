from sme_contrib.optimize.hessian import hessian
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
    h1 = hessian(r, r0)
    assert np.sqrt(np.mean(np.power(h1 - h0, 2))) / norm < 1e-4
    # compare to hessian() with smaller step size:
    h2 = hessian(r, r0, 1e-5)
    assert np.sqrt(np.mean(np.power(h2 - h0, 2))) / norm < 1e-6

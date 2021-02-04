import sme_contrib.optimize as opt
import numdifftools
import numpy as np
import os.path


def _get_abs_path(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


# analytic test function for hessian
def _r(x):
    return (
        (1.123 - x[0]) ** 2
        + 105124.124 * (x[1] - x[0] ** 2) ** 2
        + 0.123 * (x[0] ** 2) * (x[1] ** 3) * (x[2] ** 5)
    )


def test_hessian() -> None:
    # Simple test of the hessian approximation using an analytic function of three vars
    r0 = [1.86545, 1.123, 0.789]
    # get reference solution using numdifftools
    h0 = numdifftools.Hessian(_r)(r0)
    norm = np.sqrt(np.mean(np.power(h0, 2)))
    # compare to hessian() with default step size:
    h1 = opt.hessian(_r, r0)
    assert np.sqrt(np.mean(np.power(h1 - h0, 2))) / norm < 1e-4
    # compare to hessian() with smaller step size:
    h2 = opt.hessian(_r, r0, 1e-5)
    assert np.sqrt(np.mean(np.power(h2 - h0, 2))) / norm < 1e-6


# analytic test function for minimize with minimum at x = [0.375, -0.9]
def _f(x):
    return 1.23 + (x[0] - 0.375) ** 2 + (x[1] + 0.9) ** 4


def test_minimize() -> None:
    cost, res, optimizer = opt.minimize(
        _f,
        [-5.0, -5.0],
        [5.0, 5.0],
        particles=24,
        iterations=20,
        ps_options={"c1": 2.025, "c2": 2.025, "w": 0.5},
    )
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


def apply_params(model, params):
    model.compartments["Nucleus"].species["A_nucl"].diffusion_constant = params[0]


def test_steady_state() -> None:
    ss = opt.SteadyState(
        modelfile=_get_abs_path("model.xml"),
        imagefile=_get_abs_path("conc.png"),
        species=["A_nucl"],
        function_to_apply_params=apply_params,
        lower_bounds=[0.001],
        upper_bounds=[10.0],
        simulation_time=200,
        steady_state_time=20,
    )
    # obj_func should have minimum around diff_const ~ 2, since that was used to generate the concentration image
    # much higher: not enough of a gradient inside nucleus
    assert ss._obj_func([10.0]) > ss._obj_func([2.0])
    # much lower: too strong of a gradient inside nucleus
    assert ss._obj_func([0.01]) > ss._obj_func([2.0])
    # try to find parameter
    params = ss.find(particles=4, iterations=10)
    # few particles/iterations: require [0.001, 10.0] -> [-1, 5]
    assert np.abs(params[0] - 2.0) < 3.0
    assert (
        ss.get_model().compartments["Nucleus"].species["A_nucl"].diffusion_constant
        == params[0]
    )

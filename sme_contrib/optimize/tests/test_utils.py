import sme_contrib.optimize as opt
import numpy as np


def test_rescale() -> None:
    a = np.array([1.0, 3.0, 5.0])
    a_rescaled = opt.utils.rescale(a, 3.4)
    factor = 3.4 / 5.0
    assert len(a_rescaled) == 3
    assert np.abs(a_rescaled[0] - factor) < 1e-13
    assert np.abs(a_rescaled[1] - 3.0 * factor) < 1e-13
    assert np.abs(a_rescaled[2] - 5.0 * factor) < 1e-13


def test_abs_diff() -> None:
    a = np.array([1.0, 3.0, 5.0])
    b = np.array([2.0, 3.1, 5.0])
    assert np.abs(opt.utils.abs_diff(a, b) - 0.505) < 1e-13

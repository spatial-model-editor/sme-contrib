import numpy as np


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

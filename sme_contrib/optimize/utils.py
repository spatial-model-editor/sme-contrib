import numpy as np

# rescale `x` such that the maximum element equals `new_max_element`
def rescale(x, new_max_element):
    old_max_element = np.amax(x)
    return np.multiply(x, new_max_element / old_max_element)


# sum of squares difference between every element of x and y
def abs_diff(x, y):
    return 0.5 * np.sum(np.power(x - y, 2))

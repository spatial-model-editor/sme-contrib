import pyvista as pv
import numpy as np
from itertools import cycle
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def rgb_to_scalar(img: np.ndarray) -> np.ndarray:
    """
    Convert an RGB 3D image represented as a 4D tensor to a 3D image tensor where each unique RGB value is assigned a unique scalar, i.e., it contracts the dimension with the RGB values into scalars in such a way that 2 different colors are mapped to 2 different scalars, too. This is needed because PyVista doesn't work with RGB values directly and expects fields defined on a grid.

        img (np.ndarray): A 3D numpy array representing an RGB image with shape (height, width, 3).

        np.ndarray: A 2D numpy array with the same height and width as the input image, where each pixel's value
                    corresponds to a unique scalar representing the original RGB value.
    """
    reshaped = img.reshape(-1, 3, copy=True)
    unique_rgb, ridx = np.unique(reshaped, axis=0, return_inverse=True)

    values = np.arange(len(unique_rgb))
    return values[ridx].reshape(img.shape[:-1])


def make_discrete_colormap(
    cmap: str = "tab10", values: np.ndarray = np.array([])
) -> pv.LookupTable:
    """
    Create a discrete colormap for use with PyVista with as many colors as unique values in the `values`array based on a given matplotlbit colormap. The colors will possibly repeat if there are more unique values than colors in the colormap. In this case, the outcome is intended, e.g., for separability of regions in the visualization,

    Parameters:
    cmap (str): The name of the colormap to use. Default is 'tab10'.
    values (np.ndarray): An array of values to map to colors. Default is an empty array.

    Returns:
    pv.LookupTable: A PyVista LookupTable object with the values drawn from the specified colormap in RGBA format.
    """
    cm = []

    if values.size == 0:
        values = np.arange(0, 1, 1)
        cm = [
            mcolors.to_rgba(plt.get_cmap(cmap).colors[0]),
        ]
    else:
        i = 0
        for c in cycle(plt.get_cmap(cmap).colors):
            cm.append(mcolors.to_rgba(c))
            if len(cm) >= len(values):
                break
            i += 1
    lt = pv.LookupTable(
        values=np.array(cm) * 255,
        scalar_range=(0, len(values)),
        n_values=len(values),
    )

    return lt


def find_layout(num_plots: int, portrait: bool = False) -> tuple[int, int]:
    """Find a reasonable layout for a grid of subplots. This splits num_subplots into n x m subplots where n and m are as close as possible to each other. This can include a case where n x m > num_plots. Then, the superficial panels in the grid are ignored in the plotting process.

    Args:
        num_plots (int): Number of plots to arrange
        portrait (bool, optional): Whether the min or max of (n,m) should be the column number in the resulting grid. Defaults to False.

    Returns:
        tuple[int, int]: Tuple describing (n_rows, n_cols) of the grid
    """

    # for checking approximation accuracy with ints. if root > root_int, then
    # we need to adjust n_row, n_cols sucht that n_row * n_cols >= root^2
    root = np.sqrt(num_plots)
    root_int = np.rint(root)

    if np.isclose(root, root_int):
        return int(root_int), int(root_int)  # perfect square because root is an integer
    else:
        # approximation by integer root is inexact

        #  find an approximation that is close to square such that n_row * n_cols - num_plots is
        # as small as possible
        a = int(np.floor(root))
        b = int(np.ceil(root))

        a_1 = int(a - 1)
        b_1 = int(b + 1)

        # make a couple of guesses that are close to the root and select the best one
        guesses = [
            (x, y)
            for x, y in [
                (a, b),
                (a_1, b_1),
                (a, b_1),
                (a_1, b),
            ]
            if x * y >= num_plots
        ]
        best_guess = guesses[
            np.argmin([x * y for x, y in guesses])
        ]  # smallest possible approximation

        # handle orientation of the grid. min => rows for landscape, min=> cols for portrait
        return (
            (np.min(best_guess), np.max(best_guess))
            if not portrait
            else (np.max(best_guess), np.min(best_guess))
        )

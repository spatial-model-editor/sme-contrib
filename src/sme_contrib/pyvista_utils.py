import pyvista as pv
import numpy as np
from typing import Callable, Any
from itertools import cycle
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import sme


def rgb_to_scalar(img: np.ndarray) -> np.ndarray:
    """
    Convert an RGB 3D image represented as a 4D tensor to a 3D image tensor where each unique RGB value is assigned a unique scalar, i.e., it contracts the dimension with the RGB values into scalars. This is useful because PyVista doesn't work well with RGB values directly and expects fields defined on a grid, usually given by the tensor shape.

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
    Create a discrete colormap for use with PyVista.

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


def facet_grid(
    data: dict[str, np.ndarray],
    plotfuncs: dict[str, Callable],
    show_cmap: bool = False,
    cmap: str | np.ndarray | pv.LookupTable = "viridis",
    portrait: bool = False,
    linked_views: bool = True,
    plotter_kwargs: dict = {},
    plotfuncs_kwargs: dict[str, dict[str, Any]] = {},
) -> pv.Plotter:
    """
    Create a 3D facet plot using PyVista. This follows the seaborn.FacetGrid concept. This function creates a grid of subplots where each subplot is filled by a function in the plotfuncs argument. The keys for plotfuncs and data must be the same, such that plotfuncs can be unambiguously mapped over the data dictionary.
    Do not attempt to plot 2D images and 3D images into the same facet grid, as this will create odd artifacts and
    may not work as expected.
    Parameters:
    -----------
    data : dict[str, np.ndarray]
        A dictionary where keys are labels and values are numpy arrays containing the data to be plotted.
    plotfuncs : dict[str, Callable]
        A dictionary where keys are labels and values are functions with signature f(
                label:str,
                data:np.ndarray | pyvista.ImageData | pyvista.UniformGrid,
                plotter:pv.Plotter,
                panel:tuple[int, int],
                show_cmap:bool=show_cmap,
                cmap=cmap,
                **plotfuncs_kwargs
            ) -> None
    show_cmap : bool, optional
        Whether to show the color map. Default is False.
    cmap : str | np.ndarray | pv.LookupTable, optional
        The color map to use. Default is "viridis".
    portrait : bool, optional
        Whether to use a portrait layout. Default is False.
    linked_views : bool, optional
        Whether to link the views of the subplots. Default is True.
    plotter_kwargs : dict, optional
        Additional keyword arguments to pass to the PyVista Plotter.
    plotfuncs_kwargs : dict[str, dict[str, Any]], optional
        Additional keyword arguments to pass to each plotting function.

    Returns:
    --------
    pv.Plotter
        The PyVista Plotter object with the created facet plot.
    """
    if data.keys() != plotfuncs.keys():
        raise ValueError(
            "The keys for the data and plotfuncs dictionaries must be the same."
        )

    layout = find_layout(len(data), portrait=portrait)

    plotter = pv.Plotter(shape=layout, **plotter_kwargs)

    label = iter(plotfuncs.keys())

    for i in range(layout[0]):
        for j in range(layout[1]):
            current_label = next(label)
            plotfuncs[current_label](
                current_label,
                data[current_label],
                plotter,
                panel=(i, j),
                show_cmap=show_cmap,
                cmap=cmap,
                **plotfuncs_kwargs.get(current_label, {}),
            )

    if linked_views:
        plotter.link_views()

    return plotter


def facet_grid_animate(
    filename: str,
    data: list[dict[str, np.ndarray]],
    plotfuncs: dict[str, Callable],
    show_cmap: bool = False,
    cmap: str | np.ndarray | pv.LookupTable = "viridis",
    portrait: bool = False,
    linked_views: bool = True,
    titles: list[dict[str, str]] = [],
    plotter_kwargs: dict = {},
    plotfuncs_kwargs: dict[str, dict[str, Any]] = {},
) -> str:
    """
    Create a 3D animation from a series of data snapshots using PyVista.
    This series must be a list of dictionaries with the data for each frame keyed by a label used to title the panel it will be plotted into. The final plot will have as many subplots as there are labels in the data dictionaries. The keys for plotfuncs and data must be the same.
    Parameters:
    -----------
    filename : str
        The name of the output movie file.
    data : list[dict[str, np.ndarray]]
        A list of dictionaries containing the data for each timestep.
    plotfuncs : dict[str, Callable]
        A dictionary of plotting functions keyed by data label. The keys for plotfuncs and data must be the same.
    show_cmap : bool, optional
        Whether to show the color map (default is False).
    cmap : str | np.ndarray | pv.LookupTable, optional
        The colormap to use (default is "viridis").
    portrait : bool, optional
        Whether to use portrait layout (default is False).
    linked_views : bool, optional
        Whether to link the views of the subplots (default is True).
    titles : list[dict[str, str]], optional
        A list of dictionaries containing titles for each subplot (default is an empty list).
    plotter_kwargs : dict, optional
        Additional keyword arguments to pass to the PyVista Plotter (default is an empty dictionary).
    plotfuncs_kwargs : dict[str, dict[str, Any]], optional
        Additional keyword arguments to pass to each plotting function (default is an empty dictionary).
    Returns:
    --------
    str
        The filename of the created movie.
    """

    if len(titles) > 0 and len(titles) != len(data):
        raise ValueError(
            "The number of titles must be the same as the number of data dictionaries."
        )

    if data[0].keys() != plotfuncs.keys():
        raise ValueError(
            "The keys for the data and plotfuncs dictionaries must be the same."
        )

    def create_frame(
        data_dict: dict[str, np.ndarray], title: dict[str:str], layout=(1, 1)
    ):
        label = iter(data_dict.keys())
        for i in range(layout[0]):
            for j in range(layout[1]):
                current_label = next(label)
                plotfuncs[current_label](
                    title.get(current_label, current_label),
                    data_dict[current_label],
                    plotter,
                    panel=(i, j),
                    show_cmap=show_cmap,
                    cmap=cmap,
                    **plotfuncs_kwargs.get(current_label, {}),
                )

        plotter.write_frame()

    layout = find_layout(len(plotfuncs), portrait=portrait)

    plotter = pv.Plotter(shape=layout, **plotter_kwargs)

    plotter.open_movie(filename)
    create_frame(data[0], titles[0] if len(titles) > 0 else {}, layout)

    if linked_views:
        plotter.link_views()

    for i, single_timestep_data in enumerate(data[1::]):
        create_frame(
            single_timestep_data, titles[i] if len(titles) > 0 else {}, layout=layout
        )

    plotter.close()

    return filename

import pyvista as pv
import numpy as np
from typing import Callable


def rgb_to_scalar(img: np.ndarray) -> np.ndarray:
    """Convert an array of RGB values to scalar values.
        This function is necessary because pyvista does not support RGB values directly as mesh data

    Args:
        img (np.ndarray): data to be converted, of shape (n, m, 3)

    Returns:
        np.ndarray: data converted to scalar values, of shape (n, m)
    """
    reshaped = img.reshape(-1, 3, copy=True)
    unique_rgb, ridx = np.unique(reshaped, axis=0, return_inverse=True)

    values = np.arange(len(unique_rgb))
    return values[ridx].reshape(img.shape[:-1])


def _find_good_shape(num_plots: int, portrait: bool = False) -> tuple[int, int]:
    """Find a good shape (rows, columns) for a grid of plots which should be such that
        rows*columns >= num_plots and rows is as close to columns as possible and rows*columns is minimal.
        There are sophisticated ways to do this, which are way beyond what is needed here, so a simple heuristic based on sqrt(num_plots) is used.
    Args:
        num_plots (int): number of plots to distribute
        portrait (bool, optional): whether the plots should be in portrait mode. If yes, the rows will become the larger number and cols the smaller, otherwise, it will be the other way round. Defaults to False

    Returns:
        tuple[int, int]: shape of the grid (rows, columns)
    """
    root = np.sqrt(num_plots)
    root_int = np.rint(root)

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
    best_guess = guesses[np.argmin([x * y for x, y in guesses])]

    if np.isclose(root, root_int):
        return int(root_int), int(root_int)
    elif best_guess[0] * best_guess[1] >= num_plots:
        return (
            (np.min(best_guess), np.max(best_guess))
            if not portrait
            else (np.max(best_guess), np.min(best_guess))
        )
    else:
        return b, b


def _animate_with_func3D_facet(
    filename: str,
    data: dict[str, list[np.ndarray]],
    titles: dict[str, list[str]],
    plotfuncs: dict[str, Callable],
    show_cmap: bool = False,
    cmap: str | np.ndarray = "viridis",
    portrait: bool = False,
    linked_views: bool = True,
    with_titles: bool = True,
    plotter_kwargs: dict = {},
) -> str:
    """Animate a set of data with a corresponding set of plot functions.
    The The list of data for each plot must be of equal length. Each element
    of these lists will be one frame in the animation.
    The result is stored as an .mp4 file.
    Args:
        filename (str): A filename or path to store the animation. This function automatically adds an .mp4 extension to the given filename.
        data (dict[str, list[np.ndarray]]): The data to animate in each suplot. The keys are the labels for the plots and the values are lists of data to plot. Each element of these lists will be one frame in the animation. The lists must be of equal length.
        titles (dict[str, list[str]]): The titles for each subplot. The keys are the labels for the plots and the values are lists of titles to display. Each element of these lists will be the title for the corresponding frame in the animation. Useful to have timestep labels for instance.
        plotfuncs (dict[str, Callable]): Function to write the data into a frame in the plotter. Signature: func(data, plotter, show_cmap:bool = show_cmap, cmap=cmap)
        show_cmap (bool, optional): whether to show the colormap in each plot or not. Defaults to False.
        cmap (str | np.ndarray, optional): matplotlib colormap name. Defaults to "viridis".
        portrait (bool, optional): Aspect ratio mode. if this is true, the larger dimension of the plot table will be the rows. Otherwise, the larger dimension will be the columns. Defaults to False.
        linked_views (bool, optional): Whether to link all the views together for interactive plotting. If true, they will move in unison if one is moved. Defaults to True.
        with_titles (bool, optional): If the labels should be used as titles. Defaults to True.
        plotter_kwargs (dict, optional): Other keyword arguments passed to the plotter constructor. Defaults to {}.

    Returns:
        str: path to where the given animation is stored.
    """

    def create_frame(label):
        for i in range(layout[0]):
            for j in range(layout[1]):
                plotter.subplot(i, j)

                current_label = next(label)

                if with_titles:
                    plotter.add_text(titles[current_label][0])

                plotfuncs[current_label](
                    data[current_label][0], plotter, show_cmap=show_cmap, cmap=cmap
                )

                plotter.write_frame()

    layout = _find_good_shape(len(data), portrait=portrait)

    plotter = pv.Plotter(shape=layout, **plotter_kwargs)

    plotter.open_movie(filename)

    label = iter(data.keys())

    create_frame(label)

    if linked_views:
        plotter.link_views()

    current_label = next(iter(data.keys()))

    for i in range(1, len(data[current_label])):
        label = iter(data.keys())
        create_frame(label)

    plotter.close()

    return filename


def _plot3Dfacet(
    data: dict[str, np.ndarray],
    plotfuncs: dict[str, Callable],
    show_cmap: bool = False,
    cmap: str | np.ndarray = "viridis",
    portrait: bool = False,
    linked_views: bool = True,
    with_titles: bool = True,
    plotter_kwargs: dict = {},
) -> pv.Plotter:
    """Plot a set of data with a corresponding set of plot functions into a grid of subplots, similar to seaborn facetplots.

    Args:
        data (dict[str, np.ndarray]): Dictionary of data to plot. The keys are the labels for the plots and the values are the data to plot.
        plotfuncs (dict[str, Callable]): Functions that take the data and a pyvista plotter object and plot the data into the each subplot.
        show_cmap (bool, optional): whether to show the colormap in each plot or not. Defaults to False.
        cmap (str | np.ndarray, optional): matplotlib colormap name. Defaults to "viridis".
        portrait (bool, optional): Aspect ratio mode. if this is true, the larger dimension of the plot table will be the rows. Otherwise, the larger dimension will be the columns. Defaults to False.
        linked_views (bool, optional): Whether to link all the views together for interactive plotting. If true, they will move in unison if one is moved. Defaults to True.
        with_titles (bool, optional): If the labels should be used as titles. Defaults to True.
        plotter_kwargs (dict, optional): Other keyword arguments passed to the plotter constructor. Defaults to {}.

    Returns:
        pv.Plotter: pyvista plotter object. Call plotter.show() to display the plot.
    """

    layout = _find_good_shape(len(data), portrait=portrait)

    plotter = pv.Plotter(shape=layout, **plotter_kwargs)

    label = iter(data.keys())

    for i in range(layout[0]):
        for j in range(layout[1]):
            plotter.subplot(i, j)
            current_label = next(label)
            if with_titles:
                plotter.add_text(current_label)
            plotfuncs[current_label](
                data[current_label], plotter, show_cmap=show_cmap, cmap=cmap
            )

    if linked_views:
        plotter.link_views()

    return plotter

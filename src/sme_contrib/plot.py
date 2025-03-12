"""Plotting"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lscmap
from matplotlib import animation
import pyvista as pv
from typing import Any, Callable, Union

from .pyvista_utils import (
    find_layout,
)


def colormap(color, name="my colormap"):
    """
    Create a linear matplotlib colormap

    The minimum value corresponds to the color black,
    and the maximum value corresponds to the supplied ``color``.

    This color can be supplied as a triplet of floats in the range from zero
    to one, or as a hex RGB string ``"#rgb`` or ``"#rrggbb``.

    So for example, three equivalent ways to set the ``color`` to red would be
    ``(1.0, 0.0, 0.0)``, ``#f00``, or ``"#ff0000"``.

    Args:
        color: RBG triplet of floats between 0 and 1, or hex RGB string

    Returns:
        matplotlib.Colormap: the Colormap
    """
    return lscmap.from_list(name, [(0, 0, 0), color], 256)


def concentration_heatmap(
    simulation_result, species, z_slice: int = 0, title=None, ax=None, cmap=None
):
    """
    Plot 2d heatmap of species concentration

    Plots the concentration of species in the list ``species``
    from the supplied ``simulation_result`` as a 2d heatmap.

    Args:
        simulation_result (sme.SimulationResult): A simulation result to plot
        species (List of str): The species to plot
        z_slice (int): The z-slice to plot
        title (str): Optionally specify the title
        ax(matplotlib.axes._subplots.AxesSubplot): Optionally specify the axes to draw the plot on
        cmap(matplotlib.Colormap): Optionally specify the colormap to use

    Returns:
        matplotlib.axes._subplots.AxesSubplot: The axes the plot was drawn on
        matplotlib.image.AxesImage: The axes of the image
    """
    if ax is None:
        ax = plt.gca()
    if title is None:
        s = ", ".join(species)
        title = f"Concentration of {s} at time {simulation_result.time_point}"
    c = simulation_result.species_concentration[species[0]][z_slice, :]
    for i in range(1, len(species)):
        c = np.add(c, simulation_result.species_concentration[species[i]][z_slice, :])
    ax.set_title(title)
    im = ax.imshow(c, cmap=cmap)
    return ax, im


def concentration_heatmap_animation(
    simulation_results,
    species,
    z_slice: int = 0,
    title=None,
    figsize=None,
    interval=200,
):
    """
    Plot 2d animated heatmap of species concentration

    Plots the concentration of species in the list ``species``
    from the supplied list ``simulation_results`` as an animated 2d heatmap.

    Args:
        simulation_results (List of sme.SimulationResult): A simulation result to plot
        species (List of str): The species to plot
        z_slice (int): The z-slice to plot
        title (str): Optionally specify the title
        figsize ((float, float)): Optionally specify the figure size
        interval: Optionally specify the interval in ms between images

    Returns:
        matplotlib.animation.ArtistAnimation: the matplotlib animation
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    if title is None:
        s = ", ".join(species)
        title = f"Concentration of {s}"
    artists = []
    for simulation_result in simulation_results:
        c = simulation_result.species_concentration[species[0]][z_slice, :]
        for i in range(1, len(species)):
            c = np.add(
                c,
                np.array(
                    simulation_result.species_concentration[species[i]][z_slice, :]
                ),
            )
        artists.append(
            [
                ax.imshow(c, animated=True, interpolation=None),
                ax.text(
                    0.5,
                    1.01,
                    f"{title}: t = {simulation_result.time_point}",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    transform=ax.transAxes,
                ),
            ]
        )
    anim = animation.ArtistAnimation(
        fig, artists, interval=interval, blit=True, repeat=False
    )
    plt.close()
    return anim


def facet_grid_3D(
    data: dict[str, np.ndarray],
    plotfuncs: dict[str, Callable],
    show_cmap: bool = False,
    cmap: Union[str, np.ndarray, pv.LookupTable] = "viridis",
    portrait: bool = False,
    linked_views: bool = True,
    plotter_kwargs: dict = {},
    plotfuncs_kwargs: dict[str, dict[str, Any]] = {},
) -> pv.Plotter:
    """
    Create a 3D facet plot using PyVista.

    This follows the seaborn.FacetGrid concept. This function creates a grid of subplots where each subplot is filled by a function in the plotfuncs argument. The keys for plotfuncs and data must be the same, such that plotfuncs can be unambiguously mapped over the data dictionary. Do not attempt to plot 2D images and 3D images into the same facet grid, as this will create odd artifacts and may not work as expected.

    Args:
        data : (dict[str, np.ndarray]) A dictionary where keys are labels and values are numpy arrays containing the data to be plotted.
        plotfuncs : (dict[str, Callable]) A dictionary where keys are labels and values are functions with signature ``f(label:str, data:np.ndarray | pyvista.ImageData | pyvista.UniformGrid, plotter:pv.Plotter, panel:tuple[int, int], show_cmap:bool=show_cmap, cmap=cmap, **plotfuncs_kwargs )`` -> None
        show_cmap : bool, optional Whether to show the color map. Default is False.
        cmap : (str | np.ndarray | pv.LookupTable), optional The color map to use. Default is "viridis".
        portrait : (bool), optional Whether to use a portrait layout. Default is False.
        linked_views : (bool), optional Whether to link the views of the subplots. Default is True.
        plotter_kwargs : (dict, optional) Additional keyword arguments to pass to the PyVista Plotter.
        plotfuncs_kwargs : (dict[str, dict[str, Any]]), optional Additional keyword arguments to pass to each plotting function.

    Returns:
        pv.Plotter The PyVista Plotter object with the created facet plot.
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


def facet_grid_animate_3D(
    filename: str,
    data: list[dict[str, np.ndarray]],
    plotfuncs: dict[str, Callable],
    show_cmap: bool = False,
    cmap: Union[str, np.ndarray, pv.LookupTable] = "viridis",
    portrait: bool = False,
    linked_views: bool = True,
    titles: list[dict[str, str]] = [],
    plotter_kwargs: dict = {},
    plotfuncs_kwargs: dict[str, dict[str, Any]] = {},
) -> str:
    """
    Create a 3D animation from a series of data snapshots using PyVista.

    This series must be a list of dictionaries with the data for each frame keyed by a label used to title the panel it will be plotted into. The final plot will have as many subplots as there are labels in the data dictionaries. The keys for plotfuncs and data must be the same.

    Args:
        filename : (str) The name of the output movie file.
        data : (list[dict[str, np.ndarray]]) A list of dictionaries containing the data for each timestep.
        plotfuncs : (dict[str, Callable]) A dictionary of plotting functions keyed by data label. The keys for plotfuncs and data must be the same.
        show_cmap : (bool), optional Whether to show the color map (default is False).
        cmap : (str | np.ndarray | pv.LookupTable, optional) The colormap to use (default is "viridis").
        portrait : (bool), optional Whether to use portrait layout (default is False).
        linked_views : (bool), optional Whether to link the views of the subplots (default is True).
        titles : (list[dict[str, str]]), optional A list of dictionaries containing titles for each subplot (default is an empty list).
        plotter_kwargs : (dict), optional Additional keyword arguments to pass to the PyVista Plotter (default is an empty dictionary).
        plotfuncs_kwargs : (dict[str, dict[str, Any]]), optional Additional keyword arguments to pass to each plotting function (default is an empty dictionary).

    Returns:
        str The filename of the created movie.
    """

    if len(titles) > 0 and len(titles) != len(data):
        raise ValueError(
            "The number of titles must be the same as the number of data dictionaries."
        )

    if data[0].keys() != plotfuncs.keys():
        raise ValueError(
            "The keys for the data and plotfuncs dictionaries must be the same."
        )

    # main function, called for each frame in the movie
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

    # preparations
    layout = find_layout(len(plotfuncs), portrait=portrait)

    plotter = pv.Plotter(shape=layout, **plotter_kwargs)

    plotter.open_movie(filename)

    # add first frame here to set up the plotter
    create_frame(data[0], titles[0] if len(titles) > 0 else {}, layout)

    if linked_views:
        plotter.link_views()

    for i, single_timestep_data in enumerate(data[1::]):
        create_frame(
            single_timestep_data, titles[i] if len(titles) > 0 else {}, layout=layout
        )

    plotter.close()

    return filename

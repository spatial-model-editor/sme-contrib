"""Plotting"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lscmap
from matplotlib import animation
import pyvista as pv
from typing import Any, Callable, Union
import sme
from pathlib import Path

from .pyvista_utils import (
    find_layout,
    make_discrete_colormap,
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
    plotter_kwargs: Union[dict, None] = None,
    plotfuncs_kwargs: Union[dict[str, dict[str, Any]], None] = None,
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

    plotter = pv.Plotter(
        shape=layout, **(plotter_kwargs if plotter_kwargs is not None else {})
    )

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
                **(
                    plotfuncs_kwargs.get(current_label, {})
                    if plotfuncs_kwargs is not None
                    else {}
                ),
            )

    if linked_views:
        plotter.link_views()

    return plotter


def facet_grid_animate_3D(
    filename: Union[str, Path],
    data: list[dict[str, np.ndarray]],
    plotfuncs: dict[str, Callable],
    show_cmap: bool = False,
    cmap: Union[str, np.ndarray, pv.LookupTable] = "viridis",
    portrait: bool = False,
    linked_views: bool = True,
    titles: Union[list[dict[str, str]], None] = None,
    plotter_kwargs: Union[dict, None] = None,
    plotfuncs_kwargs: Union[dict[str, dict[str, Any]], None] = None,
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
    if titles is None:
        titles = []

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
                    **plotfuncs_kwargs.get(current_label, {})
                    if plotfuncs_kwargs is not None
                    else {},
                )

        plotter.write_frame()

    # preparations
    layout = find_layout(len(plotfuncs), portrait=portrait)

    plotter = pv.Plotter(
        shape=layout, **plotter_kwargs if plotter_kwargs is not None else {}
    )

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


def concentrations3D(
    simulation_result: sme.SimulationResult,
    species: list[str],
    cmap: str | np.ndarray | pv.LookupTable = "viridis",
    show_cmap: bool = False,
    plotter_kwargs: dict[str, Any] = None,
    plotfunc_kwargs: dict[str, Any] = None,
) -> pv.Plotter:
    """Plot a 3D facet grid of species concentrations.
    This function creates a 3D facet grid of species concentrations. Each panel will be a 3D plot of the concentration of a single species.
    This function is a wrapper around the facet_grid_3D function.

    Args:
        simulation_result (sme.SimulationResult): a single simulation result object, i.e., a single recorded frame of the simulations
        species (list[str]): A list of species strings
        cmap (str | np.ndarray | pv.LookupTable, optional): Name of a matplotlib colorbar. Defaults to "viridis".
        show_cmap (bool, optional): Whether or not to show the colorbar on the plot. Defaults to False.
        plotter_kwargs (dict[str, Any], optional): Additional keyword arguments for the used pyVista.Plotter. Defaults to None.
        plotfunc_kwargs (dict[str, Any], optional): Additional keyword arguments passed to plotter.add_mesh. Defaults to None.
    Raises:
        ValueError: if the data is not 3D
        ValueError: if a given species is not found in the simulation result

    Returns:
        pv.Plotter: pyvista.Plotter object the data has been plotted into
    """
    # turn the simulation result into numpy ndarray
    datadict = {}
    for s in species:
        if s not in simulation_result.species_concentration:
            raise ValueError(f"Species {s} not found in simulation result.")
        data = simulation_result.species_concentration[s]
        if data.ndim != 3:
            raise ValueError("Data must be 3D.")
        else:
            datadict[s] = data

    # create a plot function
    def plotfunc(
        label: str,
        data: np.ndarray,
        plotter: pv.Plotter,
        panel: tuple[int, int],
        show_cmap: bool,
        cmap: Union[str, np.ndarray, pv.LookupTable],
        **kwargs: dict[str, Any],
    ):
        # create a pyvista grid

        plotter.subplot(*panel)
        plotter.title = label
        plotter.add_mesh(
            data,
            scalars=data,
            label=label,
            cmap=cmap,
            show_scalar_bar=show_cmap,
            **kwargs,
        )

    # use facetGrid3D to plot it
    return facet_grid_3D(
        data=datadict,
        plotfuncs={species[i]: plotfunc for i in range(len(species))},
        show_cmap=show_cmap,
        cmap=cmap,
        portrait=False,
        linked_views=True,
        plotter_kwargs=plotter_kwargs,
        plotfuncs_kwargs=plotfunc_kwargs,
    )


def concentrationsAnimate3D(
    filename: Union[str, Path],
    simulation_results: sme.SimulationResultList,
    species: list[str],
    show_cmap: bool = False,
    cmap: Union[str, np.ndarray, pv.LookupTable] = "viridis",
    portrait: bool = False,
    titles: Union[list[dict[str, str]], None] = None,
    linked_views: bool = True,
    plotter_kwargs: dict[str, Any] = None,
    plotfunc_kwargs: dict[str, Any] = None,
) -> Union[str, Path]:
    """Animate a list of frames from a simulation result list.
    This function creates a 3D animation of the species concentrations over time. Each frame will be a 3D plot of the concentration of a single species.
    This function is a wrapper around the facet_grid_animate_3D function.
    The animation will be saved to the specified filename.

    Args:
        filename (Union[str, Path]): filename to save the animation to. Uses mp4 format.
        simulation_results (sme.SimulationResultList): a list of `SimulationResult` objects, i.e., a list of recorded frames of the simulations
        species (list[str]): list of species to plot
        show_cmap (bool, optional): Whether to show the colorbar on theplots or not. Defaults to False.
        cmap (Union[str, np.ndarray, pv.LookupTable], optional): name of matplotlib colormap or custom colormap that maps scalar values to rbp. Defaults to "viridis".
        portrait (bool, optional): Whether to use the smaller or larger number of plots as rows. Defaults to False.
        titles (Union[list[dict[str, str]], None], optional): Titles of the different plots if not just the species name is desired. Defaults to None.
        linked_views (bool, optional): link the view cameras. Defaults to True.
        plotter_kwargs (dict[str, Any], optional): Additional keyword arguments for the used pyVista.Plotter. Defaults to None. See [here](https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter) for more information
        plotfunc_kwargs (dict[str, Any], optional): Additional keyword arguments passed to plotter.add_mesh. Defaults to None. See [here](https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_mesh#) for more information.

    Returns:
        Union[str, Path]: path to the saved animation .mp4 file
    """

    def plotfunc(
        label: str,
        data: np.ndarray,
        plotter: pv.Plotter,
        panel: tuple[int, int],
        show_cmap: bool,
        cmap: Union[str, np.ndarray, pv.LookupTable],
        **kwargs: dict[str, Any],
    ):
        # create a pyvista grid
        plotter.subplot(*panel)
        plotter.title = label
        plotter.add_mesh(
            data,
            scalars=data,
            label=label,
            cmap=cmap,
            show_scalar_bar=show_cmap,
            **kwargs,
        )

    return facet_grid_animate_3D(
        filename,
        data=[
            {
                species[i]: res.species_concentration[species[i]]
                for i in range(len(species))
            }
            for res in simulation_results
        ],
        plotfuncs={species[i]: plotfunc for i in range(len(species))},
        show_cmap=show_cmap,
        cmap=cmap,
        portrait=portrait,
        linked_views=linked_views,
        titles=titles,
        plotter_kwargs=plotter_kwargs,
        plotfuncs_kwargs=plotfunc_kwargs,
    )

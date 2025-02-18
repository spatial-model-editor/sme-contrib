"""Plotting"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lscmap
from matplotlib import animation
from itertools import cycle
import matplotlib.colors as mcolors
import pyvista as pv
from typing import Any
import sme

from .pyvista_utils import (
    facet_animate3D,
    facet_plot3D,
    rgb_to_scalar,
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


def make_circular_colormap(
    cmap: str = "tab10", values: np.ndarray = np.array([])
) -> list[tuple]:
    """Create a discrete colormap of potentially repeating colors of the same size as the `values` array.

    Args:
        cmap (str, optional): matplotlib colormap name. Defaults to "tab10".
        values (np.array, optional): values to be mapped to colors. Defaults to [].

    Returns:
        list[tuple]: list of color in rgba format.
    """
    cm = [(0.0, 0.0, 0.0, 1.0)]
    i = 0
    for c in cycle(plt.get_cmap(cmap).colors):
        cm.append(mcolors.to_rgba(c))
        if len(cm) >= len(values):
            break
        i += 1
    return cm


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


def plot_concentration_image_3D(
    simulation_result: sme.SimulationResult,
    show_cmap: bool = False,
    cmap: str | np.ndarray | pv.LookupTable = "viridis",
    plotter_kwargs: dict[str, Any] = {"border": False, "notebook": False},
) -> pv.Plotter:
    """Plot a single concentration image in 3D using pyvista.

    Args:
        simulation_result (sme.SimulationResult): The simulation result object of a 3D simulation for a single timestep
        show_cmap (bool, optional): Whether the colormap should be shown or not. Defaults to False.
        cmap (str | np.ndarray | pv.LookupTable, optional): colormap to use. Either a string naming a matplotlib colormap, a numpy array of values or a pyvista lookup table mapping values ot rgba colors.  Defaults to "viridis".
        plotter_kwargs (dict[str, Any], optional): Addtitional kwargs for the pyvista.Plotter constructor Defaults to {"border": False, "notebook": False}.

    Returns:
        pv.Plotter: pyvista Plotter object
    """

    def plot_single(
        title,
        data,
        plotter,
        panel,
        show_cmap=show_cmap,
        cmap=cmap,
    ):
        _data = rgb_to_scalar(data)

        plotter.subplot(*panel)

        if title:
            plotter.add_text(title)

        img_data = pv.ImageData(
            dimensions=_data.shape,
        )
        img_data.point_data["Data"] = _data.flatten()
        img_data = img_data.points_to_cells(scalars="Data")
        plotter.subplot(0, 0)
        plotter.add_mesh(
            img_data,
            show_edges=True,
            show_scalar_bar=show_cmap,
            cmap=cmap,
        )

    return facet_plot3D(
        data={"concentrations": simulation_result.concentration_image},
        plotfuncs={"concentrations": plot_single},
        show_cmap=show_cmap,
        cmap=cmap,
        portrait=True,
        with_titles=True,
        plotter_kwargs=plotter_kwargs,
    )


def plot_species_concentration_3D(
    simulation_result: sme.SimulationResult,
    species: list[str],
    thresholds: list[float] = [],
    show_cmap: bool = False,
    cmap: str | np.ndarray | pv.LookupTable = "viridis",
    portrait: bool = False,
    linked_views: bool = True,
    with_titles: bool = True,
    plotter_kwargs: dict[str, Any] = {"border": False, "notebook": False},
) -> pv.Plotter:
    """Plot the concentration of a list of species in 3D using pyvista.
       This function creates a 3D plot of the concentration for each species in a separate subplot
    Args:
        simulation_result (sme.SimulationResult): Simulationresult object for a given timestep
        species (list[str]): list of species to plot
        thresholds (list[float], optional): Thresholds to exclude some values for each plot. If empty, it is set to 1e6 to effectively have no threshold. Defaults to [].
        show_cmap (bool, optional): Whether the colormap should be shown or not. Defaults to False.
        cmap (str | np.ndarray | pv.LookupTable, optional): colormap to use. Either a string naming a matplotlib colormap, a numpy array of values or a pyvista lookup table mapping values ot rgba colors.  Defaults to "viridis".
        portrait (bool, optional): Whether to organize plot grid (n,m) in potrait mode (smaller of (n,m) as columns) or landscape mode (smaller number as rows). Defaults to False.
        linked_views (bool, optional): If all the views should be linked togehter such that perspective changes affect all plots the same. Defaults to True.
        with_titles (bool, optional): Have a title for each plot or not. Defaults to True.
        plotter_kwargs (dict[str, Any], optional): Additional kwargs for pyvista.Plotter. Defaults to {"border": False, "notebook": False}.

    Returns:
        pv.Plotter: pyvista plotter object
    """

    def plot_single(
        label: str,
        data: np.ndarray,
        plotter: pv.Plotter,
        panel: tuple[int, int],
        show_cmap,
        cmap=cmap,
        threshold_value=1e6,
    ):
        _data = rgb_to_scalar(data) if len(data.shape) == 4 else data

        plotter.subplot(*panel)

        if with_titles:
            plotter.add_text(label)

        img_data = pv.ImageData(
            dimensions=_data.shape,
        )
        img_data.point_data["Data"] = _data.flatten()
        img_data = img_data.points_to_cells(scalars="Data")
        plotter.subplot(0, 0)
        plotter.add_mesh(
            img_data.threshold(threshold_value),
            show_edges=True,
            show_scalar_bar=show_cmap,
            cmap=cmap,
        )

    return facet_plot3D(
        data={sp: simulation_result.species_concentration[sp] for sp in species},
        plotfuncs={sp: plot_single for sp in species},
        show_cmap=show_cmap,
        cmap=cmap,
        portrait=portrait,
        with_titles=with_titles,
        linked_views=linked_views,
        plotter_kwargs=plotter_kwargs,
        plotfuncs_kwargs={
            species[i]: {"threshold_value": thresholds[i]}
            for i in range(0, len(species))
        }
        if thresholds != []
        else {},
    )


def concentrations_animation_3D(
    filename: str,
    simulation_results: sme.SimulationResultList,
    species: list[str],
    thresholds: list[float] = [],
    show_cmap: bool = False,
    cmap: str | np.ndarray | pv.LookupTable = "viridis",
    portrait: bool = False,
    linked_views: bool = True,
    with_titles: bool = True,
    plotter_kwargs: dict = {"border": False, "notebook": False},
) -> str:
    """Create an .mp4 video of the concentration of a list of species in 3D using pyvista, with one frame being one timestep for each species. In essence, this is an animated version of the plot_species_concentration_3D function.

    Args:
        filename (str): filename to save the video
        simulation_results (sme.SimulationResultList): List of simulation results for each timestep
        species (list[str]): List of species to animate
        thresholds (list[float], optional): Thresholds to limit the plotted values for each species Values larger than the threshold will be cut. Defaults to [].
        show_cmap (bool, optional): Whether the colormap should be shown or not. Defaults to False.
        cmap (str | np.ndarray | pv.LookupTable, optional): colormap to use. Either a string naming a matplotlib colormap, a numpy array of values or a pyvista lookup table mapping values ot rgba colors.  Defaults to "viridis".
        portrait (bool, optional): Whether to organize plot grid (n,m) in potrait mode (smaller of (n,m) as columns) or landscape mode (smaller number as rows). Defaults to False.
        linked_views (bool, optional): If all the views should be linked togehter such that perspective changes affect all plots the same. Defaults to True.
        with_titles (bool, optional): Have a title for each plot or not. Defaults to True.
        plotter_kwargs (dict[str, Any], optional): Additional kwargs for pyvista.Plotter. Defaults to {"border": False, "notebook": False}.

    Returns:
        str: filename of the saved video
    """

    def plot_single(
        label: str,
        data: np.ndarray,
        plotter: pv.Plotter,
        panel: tuple[int, int],
        show_cmap,
        cmap=cmap,
        threshold_value=1e6,
    ):
        _data = rgb_to_scalar(data) if len(data.shape) == 4 else data

        img_data = pv.ImageData(
            dimensions=_data.shape,
        )
        img_data.point_data["Data"] = _data.flatten()
        img_data = img_data.points_to_cells(scalars="Data")

        plotter.subplot(*panel)
        if with_titles:
            plotter.add_text(label, name=label + str(panel))

        actor = plotter.add_mesh(
            img_data.threshold(threshold_value),
            show_edges=True,
            show_scalar_bar=show_cmap,
            cmap=cmap,
            name="mesh" + label + str(panel),
        )
        actor.mapper.scalar_range = (np.min(_data), np.max(_data))

    return facet_animate3D(
        filename=filename,
        data=[s.species_concentration for s in simulation_results],
        titles=[
            {sp: f"Concentration of {sp} at t={s.time_point}" for sp in species}
            for s in simulation_results
        ],
        plotfuncs={sp: plot_single for sp in species},
        show_cmap=show_cmap,
        cmap=cmap,
        portrait=portrait,
        linked_views=linked_views,
        with_titles=with_titles,
        plotter_kwargs=plotter_kwargs,
        plotfuncs_kwargs={
            species[i]: {"threshold_value": thresholds[i]}
            for i in range(0, len(species))
        }
        if len(thresholds) > 0
        else {},
    )

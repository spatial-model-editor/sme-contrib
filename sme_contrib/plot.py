"""Plotting"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lscmap
from matplotlib import animation


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


def concentration_heatmap(simulation_result, species, title=None, ax=None, cmap=None):
    """
    Plot 2d heatmap of species concentration

    Plots the concentration of species in the list ``species``
    from the supplied ``simulation_result`` as a 2d heatmap.

    Args:
        simulation_result (sme.SimulationResult): A simulation result to plot
        species (List of str): The species to plot
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
    c = simulation_result.species_concentration[species[0]]
    for i in range(1, len(species)):
        c = np.add(c, simulation_result.species_concentration[species[i]])
    ax.set_title(title)
    im = ax.imshow(c, cmap=cmap)
    return ax, im


def concentration_heatmap_animation(
    simulation_results, species, title=None, figsize=None, interval=200
):
    """
    Plot 2d animated heatmap of species concentration

    Plots the concentration of species in the list ``species``
    from the supplied list ``simulation_results`` as an animated 2d heatmap.

    Args:
        simulation_results (List of sme.SimulationResult): A simulation result to plot
        species (List of str): The species to plot
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
        c = simulation_result.species_concentration[species[0]]
        for i in range(1, len(species)):
            c = np.add(c, np.array(simulation_result.species_concentration[species[i]]))
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

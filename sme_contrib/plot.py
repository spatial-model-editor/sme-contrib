"""Plotting"""

import numpy as np
from matplotlib import pyplot as plt
import sme


def concentration_heatmap(simulation_result, species, title=None, ax=None):
    """Plot 2d heatmap of species concentration

    Plots the concentration of species in the list ``species``
    from the supplied ``simulation_result`` as a 2d heatmap.

    Args:
        simulation_result (sme.SimulationResult): A simulation result to plot
        species (List of str): The species to plot
        title (str): Optionally specify the title
        ax(matplotlib.axes._subplots.AxesSubplot): Optionally specify the axes to draw the plot on

    Returns:
        matplotlib.axes._subplots.AxesSubplot: The axes the plot was drawn on
    """
    if ax is None:
        ax = plt.gca()
    if title is None:
        s = ",".join(species)
        title = f"Concentration of {s} at time {simulation_result.time_point}"
    c = np.array(simulation_result.species_concentration[species[0]])
    for i in range(1, len(species)):
        c = np.add(c, np.array(simulation_result.species_concentration[species[i]]))
    ax.imshow(c)
    ax.set_title(title)
    return ax

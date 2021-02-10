"""Optimization, fitting and analysis"""

import numpy as np
import multiprocessing
import pyswarms as ps
from functools import partial
from os import cpu_count
from PIL import Image
from matplotlib import pyplot as plt
import sme


def _hessian_f(f, x0, i, j, rel_eps):
    x = np.array(x0, dtype=np.float64)
    if i is None:
        return f(x)
    if j is None:
        x[i] = (1.0 + rel_eps) * x[i]
        return f(x)
    x[i] = (1.0 + rel_eps) * x[i]
    x[j] = (1.0 + rel_eps) * x[j]
    return f(x)


def hessian(f, x0, rel_eps=1e-2, processes=None):
    """Approximate Hessian of function ``f`` at point ``x0``

    Uses a `finite difference`_ approximation where the step size used
    for each element ``i`` of ``x0`` is ``rel_eps * x[i]``.

    Requires :math:`N^2 + N + 1` evalulations of ``f``, where :math:`N`
    is the number of elements of ``x0``

    The evaluations of ``f`` are done in parallel, so ``f`` must be a
    thread-safe function that can safely be called from multiple threads
    at the same time.

    Note:
        This choice of step size allows the different elements of x0 to have
        vastly different scales without causing numerical instabilities,
        but it will fail if an element of ``x0`` is equal to 0.

    Args:
        f: The function to evaluate, it should be callable as f(x0) and return a scalar
        x0: The point at which to evaluate the function, a flot or list of floats.
        rel_eps: The relative step size to use
        processes: The number of processes to use (the default ``None`` means use all available cpu cores)

    Returns:
        np.array: The Hessian as a 2d numpy array of floats

    .. _finite difference:
        https://en.wikipedia.org/wiki/Finite_difference#Multivariate_finite_differences

    """
    if processes == None:
        processes = cpu_count()
    n = len(x0)
    # make list of arguments for each f call required
    args = []
    # f(x)
    args.append((f, x0, None, None, rel_eps))
    # f([.., x_i+dx_i, ..])
    for i in range(n):
        args.append((f, x0, i, None, +rel_eps))
    # f([.., x_i-dx_i, ..])
    for i in range(n):
        args.append((f, x0, i, None, -rel_eps))
    # f([.., x_j+dx_j, .., x_i+dx_i, ..])
    for i in range(n):
        for j in range(0, i):
            args.append((f, x0, i, j, +rel_eps))
    # f([.., x_j-dx_j, .., x_i-dx_i, ..])
    for i in range(n):
        for j in range(0, i):
            args.append((f, x0, i, j, -rel_eps))
    # call f with each set of args
    pool = multiprocessing.Pool(processes)
    ff = pool.starmap(_hessian_f, args)
    pool.close()
    pool.join()
    # construct hessian elements from these values
    h = np.zeros((n, n))
    # diagonal elements
    for i in range(n):
        h[(i, i)] = ff[i + 1] - ff[0] + ff[n + i + 1] - ff[0]
        h[(i, i)] /= rel_eps * rel_eps * x0[i] * x0[i]
    # off-diagonal elements
    offset_ij = 2 * n + 1
    n_ij = (n * (n - 1)) // 2
    index_ij = 0
    for i in range(n):
        for j in range(0, i):
            h[(i, j)] = (
                ff[0]
                - ff[1 + i]
                + ff[0]
                - ff[1 + j]
                + ff[offset_ij + index_ij]
                - ff[1 + n + i]
                + ff[offset_ij + n_ij + index_ij]
                - ff[1 + n + j]
            )
            h[(i, j)] /= 2.0 * rel_eps * rel_eps * x0[i] * x0[j]
            h[(j, i)] = h[(i, j)]
            index_ij += 1
    return h


# calculate function f for each x in xs
def _minimize_f(xs, f, processes):
    pool = multiprocessing.Pool(processes=processes)
    norms = pool.map(f, xs)
    pool.close()
    pool.join()
    return np.array(norms)


# minimize objective function using particle swarm
def minimize(
    f,
    lowerbounds,
    upperbounds,
    particles=20,
    iterations=20,
    processes=None,
    ps_options=None,
):
    """Minimize function ``f`` using particle swarm

    The function ``f`` should take an array or list of parameters ``x``, and return a
    value: parameters will be found using particle swarm that minimize this value.

    Each parameter should have a specified lower and upper bound.

    The evaluations of ``f`` are done in parallel, so ``f`` must be a
    thread-safe function that can safely be called from multiple threads
    at the same time. The evaluations are parallelized over the particles
    for each iteration, so for good performance the number of particles should
    be larger than the number of processes.

    Args:
        f: The function to evaluate, it should be callable as f(x) and return a scalar
        lowerbounds: The lower bound for each element of x.
        upperbounds: The upper bound for each element of x.
        particles: The number of particles to use in the swarm
        iterations: The number of iterations to do
        processes: The number of processes to use (the default ``None`` means use all available cpu cores)
        ps_options: A map of the particle swarm hyper parameters to use

    Returns:
        ps_cost: The lowest cost
        ps_res: The parameters that gave this lowest cost
        optimizer: The PySwarms optimizer object

    .. _finite difference:
        https://en.wikipedia.org/wiki/Finite_difference#Multivariate_finite_differences

    """
    if processes == None:
        processes = cpu_count()
    if ps_options == None:
        ps_options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    optimizer = ps.single.GlobalBestPSO(
        particles,
        dimensions=len(lowerbounds),
        options=ps_options,
        bounds=(np.array(lowerbounds), np.array(upperbounds)),
    )
    ps_cost, ps_res = optimizer.optimize(
        _minimize_f, iters=iterations, f=f, processes=processes
    )
    return ps_cost, ps_res, optimizer


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


# plot 2d array as heat-map image
def _ss_plot_image(conc, title, ax=None, cmap=None):
    if ax is None:
        ax = plt.gca()
    ax.imshow(conc, cmap=cmap)
    ax.set_title(title)
    return ax


# plot 1d array as line
def _ss_plot_line(x, y, title, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_title(title)
    ax.plot(x, y)
    return ax


class SteadyState:
    """Steady state parameter fitting

    Given a model and an image of the target steady state distribution of
    a species (or the sum of multiple species), this class tries to find
    a set of parameters where the simulated model has a steady state solution
    that is as close as possible to the target image.

    Args:
        modelfile(str): The sbml file containing the model
        imagefile(str): The image file containing the target concentration
        species(List of str): The species to compare to the target concentration
        function_to_apply_params: A function that sets the parameters in the model.
            This should be a function with signature ``f(model, params)``, and
            which sets the value of the parameters to be fitted in ``model``
            according to the values in ``params``, which will be a list of floats.
        lower_bounds(List of float): The lower bound for each parameter to be fitted
        upper_bounds(List of float): The upper bound for each parameter to be fitted
        simulation_time(float): The length of time to simulate the model
        steady_state_time(float): The length of time to multiply the final rate of change of concentration.
            The cost function that is minimized is the sum of squares over all pixels
            of the difference between the final concentration and the target concentration, plus the
            sum of squares over all pixels of the difference between ``steady_state_time`` * dc/dt and
            zero. Multiplying the rate of change by a time makes the second term have the same units as
            the first term, and the relative importance of being close to steady state versus close
            to the desired concentration in the fit can be adjusted by altering ``steady_state_time``.
            The larger it is, the closer the results will be to a steady state.

    Attributes:
        params(numpy.array): The best model parameters found
        cost_history(List of float): The history of the best cost at each iteration
        cost_history_pbest(List of float): The history of the mean particle best at each iteration
    """

    def __init__(
        self,
        modelfile,
        imagefile,
        species,
        function_to_apply_params,
        lower_bounds,
        upper_bounds,
        simulation_time=1000,
        steady_state_time=200,
        timeout_seconds=10,
    ):
        self.filename = modelfile
        self.species = species
        self.set_target_image(imagefile)
        self.simulation_time = simulation_time
        self.steady_state_time = steady_state_time
        self.apply_params = function_to_apply_params
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.timeout_seconds = timeout_seconds

    def set_target_image(self, imagefile):
        """Set a new target concentration image

        Most image formats are supported. If the image has multiple
        channels, e.g. RGB or RGBA, it is first converted to grayscale.

        Note:
            This doesn't (yet) check that the image has the same dimensions
            as the model geometry, nor does it mask the pixels that lie
            outside of the model compartments to zero

        Args:
            imagefile(str): The image filename
        """
        # todo: mask image to only pixels in model
        # todo: check image dimensions match model geometry image
        img = Image.open(imagefile)
        if len(img.getbands()) > 1:
            # convert RGB or RGBA image to 8-bit grayscale
            img = img.convert("L")
        self.target_conc = np.asarray(img, dtype=np.float64)
        self.target_conc_max = np.amax(self.target_conc)

    def _get_conc(self, result):
        c = np.array(result.species_concentration[self.species[0]])
        for i in range(1, len(self.species)):
            c = np.add(c, np.array(result.species_concentration[self.species[i]]))
        return c

    def _get_dcdt(self, result):
        dcdt = np.array(result.species_dcdt[self.species[0]])
        for i in range(1, len(self.species)):
            dcdt = np.add(dcdt, np.array(result.species_dcdt[self.species[i]]))
        return dcdt

    def _rescale(self, result):
        c = self._get_conc(result)
        dcdt = self._get_dcdt(result)
        scale_factor = self.target_conc_max / np.amax(c)
        return (scale_factor * c, scale_factor * dcdt)

    def _obj_func(self, params, verbose=False):
        m = sme.open_sbml_file(self.filename)
        self.apply_params(m, params)
        results = m.simulate(
            simulation_time=self.simulation_time,
            image_interval=self.simulation_time,
            timeout_seconds=self.timeout_seconds,
            throw_on_timeout=False,
        )
        if len(results) == 1:
            # simulation fail or timeout
            print(
                f"simulation timeout with timeout {self.timeout_seconds}s, params: {params}"
            )
            return abs_diff(0, self.target_conc)
        c, dcdt = self._rescale(results[-1])
        conc_norm = abs_diff(c, self.target_conc)
        dcdt_norm = abs_diff(self.steady_state_time * dcdt, 0)
        if verbose:
            return (conc_norm, dcdt_norm, c)
        return conc_norm + dcdt_norm

    def find(self, particles=20, iterations=20, processes=None):
        """Find parameters that result in a steady state concentration close to the target image

        Uses particle swarm to minimize the difference between the rescaled concentration
        and the target image, as well as the distance from a steady state solution.

        Args:
            particles(int): The number of particles in the particle swarm
            iterations(int): The number of particle swarm iterations
            processes: The number of processes to use (the default ``None`` means use all available cpu cores)

        Returns:
            List of float: the best parameters found

        Note:
            On Windows, calling this function from a jupyter notebook can result in an error
            message of the form `Can't get attribute 'apply_params' on <module '__main__'`,
            where ``apply_params`` is the function you have defined to apply the parameters to the model.
            This is a known `issue <https://docs.python.org/3/library/multiprocessing.html#using-a-pool-of-workers>`_
            with Python multiprocessing, and a workaround is to define the ``apply_params`` function
            in a separate `.py` file and import it into the notebook.

        """
        cost, params, optimizer = minimize(
            self._obj_func,
            self.lower_bounds,
            self.upper_bounds,
            particles=particles,
            iterations=iterations,
            processes=processes,
        )
        self.cost_history = optimizer.cost_history
        self.cost_history_pbest = optimizer.mean_pbest_history
        self.conc_norm, self.dcdt_norm, self.model_conc = self._obj_func(
            params, verbose=True
        )
        self.params = params
        return params

    def hessian(self, rel_eps=0.1, processes=None):
        return hessian(self._obj_func, self.params, rel_eps, processes)

    def plot_target_concentration(self, ax=None, cmap=None):
        """Plot the target concentration as a 2d heat map

        Args:
            ax(matplotlib.axes._subplots.AxesSubplot): Optionally specify the axes to draw the plot on
            cmap(matplotlib.Colormap): Optionally specify the colormap to use

        Returns:
            matplotlib.axes._subplots.AxesSubplot: The axes the plot was drawn on
        """
        return _ss_plot_image(self.target_conc, "Target Concentration", ax, cmap)

    def plot_model_concentration(self, ax=None, cmap=None):
        """Plot the model concentration as a 2d heat map

        The model concentration is normalized such that the maximum pixel intensity
        matches the maximum pixel intensity of the target concentration image

        Args:
            ax(matplotlib.axes._subplots.AxesSubplot): Optionally specify the axes to draw the plot on
            cmap(matplotlib.Colormap): Optionally specify the colormap to use

        Returns:
            matplotlib.axes._subplots.AxesSubplot: The axes the plot was drawn on
        """
        return _ss_plot_image(self.model_conc, "Model Concentration", ax, cmap)

    def plot_cost_history(self, ax=None):
        """Plot the cost history

        The cost of the best set of parameters at each iteration of particle swarm.

        Args:
            ax(matplotlib.axes._subplots.AxesSubplot): Optionally specify the axes to draw the plot on

        Returns:
            matplotlib.axes._subplots.AxesSubplot: The axes the plot was drawn on
        """
        return _ss_plot_line(
            [*range(len(self.cost_history))], self.cost_history, "Best cost history", ax
        )

    def plot_cost_history_pbest(self, ax=None):
        """Plot the mean particle best cost history

        The mean of the best cost for each particle in the swarm, at each iteration of particle swarm.

        Args:
            ax(matplotlib.axes._subplots.AxesSubplot): Optionally specify the axes to draw the plot on

        Returns:
            matplotlib.axes._subplots.AxesSubplot: The axes the plot was drawn on
        """
        return _ss_plot_line(
            [*range(len(self.cost_history_pbest))],
            self.cost_history_pbest,
            "Mean particle best cost history",
            ax,
        )

    def plot_timeseries(self, simulation_time, image_interval_time, ax=None):
        """Plot a timeseries of the sum of concentrations

        The sum of all species concentrations summed over all pixels,
        as a function of the simulation time. This is a convenience plot
        just to see by eye how close the simulation is to a steady state.

        Args:
            ax(matplotlib.axes._subplots.AxesSubplot): Optionally specify the axes to draw the plot on

        Returns:
            matplotlib.axes._subplots.AxesSubplot: The axes the plot was drawn on
        """
        m = sme.open_sbml_file(self.filename)
        self.apply_params(m, self.params)
        results = m.simulate(
            simulation_time=simulation_time, image_interval=image_interval_time
        )
        concs = []
        times = []
        for result in results:
            concs.append(np.sum(self._get_conc(result)))
            times.append(result.time_point)
        return _ss_plot_line(times, concs, "Concentration time series", ax)

    def plot_all(self, cmap=None):
        """Generate all plots

        Helper function for interactive use in a jupyter notebook.
        Generates and shows all plots for user to see at a glance the results of the fit.

        Args:
            cmap(matplotlib.Colormap): Optionally specify the colormap to use for heatmap plots
        """
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 12))
        self.plot_cost_history(ax1)
        self.plot_cost_history_pbest(ax2)
        self.plot_timeseries(self.simulation_time, self.simulation_time / 100.0, ax3)
        plt.show()

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 12))
        self.plot_target_concentration(ax1, cmap)
        self.plot_model_concentration(ax2, cmap)
        plt.show()

    def get_model(self):
        """Returns the model with best parameters applied

        Returns:
            sme.Model: The model with the best parameters applied
        """
        m = sme.open_sbml_file(self.filename)
        self.apply_params(m, self.params)
        return m

import sme_contrib.plot as smeplot
import sme
import os.path
import pytest


def _get_abs_path(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def test_colormap() -> None:
    c1 = smeplot.colormap((1, 0, 0), "red1")
    c2 = smeplot.colormap("#ff0000", "red2")
    c3 = smeplot.colormap("#f00", "red3")
    assert c1.name == "red1"
    assert c2.name == "red2"
    assert c3.name == "red3"
    # zero -> black
    assert c1(0.0) == (0.0, 0.0, 0.0, 1.0)
    assert c2(0.0) == (0.0, 0.0, 0.0, 1.0)
    assert c3(0.0) == (0.0, 0.0, 0.0, 1.0)
    # 1 -> red
    assert c1(1.0) == (1.0, 0.0, 0.0, 1.0)
    assert c2(1.0) == (1.0, 0.0, 0.0, 1.0)
    assert c3(1.0) == (1.0, 0.0, 0.0, 1.0)


def test_concentration_heatmap() -> None:
    model_file = _get_abs_path("model.xml")
    model = sme.open_sbml_file(model_file)
    results = model.simulate(10, 10)
    # single species
    ax, im = smeplot.concentration_heatmap(results[-1], ["A_nucl"])
    assert ax.title.get_text() == "Concentration of A_nucl at time 10.0"
    assert im.get_array().shape == (100, 100)
    assert im.get_array()[0, 0] == 0.0
    # two species
    ax, im = smeplot.concentration_heatmap(results[-1], ["A_nucl", "A_cell"])
    assert ax.title.get_text() == "Concentration of A_nucl, A_cell at time 10.0"
    assert im.get_array().shape == (100, 100)
    assert im.get_array()[0, 0] == 0.0
    # specify title, existing plot axis & colormap
    colormap = smeplot.colormap((1, 0, 0), "red1")
    ax, im = smeplot.concentration_heatmap(
        results[-1], ["A_nucl", "A_cell"], title="my Title", ax=ax, cmap=colormap
    )
    assert ax.title.get_text() == "my Title"
    assert im.get_array().shape == (100, 100)
    assert im.get_array()[0, 0] == 0.0


def test_concentration_heatmap_animation() -> None:
    model_file = _get_abs_path("model.xml")
    model = sme.open_sbml_file(model_file)
    results = model.simulate(10, 10)
    n = len(results)
    # single species
    anim = smeplot.concentration_heatmap_animation(results, ["A_nucl"])
    assert sum(1 for _ in anim.new_frame_seq()) == n
    for frame, result in zip(anim.new_frame_seq(), results):
        assert (
            frame[1].get_text() == f"Concentration of A_nucl: t = {result.time_point}"
        )
    # single species: specify title
    anim = smeplot.concentration_heatmap_animation(results, ["A_nucl"], title="TTL")
    assert sum(1 for _ in anim.new_frame_seq()) == n
    for frame, result in zip(anim.new_frame_seq(), results):
        assert frame[1].get_text() == f"TTL: t = {result.time_point}"
    # two species
    anim = smeplot.concentration_heatmap_animation(results, ["A_nucl", "A_cell"])
    assert sum(1 for _ in anim.new_frame_seq()) == n
    for frame, result in zip(anim.new_frame_seq(), results):
        assert (
            frame[1].get_text()
            == f"Concentration of A_nucl, A_cell: t = {result.time_point}"
        )


def test_facet_grid_3D(exampledata):
    def plot_bloodvessel(label, data, plotter, panel, **kwargs):
        plotter.subplot(*panel)
        plotter.add_mesh(data)

    def plot_brain(label, data, plotter, panel, **kwargs):
        plotter.subplot(*panel)
        plotter.add_volume(
            data,
            cmap="viridis",
            opacity="sigmoid",  # Common opacity mapping for volume rendering
            shade=True,
            ambient=0.3,
            diffuse=0.6,
            specular=0.5,
        )

    def plot_armadillo(label, data, plotter, panel, **kwargs):
        plotter.subplot(*panel)
        plotter.add_mesh(data)

    facetgrid = smeplot.facet_grid3D(
        data={
            "armadillo": exampledata["armadillo"],
            "bloodvessel": exampledata["bloodvessel"],
            "brain": exampledata["brain"],
        },
        plotfuncs={
            "armadillo": plot_armadillo,
            "bloodvessel": plot_bloodvessel,
            "brain": plot_brain,
        },
    )
    facetgrid.show()

    assert facetgrid.shape == (1, 3)


def test_facet_grid_3D_fails(exampledata):
    def plot_bloodvessel(label, data, plotter, panel, **kwargs):
        plotter.subplot(*panel)
        plotter.add_mesh(data)

    def plot_brain(label, data, plotter, panel, **kwargs):
        plotter.subplot(*panel)
        plotter.add_volume(
            data,
            cmap="viridis",
            opacity="sigmoid",  # Common opacity mapping for volume rendering
            shade=True,
            ambient=0.3,
            diffuse=0.6,
            specular=0.5,
        )

    def plot_armadillo(label, data, plotter, panel, **kwargs):
        plotter.subplot(*panel)
        plotter.add_mesh(data)

    with pytest.raises(ValueError):
        smeplot.facet_grid3D(
            data={
                "armadillo": exampledata["armadillo"],
                "bloodvessel": exampledata["bloodvessel"],
                "brain": exampledata["brain"],
            },
            plotfuncs={
                "armadillo": plot_armadillo,
                "bloodvessel": plot_bloodvessel,
                "wrong_key": plot_brain,
            },
        )


def test_facet_grid_animation(tmp_path, exampledata):
    def plot_bloodvessel(label, data, plotter, panel, **kwargs):
        plotter.subplot(*panel)
        plotter.add_mesh(data)

    def plot_brain(label, data, plotter, panel, **kwargs):
        plotter.subplot(*panel)
        plotter.add_volume(
            data,
            cmap="viridis",
            opacity="sigmoid",  # Common opacity mapping for volume rendering
            shade=True,
            ambient=0.3,
            diffuse=0.6,
            specular=0.5,
        )

    def plot_armadillo(label, data, plotter, panel, **kwargs):
        plotter.subplot(*panel)
        plotter.add_mesh(data)

    data_for_frames = [
        {
            "armadillo": exampledata["armadillo"],
            "bloodvessel": exampledata["bloodvessel"],
            "brain": exampledata["brain"],
        },
        {
            "armadillo": exampledata["armadillo"],
            "bloodvessel": exampledata["bloodvessel"],
            "brain": exampledata["brain"],
        },
        {
            "armadillo": exampledata["armadillo"],
            "bloodvessel": exampledata["bloodvessel"],
            "brain": exampledata["brain"],
        },
    ]

    testanimation = smeplot.facet_grid_animation3D(
        tmp_path / "test.mp4",
        data=data_for_frames,
        plotfuncs={
            "armadillo": plot_armadillo,
            "bloodvessel": plot_bloodvessel,
            "brain": plot_brain,
        },
    )

    assert testanimation == tmp_path / "test.mp4"
    assert testanimation.exists()

    with pytest.raises(ValueError):
        smeplot.facet_grid_animation3D(
            tmp_path / "test.mp4",
            data=data_for_frames,
            plotfuncs={
                "armadillo": plot_armadillo,
                "bloodvessel": plot_bloodvessel,
                "wrong_key": plot_brain,
            },
        )

    with pytest.raises(ValueError):
        smeplot.facet_grid_animation3D(
            tmp_path / "test.mp4",
            data=data_for_frames,
            plotfuncs={
                "armadillo": plot_armadillo,
                "bloodvessel": plot_bloodvessel,
                "brain": plot_brain,
            },
            titles=["title1", "title2"],
        )


def test_plot_3D():
    model_file = _get_abs_path("model.xml")
    model = sme.open_sbml_file(model_file)
    results = model.simulate(100, 10)

    # single species
    plotter = smeplot.concentration_heatmap3D(
        simulation_result=results[10],
        species=["A_nucl"],
        cmap="tab10",
        show_cmap=True,
    )

    assert plotter.title == "A_nucl"
    assert plotter is not None
    assert plotter.shape == (1, 1)


def test_plot_3D_animation(tmp_path):
    model_file = _get_abs_path("model.xml")
    model = sme.open_sbml_file(model_file)
    results = model.simulate(100, 10)

    vidpath = smeplot.concentration_heatmap_animation3D(
        filename=tmp_path / "test.mp4",
        simulation_results=results,
        species=["A_nucl"],
        cmap="tab10",
        show_cmap=True,
    )

    assert vidpath is not None
    assert str(vidpath).endswith(".mp4")
    assert os.path.exists(vidpath)

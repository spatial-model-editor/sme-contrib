import sme_contrib.plot as smeplot
import numpy as np
import sme
import os.path


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
    # two species
    ax, im = smeplot.concentration_heatmap(results[-1], ["A_nucl", "A_cell"])
    assert ax.title.get_text() == "Concentration of A_nucl, A_cell at time 10.0"
    # specify title, existing plot axis & colormap
    colormap = smeplot.colormap((1, 0, 0), "red1")
    ax, im = smeplot.concentration_heatmap(
        results[-1], ["A_nucl", "A_cell"], "my Title", ax, colormap
    )
    assert ax.title.get_text() == "my Title"

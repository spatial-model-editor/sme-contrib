import sme_contrib.plot as smeplot
import numpy as np
import sme
import os.path


def _get_abs_path(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def test_concentration_heatmap() -> None:
    modelfile = _get_abs_path("model.xml")
    model = sme.open_sbml_file(modelfile)
    results = model.simulate(10, 10)
    ax = smeplot.concentration_heatmap(results[-1], ["A_nucl"])
    assert ax.title.get_text() == "Concentration of A_nucl at time 10.0"

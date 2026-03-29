import pytest
import pyvista

pyvista.Plotter().close()  # initialize pyvista vtk stuff. silly, but necessary when used with sme
from pyvista import examples

pyvista.OFF_SCREEN = True


@pytest.fixture(scope="session")
def exampledata():
    armadillo = examples.download_armadillo()
    bloodvessel = examples.download_blood_vessels()
    brain = examples.download_brain()

    return {
        "armadillo": armadillo,
        "bloodvessel": bloodvessel,
        "brain": brain,
    }

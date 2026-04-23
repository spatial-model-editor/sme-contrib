import pytest
import pyvista

pyvista.OFF_SCREEN = True

# Initialize PyVista/VTK after headless mode is enabled.
pyvista.Plotter(off_screen=True).close()
from pyvista import examples


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

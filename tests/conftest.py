import pytest
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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sme_contrib import pyvista_utils


def test_rgb_to_scalar():
    img = np.array(
        [
            [[[0, 0, 0], [255, 255, 255]], [[0, 0, 0], [255, 255, 255]]],
            [[[0, 0, 0], [255, 255, 255]], [[0, 0, 0], [255, 255, 255]]],
            [[[0, 0, 0], [255, 255, 255]], [[0, 0, 0], [255, 255, 255]]],
        ]
    )
    scalar_img = pyvista_utils.rgb_to_scalar(img)
    assert scalar_img.shape == (3, 2, 2)
    assert np.all(
        scalar_img == np.array([[[0, 1], [0, 1]], [[0, 1], [0, 1]], [[0, 1], [0, 1]]])
    )


def test_colormap_3D():
    lt = pyvista_utils.colormap_3D()
    cm = plt.get_cmap("tab10").colors
    should = (np.array([mcolors.to_rgba(cm[0])]) * 255).astype(np.int32)
    assert lt.n_values == 1
    assert lt.scalar_range == (0, 1)
    assert np.all(lt.values == should)

    lt = pyvista_utils.colormap_3D("tab20", np.array([0, 1, 2, 3]))
    assert lt.n_values == 4
    assert lt.scalar_range == (0, 4)
    cm = plt.get_cmap("tab20").colors
    should = (
        np.array(
            [
                mcolors.to_rgba(cm[0]),
                mcolors.to_rgba(cm[1]),
                mcolors.to_rgba(cm[2]),
                mcolors.to_rgba(cm[3]),
            ]
        )
        * 255
    ).astype(np.int32)
    assert np.all(lt.values == should)


def test_find_layout():
    assert pyvista_utils.find_layout(1) == (1, 1)
    assert pyvista_utils.find_layout(3) == (1, 3)
    assert pyvista_utils.find_layout(5) == (2, 3)
    assert pyvista_utils.find_layout(8) == (2, 4)
    assert pyvista_utils.find_layout(10) == (2, 5)
    assert pyvista_utils.find_layout(15) == (3, 5)
    assert pyvista_utils.find_layout(15, portrait=True) == (5, 3)
    assert pyvista_utils.find_layout(16) == (4, 4)
    assert pyvista_utils.find_layout(19) == (4, 5)
    assert pyvista_utils.find_layout(23) == (4, 6)
    assert pyvista_utils.find_layout(25) == (5, 5)
    assert pyvista_utils.find_layout(27) == (4, 7)
    assert pyvista_utils.find_layout(29) == (5, 6)
    assert pyvista_utils.find_layout(31) == (5, 7)
    assert pyvista_utils.find_layout(31, portrait=True) == (7, 5)

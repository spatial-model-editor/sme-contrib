<img align="left" width="64" height="64" src="https://raw.githubusercontent.com/spatial-model-editor/spatial-model-editor/main/core/resources/icon.iconset/icon_32x32@2x.png" alt="icon">

# sme-contrib

[![pypi](https://img.shields.io/pypi/v/sme-contrib.svg)](https://pypi.org/project/sme-contrib)
[![python versions](https://img.shields.io/pypi/pyversions/sme-contrib)](https://pypi.org/project/sme-contrib)
[![docs](https://readthedocs.org/projects/sme-contrib/badge/?version=latest)](https://sme-contrib.readthedocs.io)
[![tests](https://github.com/spatial-model-editor/sme-contrib/workflows/Tests/badge.svg)](https://github.com/spatial-model-editor/sme-contrib/actions?query=workflow%3ATests)
[![codecov](https://codecov.io/gh/spatial-model-editor/sme-contrib/branch/master/graph/badge.svg?token=jG4pg9APRN)](https://codecov.io/gh/spatial-model-editor/sme-contrib)
[![sonarcloud quality gate status](https://sonarcloud.io/api/project_badges/measure?project=spatial-model-editor_sme_contrib&metric=alert_status)](https://sonarcloud.io/dashboard?id=spatial-model-editor_sme_contrib)

A collection of useful modules for use with [sme](https://pypi.org/project/sme/),
the python interface to [Spatial Model Editor](https://spatial-model-editor.github.io).

See the [online documentation](https://sme-contrib.readthedocs.io) for more information.

## VTK conflict resolution

Because both `pyvista` and `sme` use VTK, but different versions and bundled in different ways, this leads to library conflict manifesting in code hanging for eternity, random kernel crashes in notebooks or low-level python errors. To avoid it the following rule must be observed:

- only import `sme` after `pyvista` and after `sme_contrib.plot` (which also imports `pyvista`).

an environment flag can be set to disable the automatic initialization of VTK in the `sme_contrib.plot`library to avoid further conflicts:

```python
import os
os.environ['SME_CONTRIB_SKIP_PYVISTA_INIT'] = '1' # disable automatic initialization of pyvista's VTK dependency, which can cause conflicts with sme
```

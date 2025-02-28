# Menger_Curvature

[//]: # (Badges)

| **Latest release** | [![Last release tag][badge_release]][url_latest_release] ![GitHub commits since latest release (by date) for a branch][badge_commits_since]  [![Documentation Status][badge_docs]][url_docs]|
| :----------------- | :------- |
| **Status**         | [![GH Actions Status][badge_actions]][url_actions] [![codecov][badge_codecov]][url_codecov] |
| **Community**      | [![License: GPL v2][badge_license]][url_license]  [![Powered by MDAnalysis][badge_mda]][url_mda]|

[badge_actions]: https://github.com/EtienneReboul/menger_curvature/actions/workflows/gh-ci.yaml/badge.svg
[badge_codecov]: https://codecov.io/gh/EtienneReboul/menger_curvature/branch/main/graph/badge.svg
[badge_commits_since]: https://img.shields.io/github/commits-since/EtienneReboul/menger_curvature/latest
[badge_docs]: https://readthedocs.org/projects/menger_curvature/badge/?version=latest
[badge_license]: https://img.shields.io/badge/License-GPLv2-blue.svg
[badge_mda]: https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA
[badge_release]: https://img.shields.io/github/release-pre/EtienneReboul/menger_curvature.svg
[url_actions]: https://github.com/EtienneReboul/menger_curvature/actions?query=branch%3Amain+workflow%3Agh-ci
[url_codecov]: https://codecov.io/gh/EtienneReboul/menger_curvature/branch/main
[url_docs]: https://menger_curvature.readthedocs.io/en/latest/?badge=latest
[url_latest_release]: https://github.com/EtienneReboul/menger_curvature/releases
[url_license]: https://www.gnu.org/licenses/gpl-2.0
[url_mda]: https://www.mdanalysis.org

"This project aims to provide a simple MDAkit for JIT accelerated Menger curvature calculation"

Menger_Curvature is bound by a [Code of Conduct](https://github.com/EtienneReboul/menger_curvature/blob/main/CODE_OF_CONDUCT.md).

## Installation

To build Menger_Curvature from source,
we highly recommend using virtual environments.
If possible, we strongly recommend that you use
[Anaconda](https://docs.conda.io/en/latest/) as your package manager.
Below we provide instructions both for `conda` and
for `pip`.

## With conda

Ensure that you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.

Create a virtual environment and activate it:

```bash
conda create --name menger_curvature
conda activate menger_curvature
```

Install the development and documentation dependencies:

```bash
conda env update --name menger_curvature --file devtools/conda-envs/test_env.yaml
conda env update --name menger_curvature --file docs/requirements.yaml
```

Build this package from source:

```bash
pip install -e .
```

If you want to update your dependencies (which can be risky!), run:

```bash
conda update --all
```

And when you are finished, you can exit the virtual environment with:

```bash
conda deactivate
```

## With pip

To build the package from source, run:

```bash
pip install .
```

If you want to create a development environment, install
the dependencies required for tests and docs with:

```bash
pip install ".[test,doc]"
```

## Examples

Calculate Menger curvature for the chain A of a tubulin protein trajectory in serial mode:

```python
>>> import MDAnalysis as mda
>>> from menger.analysis.mengercurvature import MengerCurvature
>>> from menger.tests.utils import make_universe
>>> md_name = "tubulin_chain_a"
>>> u = make_universe(
...     topology_name=f"{md_name}.pdb",
...     trajectory_name=f"{md_name}.dcd",
... )
>>> menger_analyser = MengerCurvature(
...     u,
...     selection="name CA and chainID A",
...     spacing=2
... )
>>> menger_analyser.run()
>>> average_curvature = menger_analyser.results.local_curvatures
>>> flexibility = menger_analyser.results.local_flexibilities
>>> menger_curvature = menger_analyser.results.curvature_array
```

Calculate Menger curvature for the chain A of a tubulin protein trajectory in parallel mode:

```python
>>> import MDAnalysis as mda
>>> from menger.analysis.mengercurvature import MengerCurvature
>>> from menger.tests.utils import make_universe
>>> md_name = "tubulin_chain_a"
>>> u = make_universe(
...     topology_name=f"{md_name}.pdb",
...     trajectory_name=f"{md_name}.dcd",
... )
>>> menger_analyser = MengerCurvature(
...     u,
...     selection="name CA and chainID A",
...     spacing=2,
...     n_workers=4
... )
>>> menger_analyser.run_parallel()
>>> average_curvature = menger_analyser.results.local_curvatures
>>> flexibility = menger_analyser.results.local_flexibilities
>>> menger_curvature = menger_analyser.results.curvature_array
```

### Copyright

The Menger_Curvature source code is hosted at <https://github.com/EtienneReboul/menger_curvature>
and is available under the GNU General Public License, version 2 (see the file [LICENSE](https://github.com/EtienneReboul/menger_curvature/blob/main/LICENSE)).

Copyright (c) 2025, LBT

### Acknowledgements

Project based on the
[MDAnalysis Cookiecutter](https://github.com/MDAnalysis/cookiecutter-mda) version 0.1.
Please cite [MDAnalysis](https://github.com/MDAnalysis/mdanalysis#citation) when using Menger_Curvature in published work.

# TerraTools
Tools to read, analyse and visualise models written by the TERRA mantle convection code.
TerraTools is released under an MIT License.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10797185.svg)](https://doi.org/10.5281/zenodo.10797185)

Homepage: [https://terratools.readthedocs.io/en/latest/](https://terratools.readthedocs.io/en/latest/)<br>
Documentation: [https://terratools.readthedocs.io/en/latest/](https://terratools.readthedocs.io/en/latest/)<br>
Source code: [https://github.com/mantle-convection-constrained/terratools](https://github.com/mantle-convection-constrained/terratools)<br>
Publication: [![DOI](https://joss.theoj.org/papers/10.21105/joss.07539/status.svg)](https://doi.org/10.21105/joss.07539)<br>


## Citing TerraTools
If you use terratools, please cite the associated
[JOSS paper](https://doi.org/10.21105/joss.07539).  You can make use of the
BibTeX entry below:

```bibtex
@article{terratools:2025,
  title = {Terratools: {A} {P}ython package to analyse {TERRA} mantle convection simulations},
  shorttitle = {Terratools},
  author = {Nowacki, Andy and Panton, James and Ward, Jamie and Myhill, Bob and Walker, Andrew and Wookey, James and Davies, J. Huw},
  year = 2025,
  month = dec,
  journal = {Journal of Open Source Software},
  volume = {10},
  number = {116},
  pages = {7539},
  issn = {2475-9066},
  doi = {10.21105/joss.07539},
  urldate = {2025-12-16},
  langid = {english}
}
```

## Installation

### Requirements
TerraTools requires Python version 3.9 or newer.

If you want to use the map plotting functions (such as `TerraModel.plot_layer`), make sure you have a working installation of [Cartopy](https://scitools.org.uk/cartopy/docs/latest/installing.html).

### Pre-installation
If you are using a Conda-like package manager (e.g.,
[Miniconda](https://docs.anaconda.com/miniconda/) or
[Mamba](https://mamba.readthedocs.io/en/latest/)), we recommend always
[creating a new environment](https://mamba.readthedocs.io/en/latest/user_guide/concepts.html#concepts)
for each project you are working on into which you install TerraTools.
For instance:
```sh
conda create -n amazing_mantle_convection_project python=3.11
conda activate amazing_mantle_convection_project
```

This is entirely optional but often prevents issues with dependency version
conflicts.

### Installing the latest released version

#### Installation with `pip`
Before installing TerraTools with `pip`, first install and/or upgrade your
version of pip:
```sh
python -m ensurepip --upgrade
```

To install the latest released version of TerraTools, then do:
```sh
python -m pip install terratools
```

#### Installation with dependency management systems
If you use a dependency management system such as
[Poetry](https://python-poetry.org/) or [Pipenv](https://pipenv.pypa.io/en/latest/)
you should add `terratools` as a dependency of your project.

### Installing the development version

You can also install the latest development version of TerraTools from source. To do this, first clone the repository onto your local machine using git:
```sh
git clone https://github.com/mantle-convection-constrained/terratools.git
```
Then navigate to the top level directory and install in development mode:
```sh
cd terratools; python -m pip install -ve .
```

### Post-installation

Finally, check you have a fully working installation:
```sh
python -c "import terratools"
```

## Reporting bugs
If you would like to report any bugs, please raise an issue on [GitHub](https://github.com/mantle-convection-constrained/terratools/issues).

## Contributing to TerraTools
If you would like to contribute bug fixes, new functions or new modules to the existing codebase, please fork the terratools repository, make the desired changes and then make a pull request on [GitHub](https://github.com/mantle-convection-constrained/terratools/pulls).

## Acknowledgement and Support
This project is supported by [NERC Large Grant MC-squared](https://www.cardiff.ac.uk/research/explore/find-a-project/view/2592859-mc2-mantle-circulation-constrained).

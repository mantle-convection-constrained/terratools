# TerraTools
Tools to read, analyse and visualise models written by the TERRA mantle convection code.
TerraTools is released under an MIT License.

Homepage: [https://terratools.readthedocs.io/en/latest/](https://terratools.readthedocs.io/en/latest/)<br>
Documentation: [https://terratools.readthedocs.io/en/latest/](https://terratools.readthedocs.io/en/latest/)<br>
Source code: [https://github.com/mantle-convection-constrained/terratools](https://github.com/mantle-convection-constrained/terratools)<br>
DOI for this version: [10.5281/zenodo.10797186](https://zenodo.org/records/10797186)

## Citing TerraTools
We are currently writing a paper for submission to JOSS. Watch this space.

## Installation

### Pre-installation

Before installing TerraTools, first install and/or upgrade your version of pip:
```
python -m ensurepip --upgrade
```
If you want to use the map plotting functions (such as `TerraModel.plot_layer`), make sure you have a working installation of [Cartopy](https://scitools.org.uk/cartopy/docs/latest/installing.html), as installation of Cartopy using pip requires [additional dependencies are met first](https://scitools.org.uk/cartopy/docs/latest/installing.html). On Mac machines, you may find that after you follow the instructions on that site, you still need to add the following command to your `~/.bashrc` or equivalent:
```
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/geos/lib/
```

### Installing the latest released version

To install the latest released version of TerraTools, please use:
```
python -m pip install terratools
```

### Installing the development version

You can also install the latest development version of TerraTools from source. To do this, first clone the repository onto your local machine using git:
```
git clone git@github.com:mantle-convection-constrained/terratools.git
```
Then navigate to the top level directory and install in development mode:
```
cd terratools; python -m pip install -ve .
```

### Post-installation

Finally, check you have a fully working installation:
```
python -c "import terratools"
```

## Reporting bugs
If you would like to report any bugs, please raise an issue on [GitHub](https://github.com/mantle-convection-constrained/terratools/issues).

## Contributing to TerraTools
If you would like to contribute bug fixes, new functions or new modules to the existing codebase, please fork the terratools repository, make the desired changes and then make a pull request on [GitHub](https://github.com/mantle-convection-constrained/terratools/pulls).

## Acknowledgement and Support
This project is supported by [NERC Large Grant MC-squared](https://www.cardiff.ac.uk/research/explore/find-a-project/view/2592859-mc2-mantle-circulation-constrained).

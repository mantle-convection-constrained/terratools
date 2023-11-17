"""
This submodule provides access to example data for TerraTools. Data is
downloaded on first use and stored in a local cache. This is managed
using pooch with file names returned by each of the public functions in
the submodule.

For example, to access the basalt table (in 'example_bas_table.dat')
you could run:

    import terratools.example_data
    filename = terratools.example_data.example_bas_table()

The filename is then the full path to the example data file.
"""
import pooch

_EXAMPLE_DATA = pooch.create(
    path=pooch.os_cache("terratools"),
    base_url="doi:10.6084/m9.figshare.24100362.v3",
    registry=None,
)

_EXAMPLE_DATA.load_registry_from_doi()


def example_bas_table():
    """
    Return the full filename for the basalt example file

    This file was created using PerpleX with the Stixrude
    and Lithgow-Bertelloni (2021) dataset, by adapting the
    script in examples/example_perplex.py

    The P-T range is 0-140 GPa (2 GPa intervals) and
    300-4300 K (200 K intervals).

    The bulk composition (molar amounts) is:
    SiO2: 52.298,
    MgO: 15.812,
    FeO: 7.121,
    CaO: 13.027,
    Al2O3: 9.489,
    Na2O: 2.244.
    """
    filename = _EXAMPLE_DATA.fetch("example_bas_table.dat")
    return filename


def example_hzb_table():
    """
    Return the full filename for the harzburgite example file

    This file was created using PerpleX with the Stixrude
    and Lithgow-Bertelloni (2021) dataset, by adapting the
    script in examples/example_perplex.py

    The P-T range is 0-140 GPa (2 GPa intervals) and
    300-4300 K (200 K intervals).

    The bulk composition (molar amounts) is:
    SiO2: 36.184,
    MgO: 56.559,
    FeO: 5.954,
    CaO: 0.889,
    Al2O3: 0.492,
    Na2O: 0.001.
    """
    filename = _EXAMPLE_DATA.fetch("example_hzb_table.dat")
    return filename


def example_lhz_table():
    """
    Return the full filename for the lherzolite example file

    This file was created using PerpleX with the Stixrude
    and Lithgow-Bertelloni (2021) dataset, by adapting the
    script in examples/example_perplex.py

    The P-T range is 0-140 GPa (2 GPa intervals) and
    300-4300 K (200 K intervals).

    The bulk composition (molar amounts) is:
    SiO2: 38.819,
    MgO: 49.894,
    FeO: 6.145,
    CaO: 2.874,
    Al2O3: 1.963,
    Na2O: 0.367.
    """
    filename = _EXAMPLE_DATA.fetch("example_lhz_table.dat")
    return filename


def example_terra_model():
    """
    Return the full filename for an example mantle convection model.

    This is a NetCDF file of a mantle convection model containing
    the temperature, composition and flow velicity fields at 64 radii
    with each radii having 2562 points.

    This model was downsampled from a higher resolution model for the
    purposes of saving memory.

    """
    filename = _EXAMPLE_DATA.fetch("example_model.nc")
    return filename


def example_terra_layer():
    """
    Return the full filename for an example layer file from a mantle
    circulation model.

    This is a concatenated NetCDF file of a Terra Model Layer, produced
    using `ncecat` which is available through the NetCDF Operators
    (NCO) package.

    It consists of a single radial layer and 163842 lateral points.

    """
    filename = _EXAMPLE_DATA.fetch("example_layer.nc")
    return filename

"""
This submodule provides access to example data for TerraTools. Data is 
downloaded on first use and stored in a local cache. This is managed
using pooch with file names returned by each of the public functions in
the submodule.

For example, to access the basalt table (in 'example_bas_table.dat')
you could run:

    import terratools.example_data
    filename = terratools.example_data.example_bas_table()
    # filename is then the full path to the example data file
"""
import pooch

_EXAMPLE_DATA = pooch.create(path=pooch.os_cache("terratools"), 
                             base_url="doi:10.6084/m9.figshare.24100362.v1",
                             registry=None)

_EXAMPLE_DATA.load_registry_from_doi()


def example_bas_table():
    """
    Return the full filename for the basalt example file
    """
    filename = _EXAMPLE_DATA.fetch('example_bas_table.dat')
    return filename


def example_hzb_table():
    """
    Return the full filename for the harzbergite example file
    """
    filename = _EXAMPLE_DATA.fetch('example_hzb_table.dat')
    return filename 


def example_lhz_table():
    """
    Return the full filename for the lherzolite example file
    """
    filename = _EXAMPLE_DATA.fetch('example_lhz_table.dat')
    return filename


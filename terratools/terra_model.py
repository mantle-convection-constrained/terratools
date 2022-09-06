"""
terra_model
===========

The terra_model module provides the TerraModel class.  This holds
the data contained within a single time slice of a TERRA mantle
convection simulation.
"""

# import netcdf4
import numpy as np

# Precision of coordinates in TerraModel
COORDINATE_TYPE = np.float64
# Precision of values in TerraModel
VALUE_TYPE = np.float64

# The following are the names of scalar fields which are allowed
# to be present in the _fields attribute of TerraModel, and a description
# of those fields.
_SCALAR_FIELDS = {
    't': 'Temperature field [K]',
    'c': 'Scalar composition field [unitless]',
    'p': 'Pressure field [GPa]',
    'vp': 'P-wave velocity [km/s]',
    'vs': 'S-wave velocity [km/s]',
    'density': 'Density [g/cm^3]',
    'qp': 'P-wave attenuation [unitless]',
    'qs': 'S-wave attenuation [unitless]'
}

_VECTOR_FIELDS = {
    'u': 'Flow field [m/s]',
    'c_hist': 'Composition histogram (_nc components) [unitles]'
}

# All fields of any kind
_ALL_FIELDS = {**_SCALAR_FIELDS, **_VECTOR_FIELDS}


class TerraModel:
    """
    Class holding a TERRA model at a single point in time.
    """

    def __init__(self):
        self._nlayers = 0
        self._npts = 0
        self._lon = np.array([])
        self._lat = np.array([])
        self._radius = np.array([])
        # If True, we are using a composition histogram apporach and
        # composition is a 'vector' field
        self._c_histogram = True
        # All the fields are held within _fields, either as scalar or
        # 'vector' fields.
        self._fields = {}
        # A set of lookup tables
        self._lookup_tables = None

    def fields(self):
        """
        Return the fields present in a TerraModel.

        :returns: list of the names of the fields present.
        """
        return self._fields.keys()

    def read_netcdf(files):
        """
        Read a TerraModel from a set of NetCDF files.

        :param files: List or iterable of file names of TERRA NetCDF model
            files
        :returns: a new TeraModel
        """
        pass

    def get_value(self, r, lat, lon, field):
        """
        Evaluate the value of field at radius r km, latitude lat degrees
        and longitude lon degrees.

        :param r: Radius in km
        :param lat: Latitude in degrees
        :param lon: Longitude in degrees
        :param field: String giving the name of the field of interest
        :returns: value of the field at that point
        """
        if not field in _ALL_FIELDS.keys():
            raise KeyError("'{}' is not a valid field name".format(field))
        if not field in self._fields.keys():
            raise KeyError("Field '{}' is not present in the model".format(field))
        pass

"""
terra_model
===========

The terra_model module provides the TerraModel class.  This holds
the data contained within a single time slice of a TERRA mantle
convection simulation.
"""

import netCDF4
import numpy as np

# Precision of coordinates in TerraModel
COORDINATE_TYPE = np.float32
# Precision of values in TerraModel
VALUE_TYPE = np.float32

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
    'qp': 'P-wave quality factor [unitless]',
    'qs': 'S-wave quality factor [unitless]',
}

# These are 'vector' fields which contain more than one component
# at each node of the grid.
_VECTOR_FIELDS = {
    'u': 'Flow field (three components) [m/s]',
    'c_hist': 'Composition histogram (_nc components) [unitles]',
}

_VECTOR_FIELD_NCOMPS = {
    'u': 3,
    'c_hist': None,
}

# All fields of any kind
_ALL_FIELDS = {**_SCALAR_FIELDS, **_VECTOR_FIELDS}

# Mapping of fields to NetCDF file variable names.
# Each field name maps to one or more variables.
# Fields which don't have a defined name should not have a key in this dict.
_FIELD_NAME_TO_VARIABLE_NAME = {
    't': ("Temperature",),
    'c_hist': ("BasaltFrac", "LherzFrac"),
    'u': ("Velocity_x", "Velocity_y", "Velocity_z"),
}

# Mapping of variable names in NetCDF files to field names
# in this module.  This is a many to one mapping.
_VARIABLE_NAME_TO_FIELD_NAME = {}
for key, vals in _FIELD_NAME_TO_VARIABLE_NAME.items():
    for val in vals:
        _VARIABLE_NAME_TO_FIELD_NAME[val] = key


class FieldNameError(Exception):
    """
    Exception type raised when trying to use an incorrect field name
    """
    def __init__(self, field):
        self.message = f"'{field}' is not a valid TerraModel field name"
        super().__init__(self.message)


class NoFieldError(Exception):
    """
    Exception type raised when trying to access a field which is not present
    """
    def __init__(self, field):
        self.message = f"Model does not contain field {field}"
        super().__init__(self.message)


class FieldDimensionError(Exception):
    """
    Exception type raised when trying to access a field which is not present
    """
    def __init__(self, model, array):
        self.message = f"Field array has incorrect first two dimensions. " + \
            f"Expected {(model._nlayers, model._npts)}; got {array.shape[0:2]}"
        super().__init__(self.message)


class TerraModel:
    """
    Class holding a TERRA model at a single point in time.
    """

    def __init__(self, r, lon, lat, fields={}, c_histogram=False, lookup_tables=None):
        """
        Construct a TerraModel.

        :param r: Radius of nodes in radial direction.  Must increase monotonically.
        :param lon: Position in longitude of lateral points (degrees)
        :param lat: Position in latitude of lateral points (degrees).  lon and
            lat must be the same length
        :param fields: dict whose keys are the names of field, and whose
            values are numpy arrays of dimension (nlayers, npts) for
            scalar fields, and (nlayers, npts, ncomps) for a field
            with ``ncomps`` components at each point
        :param c_histogram: Whether the set of compositions define
            a 'composition histogram' representing a mechanical mixture
            of compositions, or a set of endmember compositions between
            which we should interpolate
        :param lookup_tables: An iterable of SeismicLookupTable corresponding
            to the number of compositions for this model
        """

        nlayers = len(r)
        self._nlayers = nlayers

        npts = len(lon)
        if len(lat) != npts:
            raise ValueError("length of lon and lat must be the same")
        self._npts = npts

        self._lon = np.array(lon, dtype=COORDINATE_TYPE)
        self._lat = np.array(lat, dtype=COORDINATE_TYPE)
        self._radius = np.array(r, dtype=COORDINATE_TYPE)

        # Check for monotonicity of radius
        # if not np.all(self._radius[1:] - self._radius[:-1] > 0):
            # raise ValueError("radii must increase monotonically")

        # If True, we are using a composition histogram approach and
        # composition is a 'vector' field
        self._c_histogram = None

        # All the fields are held within _fields, either as scalar or
        # 'vector' fields.
        self._fields = {}

        # A set of lookup tables
        self._lookup_tables = lookup_tables

        # Check fields have the right shape and convert
        for key, val in fields.items():
            array = np.array(val, dtype=VALUE_TYPE)

            if _is_scalar_field(key):
                if array.shape != (nlayers, npts):
                    raise FieldDimensionError(model, array)

            elif _is_vector_field(key):
                expected_ncomps = _expected_vector_field_ncomps(key)
                if expected_ncomps is None:
                    if array.shape[0:2] != (nlayers, npts):
                        raise ValueError(f"Incorrect shape of field {key}. " + \
                            f"Expected ({nlayers}, {npts}, ncomps) but got " + \
                            f"{array.shape}")
                else:
                    if array.shape != (nlayers, npts, expected_ncomps):
                        raise ValueError(f"Incorrect shape of field {key}. " + \
                            f"Expected ({nlayers}, {npts}, {expected_ncomps}) but got " + \
                            f"{array.shape}")

            else:
                raise FieldNameError(key)

            self.set_field(key, array)


    def field_names(self):
        """
        Return the names of the fields present in a TerraModel.

        :returns: list of the names of the fields present.
        """
        return self._fields.keys()


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
        _check_field_name(field)
        self._check_has_field(field)


    def set_field(self, field, values):
        """
        Create a new field within a TerraModel from a predefined array.

        :param field: Name of field
        :param array: numpy.array containing the field.  For scalars it
            should have dimensions corresponding to (nlayers, npts),
            where nlayers is the number of layers and npts is the number
            of lateral points.  For multi-component fields, it should
            have dimensions (nlayers, npts, ncomps), where ncomps is the
            number of components
        """
        _check_field_name(field)
        array = np.array(values, dtype=VALUE_TYPE)
        self._check_field_shape(array)
        self._fields[field] = np.array(array, dtype=VALUE_TYPE)


    def new_field(self, name):
        """
        Create a new field with key ``name`` and return the field.

        :param name: Name of new field.
        """
        _check_field_name(name)
        self.set_field(name, np.zeros((self._nlayers, self._npts), dtype=VALUE_TYPE))


    def has_field(self, field):
        """
        Return True if this TerraModel contains a field called ``field``.

        :param field: Name of field
        :returns: True if field is present, and False otherwise
        """
        return field in self._fields.keys()


    def get_field(self, field):
        """
        Return the array containing the values of field in a TerraModel.

        :param field: Name of the field
        :returns: the field of interest as a numpy.array
        """
        self._check_has_field(field)
        return self._fields[field]


    def _check_has_field(self, field):
        """
        If field is not present in this model, raise a NoFieldError
        """
        if not self.has_field(field):
            raise NoFieldError(field)


    def _check_field_shape(self, array, scalar=True):
        """
        If the first two dimensions of array are not (nlayers, npts),
        raise an error.
        """
        if len(array.shape) not in (2, 3):
            raise FieldDimensionError(self, array)
        test_shape = array.shape if scalar else array.shape[0:2]
        if not test_shape == (self._nlayers, self._npts):
            raise FieldDimensionError(self, array)


    def number_of_compositions(self):
        """
        If a model contains a composition histogram field ('c_hist'),
        return the number of compositions; otherwise return None.

        :returns: number of compositions or None
        """
        if self.has_field("c_hist"):
            return self.get_field("c_hist").shape[2]
        else:
            return None

def read_netcdf(files, fields=None, surface_radius=6370.0, test_lateral_points=False):
    """
    Read a TerraModel from a set of NetCDF files.

    :param files: List or iterable of file names of TERRA NetCDF model
        files
    :param fields: Iterable of field names to be read in.  By default all
        fields are read in.
    :param surface_radius: Radius of the surface of the model in km
        (default 6370 km)
    :returns: a new TerraModel
    """
    # Check fields are readable
    if fields is not None:
        for field in fields:
            _check_field_name(field)

    # Passed to constructor
    _fields = {}
    _r = np.array([], dtype=COORDINATE_TYPE)
    _lat = np.array([], dtype=COORDINATE_TYPE)
    _lon = np.array([], dtype=COORDINATE_TYPE)

    for (file_number, file) in enumerate(files):
        nc = netCDF4.Dataset(file)

        # Check the file has the right things
        for dimension in ('nps', 'Depths'):
            assert dimension in nc.dimensions, \
                f"Can't find {dimension} in dimensions of file {file}"

        # Number of lateral points in this file
        npts = nc.dimensions["nps"].size
        # Number of radii
        nlayers = nc.dimensions["Depths"].size

        if file_number == 0:
            # Take the radii from the first file
            _r = surface_radius - nc["Depths"][:]
        else:
            # Check the radii are the same for this file as the first
            assert np.all(_r == surface_radius - nc["Depths"][:]), \
                f"radii in file {file} do not match those in {file[0]}"

        # Assume that the latitudes and longitudes are the same for each
        # depth slice, and so are repeated
        _lat = np.append(_lat, nc["Latitude"][0,:])
        _lon = np.append(_lon, nc["Longitude"][0,:])

        # Test this assumption
        # FIXME: This is too slow to be useful.  Work out how to do this quicker
        #        so we can actually verify this is correct.
        if test_lateral_points:
            for idep in range(1, nlayers):
                assert np.all(nc["Latitude"][0,:] == nc["Latitude"][idep,:]), \
                    f"Latitudes of depth slice {idep} do not match those of slice 0"
                assert np.all(nc["Longitude"][0,:] == nc["Longitude"][idep,:]), \
                    f"Longitudes of depth slice {idep} do not match those of slice 0"

        # Now read in fields, with some special casing
        fields_to_read = _ALL_FIELDS.keys() if fields is None else fields
        # TODO: Actually read in variables
        # for var in nc.variables:
    

    return TerraModel(r=_r, lon=_lon, lat=_lat)


def _is_valid_field_name(field):
    """
    Return True if field is a valid name of a field in a TerraModel.
    """
    return field in _ALL_FIELDS.keys()


def _variable_name_from_field(field):
    """
    Return the netCDF variable name of a field from the TerraModel field name.
    """
    return _FIELD_NAME_TO_VARIABLE_NAME[field]


def _check_field_name(field):
    """
    If field is not a valid field name, raise a FieldNameError
    """
    if not _is_valid_field_name(field):
        raise FieldNameError(field)

def _is_scalar_field(field):
    """
    Return True if field is a scalar field
    """
    return field in _SCALAR_FIELDS.keys()

def _is_vector_field(field):
    """
    Return True if field is a 'vector' field
    """
    return field in _VECTOR_FIELDS.keys()

def _expected_vector_field_ncomps(field):
    """
    Return the expected number of components in a 'vector' field,
    or None if it may take any value
    """
    return _VECTOR_FIELD_NCOMPS[field]

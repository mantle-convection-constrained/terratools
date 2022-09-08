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
    'vp': 'P-wave velocity (elastic) [km/s]',
    'vs': 'S-wave velocity (elastic) [km/s]',
    'vp_an': 'P-wave velocity (anelastic) [km/s]',
    'vs_an': 'S-wave velocity (anelastic) [km/s]',
    'density': 'Density [g/cm^3]',
    'qp': 'P-wave quality factor [unitless]',
    'qs': 'S-wave quality factor [unitless]',
}

# These are 'vector' fields which contain more than one component
# at each node of the grid.
_VECTOR_FIELDS = {
    'u_xyz': 'Flow field in Cartesian coordinates (three components) [m/s]',
    'u_geog': 'Flow field in geographic coordinates (three components) [m/s]',
    'c_hist': 'Composition histogram (_nc components) [unitles]',
}

_VECTOR_FIELD_NCOMPS = {
    'u_xyz': 3,
    'u_geog': 3,
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
    'u_xyz': ("Velocity_x", "Velocity_y", "Velocity_z"),
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
    def __init__(self, model, array, name=""):
        self.message = f"Field array {name} has incorrect first two dimensions. " + \
            f"Expected {(model._nlayers, model._npts)}; got {array.shape[0:2]}"
        super().__init__(self.message)


class TerraModel:
    """
    Class holding a TERRA model at a single point in time.
    """

    def __init__(self, r, lon, lat, fields={}, c_histogram=False,
        c_histogram_names=None, lookup_tables=None):
        """
        Construct a TerraModel.

        The basic TerraModel constructor requires the user to supply a set
        of radii ``r`` in km, and a set of lateral point coordinates as
        two separate arrays of longitude and latitude (in degrees).
        These then define the 2D grid on which the field values are defined.

        Fields are passed in as a dict mapping field name onto either a
        2D array (for scalar fields like temperature), or a 3D array
        for multicomponent fields, like flow velocity or composition histograms.

        Composition histograms enable one to translate from temperature,
        pressure and composition to seismic velocity by assuming the
        model contains a mechanical mixture of some number of different
        chemical components, and performing Voigt averaging over the
        elastic parameters predicted for these compositions, weighted
        by their proportion.  The field ``"c_hist"`` holds a 3D array
        whose last dimension gives the proportion of each component.
        When using a composition histogram, you may pass the
        ``c_histogram_name`` argument, giving the name of each component.
        The ith component name corresponds to the proportion of the ith
        slice of the last dimension of the ``"c_hist"`` array.

        Seismic lookup tables should be passed to this constructor
        when using multicomponent composition histograms; one for
        each proportion.  Note that this is not enforced, however.

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
        :param c_histogram_names: The names of each composition of the
            composition histogram
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
        if not np.all(self._radius[1:] - self._radius[:-1] > 0) and \
                not np.all(self._radius[1:] - self._radius[:-1] < 0):
            raise ValueError("radii must increase or decrease monotonically")

        # If True, we are using a composition histogram approach and
        # composition is a 'vector' field
        self._c_histogram = None
        # The names of the compositions
        self._c_hist_names = c_histogram_names

        # All the fields are held within _fields, either as scalar or
        # 'vector' fields.
        self._fields = {}

        # A set of lookup tables
        self._lookup_tables = lookup_tables

        # Check fields have the right shape and convert
        for key, val in fields.items():
            array = np.array(val, dtype=VALUE_TYPE)

            if _is_scalar_field(key):
                self._check_field_shape(val, key, scalar=True)

            elif _is_vector_field(key):
                expected_ncomps = _expected_vector_field_ncomps(key)
                if expected_ncomps is None:
                    _check_field_shape(self, val, key, scalar=False)
                else:
                    if array.shape != (nlayers, npts, expected_ncomps):
                        raise FieldDimensionError(self, val, key)

            else:
                raise FieldNameError(key)

            self.set_field(key, array)


    def field_names(self):
        """
        Return the names of the fields present in a TerraModel.

        :returns: list of the names of the fields present.
        """
        return self._fields.keys()


    def get_value(self, lon, lat, r, field, depth=False):
        """
        Evaluate the value of field at radius r km, latitude lat degrees
        and longitude lon degrees.

        :param lon: Longitude in degrees
        :param lat: Latitude in degrees
        :param r: Radius in km
        :param field: String giving the name of the field of interest
        :param depth: If True, treat r as a depth rather than a radius
        :returns: value of the field at that point
        """
        _check_field_name(field)
        self._check_has_field(field)
        if depth:
            r = np.max(self._radius) - r

        array = self.get_field(field)


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
        self._check_field_shape(array, field, scalar=_is_scalar_field(field))
        self._fields[field] = np.array(array, dtype=VALUE_TYPE)


    def new_field(self, name, ncomps=None):
        """
        Create a new, empty field with key ``name``.

        :param name: Name of new field.
        :param ncomps: Number of components for a multicomponent field.
        """
        _check_field_name(name)
        if ncomps is not None:
            self.set_field(name, np.zeros((self._nlayers, self._npts), dtype=VALUE_TYPE))
        else:
            self.set_field(name, np.zeros((self._nlayers, self._npts, ncomps), dtype=VALUE_TYPE))


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


    def _check_field_shape(self, array, name, scalar=True):
        """
        If the first two dimensions of array are not (nlayers, npts),
        raise an error.
        """
        if len(array.shape) not in (2, 3):
            raise FieldDimensionError(self, array, name)
        if (scalar and array.shape != (self._nlayers, self._npts)) or \
                (not scalar and array.shape[0:2] != (self._nlayers, self._npts)):
            raise FieldDimensionError(self, array, name)


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


    def composition_names(self):
        """
        If a model contains a composition histogram field ('c_hist'),
        return the names of the compositions; otherwise return None.

        :returns: list of composition names
        """
        if self.has_field("c_hist"):
            return self._composition_names
        else:
            return None


    def get_lateral_points(self):
        """
        Return two numpy.arrays, one each for the longitude and latitude
        (in degrees) of the lateral points of each depth slice of the fields
        in a model.

        :returns: (lon, lat) in degrees
        """
        return self._lon, self._lat


    def get_radii(self):
        """
        Return the radii of each layer in the model, in km.

        :returns: radius of each layer in km
        """
        return self._radius


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

    if len(files) == 0:
        raise ValueError("files argument cannot be empty")

    # Check fields are readable
    if fields is not None:
        for field in fields:
            _check_field_name(field)

    # Total number of lateral points and number of layers,
    # allowing us to preallocate arrays.  Consistency is checked on the next pass.
    npts_total = 0
    for (file_number, file) in enumerate(files):
        nc = netCDF4.Dataset(file)
        if "nps" not in nc.dimensions:
            raise ValueError(f"File {file} does not contain the dimension 'nps'")
        npts_total += nc.dimensions["nps"].size

        if "Depths" not in nc.dimensions:
            raise ValueError(f"File {file} does not contain the dimension 'Depths'")
        if file_number == 0:
            nlayers = nc.dimensions["Depths"].size
            # Take the radii from the first file
            _r = np.array(surface_radius - nc["Depths"][:], dtype=COORDINATE_TYPE)

    # Passed to constructor
    _fields = {}
    _lat = np.empty((npts_total,), dtype=COORDINATE_TYPE)
    _lon = np.empty((npts_total,), dtype=COORDINATE_TYPE)

    npts_pointer = 0
    for (file_number, file) in enumerate(files):
        nc = netCDF4.Dataset(file)

        # Check the file has the right things
        for dimension in ('nps', 'Depths'):
            assert dimension in nc.dimensions, \
                f"Can't find {dimension} in dimensions of file {file}"

        # Number of lateral points in this file
        npts = nc.dimensions["nps"].size
        # Range of points in whole array to fill in
        npts_range = range(npts_pointer, npts_pointer + npts)

        if file_number > 0:
            # Check the radii are the same for this file as the first
            assert np.all(_r == surface_radius - nc["Depths"][:]), \
                f"radii in file {file} do not match those in {files[0]}"

        # Assume that the latitudes and longitudes are the same for each
        # depth slice, and so are repeated
        this_slice_lat = nc["Latitude"][0,:]
        this_slice_lon = nc["Longitude"][0,:]
        _lat[npts_range] = this_slice_lat
        _lon[npts_range] = this_slice_lon

        # Test this assumption
        if test_lateral_points:
            # Indexing with a single `:` gets an N-dimensional array, not
            # a vector
            all_lats = nc["Latitude"][:]
            all_lons = nc["Longitude"][:]
            for idep in range(1, nlayers):
                assert np.all(this_slice_lat == all_lats[idep,:]), \
                    f"Latitudes of depth slice {idep} do not match those of slice 0"
                assert np.all(this_slice_lon == all_lons[idep,:]), \
                    f"Longitudes of depth slice {idep} do not match those of slice 0"

        # Now read in fields, with some special casing
        fields_to_read = _ALL_FIELDS.keys() if fields == None else fields
        fields_read = set()

        for var in nc.variables:
            # Skip 'variables' like Latitude, Longitude and Depths which
            # give the values of the dimensions
            if var in ("Latitude", "Longitude", "Depths"):
                continue

            field_name = _field_name_from_variable(var)

            if field_name not in fields_to_read:
                continue

            # Handle scalar fields
            if _is_scalar_field(field_name):
                field_data = nc[var][:]

                if field_name not in _fields.keys():
                    _fields[field_name] = np.empty(
                        (nlayers, npts_total), dtype=VALUE_TYPE)
                _fields[field_name][:,npts_range] = field_data
                fields_read.add(field_name)

            # Special case for flow field
            if "u_xyz" in fields_to_read and field_name == "u_xyz":
                if field_name in fields_read:
                    continue
                else:
                    fields_read.add(field_name)

                ncomps = _VECTOR_FIELD_NCOMPS[field_name]
                uxyz = np.empty((nlayers, npts, ncomps), dtype=VALUE_TYPE)
                uxyz[:,:,0] = nc["Velocity_x"][:]
                uxyz[:,:,1] = nc["Velocity_y"][:]
                uxyz[:,:,2] = nc["Velocity_z"][:]

                if field_name not in _fields.keys():
                    _fields[field_name] = np.empty(
                        (nlayers, npts_total, ncomps), dtype=VALUE_TYPE)
                _fields[field_name][:,npts_range,:] = uxyz

            # Special case for other vector fields; i.e. c_hist
            if "c_hist" in fields_to_read and field_name == "c_hist":
                if field_name in fields_read:
                    continue
                else:
                    fields_read.add(field_name)



        npts_pointer += npts

    # Remove duplicate points
    _, unique_indices = np.unique(np.stack((_lon, _lat)), axis=1, return_index=True)
    _lon = _lon[unique_indices]
    _lat = _lat[unique_indices]
    for (field_name, array) in _fields.items():
        ndims = array.ndim
        if ndims == 2:
            _fields[field_name] = array[:,unique_indices]
        elif ndims == 3:
            _fields[field_name] = array[:,unique_indices,:]
        else:
            # Shouldn't be able to happen
            raise ValueError(
                f"field {field_name} has an unexpected number of dimensions ({ndims})")

    return TerraModel(r=_r, lon=_lon, lat=_lat, fields=_fields)


def _is_valid_field_name(field):
    """
    Return True if field is a valid name of a field in a TerraModel.
    """
    return field in _ALL_FIELDS.keys()


def _variable_name_from_field(field):
    """
    Return the netCDF variable name(s) of a field from the TerraModel field name.
    The values returned are tuples.
    """
    return _FIELD_NAME_TO_VARIABLE_NAME[field]

def _field_name_from_variable(field):
    """
    Return the TerraModel field name of a NetCDF file variable name
    """
    return _VARIABLE_NAME_TO_FIELD_NAME[field]

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

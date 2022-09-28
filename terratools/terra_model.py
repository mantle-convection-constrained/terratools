"""
terra_model
===========

The terra_model module provides the TerraModel class.  This holds
the data contained within a single time slice of a TERRA mantle
convection simulation.
"""

import netCDF4
import numpy as np
import re
from sklearn.neighbors import NearestNeighbors

from . import geographic
from . import plot

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
    'vphi': 'Bulk sound velocity (elastic) [km/s]',
    'vp_an': 'P-wave velocity (anelastic) [km/s]',
    'vs_an': 'S-wave velocity (anelastic) [km/s]',
    'vphi_an': 'Bulk sound velocity (anelastic) [km/s]',
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
    'vp': ("Vp",),
    'vs': ("Vs",),
    'vphi': ("V_bulk",),
    'vp_an': ("Vp_anelastic",),
    'vs_an': ("Vs_anelastic",),
    'Density': ("Density",),
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
    Exception type raised when trying to set a field when the dimensions
    do not match the coordinates in the model
    """
    def __init__(self, model, array, name=""):
        self.message = f"Field array {name} has incorrect first two dimensions. " + \
            f"Expected {(model._nlayers, model._npts)}; got {array.shape[0:2]}"
        super().__init__(self.message)


class TerraModel:
    """
    Class holding a TERRA model at a single point in time.

    A TerraModel contains the coordinates of each lateral point, the
    radii of each layer, and zero or more fields at each of these points.

    Fields are either 2D (for scalar fields like temperature) or 3D
    (for multicomponent arrays like flow velocity) NumPy arrays.

    There are two kinds of methods which give you information about
    the contents of a TerraModel:

    1. Methods starting with ``get_`` return a reference to something held
       within the model.  Hence things returned
       by ``get``ters can be modified and the TerraModel instance is also
       modified.  For example, ``get_field("vp")`` returns the array containing
       the P-wave velocity for the whole model, and you can update any values
       within it.  You should **not** change the shape or length of any arrays
       returned from ``get_`` methods, as this will make the internal state
       of a TerraModel inconsistent.

    2. Methods which are simply a noun (like ``number_of_compositions``)
       return a value and this cannot be used to change the model.

    Model coordinates
    -----------------
    TerraModels have a number of layers, and within each layer there are
    a number of lateral points, of which there are always the same number.
    Points on each layer at the same point index always have the same
    coordinates.

    Use ``TerraModel.get_lateral_points()`` to obtain the longitude and
    latitude of the lateral points.

    ``TerraModel.get_radii()`` returns instead the radii (in km) of
    each of the layers in the model.

    Model fields
    ------------
    Models fields (as returned by ``TerraModel.get_field``) are simply
    NumPy arrays.  The first index is the layer index, and the second
    is the point index.

    As an example, if for a model ``m`` you obtain the temperature field
    with a call ``temp = m.get_field("t")``, the lateral coordinates
    ``lon, lat = m.get_lateral_points()`` and the radii ``r = m.get_radii()``,
    the temperature at ``lon[ip], lat[ip]`` and radius ``r[ir]`` is given
    by ``temp[ir,ip]``.

    Nearest neighbours
    ------------------
    The nearest lateral point index to an arbitrary geographic location
    can be obtained by ``m.nearest_index(lon, lat)``, whilst the nearest
    ``n`` neighbours can be obtained with ``m.nearest_indices(lon, lat, n)``.
    ``m.nearest_neighbours(lon, lat, n)`` also returns the distances to
    each near-neighbour.
    """

    def __init__(self, lon, lat, r,
            fields={}, c_histogram_names=None, lookup_tables=None):
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

        :param lon: Position in longitude of lateral points (degrees)
        :param lat: Position in latitude of lateral points (degrees).  lon and
            lat must be the same length
        :param r: Radius of nodes in radial direction.  Must increase monotonically.
        :param fields: dict whose keys are the names of field, and whose
            values are numpy arrays of dimension (nlayers, npts) for
            scalar fields, and (nlayers, npts, ncomps) for a field
            with ``ncomps`` components at each point
        :param c_histogram_names: The names of each composition of the
            composition histogram, passed as a ``c_hist`` field
        :param lookup_tables: An iterable of SeismicLookupTable corresponding
            to the number of compositions for this model
        """

        nlayers = len(r)
        self._nlayers = nlayers

        npts = len(lon)
        if len(lat) != npts:
            raise ValueError("length of lon and lat must be the same")

        # Number of lateral points
        self._npts = npts
        # Longitude and latitude in degrees of lateral points
        self._lon = np.array(lon, dtype=COORDINATE_TYPE)
        self._lat = np.array(lat, dtype=COORDINATE_TYPE)
        # Radius of each layer
        self._radius = np.array(r, dtype=COORDINATE_TYPE)

        # Check for monotonicity of radius
        if not np.all(self._radius[1:] - self._radius[:-1] > 0):
            raise ValueError("radii must increase or decrease monotonically")

        # Fit a nearest-neighbour search tree
        self._knn_tree = _fit_nn_tree(self._lon, self._lat)

        # The names of the compositions if using a composition histogram approach
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
                    self._check_field_shape(val, key, scalar=False)
                else:
                    if array.shape != (nlayers, npts, expected_ncomps):
                        raise FieldDimensionError(self, val, key)

            else:
                raise FieldNameError(key)

            self.set_field(key, array)


    def __repr__(self):
        return f"""TerraModel:
           number of radii: {self._nlayers}
             radius limits: {(np.min(self._radius), np.max(self._radius))}
  number of lateral points: {self._npts}
                    fields: {[name for name in self.field_names()]}
         composition names: {self.get_composition_names()}"""


    def field_names(self):
        """
        Return the names of the fields present in a TerraModel.

        :returns: list of the names of the fields present.
        """
        return self._fields.keys()


    def evaluate(self, lon, lat, r, field, method="triangle", depth=False):
        """
        Evaluate the value of field at radius r km, latitude lat degrees
        and longitude lon degrees.

        Note that if r is below the bottom of the model the value at the
        lowest layer is returned; and likewise if r is above the top
        of the model, the value at the highest layer is returned.

        There are two evaluation methods:

        1. 'triangle': Finds the triangle surrounding the point of
           interest and performs interpolation between the values at
           each vertex of the triangle
        2. 'nearest': Just returns the value of the closest point of
           the TerraModel

        In either case, linear interpolation is performed between the two
        surrounding layers of the model.

        :param lon: Longitude in degrees of point of interest
        :param lat: Latitude in degrees of points of interest
        :param r: Radius in km of point of interest
        :param field: String giving the name of the field of interest
        :param method: String giving the name of the evaluation method; a
            choice of 'triangle' (default) or 'nearest'.
        :param depth: If True, treat r as a depth rather than a radius
        :returns: value of the field at that point
        """
        _check_field_name(field)
        self._check_has_field(field)

        if method not in ("triangle", "nearest"):
            raise ValueError("method must be one of 'triangle' or 'nearest'")

        radii = self.get_radii()

        if depth:
            r = radii[-1] - r

        lons, lats = self.get_lateral_points()
        array = self.get_field(field)

        # Find bounding layers
        ilayer1, ilayer2 = _bounding_indices(r, radii)
        r1, r2 = radii[ilayer1], radii[ilayer2]

        if method == "triangle":
            # Get three nearest points, which should be the surrounding
            # triangle
            idx1, idx2, idx3 = self.nearest_indices(lon, lat, 3)

            # For the two layers, laterally interpolate the field
            # Note that this relies on NumPy's convention on indexing, where
            # indexing an array with fewer indices than dimensions acts
            # as if the missing, trailing dimensions were indexed with `:`.
            # (E.g., `np.ones((1,2,3))[0,0]` is `[1., 1., 1.]`.)
            val_layer1 = geographic.triangle_interpolation(
                lon, lat,
                lons[idx1], lats[idx1], array[ilayer1,idx1],
                lons[idx2], lats[idx2], array[ilayer1,idx2],
                lons[idx3], lats[idx3], array[ilayer1,idx3]
            )

            if ilayer1 == ilayer2:
                return val_layer1

            val_layer2 = geographic.triangle_interpolation(
                lon, lat,
                lons[idx1], lats[idx1], array[ilayer2,idx1],
                lons[idx2], lats[idx2], array[ilayer2,idx2],
                lons[idx3], lats[idx3], array[ilayer2,idx3]
            )

        elif method == "nearest":
            index = self.nearest_index(lon, lat)
            val_layer1 = array[ilayer1,index]

            if ilayer1 == ilayer2:
                return val_layer1

            val_layer2 = array[ilayer2,index]

        # Linear interpolation between the adjacent layers
        value = ((r2 - r)*val_layer1 + (r - r1)*val_layer2)/(r2 - r1)

        return value


    def set_field(self, field, values):
        """
        Create a new field within a TerraModel from a predefined array,
        replacing any existing field data.

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
        :returns: the new field
        """
        _check_field_name(name)
        if ncomps is not None and ncomps < 1:
            raise ValueError(f"ncomps cannot be less than 1 (is {ncomps})")

        is_vector = _is_vector_field(name)
        ncomps_expected = _expected_vector_field_ncomps(name) if is_vector else None

        nlayers = self._nlayers
        npts = self._npts

        if is_vector:
            # For a vector field, either we have not set ncomps and we use the
            # expected number, or there is no expected number and we use what is
            # passed in.  Fields without an expected number of components
            # cannot be made unless ncomps is set.
            if ncomps_expected is None:
                if ncomps is None:
                    raise ValueError(f"Field {name} has no expected number " +
                        "of components, so ncomps must be passed")
                self.set_field(name, np.zeros((nlayers, npts), dtype=VALUE_TYPE))
            else:
                if ncomps is not None:
                    if ncomps != ncomps_expected:
                        raise ValueError(f"Field {name} should have " +
                            f"{ncomps_expected} fields, but {ncomps} requested")
                else:
                    ncomps = ncomps_expected

            self.set_field(name,
                np.zeros((nlayers, npts, ncomps), dtype=VALUE_TYPE))

        else:
            # Scalar field; should not ask for ncomps at all
            if ncomps is not None:
                raise ValueError(f"Scalar field {name} cannot have {ncomps} components")
            self.set_field(name, np.zeros((nlayers, npts), dtype=VALUE_TYPE))

        return self.get_field(name)


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
        raise an error, and likewise raise an error if the array is
        not rank-2 or rank-3.

        :raises: FieldDimensionError
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


    def get_composition_names(self):
        """
        If a model contains a composition histogram field ('c_hist'),
        return the names of the compositions; otherwise return None.

        :returns: list of composition names
        """
        if self.has_field("c_hist"):
            return self._c_hist_names
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


    def nearest_index(self, lon, lat):
        """
        Return the index or indices of the lateral point(s) nearest to the
        one or more points supplied.  lon and lat may either both be a scalar or
        both an array of points; behaviour is undefined if a mix is
        provided.

        :param lon: Longitude of point(s) of interest (degrees)
        :param lat: Latitude of point(s) of interest (degrees)
        :returns: the index or indices of the nearest lateral point.
            This is a scalar for scalar input, and an array for array input.
        """
        scalar_input = False
        if np.isscalar(lon) and np.isscalar(lat):
            scalar_input = True

        indices = self.nearest_indices(lon, lat, 1)
        if scalar_input:
            return indices[0]
        else:
            return np.array([idx[0] for idx in indices])


    def nearest_indices(self, lon, lat, n):
        """
        Return the indices of the lateral point(s) nearest to the
        one or more points supplied.  lon and lat may either both be a scalar or
        both an array of points; behaviour is undefined if a mix is
        provided.

        :param lon: Longitude of point(s) of interest (degrees)
        :param lat: Latitude of point(s) of interest (degrees)
        :param n: Number of nearest neighbours to find
        :returns: the indices of the nearest n lateral points.
            This is vector for scalar input, and a vector of vectors for array
            input.
        """
        if n < 1:
            raise ValueError("n must be 1 or more")

        scalar_input = False
        if np.isscalar(lon) and np.isscalar(lat):
            scalar_input = True

        indices, _ = self.nearest_neighbors(lon, lat, n)

        return indices


    def nearest_neighbors(self, lon, lat, n):
        """
        Return the indices of the lateral point(s) nearest to the
        one or more points supplied, and the distances from the test point
        to each point.  lon and lat may either both be a scalar or
        both an array of points; behaviour is undefined if a mix is
        provided.

        Distances are in radians about the centre of the sphere; to
        convert to great circle distances, multiply by the radius of
        interest.

        :param lon: Longitude of point(s) of interest (degrees)
        :param lat: Latitude of point(s) of interest (degrees)
        :param n: Number of nearest neighbours to find
        :returns: (indices, distances), where the first item contains the
            indices of the nearest n lateral points and the second item gives
            the distances in radians about the centre on the sphere on
            which the points all lie.
            These are vectors for scalar input, and a vector of vectors for array
            input.
        """
        if n < 1:
            raise ValueError("n must be 1 or more")

        scalar_input = False

        if np.isscalar(lon) and np.isscalar(lat):
            scalar_input = True
            lon = np.array([lon])
            lat = np.array([lat])
        elif len(lon) != len(lat):
            raise ValueError("lon and lat must be the same length")

        lon_radians = np.radians(lon)
        lat_radians = np.radians(lat)
        coords = np.array([[lat, lon] for lon, lat in zip(lon_radians, lat_radians)])
        distances, indices = self._knn_tree.kneighbors(coords, n_neighbors=n)

        if scalar_input:
            return indices[0], distances[0]
        else:
            return indices, distances


    def nearest_layer(self, radius, depth=False):
        """
        Find the layer nearest to the given radius.

        :param radius: Radius of interest in km.
        :param depth: If True, treat input radius as a depth instead,
            and return index and depth rather than index and radius.
        :returns: layer index and radius of layer in km if depth is False
            (the default); otherwise return layer index and depth in km
        """
        radii = self.get_radii()
        surface_radius = radii[-1]
        if depth:
            radius = surface_radius - radius

        index = _nearest_index(radius, radii)

        if depth:
            return index, surface_radius - radii[index]
        else:
            return index, radii[index]


    def plot_layer(self, field, radius=None, index=None, depth=False,
            delta=None, extent=(-180, 180, -90, 90), method="nearest", show=True):
        """
        Create a heatmap of the values of a particular field at the model
        layer nearest to ``radius`` km.

        :param field: Name of field of interest
        :param radius: Radius in km at which to show map.  The nearest
            model layer to this radius is shown.
        :param index: Rather than using a certain radius, plot the
            field exactly at a layer index
        :param depth: If True, interpret the radius as a depth instead
        :param delta: Grid spacing of plot in degrees
        :param extent: Tuple giving the longitude and latitude extent of
            plot, in the form (min_lon, max_lon, min_lat, max_lat), all
            in degrees
        :param method: May be one of: "nearest" (plot nearest value to each
            plot grid point); or "mean" (mean value in each pixel)
        :param show: If True (the default), show the plot
        :returns: figure and axis handles
        """
        if radius is None and index is None:
            raise ValueError("Either radius or index must be given")
        if index is None:
            layer_index, layer_radius = self.nearest_layer(radius, depth)
        else:
            radii = self.get_radii()
            nlayers = len(radii)
            if index < 0 or index >= nlayers:
                raise ValueError(f"index must be between 0 and {nlayers}")

            layer_index = index
            layer_radius = radii[index]

        lon, lat = self.get_lateral_points()
        values = self.get_field(field)[layer_index]
        label = _SCALAR_FIELDS[field]

        fig, ax = plot.layer_grid(lon, lat, layer_radius, values,
            delta=delta, extent=extent, label=label)

        if depth:
            ax.set_title(f"Depth {int(layer_radius)} km")

        if show:
            fig.show()

        return fig, ax


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
    _c_hist_names = None

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

                # List of variables to read
                vars_to_read = _variable_names_from_field(field_name)
                ncomps = len(vars_to_read)
                # Convert field names to composition names
                _c_hist_names = [re.sub("Frac$", "", s).lower() for s in vars_to_read]

                if field_name not in _fields.keys():
                    _fields[field_name] = np.empty(
                        (nlayers, npts_total, ncomps), dtype=VALUE_TYPE)

                for (i, local_var) in enumerate(vars_to_read):
                    _fields["c_hist"][:,npts_range,i] = nc[local_var][:]

        nc.close()
        npts_pointer += npts

    # Check for need to sort points in increasing radius
    must_flip_radii = len(_r) > 1 and _r[1] < _r[0]
    if must_flip_radii:
        _r = np.flip(_r)

    # Remove duplicate points
    _, unique_indices = np.unique(np.stack((_lon, _lat)), axis=1, return_index=True)
    # np.unique returns indices which sort the values, so sort the
    # indices to preserve the original point order, except the duplicates
    unique_indices = np.sort(unique_indices)
    _lon = _lon[unique_indices]
    _lat = _lat[unique_indices]

    for (field_name, array) in _fields.items():
        ndims = array.ndim
        if ndims == 2:
            if must_flip_radii:
                _fields[field_name] = array[::-1,unique_indices]
            else:
                _fields[field_name] = array[:,unique_indices]
        elif ndims == 3:
            if must_flip_radii:
                _fields[field_name] = array[::-1,unique_indices,:]
            else:
                _fields[field_name] = array[:,unique_indices,:]
        else:
            # Shouldn't be able to happen
            raise ValueError(
                f"field {field_name} has an unexpected number of dimensions ({ndims})")

    return TerraModel(r=_r, lon=_lon, lat=_lat, fields=_fields,
        c_histogram_names=_c_hist_names)


def _is_valid_field_name(field):
    """
    Return True if field is a valid name of a field in a TerraModel.
    """
    return field in _ALL_FIELDS.keys()


def _variable_names_from_field(field):
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

def _fit_nn_tree(lon, lat):
    """
    Fit a nearest neighbour lookup tree to a set of longitude and
    latitude points on the surface of a sphere, where the input
    is in degrees.

    Rough timing suggests that using the geographic coordinates and
    Haversine distance takes ~440 µs per lookup, whereas using
    Cartesian coordinates and a Euclidian distance leads to
    lookups taking ~15 ms.

    # FIXME: Properly test which distance metrics and tree types
    #        are fastest.
    """
    lon_radians = np.radians(lon)
    lat_radians = np.radians(lat)
    coords = np.array([[lat, lon] for lat, lon in zip(lat_radians, lon_radians)])
    tree = NearestNeighbors(n_neighbors=1, metric="haversine").fit(coords)
    return tree

def _nearest_index(value, values):
    """
    Return the index of the nearest value in ``values``.  If
    ``value`` is smaller or greater than all ``values``, respectively
    return the minimum or maximum.

    Requires that ``values`` is sorted in increasing order.
    """
    nvals = len(values)
    index = np.searchsorted(values, value)

    # Outside the range
    if index == nvals:
        return nvals - 1
    elif index == 0:
        return index

    # Need to decide whether the one above or below is closer
    if value - values[index-1] > values[index] - value:
        return index
    else:
        return index - 1

def _bounding_indices(value, values):
    """
    Return the indices of ``values`` ``i1`` and ``i2`` where
    ``values[i1] <= value < values[i2]``.

    If ``value`` is less or greater than the smallest and largest
    values, respectively return the same index for both.
    In other words, the returned index is 0 when value is below all
    values, and ``len(values)-1`` when it is above.

    If value is exactly the same as one of the values, return the
    corresponding index twice also.

    Requires that ``values`` is sorted in increasing order.
    """
    nvals = len(values)
    index = np.searchsorted(values, value)

    # We are above or below the range of values
    if index == 0:
        return index, index
    elif index == nvals:
        return index - 1, index - 1
    elif values[index] == value:
        return index, index
    else:
        return index - 1, index

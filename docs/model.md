The main interface for dealing with TERRA models is the `TerraModel` class
which represents an instance in time within a TERRA simulation.

## `terratools.terra_model`

The `terra_model` module within `terratools` contains all the functionality
related to using `TerraModel`s.  Load it with
`from terratools import terra_model`.

## Reading NetCDF files
`terratools` defines a NetCDF [file format](file_formats.md), which can be
read using [terra_model.read_netcdf][terratools.terra_model.read_netcdf].

`terra_model.read_netcdf` will read these and return a `TerraModel` object.

## `TerraModel`
The `TerraModel` class holds the information about a mantle convection
simulation at a single time snapshot.  More details about `TerraModel`s can
be found in the [class docstring][terratools.terra_model.TerraModel].

At a minimum, a `TerraModel` instance will usually contain information about:
- the temperature field $T$ at all points (called `"t"` within `terratools`); and
- the flow field $u$ at all points (called `"u_xyz"` if in the global Cartesian
  [reference frame](reference_frames.md)).

### Fields
Depending on the contents of the model file, it may contain some of the
following fields:

* Scalar fields:
  - `"t"`: Temperature field [K]
  - `"c"`: Scalar composition field [unitless]
  - `"p"`: Pressure field [GPa]
  - `"vp"`: P-wave velocity (elastic) [km/s]
  - `"vs"`: S-wave velocity (elastic) [km/s]
  - `"vphi"`: Bulk sound velocity (elastic) [km/s]
  - `"vp_an"`: P-wave velocity (anelastic) [km/s]
  - `"vs_an"`: S-wave velocity (anelastic) [km/s]
  - `"vphi_an"`: Bulk sound velocity (anelastic) [km/s]
  - `"density"`: Density [g/cm^3]
  - `"qp"`: P-wave quality factor [unitless]
  - `"qs"`: S-wave quality factor [unitless]
  - `"visc"`: Viscosity [Pa s]
* Vector fields:
  - `"u_xyz"`: Flow field in Cartesian coordinates (three components) [m/s]
  - `"u_enu"`: Flow field in local geographic coordinates (three components) [m/s]
  - `"c_hist"`: Composition histogram [unitles]

### Field coordinates
All fields are defined in space by two indices:

- The first index gives the layer number, starting from `0` at the lowest part of
  the model.  Layer radii can be given using
  [terra_model.TerraModel.get_radii][terratools.terra_model.TerraModel.get_radii].
  For a model `m`, the `i`th layer is therfore at radius `m.get_radii()[i]`.
- The second index gives the lateral
  ([global geographic](reference_frames#Global-geographic)) position on the unit
  sphere of the point.
  [terra_model.TerraModel.get_lateral_points][terratools.terra_model.TerraModel.get_lateral_points]
  returns a tuple of `lon, lat`, where `lon` and `lat` are both the global
  geographic coordinates of each lateral point.  Therefore the coordinates of
  the `j`th lateral point for model `m` are given by
  
  ```python
  lon, lat = m.get_lateral_points()
  lon[j], lat[j]
  ```

In this way, what is really a 3D field of positions is represented as only a
2D array of values.

Scalar fields are 2D arrays, whose first index is that of the layer, and the second
is that of the lateral point.  So to get the value of the temperature $T$ at
the bottom layer (index `0`) and the last lateral point for model `m`, you would do
`m.get_field("t")[0,-1]`.

Vector fields are 3D arrays.  Their first two indices are the same as for scalar
fields, but the final index gives the component of the field at that position.
For example, to get the radial components of flow (local up) for the 10th layer,
do `m.get_field("u_enu")[9,:,2]`, noting that the 'up' component is at the third
index of the last dimension.

The fields returned by `get_field` are what is stored inside the `TerraModel`
and editing the arrays (in fact NumPy arrays) will update the model.

Fields can be created or replaced with
[m.new_field][terratools.terra_model.TerraModel.new_field] and
[m.set_field][terratools.terra_model.TerraModel.set_field], and the fields
currently present in a model can be found with
[m.field_names][terratools.terra_model.TerraModel.field_names].

### Field evaluation
One of the most common requirements of a `TerraModel` `m` is to find out what the
value of some field is at an arbitrary point within the model.  This is done
with [m.evaluate][terratools.terra_model.TerraModel.evaluate].  Evaluation
can be done by finding the nearest neighbour to the point of interest, or using
interpolation.

### Plotting
Fields can be plotted in various ways:

- Cross sections can be made with [TerraModel.plot_section][terratools.terra_model.TerraModel.plot_section]
- Depth slices can be made with [TerraModel.plot_layer][terratools.terra_model.TerraModel.plot_layer].
- Spherical harmonic power by degree across all layers can be plotted with
  [TerraModel.plot_spectral_heterogeneity][terratools.terra_model.TerraModel.plot_spectral_heterogeneity]

### Spherical harmonics
Spherical harmonics can be analysed for fields with the
[m.calc_spherical_harmonics][terratools.terra_model.TerraModel.calc_spherical_harmonics]
function.  These can then be retrieved using
[m.get_spherical_harmonics][terratools.terra_model.TerraModel.get_spherical_harmonics].

## `TerraModelLayer`

`TerraModelLayer` is a subclass of the `TerraModel` class. This holds the information about a single
layer in a simulation, typically something at the surface, such as radial stresses, or at the lowermost
layer, such as CMB heat flux. The `TerraModelLayer` class has all of the functionality of the `TerraModel`
class except for the `add_adiabat`, `get_1d_profile`, and `plot_section` methods which, if called, will
cause an exception to be raised.

To read in Terra Layer files to a `TerraModelLayer` object simply pass their paths into 
`terra_model.read_netcdf`, as you would to create and`TerraModel` object, and `terratools` will recognise
that they are for a single layer. 



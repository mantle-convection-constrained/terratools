This page describes the different reference frames used by TERRA and
`terratools`.

## Accessing individual components
In both cases below, individual elements in fields are accessed

## Global geographic frame
Most commonly, you will interact with points based on their geographic
position.  This is given in terms of
- the longitude `lon`, in °;
- the latitude `lat`, in °; and
- the radius `r`, in km.

Within `terratools`, when geographic coordinates are used as arguments
or return values from functions, they will be in the order `lon, lat, r`.

(Many functions accept the `depth=True` keyword argument which converts the
value of `r` given (or returned) into a depth below the model surface.)

The lateral points of the TERRA model mesh are given by
[terra_model.TerraModel.get_lateral_points][terratools.terra_model.TerraModel.get_lateral_points],
while [terra_model.TerraModel.get_radii][terratools.terra_model.TerraModel.get_radii]
returns the radii of the layers in the mesh.

## Global Cartesian frame
TERRA internally uses the global Cartesian reference frame, whose origin
lies at the centre of the Earth or spherical body being modelled.  In this
system, the following directions are defined:
- $x$ is from the centre towards the point where the equator and prime
  meridian runs (i.e., a longitude of 0° and latitude of 0°).
- $y$ is from the centre towards 90° east, through the equator (longitude
  90° and latitude 0°).
- $z$ is from the centre towards the north pole (latitude 90°).

Fields in a [TerraModel][terratools.terra_model.TerraModel] such as `"u_xyz"`
use this reference frame.

## Local geographic reference frame
Some fields such as `"u_enu"` use a local reference frame defined by the
directions:
- `e`: local east direction
- `n`: local north direction
- `u`: radial direction, pointing away from the centre of the Earth

![Reference frames used by terratools](images/reference_frames.svg)

# %% [markdown]
"""
# example_evaluate_points

Here we show how to extract the values of `TerraModel` fields at
individual points and sets of points.
"""

# %% [markdown]
"""
We begin by importing the required things:
"""

# %%
import terratools
from terratools.terra_model import read_netcdf
from terratools.example_data import example_terra_model
import glob

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
"""
First we download and read in the example mantle convection model:
"""

# %%
# Download and cache model
path = example_terra_model()

# read in the model
model = read_netcdf(glob.glob(path))


# %% [markdown]
"""
At this point we are able to extract points at arbitrary positions.

Let's first find a single point, at about the coordinates of Cardiff
and at 200 km below the surface:
"""

# %%
model.evaluate(lon=-3.1789, lat=51.48772, r=6170, field="t")

# %% [markdown]
"""
We can also use the `depth` keyword argument to interpret `r` as a depth
in km rather than a radius in km, getting the same answer.
"""

# %%
model.evaluate(-3.1789, 51.48772, 200, "t", depth=True)

# %% [markdown]
"""
We can also ask for multiple points at once, providing that `lon`, `lat` and `r`
are all lists or arrays of the same length.  For example, we can get a profile
along the Greenwich meridian 200 km above the core-mantle boundary:
"""

# %%
profile_lats = np.arange(-90, 90)
profile_lons = np.zeros_like(profile_lats)
profile_radii = 3680 * np.ones(len(profile_lats))
profile_ts = model.evaluate(profile_lons, profile_lats, profile_radii, "t")

plt.plot(profile_lats, profile_ts)
plt.xlabel("Latitude / °")
plt.ylabel("Temperature / K")
plt.show()

# %% [markdown]
"""
Here it is worth noting that there are two different methods by which one
can evaluate the points:
1. `"triangle"` (the default), which interpolates between neighbouring points, and
2. `"nearest"`, which just takes the nearest grid point to that of interest.
   Because of this difference, it is quicker to evaluate by nearest neighbour only.

Let's look at the difference between them for this random model:
"""

# %%
profile_ts_nearest = model.evaluate(
    profile_lons, profile_lats, profile_radii, "t", method="nearest"
)

plt.plot(profile_lats, profile_ts, label="Method: 'triangle'")
plt.plot(profile_lats, profile_ts_nearest, label="Method: 'nearest'")
plt.xlabel("Latitude / °")
plt.ylabel("Temperature / K")
plt.legend()
plt.show()

# %% [markdown]
"""
You can see that the `"triangle"` method interpolates between points, whereas
`"nearest"` gives a stepped appearance.
"""

# %% [markdown]
"""
Finally, just to visualise the interpolation across the field, we take
a set of points within a window and plot the field, with the points
scattered on top, defining a function to do the heavy lifting:
"""


# %%
def plot_field(
    model,
    lon_lims=(-20, 20),
    lat_lims=(-20, 20),
    nlons=100,
    nlats=100,
    method="triangle",
):
    """
    Plot the field ni `model` within the longitude and latitude limits
    set by `lon_lims` and `lat_lims`, respectively.  The number of points
    evaluated in each direction is given by `nlons` and `nlats`, while
    `method` is passed to `model.evaluate`.
    """
    model_lons, model_lats = model.get_lateral_points()
    inds = np.flatnonzero(
        (lon_lims[0] <= model_lons)
        * (model_lons <= lon_lims[1])
        * (lat_lims[0] <= model_lats)
        * (model_lats <= lat_lims[1])
    )

    nlons = nlats = 100
    interp_lons = np.linspace(lon_lims[0], lon_lims[1], nlons)
    interp_lats = np.linspace(lat_lims[0], lat_lims[1], nlats)
    mesh_lons, mesh_lats = np.meshgrid(interp_lons, interp_lats)
    interp_radii = 3680 * np.ones(len(mesh_lons.flat))
    interp_ts = np.reshape(
        model.evaluate(
            mesh_lons.flat, mesh_lats.flat, interp_radii, "t", method=method
        ),
        (nlons, nlats),
    )

    plt.contourf(mesh_lons, mesh_lats, interp_ts, levels=100)
    plt.colorbar(label="Temperature / K")
    plt.scatter(model_lons[inds], model_lats[inds])
    plt.xlabel("Longitude / °")
    plt.ylabel("Latitude / °")
    plt.gca().set_aspect("equal")
    plt.show()


plot_field(model)

# %% [markdown]
"""
Contrast that with when instead we only take the nearest neighbours:
"""

# %%
plot_field(model, method="nearest")

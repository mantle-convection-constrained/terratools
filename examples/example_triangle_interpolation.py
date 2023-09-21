# %% [markdown]
"""
This example shows how interpolation between points on a
spherical triangle appears.

The first function (`example_three_points`) creates three
arbitrary points on the sphere and demonstrates that the
values taken between the corners vary linearly (in great
circle distance).

The second (`example_terra`) uses a subset of the TERRA mesh
at ~22 km spacing and applies a smooth function, which we then
interpolate between, demonstrating that the field is correctly
interpolated.
"""


# %% [markdown]
"""
We begin by importing the required things:
"""

# %%

import matplotlib.pyplot as plt
import numpy as np

from terratools.terra_model import read_netcdf
from terratools.example_data import example_terra_model

from terratools.geographic import triangle_interpolation

# %% [markdown]
"""
Next define a function to create a grid of points.
"""


def grid(lon1, lon2, lat1, lat2, delta):
    lons = np.arange(lon1, lon2, delta)
    lats = np.arange(lat1, lat2, delta)
    return lons, lats


# %% [markdown]
"""
Download and read in the example model.
"""
# %%
# get the path to the downloaded model
path = example_terra_model()

# read in the model
model = read_netcdf([path])

# %% [markdown]
"""
Define a grid of points to perform the interpolation on.
Note that we extract the points from the model itself using
model get_lateral_points().
"""

# %%
# extract points
lon, lat = model.get_lateral_points()

# create grid
delta = 0.1
gridlon, gridlat = grid(-2, 2, -2, 2, delta)
nlon, nlat = len(gridlon), len(gridlat)

# %% [markdown]
"""
Perform the interpolation using model.evaluate()
with the triangle method at 5000 km radius.
"""
# %%
# loop over the grid of points

interpolated_values = np.empty((nlon, nlat))
for i in range(nlon):
    for j in range(nlat):
        interpolated_values[i, j] = model.evaluate(
            gridlon[i], gridlat[j], 5000, "t", method="triangle"
        )

# %% [markdown]
"""
Plot!
"""
# %%
plt.pcolormesh(gridlon, gridlat, np.transpose(interpolated_values))
plt.scatter(lon, lat, edgecolors="white", s=0.5)
plt.colorbar()
plt.show()

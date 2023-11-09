# %% [markdown]
"""
# example_plot_map

In this example, we show how to read in terra output in NetCDF format
and plot the temperature and bulk composition field at a radius of interest.
"""

# %% [markdown]
"""
Let's import all the necessary python objects.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from terratools.terra_model import read_netcdf
from terratools.example_data import example_terra_model
import cartopy
import cartopy.crs as ccrs

# %% [markdown]
"""
Now let's download and read in the example mantle convection model netcdf file
 and print some summary information.
"""

# %%
# Read in the netcdf files.
# Download and cache model
path = example_terra_model()

# read in the model
model = read_netcdf([path])


# %% [markdown]
"""
Plotting depth slices in terratools is very easy. First we show a basic plot for the temperature at 2800 km depth.
"""

# %%

# Note we set depth=True to define the depth as 2800
fig, ax = model.plot_layer(field="t", radius=2800, depth=True, show=False)
fig.set_size_inches(8, 6)
ax.set_title("Temperature field at 2800 km depth")
plt.show()

# %% [markdown]
"""
We can do the same thing with other scalar fields such as the bulk composition.
"""

# %%

# Add a bulk composition field to the model.
model.calc_bulk_composition()

# Note bulk composition is in the "c" field.
fig, ax = model.plot_layer(field="c", radius=2800, depth=True, show=False)
fig.set_size_inches(8, 6)
ax.set_title("Bulk composition at 2800 km depth")
plt.show()


# %% [markdown]
"""
Rather than defining a radius, we can give an index for the layer we want to plot.
"""

# %%

fig, ax = model.plot_layer(field="t", index=10, show=False)
fig.set_size_inches(8, 6)
ax.set_title("Temperature at the index 10.")
plt.show()

# %% [markdown]
"""
We can also change the sampling resolution by varying the delta argument.
"""

# %%

# plot with intervals of 5 degrees on longitude and latitude.
fig, ax = model.plot_layer(field="t", radius=2800, depth=True, delta=5, show=False)
fig.set_size_inches(8, 6)
ax.set_title("Temperature on a 5$^{\circ}$ grid at 2800 km depth.")
plt.show()


# %% [markdown]
"""
Rather than a global plot we can specify a local region to plot.
"""

# %%

# define region of interest
min_lo = -30
max_lo = 30
min_la = -30
max_la = 30

region = (min_lo, max_lo, min_la, max_la)

fig, ax = model.plot_layer(
    field="t", radius=2800, depth=True, show=False, extent=region
)
fig.set_size_inches(8, 6)
ax.set_title("Temperature in a small region at 2800 km depth.")
# ax.set_extent(region, ccrs.PlateCarree())
plt.show()

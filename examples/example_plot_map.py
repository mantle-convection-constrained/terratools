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
import glob
from terratools import terra_model as tm
import cartopy 
import cartopy.crs as ccrs 

# %% [markdown]
"""
Now lets read in the terra output netcdf files and print some summary information. 
"""

# %%
# Read in the netcdf files. 
m = tm.read_netcdf(glob.glob('/Users/earjwara/work/terra_models/muller_bb_visc1_m_bb_044_tvisc1_prim_NC_037/NC_037/*'))
print(m)

# %% [markdown]
"""
Plotting depth slices in terratools is very easy. First we show a basic plot for the temperature at 2800 km depth.
"""

# %%

# Note we set depth=True to define the depth as 2800
fig, ax = m.plot_layer(field = 't', radius = 2800, depth=True, show=False)
fig.set_size_inches(8, 6)
ax.set_title('Temperature field at 2800 km depth')
plt.show()

# %% [markdown]
"""
We can do the same thing with other scalar fields such as the bulk composition. 
"""

# %%

# Add a bulk composition field to the model.
m.calc_bulk_composition()

# Note bulk composition is in the "c" field. 
fig, ax = m.plot_layer(field = 'c', radius = 2800, depth=True, show=False)
fig.set_size_inches(8, 6)
ax.set_title('Bulk composition at 2800 km depth')
plt.show()


# %% [markdown]
"""
Rather than defining a radius, we can give an index for the layer we want to plot. 
"""

# %%

fig, ax = m.plot_layer(field = 't', index=0, show=False)
fig.set_size_inches(8, 6)
ax.set_title('Temperature at the highest radius.')
plt.show()

# %% [markdown]
"""
We can also change the sampling resolution by varying the delta argument. 
"""

# %%

# plot with intervals of 5 degrees on longitude and latitude. 
fig, ax = m.plot_layer(field = 't', radius=2800, depth=True, delta=5, show=False)
fig.set_size_inches(8, 6)
ax.set_title('Temperature on a 5$^{\circ}$ grid at 2800 km depth.')
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

fig, ax = m.plot_layer(field = 't', radius=2800, depth=True, show=False, extent=region)
fig.set_size_inches(8, 6)
ax.set_title('Temperature in a small region at 2800 km depth.')
ax.set_extent(region, ccrs.PlateCarree())
plt.show()

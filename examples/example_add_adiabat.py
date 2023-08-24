# %% [markdown]
"""
# example_add_adiabat

This example demonstrates how to read in terra output in NetCDF format 
and add a theoretical adiabat to the temperatures.
"""

# %% [markdown]
"""
Let's import all the necessary python objects. 
"""

# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import glob
from terratools import terra_model as tm
from terratools.terra_model import _calculate_adiabat

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
To see the effect of adding the adiabat, let's extract the mean 1D temperature 
in the model and plot this with radius.
"""

# %%
# Get radii and depths array
radii = m.get_radii()
depths = 6370 - radii

# Get temperature field 
temps = m.get_field('t')

# Calculate the quantiles at each radius
quantile_25 = np.quantile(temps, 0.25, axis=1)
quantile_75 =np.quantile(temps, 0.75, axis=1)

# Extract the mean 1D temperature profile 
temp_profile_no_adiabat = m.mean_1d_profile('t')

# Plot mean temperature with depth 

fig = plt.figure(figsize=(6,8))
ax = fig.add_subplot(111)
ax.plot(temp_profile_no_adiabat, radii, '-o', color='C0')
ax.plot(quantile_25, radii, '-o', color='C0')
ax.plot(quantile_75, radii, '-o', color='C0')
ax.set_xlabel('Mean Temperature (K)')
ax.set_ylabel('Radius (km)')
ax.set_title('Temperature Profile without Adiabat')
plt.show()

# %% [markdown]
"""
In the following cell, we will show what adiabat will be used. 
"""

# %%
adiabat = _calculate_adiabat(depths)

fig = plt.figure(figsize=(6,8))
ax = fig.add_subplot(111)
ax.plot(adiabat, radii, '-o', color='C0')
ax.set_xlabel('Mean Temperature (K)')
ax.set_ylabel('Radius (km)')
ax.set_title('Theoretical Adiabat')
plt.show()

# %% [markdown]
"""
Fortunately, adding an adiabat is straightforward
"""

# %%
# Add adiabat, note this will be done inplace on the TerraModel object
m.add_adiabat()

# Get new temperature field 
temps_adiabat = m.get_field('t')

# Calculate the new quantiles at each radius
quantile_25_adiabat = np.quantile(temps, 0.25, axis=1)
quantile_75_adiabat =np.quantile(temps, 0.75, axis=1)

# Extract the new mean 1D temperature profile 
temp_profile_adiabat = m.mean_1d_profile('t')


fig = plt.figure(figsize=(6,8))
ax = fig.add_subplot(111)
ax.plot(temp_profile_adiabat, radii, '-o', color='C0')
ax.plot(quantile_25_adiabat, radii, '-o', color='C0')
ax.plot(quantile_75_adiabat, radii, '-o', color='C0')
ax.set_xlabel('Mean Temperature (K)')
ax.set_ylabel('Radius (km)')
ax.set_title('Temperature Profile with Adiabat')
plt.show()

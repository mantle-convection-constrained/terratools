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
from terratools.terra_model import read_netcdf
from terratools.example_data import example_terra_model
from terratools.terra_model import _calculate_adiabat

# %% [markdown]
"""
Now let's download and read in the example mantle convection model
netcdf file and print some summary information.
"""

# %%
# Download and cache model
path = example_terra_model()

# read in the model
model = read_netcdf([path])

# %% [markdown]
"""
To see the effect of adding the adiabat, let's extract the mean 1D temperature
in the model and plot this with radius.
"""

# %%
# Get radii and depths array
radii = model.get_radii()
depths = 6370 - radii

# Get temperature field
temps = model.get_field("t")

# Calculate the quantiles at each radius
quantile_25 = np.quantile(temps, 0.25, axis=1)
quantile_75 = np.quantile(temps, 0.75, axis=1)

# Extract the mean radial temperature profile
temp_profile_no_adiabat = model.mean_radial_profile("t")

# Plot mean temperature with depth

fig = plt.figure(figsize=(6, 8))
ax = fig.add_subplot(111)
ax.plot(temp_profile_no_adiabat, radii, "-o", color="C0")
ax.plot(quantile_25, radii, "-o", color="C0")
ax.plot(quantile_75, radii, "-o", color="C0")
ax.set_xlabel("Mean Temperature (K)")
ax.set_ylabel("Radius (km)")
ax.set_title("Temperature Profile without Adiabat")
plt.show()

# %% [markdown]
"""
In the following cell, we will show what adiabat will be used.
"""

# %%
adiabat = _calculate_adiabat(depths)

fig = plt.figure(figsize=(6, 8))
ax = fig.add_subplot(111)
ax.plot(adiabat, radii, "-o", color="C0")
ax.set_xlabel("Mean Temperature (K)")
ax.set_ylabel("Radius (km)")
ax.set_title("Theoretical Adiabat")
plt.show()

# %% [markdown]
"""
Fortunately, adding an adiabat is straightforward
"""

# %%
# Add adiabat, note this will be done inplace on the TerraModel object
model.add_adiabat()

# Get new temperature field
temps_adiabat = model.get_field("t")

# Calculate the new quantiles at each radius
quantile_25_adiabat = np.quantile(temps, 0.25, axis=1)
quantile_75_adiabat = np.quantile(temps, 0.75, axis=1)

# Extract the new mean radial temperature profile
temp_profile_adiabat = model.mean_radial_profile("t")


fig = plt.figure(figsize=(6, 8))
ax = fig.add_subplot(111)
ax.plot(temp_profile_adiabat, radii, "-o", color="C0")
ax.plot(quantile_25_adiabat, radii, "-o", color="C0")
ax.plot(quantile_75_adiabat, radii, "-o", color="C0")
ax.set_xlabel("Mean Temperature (K)")
ax.set_ylabel("Radius (km)")
ax.set_title("Temperature Profile with Adiabat")
plt.show()

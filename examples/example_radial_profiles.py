# %% [markdown]
"""
In this example we will read in files to a TerraModel show how to plot a 1D profile.
First we will import all of the necessary python objects.
"""
# %%

from terratools.example_data import example_terra_model
from terratools.terra_model import read_netcdf
import os
import matplotlib.pyplot as plt


# %% [markdown]
"""
Now let's download and read in the example mantle convection model netcdf file
 and print some summary information.
"""
# %%

# Read in the netcdf files.
# Download and cache model
path = example_terra_model()

# %% [markdown]
"""
We can now read the example data into a TerraModel object.
"""
# %%

model = read_netcdf([path])

# %% [markdown]
"""
Let's inspect the fields that exist within the model
"""
# %%

print(model.field_names())

# %% [markdown]
"""
The 'c_hist' field is a set of histograms which represents the relative abundance of different
lithological end-members. From this the 'bulk compositon' can be calcualted
"""
# %%

model.calc_bulk_composition()
print(model.field_names())

# %% [markdown]
"""
We can see that 'c' has now been added to the list of model fields.

## Get a profile
Let's say we are interested in returning radial profiles of the temperature and bulk compoistion
fields at 90E,-45N. We can get these profiles using:
"""
# %%

lon = 90.0
lat = -45.0
temp_profile = model.radial_profile(lon, lat, "t")
c_profile = model.radial_profile(lon, lat, "c")

# %% [markdown]
"""
The radii are returned with:
"""
# %%

radii = model.get_radii()

# %% [markdown]
"""
And now we can plot the 1D profiles.
"""
# %%

fig, ax = plt.subplots(figsize=(6, 5), ncols=2)

ax[0].plot(temp_profile, radii)
ax[0].set_xlabel("Temperature (K)")
ax[0].set_ylabel("Radius (km)")
ax[0].set_title(f"Radial temperature profile \n at {lon}E {lat}N")

ax[1].plot(c_profile, radii)
ax[1].set_xlabel("Bulk Composition (C)")
ax[1].set_ylabel("Radius (km)")
ax[1].set_title(f"Radial composition profile \n at {lon}E {lat}N")

print(f"Plotted profile at {lon}E, {lat}N.")
plt.show()

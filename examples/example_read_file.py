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
Terra output files are currently written out serially, that is one file per process. Some
users may find it convenient to concatenate files from a dump into a single file using
`ncecat`, a tool which is available through the NetCDF operators package. Terratools also
supports reading concatenated files ``model = read_netcdf(file,cat=True)``.
"""
# %%


# %% [markdown]
"""
##Plot a 1D average profile
We can return a 1D profile of the radial average of a field using:
"""
# %%

temp_mean = model.mean_radial_profile(field="t")

# %% [markdown]
"""
And we can get the radii using:
"""
# %%

radii = model.get_radii()

# %% [markdown]
"""
And now we can plot the 1D profile.
"""
# %%

fig, ax = plt.subplots(figsize=(3, 5))

ax.plot(temp_mean, radii)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Radius (km)")
ax.set_title("1D average temperature profile")

print(f"Read files and plotted profile")

plt.show()

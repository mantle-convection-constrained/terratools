# %% [markdown]
"""
# example_convert

In this example we demonstrate how to convert old Terra output netCDF
files to version 1.0 files. Note that only conversion of .comp netCDF
files is supported as new .seis files write out seismic properties in
absolute terms whereas the old files wrote out seismic properties in
terms of the % difference compared to the radial average, from which
the absolute value cannot be reconstructed without knowledge of the
radial average.


First, lets import all of the necessary modules.
"""
# %%

from terratools import convert_files
from terratools import terra_model
import numpy as np
import os
import netCDF4 as nc4
import glob

# %% [markdown]
"""
For this example, we also need to generate some old files to convert.
In practise you will probably have old files downloaded from a
repository.

# Define function to make old files
"""


# %%
def make_old_file(filename):
    """
    Make an 'old' netcdf file
    """
    nps = 64
    nlayers = 3
    compositions = 2
    c_hist_name = ["BasaltFrac", "LherzFrac"]

    file = nc4.Dataset(filename, mode="w")
    nps_dim = file.createDimension("nps", nps)
    depths_dim = file.createDimension("Depths", nlayers)

    fields = {
        "Temperature": {"units": "K"},
        "Velocity_x": {"units": "Km/s"},
        "Velocity_y": {"units": "Km/s"},
        "Velocity_z": {"units": "Km/s"},
        "BasaltFrac": {"units": ""},
        "LherzFrac": {"units": ""},
    }

    depths_var = file.createVariable("Depths", terra_model.COORDINATE_TYPE, ("Depths",))
    depths_var.units = "Km"

    lon_var = file.createVariable(
        "Longitude", terra_model.COORDINATE_TYPE, ("Depths", "nps")
    )
    lon_var.units = "Degrees"

    lat_var = file.createVariable(
        "Latitude", terra_model.COORDINATE_TYPE, ("Depths", "nps")
    )
    lat_var.units = "Degrees"

    depths_var[:] = np.linspace(0, 2890, nlayers)
    for layer in range(nlayers):
        lon_var[layer, :] = np.linspace(0, 360, nps)
        lat_var[layer, :] = np.linspace(-90, 90, nps)

        for field in fields:
            fields[field]["vals"] = np.random.rand(nlayers, nps).astype(
                terra_model.VALUE_TYPE
            )

    for field in fields:
        this_var = file.createVariable(field, terra_model.VALUE_TYPE, ("Depths", "nps"))
        this_var[:, :] = fields[field]["vals"]
        if len(fields[field]["units"]) > 0:
            this_var.units = fields[field]["units"]

    return file


# %% [markdown]
"""
Make a sub-directory in the current location where we can store the 'old' files.
"""
# %%

directory = "./example_nc_files/"
try:
    os.mkdir(f"{directory}")
except:
    pass

# %% [markdown]
"""
We can then make some 'old' files, we'll just make 3 for this example.
"""
# %%

nfiles = 3
for filen in range(nfiles):
    oldfilepath = os.path.join(directory, f"example_file_{filen}.nc")
    oldfile = make_old_file(oldfilepath)


# %% [markdown]
"""
We can convert to the new file specification by passing their paths in as a list
to convert_files. This process overwrites the old files rather than creating
new ones. We have set test=True here to prevent calling ncks which must be in your
PATH in order to remove old netcdf variables. ncks is available through NCO (NetCDF
Operators - https://nco.sourceforge.net/ ).
"""
# %%

files = glob.glob(f"{directory}/example_file_*.nc")
convert_files.convert(files, test=True)


# %% [markdown]
"""
Lets load one of the converted files and check some of the dimensions and
variables against the old file.
"""
# %%

newfile = nc4.Dataset(f"{directory}/example_file_{nfiles-1}.nc")

print("Old latitude ", oldfile.variables["Latitude"].shape)
print("New latitude ", newfile.variables["latitude"].shape)

# %% [markdown]
"""
Finally we will have a tidy up.
"""
# %%
os.system(f"rm -rf {directory}")

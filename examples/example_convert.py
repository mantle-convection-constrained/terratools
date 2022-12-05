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
"""

# %% [markdown]
"""
First, we import convert_files from terratools and glob
"""

from terratools import convert_files
import glob

# %% [markdown]
"""
For this example, we also need to generate some old files to convert.
In practise you will probably have old files downloaded from a
repository. We can generate some random old files using the test case.
First we import necessary modules.
"""
from tests import test_convert_files
import os
import netCDF4 as nc4

# %% [markdown]
"""
Make a sub-directory in the current location where we can store the 'old' files.
"""

directory = "./old_nc_files/"
try:
    os.mkdir(f"{directory}")
except:
    pass

# %% [markdown]
"""
We can then make some 'old' files, we'll just make 3 for this example.
"""
nfiles = 3
for filen in range(nfiles):
    oldfilepath = os.path.join(directory, f"example_old_file_{filen}.nc")
    oldfile = test_convert_files.make_old_file(oldfilepath)


# %% [markdown]
"""
We can convert to the new file specification by passing their paths in as a list
to convert_files. This process overwrites the old files rather than creating
new ones. We have set test=True here to prevent calling ncks which must be in your
PATH in order to remove old netcdf variables. ncks is available through NCO (NetCDF
Operators - https://nco.sourceforge.net/ ).
"""
files = glob.glob(f"{directory}/example_old_file_*.nc")
convert_files.convert(files, test=True)


# %% [markdown]
"""
Lets load one of the converted files and check some of the dimensions and
variables against the old file.
"""

newfile = nc4.Dataset(f"{directory}/example_old_file_{nfiles-1}.nc")

print("Old latitude ", oldfile.variables["Latitude"].shape)
print("New latitude ", newfile.variables["latitude"].shape)

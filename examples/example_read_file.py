from terratools import convert_files
import glob
from tests import test_convert_files
import os
from terratools import terra_model
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # %% [markdown]
    """
    In this example we will read in files to a TerraModel show how to plot a 1D profile.
    First we must create some example data. We can use inbuilt functions in terratools to
    make some 'old' terra files and convert to the new style which is compatible with terratools.
    """
    # %%

    directory = "./example_nc_files/"
    try:
        os.mkdir(f"{directory}")
    except:
        pass
    nfiles = 4
    for filen in range(nfiles):
        oldfilepath = os.path.join(directory, f"example_file_{filen}.nc")
        oldfile = test_convert_files.make_old_file(oldfilepath)
    files = glob.glob(f"{directory}/example_file_*.nc")
    convert_files.convert(files, test=True)

    # %% [markdown]
    """
    ##Read a list of files
    We have a list of file names `files` which we can read into a TerraModel using:
    """
    # %%

    model = terra_model.read_netcdf(files)

    

    # %% [markdown]
    """
    ##Read a concatenated file
    Terra output files are currently written out serially, that is one file per process. Some
    users may find it convenient to concatenate files from a dump into a single file using
    `ncecat`, a tool which is available through the NetCDF operators package. Terratools also
    supports reading concatenated files ``model = terra_model.read_netcdf(files,cat=True)``


    ##Plot a 1D average profile
    We can return a 1D profile of the radial average of a field using:
    """
    # %%

    temp_mean = model.mean_1d_profile("t")

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

    # %% [markdown]
    """
    Finally we will clean up the example files that we generated.
    """
    # %%

    os.system(f"rm -rf {directory}")

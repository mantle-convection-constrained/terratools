import glob
import os
import terratools
from terratools.terra_model import TerraModel
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


if __name__ == "__main__":

    # %% [markdown]
    """
    In this example we will read in files to a TerraModel show how to plot a 1D profile.
    First we must create some synthetic data in lieu of reading in simulation reuslts.
    """
    # %%

    # Find where terratools is installed
    root_dir = str(Path(terratools.__file__).parent.parent)
    coords_file = f"{root_dir}/examples/data/example_mesh_points.txt"
    coords = np.loadtxt(coords_file)
    lons = coords[:, 0]
    lats = coords[:, 1]
    npts = len(lons)

    nlayers = 40
    radii = np.linspace(3480, 6370, nlayers)

    model = TerraModel(lons, lats, radii)

    # %% [markdown]
    """
    Now we create a temperature field whose values are a function of
    position in space:
    """

    # %%
    lateral_field = np.cos(np.radians(lons)) * np.sin(np.radians(lats)) * 100
    t_field = (
        radii / 10 + np.repeat(np.reshape(lateral_field, (npts, 1)), nlayers, 1)
    ).T
    model.set_field("t", t_field)

    # %% [markdown]
    """
    We can get the radii using:
    """
    # %%

    radii = model.get_radii()

    # %% [markdown]
    """
    Let's say we are interested in returning a radial profile of the temperature field
    at 90E,-45N. We can get this profile using:
    """
    # %%

    lon = 90.0
    lat = -45.0
    temp_profile = model.get_1d_profile("t", lon, lat)

    # %% [markdown]
    """
    And now we can plot the 1D profile.
    """
    # %%

    fig, ax = plt.subplots(figsize=(3, 5))

    ax.plot(temp_profile, radii)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Radius (km)")
    ax.set_title(f"1D temperature profile \n at {lon}E {lat}N")
    print(f"Plotted profile at {lon}E, {lat}N.")
    plt.show()

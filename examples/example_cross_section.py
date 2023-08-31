# %% [markdown]
"""
# example_cross_section

Here we show how to plot a cross-sections through `TerraModel` fields.
"""

# %% [markdown]
"""
We begin by importing the required things:
"""

# %%
import terratools
from terratools.terra_model import TerraModel

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
"""
Normally at this point, you would read in your model by calling
`terratools.terra_model.read_netcdf`, but here instead we create a
synthetic TERRA model just to demonstrate how to
get points out of it.

First we read in the points on which the model is defined:
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
t_field = (radii / 10 + np.repeat(np.reshape(lateral_field, (npts, 1)), nlayers, 1)).T
model.set_field("t", t_field)

# %% [markdown]
"""
At this point we are able to construct cross-sections.

To do so, we must give the coordinates of the starting point as
(longitude, latitude), the azimuth along which the section runs,
and the length of the section, with all quantities in degrees.

Let's look at the model along a 120° south-to-north section going
along the Greenwich meridian:
"""

# %%
model.plot_section("t", lon=0, lat=-60, azimuth=0, distance=120)

# %% [markdown]
"""
You can set parameters such as the spacing of points along the section
(with keyword argument `delta_distance` in °), spacing of points radially
(`delta_radius` in km), limit the depth range with `minradius` and `maxradius`,
choose the point evaluation method (`method`), and so on.

Rather than nearest-neighbour point evaluation (the default), let's use
interpolation, and only plot the field near the surface:
"""

# %%
fig, ax = model.plot_section(
    "t",
    lon=0,
    lat=-30,
    azimuth=0,
    distance=60,
    minradius=6370 - 660,
    delta_radius=10,
    method="triangle",
)

# %% [markdown]
"""
`model.plot_section` returns the figure (`fig` above) and axis handle (`ax`),
so you can save the plot using `fig.savefig("figure.pdf")`, or otherwise modify
the figure and axis using their methods.
Use the keyword argument `show=False` to `model.plot_section` to avoid
displaying the figure, which is useful in batch processing.
"""
import os
import tempfile

file = tempfile.NamedTemporaryFile(suffix=".pdf")
fig.savefig(file.name)
print(f"Saved plot to file")
file.close()

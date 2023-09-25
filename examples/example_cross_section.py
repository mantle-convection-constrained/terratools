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
from terratools.terra_model import read_netcdf
from terratools.example_data import example_terra_model

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
"""
First we download and read in the example mantle convection model:
"""

# %%
# Download and cache model
path = example_terra_model()

# read in the model
model = read_netcdf([path])

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

# %%
import os
import tempfile

file = tempfile.NamedTemporaryFile(suffix=".pdf")
fig.savefig(file.name)
print(f"Saved plot to file")
file.close()

import terratools
from terratools.terra_model import read_netcdf
from terratools.example_data import example_terra_model
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
We can convert the composition histograms of harzburgite, lherzolite and basalt
into a bulk composition scalar field using `model.calc_bulk_composition()`
"""
# %%

model.calc_bulk_composition()

# %% [markdown]
"""
We can perform spherical harmonic decompositions of scalar field. In terratools we make
use of healpy (python wrapper for healpix) to do this. For example we would do:
"""
# %%

model.calc_spherical_harmonics("c")

# %% [markdown]
"""
to calculate the spherical harmonic coefficients and power spectra of the bulk
composition field. We can also include optional arguments `nside` to change the
resolution of the healpix grid (default=2**6), `lmax` to set the maximum spherical
harmonic degree to calculate (default=16), and `savemap` toggle saving of a healpix
map (default=False).
"""
# %%

model.calc_spherical_harmonics("t", nside=2**6, lmax=20, savemap=True)

# %% [markdown]
"""
There is a 'getter' function for accessing the spherical harmonic coefficients etc
"""
# %%

sph_temp = model.get_spherical_harmonics("t")
sph_comp = model.get_spherical_harmonics("c")

# %% [markdown]
"""
We can see, for example, that changing `lmax` when calculating the spherical
harmonics for the temperature field changes the length of 'power_per_l' at
each radial layer.
"""
# %%

print(len(sph_comp[0]["power_per_l"]))
print(len(sph_temp[0]["power_per_l"]))

# %% [markdown]
"""
We can plot the power spectra of the for fields as a function of depth using:
"""
# %%

fig, ax = model.plot_spectral_heterogeneity("c", lyrmin=0, lyrmax=-1, show=False)
plt.show()

# %% [markdown]
"""
We can also plot depth slices of the spectrally filtered fields using:
"""
# %%
model.plot_hp_map("t", index=1, nside=2**5, show=False)

plt.show()

# %% [markdown]
"""
Note that either `index` or `radii` must be passed in determine which layer to plot.
"""

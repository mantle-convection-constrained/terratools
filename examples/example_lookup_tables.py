# %% [markdown]
"""
Here we show how to read in and use lookup tables using terratools.
"""

# %% [markdown]
"""
Import required modules.
"""
# %%

import numpy as np
import matplotlib.pyplot as plt
from terratools import lookup_tables

# %% [markdown]
"""
Define a function to plot the multitables
"""


# %%
def plot_multi(ax, xdats, ydat, xlabs, ylab):
    for xdat, xlab in zip(xdats, xlabs):
        ax.plot(xdat, ydat, label=xlab)
    ax.set_xlabel("V$_p$ (km/s)")
    ax.set_ylabel("Pressure (Pa)")
    ax.legend()
    plt.gca().invert_yaxis()

    return ax


# %% [markdown]
"""
Load in three example sesimic lokoup tables
This creates interpolation objects that can be queried for the sesimic property
at a temperature and pressure at each point.
"""

# %%

hzb_table = lookup_tables.SeismicLookupTable(examples.example_hzb_table())
lhz_table = lookup_tables.SeismicLookupTable(examples.example_lhz_table())
bas_table = lookup_tables.SeismicLookupTable(examples.example_bas_table())

# %% [markdown]

"""
In this example we will find Vp at a simple range of temperatures and
pressures for two compostions (hzb (C=0) and lhz(C=0.2)).
We then find the seismic properties for a compostion of C=0.15.
"""

# %%
# A range of temperatures and pressures
temperatures = np.linspace(500, 3000, 32)
pressures = np.linspace(1e9, 9e10, 32)

# Vp at temperature/pressures
vp_hzb = hzb_table.interp_points(pressures, temperatures, "vp")
vp_lhz = lhz_table.interp_points(pressures, temperatures, "vp")

# Define new and known bulk compositions
Cnew = 0.15
Chzb = 0.0
Clhz = 0.2
# vp_Cnew is vp at the defined temperatures and pressures for C=0.15 mterial.
vp_Cnew = lookup_tables.linear_interp_1d(vp_hzb, vp_lhz, 0.0, 0.2, Cnew)

print(f"vp_Cnew = {vp_Cnew}")

# Plot
fig, ax = plt.subplots(figsize=(4, 7))
plot_multi(
    ax,
    [vp_hzb, vp_lhz, vp_Cnew],
    pressures,
    ["C=0", "C=0.2", "C=0.15"],
    "Pressure (Pa)",
)
plt.show()
plt.close()

# %% [markdown]

"""
In this second example, we will add a third composition (bas (C=1.0)) and find the
Seismic properties for a mechanical mixure of known proportions of hzb, lhz and bas.
"""

# %%
# Fraction of each compostion that makes up the mechanical mixture
hzbfrac = 0.3
lhzfrac = 0.5
basfrac = 0.2

# Vp at same temps/pressures for basalt
vp_bas = bas_table.interp_points(pressures, temperatures, "vp")

# Calculate Vp of the mechanical mixture using harmonic mean
vp_mm = lookup_tables._harmonic_mean(
    [vp_bas, vp_hzb, vp_lhz], [basfrac, lhzfrac, hzbfrac]
)

fig, ax2 = plt.subplots(figsize=(4, 7))
plot_multi(
    ax2,
    [vp_hzb, vp_lhz, vp_bas, vp_mm],
    pressures,
    [
        f"hzb (frac={hzbfrac})",
        f"lhz (frac={lhzfrac})",
        f"bas (frac={basfrac})",
        "Mechanical Mixture",
    ],
    "Pressure (Pa)",
)
plt.show()
plt.close()

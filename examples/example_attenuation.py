# %% [markdown]
"""
# example_attenuation

In this example, we demonstrate how to use the seismic attenuation
classes in TerraTools.
"""

# %% [markdown]
"""
First, let us import all the objects that we will use
"""

# %%
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import terratools
from terratools.properties.profiles import prem_pressure, peridotite_solidus
from terratools.properties.attenuation import Q7g

# %% [markdown]
"""
Next, we create a geotherm interpolation function,
and define the array of depths used in the rest of the script.
"""

# %%
# Find where TerraTools is stored
module_dir = str(Path(terratools.__file__).parent)
geotherm = np.loadtxt(f"{module_dir}/properties/data/anderson_82.txt", unpack=True)
depths = geotherm[0]
temperatures = geotherm[1]
fn_geotherm = interp1d(depths, temperatures, kind="linear")
depths = np.linspace(0.0, 2880.0e3, 1001)

# %% [markdown]
"""
Depths can be converted into pressures using the PREM pressure
function, and the geotherm function just created can be used to
define the temperatures along the geotherm.
"""

# %%
# Convert depths to pressures using PREM
prem_pressure_fn = prem_pressure()
pressures = prem_pressure_fn(depths)
temperatures = fn_geotherm(depths)

# Modify the temperature profile to make it continuous with depth
idx = np.argwhere(depths > 670.0e3)
temperatures[idx] -= 150.0

# %% [markdown]
"""
In the next cell, we use the attenuation model Q7g
packaged within TerraTools to calculate the anelastic
properties at 1 Hz along the geotherm, and at
300 K above the geotherm.

In this script, we are interested only in the fractional
decrease in Vp and Vs due to anelastic effects,
so the elastic velocities are set to 1.
"""

# %%
frequency = 1.0  # Hz
p0 = Q7g.anelastic_properties(
    elastic_Vp=1.0,
    elastic_Vs=1.0,
    pressure=pressures,
    temperature=temperatures,
    frequency=frequency,
)
dT = 300.0
p1 = Q7g.anelastic_properties(
    elastic_Vp=1.0,
    elastic_Vs=1.0,
    pressure=pressures,
    temperature=temperatures + dT,
    frequency=frequency,
)

# %% [markdown]
"""
In the next cell, we print the shear quality factor at the
bottom of the profile to screen
"""

# %%
np.set_printoptions(precision=4)
print(f"Predicted Q_S at {depths[-1]/1.e3} km depth: {p0.Q_S[-1]:.4f}")

# %% [markdown]
"""
Finally, we plot the elastic and anelastic properties.
"""

# %%
fig = plt.figure(figsize=(10, 8))
ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

ax[0].plot(depths / 1.0e3, p0.V_P, label="Vp")
ax[0].plot(depths / 1.0e3, p0.V_S, label="Vs")
ax[0].plot(depths / 1.0e3, p1.V_P, label=f"Vp (+{dT} K)")
ax[0].plot(depths / 1.0e3, p1.V_S, label=f"Vs (+{dT} K)")
ax[1].plot(depths / 1.0e3, pressures / 1.0e9, label="PREM")

# Modified Anderson geotherm is 150 K colder at z > 670 km
ax[2].plot(depths / 1.0e3, temperatures, label="geotherm (Anderson, 1982, mod)")
ax[2].plot(
    depths / 1.0e3,
    temperatures + 300,
    label=f"geotherm (Anderson, 1982, mod) + {dT} K",
)

# The melting temperatures used by the attenuation model
# are given by the peridotite_solidus function
melting_temperatures = peridotite_solidus(pressures)
ax[2].plot(depths / 1.0e3, melting_temperatures, label="solidus")

ax[3].plot(depths / 1.0e3, p0.Q_S, label="QS")
ax[3].plot(depths / 1.0e3, p0.Q_K, label="QK")

ax[3].plot(depths / 1.0e3, p1.Q_S, label=f"QS (+{dT} K)")
ax[3].plot(depths / 1.0e3, p1.Q_K, label=f"QK (+{dT} K)")

# Compare the results to those from Durek and Ekstrom (1996)
prem = np.loadtxt(f"{module_dir}/properties/data/prem.txt", unpack=True)
prem_depths = prem[0]
prem_QK_L6 = prem[6]
prem_QS_L6 = prem[7]
ax[3].plot(prem_depths / 1.0e3, prem_QS_L6, label="QS (QL6; Durek and Ekstrom, 1996)")
ax[3].plot(prem_depths / 1.0e3, prem_QK_L6, label="QK (QL6; Durek and Ekstrom, 1996)")

ax[0].set_ylabel("Anelastic velocity / elastic velocity")
ax[1].set_ylabel("Pressure (GPa)")
ax[2].set_ylabel("Temperature (K)")
ax[3].set_ylabel("Q")

ax[3].set_ylim(1e1, 1.0e6)
ax[3].set_xlim(0, 2880.0)
for i in range(4):
    ax[i].set_xlabel("Depth (km)")
    ax[i].legend()

ax[3].set_yscale("log")

fig.tight_layout()
plt.show()

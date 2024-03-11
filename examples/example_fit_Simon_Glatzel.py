# %% [markdown]
"""
# example_fit_Simon_Glatzel

In this example, we demonstrate how to fit a lower mantle solidus curve
using TerraTools.
"""

# %% [markdown]
"""
First, let us import all the objects that we will use
"""

# %%

import numpy as np
from scipy.optimize import curve_fit
from terratools.properties.utilities import Simon_Glatzel_fn

# %% [markdown]

"""
Here we fit lower mantle solidus curve to Fiquet et al., 2010,
using a pin at the high pressure end of the Herzberg study.
The fitted parameters can be inserted into the
Simon_Glatzel_lower_mantle_Fiquet function, below.
"""
# %%
Pr = 36.0e9
Tr = 2800.0

Pm = np.array([22.5e9, 36.0e9, 135.0e9])
Tm = np.array([1086.0 - 5.7 * 22.5 + 390 * np.log(22.5) + 273.15, 2800.0, 4180.0])

popt, pcov = curve_fit(Simon_Glatzel_fn(Pr, Tr), Pm, Tm, [40.0e9, 0.3])

np.set_printoptions(precision=4)
print("Optimised parameters for the lower mantle solidus curve")
print("using data points from Fiquet et al. (2010):")
print(popt)

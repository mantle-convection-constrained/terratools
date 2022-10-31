import numpy as np
from scipy.interpolate import interp1d
from .utilities import read_table


def prem_pressure():
    """
    Generates an interpolating function for
    pressure as a function of depth according to PREM
    [@Dziewonski1981]

    :return: interp1d function for
    pressure as a function of depth according to PREM
    :rtype: function
    """
    prem = read_table("properties/data/prem.txt").T
    prem_depths = prem[0]
    prem_pressures = prem[2]
    return interp1d(prem_depths, prem_pressures, kind="linear")


def Simon_Glatzel_lower_mantle_Fiquet(pressure):
    """
    Simon-Glatzel model [@Simon1929] for the solidus temperature
    of the lower mantle,
    with parameters fitted on estimates of the melting point
    at 36 and 135 GPa [@Fiquet2010].

    The reference pressure (36 GPa) and temperature (2800 K)
    are chosen using published values
    [@Herzberg2000].

    :param pressure: Pressure (GPa)
    :type pressure: float or numpy array

    :return: Solidus temperature (K)
    :rtype: float or numpy array
    """
    Pr = 36.0e9
    Tr = 2800.0
    a = 3.86695953e10
    b = 3.15554341e-01
    return Tr * np.power(((pressure - Pr) / a + 1.0), b)


def peridotite_solidus(pressure):
    """
    Returns an estimate of the peridotite solidus using three studies:
    [@Hirschmann2000] (0 GPa, then linear extrapolation to 2.7 GPa)
    [@Herzberg2000] (2.7 - 22.5 GPa)
    [@Fiquet2010] (> 22.5 GPa)

    This curve is continuous with pressure, but not differentiable.

    :param pressure: Pressure (GPa)
    :type pressure: float or numpy array

    :return: Solidus temperature (K)
    :rtype: float or numpy array
    """
    # interpolation between Hirschmann (2000) at 0 GPa
    # and Herzberg et al (2000) at 2.7GPa

    if np.isscalar(pressure):
        if pressure < 2.7e9:
            T = (
                1120.661
                + (1086.0 - 5.7 * 2.7 + 390 * np.log(2.7) - 1120.661) * pressure / 2.7e9
                + 273.15
            )
        elif pressure < 22.5e9:  # Herzberg et al (2000)
            T = (
                1086.0
                - 5.7 * pressure / 1.0e9
                + 390 * np.log(pressure / 1.0e9)
                + 273.15
            )
        else:  # Fiquet et al. (2010)
            T = Simon_Glatzel_lower_mantle_Fiquet(pressure)
        return T

    else:
        T = pressure * 0.0
        T = Simon_Glatzel_lower_mantle_Fiquet(pressure)
        idx = np.where(np.all([pressure < 22.5e9, pressure >= 2.7e9], axis=0))
        T[idx] = (
            1086.0
            - 5.7 * pressure[idx] / 1.0e9
            + 390 * np.log(pressure[idx] / 1.0e9)
            + 273.15
        )
        idx = np.where(pressure < 2.7e9)
        T[idx] = (
            1120.661
            + (1086.0 - 5.7 * 2.7 + 390 * np.log(2.7) - 1120.661)
            * pressure[idx]
            / 2.7e9
            + 273.15
        )
        return T

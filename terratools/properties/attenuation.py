"""
Provides models that return quality factors and anelastic seismic velocities.
"""

import numpy as np
from collections import namedtuple
from .profiles import peridotite_solidus
from copy import deepcopy


AnelasticProperties = namedtuple(
    "AnelasticProperties", ["V_P", "V_S", "Q_S", "Q_K", "Q_P", "T_solidus"]
)


class AttenuationModelGoes(object):
    """
    This class implements a mantle seismic attenuation model
    [@Goes2004, @Maguire2016].

    Optionally, different $Q$ models can be used that correspond to
    different mantle materials. A mixing model function should be passed
    to the constructor that takes pressure and temperature as inputs and
    returns the fractions of the different materials.

    This class has an anelastic_properties method to calculate
    anelastic $V_P$ and $V_S$, $Q_S$, and $Q_K$.
    The effective $Q_S$, $Q_K$ and $\\alpha$
    (frequency dependence of the quality factor)
    are given by the linearly weighted sum
    of the $Q_S$, $Q_K$ and $\\alpha$ calculated for each material.
    """

    def __init__(self, T_solidus_function, model_mixing_function, Q_models):
        """
        Constructor for the AttenuationModelGoes class.

        :param function T_solidus_function:
            A function returning the temperature of the solidus
            as a function of pressure.
        :param function model_mixing_function:
            A function returning the amounts of different materials
            as a function of pressure and temperature.
        :param list Q_models:
            Parameter dictionaries for the attenuation models - one for each material.
        """
        self.T_solidus_function = T_solidus_function
        self.model_mixing_function = model_mixing_function
        self.Q_models = Q_models

    def anelastic_properties(
        self,
        elastic_Vp,
        elastic_Vs,
        pressure,
        temperature,
        frequency,
        dT_Q_constant_above_solidus=0,
    ):
        """
        Calculates the anelastic $V_P$ and $V_S$, $Q_S$, and $Q_K$
        according to a published model [@Maguire2016].

        The effects of anelasticity on shear wave velocity are incorporated
        using a model for the S-wave quality factor $Q_S$ that varies with
        pressure $P$ and temperature $T$ as

        $Q_S(\\omega,z,T) = Q_0 \\omega \\alpha \\exp(\\alpha g T_m(z) / T)$

        where $\\omega$ is frequency,
        $\\alpha$ is exponential frequency dependence,
        $g$ is a scaling factor and
        $T_m$ is the dry solidus melting temperature.

        $Q_K$ is chosen to be temperature independent.

        The anelastic seismic velocities are calculated as follows:

        $\\lambda = 4/3 (V_{S,\\text{el}}/V_{P,\\text{el}})^2$

        $1/Q_P = (1 - \\lambda)/Q_K + \\lambda/Q_S

        If $1/Q_P$ is negative, it is set to 0.

        $V_{P,\\text{an}} = V_{P,\\text{el}} (1 - Q_P^{-1}/(2 \\tan ( \\pi \\alpha/2)))$

        $V_{S,\\text{an}} = V_{S,\\text{el}} (1 - Q_S^{-1}/(2 \\tan ( \\pi \\alpha/2)))$

        :param elastic_Vp: The elastic P-wave velocity
        :type elastic_Vp: float or numpy array

        :param elastic_Vs: The elastic S-wave velocity
        :type elastic_Vs: float or numpy array

        :param pressure: The pressure in Pa
        :type pressure: float or numpy array

        :param temperature: The temperature in K
        :type temperature: float or numpy array

        :param frequency: The frequency of the seismic waves in Hz
        :type frequency: float

        :param dT_Q_constant_above_solidus: if the temperature > (solidus temperature + dT),
            the value of QS, QK and a are frozen at the values
            corresponding to (solidus temperature + dT).
        :type dT_Q_constant_above_solidus: float

        :return: An instance of an AnelasticProperties named tuple.
        Has the following attributes:
        V_P, V_S, Q_S, Q_K, Q_P
        """

        fractions = self.model_mixing_function(pressure, temperature)

        try:
            pressure = float(pressure)
            Tm = self.T_solidus_function(pressure)
            # Freezes QS if above a certain temperature
            if dT_Q_constant_above_solidus < temperature - Tm:
                Q_temperature = Tm + dT_Q_constant_above_solidus
            else:
                Q_temperature = deepcopy(temperature)

            QS = 0.0
            QK = 0.0
            alpha = 0.0
            for i, f in enumerate(fractions):
                Q_mod = self.Q_models[i]
                QS += f * (
                    Q_mod["Q0"]
                    * np.power(frequency, Q_mod["a"])
                    * np.exp(Q_mod["a"] * Q_mod["g"] * Tm / Q_temperature)
                )
                QK += f * Q_mod["QK"]
                alpha += f * Q_mod["a"]

        except TypeError:
            Q_temperature = deepcopy(temperature)
            Tm = self.T_solidus_function(pressure)
            idx = np.argwhere(temperature > Tm + dT_Q_constant_above_solidus)
            Q_temperature[idx] = Tm[idx] + dT_Q_constant_above_solidus

            QS = np.zeros_like(temperature)
            QK = np.zeros_like(temperature)
            alpha = np.zeros_like(temperature)
            for i, f in enumerate(fractions.T):
                Q_mod = self.Q_models[i]
                QS += f * (
                    Q_mod["Q0"]
                    * np.power(frequency, Q_mod["a"])
                    * np.exp(Q_mod["a"] * Q_mod["g"] * Tm / Q_temperature)
                )
                QK += f * Q_mod["QK"]
                alpha += f * Q_mod["a"]

        invQS = 1.0 / QS
        invQK = 1.0 / QK

        lmda = 4.0 / 3.0 * np.power(elastic_Vs / elastic_Vp, 2.0)
        invQP = (1.0 - lmda) * invQK + lmda * invQS

        try:
            if invQP < 0.0:
                invQP = 0.0
                QP = np.inf
            else:
                QP = 1.0 / invQP
        except ValueError:
            QP = np.zeros_like(temperature)
            idx = np.argwhere(invQP <= 0.0)
            invQP[idx] = 0.0
            QP[idx] = np.inf
            idx = np.argwhere(invQP > 0.0)
            QP[idx] = 1.0 / invQP[idx]

        anelastic_Vp = elastic_Vp * (1.0 - invQP / (2.0 * np.tan(np.pi * alpha / 2.0)))
        anelastic_Vs = elastic_Vs * (1.0 - invQS / (2.0 * np.tan(np.pi * alpha / 2.0)))

        return AnelasticProperties(
            V_P=anelastic_Vp, V_S=anelastic_Vs, Q_S=QS, Q_K=QK, Q_P=QP, T_solidus=Tm
        )


def mantle_domain_fractions(pressure, temperature):
    """
    This function defines the proportions of
    upper mantle, transition zone, and lower mantle
    domains as a function of pressure and temperature.

    To avoid step-changes in fractions at the top and base of
    the mantle transition zone, transition regions 2.2 GPa wide
    are implemented. At a reference temperature of 750K,
    the center of the ol-wd transition
    is at 11.1 GPa. At the same reference temperature, the center
    of the postspinel transition is at 26.1 GPa. Clapeyron slopes of
    2.4e6 Pa/K and -2.2e6 Pa/K are applied.

    :param pressure: Pressure (Pa)
    :type pressure: float or numpy array

    :param temperature: Temperature (K)
    :type temperature: float or numpy array


    :return:
        A 1D or 2D numpy array containing the effective fractions of
        upper mantle, transition zone and lower mantle material.
        If 2D, the fractions[i,j] corresponds to the ith
        P-T point and jth material.
    """

    P_smooth_halfwidth = 1.1e9
    T_ref = 750.0  # K
    pressure_tztop = 11.1e9 + 2.4e6 * (temperature - T_ref)
    pressure_tzbase = 26.1e9 - 2.2e6 * (temperature - T_ref)

    try:
        fractions = np.zeros(3)
        if pressure < pressure_tztop - P_smooth_halfwidth:
            fractions[0] = 1.0

        elif pressure < pressure_tztop + P_smooth_halfwidth:
            f = (pressure - (pressure_tztop - P_smooth_halfwidth)) / (
                2.0 * P_smooth_halfwidth
            )
            fractions[:2] = [1.0 - f, f]

        elif pressure < pressure_tzbase - P_smooth_halfwidth:
            fractions[1] = 1.0

        elif pressure < pressure_tzbase + P_smooth_halfwidth:
            f = (pressure - (pressure_tzbase - P_smooth_halfwidth)) / (
                2.0 * P_smooth_halfwidth
            )
            fractions[1:] = [1.0 - f, f]

        else:
            fractions[2] = 1.0
    except ValueError:
        fractions = np.zeros((len(pressure), 3))

        f_umtz = (pressure - (pressure_tztop - P_smooth_halfwidth)) / (
            2.0 * P_smooth_halfwidth
        )
        f_tzlm = (pressure - (pressure_tzbase - P_smooth_halfwidth)) / (
            2.0 * P_smooth_halfwidth
        )

        um_idx = np.argwhere(f_umtz <= 0.0)
        umtz_idx = np.argwhere(np.all([f_umtz >= 0.0, f_umtz <= 1.0], axis=0)).T[0]
        tz_idx = np.argwhere(np.all([f_umtz >= 1.0, f_tzlm <= 0.0], axis=0)).T[0]
        tzlm_idx = np.argwhere(np.all([f_tzlm >= 0.0, f_tzlm <= 1.0], axis=0)).T[0]
        lm_idx = np.argwhere(f_tzlm >= 1.0)

        fractions[um_idx, 0] = 1.0
        fractions[umtz_idx, :2] = np.array([1.0 - f_umtz[umtz_idx], f_umtz[umtz_idx]]).T
        fractions[tz_idx, 1] = 1.0
        fractions[tzlm_idx, 1:] = np.array([1.0 - f_tzlm[tzlm_idx], f_tzlm[tzlm_idx]]).T
        fractions[lm_idx, 2] = 1.0

    return fractions


class Q4Goes(AttenuationModelGoes):
    """
    Implements the weak T dependence attenuation model
    after [@Goes2004].

    The model uses the
    [peridotite_solidus][terratools.properties.profiles.peridotite_solidus] and
    [mantle_domain_fractions][terratools.properties.attenuation.mantle_domain_fractions]
    functions to determine the attenuation
    and proportions of upper mantle, transition zone and lower mantle
    materials as a function of pressure and temperature.

    The parameter values for the Q4 attenuation models are given in the
    following table:

    | Layer           | $Q_0$ | $g$  | $\\alpha$ | $Q_K$  |
    | --------------- | ----- | ---- | --------- | ------ |
    | upper mantle    | 0.1   | 38.0 | 0.15      | 1000.0 |
    | transition zone | 3.5   | 20.0 | 0.15      | 1000.0 |
    | lower mantle    | 35.0  | 10.0 | 0.15      | 1000.0 |

    """

    def __init__(self):
        super().__init__(
            peridotite_solidus,
            mantle_domain_fractions,
            Q_models=[
                {"Q0": 0.1, "g": 38.0, "a": 0.15, "QK": 1000.0},
                {"Q0": 3.5, "g": 20.0, "a": 0.15, "QK": 1000.0},
                {"Q0": 35.0, "g": 10.0, "a": 0.15, "QK": 1000.0},
            ],
        )


class Q6Goes(AttenuationModelGoes):
    """
    Implements the strong T dependence attenuation model
    [@Goes2004].

    The model uses the
    [peridotite_solidus][terratools.properties.profiles.peridotite_solidus] and
    [mantle_domain_fractions][terratools.properties.attenuation.mantle_domain_fractions]
    functions to determine the attenuation
    and proportions of upper mantle, transition zone and lower mantle
    materials as a function of pressure and temperature.

    The parameter values for the Q6 attenuation models are given in the
    following table:

    | Layer           | $Q_0$ | $g$  | $\\alpha$ | $Q_K$  |
    | --------------- | ----- | ---- | --------- | ------ |
    | upper mantle    | 0.1   | 38.0 | 0.15      | 1000.0 |
    | transition zone | 0.5   | 30.0 | 0.15      | 1000.0 |
    | lower mantle    | 3.5   | 20.0 | 0.15      | 1000.0 |

    """

    def __init__(self):
        super().__init__(
            peridotite_solidus,
            mantle_domain_fractions,
            Q_models=[
                {"Q0": 0.1, "g": 38.0, "a": 0.15, "QK": 1000.0},
                {"Q0": 0.5, "g": 30.0, "a": 0.15, "QK": 1000.0},
                {"Q0": 3.5, "g": 20.0, "a": 0.15, "QK": 1000.0},
            ],
        )


class Q7Goes(AttenuationModelGoes):
    """
    Implements the intermediate strength T dependence attenuation model
    [@Goes2004]. This model is most consistent with observational data
    [@Matas2007].

    The model uses the
    [peridotite_solidus][terratools.properties.profiles.peridotite_solidus] and
    [mantle_domain_fractions][terratools.properties.attenuation.mantle_domain_fractions]
    functions to determine the attenuation
    and proportions of upper mantle, transition zone and lower mantle
    materials as a function of pressure and temperature.

    The parameter values for the Q7 attenuation models are given in the
    following table:

    | Layer           | $Q_0$ | $g$  | $\\alpha$ | $Q_K$  |
    | --------------- | ----- | ---- | --------- | ------ |
    | upper mantle    | 0.1   | 38.0 | 0.15      | 1000.0 |
    | transition zone | 0.5   | 30.0 | 0.15      | 1000.0 |
    | lower mantle    | 1.5   | 26.0 | 0.15      | 1000.0 |


    """

    def __init__(self):
        super().__init__(
            peridotite_solidus,
            mantle_domain_fractions,
            Q_models=[
                {"Q0": 0.1, "g": 38.0, "a": 0.15, "QK": 1000.0},
                {"Q0": 0.5, "g": 30.0, "a": 0.15, "QK": 1000.0},
                {"Q0": 1.5, "g": 26.0, "a": 0.15, "QK": 1000.0},
            ],
        )


# Objects instantiated from attenuation classes
Q4g = Q4Goes()
Q6g = Q6Goes()
Q7g = Q7Goes()

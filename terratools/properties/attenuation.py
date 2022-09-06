import numpy as np
from collections import namedtuple
from .profiles import peridotite_solidus
from copy import deepcopy

AnelasticProperties = namedtuple('AnelasticProperties',
                                 ['V_P', 'V_S',
                                  'Q_S', 'Q_K', 'Q_P', 'T_solidus'])


class AttenuationModelGoes(object):
    """
    This class implements the mantle seismic attenuation model
    of Goes et al. (2004) and Maguire et al. (2016).
    Thanks to Saskia Goes for her detailed instructions
    and for detailing her favoured model.
    """

    def __init__(self, T_solidus_function,
                 model_mixing_function, Q_models):
        self.T_solidus_function = T_solidus_function
        self.model_mixing_function = model_mixing_function
        self.Q_models = Q_models

    def anelastic_properties(self, elastic_Vp, elastic_Vs,
                             pressure, temperature, frequency,
                             dT_Q_constant_above_solidus=0):
        """
        Calculates the anelastic Vp and Vs, QS, and QK according to the model
        used in Maguire et al., 2016.

        The effects of anelasticity on shear wave velocity are incorporated
        using a model for the S-wave quality factor QS that varies with
        temperature T and pressure P as
        QS(z,T) = Qo ω a exp(a ξ Tm / T), where
        ω is frequency,
        a is exponential frequency dependence,
        ξ is a pressure scaling factor and
        Tm is the dry solidus melting temperature.

        To avoid step-changes in QS at the top and base of
        the mantle, transition regions 2.2 GPa wide are implemented.
        At a reference temperature of 750K, the center of the ol-wd transition
        is at 11.1 GPa. At the same reference temperature, the center
        of the postspinel transition is at 26.1 GPa. Clapeyron slopes of
        2.4e6 Pa/K and -2.2e6 Pa/K are applied.

        QK is chosen to be temperature independent.

        The anelastic seismic velocities are then calculated as follows:
        lmda = 4/3 * (elastic_Vs/elastic_Vp)^2
        1/QP = (1. - lmda)/QK + lmda/QS

        If 1/QP is negative, it is set to 0.

        anelastic_Vp = elastic_Vp*(1 - invQP/(2tan(pi*alpha/2)))
        anelastic_Vs = elastic_Vs*(1 - invQS/(2tan(pi*alpha/2)))

        Arguments
        ---------
        elastic_Vp : the elastic P-wave velocity
        elastic_Vs : the elastic S-wave velocity
        pressure : the pressure in Pa
        temperature : the temperature in K
        frequency: the frequency of the seismic waves in Hz
        Q_model: a dictionary containing the parameters for the
        attenuation model
        dT_Q_constant_above_solidus: if the temperature
        > (solidus temperature + dT),
        the value of QS, QK and a are frozen at the values corresponding to
        (solidus temperature + dT).

        Returns
        -------

        An instance of an AnelasticProperties named tuple.
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

            QS = 0.
            QK = 0.
            alpha = 0.
            for i, f in enumerate(fractions):
                Q_mod = self.Q_models[i]
                QS += f * (Q_mod['Q0'] *
                           np.power(frequency, Q_mod['a']) *
                           np.exp(Q_mod['a']*Q_mod['g']*Tm/Q_temperature))
                QK += f * Q_mod['QK']
                alpha += f * Q_mod['a']

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
                QS += f * (Q_mod['Q0'] *
                           np.power(frequency, Q_mod['a']) *
                           np.exp(Q_mod['a']*Q_mod['g']*Tm/Q_temperature))
                QK += f * Q_mod['QK']
                alpha += f * Q_mod['a']

        invQS = 1./QS
        invQK = 1./QK

        lmda = 4./3.*np.power(elastic_Vs/elastic_Vp, 2.)
        invQP = (1. - lmda)*invQK + lmda*invQS

        try:
            if invQP < 0.:
                invQP = 0.
                QP = np.inf
            else:
                QP = 1./invQP
        except ValueError:
            QP = np.zeros_like(temperature)
            idx = np.argwhere(invQP <= 0.)
            invQP[idx] = 0.
            QP[idx] = np.inf
            idx = np.argwhere(invQP > 0.)
            QP[idx] = 1./invQP[idx]

        anelastic_Vp = elastic_Vp*(1. - invQP/(2.*np.tan(np.pi*alpha/2.)))
        anelastic_Vs = elastic_Vs*(1. - invQS/(2.*np.tan(np.pi*alpha/2.)))

        return AnelasticProperties(V_P=anelastic_Vp,
                                   V_S=anelastic_Vs,
                                   Q_S=QS,
                                   Q_K=QK,
                                   Q_P=QP,
                                   T_solidus=Tm)


def mantle_domain_fractions(pressure, temperature):
    """
    This function defines the proportions of
    upper mantle, transition zone, and lower mantle
    domains as a function of pressure and temperature
    """

    P_smooth_halfwidth = 1.1e9
    T_ref = 750.  # K
    pressure_tztop = 11.1e9 + 2.4e6*(temperature - T_ref)
    pressure_tzbase = 26.1e9 - 2.2e6*(temperature - T_ref)

    try:
        fractions = np.zeros(3)
        if pressure < pressure_tztop - P_smooth_halfwidth:
            fractions[0] = 1.

        elif pressure < pressure_tztop + P_smooth_halfwidth:
            f = ((pressure - (pressure_tztop - P_smooth_halfwidth))
                 / (2.*P_smooth_halfwidth))
            fractions[:2] = [1. - f, f]

        elif pressure < pressure_tzbase - P_smooth_halfwidth:
            fractions[1] = 1.

        elif pressure < pressure_tzbase + P_smooth_halfwidth:
            f = ((pressure - (pressure_tzbase - P_smooth_halfwidth))
                 / (2.*P_smooth_halfwidth))
            fractions[1:] = [1. - f, f]

        else:
            fractions[2] = 1.
    except ValueError:
        fractions = np.zeros((len(pressure), 3))

        f_umtz = ((pressure - (pressure_tztop - P_smooth_halfwidth))
                  / (2.*P_smooth_halfwidth))
        f_tzlm = ((pressure - (pressure_tzbase - P_smooth_halfwidth))
                  / (2.*P_smooth_halfwidth))

        um_idx = np.argwhere(f_umtz <= 0.)
        umtz_idx = np.argwhere(np.all([f_umtz >= 0., f_umtz <= 1.],
                                      axis=0)).T[0]
        tz_idx = np.argwhere(np.all([f_umtz >= 1., f_tzlm <= 0.],
                                    axis=0)).T[0]
        tzlm_idx = np.argwhere(np.all([f_tzlm >= 0., f_tzlm <= 1.],
                                      axis=0)).T[0]
        lm_idx = np.argwhere(f_tzlm >= 1.)

        fractions[um_idx, 0] = 1.
        fractions[umtz_idx, :2] = np.array([1. - f_umtz[umtz_idx],
                                            f_umtz[umtz_idx]]).T
        fractions[tz_idx, 1] = 1.
        fractions[tzlm_idx, 1:] = np.array([1. - f_tzlm[tzlm_idx],
                                            f_tzlm[tzlm_idx]]).T
        fractions[lm_idx, 2] = 1.

    return fractions


# Q4g - low T dependence (after Goes et al. 2004)
# Order of models is upper mantle, transition zone, lower mantle
Q4g = AttenuationModelGoes(peridotite_solidus,
                           mantle_domain_fractions,
                           Q_models=[{'Q0': 0.1, 'g': 38., 'a': 0.15,
                                      'QK': 1000.},
                                     {'Q0': 3.5, 'g': 20., 'a': 0.15,
                                      'QK': 1000.},
                                     {'Q0': 35., 'g': 10., 'a': 0.15,
                                      'QK': 1000.}])

# Q6g - strong T dependence (after Goes et al. 2004
Q6g = AttenuationModelGoes(peridotite_solidus,
                           mantle_domain_fractions,
                           Q_models=[{'Q0': 0.1, 'g': 38., 'a': 0.15,
                                      'QK': 1000.},
                                     {'Q0': 0.5, 'g': 30., 'a': 0.15,
                                      'QK': 1000.},
                                     {'Q0': 3.5, 'g': 20., 'a': 0.15,
                                      'QK': 1000.}])

# Q7g - intermediate T dependence
# (most consistent with Matas and Bukuwinski 2007)
Q7g = AttenuationModelGoes(peridotite_solidus,
                           mantle_domain_fractions,
                           Q_models=[{'Q0': 0.1, 'g': 38., 'a': 0.15,
                                      'QK': 1000.},
                                     {'Q0': 0.5, 'g': 30., 'a': 0.15,
                                      'QK': 1000.},
                                     {'Q0': 1.5, 'g': 26., 'a': 0.15,
                                      'QK': 1000.}])

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from terratools.properties.profiles import prem_pressure, peridotite_solidus
from terratools.properties.attenuation import Q7g

if __name__ == "__main__":

    prem = np.loadtxt('../terratools/properties/data/prem.txt', unpack=True)
    prem_depths = prem[0]
    prem_pressures = prem[2]
    prem_QK_L6 = prem[6]
    prem_QS_L6 = prem[7]
    prem_pressure_fn = prem_pressure()

    geotherm = np.loadtxt('../terratools/properties/data/anderson_82.txt',
                          unpack=True)
    depths = geotherm[0]
    temperatures = geotherm[1]
    fn_geotherm = interp1d(depths, temperatures, kind='linear')
    
    frequency = 1.  # Hz

    depths = np.linspace(0., 2880.e3, 1001)
    Vps = np.empty_like(depths)
    Vss = np.empty_like(depths)
    QSs = np.empty_like(depths)
    QKs = np.empty_like(depths)
    Vpsh = np.empty_like(depths)
    Vssh = np.empty_like(depths)
    QSsh = np.empty_like(depths)
    QKsh = np.empty_like(depths)
    temperatures = np.empty_like(depths)
    melting_temperatures = np.empty_like(depths)
    pressures = np.empty_like(depths)

    dT = 300.
    
    temperatures = fn_geotherm(depths)
    pressures = prem_pressure_fn(depths)
    melting_temperatures = peridotite_solidus(pressures)

    # modify the temperature profile to make it continuous with depth
    idx = np.argwhere(depths > 670.e3)
    temperatures[idx] -= 150.

    p0 = Q7g.anelastic_properties(elastic_Vp=1.,
                                  elastic_Vs=1.,
                                  depth=depths,
                                  temperature=temperatures,
                                  frequency=frequency)

    p1 = Q7g.anelastic_properties(elastic_Vp=1.,
                                  elastic_Vs=1.,
                                  depth=depths,
                                  temperature=temperatures+dT,
                                  frequency=frequency)

    np.set_printoptions(precision=4)
    print(f'Predicted Q_S at {depths[-1]/1.e3} km depth: {p0.Q_S[-1]:.4f}')

    fig = plt.figure(figsize=(10, 8))
    ax = [fig.add_subplot(2, 2, i) for i in range(1, 5)]

    ax[0].plot(depths/1.e3, p0.V_P, label='Vp')
    ax[0].plot(depths/1.e3, p0.V_S, label='Vs')
    ax[0].plot(depths/1.e3, p1.V_P, label=f'Vp (+{dT} K)')
    ax[0].plot(depths/1.e3, p1.V_S, label=f'Vs (+{dT} K)')

    ax[1].plot(depths/1.e3, pressures/1.e9, label='PREM')

    # *; -150 K at z > 670 km
    ax[2].plot(depths/1.e3, temperatures,
               label='geotherm (Anderson, 1982*)')
    ax[2].plot(depths/1.e3, temperatures+300,
               label=f'geotherm (Anderson, 1982*) + {dT} K')
    ax[2].plot(depths/1.e3, melting_temperatures, label='solidus')

    ax[3].plot(depths/1.e3, p0.Q_S, label='QS')
    ax[3].plot(depths/1.e3, p0.Q_K, label='QK')

    ax[3].plot(depths/1.e3, p1.Q_S, label=f'QS (+{dT} K)')
    ax[3].plot(depths/1.e3, p1.Q_K, label=f'QK (+{dT} K)')

    ax[3].plot(prem_depths/1.e3, prem_QS_L6,
               label='QS (QL6; Durek and Ekstrom, 1996)')
    ax[3].plot(prem_depths/1.e3, prem_QK_L6,
               label='QK (QL6; Durek and Ekstrom, 1996)')

    ax[0].set_ylabel('Anelastic velocity / elastic velocity')
    ax[1].set_ylabel('Pressure (GPa)')
    ax[2].set_ylabel('Temperature (K)')
    ax[3].set_ylabel('Q')

    ax[3].set_ylim(1e1, 1.e6)
    ax[3].set_xlim(0, 2880.)
    for i in range(4):
        ax[i].set_xlabel('Depth (km)')
        ax[i].legend()

    ax[3].set_yscale('log')

    fig.tight_layout()
    plt.show()

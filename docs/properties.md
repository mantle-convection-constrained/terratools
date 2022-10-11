TerraTools provides tools that facilitate the creation of TERRA-readable tables of
material properties.

## Elastic properties
### PerpleX

One common way to calculate material properties as a function of composition,
pressure and temperature is using the software package PerpleX [@Connolly2009].

TerraTools comes packaged with three functions to interface directly with PerpleX.
The first is
[properties.perplex.make_build_files][terratools.properties.perplex.make_build_files],
that creates a very specific type of input file; specifically one designed to tell
PerpleX's vertex program to compute the stable equilibrium assemblages at a 
fixed bulk composition over a 2D grid of pressures and temperatures. The function
allows the user to split up the problem into many smaller grids, helping ease
memory constraints.

The second PerpleX helper function in TerraTools is
[properties.perplex.run_build_files][terratools.properties.perplex.run_build_files],
that runs PerpleX's vertex and pssect software on all the build files created by
[properties.perplex.make_build_files][terratools.properties.perplex.make_build_files].

Finally, the third PerpleX helper function in TerraTools is
[properties.perplex.perplex_to_grid][terratools.properties.perplex.perplex_to_grid],
that creates a 3D numpy array of all the properties that TERRA needs, over the
pressures and temperatures of interest.

## Anelastic properties

The properties output by PerpleX are (for the most-part) properties at
infinite frequency. The seismic velocities are therefore the purely elastic ones.
To convert these seismic velocities to anelastic velocities, TerraTools provides
access to attenuation models. At present, the module includes three variations
on a simple but flexible attenuation model [@Goes2004;@Maguire2016],
packaged in the class
[properties.attenuation.AttenuationModelGoes][terratools.properties.attenuation.AttenuationModelGoes].
The effects of anelasticity on shear wave velocity are incorporated
using a model for the S-wave quality factor $Q_S$ that varies with
pressure $P$ and temperature $T$ as
$$
Q_S(\omega,z,T) = Q_0 \omega \alpha \exp \left(\alpha g \frac{T_m(z)}{T} \right)
$$
where
$\omega$ is frequency, $\alpha$ is exponential frequency dependence,
$g$ is a scaling factor and
$T_m$ is the dry solidus melting temperature.

$Q_K$ is chosen to be temperature independent.

The anelastic seismic velocities are calculated as follows:
$$
\lambda = \frac{4}{3} \left(\frac{V_{\text{S,el}}}{V_{\text{P,el}}}\right)^2
$$
$$
{Q_P}^{-1} = {(1 - \lambda)}{Q_K}^{-1}  + {\lambda}{Q_S}^{-1} 
$$
If ${Q_P}^{-1}$ is negative, it is set to 0.

$$
V_{\text{P,an}} = V_{\text{P,el}}\left(1 - {Q_P}^{-1}/(2 \tan (0.5 \pi \alpha))\right)
$$

$$
V_{\text{S,an}} = V_{\text{S,el}}\left(1 - {Q_S}^{-1}/(2 \tan (0.5 \pi \alpha))\right)
$$

The [properties.attenuation.AttenuationModelGoes][terratools.properties.attenuation.AttenuationModelGoes]
models are constructed from three parts:

- A model for the solidus temperature $T_m$ as a function of pressure, given by the function
[properties.profiles.peridotite_solidus][terratools.properties.profiles.peridotite_solidus]
designed as part of the TerraTools project to capture available experimental data
[@Hirschmann2000;@Herzberg2000;@Fiquet2010].
- A highly simplified model for the mineralogy of the mantle, split into upper mantle,
transition zone and lower mantle. The proportions of each zone as a function of pressure
and temperature are provided by the function
[properties.attenuation.mantle_domain_fractions][terratools.properties.attenuation.mantle_domain_fractions].
- Parameters for the attenuation model for each mantle mineralogy. 
Three model parameterisations are provided with different temperature dependencies on attenuation: 
[Q4Goes][terratools.properties.attenuation.Q4Goes] (weak temperature dependence),
[Q6Goes][terratools.properties.attenuation.Q6Goes] (strong temperature dependence), and
[Q7Goes][terratools.properties.attenuation.Q7Goes] (moderate temperature dependence).
The favoured model is [Q7Goes][terratools.properties.attenuation.Q7Goes], which produces
a good agreement with published studies on attenuation [@Matas2007].


| Model | Layer           | $Q_0$ | $g$  | $\alpha$ | $Q_K$  |
| ----- | --------------- | ----- | ---- | -------- | ------ |
| Q4    | upper mantle    | 0.1   | 38.0 | 0.15     | 1000.0 |
|       | transition zone | 3.5   | 20.0 | 0.15     | 1000.0 |
|       | lower mantle    | 35.0  | 10.0 | 0.15     | 1000.0 |
| Q6    | upper mantle    | 0.1   | 38.0 | 0.15     | 1000.0 |
|       | transition zone | 0.5   | 30.0 | 0.15     | 1000.0 |
|       | lower mantle    | 3.5   | 20.0 | 0.15     | 1000.0 |
| Q7    | upper mantle    | 0.1   | 38.0 | 0.15     | 1000.0 |
|       | transition zone | 0.5   | 30.0 | 0.15     | 1000.0 |
|       | lower mantle    | 1.5   | 26.0 | 0.15     | 1000.0 |

To ensure smooth behaviour, the effective $Q_S$, $Q_K$ and $\alpha$
are given by the linear proportion-weighted sum of the
$Q_S$, $Q_K$ and $\alpha$ for each material at every $P$-$T$ point.

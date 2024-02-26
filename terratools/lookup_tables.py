import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
import os
import matplotlib.pyplot as plt
from warnings import warn

# Set of all table properties, in the order they appear in the table,
# excluding pressure and temperature
TABLE_FIELDS = ("vp", "vs", "vp_an", "vs_an", "vphi", "density", "qs", "t_sol")


class SeismicLookupTable:
    def __init__(
        self,
        table_path=None,
        pressure=None,
        temperature=None,
        vp=None,
        vs=None,
        vp_an=None,
        vs_an=None,
        vphi=None,
        density=None,
        qs=None,
        t_sol=None,
    ):
        """
        Instantiate a ``SeismicLookupTable``, which can be used to
        find seismic properties at a point in pressure-temperature space.

        There are two ways of building a ``SeismicLookupTable``:

        #. Read in a file from disk using the argument ``table_path``.
        #. Pass arrays for all of the seismic properties making up the lookup
           table.

        Instances of ``SeismicLookupTable`` contain the attribute ``fields``
        which hold the different seismic properties. Under each field are:

        - [0] : Table Index
        - [1] : Field gridded in T-P space
        - [2] : Units

        This also sets up interpolator objects (eg ``self.vp_interp``)
        for rapid querying of points.

        Reading from a file
        *******************

        To read a ``SeismicLookupTable`` from file, pass a string to
        ``table_path`` (the first argument to the constructor).  This is the
        path to a file on disk where the input table is stored in plain text.

        The input table must have the following structure, where each
        row gives the set of values at one pressure-temperature point.  Note
        that the temperature increases fastest, and the pressure slowest, such
        that all temperatures are given at the first pressure point first, and
        so on.  The table below presents a simple example:

        | Pressure | Temperature | Vp | Vs | Vp_an | Vs_an | Vphi | Density | Qs | T_solidus |
        | -------- | ----------- | -- | -- | ----- | ----- | ---- | ------- | -- | --------- |
        | 0        | 500         |    |    |       |       |      |         |    |           |
        | 0        | 1000        |    |    |       |       |      |         |    |           |
        | 0        | 1500        |    |    |       |       |      |         |    |           |
        | 1e8      | 500         |    |    |       |       |      |         |    |           |
        | 1e8      | 1000        |    |    |       |       |      |         |    |           |
        | 1e8      | 1500        |    |    |       |       |      |         |    |           |

        Construction from arrays
        ************************

        If building a ``SeismicLookupTable`` from data in memory, all
        remaining arguments (``vp``, ``vs``, ``vp_an``, ``vs_an``, ``vphi``,
        ``density``, ``qs`` and ``t_sol``) must be passed in.  Each of these
        must be a 2-D array, where the first dimension ranges over the
        temperatures and the second ranges over the pressures.

        :param table_path: Path to table file, e.g. '/path/to/data/table.dat'
        :type table_path: string

        :param pressure: Iterable of pressures in GPa at which values in the
            lookup table have been evaluated.  This forms the second dimension
            of the fields in the lookup table.
        :type pressure: list, ``numpy.array`` or other iterable

        :param temperature: Iterable of temperature in K at which values in
            the lookup table have been evaluated.  This forms the first
            dimension of the fields in the lookup table.
        :type temperature: list, ``numpy.array`` or other iterable

        :param vp: Grid of elastic P-wave velocities in km/s
        :type vp: 2-D ``numpy.array``

        :param vs: Grid of elastic S-wave velocities in km/s
        :type vs: 2-D ``numpy.array``

        :param vp_an: Grid of P-wave velocities in km/s
        :type vp_an: 2-D ``numpy.array``

        :param vs_an: Grid of anelastic S-wave velocities in km/s
        :type vs_an: 2-D ``numpy.array``

        :param vphi: Grid of elastic bulk velocities in km/s
        :type vphi: 2-D ``numpy.array``

        :param density: Grid of densities in kmg/m^3
        :type density: 2-D ``numpy.array``

        :param qs: Grid of S-wave quality factors
        :type qs: 2-D ``numpy.array``

        :param t_sol: Grid of solidus temperatures in K
        :type t_sol: 2-D ``numpy.array``

        :return: Lookup table object
        """
        if table_path is not None:
            try:
                table_data = np.genfromtxt(f"{table_path}")
            except:
                table_data = np.genfromtxt(f"{table_path}", skip_header=1)

            self.pres = np.unique(table_data[:, 0])
            self.temp = np.unique(table_data[:, 1])

            nP = len(self.pres)
            nT = len(self.temp)

            # Initialise arrays for storing table columns in Temp-Pressure space
            vp = np.zeros((nT, nP))
            vs = np.zeros((nT, nP))
            vp_an = np.zeros((nT, nP))
            vs_an = np.zeros((nT, nP))
            vphi = np.zeros((nT, nP))
            density = np.zeros((nT, nP))
            qs = np.zeros((nT, nP))
            t_sol = np.zeros((nT, nP))

            pstep = np.size(self.temp)

            # Fill arrays with table data
            for i, p in enumerate(self.pres):
                vp[:, i] = table_data[0 + (i * pstep) : pstep + (i * pstep), 2]
                vs[:, i] = table_data[0 + (i * pstep) : pstep + (i * pstep), 3]
                vp_an[:, i] = table_data[0 + (i * pstep) : pstep + (i * pstep), 4]
                vs_an[:, i] = table_data[0 + (i * pstep) : pstep + (i * pstep), 5]
                vphi[:, i] = table_data[0 + (i * pstep) : pstep + (i * pstep), 6]
                density[:, i] = table_data[0 + (i * pstep) : pstep + (i * pstep), 7]
                qs[:, i] = table_data[0 + (i * pstep) : pstep + (i * pstep), 8]
                t_sol[:, i] = table_data[0 + (i * pstep) : pstep + (i * pstep), 9]
        else:
            # Check we have passed everything in
            for arg in (
                pressure,
                temperature,
                vp,
                vs,
                vp_an,
                vs_an,
                vphi,
                density,
                qs,
                t_sol,
            ):
                if arg is None:
                    raise ValueError(
                        "one or more of vp, vs, vp_an, vs_an, vphi, density, qs, t_sol not passed in"
                    )

            self.pres = pressure
            self.temp = temperature

            # Check for shapes of arrays
            nP = len(pressure)
            nT = len(temperature)
            for arg, name in zip(
                (vp, vs, vp_an, vs_an, vphi, density, qs, t_sol),
                TABLE_FIELDS,
            ):
                if arg.shape != (nT, nP):
                    raise ValueError(
                        f"array {name} has dimensions {arg.shape}, not "
                        + f"{(nT, nP)} as expected"
                    )

        self.t_max = np.max(self.temp)
        self.t_min = np.min(self.temp)
        self.p_max = np.max(self.pres)
        self.p_min = np.min(self.pres)

        # Setup interpolator objects. These can be used for rapid querying of many individual points
        self.vp_interp = RegularGridInterpolator(
            (self.pres, self.temp), vp.T, method="linear"
        )
        self.vs_interp = RegularGridInterpolator(
            (self.pres, self.temp), vs.T, method="linear"
        )
        self.vp_an_interp = RegularGridInterpolator(
            (self.pres, self.temp), vp_an.T, method="linear"
        )
        self.vs_an_interp = RegularGridInterpolator(
            (self.pres, self.temp), vs_an.T, method="linear"
        )
        self.vphi_interp = RegularGridInterpolator(
            (self.pres, self.temp), vphi.T, method="linear"
        )
        self.density_interp = RegularGridInterpolator(
            (self.pres, self.temp), density.T, method="linear"
        )
        self.qs_interp = RegularGridInterpolator(
            (self.pres, self.temp), qs.T, method="linear"
        )
        self.t_sol_interp = RegularGridInterpolator(
            (self.pres, self.temp), t_sol.T, method="linear"
        )

        # Creat dictionary which holds the interpolator objects
        self.fields = {
            "vp": [2, vp, "km/s", self.vp_interp],
            "vs": [3, vs, "km/s", self.vs_interp],
            "vp_an": [4, vp_an, "km/s", self.vp_an_interp],
            "vs_an": [5, vs_an, "km/s", self.vs_an_interp],
            "vphi": [6, vphi, "km/s", self.vphi_interp],
            "density": [7, density, "$kg/m^3$", self.density_interp],
            "qs": [8, qs, "Hz", self.qs_interp],
            "t_sol": [9, t_sol, "K", self.t_sol_interp],
        }

    #################################################
    # Need to get temp, pres, comp at given point.
    # Pressure could come from PREM or from simulation
    # Comp will be in 3 component mechanical mixture
    # We will then find the
    #################################################

    def interp_grid(self, press, temps, field):
        """
        Routine for re-gridding lookup tables into new pressure-temperature space.
        Any values of pressure or temperatures which lie outside the range
        of the original grid are set to lie at the edges of the original grid
        in P-T space.

        :param press: Pressures (Pa)
        :type press: float or numpy array
        :param temps: Temperatures (K)
        :type temps: float or numpy array
        :param field: Data field (eg. 'vs')
        :type field: string
        :return: interpolated values of a given table property
                on a 2D grid defined by press and temps
        :rtype: 2D numpy array
        :example:

        >>> t_test = [4,5,6]
        >>> p_test = 10
        >>> basalt = SeismicLookupTable('../tests/data/test_lookup_table.txt')
        >>> basalt.interp_grid(p_test, t_test, 'vs')
        """

        press = _check_bounds(press, self.pres)
        temps = _check_bounds(temps, self.temp)
        interp_func = self.fields[field][3]
        pressure_grid, temp_grid = np.meshgrid(press, temps, indexing="ij", sparse=True)
        return interp_func((pressure_grid, temp_grid))

    def interp_points(self, press, temps, field):
        """
        Routine for interpolating gridded property data at one or more
        pressure-temperature points.  Pressures and temperatures can be
        arbitrary numpy arrays or scalars; output shape is subject to
        normal broadcasting rules.

        :param press: Pressures (Pa)
        :type press: float or numpy array
        :param temps: Temperatures (K)
        :type temps: float or numpy array
        :param field: Data field (eg. 'vs')
        :type field: string

        :return: interpolated values of a given table property
                at points defined by press and temps
        :rtype: float or numpy array of same shape as inputs
        """

        press = _check_bounds(press, self.pres)
        temps = _check_bounds(temps, self.temp)

        interp_func = self.fields[field][3]
        out = interp_func((press, temps))

        return out

    def plot_table(self, ax, field, cmap="viridis_r"):
        """
        Plots the lookup table as a grid with values coloured by
        value for the field given.

        :param ax: matplotlib axis object to plot on.
        :type ax: matplotlib axis object
        :param field: Data field (eg. 'vs')
        :type field: string
        :param cmap: matplotlib colourmap.
        :type cmap: string

        :return: None
        """

        units = self.fields[field.lower()][2]
        data = self.fields[field.lower()][1]

        # temperature on x axis
        data = data.transpose()

        chart = ax.imshow(
            data,
            origin="lower",
            extent=[self.t_min, self.t_max, self.p_min, self.p_max],
            cmap=cmap,
            aspect="auto",
        )

        plt.colorbar(chart, ax=ax, label=f"{field} ({units})")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Pressure (Pa)")
        ax.set_title(f"P-T graph for {field}")
        ax.invert_yaxis()

    def plot_table_contour(self, ax, field, cmap="viridis_r"):
        """
        Plots the lookup table as contours using matplotlibs tricontourf.

        :param ax: matplotlib axis object to plot on.
        :type ax: matplotlib axis object
        :param field: Data field (eg. 'Vs')
        :type field: string
        :param cmap: matplotlib colourmap.
        :type cmap: string

        :return: None
        """
        data = self.fields[field.lower()][1]
        units = self.fields[field.lower()][2]

        chart = ax.contourf(self.pres, self.temp, np.transpose(data), cmap=cmap)

        plt.colorbar(chart, ax=ax, label=f"{field} ({units})")
        ax.set_ylabel("Temperature (K)")
        ax.set_xlabel("Pressure (Pa)")
        ax.set_title(f"P-T graph for {field}")


class MultiTables:
    def __init__(self, lookuptables):
        """
        Class to take in and process multiple tables at once.

        :param lookuptables: dictionary with keys describing the lookup table composition (e.g. "basalt")
            whose values are either:
            * the associated lookup table filename to be read in
            * a ``SeismicLookupTable`` object
        :type loookuptables: dictionary

        :return: multitable object.
        """
        self._tables = lookuptables
        self._lookup_tables = {}
        for key, value in self._tables.items():
            if isinstance(value, SeismicLookupTable):
                self._lookup_tables[key] = value
            else:
                self._lookup_tables[key] = SeismicLookupTable(value)

    def evaluate(self, P, T, fractions, field):
        """
        Returns the harmonic mean of a parameter over several lookup
        tables weighted by their fraction.

        :param P: pressure value to evaluate in Pa.
        :type P: float
        :param T: temperature value to evaluate in K.
        :type T: float
        :param fractions: relative proportions of
                          compositions. The keys in
                          the dictionary need to be
                          the same as the tables
                          attribute.
        :type fractions: dictionary
        :param field: property to evaluate, e.g. 'vs'.
        :type field: str
        :return: property 'prop' evaluated at T and P.
        :rtype: float
        """

        values = []
        fracs = []
        for key in self._tables:
            frac = fractions[key]
            value = self._lookup_tables[key].interp_points(P, T, field)
            values.append(value)
            fracs.append(frac)

        value = _harmonic_mean(data=values, fractions=fracs)

        return value


def _harmonic_mean(data, fractions):
    """
    Our own harmonic mean function. scipy.stats does have one
    but will only work on 1D arrays whereas this will take the
    mean of 2D arrays such as lookup tables also.
    If averaging lookup tables, they must have the same shape.

    :param data: data to perform harmonic mean.
    :type data: 1D or 3D numpy array. axis=0 must
                be the axis along which the 2D arrays
                change.
    :param fractions: relative fractions to weight data
    :type fractions: 1D numpy array of floats

    :return: harmonic mean of input values
    :rtype: float or 2D numpy array of floats.
    """

    m_total = np.zeros(np.shape(data[0]))
    frac_total = np.zeros(np.shape(fractions[0]))

    for i in range(len(fractions)):
        m_total += fractions[i] / data[i]
        frac_total += fractions[i]

    hmean = frac_total / m_total

    return hmean


def linear_interp_1d(vals1, vals2, c1, c2, cnew):
    """
    :param vals1: data for composition 1
    :param vals2: data for composition 2
    :param c1: C-value for composition 1
    :param c2: C-value for composition 2
    :param cnew: C-value(s) for new composition(s)

    :return: interpolated values for compositions cnew
    :rtype: float
    """

    interpolated = interp1d(
        np.array([c1, c2]),
        [vals1.flatten(), vals2.flatten()],
        fill_value="extrapolate",
        axis=0,
    )

    return interpolated(cnew)


def _check_bounds(input, check):
    """
    Check which of the valus in inputs exceeds the bounds in check
    and replace them with the min/max bound as appropriate.

    :param input: temperature or pressure of interest
    :type input: float
    :param check: range of table pressure or temperature
    :type check: 1D array of floats
    :return: output
    :rtype: numpy array of floats
    """

    if np.any(input > np.max(check)):
        warn(
            f"One or more of your inputs exceeds the table range, reverting to maximum table range"
        )

    elif np.any(input < np.min(check)):
        warn(
            f"One or more of your inputs is below the table range, reverting to minimum table range"
        )

    output = np.clip(input, np.min(check), np.max(check))

    return output

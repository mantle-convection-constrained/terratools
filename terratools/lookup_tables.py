import numpy as np
from scipy.interpolate import interp2d, interp1d
import os
import matplotlib.pyplot as plt


class SeismicLookupTable:
    def __init__(self, table_path):
        """
        Calling will create a dictionary (self.fields) containing
        fields given in seismic lookup table. Under each field are:

        - [0] : Table Index
        - [1] : Field gridded in T-P space
        - [2] : Units

        This also sets up interpolator objects (eg self.vp_interp)
        for rapid querying of points.

        Currently the input table must have the following structure, with rows
        ascending by pressure then temperature.

        | Pressure | Temperature | Vp | Vs | Vp_an | Vs_an | Vphi | Density | Qs | T_solidus |
        | -------- | ----------- | -- | -- | ----- | ----- | ---- | ------- | -- | --------- |
        | 0        | 500         |    |    |       |       |      |         |    |           |
        | 0        | 1000        |    |    |       |       |      |         |    |           |
        | 0        | 1500        |    |    |       |       |      |         |    |           |
        | 1e8      | 500         |    |    |       |       |      |         |    |           |
        | 1e8      | 1000        |    |    |       |       |      |         |    |           |
        | 1e8      | 1500        |    |    |       |       |      |         |    |           |

        :param table_path: Path to table file, e.g. '/path/to/data/table.dat'
        :type table_path: string

        :return: Lookup table object
        """
        try:
            self.table = np.genfromtxt(f"{table_path}")
        except:
            self.table = np.genfromtxt(f"{table_path}", skip_header=1)

        self.P = self.table[:, 0]
        self.T = self.table[:, 1]
        self.pres = np.unique(self.table[:, 0])
        self.temp = np.unique(self.table[:, 1])
        self.n_uniq_p = len(self.pres)
        self.n_uniq_t = len(self.temp)
        self.t_max = np.max(self.temp)
        self.t_min = np.min(self.temp)
        self.p_max = np.max(self.pres)
        self.p_min = np.min(self.pres)
        self.pstep = np.size(self.temp)

        # Initialise arrays for storing table columns in Temp-Pressure space
        Vp = np.zeros((len(self.temp), len(self.pres)))
        Vs = np.zeros((len(self.temp), len(self.pres)))
        Vp_an = np.zeros((len(self.temp), len(self.pres)))
        Vs_an = np.zeros((len(self.temp), len(self.pres)))
        Vphi = np.zeros((len(self.temp), len(self.pres)))
        Dens = np.zeros((len(self.temp), len(self.pres)))
        Qs = np.zeros((len(self.temp), len(self.pres)))
        T_sol = np.zeros((len(self.temp), len(self.pres)))

        # Fill arrays with table data
        for i, p in enumerate(self.pres):
            Vp[:, i] = self.table[
                0 + (i * self.pstep) : self.pstep + (i * self.pstep), 2
            ]
            Vs[:, i] = self.table[
                0 + (i * self.pstep) : self.pstep + (i * self.pstep), 3
            ]
            Vp_an[:, i] = self.table[
                0 + (i * self.pstep) : self.pstep + (i * self.pstep), 4
            ]
            Vs_an[:, i] = self.table[
                0 + (i * self.pstep) : self.pstep + (i * self.pstep), 5
            ]
            Vphi[:, i] = self.table[
                0 + (i * self.pstep) : self.pstep + (i * self.pstep), 6
            ]
            Dens[:, i] = self.table[
                0 + (i * self.pstep) : self.pstep + (i * self.pstep), 7
            ]
            Qs[:, i] = self.table[
                0 + (i * self.pstep) : self.pstep + (i * self.pstep), 8
            ]
            T_sol[:, i] = self.table[
                0 + (i * self.pstep) : self.pstep + (i * self.pstep), 9
            ]

        # Setup interpolator objects. These can be used for rapid querying of many individual points
        self.vp_interp = interp2d(self.pres, self.temp, Vp)
        self.vs_interp = interp2d(self.pres, self.temp, Vs)
        self.vp_an_interp = interp2d(self.pres, self.temp, Vp_an)
        self.vs_an_interp = interp2d(self.pres, self.temp, Vs_an)
        self.vphi_interp = interp2d(self.pres, self.temp, Vphi)
        self.density_interp = interp2d(self.pres, self.temp, Dens)
        self.qs_interp = interp2d(self.pres, self.temp, Qs)
        self.t_sol_interp = interp2d(self.pres, self.temp, T_sol)

        # Creat dictionary which holds the interpolator objects
        self.fields = {
            "vp": [2, Vp, "km/s", self.vp_interp],
            "vs": [3, Vs, "km/s", self.vs_interp],
            "vp_ani": [4, Vp_an, "km/s", self.vp_an_interp],
            "vs_ani": [5, Vs_an, "km/s", self.vs_an_interp],
            "vphi": [6, Vphi, "km/s", self.vphi_interp],
            "density": [7, Dens, "$kg/m^3$", self.density_interp],
            "qs": [8, Qs, "Hz", self.qs_interp],
            "t_sol": [9, T_sol, "K", self.t_sol_interp],
        }

    #################################################
    # Need to get temp, pres, comp at given point.
    # Pressure could come from PREM or from simulation
    # Comp will be in 3 component mechanical mixture
    # We will then find the
    #################################################

    def interp_grid(self, press, temps, field):
        """
        Routine for re-gridding lookup tables into new pressure-temperature space
        :param press: Pressures (Pa)
        :type press: float or numpy array
        :param temps: Temperatures (K)
        :type temps: float or numpy array
        :param field: Data field (eg. 'Vs')
        :type field: string
        :return: interpolated values of a given table property
                on a 2D grid defined by press and temps
        :rtype: 2D numpy array
        :example:

        >>> t_test = [4,5,6]
        >>> p_test = 10
        >>> basalt = SeismicLookupTable('../tests/data/test_lookup_table.txt')
        >>> basalt.interp_grid(p_test, t_test, 'Vs')
        """

        press = [press] if type(press) == int or type(press) == float else press
        temps = [temps] if type(temps) == int or type(temps) == float else temps

        _check_bounds(press, self.pres)
        _check_bounds(temps, self.temp)
        grid = self.fields[field.lower()][3]
        return grid(press, temps)

    def interp_points(self, press, temps, field):
        """
        Routine for interpolating gridded property data at one or more
        pressure-temperature points.

        :param press: Pressures (Pa)
        :type press: float or numpy array
        :param temps: Temperatures (K)
        :type temps: float or numpy array
        :param field: Data field (eg. 'Vs')
        :type field: string

        :return: interpolated values of a given table property
                at points defined by press and temps
        :rtype: 1D numpy array
        """

        # If integers are passed in then convert to indexable lists
        press = [press] if type(press) == int or type(press) == float else press
        temps = [temps] if type(temps) == int or type(temps) == float else temps

        _check_bounds(press, self.pres)
        _check_bounds(temps, self.temp)

        grid = interp2d(self.pres, self.temp, self.fields[field.lower()][1])

        out = np.zeros(len(press))
        for i in range(len(press)):
            out[i] = grid(press[i], temps[i])

        return out

    def plot_table(self, ax, field, cmap="viridis_r"):
        """
        Plots the lookup table as a grid with values coloured by
        value for the field given.

        :param ax: matplotlib axis object to plot on.
        :type ax: matplotlib axis object
        :param field: Data field (eg. 'Vs')
        :type field: string
        :param cmap: matplotlib colourmap.
        :type cmap: string

        :return: None
        """

        # get column index for field of interest
        units = self.fields[field.lower()][2]
        data = self.fields[field.lower()][1]

        # temperature on x axis
        data = data.transpose()
        print(data.shape)

        chart = ax.imshow(
            data,
            origin="lower",
            extent=[self.t_min, self.t_max, self.p_min, self.p_max],
            cmap=cmap,
            aspect="auto",
        )

        # chart = ax.tricontourf(self.P,self.T,self.table[:,i_field])

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

        # get column index for field of interest
        i_field = self.fields[field.lower()][0]
        units = self.fields[field.lower()][1]
        data = self.table[:, i_field]

        chart = ax.tricontourf(self.P, self.T, self.table[:, i_field], cmap=cmap)

        # chart = ax.tricontourf(self.P,self.T,self.table[:,i_field])

        plt.colorbar(chart, ax=ax, label=f"{field} ({units})")
        ax.set_ylabel("Temperature (K)")
        ax.set_xlabel("Pressure (Pa)")
        ax.set_title(f"P-T graph for {field}")


class MultiTables:
    def __init__(self, lookuptables):
        """
        Class to take in and process multiple tables at once.

        :param tables: dictionary with keys describing the lookup table composition (e.g. "bas")
                    and the associated lookup table filename to be read in or array of values.
        :type tables: dictionary

        :return: multitable object.
        """
        self._tables = lookuptables
        self._lookup_tables = {}
        for key in self._tables:
            self._lookup_tables[key] = SeismicLookupTable(self._tables[key])

    def evaluate(self, P, T, fractions, field):
        """
        Returns the harmonic mean of a parameter over several lookup
        tables weighted by their fraction.

        :param P: pressure value to evaluate.
        :type P: float
        :param T: temperature value to evaluate.
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

    m_total = np.zeros(data[0].shape)

    for i in range(len(fractions)):
        m_total += (1 / data[i]) * fractions[i]

    hmean = np.sum(fractions) / (m_total)

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
        print(
            f"One or more of your inputs exceeds the table range, reverting to maximum table range"
        )

    elif np.any(input < np.min(check)):
        print(
            f"One or more of your inputs is below the table range, reverting to minimum table range"
        )

    # where the values are greater than the minimum keep the same
    # but replace those below with the minimum
    output = np.where(input > np.min(check), input, np.min(check))
    # where the values are smaller than the maximum keep the same
    # but replace those above with the maximum
    output = np.where(output < np.max(check), output, np.max(check))

    return output

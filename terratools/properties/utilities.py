import numpy as np
import pkgutil


def read_table(filename):
    """
    Reads a table from a packaged dataset within terratools.

    :param string filename: path to file

    :return: The table in the file in numpy array form.
    :rtype: numpy array
    """
    datastream = pkgutil.get_data("terratools", filename)
    datalines = [
        line.strip() for line in datastream.decode("ascii").split("\n") if line.strip()
    ]
    table = []

    for line in datalines:
        if line[0] != "#":
            numbers = np.fromstring(line, sep=" ")
            table.append(numbers)
    return np.array(table)


def Simon_Glatzel_fn(Pr, Tr):
    """
    Takes a reference pressure and temperature,
    and returns a Simon-Glatzel function [@Simon1929]
    f(P, a, b) = (Tr ((P - Pr)/a + 1)^b).

    :param float Pr: Reference pressure (Pa)
    :param float Tr: Reference temperature (K)

    :return: The Simon-Glatzel function
    :rtype: function
    """

    def Simon_Glatzel(P, a, b):
        return Tr * np.power(((P - Pr) / a + 1.0), b)

    return Simon_Glatzel

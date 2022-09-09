import numpy as np
import pkgutil


def read_table(filename):
    """
    Reads a table from a packaged dataset within terratools.
    """
    datastream = pkgutil.get_data('terratools', filename)
    datalines = [line.strip()
                 for line in datastream.decode('ascii').split('\n')
                 if line.strip()]
    table = []

    for line in datalines:
        if (line[0] != '#'):
            numbers = np.fromstring(line, sep=' ')
            table.append(numbers)
    return np.array(table)


def Simon_Glatzel_fn(Pr, Tr):
    def Simon_Glatzel(P, a, b):
        return Tr * np.power(((P - Pr)/a + 1.), b)
    return Simon_Glatzel

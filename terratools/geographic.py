"""
This module deals with transformation between spherical and
Cartesian coordinate systems.
"""

import numpy as np


def geog2cart(lon, lat, r, radians=False):
    """
    Convert coordinates in a geographic system to a Cartesian system.
    If all inputs are scalars, output is also scalar; while if all
    inputs are arrays, then outputs are arrays.  Behaviour is
    undefined if using a mix of scalars and arrays.

    The Cartesian system is defined as:
    - x passes through longitude = 0 and latitude = 0
    - y passes through longitude  = 90° and latitude = 0
    - z passes through the north pole.

    :param lon: Longitude of point(s) in degrees (default), or
        in radians if ``radians`` is ``True``
    :type lon: float or numpy array
    :param lat: Latitude of point(s) in degrees (default), or
        in radians if ``radians`` is ``True``
    :type lat: float or numpy array
    :param r: Radius of point(s).  The units of x, y and z will be
        the same as those of ``r``.
    :type r: float or numpy array
    :param radians: If ``True``, input are in radians; otherwise they
        are assumed to be in degrees.
    :type radians: bool
    :returns: for scalar input, the three coordinates x, y and z; for
        vector inputs, vectors of x, y and z
    :rtype: 3 floats or 3 numpy arrays
    """
    if np.any(r < 0):
        raise ValueError("radius cannot be negative")

    if not radians:
        lon = np.radians(lon)
        lat = np.radians(lat)

    x = r * np.cos(lon) * np.cos(lat)
    y = r * np.sin(lon) * np.cos(lat)
    z = r * np.sin(lat)

    return x, y, z


def cart2geog(x, y, z, radians=False):
    """
    Convert coordinates in a Cartesian system to a geographic one.
    If all coordinates are zero, then return all zeros for
    longitude, latitude and radius.

    If all inputs are scalars, output is also scalar; while if all
    inputs are arrays, then outputs are arrays.  Behaviour is
    undefined if using a mix of scalars and arrays.

    The Cartesian system is defined as:
    - x passes through longitude = 0 and latitude = 0
    - y passes through longitude  = 90° and latitude = 0
    - z passes through the north pole.

    :param x: x coordinate(s)
    :type x: float or numpy array
    :param y: y coordinate(s)
    :type y: float or numpy array
    :param z: z coordinate(s)
    :type z: float or numpy array
    :param radians: If ``True``, return output in radians; otherwise
        return output in degrees.
    :type radians: bool
    :returns: longitude, latitude and radius.  Longitude and latitude
        are in degrees, unless ``radians`` is ``True``.  Radius has the
        same units as x, y, and z.
    :rtype: 3 floats or 3 numpy arrays
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    scalar_input = np.isscalar(r)

    if scalar_input and r == 0:
        return 0.0, 0.0, 0.0

    lon = np.arctan2(y, x)
    lat = np.arcsin(z / r)

    # Fix any coordinates where r was zero
    if not scalar_input:
        undef_inds = np.where(r == 0)
        lon[undef_inds] = 0
        lat[undef_inds] = 0
        r[undef_inds] = 0

    if not radians:
        lon = np.degrees(lon)
        lat = np.degrees(lat)

    return lon, lat, r


def angular_distance(lon1, lat1, lon2, lat2, radians=False):
    """
    Compute the great circle distance between two points on the unit
    sphere.  The output is always equivalent to the angle in radians
    subtended between the vectors connecting the centre of the sphere
    to the two points.

    Note that all input angles are in degrees, unless ``radians`` is
    ``True``.

    :param lon1: Longitude of first point
    :type lon1: float or numpy array
    :param lat1: Latitude of first point
    :type lat1: float or numpy array
    :param lon2: Longitude of second point
    :type lon2: float or numpy array
    :param lat2: Latitude of second point
    :type lat2: float or numpy array
    :param radians: If True, input angles are in radians; otherwise they are
        in degrees.
    :type radians: bool
    :returns: angular distance between two points
    :rtype: float or numpy array
    """
    if not radians:
        lon1 = np.radians(lon1)
        lat1 = np.radians(lat1)
        lon2 = np.radians(lon2)
        lat2 = np.radians(lat2)

    sin_lat1 = np.sin(lat1)
    sin_lat2 = np.sin(lat2)
    cos_lat1 = np.cos(lat1)
    cos_lat2 = np.cos(lat2)
    cos_lon2_lon1 = np.cos(lon2 - lon1)

    distance = np.arctan2(
        np.sqrt(
            (cos_lat2 * np.sin(lon2 - lon1)) ** 2
            + (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_lon2_lon1) ** 2
        ),
        sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_lon2_lon1,
    )

    return distance


def angular_step(lon, lat, azimuth, distance, radians=False):
    """
    Compute the final point on the surface of the sphere reached by travelling
    along a great circle from some starting point along an initial azimuth.
    The distance covered is in terms of the angle subtended between the start
    and end points at the centre of the sphere.

    Note that all input angles are in degrees, unless ``radians`` is
    ``True``.

    :param lon: Longitude of starting point
    :type lon: float or numpy array
    :param lat: Latitude of starting point
    :type lat: float or numpy array
    :param azimuth: Azimuth along which to travel from starting point
    :type azimuth: float or numpy array
    :param distance: Great circle distance between starting and final points
        in terms of angle subtended between them at the centre of the sphere
    :type distance: float or numpy array
    :param radians: If ``True``, input angles are in radians and the output
        is in radians also; otherwise all input and output are in degrees.
    :type radians: book
    :returns: final longitude and latitude
    :rtype: float, float or two numpy arrays
    """
    if not radians:
        lon = np.radians(lon)
        lat = np.radians(lat)
        azimuth = np.radians(azimuth)
        distance = np.radians(distance)

    lat2 = np.arcsin(
        np.sin(lat) * np.cos(distance)
        + np.cos(lat) * np.sin(distance) * np.cos(azimuth)
    )
    lon2 = lon + np.arctan2(
        np.sin(azimuth) * np.sin(distance) * np.cos(lat),
        np.cos(distance) - np.sin(lat) * np.sin(lat2),
    )

    if not radians:
        lon2, lat2 = np.degrees(lon2), np.degrees(lat2)

    return lon2, lat2


def azimuth(lon1, lat1, lon2, lat2, radians=False):
    """
    Compute the azimuth from point 1 (lon1, lat1) to point 2
    (lon2, lat2) on the surface of a sphere.

    Note that all input angles are in degrees, unless ``radians`` is
    ``False``.

    :param lon1: Longitude of first point
    :type lon1: float or numpy array
    :param lat1: Latitude of first point
    :type lat1: float or numpy array
    :param lon2: Longitude of second point
    :type lon2: float or numpy array
    :param lat2: Latitude of second point
    :type lat2: float or numpy array
    :param radians: If True, input angles are in radians; otherwise they are
        in degrees.
    :type radians: bool
    :returns: angular distance between two points, in radians if
        ``radians`` is ``True``, and in degrees otherwise.
    :rtype: float or numpy array
    """
    if not radians:
        lon1 = np.radians(lon1)
        lat1 = np.radians(lat1)
        lon2 = np.radians(lon2)
        lat2 = np.radians(lat2)

    azimuth = np.arctan2(
        np.sin(lon2 - lon1) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1),
    )

    if not radians:
        return np.degrees(azimuth)
    else:
        return azimuth


def spherical_triangle_area(
    lon1, lat1, lon2, lat2, lon3, lat3, r=1, radians=False, tol=None
):
    """
    Return the spherical area covered by a spherical triangle defined
    by three geographic coordinates (lon1, lat1), (lon2, lat2) and
    (lon3, lat3), on a sphere with radius r.

    Note: Where two sides of the triangle are similar in length (and the third
    is about half the other two), l'Huilier's formula becomes unstable.  To guard
    against that case, we use a different expression;
    see [here](http://en.wikipedia.org/wiki/Spherical_trigonometry#Area_and_spherical_excess).

    Note that all input angles are in degrees, unless ``radians`` is
    ``False``.

    :param lon1: Longitude of first point
    :type lon1: float or numpy array
    :param lat1: Latitude of first point
    :type lat1: float or numpy array
    :param lon2: Longitude of second point
    :type lon2: float or numpy array
    :param lat2: Latitude of second point
    :type lat2: float or numpy array
    :param lon3: Longitude of third point
    :type lon3: float or numpy array
    :param lat3: Latitude of third point
    :type lat3: float or numpy array
    :param radians: Whether input is in radians
    :type radians: bool
    :param tol: Angle tolerance (always in radians) for collinearity test
        of three points
    :type tol: float
    :returns: spherical area of triangle
    :rtype: float or numpy array
    """

    if not radians:
        lon1 = np.radians(lon1)
        lat1 = np.radians(lat1)
        lon2 = np.radians(lon2)
        lat2 = np.radians(lat2)
        lon3 = np.radians(lon3)
        lat3 = np.radians(lat3)

    arc1 = angular_distance(lon2, lat2, lon3, lat3, radians=True)
    arc2 = angular_distance(lon3, lat3, lon1, lat1, radians=True)

    # Angle subtended at point 3
    az1 = azimuth(lon3, lat3, lon1, lat1, radians=True)
    az2 = azimuth(lon3, lat3, lon2, lat2, radians=True)

    # Change in azimuth
    c = np.abs(az2 - az1)
    c = np.minimum(c, 2.0 * np.pi - c)

    tan_arc1 = np.tan(arc1)
    tan_arc2 = np.tan(arc2)

    area = (
        2
        * np.arctan(
            tan_arc1 * tan_arc2 * np.sin(c) / (1 + tan_arc1 * tan_arc2 * np.cos(c))
        )
        * r**2
    )

    return area


def triangle_interpolation(
    lon, lat, lon1, lat1, val1, lon2, lat2, val2, lon3, lat3, val3, radians=False
):
    """
    Find the interpolated value of a point at (lon, lat), calculated
    by computing the weighted average of the three surrounding values.
    Hence the point of interest must lie within the triangle defined
    by the three other points.

    The weighting is computed as follows.  The three spherical triangles
    connecting the point of interest P to the outer triangle's vertices
    A, B and C are found.  Then the area of triangles APB, BPC and
    CPA are computed, and the weighted average of values a at A, b at B
    and c at C is computed by giving each value the weight of the area
    of its opposite triangle, divided by the total area.

    <pre>
                x A (point 1, val1)
               /'\\
              / ` \\
             /   | \\
            /   _+  \\
           /__--   ` \\
        C x__________x B (point 2, val2)
      (point 3, val3)
    </pre>

    Note that all input angles are in degrees, unless ``radians`` is
    ``False``.

    :param lon: Longitude of point of interest within the outer triangle.
    :type lon: float or numpy array
    :param lat: Latitude of piont of interest.
    :type lat: float or numpy array
    :param lon1: Longitude of first point
    :type lon1: float or numpy array
    :param lat1: Latitude of first point of surrounding triangle
    :type lat1: float or numpy array
    :param val1: Value at first point of surrounding triangle
    :type val1: float or numpy array
    :param lon2: Longitude of second point of surrounding triangle
    :type lon2: float or numpy array
    :param lat2: Latitude of second point of surrounding triangle
    :type lat2: float or numpy array
    :param val2: Value at second point of surrounding triangle
    :type val2: float or numpy array
    :param lon3: Longitude of third point of surrounding triangle
    :type lon3: float or numpy array
    :param lat3: Latitude of third point of surrounding triangle
    :type lat3: float or numpy array
    :param val3: Value at third point of surrounding triangle
    :type val3: float or numpy array
    :param radians: Whether input is in radians
    :type radians: bool
    :returns: interpolated value
    :rtype: float or numpy array
    """
    if not radians:
        lon = np.radians(lon)
        lat = np.radians(lat)
        lon1 = np.radians(lon1)
        lat1 = np.radians(lat1)
        lon2 = np.radians(lon2)
        lat2 = np.radians(lat2)
        lon3 = np.radians(lon3)
        lat3 = np.radians(lat3)

    area12 = spherical_triangle_area(lon, lat, lon1, lat1, lon2, lat2, radians=True)
    area23 = spherical_triangle_area(lon, lat, lon2, lat2, lon3, lat3, radians=True)
    area31 = spherical_triangle_area(lon, lat, lon3, lat3, lon1, lat1, radians=True)

    total_area = area12 + area23 + area31

    interpolated_value = (area23 * val1 + area31 * val2 + area12 * val3) / total_area

    return interpolated_value

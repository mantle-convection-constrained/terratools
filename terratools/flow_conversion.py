"""
This module deals with the conversion of flow velocity vectors
from cartesian to geographic vector. Taken from Andrew Walker's
package pyflowng (private repository).
"""

import numpy as np
import terratools.geographic as gt


def get_rotmat_to_geographical(lon, lat):
    """
    Return the rotation matrix at a longitude/latitude
    to convert the cartesian flow vector into
    geographical coordinates.

    :param lon: longitude
    :type lon: float
    :param lat: latitude
    :type lat: float
    :return trans: transformation matrix
    :rtype trans: 2D numpy array
    """

    # check longitude and latitude are integers or floats

    if isinstance(lat, np.ndarray):
        if lon.dtype.kind != "f" and lon.dtype.kind == "i":
            raise AssertionError("longitude needs to be integer or float")
    elif np.isscalar(lon):
        if not isinstance(lon, (int, float, np.floating, np.integer)):
            raise AssertionError("longitude needs to be integer or float")
    else:
        raise AssertionError("longitude type not acceptable")

    if isinstance(lat, np.ndarray):
        if lat.dtype.kind != "f" and lat.dtype.kind == "i":
            raise AssertionError("latitude needs to be integer or float")
    elif np.isscalar(lat):
        if not isinstance(lat, (int, float, np.floating, np.integer)):
            raise AssertionError("latitude needs to be integer or float")
    else:
        raise AssertionError("latitude type not acceptable")

    # Unit vectors on global system
    x_hat = np.array([1.0, 0.0, 0.0])
    y_hat = np.array([0.0, 1.0, 0.0])
    z_hat = np.array([0.0, 0.0, 1.0])

    # Find unit vector pointing to location in cart space
    (x0, y0, z0) = gt.geog2cart(lon, lat, 1.0)

    u_hat = np.array([x0, y0, z0])
    u_hat = u_hat / np.sqrt(np.sum(np.power(u_hat, 2.0)))

    # Local West and East unit vector(s)
    e_hat = np.cross(z_hat, u_hat)
    e_hat = e_hat / np.sqrt(np.sum(np.power(e_hat, 2.0)))
    w_hat = np.cross(u_hat, z_hat)
    w_hat = w_hat / np.sqrt(np.sum(np.power(w_hat, 2.0)))

    # Local North unit vector
    n_hat = np.cross(u_hat, e_hat)
    # No need to normalise, p and e are already orthoganol.

    # http://www.kwon3d.com/theory/transform/transform.html

    trans = np.array(
        [
            [np.dot(x_hat, e_hat), np.dot(y_hat, e_hat), np.dot(z_hat, e_hat)],
            [np.dot(x_hat, n_hat), np.dot(y_hat, n_hat), np.dot(z_hat, n_hat)],
            [np.dot(x_hat, u_hat), np.dot(y_hat, u_hat), np.dot(z_hat, u_hat)],
        ]
    )

    return trans


def rotate_vector(vec, lon, lat):
    """
    Convert the cartesian flow vector into
    geographical vector.

    :param vec: cartesian flow vector in [vx,vy,vz]
    :type vec: 1D numpy array of floats
    :param lon: longitude
    :type lon: float
    :param lat: latitude
    :type lat: float
    :return vec: geographical flow vector in [east, north, up] ('up' is radial +ve outwards)
    :rtype vec: 1D numpy array of floats
    """

    # convert to array if vec is a list
    vec = np.array(vec)

    if isinstance(vec, np.ndarray):
        if vec.dtype.kind != "f" and vec.dtype.kind != "i":
            raise AssertionError("flow vector needs to hold integers or floats.")
    else:
        raise AssertionError("flow vector needs to be a numpy array.")

    # get transformation matrix
    trans = get_rotmat_to_geographical(lon, lat)
    # apply transform to cartesian vector
    vec = np.matmul(vec, np.transpose(trans))
    return vec

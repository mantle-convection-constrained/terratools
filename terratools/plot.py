"""
plot
====

The `plot` module contains internal functions for performing
plotting of TerraModels and other classes in terratools.
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.stats


def layer_grid(lon, lat, radius, values, delta=None, extent=(-180, 180, -90, 90),
        label=None, method="nearest", **subplots_kwargs):
    """
    Take a set of arbitrary points in longitude and latitude, each
    with a different value, and create a grid of these values, which
    is then plotted on a map.

    :param lon: Set of longitudes in degrees
    :param lat: Set of latitudes in degrees
    :param radius: Radius of layer in km
    :param values: Set of the values taken at each point
    :param delta: Grid spacing in degrees
    :param extent: Tuple giving min longitude, max longitude, min latitude
        and max latitude (all in degrees), defining the region to plot
    :param label: Label for values; e.g. "Temperature / K"
    :param method: Can be one of:
        - "nearest": nearest neighbour only;
        - "mean": mean of all values within each grid point
    :param **kwargs: Extra keyword arguments passed to
        `matplotlib.pyplot.subplots`
    :returns: tuple of figure and axis handles, respectively
    """
    fig, ax = plt.subplots(
        subplot_kw={'projection': ccrs.EqualEarth()},
        **subplots_kwargs
    )

    if len(extent) != 4:
        raise ValueError("extent must contain four values")
    elif extent[1] <= extent[0]:
        raise ValueError("maximum longitude must be more than minimum; have" +
            f"min = {extent[0]} and max = {extent[1]}")
    elif extent[3] <= extent[2]:
        raise ValueError("maximum latitude must be more than minimum; have" +
            f"min = {extent[2]} and max = {extent[3]}")

    minlon, maxlon, minlat, maxlat = extent

    if delta is None:
        lonrange = maxlon - minlon
        latrange = maxlat - minlat
        max_range = max(lonrange, latrange)
        delta = max_range/200
    elif delta <= 0:
            raise ValueError("delta must be more than 0")

    grid_lons = np.arange(minlon, maxlon, delta)
    grid_lats = np.arange(minlat, maxlat, delta)
    grid_lon, grid_lat = np.meshgrid(grid_lons, grid_lats)

    if method == "nearest":
        grid = scipy.interpolate.griddata((lon, lat), values, (grid_lon, grid_lat),
            method="nearest")
    elif method == "mean":
        grid, _, _, _ = scipy.stats.binned_statistic_2d(lon, lat, values,
            bins=[grid_lons, grid_lats])
        grid = np.transpose(grid)
    else:
        raise ValueError(f"unsupported method '{method}")

    grid = np.flip(grid, axis=0)

    transform = ccrs.PlateCarree()

    contours = ax.imshow(grid, transform=transform)
    ax.set_title(f'Radius {int(radius)} km')
    ax.set_xlabel(f'{label}', fontsize=12)

    cbar = plt.colorbar(contours, ax=ax, orientation='horizontal', pad=0.05,
        aspect=30, shrink=0.5, label=(label if label is not None else ""))

    ax.coastlines()

    return fig, ax

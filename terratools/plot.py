"""
This submodule contains internal functions for performing
plotting of TerraModels and other classes in terratools.
"""

_CARTOPY_INSTALLED = True
_CARTOPY_NOT_INSTALLED_EXCEPTION = None

try:
    import cartopy.crs as ccrs
except ImportError as exception:
    _CARTOPY_INSTALLED = False
    _CARTOPY_NOT_INSTALLED_EXCEPTION = exception

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import scipy.interpolate
import scipy.stats
import sys


def layer_grid(
    lon,
    lat,
    radius,
    values,
    delta=None,
    extent=(-180, 180, -90, 90),
    label=None,
    method="nearest",
    coastlines=True,
    **subplots_kwargs,
):
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
        * "nearest": nearest neighbour only;
        * "mean": mean of all values within each grid point
    :param coastlines: If ``True`` (the default) use cartopy
        to plot coastlines.  Otherwise, do not plot coastlines.
        This works around an issue with Cartopy when
        installed in certain situations.  See
        https://github.com/SciTools/cartopy/issues/879 for details.
    :param **subplots_kwargs: Extra keyword arguments passed to
        `matplotlib.pyplot.subplots`
    :returns: tuple of figure and axis handles, respectively
    """
    if not _CARTOPY_INSTALLED:
        sys.stderr.write("layer_grid require cartopy to be installed")
        raise _CARTOPY_NOT_INSTALLED_EXCEPTION

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.EqualEarth(), **subplots_kwargs}
    )

    if len(extent) != 4:
        raise ValueError("extent must contain four values")
    elif extent[1] <= extent[0]:
        raise ValueError(
            "maximum longitude must be more than minimum; have"
            + f"min = {extent[0]} and max = {extent[1]}"
        )
    elif extent[3] <= extent[2]:
        raise ValueError(
            "maximum latitude must be more than minimum; have"
            + f"min = {extent[2]} and max = {extent[3]}"
        )

    minlon, maxlon, minlat, maxlat = extent

    if delta is None:
        lonrange = maxlon - minlon
        latrange = maxlat - minlat
        max_range = max(lonrange, latrange)
        delta = max_range / 200
    elif delta <= 0:
        raise ValueError("delta must be more than 0")

    grid_lons = np.arange(minlon, maxlon, delta)
    grid_lats = np.arange(minlat, maxlat, delta)
    grid_lon, grid_lat = np.meshgrid(grid_lons, grid_lats)

    if method == "nearest":
        grid = scipy.interpolate.griddata(
            (lon, lat), values, (grid_lon, grid_lat), method="nearest"
        )
    elif method == "mean":
        grid, _, _, _ = scipy.stats.binned_statistic_2d(
            lon, lat, values, bins=[grid_lons, grid_lats]
        )
        grid = np.transpose(grid)
    else:
        raise ValueError(f"unsupported method '{method}'")

    grid = np.flip(grid, axis=0)

    transform = ccrs.PlateCarree()

    contours = ax.imshow(grid, transform=transform, extent=extent)
    ax.set_title(f"Radius {int(radius)} km")
    ax.set_xlabel(f"{label}", fontsize=12)

    cbar = plt.colorbar(
        contours,
        ax=ax,
        orientation="horizontal",
        pad=0.05,
        aspect=30,
        shrink=0.5,
        label=(label if label is not None else ""),
    )

    # This leads to a segfault on machines where cartopy is not installed
    # from conda-forge, or where it was not built from source:
    # https://github.com/SciTools/cartopy/issues/879
    if coastlines:
        ax.coastlines()

    return fig, ax


def plot_section(
    distances, radii, grid, label=None, show=True, levels=25, cmap="turbo"
):
    """
    Create a plot of a cross-section.

    :param distances: Distances along cross section, given as the angle
        subtended at the Earth's centre between the starting and
        end points of the section, in degrees.
    :type distance: set of floats

    :param radii: Radii of cross section.
    :type minradius: set of floats

    :param grid: Values of the field evaluated at each distance and radius
        point, where the first axis given the distance index, and the second
        axis gives the radius index.
    :type grid: 2d array

    :param label: Label for colour scale
    :type label: str

    :param levels: Number of levels or set of levels to plot
    :type levels: int or set of floats

    :param cmap: Colour map to be used (default "turbo")
    :type cmap: str

    :param show: If `True` (default), show the plot
    :type show: bool

    :returns: figure and axis handles
    """

    distances_radians = np.radians(distances)
    min_distance = np.min(distances_radians)
    max_distance = np.max(distances_radians)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    contours = ax.contourf(distances_radians, radii, grid.T, levels=levels, cmap=cmap)
    ax.set_rorigin(0)
    ax.set_thetalim(min_distance, max_distance)
    # Rotate plot so that the middle of the section is up
    ax.set_theta_offset((min_distance + max_distance) / 2 + np.pi / 2)
    # Make distance increase to the right (i.e., clockwise)
    ax.set_theta_direction(-1)

    cbar = plt.colorbar(
        contours,
        ax=ax,
        orientation="horizontal",
        pad=0.05,
        aspect=30,
        shrink=0.5,
        label=(label if label is not None else ""),
    )

    if show:
        plt.show()

    return fig, ax


def spectral_heterogeneity(
    indat,
    title,
    depths,
    lmin,
    lmax,
    saveplot,
    savepath,
    lyrmin,
    lyrmax,
    **subplots_kwargs,
):
    """
    Creates a contour plot from the power spectrum over depth
    :param indat: array containing power spectrum at each radial layer.
        shape (nr,lmax+1)
    :param depths: array containing depths corresponding to power spectra
    :param lmin: minimum spherical harmonic degree to plot
    :param lmax: maximum spherical harmonic degree to plot
    :param saveplot: flag to save figure
    :param savepath: path under which to save figure
    :param lyrmin: minimum layer to plot
    :param lyrmax: maximum layer to plot
    :param **subplots_kwargs: Extra keyword arguments passed to
            `matplotlib.pyplot.subplots`
    :returns: tuple of figure and axis handles, respectively
    """

    logged = np.log(indat[lyrmin:lyrmax, lmin : lmax + 1])
    deps = depths[lyrmin:lyrmax]

    fig, ax = plt.subplots(figsize=(8, 6), **subplots_kwargs)

    plotmin = np.min(logged)
    plotmax = np.max(logged)
    levels = np.linspace(plotmin, plotmax, 10)
    cs = ax.contourf(np.arange(lmin, lmax + 1), deps, logged, levels=levels)
    ax.set_ylabel("Depth (km)", fontsize=12)
    ax.set_xlabel("L", fontsize=12)
    ax.set_xlim(lmin - 1, lmax + 1)
    if title == None:
        ax.set_title(f"Spherical Harmonic Power Spectrum")
    else:
        ax.set_title(f"Spherical Harmonic Power Spectrum \n for {title} field")
    plt.gca().invert_yaxis()
    cbar = fig.colorbar(cs, ax=ax, shrink=0.9, orientation="horizontal", pad=0.1)
    cbar.set_label("ln(Power)", fontsize=12)

    if saveplot:
        if savepath == None:
            savepath = "."
        if title == None:
            title = ""
        plt.savefig(
            f"{savepath}/powers_{title}.pdf", format="pdf", dpi=200, bbox_inches="tight"
        )

    return fig, ax


def plumes_3d(
    plmobj,
    elev=10,
    azim=70,
    roll=0,
    dist=20,
    cmap="terrain",
    **subplots_kwargs,
):
    """
    Generate 3D scatter plot of grid points which correspond to plumes
    coloured by plumeID. First convert from lon,lat,depth to cartesian
    for plotting.

    :param elev: camera elevation (degrees)
    :param azim: camera azimuth (degrees)
    :param roll: camera roll (degrees)
    :param dist: camera distance (unitless)
    :param cmap: string corresponding to matplotlib colourmap
    """

    nplms = plmobj.n_plms

    map = mpl.colormaps[cmap]
    maplin = map(np.linspace(0, 1, nplms))

    fig = plt.figure(figsize=(12, 12))
    ax = plt.subplot(projection="3d")

    ax.dist = dist

    for i in range(nplms):
        plm_pnts = plmobj.plm_coords[i]
        for j, depth in enumerate(plmobj.plm_depths[i]):
            r = plmobj._model.get_radii()[-1] - plm_pnts[j][:, 2]
            x, y, z = _latlon2xyz(plm_pnts[j][:, 1], plm_pnts[j][:, 0], r)
            if j == 0:
                ax.scatter(x, y, z, color=[maplin[i]], label=str(i))
            else:
                ax.scatter(x, y, z, color=[maplin[i]])

    # draw sphere
    u, v = np.mgrid[0 : 2 * np.pi : 100j, 0 : np.pi : 50j]
    x = 3480 * np.cos(u) * np.sin(v)
    y = 3480 * np.sin(u) * np.sin(v)
    z = 3480 * np.cos(v)
    ax.plot_surface(x, y, z, color="gray")
    ax.view_init(
        elev=elev,
        azim=azim,
        roll=roll,
    )
    ax.set_title(f"{nplms} plumes detected", y=0.9)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.legend(title="PlumeID", loc="lower center", ncol=int(nplms / 2))

    return fig, ax


def point(
    ax,
    lon,
    lat,
    color="red",
    size=4,
    text=None,
    textcolor="black",
    fontsize=11,
    **subplots_kwargs,
):
    """
    Plot point(s) onto a axis.
    :param ax: axis handle on which to draw
    :param lon: array of longitudinal points
    :param lat: array of latitudinal points
    :param color: color of points
    :param size: size of point to plot
    :param text: string to label point
    :param textcolor: color of text
    :param fontsize: fontsize for text:
    :param **subplots_kwargs: Extra keyword arguments passed to
            `matplotlib.pyplot.subplots`
    """
    transform = ccrs.PlateCarree()
    ax.scatter(lon, lat, transform=transform, color=color, s=size)
    if text != None:
        ax.text(
            lon, lat, f"{text}", transform=transform, c=textcolor, fontsize=fontsize
        )


def _latlon2xyz(lats, lons, r):
    """
    Lats, lons, r(kilometers) taken as input (flattened)
    returns cartesian x,y,z centered around the center
    of the Earth.
    """
    lats = np.deg2rad(lats)
    lons = np.deg2rad(lons)
    x = r * np.cos(lats) * np.cos(lons)
    y = r * np.cos(lats) * np.sin(lons)
    z = r * np.sin(lats)
    return x, y, z

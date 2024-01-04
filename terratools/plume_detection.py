"""
This module deals with the detection of plumes in a TerraModel.
Written by James Panton and Alistair Campbell.
"""

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans


class AlgorithmError(Exception):
    """
    Exception type raised when incorrect algorithm passed to plume_dbscan
    """

    def __init__(self, alg):
        self.message = f"'{alg}' is not a supported clustering algorithm, please use DBSCAN or HDBSCAN."
        super().__init__(self.message)


def plume_kmeans(model, depth_range=(400, 2600), n_init="auto"):
    """
    Performs K-means clustering on the model at each depth containing a layer within the range specified
    Currently only works when nclust=2

    :param model: TerraModel object
    :param depth_range: (min_depth, max_depth) over which to look for plumes
    :param n_init: number of times to run k-means with different starting centroids
    :return plume_binary: Array of shape (maxlyr-minlyr+1,nps) containing binary
        information on whether a plume was detected
    :return toplot: Layers of the TerraModel which were searched for plumes
    :return depths_flip: Corresponding depths of layers which were searched for plumes.
    """

    # We will be looking for 2 clusters - those which are plume like and those which are not.
    nclust = 2

    interpolation = "nearest"  # for gridding of data - may be linear, nearest, cubic

    # Separate the minimum and maximum depths to be used by K-means
    mindepth = depth_range[0]
    maxdepth = depth_range[1]

    # Determine the closest layers to the specified min and max depths
    minlyr, mindepth = model.nearest_layer(mindepth, depth=True)
    maxlyr, maxdepth = model.nearest_layer(maxdepth, depth=True)

    # get the depths of each layer used in the model
    depths = model.get_radii()[-1] - model.get_radii()
    depths_flip = np.flip(depths[maxlyr : minlyr + 1])

    # Create a list of all of the depth layers in the model that are to be used by K-means
    toplot = np.arange(minlyr, maxlyr - 1, -1)

    # Define the longitudes and latitudes of the gridpoints on the model
    lonlat = np.zeros((model.get_lateral_points()[1].shape[0], 2))
    lonlat[:, 0], lonlat[:, 1] = model.get_lateral_points()

    # Initialize array to store
    plume_binary = np.zeros((np.shape(lonlat)[0], len(toplot)))

    # Loop over each depth layer in the model, using K-means clustering on the data for each layer.
    for depth_num, layer in enumerate(toplot):
        ######Produce an array containing only the grid points which are members of the final cluster (highest temperature and velocity)
        #####plume_binary[plume_binary<(nclust-1)]=0

        # Obtain the temperatures and velocities from the model at the current depth number
        data_t = model.get_field("t")[layer, :]
        data_c = model.calc_bulk_composition()
        data_v = model.get_field("u_enu")[layer, :, 2]

        # Determine the temperature values which are greater than the layer mean
        data_t_bool = data_t > np.mean(data_t)
        # And only positive radial velocities
        data_v_bool = data_v > 0

        #         #Normalise the temperature and radial velocity fields such that their largest values must equal 1
        #         #Next, replace all of the below average values with zero
        data_t_norm = (
            normalize(data_t.reshape(1, -1) - np.mean(data_t))[0] * data_t_bool
        )
        data_v_norm = normalize(data_v.reshape(1, -1))[0] * data_v_bool

        # data_t_norm=(data_t-np.mean(data_t))*data_t_bool
        # data_t_norm=
        # data_v_norm=data_v*data_v_bool
        # data_v_norm=data_v**2

        # Multiply the new normalised t and v fields together
        prod_t_v = data_t_norm * data_v_norm

        # Perform K-means cluster analysis on the combined t and v fields
        clusters = kmeans_analysis(prod_t_v, nclust, n_init=n_init)

        # #Combine the results from each layer to form a 3D array
        plume_binary[:, depth_num] = clusters.flatten()

    # Produce an array containing only the grid points which are members of the final cluster
    # (highest temperature and velocity)
    plume_binary[plume_binary < (nclust - 1)] = 0

    return plume_binary, toplot, depths_flip


def kmeans_analysis(gridin, nclusts, n_init="auto"):
    """
    perform kmeans clusters analysis on given grid using sklearn package
    for nclust clusters. Return grid filled with values 0 to nclust-1.

    :param gridin: Input data for K-means analysis
    :param nclusts: Number of clusters to search for
    :param n_init: Numer of times to run K-means with different starting centroids
    :return result_clusts: Result of K-means analysis with clusters numerated
    """
    from sklearn.cluster import KMeans

    # Perform K-means cluster analysis
    kmeans = KMeans(n_clusters=nclusts, random_state=0, n_init=n_init).fit(
        gridin.reshape(-1, 1)
    )
    result = kmeans.cluster_centers_[kmeans.labels_]
    # cluster_pic=pic2show[:,0].reshape(grid_shape[0], grid_shape[1])

    # Numerate the clusters
    result_clusts = np.zeros_like(result)
    for i, uval in enumerate(np.unique(result)):
        mask = result == uval
        result_clusts = result_clusts + (mask * i)

    return result_clusts


def plume_dbscan(
    model,
    kmeans,
    algorithm="HDBSCAN",
    epsilon=150,
    minsamples=150,
    depth_range=(400, 2600),
):
    """
    Performs either DBSCAN or HDBSCAN on the output array from the K-means function.

    :param model: Input TerraModel
    :param kmeans: Result from the K-means analysis
    :param algorithm: Spatial clustering algorithm - 'DBSCAN' and 'HDBSCAN' supported
    :param epsilon: Threshold distance parameter for DBSCAN, min_cluster_size for HDBSCAN
    :param minsamples: Minimum number of samples in a neighbourhood for DBSCAN and HDBSCAN
    :param depth_range: (min_depth, max_depth) over which to look for plumes

    :return labels: Corresponding cluster label for each input point (-1 = noise)
    :return n_clusts: Number of clusters detected
    :return n_noise: Number of noise points
    """

    # Separate the minimum and maximum depths to be used by the clustering algorithm
    mindepth = depth_range[0]
    maxdepth = depth_range[1]

    # Get the depths of each layer used in the model
    # nr=len(model.get_radii())
    radii = model.get_radii()
    depths = model.get_radii()[-1] - model.get_radii()

    # Reorder the depths array to go from the bottom of the mantle to the top
    depths_flip = np.flip(depths)  # [13:58] (cut)

    # Determine the closest layers to the specified min and max depths
    minlyr, mindepth = model.nearest_layer(mindepth, depth=True)
    maxlyr, maxdepth = model.nearest_layer(maxdepth, depth=True)

    # Create a list of all of the depth layers in the model that are to be used by density based clustering
    toplot = np.arange(minlyr, maxlyr - 1, -1)

    # Define the longitudes and latitudes of the gridpoints on the model
    lonlat = np.zeros((model.get_lateral_points()[1].shape[0], 3))
    lonlat[:, 0], lonlat[:, 1] = model.get_lateral_points()

    # Create an array containing the coordinates of each point detected
    # as being part of a plume by K-means
    pnts_in_plm = np.argwhere(kmeans)
    pnts = np.zeros((np.shape(pnts_in_plm)[0], 3))
    n = 0
    for i, layer in enumerate(toplot):
        boolarr = kmeans[:, i].astype(dtype=bool)
        pnts[n : (n + np.sum(boolarr)), 0] = lonlat[:, 0][boolarr]  # fill lons
        pnts[n : (n + np.sum(boolarr)), 1] = lonlat[:, 1][boolarr]  # fill lats
        pnts[n : (n + np.sum(boolarr)), 2] = radii[layer]  # fill depths
        n = n + np.sum(boolarr)

    # Create a normalised version of the coordinate array
    pnts_norm = np.zeros(np.shape(pnts))
    pnts_norm[:, 0] = pnts[:, 0] / np.max(pnts[:, 0])
    pnts_norm[:, 1] = pnts[:, 1] / np.max(pnts[:, 1])
    pnts_norm[:, 2] = pnts[:, 2] / np.max(pnts[:, 2])

    if algorithm == "DBSCAN":  # Perform DBSCAN on the data
        # EPSILON=10
        # MINSAMPLES=100

        density_scan = DBSCAN(eps=epsilon, min_samples=minsamples).fit(pnts_norm)

    elif algorithm == "HDBSCAN":  # Perform HDBSCAN on the data
        # MINCLUST=150
        # MINSAMPLE=150

        density_scan = HDBSCAN(min_cluster_size=epsilon, min_samples=minsamples).fit(
            pnts_norm
        )

    else:
        raise AlgorithmError(algorithm)

    # Assign each point in the data a label equal to the number of the cluster
    # that it was assigned to by the density based clustering algorithm
    labels = density_scan.labels_

    # Determine the number of plumes
    n_clusts = len(set(labels)) - (1 if -1 in labels else 0)

    # Determine the number of noise points
    n_noise = list(labels).count(-1)

    print(f"Detected {n_clusts} plumes with {n_noise} noise points")

    return labels, n_clusts, n_noise


def plume_centroids(plumeID, plm_obj):
    """
    Returns the coordinates of the centroids of each depth layer within the plume

    :param plumeID: plumeID number for the plume in question
    :param plm_obj: Plume object
    :return plume_nth_centroids: Centroid of plume at each radial layer that it
        was detected
    """
    # Select the points within plume 'plumeID'
    plume_nth = plm_obj._pnts_plms[plm_obj._plm_clusts == plumeID]

    # Obtain a list of depths occupied by the plume
    plume_nth_depths = np.unique(plume_nth[:, 2])

    # Create an array to contain the lon, lat and depth coordinates of the centroid of each depth
    plume_nth_centroids = np.zeros((len(plume_nth_depths), 3))

    # Calculate the centroids of each layer
    for i, j in enumerate(plume_nth_depths):
        mask = plume_nth[:, 2] == j
        plume_nth_depth = plume_nth[mask]
        lon_cent, lat_cent = get_centre(plume_nth_depth[:, 0], plume_nth_depth[:, 1])

        plume_nth_centroids[i, 0] = lon_cent
        plume_nth_centroids[i, 1] = lat_cent
        plume_nth_centroids[i, 2] = j

    return plume_nth_centroids


# def plume_radials(reduced_field,plm_obj,plumeID):
#    """
#    Get the radial stats
#    """
#
#    fld=


def get_centre(lons, lats):
    """
    Function to take in data lons and lats of a contour line and
    returns the centroid lon, lat

    :param lons: Input longitudes
    :param lats: Input latitudes
    :return lon_out: longitude of centroid
    :return lat_out: latitude of centroid
    """

    bigX = 0.0
    bigY = 0.0
    bigZ = 0.0

    if np.size(lons) != np.size(lats):
        print("Error!: lons and lats inputs must be same size")
        # sys.exit()
    else:
        npts = np.size(lons)
        for lon, lat in zip(lons, lats):
            lon_r = deg2rad(lon)
            lat_r = deg2rad(lat)

            a = np.cos(lat_r) * np.cos(lon_r)
            b = np.cos(lat_r) * np.sin(lon_r)
            c = np.sin(lat_r)

            bigX += a
            bigY += b
            bigZ += c

        bigX /= npts
        bigY /= npts
        bigZ /= npts

        lon_out = np.arctan2(bigY, bigX)
        hyp = np.sqrt(bigX * bigX + bigY * bigY)
        lat_out = np.arctan2(bigZ, hyp)

        lon_out = rad2deg(lon_out)
        lat_out = rad2deg(lat_out)
    return lon_out, lat_out


def rad2deg(indat):
    """
    convert radians to degrees

    :param indat: Input radians
    :return outdat: Output degress
    """
    outdat = indat * 180.0 / np.pi
    return outdat


def deg2rad(indat):
    """
    convert degrees to radians

    :param indat: Input degrees
    :return outdat: Output radians
    """
    outdat = indat * np.pi / 180.0
    return outdat

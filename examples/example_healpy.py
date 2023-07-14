from terratools import convert_files
import glob
from tests import test_convert_files
import os
from terratools import terra_model
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # %% [markdown]
    """
    For the purposes of this example we first create some example 'old' data
    files and convert to 'new' files.
    """

    directory = "./example_nc_files/"
    try:
        os.mkdir(f"{directory}")
    except:
        pass
    nfiles = 4
    for filen in range(nfiles):
        oldfilepath = os.path.join(directory, f"example_file_{filen}.nc")
        oldfile = test_convert_files.make_old_file(oldfilepath)
    files = glob.glob(f"{directory}/example_file_*.nc")
    convert_files.convert(files, test=True)

    # %% [markdown]
    """
    We can now read the files in as a TerraModel
    """

    model = terra_model.read_netcdf(files)

    # %% [markdown]
    """
    We can convert the composition histograms of harzburgite, lherzolite and basalt
    into a bulk composition scalar field using `model.calc_bulk_composition()`
    """

    model.calc_bulk_composition()

    # %% [markdown]
    """
    We can perform spherical harmonic decompositions of scalar field. In terratools we make
    use of healpy (python wrapper for healpix) to do this.

    `hp_sph(self, field, fname, nside=2**6, lmax=16, savemap=False)`:

        :param `field`: input field of shape (nr, nps)
        :param `fname`: string, name under which to save spherical harmonic
            coefficients and power spectra ``data.sph[fname]``
        :param `nside`: healpy param, number of sides for healpix grid, power
                      of 2 less than 2**30 (default 2**6)
        :param `lmax`: maximum spherical harmonic degree (default 16)
        :param `savemap`: Default (``False``) do not save the healpix map

    For example we would do:
    `model.hp_sph(model.get_field('t'),'temp')`
    to get the spherical harmonic coefficients and power spectra of the temperature field.
    """

    model.hp_sph(model.get_field("t"), "temp")
    model.hp_sph(model.get_field("c"), "comp")

    for key in model.sph.keys():
        print(key)

    # %% [markdown]
    """
    We can plot the power spectra of the for fields as a function of depth using
    `model.plot_spectral_heterogeneity('fname')

    `plot_spectral_heterogeneity(self,fname,title=None,saveplot=False,savepath=None,
        lmin=1,lmax=None,lyrmin=1,lyrmax=-1,show=True,**subplots_kwargs,)`:

        Plot spectral heterogenity maps of the given field, that is the power
        spectrum over depth.
        :param `fname`: name of field to plot as created using model.hp_sph()
        :param `title`: title of plot (string)
        :param `saveplot`: flag to save an image of the plot to file
        :param `savepath`: path under which to save plot to
        :param `lmin`: minimum spherical harmonic degree to plot (default=1)
        :param `lmax`: maximum spherical harmonic degree to plot (default to plot all)
        :param `lyrmin`: min layer to plot (default omits boundary)
        :param `lyrmax`: max layer to plot (default omits boundary)
        :param `show`: if True (default) show the plot
        :param **subplot_kwargs: Extra keyword arguments passed to
            `matplotlib.pyplot.subplots`
        :returns: figure and axis handles


    """

    fig, ax = model.plot_spectral_heterogeneity("comp", lyrmin=0, lyrmax=-1, show=False)

    plt.show()

    # %% [markdown]
    """
    We can also plot depth slices of the spectrally filtered fields using
    `model.plot_hp_map('fname',index=1)`

    `plot_hp_map(self,fname,index=None,radius=None,nside=2**6,title=None,delta=None,
        extent=(-180, 180, -90, 90), method="nearest",show=True,**subplots_kwargs,)`:

        Create heatmap of a field recreated from the spherical harmonic coefficients
        :param `fname`: name of field as created using ``data.hp_sph()``
        :param `index`: index of layer to plot
        :param `radius`: radius to plot (nearest model radius is shown)
        :param `nside`: healpy param, number of sides for healpix grid, power
            of 2 less than 2**30 (default 2**6)
        :param `title`: name of field to be included in title
        :param `delta`: Grid spacing of plot in degrees
        :param `extent`: Tuple giving the longitude and latitude extent of
            plot, in the form (min_lon, max_lon, min_lat, max_lat), all
            in degrees
        :param `method`: May be one of: "nearest" (plot nearest value to each
            plot grid point); or "mean" (mean value in each pixel)
        :param `show`: If True (the default), show the plot
        :param **subplot_kwargs: Extra keyword arguments passed to
            `matplotlib.pyplot.subplots`
        :returns: figure and axis handles

    Note that either `index` or `radius` must be passed in
    """
    model.plot_hp_map("temp", index=1, nside=2**5, show=False)

    plt.show()

    os.system(f"rm -rf {directory}")

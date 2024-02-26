import netCDF4 as nc4
import sys
import os
import stat
import numpy as np
from distutils.spawn import find_executable


class FileTypeError(Exception):
    """
    Exception type raised when trying to convert wrong file type
    """

    def __init__(self, file):
        self.message = (
            f"Conversion of .seis files not supported. "
            + "Please convert .comp files and use terra_model to determine seismic properties."
        )
        super().__init__(self.message)


class DepthDimError(Exception):
    """
    Exception type raised when trying to convert layer file that already has depth dimension
    """

    def __init__(self, file):
        self.message = f"File {file} already contains a depth dimension, conversion is not supported."
        super().__init__(self.message)


class CopyError(Exception):
    """
    Exception type raised when values of variable in old file and new file do not match
    """

    def __init__(self, file, var):
        self.message = f"Variable {var} in file {file} was incorrectly converted"
        super().__init__(self.message)


def convert(files, test=False):
    """
    Call to convert files from 'old' (pre-versioning) Terra netCDF files to the new standard.
    Only .comp file types may be converted as the 'old' .seis files have sesimic velocities
    written as perturbation from the radial mean while the new ones write the absolute values.

    :param files: List of files to be converted
    """

    # Check if ncks exists - for cleaning up old variables
    cleanup = _tool_exists("ncks")

    for file in files:
        if not test:
            # ensure write permissions before opening
            os.chmod(
                file,
                stat.S_IWUSR
                | stat.S_IWGRP
                | stat.S_IWOTH
                | stat.S_IRUSR
                | stat.S_IRGRP
                | stat.S_IROTH,
            )
        data = nc4.Dataset(file, mode="a")
        depths = data["Depths"][:]
        variables = (
            data.variables.copy()
        )  # Copy so not overwriting dictionary keys in loop
        for var in variables:
            if "anelastic" in var:
                raise (FileTypeError(file))

        path, fname = file.rsplit("/", 1)
        for i, var in enumerate(variables):
            # Skip compositions and coordinates for now
            if "Frac" in var or "Latitude" in var or "Longitude" in var:
                continue

            # rename velocity and Depth units in lower case
            if "Velocity" in var or "Depths" in var:
                data[var].units = data[var].units.lower()

            # rename variables to lowercase
            try:
                data.renameVariable(var, var.lower())
            except:
                continue

        # rename depth
        data.renameDimension("Depths", "depths")
        data.renameVariable("Latitude", "Lat_old")
        data.renameVariable("Longitude", "Lon_old")

        # create new variable
        comp_dim = data.createDimension("compositions", 2)
        comp_fracs_var = data.createVariable(
            "composition_fractions", np.float64, ("compositions", "depths", "nps")
        )

        harzfrac = 1 - data["BasaltFrac"][:, :] - data["LherzFrac"][:, :]
        comp_fracs_var[0, :, :] = harzfrac
        comp_fracs_var[1, :, :] = data["LherzFrac"][:, :]
        comp_fracs_var.composition_1_name = "Harzburgite"
        comp_fracs_var.composition_1_c = 0.0
        comp_fracs_var.composition_2_name = "Lherzolite"
        comp_fracs_var.composition_2_c = 0.2
        comp_fracs_var.composition_3_name = "Basalt"
        comp_fracs_var.composition_3_c = 1.0

        # cannot remove a single dimension from coordinates
        lat_var = data.createVariable("latitude", np.float64, ("nps"))
        lon_var = data.createVariable("longitude", np.float64, ("nps"))

        lat_var[:] = data["Lat_old"][0, :]
        lat_var.units = "degrees"
        lon_var[:] = data["Lon_old"][0, :]
        lon_var.units = "degrees"

        data["depths"][:] = depths

        # Global variables
        data.version = 1.0
        data.nth_comp = "bas_frac = 1 - hzb_frac - lhz_frac"

        data.close()

        # Have to use ncks (available through NCO - netCDF operators)
        if cleanup and not test:
            os.system(f"ncks -C -O -x -v BasaltFrac {path}/{fname} {path}/{fname}_new")
            os.system(f"mv {path}/{fname}_new {path}/{fname}")
            os.system(f"ncks -C -O -x -v LherzFrac {path}/{fname} {path}/{fname}_new")
            os.system(f"mv {path}/{fname}_new {path}/{fname}")
            os.system(f"ncks -C -O -x -v Lon_old {path}/{fname} {path}/{fname}_new")
            os.system(f"mv {path}/{fname}_new {path}/{fname}")
            os.system(f"ncks -C -O -x -v Lat_old {path}/{fname} {path}/{fname}_new")
            os.system(f"mv {path}/{fname}_new {path}/{fname}")
        elif not cleanup and not test:
            print("ncks is not available on your PATH so cannot clean up old variables")
            print("ncks is available with NCO (NetCDF Operators)")


def convert_layer(files, newfile_suff="convert", replace=False):
    """
    Converts old single layer files into new format which includes depth dimension

    :param files: list of files to convert
    :type files: list of strings

    :param newfile_suff: string to append to new files
    :type newfile_suff: str

    :param replace: toggle replacement of old files
    :type replace: bool
    """

    for file in files:
        dat = nc4.Dataset(file)
        newfile_name = f"{file}_{newfile_suff}"
        if os.path.exists(newfile_name):
            os.remove(newfile_name)
        _touch(newfile_name)
        os.chmod(
            newfile_name,
            stat.S_IWUSR
            | stat.S_IWGRP
            | stat.S_IWOTH
            | stat.S_IRUSR
            | stat.S_IRGRP
            | stat.S_IROTH,
        )
        newfile = nc4.Dataset(newfile_name, "w", format="NETCDF3_CLASSIC")
        for name in dat.ncattrs():
            newfile.setncattr(name, dat.getncattr(name))
        # copy dimensions
        for name in dat.dimensions:
            if name == "depths" or name == "depth":
                raise (DepthDimError(file))
            if dat.dimensions[name].isunlimited():
                newfile.createDimension(name, None)
            else:
                size = dat.dimensions[name].size
                newfile.createDimension(name, size)
        # Now add the depth dimension
        newfile.createDimension("depths", 1)

        for name, variable in dat.variables.items():
            if len(variable.dimensions) == 2:
                x = newfile.createVariable(
                    name,
                    variable.datatype,
                    (
                        variable.dimensions[0],
                        "depths",
                        variable.dimensions[1],
                    ),
                )
            else:
                name_new = name.replace(" ", "_")
                x = newfile.createVariable(
                    name_new, variable.datatype, ("depths",) + variable.dimensions
                )
            newfile.variables[name_new][:] = dat.variables[name][:]
            for attr in variable.ncattrs():
                val = variable.getncattr(attr)
                x.setncattr(attr, val)

            # test that old file and new file have same contents
            if not np.all(newfile[name_new][:] == dat[name][:]):
                raise CopyError(file, name)

        # create new depth variable
        x = newfile.createVariable("depths", variable.datatype, ("depths",))
        try:
            depth_from_attr = dat.getncattr("depth (km)")
        except:
            depth_from_attr = 0.0
        newfile.variables["depths"][:] = depth_from_attr
        x.setncattr("units", "km")
        x.setncattr("radius", 6370.0)

        newfile.version = 1.0
        newfile.close()

        if replace:
            # remove old file then rename newfile
            os.remove(file)
            os.rename(newfile_name, file)


def _touch(path):
    with open(path, "a"):
        os.utime(path, None)


def _tool_exists(toolname):
    """Check whether `toolname` exists on PATH."""

    return find_executable(toolname) is not None


if __name__ == "__main__":
    import sys
    import glob

    file_base = sys.argv[1]
    file_suffix = sys.argv[2]

    files = glob.glob(f"{file_base}*{file_suffix}")
    if len(file_base) == 0:
        sys.stderr.write(
            f"""

        usage: python {file_base} {file_suffix}

        Positionsal Arguments:
        file_base            : base of files to be converted including path
        file_suffix          : suffix of files to be converted

        """
        )
        sys.exit(1)

    convert(files)

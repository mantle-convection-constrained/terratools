import os
import shutil
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from datetime import datetime

from terratools.properties.perplex import make_build_files
from terratools.properties.perplex import run_build_files
from terratools.properties.perplex import perplex_to_grid
from terratools.properties.attenuation import Q7g

if __name__ == "__main__":
    perplex_path = ""

    url_datafiles = "https://www.perplex.ethz.ch/perplex/datafiles"
    url_sol = f"{url_datafiles}/stx21_solution_model.dat"
    url_mbr = f"{url_datafiles}/stx21ver.dat"
    urllib.request.urlretrieve(url_sol, "stx21_solution_model.dat")
    urllib.request.urlretrieve(url_mbr, "stx21ver.dat")

    # In this script, we obtain properties for lherzolite
    project_name = "output-lhz"
    molar_composition = {
        "SIO2": 38.819,
        "MGO": 49.894,
        "FEO": 6.145,
        "CAO": 2.874,
        "AL2O3": 1.963,
        "NA2O": 0.367,
    }

    # Split the PerpleX computations into four parts
    # using the following P-T boundaries
    pressure_bounds = [0.0, 30.0e9, 150.0e9]
    temperature_bounds = [200.0, 2200.0, 4500.0]
    endmember_file = "./stx21ver.dat"
    solution_file = "./stx21_solution_model.dat"
    option_file = "../terratools/properties/data/perplex_option.dat"
    solutions = [
        "C2/c",
        "Wus",
        "Pv",
        "Pl",
        "Sp",
        "O",
        "Wad",
        "Ring",
        "Opx",
        "Cpx",
        "Aki",
        "Gt",
        "Ppv",
        "CF",
        "NaAl",
    ]
    excludes = []

    # Delete preexisting directory if it exists
    try:
        shutil.rmtree(project_name, ignore_errors=False, onerror=None)
    except FileNotFoundError:
        pass

    # Make PerpleX build files
    make_build_files(
        project_name,
        molar_composition,
        pressure_bounds,
        temperature_bounds,
        endmember_file,
        solution_file,
        option_file,
        solutions,
        excludes,
    )

    # Remove downloaded files that are now copied to the project directory
    os.remove(endmember_file)
    os.remove(solution_file)

    if perplex_path != "":
        # Run the build files in the project directory
        # through vertex and pssect
        run_build_files(project_name, perplex_path)

        # Create the property grid at the desired resolution
        pressures = np.linspace(0.0, 145.0e9, 146)
        temperatures = np.linspace(300.0, 3300.0, 121)
        grid = perplex_to_grid(
            project_name,
            pressure_bounds,
            temperature_bounds,
            pressures,
            temperatures,
            perplex_path,
        )

        out = grid

        # Make a plot of the properties
        fig = plt.figure(figsize=(10, 8))
        ax = [fig.add_subplot(2, 2, i) for i in range(1, 4)]

        im0 = ax[0].contourf(
            out[:, :, 0] / 1.0e9, out[:, :, 1], out[:, :, 2], 100, cmap="rainbow"
        )
        im1 = ax[1].contourf(
            out[:, :, 0] / 1.0e9, out[:, :, 1], out[:, :, 3], 100, cmap="rainbow"
        )
        im2 = ax[2].contourf(
            out[:, :, 0] / 1.0e9, out[:, :, 1], out[:, :, 4], 100, cmap="rainbow"
        )

        im1.set_clim(0, 15)

        c0 = fig.colorbar(im0, ax=ax[0])
        c1 = fig.colorbar(im1, ax=ax[1])
        c2 = fig.colorbar(im2, ax=ax[2])

        c0.ax.set_ylabel("Density (kg/m$^3$)")
        c1.ax.set_ylabel("Vp (km/s)")
        c2.ax.set_ylabel("Vs (km/s)")

        for i in range(3):
            ax[i].set_xlabel("Pressure (GPa)")
            ax[i].set_ylabel("Temperature (K)")

        fig.tight_layout()
        plt.show()

        # Calculate the anelastic properties

        # First, flatten the property table
        out = out.reshape(out.shape[0] * out.shape[1], 5)

        P, T, rho, Vp, Vs = out.T
        frequency = 1.0

        p0 = Q7g.anelastic_properties(
            elastic_Vp=Vp, elastic_Vs=Vs, pressure=P, temperature=T, frequency=frequency
        )

        # Pressure(Pa),
        # Temperature(K),  Vp(elastic(km s-1)),  Vs(elastic(km s-1)),
        # Vp(anelastic), Vs(anelastic),
        # Bulk velocity(elastic(km s-1)),  Density (g cm-3), QS
        Vphi = np.sqrt(Vp * Vp - 4.0 / 3.0 * Vs * Vs)
        out_data = np.array(
            [P, T, Vp, Vs, p0.V_P, p0.V_S, Vphi, rho, p0.Q_S, p0.T_solidus]
        ).T

        header = (
            f"P(Pa) T(K) Vp(elastic, km/s) "
            "Vs(elastic, km/s) Vp(anelastic, "
            f"{frequency} Hz, km/s) "
            f"Vs(anelastic, {frequency} Hz, km/s) "
            "Vphi(elastic, km/s) Density(kg/m^3) "
            f"Q_S({frequency} Hz) T(solidus, K). "
            f"File produced on {datetime.today()}"
        )

        np.savetxt(
            f"{project_name}_anelastic_properties.dat",
            out_data,
            fmt="%.5f",
            header=header,
        )

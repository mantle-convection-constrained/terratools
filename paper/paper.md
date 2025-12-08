---
title: 'terratools: A Python package to analyse TERRA mantle convection simulations'
tags:
  - Python
  - Earth sciences
  - mantle convection
authors:
  - name: Andy Nowacki
    orcid: 0000-0001-7669-7383
    equal-contrib: true
    affiliation: 1
  - name: James Panton
    orcid: 0000-0002-1345-9406
    affiliation: 2
  - name: Jamie Ward
    orcid: 0000-0001-5697-4590
    affiliation: 1
  - name: Bob Myhill
    orcid: 0000-0001-9489-5236
    affiliation: 3
  - name: Andrew Walker
    orcid: 0000-0003-3121-3255
    affiliation: 4
  - name: James Wookey
    orcid: 0000-0002-7403-4380
    affiliation: 3
  - name: J. Huw Davies
    orcid: 0000-0003-2656-0260
    affiliation: 2

affiliations:
 - name: School of Earth and Environment, University of Leeds, UK
   index: 1
 - name: School of Earth and Environmental Sciences, Cardiff University, UK
   index: 2
 - name: School of Earth Sciences, University of Bristol, UK
   index: 3
 - name: Department of Earth Sciences, University of Oxford, UK
   index: 4
date: 14 July 2023
bibliography: paper.bib

---

# Summary

Fluid-like convection of Earth’s rocky mantle drives processes such as plate tectonics that shape the surface and explains the evolution of our planet on the longest time scales. Because of this, computer simulations of mantle convection have become important for our understanding of the Earth and large scale simulation codes have been created by the community [@Tackley@2008; @Kronbichler2012; @Davies2011; @Moresi2014; @Zhong2000]. One example is TERRA [@baumgardner1985; @bunge1997], a large parallel program using the finite element method to simulate convection.  TERRA is written in Fortran and runs on supercomputers, producing gigabytes of files for each timestep. Handling these outputs is non-trivial because the output reflects the structured grid of finite element mesh and the parallel decomposition used to execute the code. Furthermore, existing closed-source tools in Fortran and other compiled languages present a significant barrier to further development of model analysis tools. Here we describe terratools, a Python package designed to enable reproducible and repeatable post-processing analysis of the outputs of TERRA simulations. Documentation is available via a dedicated website (https://terratools.readthedocs.io/).

# Statement of need

TERRA is a widely-used and powerful simulation package which underlies a large amount of research into the Earth's mantle [@Ghelichkhan2021; @panton2023; @taiwo2023], but before now there has not been any open-source software which can be shared amongst different scientific groups to aid in analysing the results of TERRA simulations.  This has slowed the development of new analyses and enforced duplicated effort.  TERRA also uses a particular meshing of the sphere, and taking advantage of this for efficient computation is not straightforward.  In addition, there are a number of choices in how some analyses are done, for example in how physical model parameters such as temperature and density are translated into geophysical observables such as seismic wave velocity, and it is not always transparent which choices have been made in any particular study.

terratools addresses this by providing a high-level abstraction over the details of a TERRA model and encapsulating these in a class, `TerraModel`, which permits non-specialist programmers with Python experience to examine mantle convection simulations in new and existing ways.

# Current functionality

As well as a high-level abstraction over simulation timesteps, a number of analytical workflows come with terratools and are constantly being added to.  For instance, radial average profiles, local one-dimensional profiles, arbitrary point extraction and spherical harmonic analysis are available already.  Conversion from model parameters (temperature and composition) to seismic parameters (P- and S-wave velocities, attenuation) is supported using pre-computed conversion lookup tables, but tools to create these are also provided. We also provide tools for identifying upwelling features (mantle plumes) in simulations.

terratools defines a versioned and open-source file format based upon NetCDF [@netcdf] for TERRA models, making the exchange of simulation snapshots simpler and removing the need for different groups to rewrite file readers.  As files may be many gigabytes in size, this also enables more efficient file reading and writing, saving time.

# Acknowledgements

We acknowledge funding from NERC Large Grant 'Mantle Circulation Constrained (MC²)' (NE/T012595/1, NE/T012633/1, NE/T012684/1).

# References

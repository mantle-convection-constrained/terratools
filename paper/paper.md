---
title: 'TerraTools: A Python package for access to Mantle Convection Simulations'
tags:
  - Python
  - earth sciences
  - mantle convection
authors:
  - name: Andy Nowacki
    orcid: 0000-0001-7669-7383
#    equal-contrib: true
    affiliation: 1
  - name: James Panton
    orcid: 0000-0002-1345-9406
    affiliation: 2
  - name: Jamie Ward
    orcid: 0000-0001-5697-4590
    affiliation: 1
  - name: Bob Myhill
    orcid: 0000-0000-0000-0000
    affiliation: 3
  - name: Andrew Walker
    orcid: 00000-0003-3121-3255
    affiliation: 4
  - name: James Wookey
    orcid: 0000-0002-7403-4380
    affiliation: 3
  - name: J. Huw Davies
    orcid: 0000-0003-2656-0260
    affiliation: 2

affiliations:
 - name: School of Earth and Environment, University of Leeds , UK
   index: 1
 - name: School of Earth and Environmental Sciences, Cardiff University, UK
   index: 2
 - name: School of Earth Sciences, University of Bristol, UK
   index: 3
 - name: Department of Earth Sciences, University of Oxford, UK
   index: 3
date: 7 September 2022
bibliography: paper.bib

---

# Summary

Fluid-like convection of Earth’s rocky mantle drives processes such as plate tectonics that shape the surface and explains the evolution of our planet on the longest time scales. Because of this computer simulations of mantle convection have become important for our understanding of the Earth and large scale simulation codes have been created by the community [refs?]. One example is TERRA [@baumgardner1985; @bunge1997], a large parallel programme using the the finite element method to simulate convection including [key features]. While TERRA is written in Fortran and runs on supercomputers, handling its output is non- trivial because the output reflects the structured grid of finite element mesh and the parallel decomposition used to execute the code and processing the data often needs integration with other tools. Here we describe TerraTools, a Python package designed to [key things]. The package has been developed to aid in research associated with a large collaborative grant [issues about this - usability and stuff].

# Statement of need

`TerraTools` statement of need goes here.

# Acknowledgements

We acknowledge funding from NERC Large Grant `Mantle Circulation Constrained (MC$^2)` (NE/T012595/1).

# References
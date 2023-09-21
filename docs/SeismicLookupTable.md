TerraTools allows for interrogation of seismic lookup tables by interpolation and plotting. Futhermore, TerraTools can convert terra output in temperature and composition to a seismic model for given look up tables for each composition present. 

## Seismic lookup tables and how to read them in

A seismic lookup table describes how seismic properties such as Vp (p-wave velocity) or density vary with pressure and temperature for a given composition. In TerraTools, you can make a SeismicLookupTable by either reading in a file from disk or passing arrays to the SeismicLookupTable class. 

When reading in a lookup table file via `[SeismicLookupTable(table_path='path_to_file')][terratools.lookup_tables.SeismicLookupTable]`, the file needs to be in the format shown below. Note that the temperatures increase first and then the pressures. In other words, the values at all temperatures are given for each pressure point first, then all values for the temperatures at the second pressure point and so on. The table below gives an example of this. 

| Pressure | Temperature | Vp | Vs | Vp_an | Vs_an | Vphi | Density | Qs | T_solidus |
| -------- | ----------- | -- | -- | ----- | ----- | ---- | ------- | -- | --------- |
| 0        | 500         |    |    |       |       |      |         |    |           |
| 0        | 1000        |    |    |       |       |      |         |    |           |
| 0        | 1500        |    |    |       |       |      |         |    |           |
| 1e8      | 500         |    |    |       |       |      |         |    |           |
| 1e8      | 1000        |    |    |       |       |      |         |    |           |
| 1e8      | 1500        |    |    |       |       |      |         |    |           |


If constructing from data in memory, 2D arrays of ``vp``, ``vs``, ``vp_an``, ``vs_an``, ``vphi``, ``density``, ``qs`` and ``t_sol`` of dimensions [n_temps, n_pressure] and the corresponding temperatures (``temperature``) and pressures (``pressure``) need to be passed to SeismicLookupTable. 


## Functions with seismic lookup table

Once read in, terratools offers several methods to interpolate and plot the tables. `[SeismicLookupTable.interp_grid()][terratools.lookup_tables.SeismicLookupTable.interp_grid]` will regrid the whole lookup table for a given field to new pressures and temperatures. `[SeismicLookupTable.interp_points()][terratools.lookup_tables.SeismicLookupTable.interp_points]` will interpolate the field to specific pressure and temperature pairs. 

To visualise the tables, one can use either `[SeismicLookupTable.plot_table()][terratools.lookup_tables.SeismicLookupTable.plot_table]` giving a field and matplotlib axis to plot the values in each cell, or one can use `[SeismicLookupTable.plot_table_contour()][terratools.lookup_tables.SeismicLookupTable.plot_table_contour]` giving the same arguments but plotting the table with contours of the parameter in question. 

## MultiTables 

If more than one composition is present, terratools can evaluate the seismic properties using the MultiTables class. To initiate the class, a dictionary needs to be provided as ``lookuptables`` where the keys are the composition names (e.g. 'basalt') and the values are the lookup table objects for that composition as described above. 

Once the class has been initialisd, one can evaluate the seismic property of interest with the `[MultiTables.evaluate()][terratools.lookup_tables.MultiTables.evaluate()]` function where the user needs to give the pressure, temperature and field of interest as well as the relative fractions of the compositions. The property is found by interpolating each of the lookuptables to the pressure and temperature of interest then performing an harmonic mean using the fractions as weights. 


## Converting to a seismic velocity model

First, if no lookup tables are part of the TerraModel object, the look up tables need to be added by using `[TerraModel.add_lookup_table()][terratools.terra_model.TerraModel.add_lookup_table]`. Note when adding the tables, they keys must have the same name as the compositions in TerraModel. Once TerraModel has the lookup tables, the seismic properties at a location can be evaluated with `[TerraModel.evaluate_from_lookup_tables()][terratools.terra_model.TerraModel.evaluate_from_lookup_tables]` passing the latitude, longitude and radius.
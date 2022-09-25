
Changelog
=========
1.1.0 (2022-09-25)
------------------
New features

* Added support for Python 3.9, 3.10.
* Silixa's xml version 8 is now supported

Bug fixes

* Loading in untested sensornet files will not give a UnboundLocalError error anymore
* Sensornet .ddf file version check is now more robust (commas are replaced to periods)
* Changed matplotlib's deprecated DivergingNorm to TwoSlopeNorm
* Updated the stokes_variance_linear docstring to remove incorrect and duplicate information
* Adjusted resample_datastore to avoid using deprecated 'base' kwarg, instead using the new arguments 'origin' and 'offset'. See http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html

Others

* Notebook 04 on Stokes variance has been updated to explain the different estimation methods for the variance, and their importance.
* Notebook 07 and 08 have been updated to take into account the changes in ds.stokes_variance.
* Silenced xarray's __slots__ warning
* Deprecated Python 3.6
* CI has been moved to GitHub Actions. Travis-CI and AppVeyor have been removed.

1.0.2 (2020-05-04)
------------------
* Same as v1.0.1

1.0.1 (2020-05-04)
------------------
New features

* st_var can now be array-like. The array-like can vary in x or time, or both.
* When converting from a xr.Dataset to a DataStore object, the attributes arenow transfered over
* Added 'verbose' kwarg to shift_double_ended utility function to silence theconsole output

Bug fixes

* If the '_sections' attribute is missing from a DataStore object it is automagically re-added.
* Assert that stokes variances are passed when running a double-ended WLS calibration
* Added check for NaN/inf values in wls_sparse solver to avoid unclear error messages
* Before calibration, the values of the used reference temperatures are checked if they are valid (float dtype, no NaN/inf values, correct time dimension)

Others

* European Geosciences Union conference 2020 presentation added
* Updated documentation with content article
* Use YAPF formatting of the Python files
* Travis-CI.org integration in GitHub restored.

1.0.0 (2020-03-30)
------------------
* First major release
* Reduced flexibility in defining names under which the Stokes are stored
* 4 Averaging functions implemented, with uncertainty estimation. See notebook 16 for the various options
* Notebook about transient attenuation caused by for example connectors
* Bug in singel ended transient attenuation
* Check for consistent number of measurement locations across read files

0.9.2 (2020-03-17)
------------------
* Reduced flexibility in defining Stokes labels

0.9.1 (2020-03-17)
------------------
* Same as 0.9.2

0.9.0 (2020-03-16)
------------------
* Increased precision of least squares solver, as this was limiting theprecision of the parameter estimation
* The variance of the noise in the Stokes can change linear with the intensity
* Improved residuals plot function
* Reduced the number of equations for double ended calibration
* Support for transient attenuation due to connectors along fibers
* Matching sections

0.8.0 (2020-02-14)
------------------
* Valentine edition
* Added example for fixing parameters
* Bug fixed in routine for reading Sensornet files (Bor van der Scheer)
* Official support for Python 3.8
* When the datastore is printed to the screen, the mean and std of thereference temperature is plotted
* Integrated differential attenuation is reformulated. Starts integrating atthe first reference section location.
* Estimation of the variance of the Stokes that is linear dependent on theintensity (Poisson)
* Removed `__slots__` attribute

0.7.4 (2020-01-26)
------------------
* Update automated zenodo reference requires to draft a new release

0.7.3 (2020-01-24)
------------------
* Solved xlim in subplots of plot_residuals_reference_sections funciton
* Solved YAML deprecation related problems
* Introduced new approach for double ended calibration, with a different Cfor the forward and backward channel
* First code added for time variant asymmetric attenuation, such as connectors.
* First code added for matching sections

0.7.2 (2019-11-22)
------------------
* Set alpha and or gamma to known value, with corresponding variance.
* Bug in computation of the weights for single and double-ended calibration
* Added notebook about merging two single ended measurements
* Added example notebook on how to create a custom datastore
* Added notebook examples for loading in data from the different manufa..
* Loading AP Sensing files and tests
* Loading Sensortran files

0.7.0 (2019-11-07)
------------------
* Ensure order of dimension upon initialization of DataStore. Resamplingwould lead to issues
* Bug in section definition (reported by Robert Law)
* Rewritten calibration solvers to align with article of this package
* Removed old calibration solvers
* New possibilities of saving and loading large DataStores saved to multiplenetCDF files

0.6.7 (2019-11-01)
------------------
* Use twine to check if the compiled package meets all the requirements of Pypi

0.6.6 (2019-11-01)
------------------
* Use twine to check if the compiled package meets all the requirements of Pypi

0.6.5 (2019-11-01)
------------------
* Major bug fix version.
* More flexibility in defining the time and space dimensions
* Fixed unsave yaml loading
* Added support for Silixa 7 files
* Start using `__slots__` as it is something new
* xarray doesn't have the attribute `._initialized` anymore. Rewritten teststo make more sense by checking the sum of the Stokes instead.
* Support for double ended Sensornet files and tests
* Bug fixing

0.6.4 (2019-04-09)
------------------
* More flexibility in defining the time dimension
* Cleanup of some plotting functions

0.6.3 (2019-04-03)
------------------
* Added reading support for zipped silixa files. Still rarely fails due to upstream bug.
* pretty __repr__
* Reworked double ended calibration procedure. Integrated differential attenuation outside of reference sections is now calculated seperately.
* New approach for estimation of Stokes variance. Not restricted to a decaying exponential
* Bug in averaging TMPF and TMPB to TMPW
* Modified residuals plot, especially useful for long fibers (Great work Bart!)
* Example notebooks updatred accordingly
* Bug in `to_netcdf` when passing encodings
* Better support for sections that are not related to a timeseries.

0.6.2 (2019-02-26)
------------------
* Double-ended weighted calibration procedure is rewritten so that the integrated differential attenuation outside of the reference sections is calculated seperately. Better memory usage and faster
* Other calibration routines cleaned up
* Official support for Python 3.7
* Coverage figures are now trustworthy
* String representation improved
* Include test for aligning double ended measurements
* Example for aligning double ended measurements

0.6.1 (2019-01-04)
------------------
* Many examples were shown in the documentation
* Fixed verbose settings of solvers
* Revised example notebooks
* Moved to 80 characters per line (PEP)
* More Python formatting using YAPF
* Use example of `plot_residuals_reference_sections` function in Stokes variance example notebook
* Support Python 3.7

0.6.0 (2018-12-08)
------------------
* Reworked the double-ended calibration routine and the routine for confidence intervals. The integrated differential attenuation is not zero at x=0 anymore.
* Verbose commands carpentry
* Bug fixed that would make the read_silixa routine crash if there are copies of the same file in the same folder
* Routine to read sensornet files. Only single-ended configurations supported for now. Anyone has double-ended measurements?
* Lazy calculation of the confidence intervals
* Bug solved. The x-coordinates where not calculated correctly. The bug only appeared for measurements along long cables.
* Example notebook of importing a timeseries. For example, importing measurments from an external temperature sensor for calibration.
* Updated documentation


0.5.3 (2018-10-26)
------------------
* No changes

0.5.2 (2018-10-26)
------------------
* New resample_datastore method (see basic usage notebook)
* New notebook on basic usage of DataStore
* Support for Silixa v4 (Windows xp based system) and Silixa v6 (Windows 7) measurement files
* The representation string now includes the sections
* Reorganized the IO related files
* CI: Add appveyor to continuesly test on Windows platform
* Auto load Silixa files to memory option, if size is small

0.5.1 (2018-10-19)
------------------
* Rewritten the routine that reads Silixa measurement files
* dts-calibration is now citable
* Refractored the MC confidence interval routine
* MC confidence interval routine speed up, with full dask support
* Link to mybinder.org to try the example notebooks online
* Added a few missing dependencies
* The routine to read the Silixa files is completely refractored. Faster, smarter. Supports both the path to a directory and a list of file paths.
* Changed imports from dtscalibration to be relative

0.4.0 (2018-09-06)
------------------
* Single ended calibration
* Confidence intervals for single ended calibration
* Example notebooks have figures embedded
* Several bugs squashed
* Reorganized DataStore functions


0.2.0 (2018-08-16)
------------------
* Double ended calibration
* Confidence intervals for double ended calibration


0.1.0 (2018-08-01)
------------------
* First release on PyPI.


Changelog
=========

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

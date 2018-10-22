
Changelog
=========

Head
----

* CI: Add appveyor to continuesly test on Windows platform
* Auto load Silixa files to memory option, if size is small

0.5.1 (2018-10-19)
------------------

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

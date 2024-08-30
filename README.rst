========
Overview
========

.. start-badges

.. list-table::

    * - Docs
      - |docs|
    * - Tests
      - |tests|
    * - Package
      - | |version| |supported-versions| |commits-since|
    * - Citable
      - |zenodo|
    * - Example notebooks
      - |example-notebooks|

.. |docs| image:: https://readthedocs.org/projects/python-dts-calibration/badge/?style=flat
    :target: https://python-dts-calibration.readthedocs.io/en/latest/
    :alt: Documentation Status

.. |tests| image:: https://github.com/dtscalibration/python-dts-calibration/actions/workflows/build.yml/badge.svg
    :target: https://github.com/dtscalibration/python-dts-calibration/actions/workflows/build.yml
    :alt: Test Status

.. |version| image:: https://img.shields.io/pypi/v/dtscalibration.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/dtscalibration

.. |commits-since| image:: https://img.shields.io/github/commits-since/dtscalibration/python-dts-calibration/v3.0.3.svg
    :alt: Commits since latest release
    :target: https://github.com/dtscalibration/python-dts-calibration/compare/v1.1.1...main

.. |wheel| image:: https://img.shields.io/pypi/wheel/dtscalibration.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/dtscalibration

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/dtscalibration.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/dtscalibration

.. |zenodo| image:: https://zenodo.org/badge/143077491.svg
   :alt: It would be greatly appreciated if you could cite this package in eg articles presentations
   :target: https://zenodo.org/badge/latestdoi/143077491

.. |example-notebooks| image:: https://mybinder.org/badge.svg
   :alt: Interactively run the example notebooks online
   :target: https://mybinder.org/v2/gh/dtscalibration/python-dts-calibration/main?labpath=docs%2Fnotebooks

.. end-badges

A Python package to load Distributed Temperature Sensing files, perform a calibration, and plot the result. A detailed description of the calibration procedure can be found at https://doi.org/10.3390/s20082235 .

Do you have questions, ideas or just want to say hi? Please leave a message on the ` discussions page <https://github.com/dtscalibration/python-dts-calibration/discussions>`_!

Installation
============

.. code-block:: zsh

    pip install dtscalibration

Or the development version directly from GitHub

.. code-block:: zsh

    pip install https://github.com/dtscalibration/python-dts-calibration/zipball/main --upgrade

Package features
================
DTS measures temperature by calibrating backscatter measurements to sections with a known temperature. DTS devices provide a simple interface to perform a limited calibration. Re-calibrating your measurements with this Python package gives you better temperature estimates and additional options.

* Advanced calibration routine
   * Supports `single <https://github.com/dtscalibration/python-dts-calibration/blob/main/docs/notebooks/07Calibrate_single_ended.ipynb>`_- and `double-ended <https://github.com/dtscalibration/python-dts-calibration/blob/main/docs/notebooks/08Calibrate_double_ended.ipynb>`_ setups
   * Compute uncertainty of the calibrated temperature
   * All measurements are used to estimate parameter values that are constant over time.
   * Weighted least-squares calibration
   * `Fixing parameters to a previously determined value <https://github.com/dtscalibration/python-dts-calibration/blob/main/docs/notebooks/13Fixed_parameter_calibration.ipynb>`_
   * `(Asymmetric) step loss correction <https://github.com/dtscalibration/python-dts-calibration/blob/main/docs/notebooks/14Lossy_splices.ipynb>`_ so that fiber connectors can be used instead of welds/splices.
   * `Matching temperature sections <https://github.com/dtscalibration/python-dts-calibration/blob/main/docs/notebooks/15Matching_sections.ipynb>`_ to support J-configurations
* Dynamic reference section definition
* Tools for merging and aligning double-ended setups
* Data formats of most manufacturers are supported

Devices currently supported
===========================
* Silixa Ltd.: **Ultima** & **XT-DTS** .xml files *(up to version 8.1)*
* Sensornet Ltd.: **Oryx**, **Halo** & **Sentinel** .ddf files
* AP Sensing: **N4386B** .xml files *(single ended only)*
* SensorTran: **SensorTran 5100** .dat binary files *(single ended only)*

Documentation
=============

* A full calibration procedure for single-ended setups is presented in notebook `07Calibrate_single_ended.ipynb <https://github.com/dtscalibration/python-dts-calibration/blob/main/docs/notebooks/07Calibrate_single_ended.ipynb>`_ and for double-ended setups in `08Calibrate_double_ended.ipynb <https://github.com/dtscalibration/python-dts-calibration/blob/main/docs/notebooks/08Calibrate_double_ended.ipynb>`_.
* Documentation at `readthedocs <https://python-dts-calibration.readthedocs.io/en/latest/>`_.
* Example notebooks (`./docs/notebooks`) that work within the browser can be viewed `here <https://mybinder.org/v2/gh/dtscalibration/python-dts-calibration/main?labpath=docs%2Fnotebooks>`_.

How to cite
===========
The following article explains and discusses the calibration procedure:

    des Tombe, B., Schilperoort, B., & Bakker, M. (2020). Estimation of Temperature and Associated Uncertainty from Fiber-Optic Raman-Spectrum Distributed Temperature Sensing. Sensors, 20(8), 2235. https://doi.org/10.3390/s20082235

Cite the specific implementation / repository via Zenodo:

1. Check the version of `dtscalibration` that is used in your Python console with:

    >>> # The following line introduces the .dts accessor for xarray datasets
    >>> import dtscalibration  # noqa: E401
    >>> dtscalibration.__version__
    '3.0.1'
2. Go to `Zenodo <https://zenodo.org/search?q=conceptrecid:%221410097%22&sort=-version&all_versions=True>`_ and follow the link to the version of interest.
3. The citation is found on the bottom right of the page.

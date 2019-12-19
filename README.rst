========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - Docs
      - |docs|
    * - Tests
      - | |travis|
        | |appveyor|
        | |codecov|
    * - Package
      - | |version|
        | |wheel|
        | |supported-versions|
        | |supported-implementations|
        | |commits-since|
    * - Citable
      - |zenodo|
    * - Example notebooks
      - |example-notebooks|

.. |docs| image:: https://readthedocs.org/projects/python-dts-calibration/badge/?style=flat
    :target: https://python-dts-calibration.readthedocs.io/en/latest/
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/dtscalibration/python-dts-calibration.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/dtscalibration/python-dts-calibration

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/we2caropyby30nd1?svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/bdestombe/python-dts-calibration

.. |codecov| image:: https://codecov.io/github/dtscalibration/python-dts-calibration/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/dtscalibration/python-dts-calibration

.. |version| image:: https://img.shields.io/pypi/v/dtscalibration.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/dtscalibration

.. |commits-since| image:: https://img.shields.io/github/commits-since/dtscalibration/python-dts-calibration/v0.7.2.svg
    :alt: Commits since latest release
    :target: https://github.com/dtscalibration/python-dts-calibration/compare/v0.7.2...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/dtscalibration.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/dtscalibration

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/dtscalibration.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/dtscalibration

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/dtscalibration.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/dtscalibration

.. |zenodo| image:: https://zenodo.org/badge/143077491.svg
   :alt: It would be greatly appreciated if you could cite this package in eg articles presentations
   :target: https://zenodo.org/badge/latestdoi/143077491

.. |example-notebooks| image:: https://mybinder.org/badge.svg
   :alt: Interactively run the example notebooks online
   :target: https://mybinder.org/v2/gh/dtscalibration/python-dts-calibration/master?filepath=examples%2Fnotebooks

.. end-badges

A Python package to load raw DTS files, perform a calibration, and plot the result

* Free software: BSD 3-Clause License


Installation
============

::

    pip install dtscalibration
    
Or the development version directly from GitHub

::

    pip install https://github.com/dtscalibration/python-dts-calibration/zipball/master --upgrade

Package features
================
* Advanced calibration routine
   * Both single- and double-ended setups
   * Confidence intervals of calibrated temperature
   * Time integration of calibration parameters
   * Fixing parameters to a previously determined value
   * Weighted least-squares calibration
* Dynamic reference section definition
* Tools for merging and aligning double-ended setups
* Data formats of most manufacturers are supported

Devices currently supported
===========================
* Silixa Ltd.: **Ultima** & **XT-DTS** .xml files *(up to version 7.0)*
* Sensornet Ltd.: **Oryx** & **Halo** .ddf files
* AP Sensing: **CP320** .xml files *(single ended only)*
* SensorTran: **SensorTran 5100** .dat binary files *(single ended only)*

Learn by examples
=================
Interactively run the example notebooks online by clicking the example-notebooks button in the beginning of this README.

Documentation
=============

https://python-dts-calibration.readthedocs.io/


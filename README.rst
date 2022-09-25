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

.. |commits-since| image:: https://img.shields.io/github/commits-since/dtscalibration/python-dts-calibration/v1.1.1.svg
    :alt: Commits since latest release
    :target: https://github.com/dtscalibration/python-dts-calibration/compare/v1.1.1...master

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
   :target: https://mybinder.org/v2/gh/dtscalibration/python-dts-calibration/master?filepath=examples%2Fnotebooks

.. end-badges

A Python package to load Distributed Temperature Sensing files, perform a calibration, and plot the result. A detailed description of the calibration procedure can be found at https://doi.org/10.3390/s20082235 .

* Free software: BSD 3-Clause License


Installation
============

.. code-block:: zsh

    pip install dtscalibration

Or the development version directly from GitHub

.. code-block:: zsh

    pip install https://github.com/dtscalibration/python-dts-calibration/zipball/master --upgrade

Package features
================
* Advanced calibration routine
   * Both single- and double-ended setups
   * Confidence intervals of calibrated temperature
   * Time integration of calibration parameters
   * Weighted least-squares calibration
   * `Fixing parameters to a previously determined value <..//master/examples/notebooks/13Fixed_parameter_calibration.ipynb>`_
   * `(Asymmetric) step loss correction <../master/examples/notebooks/14Lossy_splices.ipynb>`_
   * `Matching temperature sections <../master/examples/notebooks/15Matching_sections.ipynb>`_
* Dynamic reference section definition
* Tools for merging and aligning double-ended setups
* Data formats of most manufacturers are supported

Devices currently supported
===========================
* Silixa Ltd.: **Ultima** & **XT-DTS** .xml files *(up to version 8.1)*
* Sensornet Ltd.: **Oryx** & **Halo** .ddf files
* AP Sensing: **CP320** .xml files *(single ended only)*
* SensorTran: **SensorTran 5100** .dat binary files *(single ended only)*

Learn by examples
=================
Interactively run the example notebooks online by clicking `here <https://mybinder.org/v2/gh/dtscalibration/python-dts-calibration/master?filepath=examples%2Fnotebooks>`_.

Documentation
=============

https://python-dts-calibration.readthedocs.io/

How to cite
===========
The following article explains and discusses the calibration procedure:

    des Tombe, B., Schilperoort, B., & Bakker, M. (2020). Estimation of Temperature and Associated Uncertainty from Fiber-Optic Raman-Spectrum Distributed Temperature Sensing. Sensors, 20(8), 2235. https://doi.org/10.3390/s20082235

Cite the specific implementation / repository via Zenodo:

1. Check the version of `dtscalibration` that is used in your Python console with:

    >>> import dtscalibration
    >>> dtscalibration.__version__
    '1.0.0'
2. Go to `Zenodo <https://zenodo.org/search?q=conceptrecid:%221410097%22&sort=-version&all_versions=True>`_ and follow the link to the version of interest.
3. The citation is found on the bottom right of the page.

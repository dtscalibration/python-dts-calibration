========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
    * - citable
      - |zenodo|

.. |docs| image:: https://readthedocs.org/projects/python-dts-calibration/badge/?style=flat
    :target: https://readthedocs.org/projects/python-dts-calibration
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/bdestombe/python-dts-calibration.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/bdestombe/python-dts-calibration

.. |codecov| image:: https://codecov.io/github/bdestombe/python-dts-calibration/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/bdestombe/python-dts-calibration

.. |version| image:: https://img.shields.io/pypi/v/dtscalibration.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/dtscalibration

.. |commits-since| image:: https://img.shields.io/github/commits-since/bdestombe/python-dts-calibration/v0.4.0.svg
    :alt: Commits since latest release
    :target: https://github.com/bdestombe/python-dts-calibration/compare/v0.4.0...master

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

.. end-badges

A Python package to load raw DTS files, perform a calibration, and plot the result

* Free software: BSD 3-Clause License

Installation
============

::

    pip install dtscalibration

Documentation
=============

https://python-dts-calibration.readthedocs.io/

Development
===========

To run the all tests run::

    tox


To bump version and docs::

    tox
    git status  # to make sure no unversioned modifications are in the repository
    git add --all
    bumpversion patch
    rm -rf build
    rm -rf src/*.egg-info
    rm -rf dist/*
    tox
    python setup.py clean --all sdist bdist_wheel
    twine upload --repository-url https://upload.pypi.org/legacy/ dist/dtscalibration*

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
        | |appveyor|
        | |codecov|
    * - package
      - | |version|
        | |wheel|
        | |supported-versions|
        | |supported-implementations|
        | |commits-since|
    * - citable
      - |zenodo|
    * - Example notebooks
      - |example-notebooks|

.. |docs| image:: https://readthedocs.org/projects/python-dts-calibration/badge/?style=flat
    :target: https://readthedocs.org/projects/python-dts-calibration
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/bdestombe/python-dts-calibration.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/bdestombe/python-dts-calibration

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/bdestombe/python-dts-calibration?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/bdestombe/python-dts-calibration

.. |codecov| image:: https://codecov.io/github/bdestombe/python-dts-calibration/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/bdestombe/python-dts-calibration

.. |version| image:: https://img.shields.io/pypi/v/dtscalibration.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/dtscalibration

.. |commits-since| image:: https://img.shields.io/github/commits-since/bdestombe/python-dts-calibration/v0.5.1.svg
    :alt: Commits since latest release
    :target: https://github.com/bdestombe/python-dts-calibration/compare/v0.5.1...master

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
   :target: https://mybinder.org/v2/gh/bdestombe/python-dts-calibration/master?filepath=examples%2Fnotebooks

.. end-badges

A Python package to load raw DTS files, perform a calibration, and plot the result

* Free software: BSD 3-Clause License

Installation
============

::

    pip install dtscalibration

Learn by examples
=================
Interactively run the example notebooks online by clicking the launch-binder button.

Documentation
=============

https://python-dts-calibration.readthedocs.io/

Development
===========

To run the all tests run:

.. code-block:: zsh

    tox


To bump version and docs:

.. code-block:: zsh

    git status          # to make sure no unversioned modifications are in the repository
    tox                 # Performes tests and creates documentation and runs notebooks
    git status          # Only notebook related files should be shown
    git add --all       # Add all notebook related files to local version
    git commit -m "Updated notebook examples to reflect recent changes"
    
.. code-block:: zsh

    # update CHANGELOG.rst with the recent commits
    # update AUTHORS.rst
    
.. code-block:: zsh

    bumpversion patch   # (major, minor, patch)
    git push
    rm -rf build        # Clean local folders (not synced) used for pip wheel
    rm -rf src/*.egg-info
    rm -rf dist/*
    python setup.py clean --all sdist bdist_wheel
    twine upload --repository-url https://upload.pypi.org/legacy/ dist/dtscalibration*
    
On GitHub draft a new release

.. code-block:: zsh

    # GitHub > Code > Releases > Draft a new release
    # Tag: v1.2.3
    # Title: v1.2.3
    # Describtion: Copy-paste the new part of CHANGELOG.rst


3. Define calibration sections
==============================

The goal of this notebook is to show how you can define calibration
sections. That means that we define certain parts of the fiber to a
timeseries of temperature measurements. Here, we assume the temperature
timeseries is already part of the ``DataStore`` object.

.. code:: ipython3

    import os
    
    from dtscalibration import read_silixa_files

.. code:: ipython3

    filepath = os.path.join('..', '..', 'tests', 'data', 'double_ended2')
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf='UTC',
        file_ext='*.xml')


.. parsed-literal::

    6 files were found, each representing a single timestep
    6 recorded vars were found: LAF, ST, AST, REV-ST, REV-AST, TMP
    Recorded at 1693 points along the cable
    The measurement is double ended
    Reading the data from disk


First we have a look at which temperature timeseries are available for
calibration. Therefore we access ``ds.data_vars`` and we find
``probe1Temperature`` and ``probe2Temperature`` that refer to the
temperature measurement timeseries of the two probes attached to the
Ultima.

Alternatively, we can access the ``ds.timeseries_keys`` property to list
all timeseries that can be used for calibration.

.. code:: ipython3

    print(ds.timeseries_keys)    # list the available timeseeries
    ds.probe1Temperature.plot(figsize=(12, 8));  # plot one of the timeseries


.. parsed-literal::

    ['acquisitionTime', 'referenceTemperature', 'probe1Temperature', 'probe2Temperature', 'referenceProbeVoltage', 'probe1Voltage', 'probe2Voltage', 'userAcquisitionTimeFW', 'userAcquisitionTimeBW']


.. parsed-literal::

    /Users/bfdestombe/Projects/dts-calibration/python-dts-calibration-dev/.tox/docs/lib/python3.7/site-packages/pandas/plotting/_matplotlib/converter.py:103: FutureWarning: Using an implicitly registered datetime converter for a matplotlib plotting method. The converter was registered by pandas on import. Future versions of pandas will require you to explicitly register matplotlib converters.
    
    To register the converters:
    	>>> from pandas.plotting import register_matplotlib_converters
    	>>> register_matplotlib_converters()
      warnings.warn(msg, FutureWarning)


A calibration is needed to estimate temperature from Stokes and
anti-Stokes measurements. There are three unknowns for a single ended
calibration procedure :math:`\gamma`, :math:`C`, and :math:`\alpha`. The
parameters :math:`\gamma` and :math:`\alpha` remain constant over time,
while :math:`C` may vary.

At least two calibration sections of different temperatures are needed
to perform a decent calibration procedure.

This setup has two baths, named ‘cold’ and ‘warm’. Each bath has 2
sections. ``probe1Temperature`` is the temperature timeseries of the
cold bath and ``probe2Temperature`` is the temperature timeseries of the
warm bath.

+---------+---------------------------+-------------+-----------------+
| Name    | Name reference            | Number of   | Location of     |
| section | temperature time series   | sections    | sections (m)    |
+=========+===========================+=============+=================+
| Cold    | probe1Temperature         | 2           | 7.5-17.0;       |
| bath    |                           |             | 70.0-80.0       |
+---------+---------------------------+-------------+-----------------+
| Warm    | probe2Temperature         | 2           | 24.0-34.0;      |
| bath    |                           |             | 85.0-95.0       |
+---------+---------------------------+-------------+-----------------+

Sections are defined in a dictionary with its keywords of the names of
the reference temperature time series. Its values are lists of slice
objects, where each slice object is a section.

Note that slice is part of the standard Python library and no import is
required.

.. code:: ipython3

    sections = {
        'probe1Temperature': [slice(7.5, 17.), slice(70., 80.)],  # cold bath
        'probe2Temperature': [slice(24., 34.), slice(85., 95.)],  # warm bath
        }
    ds.sections = sections

.. code:: ipython3

    ds.sections




.. parsed-literal::

    {'probe1Temperature': [slice(7.5, 17.0, None), slice(70.0, 80.0, None)],
     'probe2Temperature': [slice(24.0, 34.0, None), slice(85.0, 95.0, None)]}



NetCDF files do not support reading/writing python dictionaries.
Internally the sections dictionary is stored in ``ds._sections`` as a
string encoded with yaml, which can be saved to a netCDF file. Each time
the sections dictionary is requested, yaml decodes the string and
evaluates it to the Python dictionary.


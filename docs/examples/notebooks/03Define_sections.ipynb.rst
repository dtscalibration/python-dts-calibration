
3. Define calibration sections
==============================

.. code:: ipython3

    import os
    import glob
    
    from dtscalibration import read_silixa_files

.. code:: ipython3

    try:
        wd = os.path.dirname(os.path.realpath(__file__))
    except:
        wd = os.getcwd()
    
    filepath = os.path.join(wd, '..', '..', 'tests', 'data', 'double_ended2')
    timezone_netcdf = 'UTC'
    timezone_input_files = 'Europe/Amsterdam'
    file_ext = '*.xml'
    
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf=timezone_netcdf,
        timezone_input_files=timezone_input_files,
        file_ext=file_ext)


.. parsed-literal::

    6 files were found, each representing a single timestep
    6 recorded vars were found: LAF, ST, AST, REV-ST, REV-AST, TMP
    Recorded at 1693 points along the cable
    The measurement is double ended


.. code:: ipython3

    ds.probe1Temperature.plot()




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x11c54c7b8>]



A calibration is needed to estimate temperature from Stokes and
anti-Stokes measurements. There are three unknowns for a single ended
calibration procedure :math:`\gamma`, :math:`C`, and :math:`\alpha`. The
parameters :math:`\gamma` and :math:`\alpha` remain constant over time,
while :math:`C` may vary.

At least two calibration sections of different temperatures are needed
to perform a decent calibration procedure.

+---------+---------------------------+-------------+-----------------+
| Name    | Name reference            | Number of   | Location of     |
| section | temperature time series   | stretches   | sections (m)    |
+=========+===========================+=============+=================+
| Cold    | probe1Temperature         | 2           | 7.5-17.0;       |
| bath    |                           |             | 70.0-80.0       |
+---------+---------------------------+-------------+-----------------+
| Warm    | probe2Temperature         | 2           | 24.0-34.0;      |
| bath    |                           |             | 85.0-95.0       |
+---------+---------------------------+-------------+-----------------+

Each section requires a reference temperature time series, such as the
temperature measured by an external temperature sensor. They should
already be part of the DataStore object.

Sections are defined in a dictionary with its keywords of the names of
the reference temperature time series. Its values are lists of slice
objects, where each slice object is a stretch.

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

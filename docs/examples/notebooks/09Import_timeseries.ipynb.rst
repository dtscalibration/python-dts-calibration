9. Import a time series
=======================

In this tutorial we are adding a timeseries to the DataStore object.
This might be useful if the temperature in one of the calibration baths
was measured with an external device. It requires three steps to add the
measurement files to the DataStore object: 1. Load the measurement files
(e.g., csv, txt) with pandas into a pandas.Series object 2. Add the
pandas.Series object to the DataStore 3. Align the time to that of the
DTS measurement (required for calibration)

.. code:: ipython3

    import pandas as pd
    import os
    
    from dtscalibration import read_silixa_files

Step 1: load the measurement files
----------------------------------

.. code:: ipython3

    filepath = os.path.join('..', '..', 'tests', 'data', 
                            'external_temperature_timeseries', 
                            'Loodswaternet2018-03-28 02h.csv')
    
    # Bonus:
    print(filepath, '\n')
    with open(filepath, 'r') as f:
        head = [next(f) for _ in range(5)]
    print(' '.join(head))


.. parsed-literal::

    ../../tests/data/external_temperature_timeseries/Loodswaternet2018-03-28 02h.csv 
    
    "time","Pt100 2"
     2018-03-28 02:00:05, 12.748
     2018-03-28 02:00:10, 12.747
     2018-03-28 02:00:15, 12.746
     2018-03-28 02:00:20, 12.747
    


.. code:: ipython3

    ts = pd.read_csv(filepath, sep=',', index_col=0, parse_dates=True, 
                     squeeze=True, engine='python')  # the latter 2 kwargs are to ensure a pd.Series is returned
    ts = ts.tz_localize('Europe/Amsterdam')  # set the timezone

.. code:: ipython3

    ts.head()  # Double check the timezone




.. parsed-literal::

    time
    2018-03-28 02:00:05+02:00    12.748
    2018-03-28 02:00:10+02:00    12.747
    2018-03-28 02:00:15+02:00    12.746
    2018-03-28 02:00:20+02:00    12.747
    2018-03-28 02:00:26+02:00    12.747
    Name: Pt100 2, dtype: float64



Now we quickly create a DataStore from xml-files with Stokes
measurements to add the external timeseries to

.. code:: ipython3

    filepath_ds = os.path.join('..', '..', 'tests', 'data', 'double_ended2')
    ds = read_silixa_files(directory=filepath_ds,
                           timezone_netcdf='UTC',
                           file_ext='*.xml')


.. parsed-literal::

    6 files were found, each representing a single timestep
    6 recorded vars were found: LAF, ST, AST, REV-ST, REV-AST, TMP
    Recorded at 1693 points along the cable
    The measurement is double ended
    Reading the data from disk


Step 2: Add the temperature measurements of the external probe to the DataStore.
--------------------------------------------------------------------------------

First add the coordinates

.. code:: ipython3

    ds.coords['time_external'] = ts.index.values

Second we add the measured values

.. code:: ipython3

    ds['external_probe'] = (('time_external',), ts)

Step 3: Align the time of the external measurements to the Stokes measurement times
-----------------------------------------------------------------------------------

We linearly interpolate the measurements of the external sensor to the
times we have DTS measurements

.. code:: ipython3

    ds['external_probe_dts'] = ds['external_probe'].interp(time_external=ds.time)

.. code:: ipython3

    print(ds.data_vars)


.. parsed-literal::

    Data variables:
        ST                     (x, time) float64 1.281 -0.5321 ... -43.44 -41.08
        AST                    (x, time) float64 0.4917 1.243 ... -30.14 -32.09
        REV-ST                 (x, time) float64 0.4086 -0.568 ... 4.822e+03
        REV-AST                (x, time) float64 2.569 -1.603 ... 4.224e+03
        TMP                    (x, time) float64 196.1 639.1 218.7 ... 8.442 18.47
        acquisitionTime        (time) float32 2.098 2.075 2.076 2.133 2.085 2.062
        referenceTemperature   (time) float32 21.0536 21.054 ... 21.0531 21.057
        probe1Temperature      (time) float32 4.36149 4.36025 ... 4.36021 4.36118
        probe2Temperature      (time) float32 18.5792 18.5785 ... 18.5805 18.5723
        referenceProbeVoltage  (time) float32 0.121704 0.121704 ... 0.121705
        probe1Voltage          (time) float32 0.114 0.114 0.114 0.114 0.114 0.114
        probe2Voltage          (time) float32 0.121 0.121 0.121 0.121 0.121 0.121
        userAcquisitionTimeFW  (time) float32 2.0 2.0 2.0 2.0 2.0 2.0
        userAcquisitionTimeBW  (time) float32 2.0 2.0 2.0 2.0 2.0 2.0
        external_probe         (time_external) float64 12.75 12.75 ... 12.76 12.76
        external_probe_dts     (time) float64 12.75 12.75 12.75 12.75 12.75 12.75


Now we can use ``external_probe_dts`` when we define sections and use it
for calibration


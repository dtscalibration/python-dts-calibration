A2. Loading sensornet files
===========================

This example loads sensornet files. Both single-ended and double-ended
measurements are supported.

.. code:: ipython3

    import os
    import glob
    
    from dtscalibration import read_sensornet_files

The example data files are located in
``./python-dts-calibration/tests/data``.

.. code:: ipython3

    filepath = os.path.join('..', '..', 'tests', 'data', 'sensornet_oryx_v3.7')
    print(filepath)


.. parsed-literal::

    ..\..\tests\data\sensornet_oryx_v3.7
    

.. code:: ipython3

    filepathlist = sorted(glob.glob(os.path.join(filepath, '*.ddf')))
    filenamelist = [os.path.basename(path) for path in filepathlist]
    
    for fn in filenamelist:
        print(fn)


.. parsed-literal::

    channel 1 20180107 202119 00001.ddf
    channel 1 20180107 202149 00001.ddf
    channel 1 20180107 202219 00001.ddf
    channel 1 20180107 202249 00001.ddf
    channel 1 20180107 202319 00001.ddf
    channel 1 20180107 202349 00001.ddf
    channel 1 20180107 202418 00001.ddf
    

We will simply load in the sensornet files. As the sensornet files are
of low spatial and temporal resolution, reading the data lazily into
dask is not supported.

.. code:: ipython3

    ds = read_sensornet_files(directory=filepath)


.. parsed-literal::

    7 files were found, each representing a single timestep
    Recorded at 2068 points along the cable
    The measurement is single ended
    

The object tries to gather as much metadata from the measurement files
as possible (temporal and spatial coordinates, filenames, temperature
probes measurements). All other configuration settings are loaded from
the first files and stored as attributes of the ``DataStore``.

.. code:: ipython3

    print(ds)


.. parsed-literal::

    <dtscalibration.DataStore>
    Sections:                  ()
    Dimensions:                (time: 7, x: 2068)
    Coordinates:
      * x                      (x) float64 -747.0 -746.0 ... 1.349e+03 1.35e+03
        filename               (time) <U35 'channel 1 20180107 202119 00001.ddf' ... 'channel 1 20180107 202418 00001.ddf'
        timestart              (time) datetime64[ns] 2018-01-07T20:20:49 ... 2018-01-07T20:23:48
        timeend                (time) datetime64[ns] 2018-01-07T20:21:19 ... 2018-01-07T20:24:18
      * time                   (time) datetime64[ns] 2018-01-07T20:21:04 ... 2018-01-07T20:24:03
        acquisitiontimeFW      (time) timedelta64[ns] 00:00:30 00:00:30 ... 00:00:30
    Data variables:
        ST                     (x, time) float64 0.675 -0.21 ... -0.036 -0.102
        AST                    (x, time) float64 -0.228 -0.049 ... 0.057 0.399
        TMP                    (x, time) float64 -64.76 -92.32 ... 407.0 -200.0
        probe1Temperature      (time) float64 3.12 3.09 3.09 3.07 3.07 3.12 3.07
        probe2Temperature      (time) float64 1.259e+03 1.259e+03 ... 1.259e+03
        referenceTemperature   (time) float64 15.34 15.37 15.34 ... 15.34 15.31
        gamma_ddf              (time) float64 498.8 498.8 498.8 ... 498.8 498.8
        k_internal             (time) float64 0.2786 0.2787 0.2786 ... 0.2785 0.2785
        k_external             (time) float64 0.2786 0.2787 0.2786 ... 0.2785 0.2785
        userAcquisitionTimeFW  (time) float64 30.0 30.0 30.0 30.0 30.0 30.0 30.0
        userAcquisitionTimeBW  (time) float64 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    Attributes:
        DTS Sentinel unit serial number::  SN409017
        Multiplexer serial number::        ORYX
        Hardware model number::            OX4
        Software version number::          ORYX F/W v1.02 Oryx Data Collector v3....
        data status:                       ok
        installation:                      speulderbos2017nov21
        differential loss correction:      single-ended
        forward channel:                   channel 1
        reverse channel:                   N/A
    
    .. and many more attributes. See: ds.attrs
    

Double ended sensornet files are also supported. Note the REV-ST and
REV-AST data variables.

.. code:: ipython3

    filepath = os.path.join('..', '..', 'tests', 'data', 'sensornet_halo_v1.0')
    ds = read_sensornet_files(directory=filepath)
    print(ds)


.. parsed-literal::

    5 files were found, each representing a single timestep
    Recorded at 978 points along the cable
    The measurement is double ended
    <dtscalibration.DataStore>
    Sections:                  ()
    Dimensions:                (time: 5, x: 664)
    Coordinates:
      * x                      (x) float64 -49.28 -47.25 ... 1.294e+03 1.296e+03
        filename               (time) <U32 'channel 1 20030111 002 00001.ddf' ... 'channel 1 20030111 002 00005.ddf'
        timeFWstart            (time) datetime64[ns] 2003-01-11T03:05:09 ... 2003-01-11T03:21:09
        timeFWend              (time) datetime64[ns] 2003-01-11T03:06:09 ... 2003-01-11T03:22:09
        timeFW                 (time) datetime64[ns] 2003-01-11T03:05:39 ... 2003-01-11T03:21:39
        timeBWstart            (time) datetime64[ns] 2003-01-11T03:06:09 ... 2003-01-11T03:22:09
        timeBWend              (time) datetime64[ns] 2003-01-11T03:07:09 ... 2003-01-11T03:23:09
        timeBW                 (time) datetime64[ns] 2003-01-11T03:06:39 ... 2003-01-11T03:22:39
        timestart              (time) datetime64[ns] 2003-01-11T03:05:09 ... 2003-01-11T03:21:09
        timeend                (time) datetime64[ns] 2003-01-11T03:07:09 ... 2003-01-11T03:23:09
      * time                   (time) datetime64[ns] 2003-01-11T03:06:09 ... 2003-01-11T03:22:09
        acquisitiontimeFW      (time) timedelta64[ns] 00:01:00 00:01:00 ... 00:01:00
        acquisitiontimeBW      (time) timedelta64[ns] 00:01:00 00:01:00 ... 00:01:00
    Data variables:
        ST                     (x, time) float64 1.882e+03 1.876e+03 ... 359.0 358.7
        AST                    (x, time) float64 2.137e+03 2.135e+03 ... 347.3 347.2
        TMP                    (x, time) float64 84.19 71.0 81.6 ... 38.62 38.69
        probe1Temperature      (time) float64 nan nan nan nan nan
        probe2Temperature      (time) float64 nan nan nan nan nan
        referenceTemperature   (time) float64 34.42 34.31 34.25 34.25 34.25
        gamma_ddf              (time) float64 510.4 510.4 510.4 510.4 510.4
        k_internal             (time) float64 0.1902 0.1898 0.1898 0.1898 0.1898
        k_external             (time) float64 0.1902 0.1898 0.1898 0.1898 0.1898
        userAcquisitionTimeFW  (time) float64 60.05 60.05 60.05 60.05 60.05
        userAcquisitionTimeBW  (time) float64 60.08 60.06 60.05 60.05 60.05
        REV-ST                 (x, time) float64 354.1 352.1 ... 1.76e+03 1.759e+03
        REV-AST                (x, time) float64 342.0 340.5 ... 2.242e+03 2.241e+03
    Attributes:
        DTS Sentinel unit serial number::  SN307009
        Multiplexer serial number::        multiplexer serial number
        Hardware model number::            HL4
        Software version number::          Halo DTS v1.0
        data status:                       ok
        installation:                      NYAN30AUG2019
        differential loss correction:      combined
        forward channel:                   channel 1
        reverse channel:                   channel 1 reverse
    
    .. and many more attributes. See: ds.attrs
    

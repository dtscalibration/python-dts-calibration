A2. Loading sensornet files
===========================

This example loads sensornet files. Both single-ended and double-ended
measurements are supported.

.. code:: ipython3

    import os
    import glob
    
    from dtscalibration import read_sensornet_files


.. parsed-literal::

    /usr/lib/python3.7/typing.py:845: FutureWarning: xarray subclass DataStore should explicitly define __slots__
      super().__init_subclass__(*args, **kwargs)


The example data files are located in
``./python-dts-calibration/tests/data``.

.. code:: ipython3

    filepath = os.path.join('..', '..', 'tests', 'data', 'sensornet_oryx_v3.7')
    print(filepath)


.. parsed-literal::

    ../../tests/data/sensornet_oryx_v3.7


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
    Dimensions:                (time: 7, x: 1380)
    Coordinates:
      * x                      (x) float64 -49.97 -48.96 ... 1.348e+03 1.349e+03
        filename               (time) <U35 'channel 1 20180107 202119 00001.ddf' ... 'channel 1 20180107 202418 00001.ddf'
        timestart              (time) datetime64[ns] 2018-01-07T20:20:49 ... 2018-01-07T20:23:48
        timeend                (time) datetime64[ns] 2018-01-07T20:21:19 ... 2018-01-07T20:24:18
      * time                   (time) datetime64[ns] 2018-01-07T20:21:04 ... 2018-01-07T20:24:03
        acquisitiontimeFW      (time) timedelta64[ns] 00:00:30 00:00:30 ... 00:00:30
    Data variables:
        st                     (x, time) float64 1.482e+03 1.482e+03 ... -0.324
        ast                    (x, time) float64 956.4 956.4 954.7 ... -0.121 0.458
        tmp                    (x, time) float64 17.5 17.51 17.22 ... 700.0 312.9
        probe1Temperature      (time) float64 3.12 3.09 3.09 3.07 3.07 3.12 3.07
        probe2Temperature      (time) float64 1.259e+03 1.259e+03 ... 1.259e+03
        referenceTemperature   (time) float64 15.34 15.37 15.34 ... 15.34 15.31
        gamma_ddf              (time) float64 498.8 498.8 498.8 ... 498.8 498.8
        k_internal             (time) float64 0.2786 0.2787 0.2786 ... 0.2785 0.2785
        k_external             (time) float64 0.2786 0.2787 0.2786 ... 0.2785 0.2785
        userAcquisitionTimeFW  (time) float64 30.0 30.0 30.0 30.0 30.0 30.0 30.0
        userAcquisitionTimeBW  (time) float64 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    Attributes:
        DTS Sentinel unit serial number::  SN409017\n
        Multiplexer serial number::        ORYX\n
        Hardware model number::            OX4\n
        Software version number::          ORYX F/W v1,02 Oryx Data Collector v3....
        data status:                       ok\n
        installation:                      speulderbos2017nov21\n
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
    Dimensions:                (time: 5, x: 712)
    Coordinates:
      * x                      (x) float64 -49.28 -47.25 ... 1.391e+03 1.393e+03
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
        st                     (x, time) float64 1.882e+03 1.876e+03 ... -0.54
        ast                    (x, time) float64 2.137e+03 2.135e+03 ... -0.681
        tmp                    (x, time) float64 84.19 71.0 81.6 ... -44.31 -200.0
        probe1Temperature      (time) float64 nan nan nan nan nan
        probe2Temperature      (time) float64 nan nan nan nan nan
        referenceTemperature   (time) float64 34.42 34.31 34.25 34.25 34.25
        gamma_ddf              (time) float64 510.4 510.4 510.4 510.4 510.4
        k_internal             (time) float64 0.1902 0.1898 0.1898 0.1898 0.1898
        k_external             (time) float64 0.1902 0.1898 0.1898 0.1898 0.1898
        userAcquisitionTimeFW  (time) float64 60.05 60.05 60.05 60.05 60.05
        userAcquisitionTimeBW  (time) float64 60.08 60.06 60.05 60.05 60.05
        rst                    (x, time) float64 -0.384 -0.36 ... 1.76e+03 1.759e+03
        rast                   (x, time) float64 -0.535 -0.633 ... 2.241e+03
    Attributes:
        DTS Sentinel unit serial number::  SN307009\n
        Multiplexer serial number::        multiplexer serial number\n
        Hardware model number::            HL4\n
        Software version number::          Halo DTS v1.0\n
        data status:                       ok\n
        installation:                      NYAN30AUG2019\n
        differential loss correction:      combined
        forward channel:                   channel 1
        reverse channel:                   channel 1 reverse
    
    .. and many more attributes. See: ds.attrs


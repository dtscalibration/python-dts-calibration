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


.. parsed-literal::

    /home/bart/git/travis_fix/python-dts-calibration/src/dtscalibration/io.py:1835: FutureWarning: Using .astype to convert from timezone-aware dtype to timezone-naive dtype is deprecated and will raise in a future version.  Use obj.tz_localize(None) or obj.tz_convert('UTC').tz_localize(None) instead
      'time', pd.DatetimeIndex(v).tz_localize(


The object tries to gather as much metadata from the measurement files
as possible (temporal and spatial coordinates, filenames, temperature
probes measurements). All other configuration settings are loaded from
the first files and stored as attributes of the ``DataStore``.

.. code:: ipython3

    print(ds)


.. parsed-literal::

    <dtscalibration.DataStore>
    Sections:                  ()
    Dimensions:                (x: 1380, time: 7, trans_att: 0)
    Coordinates:
      * x                      (x) float64 -49.97 -48.96 ... 1.348e+03 1.349e+03
        filename               (time) <U35 'channel 1 20180107 202119 00001.ddf' ...
        timestart              (time) datetime64[ns] 2018-01-07T20:20:49 ... 2018...
        timeend                (time) datetime64[ns] 2018-01-07T20:21:19 ... 2018...
      * time                   (time) datetime64[ns] 2018-01-07T20:21:04 ... 2018...
        acquisitiontimeFW      (time) timedelta64[ns] 00:00:30 00:00:30 ... 00:00:30
      * trans_att              (trans_att) float64 
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
    Attributes: (12/21)
        DTS Sentinel unit serial number:  SN409017
        Multiplexer serial number:        ORYX
        Hardware model number:            OX4
        Software version number:          ORYX F/W v1,02 Oryx Data Collector v3.7...
        data status:                      ok
        installation:                     speulderbos2017nov21
        ...                               ...
        fibre end:                        0.00
        default loss term dB per km:      0.3730
    
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
    Dimensions:                (x: 712, time: 5, trans_att: 0)
    Coordinates: (12/14)
      * x                      (x) float64 -49.28 -47.25 ... 1.391e+03 1.393e+03
        filename               (time) <U32 'channel 1 20030111 002 00003.ddf' ......
        timeFWstart            (time) datetime64[ns] 2003-01-11T03:13:10 ... 2003...
        timeFWend              (time) datetime64[ns] 2003-01-11T03:14:10 ... 2003...
        timeFW                 (time) datetime64[ns] 2003-01-11T03:13:40 ... 2003...
        timeBWstart            (time) datetime64[ns] 2003-01-11T03:14:10 ... 2003...
        ...                     ...
        timestart              (time) datetime64[ns] 2003-01-11T03:13:10 ... 2003...
        timeend                (time) datetime64[ns] 2003-01-11T03:15:10 ... 2003...
      * time                   (time) datetime64[ns] 2003-01-11T03:14:10 ... 2003...
        acquisitiontimeFW      (time) timedelta64[ns] 00:01:00 00:01:00 ... 00:01:00
        acquisitiontimeBW      (time) timedelta64[ns] 00:01:00 00:01:00 ... 00:01:00
      * trans_att              (trans_att) float64 
    Data variables: (12/13)
        st                     (x, time) float64 1.877e+03 1.876e+03 ... -0.54
        ast                    (x, time) float64 2.139e+03 2.138e+03 ... -0.681
        tmp                    (x, time) float64 81.6 60.57 71.0 ... -47.22 -200.0
        probe1Temperature      (time) float64 nan nan nan nan nan
        probe2Temperature      (time) float64 nan nan nan nan nan
        referenceTemperature   (time) float64 34.25 34.25 34.31 34.42 34.25
        ...                     ...
        k_internal             (time) float64 0.1898 0.1898 0.1898 0.1902 0.1898
        k_external             (time) float64 0.1898 0.1898 0.1898 0.1902 0.1898
        userAcquisitionTimeFW  (time) float64 60.05 60.05 60.05 60.05 60.05
        userAcquisitionTimeBW  (time) float64 60.05 60.05 60.06 60.08 60.05
        rst                    (x, time) float64 -0.504 -0.459 ... 1.759e+03
        rast                   (x, time) float64 -0.622 -0.663 ... 2.241e+03
    Attributes: (12/21)
        DTS Sentinel unit serial number:  SN307009
        Multiplexer serial number:        multiplexer serial number
        Hardware model number:            HL4
        Software version number:          Halo DTS v1.0
        data status:                      ok
        installation:                     NYAN30AUG2019
        ...                               ...
        fibre end:                        1298.10
        default loss term dB per km:      0.3938
    
    .. and many more attributes. See: ds.attrs


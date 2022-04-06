1. Load your first measurement files
====================================

This notebook is located in
https://github.com/bdestombe/python-dts-calibration/tree/master/examples/notebooks

The goal of this notebook is to show the different options of loading
measurements from raw DTS files. These files are loaded into a
``DataStore`` object. This object has various methods for calibration,
plotting. The current supported devices are: - Silixa - Sensornet

This example loads Silixa files. Both single-ended and double-ended
measurements are supported. The first step is to load the correct read
routine from ``dtscalibration``. - Silixa ->
``dtscalibration.read_silixa_files`` - Sensornet ->
``dtscalibration.read_sensornet_files``

.. code:: ipython3

    import os
    import glob
    
    from dtscalibration import read_silixa_files

The example data files are located in
``./python-dts-calibration/tests/data``.

.. code:: ipython3

    filepath = os.path.join('..', '..', 'tests', 'data', 'double_ended2')
    print(filepath)


.. parsed-literal::

    ../../tests/data/double_ended2


.. code:: ipython3

    # Bonus: Just to show which files are in the folder
    filepathlist = sorted(glob.glob(os.path.join(filepath, '*.xml')))
    filenamelist = [os.path.basename(path) for path in filepathlist]
    
    for fn in filenamelist:
        print(fn)


.. parsed-literal::

    channel 1_20180328014052498.xml
    channel 1_20180328014057119.xml
    channel 1_20180328014101652.xml
    channel 1_20180328014106243.xml
    channel 1_20180328014110917.xml
    channel 1_20180328014115480.xml


Define in which timezone the measurements are taken. In this case it is
the timezone of the Silixa Ultima computer was set to
‘Europe/Amsterdam’. The default timezone of netCDF files is ``UTC``. All
the steps after loading the raw files are performed in this timezone.
Please see www..com for a full list of supported timezones. We also
explicitely define the file extension (``.xml``) because the folder is
polluted with files other than measurement files.

.. code:: ipython3

    ds = read_silixa_files(directory=filepath,
                           timezone_netcdf='UTC',
                           file_ext='*.xml')


.. parsed-literal::

    6 files were found, each representing a single timestep
    6 recorded vars were found: LAF, ST, AST, REV-ST, REV-AST, TMP
    Recorded at 1693 points along the cable
    The measurement is double ended
    Reading the data from disk


.. parsed-literal::

    /home/bart/git/travis_fix/python-dts-calibration/src/dtscalibration/io.py:1843: FutureWarning: Using .astype to convert from timezone-aware dtype to timezone-naive dtype is deprecated and will raise in a future version.  Use obj.tz_localize(None) or obj.tz_convert('UTC').tz_localize(None) instead
      'time', pd.DatetimeIndex(v).tz_convert(timezone_netcdf).astype(


The object tries to gather as much metadata from the measurement files
as possible (temporal and spatial coordinates, filenames, temperature
probes measurements). All other configuration settings are loaded from
the first files and stored as attributes of the ``DataStore``.

.. code:: ipython3

    print(ds)


.. parsed-literal::

    <dtscalibration.DataStore>
    Sections:                  ()
    Dimensions:                (x: 1693, time: 6, trans_att: 0)
    Coordinates: (12/15)
      * x                      (x) float64 -80.5 -80.38 -80.25 ... 134.3 134.4 134.5
        filename               (time) <U31 'channel 1_20180328014052498.xml' ... ...
        filename_tstamp        (time) int64 20180328014052498 ... 20180328014115480
        timeFWstart            (time) datetime64[ns] 2018-03-28T00:40:52.097000 ....
        timeFWend              (time) datetime64[ns] 2018-03-28T00:40:54.097000 ....
        timeFW                 (time) datetime64[ns] 2018-03-28T00:40:53.097000 ....
        ...                     ...
        timestart              (time) datetime64[ns] 2018-03-28T00:40:52.097000 ....
        timeend                (time) datetime64[ns] 2018-03-28T00:40:56.097000 ....
      * time                   (time) datetime64[ns] 2018-03-28T00:40:54.097000 ....
        acquisitiontimeFW      (time) timedelta64[ns] 00:00:02 00:00:02 ... 00:00:02
        acquisitiontimeBW      (time) timedelta64[ns] 00:00:02 00:00:02 ... 00:00:02
      * trans_att              (trans_att) float64 
    Data variables: (12/14)
        st                     (x, time) float64 1.281 -0.5321 ... -43.44 -41.08
        ast                    (x, time) float64 0.4917 1.243 ... -30.14 -32.09
        rst                    (x, time) float64 0.4086 -0.568 ... 4.822e+03
        rast                   (x, time) float64 2.569 -1.603 ... 4.224e+03
        tmp                    (x, time) float64 196.1 639.1 218.7 ... 8.442 18.47
        acquisitionTime        (time) float32 2.098 2.075 2.076 2.133 2.085 2.062
        ...                     ...
        probe2Temperature      (time) float32 18.58 18.58 18.58 18.58 18.58 18.57
        referenceProbeVoltage  (time) float32 0.1217 0.1217 0.1217 ... 0.1217 0.1217
        probe1Voltage          (time) float32 0.114 0.114 0.114 0.114 0.114 0.114
        probe2Voltage          (time) float32 0.121 0.121 0.121 0.121 0.121 0.121
        userAcquisitionTimeFW  (time) float32 2.0 2.0 2.0 2.0 2.0 2.0
        userAcquisitionTimeBW  (time) float32 2.0 2.0 2.0 2.0 2.0 2.0
    Attributes: (12/351)
        uid:                                                                     ...
        nameWell:                                                                ...
        nameWellbore:                                                            ...
        name:                                                                    ...
        indexType:                                                               ...
        startIndex:uom:                                                          ...
        ...                                                                                                                                                    ...
        customData:UserConfiguration:ChannelConfiguration_3:FibreCorrectionConfig...
        customData:UserConfiguration:ChannelConfiguration_3:FibreCorrectionConfig...
    
    .. and many more attributes. See: ds.attrs




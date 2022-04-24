A3. Loading AP Sensing files
============================

This example loads AP sensing files. Only single-ended files are
currently supported. Just like with Silixa’s devices, the AP Sensing
data is in .xml files

.. code:: ipython3

    import os
    import glob
    
    from dtscalibration import read_apsensing_files

The example data files are located in
``./python-dts-calibration/tests/data``.

.. code:: ipython3

    filepath = os.path.join('..', '..', 'tests', 'data', 'ap_sensing')
    print(filepath)


.. parsed-literal::

    ../../tests/data/ap_sensing


.. code:: ipython3

    filepathlist = sorted(glob.glob(os.path.join(filepath, '*.xml')))
    filenamelist = [os.path.basename(path) for path in filepathlist]
    
    for fn in filenamelist:
        print(fn)


.. parsed-literal::

    _AP Sensing_N4386B_3_20180118201727.xml
    _AP Sensing_N4386B_3_20180118202957.xml
    _AP Sensing_N4386B_3_20180118205357.xml


We will simply load in the .xml files

.. code:: ipython3

    ds = read_apsensing_files(directory=filepath)


.. parsed-literal::

    3 files were found, each representing a single timestep
    4 recorded vars were found: LAF, TEMP, ST, AST
    Recorded at 7101 points along the cable
    The measurement is single ended
    Reading the data from disk


The object tries to gather as much metadata from the measurement files
as possible (temporal and spatial coordinates, filenames, temperature
probes measurements). All other configuration settings are loaded from
the first files and stored as attributes of the ``DataStore``.

Calibration follows as usual (see the other notebooks).

.. code:: ipython3

    print(ds)


.. parsed-literal::

    <dtscalibration.DataStore>
    Sections:                  ()
    Dimensions:       (x: 7101, time: 3, trans_att: 0)
    Coordinates:
      * x             (x) float64 0.0 0.5 1.0 1.5 ... 3.549e+03 3.55e+03 3.55e+03
        filename      (time) <U39 '_AP Sensing_N4386B_3_20180118201727.xml' ... '...
      * time          (time) datetime64[ns] 2018-01-18T20:17:27 ... 2018-01-18T20...
      * trans_att     (trans_att) float64 
    Data variables:
        tmp           (x, time) float64 12.16 11.32 12.26 ... 17.68 15.08 17.83
        st            (x, time) float64 1.098 1.105 1.101 ... 3.39e-18 3.409e-18
        ast           (x, time) float64 0.1888 0.1891 0.1895 ... 4.838e-19 4.945e-19
        creationDate  (time) datetime64[ns] 2018-01-18T20:17:27 ... 2018-01-18T20...
    Attributes: (12/52)
        wellbore:uid:                                                            ...
        wellbore:name:                                                           ...
        wellbore:dtsInstalledSystemSet:dtsInstalledSystem:uid:                   ...
        wellbore:dtsInstalledSystemSet:dtsInstalledSystem:name:                  ...
        wellbore:dtsInstalledSystemSet:dtsInstalledSystem:fiberInformation:fiber:...
        wellbore:dtsInstalledSystemSet:dtsInstalledSystem:fiberInformation:fiber:...
        ...                                                                             ...
        wellbore:wellLogSet:wellLog:blockInfo:blockCurveInfo_3:curveId:          ...
        wellbore:wellLogSet:wellLog:blockInfo:blockCurveInfo_3:columnIndex:      ...
    
    .. and many more attributes. See: ds.attrs


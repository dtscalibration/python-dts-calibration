A4. Loading sensortran files
============================

This example loads sensortran files. Only single-ended measurements are
currently supported. Sensortran files are in binary format. The library
requires the ``*BinaryRawDTS.dat`` and ``*BinaryTemp.dat`` files.

.. code:: ipython3

    import os
    import glob
    import matplotlib.pyplot as plt
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
        
    from dtscalibration import read_sensortran_files


.. parsed-literal::

    /usr/lib/python3.7/typing.py:845: FutureWarning: xarray subclass DataStore should explicitly define __slots__
      super().__init_subclass__(*args, **kwargs)


The example data files are located in
``./python-dts-calibration/tests/data``.

.. code:: ipython3

    filepath = os.path.join('..', '..', 'tests', 'data', 'sensortran_binary')
    print(filepath)


.. parsed-literal::

    ../../tests/data/sensortran_binary


.. code:: ipython3

    filepathlist = sorted(glob.glob(os.path.join(filepath, '*.dat')))
    filenamelist = [os.path.basename(path) for path in filepathlist]
    
    for fn in filenamelist:
        print(fn)


.. parsed-literal::

    15_56_47_BinaryRawDTS.dat
    15_56_47_BinaryTemp.dat
    16_11_31_BinaryRawDTS.dat
    16_11_31_BinaryTemp.dat
    16_29_23_BinaryRawDTS.dat
    16_29_23_BinaryTemp.dat


We will simply load in the binary files

.. code:: ipython3

    ds = read_sensortran_files(directory=filepath)


.. parsed-literal::

    3 files were found, each representing a single timestep
    Recorded at 11582 points along the cable
    The measurement is single ended


The object tries to gather as much metadata from the measurement files
as possible (temporal and spatial coordinates, filenames, temperature
probes measurements). All other configuration settings are loaded from
the first files and stored as attributes of the ``DataStore``.
Sensortran's data files contain less information than the other
manufacturer's devices, one being the acquisition time. The acquisition
time is needed for estimating variances, and is set a constant 1s.

.. code:: ipython3

    print(ds)


.. parsed-literal::

    <dtscalibration.DataStore>
    Sections:                  ()
    Dimensions:                (time: 3, x: 11582)
    Coordinates:
      * x                      (x) float32 -451.37958 -450.87354 ... 5408.9644
        filename               (time) <U25 '15_56_47_BinaryRawDTS.dat' ... '16_29_23_BinaryRawDTS.dat'
        filename_temp          (time) <U23 '15_56_47_BinaryTemp.dat' ... '16_29_23_BinaryTemp.dat'
        timestart              (time) datetime64[ns] 2009-09-24T00:56:46 ... 2009-09-24T01:29:22
        timeend                (time) datetime64[ns] 2009-09-24T00:56:47 ... 2009-09-24T01:29:23
      * time                   (time) datetime64[ns] 2009-09-24T00:56:47 ... 2009-09-24T01:29:23
        acquisitiontimeFW      (time) timedelta64[ns] 00:00:01 00:00:01 00:00:01
    Data variables:
        st                     (x, time) int32 39040680 39057147 ... 39071213
        ast                    (x, time) int32 39048646 39064414 ... 39407668
        tmp                    (x, time) float64 -273.1 -273.1 ... 82.41 82.71
        referenceTemperature   (time) float64 28.61 29.24 30.29
        st_zero                (time) float64 3.904e+07 3.906e+07 3.907e+07
        ast_zero               (time) float64 3.905e+07 3.907e+07 3.908e+07
        userAcquisitionTimeFW  (time) float64 1.0 1.0 1.0
    Attributes:
        survey_type:                 2
        hdr_version:                 3
        x_units:                     n/a
        y_units:                     counts
        num_points:                  12000
        num_pulses:                  25000
        channel_id:                  1
        num_subtraces:               354
        num_skipped:                 0
    
    .. and many more attributes. See: ds.attrs


The sensortran files differ from other manufacturers, in that they
return the 'counts' of the Stokes and anti-Stokes signals. These are not
corrected for offsets, which has to be done manually for proper
calibration.

Based on the data available in the binary files, the library estimates a
zero-count to correct the signals, but this is not perfectly accurate or
constant over time. For proper calibration, the offsets would have to be
incorporated into the calibration routine.

.. code:: ipython3

    ds




.. raw:: html

    <pre>&lt;dtscalibration.DataStore&gt;
    Sections:                  ()
    Dimensions:                (time: 3, x: 11582)
    Coordinates:
      * x                      (x) float32 -451.37958 -450.87354 ... 5408.9644
        filename               (time) &lt;U25 &#x27;15_56_47_BinaryRawDTS.dat&#x27; ... &#x27;16_29_23_BinaryRawDTS.dat&#x27;
        filename_temp          (time) &lt;U23 &#x27;15_56_47_BinaryTemp.dat&#x27; ... &#x27;16_29_23_BinaryTemp.dat&#x27;
        timestart              (time) datetime64[ns] 2009-09-24T00:56:46 ... 2009-09-24T01:29:22
        timeend                (time) datetime64[ns] 2009-09-24T00:56:47 ... 2009-09-24T01:29:23
      * time                   (time) datetime64[ns] 2009-09-24T00:56:47 ... 2009-09-24T01:29:23
        acquisitiontimeFW      (time) timedelta64[ns] 00:00:01 00:00:01 00:00:01
    Data variables:
        st                     (x, time) int32 39040680 39057147 ... 39071213
        ast                    (x, time) int32 39048646 39064414 ... 39407668
        tmp                    (x, time) float64 -273.1 -273.1 ... 82.41 82.71
        referenceTemperature   (time) float64 28.61 29.24 30.29
        st_zero                (time) float64 3.904e+07 3.906e+07 3.907e+07
        ast_zero               (time) float64 3.905e+07 3.907e+07 3.908e+07
        userAcquisitionTimeFW  (time) float64 1.0 1.0 1.0
    Attributes:
        survey_type:                 2
        hdr_version:                 3
        x_units:                     n/a
        y_units:                     counts
        num_points:                  12000
        num_pulses:                  25000
        channel_id:                  1
        num_subtraces:               354
        num_skipped:                 0
    
    .. and many more attributes. See: ds.attrs</pre>



.. code:: ipython3

    ds0 = ds.isel(time=0)
    
    plt.figure()
    ds0.st.plot(label='Stokes signal')
    plt.axhline(ds0.st_zero.values, c='r', label="'zero' measurement")
    plt.legend()
    plt.title('')
    plt.axhline(c='k')




.. parsed-literal::

    <matplotlib.lines.Line2D at 0x7fcadfa03050>



After a correction and rescaling (for human readability) the data will
look more like other manufacturer's devices

.. code:: ipython3

    ds['st'] = (ds.st - ds.st_zero)/1e4
    ds['ast'] = (ds.ast - ds.ast_zero)/1e4

.. code:: ipython3

    ds.isel(time=0).st.plot(label='Stokes intensity')
    ds.isel(time=0).ast.plot(label='anti-Stokes intensity')
    plt.legend()
    plt.axhline(c='k', lw=1)
    plt.xlabel('')
    plt.title('')
    plt.ylim([-50,500])




.. parsed-literal::

    (-50, 500)




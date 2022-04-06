11. Merge two single-ended measurements into a double-ended datastore
=====================================================================

Often people have set up a double ended cable to be measured as if it
were two single ended setups, to prevent loss of data as some
manufacturers throw out some raw data when doing double ended
measurements. To do the double ended calibration the two channels have
to be merged into a single datastore and aligned.

.. code:: ipython3

    import os
    from dtscalibration import read_silixa_files
    from dtscalibration.datastore_utils import suggest_cable_shift_double_ended, shift_double_ended, merge_double_ended
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

We load in both channels into seperate datastores

.. code:: ipython3

    filepath_ch1 = os.path.join('..', '..', 'tests', 'data', 'double_single_ended', 'channel_1')
    filepath_ch2 = os.path.join('..', '..', 'tests', 'data', 'double_single_ended', 'channel_2')
    
    ds_ch1 = read_silixa_files(
        directory=filepath_ch1,
        timezone_netcdf='UTC',
        file_ext='*.xml')
    
    ds_ch2 = read_silixa_files(
        directory=filepath_ch2,
        timezone_netcdf='UTC',
        file_ext='*.xml')


.. parsed-literal::

    6 files were found, each representing a single timestep
    4 recorded vars were found: LAF, ST, AST, TMP
    Recorded at 8936 points along the cable
    The measurement is single ended
    Reading the data from disk


.. parsed-literal::

    /home/bart/git/travis_fix/python-dts-calibration/src/dtscalibration/io.py:1843: FutureWarning: Using .astype to convert from timezone-aware dtype to timezone-naive dtype is deprecated and will raise in a future version.  Use obj.tz_localize(None) or obj.tz_convert('UTC').tz_localize(None) instead
      'time', pd.DatetimeIndex(v).tz_convert(timezone_netcdf).astype(


.. parsed-literal::

    6 files were found, each representing a single timestep
    4 recorded vars were found: LAF, ST, AST, TMP
    Recorded at 8936 points along the cable
    The measurement is single ended
    Reading the data from disk


.. parsed-literal::

    /home/bart/git/travis_fix/python-dts-calibration/src/dtscalibration/io.py:1843: FutureWarning: Using .astype to convert from timezone-aware dtype to timezone-naive dtype is deprecated and will raise in a future version.  Use obj.tz_localize(None) or obj.tz_convert('UTC').tz_localize(None) instead
      'time', pd.DatetimeIndex(v).tz_convert(timezone_netcdf).astype(


When plotting the data the two datasets already look quite similar as it
is a duplex measurement.

.. code:: ipython3

    ds_ch1.isel(time=0).st.plot(label='ST ch1', lw=2)
    ds_ch2.isel(time=0).st.plot(label='ST ch2')
    plt.legend()




.. parsed-literal::

    <matplotlib.legend.Legend at 0x7f27ae5cab20>




.. image:: 11Merge_single_measurements_into_double.ipynb_files/11Merge_single_measurements_into_double.ipynb_6_1.png


The second channel has to be flipped around to align the fibers
correctly. To do this properly, a cable length has to be defined. This
is the distance along the fiber between the connectors of channel 1 and
channel 2.

Make sure the two datasets are aligned time-wise and have the same
amount of measurements in time. Now we can merge the two channels using
the utilify function *merge_double_ended*

.. code:: ipython3

    cable_length = 2017.7
    
    ds = merge_double_ended(ds_fw = ds_ch1,
                            ds_bw = ds_ch2,
                            cable_length = cable_length,
                            plot_result = False)
    
    print((ds.isel(time=0).st - ds.isel(time=0).rst).sum().values)


.. parsed-literal::

    -3712866.0382


To perfectly align the two measurements we can use the alignment
utility. Before we do so, we select only the data of the phyisical cable
and 10 meters of the internal reference coil.

It turns out we were off by 3 datapoints, so let’s shift it by that.

.. code:: ipython3

    ds = ds.sel(x=slice(-10, cable_length + 10))
    
    shift1, shift2 = suggest_cable_shift_double_ended(ds.isel(time=[0,-1]).compute(),
                                                      np.arange(-10, 10, 1, dtype=int))
    
    ds = shift_double_ended(ds, shift1)


.. parsed-literal::

    I dont know what to do with the following data ['tmp']
    I dont know what to do with the following data ['tmp']
    I dont know what to do with the following data ['tmp']



.. image:: 11Merge_single_measurements_into_double.ipynb_files/11Merge_single_measurements_into_double.ipynb_10_1.png


Now we can calibrate the data double ended as usual.

10. Align double ended measurements
===================================

The cable length was initially configured during the DTS measurement.
For double ended measurements it is important to enter the correct
length so that the forward channel and the backward channel are aligned.

This notebook shows how to better align the forward and the backward
measurements. Do this before the calibration steps.

.. code:: ipython3

    import os
    from dtscalibration import read_silixa_files
    from dtscalibration.datastore_utils import suggest_cable_shift_double_ended, shift_double_ended
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

.. code:: ipython3

    suggest_cable_shift_double_ended?

.. code:: ipython3

    filepath = os.path.join('..', '..', 'tests', 'data', 'double_ended2')
    
    ds_aligned = read_silixa_files(
        directory=filepath,
        timezone_netcdf='UTC',
        file_ext='*.xml')  # this one is already correctly aligned


.. parsed-literal::

    6 files were found, each representing a single timestep
    6 recorded vars were found: LAF, ST, AST, REV-ST, REV-AST, TMP
    Recorded at 1693 points along the cable
    The measurement is double ended
    Reading the data from disk


.. parsed-literal::

    /home/bart/git/travis_fix/python-dts-calibration/src/dtscalibration/io.py:1843: FutureWarning: Using .astype to convert from timezone-aware dtype to timezone-naive dtype is deprecated and will raise in a future version.  Use obj.tz_localize(None) or obj.tz_convert('UTC').tz_localize(None) instead
      'time', pd.DatetimeIndex(v).tz_convert(timezone_netcdf).astype(


Because our loaded files were already nicely aligned, we are purposely
offsetting the forward and backward channel by 3 \`spacial indices’.

.. code:: ipython3

    ds_notaligned = shift_double_ended(ds_aligned, 3)


.. parsed-literal::

    I dont know what to do with the following data ['tmp']


The device-calibrated temperature doesnot have a valid meaning anymore
and is dropped

.. code:: ipython3

    suggested_shift = suggest_cable_shift_double_ended(
        ds_notaligned, 
        np.arange(-5, 5), 
        plot_result=True, 
        figsize=(12,8))


.. parsed-literal::

    /home/bart/git/travis_fix/python-dts-calibration/src/dtscalibration/datastore_utils.py:308: RuntimeWarning: invalid value encountered in log
      i_f = np.log(st / ast)
    /home/bart/git/travis_fix/python-dts-calibration/src/dtscalibration/datastore_utils.py:309: RuntimeWarning: invalid value encountered in log
      i_b = np.log(rst / rast)



.. image:: 10Align_double_ended_measurements.ipynb_files/10Align_double_ended_measurements.ipynb_8_1.png


The two approaches suggest a shift of -3 and -4. It is up to the user
which suggestion to follow. Usually the two suggested shift are close

.. code:: ipython3

    ds_restored = shift_double_ended(ds_notaligned, suggested_shift[0])

.. code:: ipython3

    print(ds_aligned.x, 3*'\n', ds_restored.x)


.. parsed-literal::

    <xarray.DataArray 'x' (x: 1693)>
    array([-80.5043, -80.3772, -80.2501, ..., 134.294 , 134.421 , 134.548 ])
    Coordinates:
      * x        (x) float64 -80.5 -80.38 -80.25 -80.12 ... 134.2 134.3 134.4 134.5
    Attributes:
        name:              distance
        description:       Length along fiber
        long_description:  Starting at connector of forward channel
        units:             m 
    
    
     <xarray.DataArray 'x' (x: 1687)>
    array([-80.123 , -79.9959, -79.8688, ..., 133.913 , 134.04  , 134.167 ])
    Coordinates:
      * x        (x) float64 -80.12 -80.0 -79.87 -79.74 ... 133.8 133.9 134.0 134.2
    Attributes:
        name:              distance
        description:       Length along fiber
        long_description:  Starting at connector of forward channel
        units:             m


Note that our fiber has become shorter by 2*3 spatial indices

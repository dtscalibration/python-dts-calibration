12. Creating a DataStore from numpy arrays
==========================================

The goal of this notebook is to demonstrate how to create a
``DataStore`` from scratch. This can be useful if your device is not
supported or if you would like to integrate the ``dtscalibration``
library in your current routine.

.. code:: ipython3

    import numpy as np
    import os
    import matplotlib.pyplot as plt
    import xarray as xr
    
    from dtscalibration import DataStore, read_silixa_files

For a ``DataStore`` object, a few things are needed:

-  timestamps

-  Stokes signal

-  anti-Stokes signal

-  x (length along fiber)

Let’s grab the data from an existing silixa dataset:

.. code:: ipython3

    filepath = os.path.join('..', '..', 'tests', 'data', 'single_ended')
    
    ds_silixa = read_silixa_files(directory=filepath,
                                  silent=True)


.. parsed-literal::

    /home/bart/git/travis_fix/python-dts-calibration/src/dtscalibration/io.py:1843: FutureWarning: Using .astype to convert from timezone-aware dtype to timezone-naive dtype is deprecated and will raise in a future version.  Use obj.tz_localize(None) or obj.tz_convert('UTC').tz_localize(None) instead
      'time', pd.DatetimeIndex(v).tz_convert(timezone_netcdf).astype(


We will get all the numpy arrays from this ``DataStore`` to create a new
one from ‘scratch’.

Let’s start with the most basic data:

.. code:: ipython3

    x = ds_silixa.x.values
    time = ds_silixa.time.values
    ST = ds_silixa.st.values
    AST = ds_silixa.ast.values

Now this data has to be inserted into an xarray ``Dataset``

.. code:: ipython3

    ds = xr.Dataset()
    ds['x'] = ('x', x)
    ds['time'] = ('time', time)
    ds['st'] = (['x', 'time'], ST)
    ds['ast'] = (['x', 'time'], AST)

.. code:: ipython3

    ds = DataStore(ds)
    print(ds)


.. parsed-literal::

    <dtscalibration.DataStore>
    Sections:                  ()
    Dimensions:    (x: 1461, time: 3, trans_att: 0)
    Coordinates:
      * x          (x) float64 -80.74 -80.62 -80.49 -80.36 ... 104.6 104.7 104.8
      * time       (time) datetime64[ns] 2018-05-04T12:22:17.710000 ... 2018-05-0...
      * trans_att  (trans_att) float64 
    Data variables:
        st         (x, time) float64 -0.8058 0.4287 -0.513 ... 27.99 27.83 28.81
        ast        (x, time) float64 -0.2459 -0.5932 0.1111 ... 36.2 35.7 35.16
    Attributes:
        _sections:  null\n...\n


For calibration, a few more paramaters are needed:

-  acquisition time (for calculating residuals for WLS calibration)

-  reference temperatures

-  a double ended flag

We’ll put these into the custom ``DataStore``:

.. code:: ipython3

    ds['acquisitiontimeFW'] = ds_silixa['acquisitiontimeFW'].values
    ds['temp1'] = ds_silixa['probe1Temperature']
    ds['temp2'] = ds_silixa['probe2Temperature']
    
    ds.attrs['isDoubleEnded'] = '0'

Now we can calibrate the data as usual (ordinary least squares in this
example).

.. code:: ipython3

    ds = ds.sel(x=slice(-30, 101))
    sections = {
                'temp1':    [slice(20, 25.5)],  # warm bath
                'temp2':    [slice(5.5, 15.5)],  # cold bath
                }
    ds.sections = sections
    
    ds.calibration_single_ended(method='ols')
    
    ds.isel(time=0).tmpf.plot()




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x7f813e235df0>]



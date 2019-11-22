2. Common DataStore functions
=============================

Examples of how to do some of the more commonly used functions:

1. mean, min, max, std
2. Selecting
3. Selecting by index
4. Downsample (time dimension)
5. Upsample / Interpolation (length and time dimension)

.. code:: ipython3

    import os
    
    from dtscalibration import read_silixa_files

First we load the raw measurements into a ``DataStore`` object, as we
learned from the previous notebook.

.. code:: ipython3

    filepath = os.path.join('..', '..', 'tests', 'data', 'single_ended')
    
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf='UTC',
        file_ext='*.xml')


.. parsed-literal::

    3 files were found, each representing a single timestep
    4 recorded vars were found: LAF, ST, AST, TMP
    Recorded at 1461 points along the cable
    The measurement is single ended
    Reading the data from disk


0 Access the data
-----------------

The implemented read routines try to read as much data from the raw DTS
files as possible. Usually they would have coordinates (time and space)
and Stokes and anti Stokes measurements. We can access the data by key.
It is presented as a DataArray. More examples are found at
http://xarray.pydata.org/en/stable/indexing.html

.. code:: ipython3

    ds['ST']  # is the data stored, presented as a DataArray




.. parsed-literal::

    <xarray.DataArray 'ST' (x: 1461, time: 3)>
    array([[-8.05791e-01,  4.28741e-01, -5.13021e-01],
           [-4.58870e-01, -1.24484e-01,  9.68469e-03],
           [ 4.89174e-01, -9.57734e-02,  5.62837e-02],
           ...,
           [ 4.68457e+01,  4.72201e+01,  4.79139e+01],
           [ 3.76634e+01,  3.74649e+01,  3.83160e+01],
           [ 2.79879e+01,  2.78331e+01,  2.88055e+01]])
    Coordinates:
      * x                  (x) float64 -80.74 -80.62 -80.49 ... 104.6 104.7 104.8
        filename           (time) <U31 'channel 2_20180504132202074.xml' ... 'channel 2_20180504132303723.xml'
        filename_tstamp    (time) int64 20180504132202074 ... 20180504132303723
        timestart          (time) datetime64[ns] 2018-05-04T12:22:02.710000 ... 2018-05-04T12:23:03.716000
        timeend            (time) datetime64[ns] 2018-05-04T12:22:32.710000 ... 2018-05-04T12:23:33.716000
      * time               (time) datetime64[ns] 2018-05-04T12:22:17.710000 ... 2018-05-04T12:23:18.716000
        acquisitiontimeFW  (time) timedelta64[ns] 00:00:30 00:00:30 00:00:30
    Attributes:
        name:         ST
        description:  Stokes intensity
        units:        -



.. code:: ipython3

    ds['TMP'].plot(figsize=(12, 8));

1 mean, min, max
----------------

The first argument is the dimension. The function is taken along that
dimension. ``dim`` can be any dimension (e.g., ``time``, ``x``). The
returned ``DataStore`` does not contain that dimension anymore.

Normally, you would like to keep the attributes (the informative texts
from the loaded files), so set ``keep_attrs`` to ``True``. They don’t
take any space compared to your Stokes data, so keep them.

Note that also the sections are stored as attribute. If you delete the
attributes, you would have to redefine the sections.

.. code:: ipython3

    ds_min = ds.mean(dim='time', keep_attrs=True)  # take the minimum of all data variables (e.g., Stokes, Temperature) along the time dimension

.. code:: ipython3

    ds_max = ds.max(dim='x', keep_attrs=True)  # Take the maximum of all data variables (e.g., Stokes, Temperature) along the x dimension

.. code:: ipython3

    ds_std = ds.std(dim='time', keep_attrs=True)  # Calculate the standard deviation along the time dimension

2 Selecting
-----------

What if you would like to get the maximum temperature between
:math:`x >= 20` m and :math:`x < 35` m over time? We first have to
select a section along the cable.

.. code:: ipython3

    section = slice(20., 35.)
    section_of_interest = ds.sel(x=section)

.. code:: ipython3

    section_of_interest_max = section_of_interest.max(dim='x')

What if you would like to have the measurement at approximately
:math:`x=20` m?

.. code:: ipython3

    point_of_interest = ds.sel(x=20., method='nearest')

3 Selecting by index
--------------------

What if you would like to see what the values on the first timestep are?
We can use isel (index select)

.. code:: ipython3

    section_of_interest = ds.isel(time=slice(0, 2))  # The first two time steps

.. code:: ipython3

    section_of_interest = ds.isel(x=0)

4 Downsample (time dimension)
-----------------------------

We currently have measurements at 3 time steps, with 30.001 seconds
inbetween. For our next exercise we would like to down sample the
measurements to 2 time steps with 47 seconds inbetween. The calculated
variances are not valid anymore. We use the function
``resample_datastore``.

.. code:: ipython3

    ds_resampled = ds.resample_datastore(how='mean', time="47S")

5 Upsample / Interpolation (length and time dimension)
------------------------------------------------------

So we have measurements every 0.12 cm starting at :math:`x=0` m. What if
we would like to change our coordinate system to have a value every 12
cm starting at :math:`x=0.05` m. We use (linear) interpolation,
extrapolation is not supported. The calculated variances are not valid
anymore.

.. code:: ipython3

    x_old = ds.x.data
    x_new = x_old[:-1] + 0.05 # no extrapolation
    ds_xinterped = ds.interp(coords={'x': x_new})

We can do the same in the time dimension

.. code:: ipython3

    import numpy as np
    time_old = ds.time.data
    time_new = time_old + np.timedelta64(10, 's')
    ds_tinterped = ds.interp(coords={'time': time_new})


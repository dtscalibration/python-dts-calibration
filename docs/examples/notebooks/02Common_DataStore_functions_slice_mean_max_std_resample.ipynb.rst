
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

.. code:: ipython3

    try:
        wd = os.path.dirname(os.path.realpath(__file__))
    except:
        wd = os.getcwd()
    
    filepath = os.path.join(wd, '..', '..', 'tests', 'data', 'single_ended')
    timezone_netcdf = 'UTC'
    timezone_input_files = 'Europe/Amsterdam'
    file_ext = '*.xml'
    
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf=timezone_netcdf,
        timezone_input_files=timezone_input_files,
        file_ext=file_ext)


.. parsed-literal::

    3 files were found, each representing a single timestep
    4 recorded vars were found: LAF, ST, AST, TMP
    Recorded at 1461 points along the cable
    The measurement is single ended
    

1 mean, min, max
----------------

The first argument is the dimension. The function is taken along that
dimension. ``dim`` can be any dimension (e.g., ``time``, ``x``). The
returned ``DataStore`` does not contain that dimension anymore.

Normally, you would like to keep the attributes (the informative texts
from the loaded files), so set ``keep_attrs`` to ``True``.

Note that also the sections are stored as attribute. If you delete the
attributes, you would have to redefine the sections.

.. code:: ipython3

    ds_min = ds.mean(dim='time', keep_attrs=True)

.. code:: ipython3

    ds_max = ds.max(dim='x', keep_attrs=True)

.. code:: ipython3

    ds_std = ds.std(dim='time', keep_attrs=True)

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

    section_of_interest = ds.sel(x=20., method='nearest')

3 Selecting by index
--------------------

What if you would like to see what the values on the first timestep are?
We can use isel (index select)

.. code:: ipython3

    section_of_interest = ds.isel(x=0)

.. code:: ipython3

    section_of_interest = ds.isel(time=slice(0, 2))  # The first two time steps

4 Downsample (time dimension)
-----------------------------

We currently have measurements at 3 time steps, with 30.001 seconds
inbetween. For our next exercise we would like to down sample the
measurements to 2 time steps with 46 seconds inbetween. The calculated
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

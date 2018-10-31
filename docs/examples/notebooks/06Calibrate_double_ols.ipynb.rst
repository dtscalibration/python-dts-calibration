
6. Calibration of double ended measurement with OLS
===================================================

A double ended calibration is performed with ordinary least squares.
Over all timesteps simultaneous. :math:`\gamma` and :math:`\alpha`
remain constant, while :math:`C` varies over time. The weights are
considered equal here and no variance is calculated.

.. code:: ipython3

    import os
    
    from dtscalibration import read_silixa_files
    import matplotlib.pyplot as plt
    %matplotlib inline

.. code:: ipython3

    try:
        wd = os.path.dirname(os.path.realpath(__file__))
    except:
        wd = os.getcwd()
    
    filepath = os.path.join(wd, '..', '..', 'tests', 'data', 'double_ended2')
    timezone_netcdf = 'UTC'
    timezone_input_files = 'Europe/Amsterdam'
    file_ext = '*.xml'
    
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf=timezone_netcdf,
        timezone_input_files=timezone_input_files,
        file_ext=file_ext)
    
    ds100 = ds.sel(x=slice(0, 100))  # only calibrate parts of the fiber
    sections = {
        'probe1Temperature': [slice(7.5, 17.), slice(70., 80.)],  # cold bath
        'probe2Temperature': [slice(24., 34.), slice(85., 95.)],  # warm bath
        }


.. parsed-literal::

    6 files were found, each representing a single timestep
    6 recorded vars were found: LAF, ST, AST, REV-ST, REV-AST, TMP
    Recorded at 1693 points along the cable
    The measurement is double ended
    

.. code:: ipython3

    print(ds100.calibration_double_ended.__doc__)


.. parsed-literal::

    
    
            Parameters
            ----------
            sections : dict, optional
            st_label : str
                Label of the forward stokes measurement
            ast_label : str
                Label of the anti-Stoke measurement
            rst_label : str
                Label of the reversed Stoke measurement
            rast_label : str
                Label of the reversed anti-Stoke measurement
            st_var : float, optional
                The variance of the measurement noise of the Stokes signals in the forward
                direction Required if method is wls.
            ast_var : float, optional
                The variance of the measurement noise of the anti-Stokes signals in the forward
                direction. Required if method is wls.
            rst_var : float, optional
                The variance of the measurement noise of the Stokes signals in the backward
                direction. Required if method is wls.
            rast_var : float, optional
                The variance of the measurement noise of the anti-Stokes signals in the backward
                direction. Required if method is wls.
            store_c : str
                Label of where to store C
            store_gamma : str
                Label of where to store gamma
            store_alphaint : str
                Label of where to store alphaint
            store_alpha : str
                Label of where to store alpha
            store_tmpf : str
                Label of where to store the calibrated temperature of the forward direction
            store_tmpb : str
                Label of where to store the calibrated temperature of the backward direction
            variance_suffix : str, optional
                String appended for storing the variance. Only used when method is wls.
            method : {'ols', 'wls'}
                Use 'ols' for ordinary least squares and 'wls' for weighted least squares
            store_tempvar : str
                If defined, the variance of the error is calculated
            conf_ints : iterable object of float, optional
                A list with the confidence boundaries that are calculated. E.g., to cal
            conf_ints_size : int, optional
                Size of the monte carlo parameter set used to calculate the confidence interval
            ci_avg_time_flag : bool, optional
                The confidence intervals differ per time step. If you would like to calculate confidence
                intervals of all time steps together. ‘We can say with 95% confidence that the
                temperature remained between this line and this line during the entire measurement
                period’.
            da_random_state : dask.array.random.RandomState
                The seed for dask. Makes random not so random. To produce reproducable results for
                testing environments.
            solver : {'sparse', 'stats'}
                Either use the homemade weighted sparse solver or the weighted dense matrix solver of
                statsmodels
    
            Returns
            -------
    
            
    

.. code:: ipython3

    st_label = 'ST'
    ast_label = 'AST'
    rst_label = 'REV-ST'
    rast_label = 'REV-AST'
    ds100.calibration_double_ended(sections=sections,
                                   st_label=st_label,
                                   ast_label=ast_label,
                                   rst_label=rst_label,
                                   rast_label=rast_label,
                                   method='ols')

.. code:: ipython3

    ds1 = ds100.isel(time=0)  # take only the first timestep
    
    ds1.TMPF.plot(linewidth=1, label='User cali. Forward')  # plot the temperature calibrated by us
    ds1.TMPB.plot(linewidth=1, label='User cali. Backward')  # plot the temperature calibrated by us
    ds1.TMP.plot(linewidth=1, label='Device calibrated')  # plot the temperature calibrated by the device
    plt.legend()




.. parsed-literal::

    <matplotlib.legend.Legend at 0x1a2d1b00>




.. image:: 06Calibrate_double_ols.ipynb_files%5C06Calibrate_double_ols.ipynb_6_1.png


Lets compare our calibrated values with the device calibration. Lets
average the temperature of the forward channel and the backward channel
first.

.. code:: ipython3

    ds1['TMPAVG'] = (ds1.TMPF + ds1.TMPB) / 2
    ds1_diff = ds1.TMP - ds1.TMPAVG
    ds1_diff.plot()




.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x1a84ca20>]




.. image:: 06Calibrate_double_ols.ipynb_files%5C06Calibrate_double_ols.ipynb_8_1.png


The device calibration sections and calibration sections defined by us
differ. The device only allows for 2 sections, one per thermometer. And
most likely the :math:`\gamma` is fixed in the device calibration.

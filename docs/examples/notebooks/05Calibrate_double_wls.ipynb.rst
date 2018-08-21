
5. Calibration of double ended measurement with WLS and confidence intervals
============================================================================

A double ended calibration is performed with weighted least squares.
Over all timesteps simultaneous. :math:`\gamma` and :math:`\alpha`
remain constant, while :math:`C` varies over time. The weights are not
considered equal here. The weights kwadratically decrease with the
signal strength of the measured Stokes and anti-Stokes signals.

The confidence intervals can be calculated as the weights are correctly
defined.

.. code:: ipython3

    import os
    
    from dtscalibration import read_xml_dir
    # import matplotlib.pyplot as plt


.. parsed-literal::

    /Users/bfdestombe/Projects/dts-calibration/python-dts-calibration/.tox/docs/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
      return f(*args, **kwds)


.. code:: ipython3

    try:
        wd = os.path.dirname(os.path.realpath(__file__))
    except:
        wd = os.getcwd()
    
    filepath = os.path.join(wd, '..', '..', 'tests', 'data', 'double_ended2')
    timezone_netcdf = 'UTC',
    timezone_ultima_xml = 'Europe/Amsterdam'
    file_ext = '*.xml'
    
    ds_ = read_xml_dir(filepath,
                      timezone_netcdf=timezone_netcdf,
                      timezone_ultima_xml=timezone_ultima_xml,
                      file_ext=file_ext)
    
    ds = ds_.sel(x=slice(0, 100))  # only calibrate parts of the fiber
    sections = {
        'probe1Temperature': [slice(7.5, 17.), slice(70., 80.)],  # cold bath
        'probe2Temperature': [slice(24., 34.), slice(85., 95.)],  # warm bath
        }
    ds.sections = sections


.. parsed-literal::

    6 files were found, each representing a single timestep
    6 recorded vars were found: LAF, ST, AST, REV-ST, REV-AST, TMP
    Recorded at 1693 points along the cable
    processing file 1 out of 6


.. code:: ipython3

    st_label = 'ST'
    ast_label = 'AST'
    rst_label = 'REV-ST'
    rast_label = 'REV-AST'

First calculate the variance in the measured Stokes and anti-Stokes
signals, in the forward and backward direction.

The Stokes and anti-Stokes signals should follow a smooth decaying
exponential. This function fits a decaying exponential to each reference
section for each time step. The variance of the residuals between the
measured Stokes and anti-Stokes signals and the fitted signals is used
as an estimate of the variance in measured signals.

.. code:: ipython3

    st_var, resid = ds.variance_stokes(st_label=st_label, suppress_info=1)
    ast_var, _ = ds.variance_stokes(st_label=ast_label, suppress_info=1)
    rst_var, _ = ds.variance_stokes(st_label=rst_label, suppress_info=1)
    rast_var, _ = ds.variance_stokes(st_label=rast_label, suppress_info=1)

Similar to the ols procedure, we make a single function call to
calibrate the temperature. If the method is ``wls`` and confidence
intervals are passed to ``conf_ints``, confidence intervals calculated.
As weigths are correctly passed to the least squares procedure, the
covariance matrix can be used. This matrix holds the covariances between
all the parameters. A large parameter set is generated from this matrix,
assuming the parameter space is normally distributed with their mean at
the best estimate of the least squares procedure.

The large parameter set is used to calculate a large set of
temperatures. By using ``percentiles`` or ``quantile`` the 95%
confidence interval of the calibrated temperature between 2.5% and 97.5%
are calculated.

The confidence intervals differ per time step. If you would like to
calculate confidence intervals of all time steps together you have the
option ``ci_avg_time_flag=True``. 'We can say with 95% confidence that
the temperature remained between this line and this line during the
entire measurement period'.

.. code:: ipython3

    ds.calibration_double_ended(sections=sections,
                                st_label=st_label,
                                ast_label=ast_label,
                                rst_label=rst_label,
                                rast_label=rast_label,
                                st_var=st_var,
                                ast_var=ast_var,
                                rst_var=rst_var,
                                rast_var=rast_var,
                                method='wls',
                                # conf_ints=[0.00135, 0.025, 0.15865, 0.5, 0.84135, 0.975, 0.99865],
                                conf_ints=[0.025, 0.5, 0.975],
                                ci_avg_time_flag=0,
                                store_tempvar='_var',
                                conf_ints_size=500,
                                solver='stats',
                                x_alpha_set_zero=0.)

.. code:: ipython3

    # ds1 = ds.isel(time=0)  # take only the first timestep
    # ds1.TMPF.plot(linewidth=0.7)
    # ds1.TMPF_MC.isel(CI=0).plot(linewidth=0.7, label='CI: 2.5%')
    # ds1.TMPF_MC.isel(CI=1).plot(linewidth=0.7, label='CI: 97.5%')
    # plt.legend()
    # plt.show()

The DataArrays ``TMPF_MC`` and ``TMPB_MC`` and the dimension ``CI`` are
added. ``MC`` stands for monte carlo and the ``CI`` dimension holds the
confidence interval 'coordinates'.

.. code:: ipython3

    ds.data_vars




.. parsed-literal::

    Data variables:
        ST                     (x, time) float32 4049.08 4044.32 4046.2 4045.23 ...
        AST                    (x, time) float32 3293.22 3296.04 3280.75 3287.3 ...
        REV-ST                 (x, time) float32 4060.72 4037.16 4029.5 4042.97 ...
        REV-AST                (x, time) float32 3350.43 3333.43 3324.93 3332.45 ...
        TMP                    (x, time) float32 16.6912 16.8743 16.5069 16.5165 ...
        acquisitionTime        (time) float32 2.098 2.075 2.076 2.133 2.085 2.062
        referenceTemperature   (time) float32 21.0536 21.054 21.0497 21.0519 ...
        probe1Temperature      (time) float32 4.36149 4.36025 4.35911 4.36002 ...
        probe2Temperature      (time) float32 18.5792 18.5785 18.5848 18.5814 ...
        referenceProbeVoltage  (time) float32 0.121704 0.121704 0.121703 ...
        probe1Voltage          (time) float32 0.114 0.114 0.114 0.114 0.114 0.114
        probe2Voltage          (time) float32 0.121 0.121 0.121 0.121 0.121 0.121
        userAcquisitionTimeFW  (time) float32 2.0 2.0 2.0 2.0 2.0 2.0
        userAcquisitionTimeBW  (time) float32 2.0 2.0 2.0 2.0 2.0 2.0
        gamma                  float64 482.6
        alphaint               float64 0.00244
        alpha                  (x) float64 2.083e-13 -0.002081 0.004402 0.003867 ...
        c                      (time) float64 1.464 1.464 1.463 1.464 1.464 1.464
        gamma_var              float64 0.04083
        alphaint_var           float64 3.757e-07
        alpha_var              (x) float64 8.328e-33 2.672e-07 2.653e-07 ...
        c_var                  (time) float64 6.009e-07 6.009e-07 6.009e-07 ...
        TMPF                   (x, time) float64 15.76 16.01 15.29 15.56 15.73 ...
        TMPB                   (x, time) float64 17.83 17.86 17.91 17.6 17.58 ...
        TMPF_MC                (CI, x, time) float64 15.03 15.26 14.55 14.83 ...
        TMPB_MC                (CI, x, time) float64 16.69 16.79 16.74 16.38 ...
        TMPF_MC_var            (x, time) float64 0.1529 0.1256 0.1386 0.1451 ...
        TMPB_MC_var            (x, time) float64 0.3988 0.3429 0.4192 0.4297 ...



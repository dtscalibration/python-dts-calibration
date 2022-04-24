5. Calibration of single-ended measurement with OLS
===================================================

Use WLS instead. See
``examples/notebooks/07Calibrate_single_wls.ipynb``.

A double ended calibration is performed with Ordinary Least Squares.
Over all timesteps simultaneous. :math:`\gamma` and :math:`\alpha`
remain constant, while :math:`C` varies over time. The weights are
considered equal here and no variance or confidence interval is
calculated.

Note that the internal reference section can not be used since there is
a connector between the internal and external fiber and therefore the
integrated differential attenuation cannot be considered to be linear
anymore.

.. code:: ipython3

    import os
    
    from dtscalibration import read_silixa_files
    import matplotlib.pyplot as plt
    
    %matplotlib inline

.. code:: ipython3

    filepath = os.path.join('..', '..', 'tests', 'data', 'single_ended')
    
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf='UTC',
        file_ext='*.xml')
    
    ds100 = ds.sel(x=slice(-30, 101))  # only calibrate parts of the fiber, in meters
    sections = {
                'probe1Temperature':    [slice(20, 25.5)],  # warm bath
                'probe2Temperature':    [slice(5.5, 15.5)],  # cold bath
                }
    ds100.sections = sections


.. parsed-literal::

    3 files were found, each representing a single timestep
    4 recorded vars were found: LAF, ST, AST, TMP
    Recorded at 1461 points along the cable
    The measurement is single ended
    Reading the data from disk


.. parsed-literal::

    /home/bart/git/travis_fix/python-dts-calibration/src/dtscalibration/io.py:1843: FutureWarning: Using .astype to convert from timezone-aware dtype to timezone-naive dtype is deprecated and will raise in a future version.  Use obj.tz_localize(None) or obj.tz_convert('UTC').tz_localize(None) instead
      'time', pd.DatetimeIndex(v).tz_convert(timezone_netcdf).astype(


.. code:: ipython3

    print(ds100.calibration_single_ended.__doc__)


.. parsed-literal::

    
            Calibrate the Stokes (`ds.st`) and anti-Stokes (`ds.ast`) data to
            temperature using fiber sections with a known temperature
            (`ds.sections`) for single-ended setups. The calibrated temperature is
            stored under `ds.tmpf` and its variance under `ds.tmpf_var`.
    
            In single-ended setups, Stokes and anti-Stokes intensity is measured
            from a single end of the fiber. The differential attenuation is assumed
            constant along the fiber so that the integrated differential attenuation
            may be written as (Hausner et al, 2011):
    
            .. math::
    
                \int_0^x{\Delta\alpha(x')\,\mathrm{d}x'} \approx \Delta\alpha x
    
            The temperature can now be written from Equation 10 [1]_ as:
    
            .. math::
    
                T(x,t)  \approx \frac{\gamma}{I(x,t) + C(t) + \Delta\alpha x}
    
            where
    
            .. math::
    
                I(x,t) = \ln{\left(\frac{P_+(x,t)}{P_-(x,t)}\right)}
    
    
            .. math::
    
                C(t) = \ln{\left(\frac{\eta_-(t)K_-/\lambda_-^4}{\eta_+(t)K_+/\lambda_+^4}\right)}
    
            where :math:`C` is the lumped effect of the difference in gain at
            :math:`x=0` between Stokes and anti-Stokes intensity measurements and
            the dependence of the scattering intensity on the wavelength. The
            parameters :math:`P_+` and :math:`P_-` are the Stokes and anti-Stokes
            intensity measurements, respectively.
            The parameters :math:`\gamma`, :math:`C(t)`, and :math:`\Delta\alpha`
            must be estimated from calibration to reference sections, as discussed
            in Section 5 [1]_. The parameter :math:`C` must be estimated
            for each time and is constant along the fiber. :math:`T` in the listed
            equations is in Kelvin, but is converted to Celsius after calibration.
    
            Parameters
            ----------
            store_p_cov : str
                Key to store the covariance matrix of the calibrated parameters
            store_p_val : str
                Key to store the values of the calibrated parameters
            p_val : array-like, optional
                Define `p_val`, `p_var`, `p_cov` if you used an external function
                for calibration. Has size 2 + `nt`. First value is :math:`\gamma`,
                second is :math:`\Delta \alpha`, others are :math:`C` for each
                timestep.
            p_var : array-like, optional
                Define `p_val`, `p_var`, `p_cov` if you used an external function
                for calibration. Has size 2 + `nt`. First value is :math:`\gamma`,
                second is :math:`\Delta \alpha`, others are :math:`C` for each
                timestep.
            p_cov : array-like, optional
                The covariances of `p_val`.
                If set to False, no uncertainty in the parameters is propagated
                into the confidence intervals. Similar to the spec sheets of the DTS
                manufacturers. And similar to passing an array filled with zeros.
            sections : Dict[str, List[slice]], optional
                If `None` is supplied, `ds.sections` is used. Define calibration
                sections. Each section requires a reference temperature time series,
                such as the temperature measured by an external temperature sensor.
                They should already be part of the DataStore object. `sections`
                is defined with a dictionary with its keywords of the
                names of the reference temperature time series. Its values are
                lists of slice objects, where each slice object is a fiber stretch
                that has the reference temperature. Afterwards, `sections` is stored
                under `ds.sections`.
            st_var, ast_var : float, callable, array-like, optional
                The variance of the measurement noise of the Stokes signals in the
                forward direction. If `float` the variance of the noise from the
                Stokes detector is described with a single value.
                If `callable` the variance of the noise from the Stokes detector is
                a function of the intensity, as defined in the callable function.
                Or manually define a variance with a DataArray of the shape
                `ds.st.shape`, where the variance can be a function of time and/or
                x. Required if method is wls.
            store_c : str
                Label of where to store C
            store_gamma : str
                Label of where to store gamma
            store_dalpha : str
                Label of where to store dalpha; the spatial derivative of alpha.
            store_alpha : str
                Label of where to store alpha; The integrated differential
                attenuation.
                alpha(x=0) = 0
            store_ta : str
                Label of where to store transient alpha's
            store_tmpf : str
                Label of where to store the calibrated temperature of the forward
                direction
            variance_suffix : str
                String appended for storing the variance. Only used when method
                is wls.
            method : {'ols', 'wls'}
                Use `'ols'` for ordinary least squares and `'wls'` for weighted least
                squares. `'wls'` is the default, and there is currently no reason to
                use `'ols'`.
            solver : {'sparse', 'stats'}
                Either use the homemade weighted sparse solver or the weighted
                dense matrix solver of statsmodels. The sparse solver uses much less
                memory, is faster, and gives the same result as the statsmodels
                solver. The statsmodels solver is mostly used to check the sparse
                solver. `'stats'` is the default.
            matching_sections : List[Tuple[slice, slice, bool]], optional
                Provide a list of tuples. A tuple per matching section. Each tuple
                has three items. The first two items are the slices of the sections
                that are matched. The third item is a boolean and is True if the two
                sections have a reverse direction ("J-configuration").
            transient_att_x, transient_asym_att_x : iterable, optional
                Depreciated. See trans_att
            trans_att : iterable, optional
                Splices can cause jumps in differential attenuation. Normal single
                ended calibration assumes these are not present. An additional loss
                term is added in the 'shadow' of the splice. Each location
                introduces an additional nt parameters to solve for. Requiring
                either an additional calibration section or matching sections.
                If multiple locations are defined, the losses are added.
            fix_gamma : Tuple[float, float], optional
                A tuple containing two floats. The first float is the value of
                gamma, and the second item is the variance of the estimate of gamma.
                Covariances between gamma and other parameters are not accounted
                for.
            fix_dalpha : Tuple[float, float], optional
                A tuple containing two floats. The first float is the value of
                dalpha (:math:`\Delta \alpha` in [1]_), and the second item is the
                variance of the estimate of dalpha.
                Covariances between alpha and other parameters are not accounted
                for.
    
            Returns
            -------
    
            References
            ----------
            .. [1] des Tombe, B., Schilperoort, B., & Bakker, M. (2020). Estimation
                of Temperature and Associated Uncertainty from Fiber-Optic Raman-
                Spectrum Distributed Temperature Sensing. Sensors, 20(8), 2235.
                https://doi.org/10.3390/s20082235
    
            Examples
            --------
            - `Example notebook 7: Calibrate single ended <https://github.com/dtscalibration/python-dts-calibration/blob/master/examples/notebooks/07Calibrate_single_wls.ipynb>`_
    
            


.. code:: ipython3

    ds100.calibration_single_ended(method='ols')

Lets compare our calibrated values with the device calibration

.. code:: ipython3

    ds1 = ds100.isel(time=0)  # take only the first timestep
    
    ds1.tmpf.plot(linewidth=1, figsize=(12, 8), label='User calibrated')  # plot the temperature calibrated by us
    ds1.tmp.plot(linewidth=1, label='Device calibrated')  # plot the temperature calibrated by the device
    plt.title('Temperature at the first time step')
    plt.legend();



.. image:: 05Calibrate_single_ols.ipynb_files/05Calibrate_single_ols.ipynb_7_0.png



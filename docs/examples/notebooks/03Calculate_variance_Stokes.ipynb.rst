
3. Calculate variance of Stokes and anti-Stokes measurements
============================================================

.. code:: ipython3

    import os
    
    from dtscalibration import read_xml_dir
    
    # from matplotlib import pyplot as plt


.. parsed-literal::

    /Users/bfdestombe/PycharmProjects/dts-calibration/python-dts-calibration/.tox/docs/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
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
    
    ds = read_xml_dir(filepath,
                      timezone_netcdf=timezone_netcdf,
                      timezone_ultima_xml=timezone_ultima_xml,
                      file_ext=file_ext)
    sections = {
        'probe1Temperature': [slice(7.5, 17.), slice(70., 80.)],  # cold bath
        'probe2Temperature': [slice(24., 34.), slice(85., 95.)],  # warm bath
        }


.. parsed-literal::

    6 files were found, each representing a single timestep
    6 recorded vars were found: LAF, ST, AST, REV-ST, REV-AST, TMP
    Recorded at 1693 points along the cable
    processing file 1 out of 6


.. code:: ipython3

    print(ds.variance_stokes.__doc__)


.. parsed-literal::

    
            Calculates the variance between the measurements and a best fit exponential at each
            reference section. This fits a two-parameter exponential to the stokes measurements. The
            temperature is constant and there are no splices/sharp bends in each reference section.
            Therefore all signal decrease is due to differential attenuation, which is the same for
            each reference section. The scale of the exponential does differ per reference section.
    
            Assumptions: 1) the temperature is the same along a reference section. 2) no sharp bends
            and splices in the reference sections. 3) Same type of optical cable in each reference
            section.
    
            Parameters
            ----------
            st_label : str
                label of the Stokes, anti-Stokes measurement.
                E.g., ST, AST, REV-ST, REV-AST
            sections : dict, optional
                Define sections. See documentation
    
            Returns
            -------
            I_var : float
                Variance of the residuals between measured and best fit
            resid : array_like
                Residuals between measured and best fit
            


.. code:: ipython3

    I_var, residuals = ds.variance_stokes(st_label='ST', sections=sections, use_statsmodels=False)
    print("The variance of the Stokes signal along the reference sections "
          "is approximately {}".format(I_var))


.. parsed-literal::

     
    LSQR            Least-squares solution of  Ax = b
    The matrix A has     1854 rows  and       25 cols
    damp = 0.00000000000000e+00   calc_var =        0
    atol = 1.00e-08                 conlim = 1.00e+08
    btol = 1.00e-08               iter_lim =       50
     
       Itn      x[0]       r1norm     r2norm   Compatible    LS      Norm A   Cond A
         0  0.00000e+00   5.088e+02  5.088e+02    1.0e+00  2.4e+01
         1  2.96930e-02   2.687e+02  2.687e+02    1.3e-02  2.0e-02   1.5e+04  1.0e+00
         2  9.86757e-03   1.350e+02  1.350e+02    6.4e-03  2.9e-02   1.5e+04  4.4e+01
         3 -1.01526e-03   5.705e+00  5.705e+00    2.7e-04  2.2e-02   1.5e+04  6.1e+01
         4 -1.02898e-03   4.527e+00  4.527e+00    2.1e-04  1.1e-04   1.5e+04  6.7e+01
         5 -1.02851e-03   4.527e+00  4.527e+00    2.1e-04  1.5e-03   1.5e+04  7.0e+01
         6 -1.02898e-03   4.527e+00  4.527e+00    2.1e-04  1.7e-06   2.1e+04  1.0e+02
         7 -1.02898e-03   4.527e+00  4.527e+00    2.1e-04  6.3e-09   2.1e+04  1.1e+02
     
    LSQR finished
    The least-squares solution is good enough, given atol     
     
    istop =       2   r1norm = 4.5e+00   anorm = 2.1e+04   arnorm = 5.9e-04
    itn   =       7   r2norm = 4.5e+00   acond = 1.1e+02   xnorm  = 9.8e-01
     
    The variance of the Stokes signal along the reference sections is approximately 40.161692081870584


.. code:: ipython3

    # plt.hist(residuals, bins=50, density=True);

.. code:: ipython3

    # plt.plot(residuals)  # not precisely randoms


3. Calculate variance of Stokes and anti-Stokes measurements
============================================================

.. code:: ipython3

    import os
    
    from dtscalibration import read_xml_dir
    from matplotlib import pyplot as plt

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
    Dask: Setting up handle for delayed readout. 1 out of 6
    Dask: Setting up handle for delayed readout. 6 out of 6
    Directly reading time and extra info from xml files. 1 out of 6
    Directly reading time and extra info from xml files. 6 out of 6


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
    
            Idea from discussion at page 127 in Richter, P. H. (1995). Estimating errors in
            least-squares fitting. For weights used error propagation:
            w^2 = 1/sigma(lny)^2 = y^2/sigma(y)^2 = y^2
    
            Parameters
            ----------
            use_statsmodels
            suppress_info
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

    I_var, residuals = ds.variance_stokes(st_label='ST', 
                                          sections=sections, 
                                          use_statsmodels=False, 
                                          debug_high_stokes_variance=False)
    print("The variance of the Stokes signal along the reference sections "
          "is approximately {}".format(I_var))


.. parsed-literal::

    The variance of the Stokes signal along the reference sections is approximately 40.15998656786007


.. code:: ipython3

    residuals.plot(figsize=(12, 8))




.. parsed-literal::

    <matplotlib.collections.QuadMesh at 0x11bac6630>



.. code:: ipython3

    residuals.plot.hist(bins=50);

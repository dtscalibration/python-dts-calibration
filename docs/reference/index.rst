Reference
=========

Load the data
-------------

See example notebooks 01, A2, A3, and A4. Import directly from `dtscalibration`.

.. currentmodule:: dtscalibration
.. autosummary::
    :toctree: ./generated
    :nosignatures:

    read_apsensing_files
    read_sensornet_files
    read_sensortran_files
    read_silixa_files

Compute the variance in the Stokes measurements
-----------------------------------------------

See example notebooks 04. Import from `dtscalibration.variance_stokes`.

.. currentmodule:: dtscalibration.variance_stokes
.. autosummary::
    :toctree: ./generated
    :nosignatures:

    variance_stokes_constant
    variance_stokes_linear
    variance_stokes_exponential


The DTS Accessor
----------------

These methods are available as an `xarray.Dataset` accessor. Add 
`# The following line introduces the .dts accessor for xarray datasets
import dtscalibration  # noqa: E401 ` to your import 
statements. See example natebooks 07, 08, and 17.

.. currentmodule:: xarray.Dataset
.. autosummary::
    :toctree: generated/
    :template: autosummary/accessor_method.rst
    :nosignatures:

    dts.sections
    dts.calibrate_single_ended
    dts.calibrate_double_ended
    dts.monte_carlo_single_ended
    dts.monte_carlo_double_ended
    dts.average_monte_carlo_single_ended
    dts.average_monte_carlo_double_ended
    dts.get_default_encoding
    dts.get_timeseries_keys
    dts.matching_sections
    dts.ufunc_per_section

Plot the results
----------------

Import from `dtscalibration.plot`.

.. currentmodule:: dtscalibration.plot
.. autosummary::
    :toctree: ./generated
    :nosignatures:

    plot_residuals_reference_sections
    plot_residuals_reference_sections_single
    plot_accuracy
    plot_sigma_report
    plot_location_residuals_double_ended
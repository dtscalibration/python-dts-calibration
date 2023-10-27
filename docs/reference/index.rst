Reference
=========

Load the data
-------------

See example notebooks 01, A2, A3, and A4.

.. automodule:: dtscalibration.io
   :members: dtscalibration.read_apsensing_files
   :nosignatures:

Compute the variance in the Stokes measurements
-----------------------------------------------

See example notebooks 04 and have a look at the docstring of the dtscalibration.variance_stokes funcitons.

.. automodule:: dtscalibration.variance_stokes
   :members:
   :nosignatures:


The DTS Accessor
----------------

See example natebooks 07, 08, and 17.

.. currentmodule:: xarray
.. autosummary::
    :toctree: generated/
    :template: autosummary/accessor_method.rst
    :nosignatures:

    Dataset.dts.sections
    Dataset.dts.calibrate_single_ended
    Dataset.dts.calibrate_double_ended
    Dataset.dts.monte_carlo_single_ended
    Dataset.dts.monte_carlo_double_ended
    Dataset.dts.average_monte_carlo_single_ended
    Dataset.dts.average_monte_carlo_double_ended
    Dataset.dts.get_default_encoding
    Dataset.dts.get_timeseries_keys
    Dataset.dts.matching_sections
    Dataset.dts.ufunc_per_section

Plot the results
----------------

.. automodule:: dtscalibration.plot
   :members:
   :nosignatures:
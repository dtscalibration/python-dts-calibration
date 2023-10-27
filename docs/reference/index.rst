Reference
=========

Load the data
-------------

.. automodule:: dtscalibration.io
   :members:
   :nosignatures:

Compute the variance in the Stokes measurements
-----------------------------------------------

.. automodule:: dtscalibration.variance_stokes
   :members:
   :nosignatures:

The DTS Accessor
----------------

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
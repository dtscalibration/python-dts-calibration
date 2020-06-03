DataStore
=========

.. currentmodule:: dtscalibration

.. autoclass:: DataStore
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~DataStore.channel_configuration
      ~DataStore.chbw
      ~DataStore.chfw
      ~DataStore.is_double_ended
      ~DataStore.sections
      ~DataStore.timeseries_keys

   .. rubric:: Methods Summary

   .. autosummary::

      ~DataStore.average_double_ended
      ~DataStore.average_single_ended
      ~DataStore.calibration_double_ended
      ~DataStore.calibration_single_ended
      ~DataStore.check_deprecated_kwargs
      ~DataStore.conf_int_double_ended
      ~DataStore.conf_int_single_ended
      ~DataStore.get_default_encoding
      ~DataStore.get_section_indices
      ~DataStore.get_time_dim
      ~DataStore.i_var
      ~DataStore.in_confidence_interval
      ~DataStore.inverse_variance_weighted_mean
      ~DataStore.inverse_variance_weighted_mean_array
      ~DataStore.rename_labels
      ~DataStore.resample_datastore
      ~DataStore.temperature_residuals
      ~DataStore.to_mf_netcdf
      ~DataStore.to_netcdf
      ~DataStore.ufunc_per_section
      ~DataStore.variance_stokes
      ~DataStore.variance_stokes_constant
      ~DataStore.variance_stokes_exponential
      ~DataStore.variance_stokes_linear

   .. rubric:: Attributes Documentation

   .. autoattribute:: channel_configuration
   .. autoattribute:: chbw
   .. autoattribute:: chfw
   .. autoattribute:: is_double_ended
   .. autoattribute:: sections
   .. autoattribute:: timeseries_keys

   .. rubric:: Methods Documentation

   .. automethod:: average_double_ended
   .. automethod:: average_single_ended
   .. automethod:: calibration_double_ended
   .. automethod:: calibration_single_ended
   .. automethod:: check_deprecated_kwargs
   .. automethod:: conf_int_double_ended
   .. automethod:: conf_int_single_ended
   .. automethod:: get_default_encoding
   .. automethod:: get_section_indices
   .. automethod:: get_time_dim
   .. automethod:: i_var
   .. automethod:: in_confidence_interval
   .. automethod:: inverse_variance_weighted_mean
   .. automethod:: inverse_variance_weighted_mean_array
   .. automethod:: rename_labels
   .. automethod:: resample_datastore
   .. automethod:: temperature_residuals
   .. automethod:: to_mf_netcdf
   .. automethod:: to_netcdf
   .. automethod:: ufunc_per_section
   .. automethod:: variance_stokes
   .. automethod:: variance_stokes_constant
   .. automethod:: variance_stokes_exponential
   .. automethod:: variance_stokes_linear

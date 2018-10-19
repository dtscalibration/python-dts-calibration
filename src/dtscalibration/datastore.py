# coding=utf-8
import glob
import inspect
import os

import dask.array as da
import numpy as np
import scipy.sparse as sp
import scipy.stats as sst
import xarray as xr
import yaml
from scipy.sparse import linalg as ln

from .calibrate_utils import calibration_double_ended_ols
from .calibrate_utils import calibration_double_ended_wls
from .calibrate_utils import calibration_single_ended_ols
from .calibrate_utils import calibration_single_ended_wls
from .datastore_utils import check_dims
from .datastore_utils import check_timestep_allclose
from .datastore_utils import read_silixa_files_routine


class DataStore(xr.Dataset):
    """The data class that stores the measurements, contains calibration methods to relate Stokes
    and anti-Stokes to temperature. The user should never initiate this class directly,
    but use read_xml_dir or open_datastore functions instead.

        Parameters
        ----------
        data_vars : dict-like, optional
            A mapping from variable names to :py:class:`~xarray.DataArray`
            objects, :py:class:`~xarray.Variable` objects or tuples of the
            form ``(dims, data[, attrs])`` which can be used as arguments to
            create a new ``Variable``. Each dimension must have the same length
            in all variables in which it appears.
        coords : dict-like, optional
            Another mapping in the same form as the `variables` argument,
            except the each item is saved on the datastore as a "coordinate".
            These variables have an associated meaning: they describe
            constant/fixed/independent quantities, unlike the
            varying/measured/dependent quantities that belong in `variables`.
            Coordinates values may be given by 1-dimensional arrays or scalars,
            in which case `dims` do not need to be supplied: 1D arrays will be
            assumed to give index values along the dimension with the same
            name.
        attrs : dict-like, optional
            Global attributes to save on this datastore.
        sections : dict, optional
            Sections for calibration. The dictionary should contain key-var couples
            in which the key is the name of the calibration temp time series. And
            the var is a list of slice objects as 'slice(start, stop)'; start and
            stop in meter (float).
        compat : {'broadcast_equals', 'equals', 'identical'}, optional
            String indicating how to compare variables of the same name for
            potential conflicts when initializing this datastore:

            - 'broadcast_equals': all values must be equal when variables are
              broadcast against each other to ensure common dimensions.
            - 'equals': all values and dimensions must be the same.
            - 'identical': all values, dimensions and attributes must be the
              same.

        See Also
        --------
        dtscalibration.read_xml_dir : Load measurements stored in XML-files
        dtscalibration.open_datastore : Load (calibrated) measurements from netCDF-like file
        """

    def __init__(self, *args, **kwargs):
        super(DataStore, self).__init__(*args, **kwargs)

        if '_sections' not in self.attrs:
            self.attrs['_sections'] = yaml.dump(None)

        if 'sections' in kwargs:
            self.sections = kwargs['sections']

    # noinspection PyIncorrectDocstring
    @property
    def sections(self):
        """
        Define calibration sections. Each section requires a reference temperature time series,
        such as the temperature measured by an external temperature sensor. They should already be
        part of the DataStore object.

        Please look at the example notebook on `sections` if you encounter difficulties.

        Parameters
        ----------
        sections : dict
            Sections are defined in a dictionary with its keywords of the names of the reference
            temperature time series. Its values are lists of slice objects, where each slice object
            is a stretch.
        Returns
        -------

        """
        assert hasattr(self, '_sections'), 'first set the sections'
        return yaml.load(self.attrs['_sections'])

    @sections.setter
    def sections(self, sections):
        if sections:
            assert isinstance(sections, dict)

            for key in sections:
                assert key in self.data_vars, 'The keys of the sections-dictionary should refer ' \
                                              'to a valid timeserie already stored in ds.data_vars'
            check_dims(self, sections, ('time',))

            for k, v in sections.items():
                assert isinstance(v, (list, tuple)), 'The values of the sections-dictionary ' \
                                                     'should be lists of slice objects.'

                for vi in v:
                    assert isinstance(vi, slice), 'The values of the sections-dictionary should ' \
                                                  'be lists of slice objects.'

        self.attrs['_sections'] = yaml.dump(sections)
        pass

    @sections.deleter
    def sections(self):
        self.sections = None
        pass

    @property
    def is_double_ended(self):
        return bool(int(self.attrs['customData:isDoubleEnded']))

    @property
    def chfw(self):
        return int(self.attrs['customData:forwardMeasurementChannel']) - 1  # zero-based

    @property
    def chbw(self):
        if self.is_double_ended:
            return int(self.attrs['customData:reverseMeasurementChannel']) - 1  # zero-based
        else:
            return None

    @property
    def channel_configuration(self):
        d = {
            'chfw': {
                'st_label':              'ST',
                'ast_label':             'AST',
                'acquisitiontime_label': 'userAcquisitionTimeFW',
                'time_start_label':      'timeFWstart',
                'time_label':            'timeFW',
                'time_end_label':        'timeFWend',
                },
            'chbw': {
                'st_label':              'REV-ST',
                'ast_label':             'REV-AST',
                'acquisitiontime_label': 'userAcquisitionTimeBW',
                'time_start_label':      'timeBWstart',
                'time_label':            'timeBW',
                'time_end_label':        'timeBWend',
                }
            }
        return d

    def to_netcdf(self, path=None, mode='w', format=None, group=None,
                  engine=None, encoding=None, unlimited_dims=None,
                  compute=True):
        """Write datastore contents to a netCDF file.
        Parameters
        ----------
        path : str, Path or file-like object, optional
            Path to which to save this dataset. File-like objects are only
            supported by the scipy engine. If no path is provided, this
            function returns the resulting netCDF file as bytes; in this case,
            we need to use scipy, which does not support netCDF version 4 (the
            default format becomes NETCDF3_64BIT).
        mode : {'w', 'a'}, optional
            Write ('w') or append ('a') mode. If mode='w', any existing file at
            this location will be overwritten. If mode='a', existing variables
            will be overwritten.
        format : {'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT','NETCDF3_CLASSIC'}, optional
            File format for the resulting netCDF file:
            * NETCDF4: Data is stored in an HDF5 file, using netCDF4 API
              features.
            * NETCDF4_CLASSIC: Data is stored in an HDF5 file, using only
              netCDF 3 compatible API features.
            * NETCDF3_64BIT: 64-bit offset version of the netCDF 3 file format,
              which fully supports 2+ GB files, but is only compatible with
              clients linked against netCDF version 3.6.0 or later.
            * NETCDF3_CLASSIC: The classic netCDF 3 file format. It does not
              handle 2+ GB files very well.
            All formats are supported by the netCDF4-python library.
            scipy.io.netcdf only supports the last two formats.
            The default format is NETCDF4 if you are saving a file to disk and
            have the netCDF4-python library available. Otherwise, xarray falls
            back to using scipy to write netCDF files and defaults to the
            NETCDF3_64BIT format (scipy does not support netCDF4).
        group : str, optional
            Path to the netCDF4 group in the given file to open (only works for
            format='NETCDF4'). The group(s) will be created if necessary.
        engine : {'netcdf4', 'scipy', 'h5netcdf'}, optional
            Engine to use when writing netCDF files. If not provided, the
            default engine is chosen based on available dependencies, with a
            preference for 'netcdf4' if writing to a file on disk.
        encoding : dict, optional
            defaults to reasonable compression. Use encoding={} to disable encoding.
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{'my_variable': {'dtype': 'int16', 'scale_factor': 0.1,
                               'zlib': True}, ...}``
            The `h5netcdf` engine supports both the NetCDF4-style compression
            encoding parameters ``{'zlib': True, 'complevel': 9}`` and the h5py
            ones ``{'compression': 'gzip', 'compression_opts': 9}``.
            This allows using any compression plugin installed in the HDF5
            library, e.g. LZF.
        unlimited_dims : sequence of str, optional
            Dimension(s) that should be serialized as unlimited dimensions.
            By default, no dimensions are treated as unlimited dimensions.
            Note that unlimited_dims may also be set via
            ``dataset.encoding['unlimited_dims']``.
        compute: boolean
            If true compute immediately, otherwise return a
            ``dask.delayed.Delayed`` object that can be computed later.
        """
        if encoding is None:
            encoding = self.get_default_encoding()

        if engine is None:
            engine = 'netcdf4'

        # Fix Bart Schilperoort: netCDF doesn't like None's
        for attribute, value in self.attrs.items():
            if value is None:
                self.attrs[attribute] = ''

        return super(DataStore, self).to_netcdf(
            path, mode, format=format, group=group,
            engine=engine, encoding=encoding,
            unlimited_dims=unlimited_dims,
            compute=compute)

    def get_default_encoding(self):
        # TODO: set scale parameter
        compdata = dict(zlib=True, complevel=6, shuffle=False)  # , least_significant_digit=None
        compcoords = dict(zlib=True, complevel=4)

        encoding = {var: compdata for var in self.data_vars}
        encoding.update({var: compcoords for var in self.coords})
        return encoding

    def variance_stokes(self, st_label, sections=None, use_statsmodels=False,
                        suppress_info=True, reshape_residuals=True):
        """
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
        """
        if sections:
            self.sections = sections
        else:
            assert self.sections, 'sections are not defined'

        check_dims(self, [st_label], correct_dims=('x', 'time'))
        check_timestep_allclose(self, eps=0.01)

        nt = self['time'].size

        len_stretch_list = []
        y_list = []  # intensities of stokes
        x_list = []  # length rel to start of section. for alpha

        for k, stretches in self.sections.items():
            for stretch in stretches:
                y_list.append(self[st_label].sel(x=stretch).data.T.reshape(-1))
                _x = self.x.sel(x=stretch).data.copy()
                _x -= _x[0]
                x_list.append(da.tile(_x, nt))
                len_stretch_list.append(_x.size)

        n_sections = len(len_stretch_list)
        n_locs = sum(len_stretch_list)

        x = np.concatenate(x_list)  # coordinates are already in memory
        y = np.asarray(da.concatenate(y_list))

        data1 = x
        data2 = np.ones(sum(len_stretch_list) * nt)
        data = np.concatenate([data1, data2])

        # alpha is the same for all -> one column
        coords1row = np.arange(nt * n_locs)
        coords1col = np.zeros_like(coords1row)

        # second calibration parameter is different per section and per timestep
        coords2row = np.arange(nt * n_locs)
        coords2col = np.hstack([
            np.repeat(np.arange(i * nt + 1, (i + 1) * nt + 1), in_locs)
            for i, in_locs in enumerate(len_stretch_list)
            ])  # C for
        coords = (np.concatenate([coords1row, coords2row]),
                  np.concatenate([coords1col, coords2col]))

        lny = np.log(y)
        w = y.copy()  # 1/std.

        ddof = 1 + nt * n_sections  # see numpy documentation on ddof

        if use_statsmodels:
            # returns the same answer with statsmodel
            import statsmodels.api as sm

            X = sp.coo_matrix((data, coords),
                              shape=(nt * n_locs, 1 + nt * n_sections),
                              dtype=float,
                              copy=False)

            mod_wls = sm.WLS(lny, X.todense(), weights=w ** 2)
            res_wls = mod_wls.fit()
            # print(res_wls.summary())
            a = res_wls.params

        else:
            wdata = data * np.hstack((w, w))
            wX = sp.coo_matrix((wdata, coords),
                               shape=(nt * n_locs, 1 + nt * n_sections),
                               dtype=float,
                               copy=False)

            wlny = (lny * w)

            p0_est = np.asarray([0.] + nt * n_sections * [8])
            # noinspection PyTypeChecker
            a = ln.lsqr(wX, wlny,
                        x0=p0_est,
                        show=not suppress_info,
                        calc_var=False)[0]

        C_expand_to_sec = np.hstack([
            np.repeat(a[i * nt + 1:(i + 1) * nt + 1], leni)
            for i, leni in enumerate(len_stretch_list)
            ])
        I_est = np.exp(C_expand_to_sec) * np.exp(x * a[0])
        resid = I_est - y
        var_I = resid.std(ddof=ddof) ** 2

        if not reshape_residuals:
            return var_I, resid
        else:
            # restructure the residuals, such that they can be plotted and added to ds
            resid_res = []
            for leni, lenis, lenie in zip(
                len_stretch_list,
                nt * np.cumsum([0] + len_stretch_list[:-1]),
                    nt * np.cumsum(len_stretch_list)):

                resid_res.append(resid[lenis:lenie].reshape(leni, nt))

            _resid = np.concatenate(resid_res)
            _resid_x = self.ufunc_per_section(label='x', calc_per='all')
            isort = np.argsort(_resid_x)
            resid_x = _resid_x[isort]
            resid = _resid[isort, :]

            ix_resid = np.array([np.argmin(np.abs(ai - self.x.data)) for ai in resid_x])

            resid_sorted = np.full(shape=self[st_label].shape, fill_value=np.nan)
            resid_sorted[ix_resid, :] = resid
            resid_da = xr.DataArray(data=resid_sorted,
                                    dims=('x', 'time'),
                                    coords={
                                        'x':    self.x,
                                        'time': self.time})

            return var_I, resid_da

    def inverse_variance_weighted_mean(self,
                                       tmp1='TMPF',
                                       tmp2='TMPB',
                                       tmp1_var='TMPF_MC_var',
                                       tmp2_var='TMPB_MC_var',
                                       tmpw_store='TMPW',
                                       tmpw_var_store='TMPW_var'):
        """
        Average two temperature datasets with the inverse of the variance as weights. The two
        temperature datasets `tmp1` and `tmp2` with their variances `tmp1_var` and `tmp2_var`,
        respectively. Are averaged and stored in the DataStore.

        Parameters
        ----------
        tmp1 : str
            The label of the first temperature dataset that is averaged
        tmp2 : str
            The label of the second temperature dataset that is averaged
        tmp1_var : str
            The variance of tmp1
        tmp2_var : str
            The variance of tmp2
        tmpw_store : str
            The label of the averaged temperature dataset
        tmpw_var_store : str
            The label of the variance of the averaged temperature dataset

        Returns
        -------

        """

        self[tmpw_var_store] = 1 / (1 / self[tmp1_var] +
                                    1 / self[tmp2_var])

        self[tmpw_store] = (self[tmp1] / self[tmp1_var] +
                            self[tmp2] / self[tmp2_var]) * self[tmpw_var_store]

        pass

    def inverse_variance_weighted_mean_array(self,
                                             tmp_label='TMPF',
                                             tmp_var_label='TMPF_MC_var',
                                             tmpw_store='TMPW',
                                             tmpw_var_store='TMPW_var',
                                             dim='time'):
        """
        Calculates the weighted average across a dimension.

        Parameters
        ----------

        Returns
        -------

        See Also
        --------
        - https://en.wikipedia.org/wiki/Inverse-variance_weighting

        """
        self[tmpw_var_store] = 1 / (1 / self[
            tmp_var_label]).sum(dim=dim)

        self[tmpw_store] = (self[tmp_label] / self[tmp_var_label]).sum(dim=dim) / (1 / self[
            tmp_var_label]).sum(dim=dim)

        pass

    def calibration_single_ended(self,
                                 sections=None,
                                 st_label='ST',
                                 ast_label='AST',
                                 st_var=None,
                                 ast_var=None,
                                 store_c='c',
                                 store_gamma='gamma',
                                 store_dalpha='dalpha',
                                 store_alpha='alpha',
                                 store_tmpf='TMPF',
                                 store_p_cov='p_cov',
                                 store_p_sol='p_val',
                                 variance_suffix='_var',
                                 method='ols',
                                 store_tempvar=None,
                                 conf_ints=None,
                                 conf_ints_size=100,
                                 ci_avg_time_flag=False,
                                 solver='sparse',
                                 da_random_state=None,
                                 ):
        """

        Parameters
        ----------
        sections : dict, optional
        st_label : str
            Label of the forward stokes measurement
        ast_label : str
            Label of the anti-Stoke measurement
        st_var : float, optional
            The variance of the measurement noise of the Stokes signals in the forward
            direction Required if method is wls.
        ast_var : float, optional
            The variance of the measurement noise of the anti-Stokes signals in the forward
            direction. Required if method is wls.
        store_c : str
            Label of where to store C
        store_gamma : str
            Label of where to store gamma
        store_dalpha : str
            Label of where to store dalpha; the spatial derivative  of alpha.
        store_alpha : str
            Label of where to store alpha; The integrated differential attenuation.
            alpha(x=0) = 0
        store_tmpf : str
            Label of where to store the calibrated temperature of the forward direction
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

        """

        if sections:
            self.sections = sections
        else:
            assert self.sections, 'sections are not defined'

        check_dims(self,
                   [st_label, ast_label],
                   correct_dims=('x', 'time'))

        if method == 'ols':
            nt, z, p0_ = calibration_single_ended_ols(
                self, st_label, ast_label)

            p0 = p0_[0]
            gamma = p0[0]
            dalpha = p0[1]
            c = p0[2:nt + 2]

            # Can not estimate parameter variance with ols
            gammavar = None
            dalphavar = None
            cvar = None

        elif method == 'wls':
            for vari in [st_var, ast_var]:
                assert isinstance(vari, float)

            nt, z, p_sol, p_var, p_cov = calibration_single_ended_wls(
                self, st_label, ast_label,
                st_var, ast_var, solver=solver)

            gamma = p_sol[0]
            dalpha = p_sol[1]
            c = p_sol[2:nt + 2]

            # Estimate of the standard error - sqrt(diag of the COV matrix) - is not squared
            gammavar = p_var[0]
            dalphavar = p_var[1]
            cvar = p_var[2:nt + 2]

        else:
            raise ValueError('Choose a valid method')

        # store calibration parameters in DataStore
        self[store_gamma] = (tuple(), gamma)
        self[store_dalpha] = (tuple(), dalpha)
        self[store_alpha] = (('x',), dalpha * self.x.data)
        self[store_c] = (('time',), c)

        # store variances in DataStore
        if method == 'wls':
            self[store_gamma + variance_suffix] = (tuple(), gammavar)
            self[store_dalpha + variance_suffix] = (tuple(), dalphavar)
            self[store_c + variance_suffix] = (('time',), cvar)

        # deal with FW
        if store_tmpf:
            tempF_data = gamma / \
                         (np.log(self[st_label].data / self[ast_label].data)
                          + c + self.x.data[:, None] * dalpha) - 273.15
            self[store_tmpf] = (('x', 'time'), tempF_data)

        if store_p_sol and method == 'wls':
            self[store_p_sol] = (('params1',), p_sol)
            _p_sol = store_p_sol
        elif method == 'wls':
            _p_sol = p_sol
        else:
            _p_sol = None

        if store_p_cov and method == 'wls':
            self[store_p_cov] = (('params1', 'params2'), p_cov)
            _p_cov = store_p_cov
        elif method == 'wls':
            _p_cov = p_cov
        else:
            _p_cov = None

        if conf_ints:
            assert method == 'wls'
            self.conf_int_single_ended(
                p_sol=_p_sol,
                p_cov=_p_cov,
                st_label=st_label,
                ast_label=ast_label,
                st_var=st_var,
                ast_var=ast_var,
                store_tmpf=store_tmpf,
                store_tempvar=store_tempvar,
                conf_ints=conf_ints,
                conf_ints_size=conf_ints_size,
                ci_avg_time_flag=ci_avg_time_flag,
                da_random_state=da_random_state)
        pass

    def calibration_double_ended(self,
                                 sections=None,
                                 st_label='ST',
                                 ast_label='AST',
                                 rst_label='REV-ST',
                                 rast_label='REV-AST',
                                 st_var=None,
                                 ast_var=None,
                                 rst_var=None,
                                 rast_var=None,
                                 store_c='c',
                                 store_gamma='gamma',
                                 store_alphaint='alphaint',
                                 store_alpha='alpha',
                                 store_tmpf='TMPF',
                                 store_tmpb='TMPB',
                                 store_p_cov='p_cov',
                                 store_p_sol='p_val',
                                 variance_suffix='_var',
                                 method='ols',
                                 store_tempvar=None,
                                 conf_ints=None,
                                 conf_ints_size=100,
                                 ci_avg_time_flag=False,
                                 solver='sparse',
                                 da_random_state=None,
                                 dtype32=False):
        """

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

        """

        if sections:
            self.sections = sections
        else:
            assert self.sections, 'sections are not defined'

        check_dims(self,
                   [st_label, ast_label, rst_label, rast_label],
                   correct_dims=('x', 'time'))

        if method == 'ols':
            nt, z, p0_ = calibration_double_ended_ols(
                self, st_label, ast_label, rst_label, rast_label)

            p0 = p0_[0]
            gamma = p0[0]
            alphaint = p0[1]
            c = p0[2:nt + 2]
            alpha = p0[nt + 2:]

            # Can not estimate parameter variance with ols
            gammavar = None
            alphaintvar = None
            cvar = None
            alphavar = None

        elif method == 'wls':
            for vari in [st_var, ast_var, rst_var, rast_var]:
                assert isinstance(vari, float)

            nt, z, p_sol, p_var, p_cov = calibration_double_ended_wls(
                self, st_label, ast_label, rst_label, rast_label,
                st_var, ast_var, rst_var, rast_var, solver=solver, dtype32=dtype32)

            gamma = p_sol[0]
            alphaint = p_sol[1]
            c = p_sol[2:nt + 2]
            alpha = p_sol[nt + 2:]

            # Estimate of the standard error - sqrt(diag of the COV matrix) - is not squared
            gammavar = p_var[0]
            alphaintvar = p_var[1]
            cvar = p_var[2:nt + 2]
            alphavar = p_var[nt + 2:]

        else:
            raise ValueError('Choose a valid method')

        # store calibration parameters in DataStore
        self[store_gamma] = (tuple(), gamma)
        self[store_alphaint] = (tuple(), alphaint)
        self[store_alpha] = (('x',), alpha)
        self[store_c] = (('time',), c)

        # store variances in DataStore
        if method == 'wls':
            self[store_gamma + variance_suffix] = (tuple(), gammavar)
            self[store_alphaint + variance_suffix] = (tuple(), alphaintvar)
            self[store_alpha + variance_suffix] = (('x',), alphavar)
            self[store_c + variance_suffix] = (('time',), cvar)

        # deal with FW
        if store_tmpf:
            tempF_data = gamma / \
                         (np.log(self[st_label].data / self[ast_label].data)
                          + c + alpha[:, None]) - 273.15
            self[store_tmpf] = (('x', 'time'), tempF_data)

        # deal with BW
        if store_tmpb:
            tempB_data = gamma / \
                         (np.log(self[rst_label].data / self[rast_label].data)
                          + c - alpha[:, None] + alphaint) - 273.15
            self[store_tmpb] = (('x', 'time'), tempB_data)

        if store_p_sol and method == 'wls':
            self[store_p_sol] = (('params1',), p_sol)
            _p_sol = store_p_sol
        elif method == 'wls':
            _p_sol = p_sol
        else:
            _p_sol = None

        if store_p_cov and method == 'wls':
            self[store_p_cov] = (('params1', 'params2'), p_cov)
            _p_cov = store_p_cov
        elif method == 'wls':
            _p_cov = p_cov
        else:
            _p_cov = None

        if conf_ints:
            assert method == 'wls'
            self.conf_int_double_ended(
                p_sol=_p_sol,
                p_cov=_p_cov,
                st_label=st_label,
                ast_label=ast_label,
                rst_label=rst_label,
                rast_label=rast_label,
                st_var=st_var,
                ast_var=ast_var,
                rst_var=rst_var,
                rast_var=rast_var,
                # store_c=store_c,
                # store_gamma=store_gamma,
                # store_alphaint=store_alphaint,
                # store_alpha=store_alpha,
                store_tmpf=store_tmpf,
                store_tmpb=store_tmpb,
                store_tempvar=store_tempvar,
                conf_ints=conf_ints,
                conf_ints_size=conf_ints_size,
                ci_avg_time_flag=ci_avg_time_flag,
                da_random_state=da_random_state)

        pass

    def conf_int_single_ended(self,
                              p_sol,
                              p_cov,
                              st_label='ST',
                              ast_label='AST',
                              st_var=None,
                              ast_var=None,
                              store_tmpf='TMPF',
                              store_tempvar='_var',
                              conf_ints=None,
                              conf_ints_size=100,
                              ci_avg_time_flag=False,
                              da_random_state=None
                              ):
        assert conf_ints

        if da_random_state:
            state = da_random_state
        else:
            state = da.random.RandomState()

        no, nt = self[st_label].data.shape
        npar = nt + 2  # number of parameters

        assert isinstance(p_sol, (str, np.ndarray, np.generic))
        if isinstance(p_sol, str):
            p_sol = self[p_sol].data
        assert p_sol.shape == (npar,)

        assert isinstance(p_cov, (str, np.ndarray, np.generic))
        if isinstance(p_cov, str):
            p_cov = self[p_cov].data
        assert p_cov.shape == (npar, npar)

        p_mc = sst.multivariate_normal.rvs(mean=p_sol,
                                           cov=p_cov,
                                           size=conf_ints_size)
        self.coords['MC'] = range(conf_ints_size)
        self.coords['CI'] = conf_ints

        gamma = p_mc[:, 0]
        dalpha = p_mc[:, 1]
        c = p_mc[:, 2:nt + 2]
        self['gamma_MC'] = (('MC',), gamma)
        self['dalpha_MC'] = (('MC',), dalpha)
        self['c_MC'] = (('MC', 'time',), c)

        rshape = (self.MC.size, self.x.size, self.time.size)
        r2shape = {
            0: -1,
            1: 'auto',
            2: -1}

        self['r_st'] = (('MC', 'x', 'time'), state.normal(
            loc=self[st_label].data,
            scale=st_var ** 0.5,
            size=rshape,
            chunks=r2shape))
        self['r_ast'] = (('MC', 'x', 'time'), state.normal(
            loc=self[ast_label].data,
            scale=ast_var ** 0.5,
            size=rshape,
            chunks=r2shape))

        self[store_tmpf + '_MC'] = self['gamma_MC'] / (xr.ufuncs.log(
            self['r_st'] / self['r_ast']) + self['c_MC'] + self[
                                                           'dalpha_MC'] * self.x) - 273.15

        del p_mc
        drop_var = ['gamma_MC',
                    'dalpha_MC',
                    'c_MC',
                    'MC',
                    'r_st',
                    'r_ast']
        for k in drop_var:
            del self[k]

        if ci_avg_time_flag:
            avg_dims = ['MC', 'time']
        else:
            avg_dims = ['MC']

        avg_axis = self[store_tmpf + '_MC'].get_axis_num(avg_dims)

        if store_tempvar:
            self[store_tmpf + '_MC' + store_tempvar] = (self[store_tmpf + '_MC'] - self[
                store_tmpf]).std(dim=avg_dims) ** 2

        if ci_avg_time_flag:
            new_chunks = ((len(conf_ints),),) + self[store_tmpf + '_MC'].chunks[1]
        else:
            new_chunks = ((len(conf_ints),),) + self[store_tmpf + '_MC'].chunks[1:]

        q = self[store_tmpf + '_MC'].data.map_blocks(
            lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
            chunks=new_chunks,  #
            drop_axis=avg_axis,  # avg dimesnions are dropped from input arr
            new_axis=0)  # The new CI dimension is added as first axis
        self[store_tmpf + '_MC'] = (('CI', 'x', 'time'), q)

    def conf_int_double_ended(self,
                              p_sol=None,
                              p_cov=None,
                              st_label='ST',
                              ast_label='AST',
                              rst_label='REV-ST',
                              rast_label='REV-AST',
                              st_var=None,
                              ast_var=None,
                              rst_var=None,
                              rast_var=None,
                              store_tmpf='TMPF',
                              store_tmpb='TMPB',
                              store_tempvar='_var',
                              conf_ints=None,
                              conf_ints_size=100,
                              ci_avg_time_flag=False,
                              da_random_state=None):
        """

        Parameters
        ----------
        p_sol : array-like
            parameter solution directly from calibration_double_ended_wls
        p_cov : array-like
            parameter covariance at the solution directly from calibration_double_ended_wls
        st_label
        ast_label
        rst_label
        rast_label
        st_var
        ast_var
        rst_var
        rast_var
        store_c
        store_gamma
        store_alphaint
        store_alpha
        store_tmpf
        store_tmpb
        store_tempvar
        conf_ints
        conf_ints_size
        ci_avg_time_flag
        da_random_state

        Returns
        -------

        """
        if da_random_state:
            # In testing environments
            assert isinstance(da_random_state, da.random.RandomState)
            state = da_random_state
        else:
            state = da.random.RandomState()

        assert conf_ints

        no, nt = self[st_label].data.shape
        npar = nt + 2 + no  # number of parameters

        assert isinstance(p_sol, (str, np.ndarray, np.generic))
        if isinstance(p_sol, str):
            p_sol = self[p_sol].data
        assert p_sol.shape == (npar,)

        assert isinstance(p_cov, (str, np.ndarray, np.generic))
        if isinstance(p_cov, str):
            p_cov = self[p_cov].data
        assert p_cov.shape == (npar, npar)

        p_mc = sst.multivariate_normal.rvs(mean=p_sol,
                                           cov=p_cov,
                                           size=conf_ints_size)  # this one takes long
        self.coords['MC'] = range(conf_ints_size)
        self.coords['CI'] = conf_ints

        rshape = (self.MC.size, self.x.size, self.time.size)
        r2shape = {
            0: -1,
            1: 'auto',
            2: -1}

        self['r_st'] = (('MC', 'x', 'time'), state.normal(
            loc=self[st_label].data,
            scale=st_var ** 0.5,
            size=rshape,
            chunks=r2shape))
        self['r_ast'] = (('MC', 'x', 'time'), state.normal(
            loc=self[ast_label].data,
            scale=ast_var ** 0.5,
            size=rshape,
            chunks=r2shape))
        self['r_rst'] = (('MC', 'x', 'time'), state.normal(
            loc=self[rst_label].data,
            scale=rst_var ** 0.5,
            size=rshape,
            chunks=r2shape))
        self['r_rast'] = (('MC', 'x', 'time'), state.normal(
            loc=self[rast_label].data,
            scale=rast_var ** 0.5,
            size=rshape,
            chunks=r2shape))

        gamma = p_mc[:, 0]
        alphaint = p_mc[:, 1]
        c = p_mc[:, 2:nt + 2]
        alpha = p_mc[:, nt + 2:]

        self['gamma_MC'] = (('MC',), gamma)
        self['alphaint_MC'] = (('MC',), alphaint)
        self['alpha_MC'] = (('MC', 'x',), alpha)
        self['c_MC'] = (('MC', 'time',), c)

        self[store_tmpf + '_MC'] = self['gamma_MC'] / (xr.ufuncs.log(
            self['r_st'] / self['r_ast']) + self['c_MC'] + self[
                                                           'alpha_MC']) - 273.15
        self[store_tmpb + '_MC'] = self['gamma_MC'] / (xr.ufuncs.log(
            self['r_rst'] / self['r_rast']) + self['c_MC'] - self[
                                                           'alpha_MC'] + self[
                                                           'alphaint_MC']) - 273.15

        del p_mc
        drop_var = ['gamma_MC',
                    'alphaint_MC',
                    'alpha_MC',
                    'c_MC',
                    'MC',
                    'r_st',
                    'r_ast',
                    'r_rst',
                    'r_rast']
        for k in drop_var:
            del self[k]

        if ci_avg_time_flag:
            avg_dims = ['MC', 'time']
        else:
            avg_dims = ['MC']

        avg_axis = self[store_tmpf + '_MC'].get_axis_num(avg_dims)

        if store_tempvar:
            self[store_tmpf + '_MC' + store_tempvar] = (self[store_tmpf + '_MC'] - self[
                store_tmpf]).std(dim=avg_dims) ** 2
            self[store_tmpb + '_MC' + store_tempvar] = (self[store_tmpb + '_MC'] - self[
                store_tmpb]).std(dim=avg_dims) ** 2

        if ci_avg_time_flag:
            new_chunks = ((len(conf_ints),),) + self[store_tmpf + '_MC'].chunks[1]
        else:
            new_chunks = ((len(conf_ints),),) + self[store_tmpf + '_MC'].chunks[1:]

        q = self[store_tmpf + '_MC'].data.map_blocks(
            lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
            chunks=new_chunks,  #
            drop_axis=avg_axis,  # avg dimesnions are dropped from input arr
            new_axis=0)  # The new CI dimension is added as first axis
        self[store_tmpf + '_MC'] = (('CI', 'x', 'time'), q)

        q = self[store_tmpb + '_MC'].data.map_blocks(
            lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
            chunks=new_chunks,  #
            drop_axis=avg_axis,  # avg dimesnions are dropped from input arr
            new_axis=0)  # The new CI dimension is added as first axis
        self[store_tmpb + '_MC'] = (('CI', 'x', 'time'), q)

    def ufunc_per_section(self,
                          func=None,
                          label=None,
                          subtract_from_label=None,
                          temp_err=False,
                          ref_temp_broadcasted=False,
                          calc_per='stretch',
                          **func_kwargs):
        """
        User function applied to parts of the cable. Super useful, many options and slightly
        complicated.

        The function `func` is taken over all the timesteps and calculated per `calc_per`. This
        is returned as a dictionary

        Parameters
        ----------
        func : callable
            A numpy function, or lambda function to apple to each 'calc_per'.
        label
        subtract_from_label
        temp_err : bool
            The argument of the function is label minus the reference temperature.
        ref_temp_broadcasted : bool
        calc_per : {'all', 'per_section', 'per_stretch'}
        func_kwargs : dict
            Dictionary with options that are passed to func

        Returns
        -------

        """

        if not func:
            def func(a):
                return a

        elif isinstance(func, str) and func == 'var':
            def func(a):
                return np.std(a) ** 2

        else:
            assert callable(func)

        out = dict()

        for k, section in self.sections.items():

            out[k] = []
            for stretch in section:
                arg1 = self[label].sel(x=stretch).data

                if subtract_from_label:
                    # calculate std wrt other series
                    check_dims(self, [subtract_from_label], correct_dims=('x', 'time'))

                    assert not temp_err

                    arg2 = self[subtract_from_label].sel(x=stretch).data
                    out[k].append(arg1 - arg2)

                elif temp_err:
                    # calculate std wrt reference temperature
                    arg2 = self[k].data
                    out[k].append(arg1 - arg2)

                elif ref_temp_broadcasted:
                    assert not temp_err
                    assert not subtract_from_label

                    arg2 = da.broadcast_to(self[k].data, arg1.shape)
                    out[k].append(arg2)

                else:
                    # calculate std wrt mean value
                    out[k].append(arg1)

            if calc_per == 'stretch':
                out[k] = [func(argi, **func_kwargs) for argi in out[k]]

            elif calc_per == 'section':
                out[k] = func(da.concatenate(out[k]), **func_kwargs)

        if calc_per == 'all':
            out = {k: da.concatenate(section) for k, section in out.items()}
            out = func(da.concatenate(list(out.values()), axis=0), **func_kwargs)

            if hasattr(out, 'chunks') and 'x' in self[label].dims:
                # also sum the chunksize in the x dimension
                # first find out where the x dim is
                ixdim = self[label].dims.index('x')
                c_old = out.chunks
                c_new = list(c_old)
                c_new[ixdim] = sum(c_old[ixdim])
                out = out.rechunk(c_new)

        return out


def open_datastore(filename_or_obj, group=None, decode_cf=True,
                   mask_and_scale=None, decode_times=True, autoclose=False,
                   concat_characters=True, decode_coords=True, engine=None,
                   chunks=None, lock=None, cache=None, drop_variables=None,
                   backend_kwargs=None, **kwargs):
    """Load and decode a datastore from a file or file-like object.
    Parameters
    ----------
    filename_or_obj : str, Path, file or xarray.backends.*DataStore
        Strings and Path objects are interpreted as a path to a netCDF file
        or an OpenDAP URL and opened with python-netCDF4, unless the filename
        ends with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). File-like objects are opened
        with scipy.io.netcdf (only netCDF3 supported).
    group : str, optional
        Path to the netCDF4 group in the given file to open (only works for
        netCDF4 files).
    decode_cf : bool, optional
        Whether to decode these variables, assuming they were saved according
        to CF conventions.
    mask_and_scale : bool, optional
        If True, replace array values equal to `_FillValue` with NA and scale
        values according to the formula `original_values * scale_factor +
        add_offset`, where `_FillValue`, `scale_factor` and `add_offset` are
        taken from variable attributes (if they exist).  If the `_FillValue` or
        `missing_value` attribute contains multiple values a warning will be
        issued and all array values matching one of the multiple values will
        be replaced by NA. mask_and_scale defaults to True except for the
        pseudonetcdf backend.
    decode_times : bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
    autoclose : bool, optional
        If True, automatically close files to avoid OS Error of too many files
        being open.  However, this option doesn't work with streams, e.g.,
        BytesIO.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
    decode_coords : bool, optional
        If True, decode the 'coordinates' attribute to identify coordinates in
        the resulting dataset.
    engine : {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'pynio', 'pseudonetcdf'}, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        'netcdf4'.
    chunks : int or dict, optional
        If chunks is provided, it used to load the new dataset into dask
        arrays. ``chunks={}`` loads the dataset with dask using a single
        chunk for all arrays.
    lock : False, True or threading.Lock, optional
        If chunks is provided, this argument is passed on to
        :py:func:`dask.array.from_array`. By default, a global lock is
        used when reading data from netCDF files with the netcdf4 and h5netcdf
        engines to avoid issues with concurrent access when using dask's
        multithreaded backend.
    cache : bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False. Does not
        change the behavior of coordinates corresponding to dimensions, which
        always load their data from disk into a ``pandas.Index``.
    drop_variables: string or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    backend_kwargs: dictionary, optional
        A dictionary of keyword arguments to pass on to the backend. This
        may be useful when backend options would improve performance or
        allow user control of dataset processing.
    Returns
    -------
    dataset : Dataset
        The newly created dataset.
    See Also
    --------
    read_xml_dir
    """

    xr_kws = inspect.signature(xr.open_dataset).parameters.keys()

    ds_kwargs = {k: v for k, v in kwargs.items() if k not in xr_kws}

    if chunks is None:
        chunks = {
            'x':    'auto',
            'time': -1}

    ds_xr = xr.open_dataset(
        filename_or_obj, group=group, decode_cf=decode_cf,
        mask_and_scale=mask_and_scale, decode_times=decode_times, autoclose=autoclose,
        concat_characters=concat_characters, decode_coords=decode_coords, engine=engine,
        chunks=chunks, lock=lock, cache=cache, drop_variables=drop_variables,
        backend_kwargs=backend_kwargs)

    ds = DataStore(data_vars=ds_xr.data_vars,
                   coords=ds_xr.coords,
                   attrs=ds_xr.attrs,
                   **ds_kwargs)

    ds_xr.close()

    return ds


def read_silixa_files(
    filepathlist=None,
    directory=None,
    file_ext='*.xml',
    timezone_netcdf='UTC',
    timezone_ultima_xml='UTC',
    silent=False,
        **kwargs):

    """Read a folder with measurement files. Each measurement file contains values for a
    single timestep. Remember to check which timezone you are working in.

    Parameters
    ----------
    filepathlist : list of str, optional
        List of paths that point the the silixa files
    directory : str, Path, optional
        Path to folder
    timezone_netcdf : str, optional
        Timezone string of the netcdf file. UTC follows CF-conventions.
    timezone_ultima_xml : str, optional
        Timezone string of the measurement files. Remember to check when measurements are taken.
        Also if summertime is used.
    file_ext : str, optional
        file extension of the measurement files
    silent : bool
        If set tot True, some verbose texts are not printed to stdout/screen
    kwargs : dict-like, optional
        keyword-arguments are passed to DataStore initialization

    Returns
    -------
    datastore : DataStore
        The newly created datastore.
    """

    if not filepathlist:
        filepathlist = sorted(glob.glob(os.path.join(directory, file_ext)))

    # Make sure that the list of files contains any files
    assert len(filepathlist) >= 1, 'No measurement files found in provided list/directory'

    # read raw files:
    data_vars, coords, attrs = read_silixa_files_routine(
        filepathlist,
        timezone_netcdf=timezone_netcdf,
        timezone_ultima_xml=timezone_ultima_xml,
        silent=silent)

    ds = DataStore(data_vars=data_vars,
                   coords=coords,
                   attrs=attrs,
                   **kwargs)

    return ds


def plot_dask(arr, file_path=None):
    from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler, visualize

    with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
        out = arr.compute()

    arr.visualize(file_path)

    visualize([prof, rprof, cprof], show=True)

    return out

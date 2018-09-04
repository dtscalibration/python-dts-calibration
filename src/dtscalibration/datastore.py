# coding=utf-8
import glob
import inspect
import os

import dask.array as da
import numpy as np
import scipy.sparse as sp
import xarray as xr
import yaml
from scipy.sparse import linalg as ln

from dtscalibration.calibrate_utils import calibration_double_ended_ols
from dtscalibration.calibrate_utils import calibration_double_ended_wls
from dtscalibration.calibrate_utils import calibration_single_ended_ols
from dtscalibration.calibrate_utils import calibration_single_ended_wls
from dtscalibration.calibrate_utils import conf_int_double_ended
from dtscalibration.calibrate_utils import conf_int_single_ended
from dtscalibration.datastore_utils import check_dims
from dtscalibration.datastore_utils import coords_time
from dtscalibration.datastore_utils import grab_data2


class DataStore(xr.Dataset):
    """To load data from a file or file-like object, use the `open_datastore`
        function.

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
        """

    def __init__(self, *args, **kwargs):
        super(DataStore, self).__init__(*args, **kwargs)

        if '_sections' not in self.attrs:
            self.attrs['_sections'] = yaml.dump(None)

        if 'sections' in kwargs:
            self.sections = kwargs['sections']

    @property
    def sections(self):
        assert hasattr(self, '_sections'), 'first set the sections'
        return yaml.load(self.attrs['_sections'])

    @sections.setter
    def sections(self, sections):
        if sections:
            assert isinstance(sections, dict)

            for k, v in sections.items():
                for vi in v:
                    assert isinstance(vi, slice)

        self.attrs['_sections'] = yaml.dump(sections)
        pass

    @sections.deleter
    def sections(self):
        self.sections = None
        pass

    def variance_stokes(self, st_label, sections=None, use_statsmodels=False,
                        suppress_info=True, debug_high_stokes_variance=False):
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

        resid_da = xr.DataArray(data=np.full(shape=self.ST.shape, fill_value=np.nan),
                                dims=('x', 'time'),
                                coords={'x': self.x, 'time': self.time})

        ix_resid = np.array([np.argmin(np.abs(ai - self.x.data)) for ai in resid_x])
        self.x.sel(x=resid_x, method='nearest')
        resid_da[ix_resid, :] = resid

        return var_I, resid_da

    def inverse_variance_weighted_mean(self,
                                       tmp1='TMPF',
                                       tmp2='TMPB',
                                       tmp1_var='TMPF_MC_var',
                                       tmp2_var='TMPB_MC_var',
                                       tmpw='TMPW',
                                       tmpw_var='TMPW_var'):
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
        tmpw : str
            The label of the averaged temperature dataset
        tmpw_var : str
            The label of the variance of the averaged temperature dataset

        Returns
        -------

        """

        self[tmpw_var] = 1 / (1 / self[tmp1_var] +
                              1 / self[tmp2_var])

        self[tmpw] = (self[tmp1] / self[tmp1_var] +
                      self[tmp2] / self[tmp2_var]) * self[tmpw_var]

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
                                 variance_suffix='_var',
                                 method='ols',
                                 store_tempvar=None,
                                 conf_ints=None,
                                 conf_ints_size=100,
                                 ci_avg_time_flag=False,
                                 solver='sparse',
                                 da_random_state=None
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

        if conf_ints:
            assert method == 'wls'
            conf_int_single_ended(
                self,
                p_sol,
                p_cov,
                st_label=st_label,
                ast_label=ast_label,
                st_var=st_var,
                ast_var=ast_var,
                store_c=store_c,
                store_gamma=store_gamma,
                store_dalpha=store_dalpha,
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
                                 variance_suffix='_var',
                                 method='ols',
                                 store_tempvar=None,
                                 conf_ints=None,
                                 conf_ints_size=100,
                                 ci_avg_time_flag=False,
                                 solver='sparse',
                                 da_random_state=None
                                 ):
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
                st_var, ast_var, rst_var, rast_var, solver=solver)

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

        if conf_ints:
            assert method == 'wls'
            conf_int_double_ended(
                self,
                p_sol,
                p_cov,
                st_label=st_label,
                ast_label=ast_label,
                rst_label=rst_label,
                rast_label=rast_label,
                st_var=st_var,
                ast_var=ast_var,
                rst_var=rst_var,
                rast_var=rast_var,
                store_c=store_c,
                store_gamma=store_gamma,
                store_alphaint=store_alphaint,
                store_alpha=store_alpha,
                store_tmpf=store_tmpf,
                store_tmpb=store_tmpb,
                store_tempvar=store_tempvar,
                conf_ints=conf_ints,
                conf_ints_size=conf_ints_size,
                ci_avg_time_flag=ci_avg_time_flag,
                da_random_state=da_random_state)

        pass

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

                    arg2 = np.broadcast_to(self[k].data, arg1.shape)
                    out[k].append(arg2)

                else:
                    # calculate std wrt mean value
                    out[k].append(arg1)

            if calc_per == 'stretch':
                out[k] = [func(argi, **func_kwargs) for argi in out[k]]

            elif calc_per == 'section':
                out[k] = func(np.concatenate(out[k]), **func_kwargs)

        if calc_per == 'all':
            out = {k: np.concatenate(section) for k, section in out.items()}
            out = func(np.concatenate(list(out.values()), axis=0), **func_kwargs)

        return out


def open_datastore(filename_or_obj, **kwargs):
    """Load and decode a datastore from a file or file-like object
    into a DataStore object.

    Parameters
    ----------
    filename_or_obj : str, Path, file or xarray.backends.*DataStore
        Strings and Path objects are interpreted as a path to a netCDF file
        or an OpenDAP URL and opened with python-netCDF4, unless the filename
        ends with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). File-like objects are opened
        with scipy.io.netcdf (only netCDF3 supported).
    Returns
    -------
    datastore : DataStore
        The newly created datastore.
    See Also
    --------
    read_xml_dir
    """

    xr_kws = inspect.signature(xr.open_dataset).parameters.keys()

    xr_kwargs = {k: v for k, v in kwargs.items() if k in xr_kws}
    ds_kwargs = {k: v for k, v in kwargs.items() if k not in xr_kws}

    ds_xr = xr.open_dataset(filename_or_obj, **xr_kwargs)

    ds = DataStore(data_vars=ds_xr.data_vars,
                   coords=ds_xr.coords,
                   attrs=ds_xr.attrs,
                   **ds_kwargs)
    return ds


def read_xml_dir(filepath,
                 timezone_netcdf='UTC',
                 timezone_ultima_xml='Europe/Amsterdam',
                 file_ext='*.xml',
                 **kwargs):
    """Read a folder with measurement files. Each measurement file contains values for a
    single timestep. Remember to check which timezone you are working in.

    Parameters
    ----------
    filepath : str, Path
        Path to folder
    timezone_netcdf : str, optional
        Timezone string of the netcdf file. UTC follows CF-conventions.
    timezone_ultima_xml : str, optional
        Timezone string of the measurement files. Remember to check when measurements are taken.
        Also if summertime is used.
    file_ext : str, optional
        file extension of the measurement files
    kwargs : dict-like, optional
        keyword-arguments are passed to DataStore initialization

    Returns
    -------
    datastore : DataStore
        The newly created datastore.
    """
    log_attrs = {
        'x':                     {
            'description':      'Length along fiber',
            'long_describtion': 'Starting at connector of forward channel',
            'units':            'm'},
        'TMP':                   {
            'description': 'temperature calibrated by device',
            'units':       'degC'},
        'ST':                    {
            'description': 'Stokes intensity',
            'units':       '-'},
        'AST':                   {
            'description': 'anti-Stokes intensity',
            'units':       '-'},
        'REV-ST':                {
            'description': 'reverse Stokes intensity',
            'units':       '-'},
        'REV-AST':               {
            'description': 'reverse anti-Stokes intensity',
            'units':       '-'},
        'acquisitionTime':       {
            'description':      'Measurement duration of forward channel',
            'long_describtion': 'Actual measurement duration of forward channel',
            'units':            'seconds'},
        'userAcquisitionTimeFW': {
            'description':      'Measurement duration of forward channel',
            'long_describtion': 'Desired measurement duration of forward channel',
            'units':            'seconds'},
        'userAcquisitionTimeBW': {
            'description':      'Measurement duration of backward channel',
            'long_describtion': 'Desired measurement duration of backward channel',
            'units':            'seconds'},
        }

    filepathlist = sorted(glob.glob(os.path.join(filepath, file_ext)))
    filenamelist = [os.path.basename(path) for path in filepathlist]
    assert len(filepathlist) >= 1, 'No measurement files with extension {} found in {}'.format(
        file_ext, filepath)

    array, timearr, meta, extra = grab_data2(filepathlist)

    coords = {
        'x':        ('x', array['LAF'][:, 0], log_attrs['x']),
        'filename': ('time', filenamelist)}
    tcoords = coords_time(extra, timearr,
                          timezone_netcdf=timezone_netcdf,
                          timezone_ultima_xml=timezone_ultima_xml)
    coords.update(tcoords)

    dataset_dict = {}
    for name in array.dtype.names:
        if name in ['TMP', 'ST', 'AST', 'REV-ST', 'REV-AST']:
            dataset_dict[name] = (['x', 'time'], array[name], log_attrs[name])

        elif name == 'LAF':
            continue

        else:
            print(name)
            assert 0

    for key, item in extra.items():
        if key in log_attrs:
            dataset_dict[key] = (['time'], item['array'], log_attrs[key])

        else:
            dataset_dict[key] = (['time'], item['array'])

    ds = DataStore(data_vars=dataset_dict,
                   coords=coords,
                   attrs=meta,
                   **kwargs)

    return ds

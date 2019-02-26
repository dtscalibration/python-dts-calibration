# coding=utf-8
import glob
import inspect
import os
from typing import Dict
from typing import List

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
from .io import read_sensornet_files_routine_v3
from .io import read_silixa_files_routine_v4
from .io import read_silixa_files_routine_v6
from .io import silixa_xml_version_check

dtsattr_namelist = ['double_ended_flag']


class DataStore(xr.Dataset):
    """The data class that stores the measurements, contains calibration
    methods to relate Stokes and anti-Stokes to temperature. The user should
    never initiate this class directly, but use read_xml_dir or open_datastore
    functions instead.

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
            Sections for calibration. The dictionary should contain key-var
            couples in which the key is the name of the calibration temp time
            series. And the var is a list of slice objects as 'slice(start,
            stop)'; start and stop in meter (float).
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
        dtscalibration.open_datastore : Load (calibrated) measurements from
        netCDF-like file
        """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if '_sections' not in self.attrs:
            self.attrs['_sections'] = yaml.dump(None)

        if 'sections' in kwargs:
            self.sections = kwargs['sections']

    def __unicode__(self):
        # __repr__ from xarray is used and edited.
        #   'xarray' is prepended. so we remove it and add 'dtscalibration'
        s = xr.core.formatting.dataset_repr(self)
        name_module = type(self).__name__
        preamble_new = u'<dtscalibration.%s>' % name_module

        # Add sections to new preamble
        preamble_new += '\nSections:\n'
        if hasattr(self, '_sections') and self.sections:
            unit = self.x.units
            for k, v in self.sections.items():
                preamble_new += '    {0: <23}'.format(k)
                vl = ['{0:.2f}{2} - {1:.2f}{2}'.format(vi.start, vi.stop, unit)
                      for vi in v]
                preamble_new += ' and '.join(vl) + '\n'

        else:
            preamble_new += 18 * ' ' + '()\n'

        # add new preamble to the remainder of the former __repr__
        len_preamble_old = 8 + len(name_module) + 2

        # untill the attribute listing
        attr_index = s.find('Attributes:')

        # abbreviate attribute listing
        attr_list_all = s[attr_index:].split(sep='\n')
        if len(attr_list_all) > 10:
            s_too_many = ['\n.. and many more attributes. See: ds.attrs']
            attr_list = attr_list_all[:10] + s_too_many
        else:
            attr_list = attr_list_all

        s_out = (preamble_new +
                 s[len_preamble_old:attr_index] +
                 '\n'.join(attr_list))

        # return new __repr__
        return s_out

    # noinspection PyIncorrectDocstring
    @property
    def sections(self):
        """
        Define calibration sections. Each section requires a reference
        temperature time series, such as the temperature measured by an
        external temperature sensor. They should already be part of the
        DataStore object.

        Please look at the example notebook on `sections` if you encounter
        difficulties.

        Parameters
        ----------
        sections : Dict[str, List[slice]]
            Sections are defined in a dictionary with its keywords of the
            names of the reference
            temperature time series. Its values are lists of slice objects,
            where each slice object
            is a stretch.
        Returns
        -------

        """
        assert hasattr(self, '_sections'), 'first set the sections'
        return yaml.load(self.attrs['_sections'])

    @sections.setter
    def sections(self, sections: Dict[str, List[slice]]):
        if sections:
            assert isinstance(sections, dict)

            for key in sections:
                assert key in self.data_vars, 'The keys of the ' \
                                              'sections-dictionary should ' \
                                              'refer ' \
                                              'to a valid timeserie already ' \
                                              'stored in ds.data_vars'
            check_dims(self, sections, ('time',))

            for k, v in sections.items():
                assert isinstance(v, (list, tuple)), \
                    'The values of the sections-dictionary ' \
                    'should be lists of slice objects.'

                for vi in v:
                    assert isinstance(vi, slice), \
                        'The values of the sections-dictionary should ' \
                        'be lists of slice objects.'
                    assert self.x.sel(x=vi).size > 0, \
                        f'Better define the {k} section. You tried {vi}, ' \
                        'which is out of reach'

        self.attrs['_sections'] = yaml.dump(sections)
        pass

    @sections.deleter
    def sections(self):
        self.sections = None
        pass

    @property
    def is_double_ended(self):
        """

        Returns
        -------

        """
        if 'isDoubleEnded' in self.attrs:
            return bool(int(self.attrs['isDoubleEnded']))
        elif 'customData:isDoubleEnded' in self.attrs:
            # backward compatible to when only silixa files were supported
            return bool(int(self.attrs['customData:isDoubleEnded']))
        else:
            assert 0

    @is_double_ended.setter
    def is_double_ended(self, flag: bool):
        self.attrs['isDoubleEnded'] = flag
        pass

    @property
    def chfw(self):
        """

        Returns
        -------

        """
        return int(self.attrs['forwardMeasurementChannel']) - 1  # zero-based

    @property
    def chbw(self):
        """

        Returns
        -------

        """
        if self.is_double_ended:
            return int(
                self.attrs['reverseMeasurementChannel']) - 1  # zero-based
        else:
            return None

    @property
    def channel_configuration(self):
        """

        Returns
        -------

        """
        d = {
            'chfw':
                {
                    'st_label': 'ST',
                    'ast_label': 'AST',
                    'acquisitiontime_label': 'userAcquisitionTimeFW',
                    'time_start_label': 'timeFWstart',
                    'time_label': 'timeFW',
                    'time_end_label': 'timeFWend'},
            'chbw':
                {
                    'st_label': 'REV-ST',
                    'ast_label': 'REV-AST',
                    'acquisitiontime_label': 'userAcquisitionTimeBW',
                    'time_start_label': 'timeBWstart',
                    'time_label': 'timeBW',
                    'time_end_label': 'timeBWend'}}
        return d

    @property
    def timeseries_keys(self):
        """
        Returns the keys of all timeseires that can be used for calibration.
        """
        return [k for k, v in self.data_vars.items() if v.dims == ('time',)]

    def resample_datastore(
            self,
            how,
            freq=None,
            dim=None,
            skipna=None,
            closed=None,
            label=None,
            base=0,
            keep_attrs=True,
            **indexer):
        """Returns a resampled DataStore. Always define the how.
        Handles both downsampling and upsampling. If any intervals contain no
        values from the original object, they will be given the value ``NaN``.
        Parameters
        ----------
        freq
        dim
        how : str
            Any function that is available via groupby. E.g., 'mean'
            http://pandas.pydata.org/pandas-docs/stable/groupby.html#groupby
            -dispatch
        skipna : bool, optional
            Whether to skip missing values when aggregating in downsampling.
        closed : 'left' or 'right', optional
            Side of each interval to treat as closed.
        label : 'left or 'right', optional
            Side of each interval to use for labeling.
        base : int, optional
            For frequencies that evenly subdivide 1 day, the "origin" of the
            aggregated intervals. For example, for '24H' frequency, base could
            range from 0 through 23.
        keep_attrs : bool, optional
            If True, the object's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **indexer : {dim: freq}
            Dictionary with a key indicating the dimension name to resample
            over and a value corresponding to the resampling frequency.
        Returns
        -------
        resampled : same type as caller
            This object resampled.
            """

        from xarray.core.dataarray import DataArray
        import pandas as pd

        RESAMPLE_DIM = '__resample_dim__'

        if (freq and indexer) or (dim and indexer):
            raise TypeError(
                "If passing an 'indexer' then 'dim' "
                "and 'freq' should not be used")

        if indexer:
            dim, freq = indexer.popitem()

        if isinstance(dim, str):
            dim = self[dim]
        else:
            raise TypeError(
                "Dimension name should be a string; "
                "was passed %r" % dim)

        if how is None:
            how = 'mean'

        group = DataArray(dim, [(dim.dims, dim)], name=RESAMPLE_DIM)
        grouper = pd.Grouper(
            freq=freq, how=how, closed=closed, label=label, base=base)
        gb = self._groupby_cls(self, group, grouper=grouper)
        if isinstance(how, str):
            f = getattr(gb, how)
            if how in ['first', 'last']:
                result = f(skipna=skipna, keep_attrs=False)
            elif how == 'count':
                result = f(dim=dim.name, keep_attrs=False)
            else:
                result = f(dim=dim.name, skipna=skipna, keep_attrs=False)
        else:
            result = gb.reduce(how, dim=dim.name, keep_attrs=False)
        result = result.rename({RESAMPLE_DIM: dim.name})

        if keep_attrs:
            attrs = self.attrs
        else:
            attrs = None

        out = DataStore(
            data_vars=result.data_vars, coords=result.coords, attrs=attrs)

        return out

    def to_netcdf(
            self,
            path=None,
            mode='w',
            format=None,
            group=None,
            engine=None,
            encoding=None,
            unlimited_dims=None,
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
        format : {'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT',
        'NETCDF3_CLASSIC'}, optional
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
            defaults to reasonable compression. Use encoding={} to disable
            encoding.
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
            path,
            mode,
            format=format,
            group=group,
            engine=engine,
            encoding=encoding,
            unlimited_dims=unlimited_dims,
            compute=compute)

    def get_default_encoding(self):
        """

        Returns
        -------

        """
        # TODO: set scale parameter
        compdata = dict(
            zlib=True, complevel=6,
            shuffle=False)  # , least_significant_digit=None
        compcoords = dict(zlib=True, complevel=4)

        encoding = {var: compdata for var in self.data_vars}
        encoding.update({var: compcoords for var in self.coords})
        return encoding

    def variance_stokes(
            self,
            st_label,
            sections=None,
            use_statsmodels=False,
            suppress_info=True,
            reshape_residuals=True):
        """Calculates the variance between the measurements and a best fit
        exponential at each reference section. This fits a two-parameter
        exponential to the stokes measurements. The temperature is constant
        and there are no splices/sharp bends in each reference section.
        Therefore all signal decrease is due to differential attenuation,
        which is the same for each reference section. The scale of the
        exponential does differ per reference section.

        Assumptions: 1) the temperature is the same along a reference
        section. 2) no sharp bends and splices in the reference sections. 3)
        Same type of optical cable in each reference section.

        Idea from discussion at page 127 in Richter, P. H. (1995). Estimating
        errors in least-squares fitting. For weights used error propagation:
        w^2 = 1/sigma(lny)^2 = y^2/sigma(y)^2 = y^2

        Parameters
        ----------
        reshape_residuals
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

        nt = self.time.size

        len_stretch_list = []  # number of reference points per section (
        # spatial)
        y_list = []  # intensities of stokes
        x_list = []  # length rel to start of section. for alpha

        for k, stretches in self.sections.items():
            for stretch in stretches:
                y_list.append(self[st_label].sel(x=stretch).data.T.reshape(-1))
                _x = self.x.sel(x=stretch).data.copy()
                _x -= _x[0]
                x_list.append(da.tile(_x, nt))
                len_stretch_list.append(_x.size)

        n_sections = len(len_stretch_list)  # number of sections
        n_locs = sum(len_stretch_list)  # total number of locations along cable used
        # for reference.

        x = np.concatenate(x_list)  # coordinates are already in memory
        y = np.concatenate(y_list)

        data1 = x
        data2 = np.ones(sum(len_stretch_list) * nt)
        data = np.concatenate([data1, data2])

        # alpha is NOT the same for all -> one column per section
        coords1row = np.arange(nt * n_locs)
        coords1col = np.hstack(
            [np.ones(in_locs * nt) * i
                for i, in_locs in enumerate(len_stretch_list)])  # C for

        # second calibration parameter is different per section and per timestep
        coords2row = np.arange(nt * n_locs)
        coords2col = np.hstack(
            [np.repeat(
                np.arange(i * nt + n_sections, (i + 1) * nt + n_sections),
                in_locs)
                for i, in_locs in enumerate(len_stretch_list)])  # C for
        coords = (
            np.concatenate([coords1row, coords2row]),
            np.concatenate([coords1col, coords2col]))

        lny = np.log(y)
        w = y.copy()  # 1/std.

        ddof = n_sections + nt * n_sections  # see numpy documentation on ddof

        if use_statsmodels:
            # returns the same answer with statsmodel
            import statsmodels.api as sm

            X = sp.coo_matrix(
                (data, coords),
                shape=(nt * n_locs, n_sections + nt * n_sections),
                dtype=float,
                copy=False)

            mod_wls = sm.WLS(lny, X.todense(), weights=w**2)
            res_wls = mod_wls.fit()
            # print(res_wls.summary())
            a = res_wls.params

        else:
            wdata = data * np.hstack((w, w))
            wX = sp.coo_matrix(
                (wdata, coords),
                shape=(nt * n_locs, n_sections + nt * n_sections),
                dtype=float,
                copy=False)

            wlny = (lny * w)

            p0_est = np.asarray(n_sections * [0.] + nt * n_sections * [8])
            # noinspection PyTypeChecker
            a = ln.lsqr(
                wX, wlny, x0=p0_est, show=not suppress_info, calc_var=False)[0]

        beta = a[:n_sections]
        beta_expand_to_sec = np.hstack(
            [
                np.repeat(beta[i], leni * nt)
                for i, leni in enumerate(len_stretch_list)])
        G = a[n_sections:]
        G_expand_to_sec = np.hstack(
            [
                np.repeat(G[i * nt:(i + 1) * nt], leni)
                for i, leni in enumerate(len_stretch_list)])

        I_est = np.exp(G_expand_to_sec) * np.exp(x * beta_expand_to_sec)
        resid = I_est - y
        var_I = resid.std(ddof=ddof)**2

        if not reshape_residuals:
            return var_I, resid
        else:
            # restructure the residuals, such that they can be plotted and
            # added to ds
            resid_res = []
            for leni, lenis, lenie in zip(
                    len_stretch_list,
                    nt * np.cumsum([0] + len_stretch_list[:-1]),
                    nt * np.cumsum(len_stretch_list)):

                resid_res.append(
                    resid[lenis:lenie].reshape((leni, nt), order='F'))

            _resid = np.concatenate(resid_res)
            _resid_x = self.ufunc_per_section(label='x', calc_per='all')
            isort = np.argsort(_resid_x)
            resid_x = _resid_x[isort]  # get indices from ufunc directly
            resid = _resid[isort, :]

            ix_resid = np.array(
                [np.argmin(np.abs(ai - self.x.data)) for ai in resid_x])

            resid_sorted = np.full(
                shape=self[st_label].shape, fill_value=np.nan)
            resid_sorted[ix_resid, :] = resid
            resid_da = xr.DataArray(
                data=resid_sorted,
                dims=('x', 'time'),
                coords={
                    'x': self.x,
                    'time': self.time})

            return var_I, resid_da

    def i_var_fw(self, st_var, ast_var, st_label='ST', ast_label='AST'):
        st = self[st_label]
        ast = self[ast_label]
        return 1 / st**2 * st_var + 1 / ast**2 * ast_var

    def i_var_bw(self, rst_var, rast_var, rst_label='ST', rast_label='AST'):
        rst = self[rst_label]
        rast = self[rast_label]
        return 1 / rst**2 * rst_var + 1 / rast**2 * rast_var

    def inverse_variance_weighted_mean(
            self,
            tmp1='TMPF',
            tmp2='TMPB',
            tmp1_var='TMPF_MC_var',
            tmp2_var='TMPB_MC_var',
            tmpw_store='TMPW',
            tmpw_var_store='TMPW_var'):
        """
        Average two temperature datasets with the inverse of the variance as
        weights. The two
        temperature datasets `tmp1` and `tmp2` with their variances
        `tmp1_var` and `tmp2_var`,
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

        self[tmpw_var_store] = 1 / (1 / self[tmp1_var] + 1 / self[tmp2_var])

        self[tmpw_store] = (
            self[tmp1] / self[tmp1_var] +
            self[tmp2] / self[tmp2_var]) * self[tmpw_var_store]

        pass

    def inverse_variance_weighted_mean_array(
            self,
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
        self[tmpw_var_store] = 1 / (1 / self[tmp_var_label]).sum(dim=dim)

        self[tmpw_store] = (self[tmp_label] / self[tmp_var_label]).sum(
            dim=dim) / (1 / self[tmp_var_label]).sum(dim=dim)

        pass

    def calibration_single_ended(
            self,
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
            store_p_val='p_val',
            variance_suffix='_var',
            method='ols',
            solver='sparse',
            nt=None,
            z=None,
            p_val=None,
            p_var=None,
            p_cov=None):
        """

        Parameters
        ----------
        store_p_cov : str
            Key to store the covariance matrix of the calibrated parameters
        store_p_val : str
            Key to store the values of the calibrated parameters
        nt : int, optional
            Number of timesteps. Should be defined if method=='external'
        z : array-like, optional
            Distances. Should be defined if method=='external'
        p_val
        p_var
        p_cov
        sections : dict, optional
        st_label : str
            Label of the forward stokes measurement
        ast_label : str
            Label of the anti-Stoke measurement
        st_var : float, optional
            The variance of the measurement noise of the Stokes signals in
            the forward
            direction Required if method is wls.
        ast_var : float, optional
            The variance of the measurement noise of the anti-Stokes signals
            in the forward
            direction. Required if method is wls.
        store_c : str
            Label of where to store C
        store_gamma : str
            Label of where to store gamma
        store_dalpha : str
            Label of where to store dalpha; the spatial derivative  of alpha.
        store_alpha : str
            Label of where to store alpha; The integrated differential
            attenuation.
            alpha(x=0) = 0
        store_tmpf : str
            Label of where to store the calibrated temperature of the forward
            direction
        variance_suffix : str, optional
            String appended for storing the variance. Only used when method
            is wls.
        method : {'ols', 'wls'}
            Use 'ols' for ordinary least squares and 'wls' for weighted least
            squares
        solver : {'sparse', 'stats'}
            Either use the homemade weighted sparse solver or the weighted
            dense matrix solver of
            statsmodels

        Returns
        -------

        """

        if sections:
            self.sections = sections
        else:
            assert self.sections, 'sections are not defined'

        check_dims(self, [st_label, ast_label], correct_dims=('x', 'time'))

        if method == 'ols':
            nt, z, p_val = calibration_single_ended_ols(
                self, st_label, ast_label)

        elif method == 'wls' or method == 'external':
            if method == 'wls':
                for vari in [st_var, ast_var]:
                    assert isinstance(vari, float)

                nt, z, p_val, p_var, p_cov = calibration_single_ended_wls(
                    self, st_label, ast_label, st_var, ast_var, solver=solver)
            else:
                for input_item in [nt, z, p_val, p_var, p_cov]:
                    assert input_item is not None

        else:
            raise ValueError('Choose a valid method')

        # store calibration parameters in DataStore
        gamma = p_val[0]
        dalpha = p_val[1]
        c = p_val[2:nt + 2]

        self[store_gamma] = (tuple(), gamma)
        self[store_dalpha] = (tuple(), dalpha)
        self[store_alpha] = (('x',), dalpha * self.x.data)
        self[store_c] = (('time',), c)

        # store variances in DataStore
        if method == 'wls' or method == 'external':
            gammavar = p_var[0]
            dalphavar = p_var[1]
            cvar = p_var[2:nt + 2]

            self[store_gamma + variance_suffix] = (tuple(), gammavar)
            self[store_dalpha + variance_suffix] = (tuple(), dalphavar)
            self[store_c + variance_suffix] = (('time',), cvar)

        # deal with FW
        if store_tmpf:
            tempF_data = gamma / \
                         (np.log(self[st_label].data / self[ast_label].data)
                          + c + self.x.data[:, None] * dalpha) - 273.15
            self[store_tmpf] = (('x', 'time'), tempF_data)

        if store_p_val and (method == 'wls' or method == 'external'):
            self[store_p_val] = (('params1',), p_val)
        else:
            pass

        if store_p_cov and (method == 'wls' or method == 'external'):
            self[store_p_cov] = (('params1', 'params2'), p_cov)
        else:
            pass

        pass

    def calibration_double_ended(
            self,
            sections=None,
            st_label='ST',
            ast_label='AST',
            rst_label='REV-ST',
            rast_label='REV-AST',
            st_var=None,
            ast_var=None,
            rst_var=None,
            rast_var=None,
            store_d='d',
            store_gamma='gamma',
            store_alpha='alpha',
            store_tmpf='TMPF',
            store_tmpb='TMPB',
            store_p_cov='p_cov',
            store_p_val='p_val',
            variance_suffix='_var',
            method='ols',
            solver='sparse',
            nt=None,
            z=None,
            p_val=None,
            p_var=None,
            p_cov=None):
        """

        Parameters
        ----------
        store_p_cov
        store_p_val
        nt
        z
        p_val
        p_var
        p_cov
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
            The variance of the measurement noise of the Stokes signals in
            the forward
            direction Required if method is wls.
        ast_var : float, optional
            The variance of the measurement noise of the anti-Stokes signals
            in the forward
            direction. Required if method is wls.
        rst_var : float, optional
            The variance of the measurement noise of the Stokes signals in
            the backward
            direction. Required if method is wls.
        rast_var : float, optional
            The variance of the measurement noise of the anti-Stokes signals
            in the backward
            direction. Required if method is wls.
        store_d : str
            Label of where to store D. Equals the integrated differential
            attenuation at x=0
            And should be equal to half the total integrated differential
            attenuation.
        store_gamma : str
            Label of where to store gamma
        store_alpha : str
            Label of where to store alpha
        store_tmpf : str
            Label of where to store the calibrated temperature of the forward
            direction
        store_tmpb : str
            Label of where to store the calibrated temperature of the
            backward direction
        variance_suffix : str, optional
            String appended for storing the variance. Only used when method
            is wls.
        method : {'ols', 'wls', 'external'}
            Use 'ols' for ordinary least squares and 'wls' for weighted least
            squares
        solver : {'sparse', 'stats'}
            Either use the homemade weighted sparse solver or the weighted
            dense matrix solver of
            statsmodels

        Returns
        -------

        """

        if sections:
            self.sections = sections
        else:
            assert self.sections, 'sections are not defined'

        check_dims(
            self, [st_label, ast_label, rst_label, rast_label],
            correct_dims=('x', 'time'))

        if method == 'ols':
            nt, z, p_val = calibration_double_ended_ols(
                self, st_label, ast_label, rst_label, rast_label)

        elif method == 'wls' or method == 'external':
            # External is also/always weighted
            if method == 'wls':
                for vari in [st_var, ast_var, rst_var, rast_var]:
                    assert isinstance(vari, float)

                nt, z, p_val, p_var, p_cov = calibration_double_ended_wls(
                    self,
                    st_label,
                    ast_label,
                    rst_label,
                    rast_label,
                    st_var,
                    ast_var,
                    rst_var,
                    rast_var,
                    solver=solver)
            else:
                for input_item in [nt, z, p_val, p_var, p_cov]:
                    assert input_item is not None

        else:
            raise ValueError('Choose a valid method')

        gamma = p_val[0]
        d = p_val[1:nt + 1]
        alpha = p_val[nt + 1:]

        # store calibration parameters in DataStore
        self[store_gamma] = (tuple(), gamma)
        self[store_alpha] = (('x',), alpha)
        self[store_d] = (('time',), d)

        # store variances in DataStore
        if method == 'wls' or method == 'external':
            # the variances only have ameaning if the observations are weighted
            gammavar = p_var[0]
            dvar = p_var[1:nt + 1]
            alphavar = p_var[nt + 1:]

            self[store_gamma + variance_suffix] = (tuple(), gammavar)
            self[store_alpha + variance_suffix] = (('x',), alphavar)
            self[store_d + variance_suffix] = (('time',), dvar)

        # deal with FW
        if store_tmpf:
            tempF_data = gamma / \
                         (np.log(self[st_label].data / self[ast_label].data)
                          + d + alpha[:, None]) - 273.15
            self[store_tmpf] = (('x', 'time'), tempF_data)

        # deal with BW
        if store_tmpb:
            tempB_data = gamma / \
                         (np.log(self[rst_label].data / self[rast_label].data)
                          + d - alpha[:, None]) - 273.15
            self[store_tmpb] = (('x', 'time'), tempB_data)

        if store_p_val and (method == 'wls' or method == 'external'):
            # TODO: add params1 dimension
            self[store_p_val] = (('params1',), p_val)
        else:
            pass

        if store_p_cov and (method == 'wls' or method == 'external'):
            # TODO: add params1 dimension
            # TODO: add params2 dimension
            self[store_p_cov] = (('params1', 'params2'), p_cov)
        else:
            pass

        pass

    def conf_int_single_ended(
            self,
            p_val='p_val',
            p_cov='p_cov',
            st_label='ST',
            ast_label='AST',
            st_var=None,
            ast_var=None,
            store_tmpf='TMPF',
            store_tempvar='_var',
            conf_ints=None,
            conf_ints_size=100,
            ci_avg_time_flag=False,
            ci_avg_x_flag=False,
            da_random_state=None,
            remove_mc_set_flag=True):
        """

        Parameters
        ----------
        p_val : array-like or string
            parameter solution directly from calibration_double_ended_wls
        p_cov : array-like or string or bool
            parameter covariance at p_val directly from
            calibration_double_ended_wls. If set to False, no uncertainty in
            the parameters is propagated into the confidence intervals.
            Similar to the spec sheets of the DTS manufacturers. And similar to
            passing an array filled with zeros. If set to string, the p_cov
            is retreived by accessing ds[p_cov] . See p_cov keyword argument in
            the calibration routine.
        st_label : str
            Key of the forward Stokes
        ast_label : str
            Key of the forward anti-Stokes
        st_var : float
            Float of the variance of the Stokes signal
        ast_var : float
            Float of the variance of the anti-Stokes signal
        store_tmpf : str
            Key of how to store the Forward calculated temperature. Is
            calculated using the
            forward Stokes and anti-Stokes observations.
        store_tempvar : str
            a string that is appended to the store_tmp_ keys. and the
            variance is calculated
            for those store_tmp_ keys
        conf_ints : iterable object of float
            A list with the confidence boundaries that are calculated. Valid
            values are between
            [0, 1].
        conf_ints_size : int
            Size of the monte carlo parameter set used to calculate the
            confidence interval
        ci_avg_time_flag : bool
            The confidence intervals differ per time step. If you would like
            to calculate confidence
            intervals of all time steps together. ‘We can say with 95%
            confidence that the
            temperature remained between this line and this line during the
            entire measurement
            period’.
        ci_avg_x_flag : bool
            Similar to `ci_avg_time_flag` but then over the x-dimension
            instead of the time-dimension
        da_random_state
            For testing purposes. Similar to random seed. The seed for dask.
            Makes random not so random. To produce reproducable results for
            testing environments.
        remove_mc_set_flag : bool
            Remove the monte carlo data set, from which the CI and the
            variance are calculated.
        """

        assert conf_ints

        if da_random_state:
            state = da_random_state
        else:
            state = da.random.RandomState()

        no, nt = self[st_label].data.shape
        npar = nt + 2  # number of parameters

        self.coords['MC'] = range(conf_ints_size)
        self.coords['CI'] = conf_ints

        assert isinstance(p_val, (str, np.ndarray, np.generic))
        if isinstance(p_val, str):
            p_val = self[p_val].data
        assert p_val.shape == (npar,)

        assert isinstance(p_cov, (str, np.ndarray, np.generic, bool))
        if isinstance(p_cov, bool) and not p_cov:
            gamma = p_val[0]
            dalpha = p_val[1]
            c = p_val[2:nt + 2]
            self['gamma_MC'] = (tuple(), gamma)
            self['dalpha_MC'] = (tuple(), dalpha)
            self['c_MC'] = (('time',), c)

        elif isinstance(p_cov, bool) and p_cov:
            raise NotImplementedError(
                'Not an implemented option. Check p_cov argument')

        else:
            if isinstance(p_cov, str):
                p_cov = self[p_cov].data
            assert p_cov.shape == (npar, npar)

            p_mc = sst.multivariate_normal.rvs(
                mean=p_val, cov=p_cov, size=conf_ints_size)

            gamma = p_mc[:, 0]
            dalpha = p_mc[:, 1]
            c = p_mc[:, 2:nt + 2]

            self['gamma_MC'] = (('MC',), gamma)
            self['dalpha_MC'] = (('MC',), dalpha)
            self['c_MC'] = ((
                'MC',
                'time',
            ), c)

        rshape = (self.MC.size, self.x.size, self.time.size)

        if ci_avg_x_flag and not ci_avg_time_flag:
            r2shape = {0: -1, 1: -1, 2: 'auto'}

        elif ci_avg_x_flag and ci_avg_time_flag:
            r2shape = {0: -1, 1: 'auto', 2: 'auto'}

        else:
            r2shape = {0: -1, 1: 'auto', 2: -1}

        for k, st_labeli, st_vari in zip(
            ['r_st', 'r_ast'],
            [st_label, ast_label],
                [st_var, ast_var]):
            if hasattr(self[st_labeli].data, 'chunks'):
                chunks = self[st_label].chunks
                self[k] = (
                    ('MC', 'x', 'time'),
                    state.normal(
                        loc=da.rechunk(self[st_labeli].data, chunks=chunks),
                        scale=st_vari**0.5,
                        size=rshape,
                        chunks=r2shape))
            else:
                self[k] = (
                    ('MC', 'x', 'time'),
                    state.normal(
                        loc=da.from_array(self[st_labeli].data, chunks={}),
                        scale=st_vari**0.5,
                        size=rshape,
                        chunks=r2shape))

        self[store_tmpf + '_MC_set'] = self['gamma_MC'] / (
            np.log(self['r_st'] / self['r_ast']) + self['c_MC'] +
            self['dalpha_MC'] * self.x) - 273.15

        if ci_avg_time_flag and not ci_avg_x_flag:
            avg_dims = ['MC', 'time']
        elif ci_avg_x_flag and not ci_avg_time_flag:
            avg_dims = ['MC', 'x']
        elif ci_avg_x_flag and ci_avg_time_flag:
            avg_dims = ['MC', 'time', 'x']
        else:
            avg_dims = ['MC']

        avg_axis = self[store_tmpf + '_MC_set'].get_axis_num(avg_dims)

        self[store_tmpf + '_MC' + store_tempvar] = (
            self[store_tmpf + '_MC_set'] - self[store_tmpf]).std(
                dim=avg_dims)**2

        if ci_avg_time_flag and not ci_avg_x_flag:
            chunks_axis = self[store_tmpf + '_MC_set'].get_axis_num('x')
            new_chunks = ((len(conf_ints),),) + (
                self[store_tmpf + '_MC_set'].chunks[chunks_axis],)
        elif ci_avg_x_flag and not ci_avg_time_flag:
            chunks_axis = self[store_tmpf + '_MC_set'].get_axis_num('time')
            new_chunks = ((len(conf_ints),),) + (
                self[store_tmpf + '_MC_set'].chunks[chunks_axis],)
        elif ci_avg_x_flag and ci_avg_time_flag:
            new_chunks = (len(conf_ints),)
        else:
            new_chunks = (
                (len(conf_ints),),) + self[store_tmpf + '_MC_set'].chunks[1:]

        q = self[store_tmpf + '_MC_set'].data.map_blocks(
            lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
            chunks=new_chunks,  #
            drop_axis=avg_axis,  # avg dimesnions are dropped from input arr
            new_axis=0)  # The new CI dimension is added as first axis

        if ci_avg_time_flag and not ci_avg_x_flag:
            self[store_tmpf + '_MC'] = (('CI', 'x'), q)
        elif ci_avg_x_flag and not ci_avg_time_flag:
            self[store_tmpf + '_MC'] = (('CI', 'time'), q)
        elif ci_avg_x_flag and ci_avg_time_flag:
            self[store_tmpf + '_MC'] = (('CI',), q)
        else:
            self[store_tmpf + '_MC'] = (('CI', 'x', 'time'), q)

        if remove_mc_set_flag:
            drop_var = [
                'gamma_MC', 'dalpha_MC', 'c_MC', 'MC', 'r_st', 'r_ast',
                store_tmpf + '_MC_set']
            for k in drop_var:
                del self[k]

        pass

    def conf_int_double_ended(
            self,
            p_val='p_val',
            p_cov='p_cov',
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
            store_tmpw='TMPW',
            store_tempvar='_var',
            conf_ints=None,
            conf_ints_size=100,
            ci_avg_time_flag=False,
            da_random_state=None,
            remove_mc_set_flag=True):
        """

        Parameters
        ----------
        p_val : array-like or string
            parameter solution directly from calibration_double_ended_wls
        p_cov : array-like or string
            parameter covariance at the solution directly from
            calibration_double_ended_wls
            If set to False, no uncertainty in the parameters is propagated
            into the confidence
            intervals. Similar to the spec sheets of the DTS manufacturers.
            And similar to
            passing an array filled with zeros
        st_label : str
            Key of the forward Stokes
        ast_label : str
            Key of the forward anti-Stokes
        rst_label : str
            Key of the backward Stokes
        rast_label : str
            Key of the backward anti-Stokes
        st_var : float
            Float of the variance of the Stokes signal
        ast_var : float
            Float of the variance of the anti-Stokes signal
        rst_var : float
            Float of the variance of the backward Stokes signal
        rast_var : float
            Float of the variance of the backward anti-Stokes signal
        store_tmpf : str
            Key of how to store the Forward calculated temperature. Is
            calculated using the
            forward Stokes and anti-Stokes observations.
        store_tmpb : str
            Key of how to store the Backward calculated temperature. Is
            calculated using the
            backward Stokes and anti-Stokes observations.
        store_tmpw : str
            Key of how to store the forward-backward-weighted temperature.
            First, the variance of
            TMPF and TMPB are calculated. The Monte Carlo set of TMPF and
            TMPB are averaged,
            weighted by their variance. The median of this set is thought to
            be the a reasonable
            estimate of the temperature
        store_tempvar : str
            a string that is appended to the store_tmp_ keys. and the
            variance is calculated
            for those store_tmp_ keys
        conf_ints : iterable object of float
            A list with the confidence boundaries that are calculated. Valid
            values are between
            [0, 1].
        conf_ints_size : int
            Size of the monte carlo parameter set used to calculate the
            confidence interval
        ci_avg_time_flag : bool
            The confidence intervals differ per time step. If you would like
            to calculate confidence
            intervals of all time steps together. ‘We can say with 95%
            confidence that the
            temperature remained between this line and this line during the
            entire measurement
            period’.
        da_random_state
            For testing purposes. Similar to random seed. The seed for dask.
            Makes random not so random. To produce reproducable results for
            testing environments.
        remove_mc_set_flag : bool
            Remove the monte carlo data set, from which the CI and the
            variance are calculated.

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
        npar = nt + 1 + no  # number of parameters

        self.coords['MC'] = range(conf_ints_size)
        self.coords['CI'] = conf_ints

        assert isinstance(p_val, (str, np.ndarray, np.generic))
        if isinstance(p_val, str):
            p_val = self[p_val].data
        assert p_val.shape == (npar,)

        assert isinstance(p_cov, (str, np.ndarray, np.generic, bool))

        if isinstance(p_cov, bool) and not p_cov:
            gamma = p_val[0]
            d = p_val[1:nt + 1]
            alpha = p_val[nt + 1:]

            self['gamma_MC'] = (tuple(), gamma)
            self['alpha_MC'] = (('x',), alpha)
            self['d_MC'] = (('time',), d)

        elif isinstance(p_cov, bool) and p_cov:
            raise NotImplementedError(
                'Not an implemented option. Check p_cov argument')

        else:
            if isinstance(p_cov, str):
                p_cov = self[p_cov].data
            assert p_cov.shape == (npar, npar)

            p_mc = sst.multivariate_normal.rvs(
                mean=p_val, cov=p_cov, size=conf_ints_size)  # this
            # one takes long

            gamma = p_mc[:, 0]
            d = p_mc[:, 1:nt + 1]
            alpha = p_mc[:, nt + 1:]

            self['gamma_MC'] = (('MC',), gamma)
            self['alpha_MC'] = ((
                'MC',
                'x',
            ), alpha)
            self['d_MC'] = ((
                'MC',
                'time',
            ), d)

        rshape = (self.MC.size, self.x.size, self.time.size)
        r2shape = {0: -1, 1: 'auto', 2: -1}

        for k, st_labeli, st_vari in zip(
            ['r_st', 'r_ast', 'r_rst', 'r_rast'],
            [st_label, ast_label, rst_label, rast_label],
                [st_var, ast_var, rst_var, rast_var]):
            if hasattr(self[st_labeli].data, 'chunks'):
                # if it is a dask array, we don't need to chunk it
                self[k] = (
                    ('MC', 'x', 'time'),
                    state.normal(
                        loc=self[st_labeli].data,
                        scale=st_vari**0.5,
                        size=rshape,
                        chunks=r2shape))
            else:
                # if it is a numpy array (it apparently fits in the memory so
                # we dont need to
                # chunk it)
                self[k] = (
                    ('MC', 'x', 'time'),
                    state.normal(
                        loc=da.from_array(self[st_labeli].data, chunks={}),
                        scale=st_vari**0.5,
                        size=rshape,
                        chunks=r2shape))

        self[store_tmpf + '_MC_set'] = self['gamma_MC'] / (
            np.log(self['r_st'] / self['r_ast']) + self['d_MC'] +
            self['alpha_MC']) - 273.15
        self[store_tmpb + '_MC_set'] = self['gamma_MC'] / (
            np.log(self['r_rst'] / self['r_rast']) + self['d_MC'] -
            self['alpha_MC']) - 273.15

        if ci_avg_time_flag:
            avg_dims = ['MC', 'time']
            ci_dims = ('CI', 'x')
        else:
            avg_dims = ['MC']
            ci_dims = ('CI', 'x', 'time')

        avg_axis = self[store_tmpf + '_MC_set'].get_axis_num(avg_dims)

        self[store_tmpf + '_MC' + store_tempvar] = (
            self[store_tmpf + '_MC_set'] - self[store_tmpf]).std(
                dim=avg_dims)**2
        self[store_tmpb + '_MC' + store_tempvar] = (
            self[store_tmpb + '_MC_set'] - self[store_tmpb]).std(
                dim=avg_dims)**2

        if ci_avg_time_flag:
            new_chunks = ((len(conf_ints),),) + (
                self[store_tmpf + '_MC_set'].chunks[1],)
        else:
            new_chunks = (
                (len(conf_ints),),) + self[store_tmpf + '_MC_set'].chunks[1:]

        q = self[store_tmpf + '_MC_set'].data.map_blocks(
            lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
            chunks=new_chunks,  #
            drop_axis=avg_axis,  # avg dimesnions are dropped from input arr
            new_axis=0)  # The new CI dimension is added as first axis
        self[store_tmpf + '_MC'] = (ci_dims, q)

        q = self[store_tmpb + '_MC_set'].data.map_blocks(
            lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
            chunks=new_chunks,  #
            drop_axis=avg_axis,  # avg dimesnions are dropped from input arr
            new_axis=0)  # The new CI dimension is added as first axis
        self[store_tmpb + '_MC'] = (ci_dims, q)

        # Calculate the weighted MC_set
        weightf = (
            1 / (
                1 / self[store_tmpf + '_MC' + store_tempvar] +
                1 / self[store_tmpb + '_MC' + store_tempvar]) /
            self[store_tmpf + '_MC' + store_tempvar])

        weightb = (
            1 / (
                1 / self[store_tmpf + '_MC' + store_tempvar] +
                1 / self[store_tmpb + '_MC' + store_tempvar]) /
            self[store_tmpb + '_MC' + store_tempvar])

        # np.testing.assert_almost_equal(np.all(weightf + weightb), 1)

        self[store_tmpw + '_MC_set'] = (
            weightf * self[store_tmpf + '_MC_set'] +
            weightb * self[store_tmpb + '_MC_set']).transpose(
                'MC', 'x', 'time')

        # Calculate the CI of the weighted MC_set
        q = self[store_tmpw + '_MC_set'].data.map_blocks(
            lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
            chunks=new_chunks,  #
            drop_axis=avg_axis,  # avg dimensions are dropped from input arr
            new_axis=0)  # The new CI dimension is added as first axis
        self[store_tmpw + '_MC'] = (ci_dims, q)

        # Calculate the mean of the weighted MC_set
        self[store_tmpw] = (
            ('x', 'time'), self[store_tmpw + '_MC_set'].mean(dim='MC').data)

        # Calculate the variance of the weighted MC_set
        self[store_tmpw + '_MC' + store_tempvar] = (
            self[store_tmpw + '_MC_set'] - self[store_tmpw]).std(
                dim=avg_dims)**2

        # Clean up the garbage. All arrays with a Monte Carlo dimension.
        # remove_MC_set = [k for k, v in self.data_vars.items() if 'MC' in
        # v.dims]
        if remove_mc_set_flag:
            remove_MC_set = [
                'r_st', 'r_ast', 'r_rst', 'r_rast', 'gamma_MC', 'alpha_MC',
                'd_MC', 'TMPF_MC_set', 'TMPB_MC_set', 'TMPW_MC_set', 'MC']
            for k in remove_MC_set:
                del self[k]

    def temperature_residuals(self, label=None):
        """

        Parameters
        ----------
        label : str
            The key of the temperature DataArray

        Returns
        -------
        resid_da : xarray.DataArray
            The residuals as DataArray
        """
        resid_temp = self.ufunc_per_section(
            label=label, temp_err=True, calc_per='all')
        resid_x = self.ufunc_per_section(label='x', calc_per='all')

        resid_ix = np.array(
            [np.argmin(np.abs(ai - self.x.data)) for ai in resid_x])

        resid_sorted = np.full(shape=self[label].shape, fill_value=np.nan)
        resid_sorted[resid_ix, :] = resid_temp
        resid_da = xr.DataArray(
            data=resid_sorted,
            dims=('x', 'time'),
            coords={
                'x': self.x,
                'time': self.time})
        return resid_da

    def ufunc_per_section(
            self,
            func=None,
            label=None,
            subtract_from_label=None,
            temp_err=False,
            x_indices=False,
            ref_temp_broadcasted=False,
            calc_per='stretch',
            **func_kwargs):
        """
        User function applied to parts of the cable. Super useful,
        many options and slightly
        complicated.

        The function `func` is taken over all the timesteps and calculated
        per `calc_per`. This
        is returned as a dictionary

        Parameters
        ----------
        func : callable, str
            A numpy function, or lambda function to apple to each 'calc_per'.
        label
        subtract_from_label
        temp_err : bool
            The argument of the function is label minus the reference
            temperature.
        x_indices : bool
            To retreive an integer array with the indices of the
            x-coordinates in the section/stretch
        ref_temp_broadcasted : bool
        calc_per : {'all', 'section', 'stretch'}
        func_kwargs : dict
            Dictionary with options that are passed to func

        TODO: Spend time on creating a slice instead of appendng everything
        to a list and concatenating after.


        Returns
        -------

        Examples
        --------
        # Calculate the variance of the residuals in the along ALL the
        # reference sections wrt the temperature of the water baths
        TMPF_var = d.ufunc_per_section(
            func='var',
            calc_per='all',
            label='TMPF',
            temp_err=True
            )

        # Calculate the variance of the residuals in the along PER
        # reference section wrt the temperature of the water baths
        TMPF_var = d.ufunc_per_section(
            func='var',
            calc_per='stretch',
            label='TMPF',
            temp_err=True
            )

        # Calculate the variance of the residuals in the along PER
        # water bath wrt the temperature of the water baths
        TMPF_var = d.ufunc_per_section(
            func='var',
            calc_per='section',
            label='TMPF',
            temp_err=True
            )

        # Obtain the coordinates of the measurements per section
        locs = d.ufunc_per_section(
            func=None,
            label='x',
            temp_err=False,
            ref_temp_broadcasted=False,
            calc_per='stretch')

        # Number of observations per stretch
        nlocs = d.ufunc_per_section(
            func=len,
            label='x',
            temp_err=False,
            ref_temp_broadcasted=False,
            calc_per='stretch')

        # x-coordinate index
        ix_loc = d.ufunc_per_section(x_indices=True)


        Note
        ----
        If self[label] or self[subtract_from_label] is a Dask array, a Dask
        array is returned
        Else a numpy array is returned
        """

        if not func:

            def func(a):
                """

                Parameters
                ----------
                a

                Returns
                -------

                """
                return a

        elif isinstance(func, str) and func == 'var':

            def func(a):
                """

                Parameters
                ----------
                a

                Returns
                -------

                """
                return np.std(a)**2

        else:
            assert callable(func)

        assert calc_per in ['all', 'section', 'stretch']

        if not x_indices and \
            hasattr(self[label].data, 'chunks') or \
            (subtract_from_label and
             hasattr(self[subtract_from_label].data, 'chunks')):
            concat = da.concatenate
        else:
            concat = np.concatenate

        out = dict()

        for k, section in self.sections.items():
            out[k] = []
            for stretch in section:

                if not x_indices:
                    arg1 = self[label].sel(x=stretch).data

                else:
                    assert not subtract_from_label
                    assert not temp_err
                    assert not ref_temp_broadcasted
                    # so it is slicable with x-indices
                    self['_x_indices'] = self.x.astype(int) * 0 + range(
                        self.x.size)
                    arg1 = self['_x_indices'].sel(x=stretch).data
                    del self['_x_indices']

                if subtract_from_label:
                    # calculate std wrt other series
                    # check_dims(self, [subtract_from_label],
                    #            correct_dims=('x', 'time'))

                    assert not temp_err

                    arg2 = self[subtract_from_label].sel(x=stretch).data
                    out[k].append(arg1 - arg2)

                elif temp_err:
                    # calculate std wrt reference temperature of the
                    # corresponding bath
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
                out[k] = func(concat(out[k]), **func_kwargs)

        if calc_per == 'all':
            out = {k: concat(section) for k, section in out.items()}
            out = func(concat(list(out.values()), axis=0), **func_kwargs)

            if (hasattr(out, 'chunks') and len(out.chunks) > 0 and
                    'x' in self[label].dims):
                # also sum the chunksize in the x dimension
                # first find out where the x dim is
                ixdim = self[label].dims.index('x')
                c_old = out.chunks
                c_new = list(c_old)
                c_new[ixdim] = sum(c_old[ixdim])
                out = out.rechunk(c_new)

        return out


def open_datastore(
        filename_or_obj,
        group=None,
        decode_cf=True,
        mask_and_scale=None,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        engine=None,
        chunks=None,
        lock=None,
        cache=None,
        drop_variables=None,
        backend_kwargs=None,
        **kwargs):
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
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
    decode_coords : bool, optional
        If True, decode the 'coordinates' attribute to identify coordinates in
        the resulting dataset.
    engine : {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'pynio',
    'pseudonetcdf'}, optional
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
        chunks = {'x': 'auto', 'time': -1}

    ds_xr = xr.open_dataset(
        filename_or_obj,
        group=group,
        decode_cf=decode_cf,
        mask_and_scale=mask_and_scale,
        decode_times=decode_times,
        concat_characters=concat_characters,
        decode_coords=decode_coords,
        engine=engine,
        chunks=chunks,
        lock=lock,
        cache=cache,
        drop_variables=drop_variables,
        backend_kwargs=backend_kwargs)

    ds = DataStore(
        data_vars=ds_xr.data_vars,
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
        silent=False,
        load_in_memory='auto',
        **kwargs):
    """Read a folder with measurement files. Each measurement file contains
    values for a
    single timestep. Remember to check which timezone you are working in.

    The silixa files are already timezone aware

    Parameters
    ----------
    filepathlist : list of str, optional
        List of paths that point the the silixa files
    directory : str, Path, optional
        Path to folder
    timezone_netcdf : str, optional
        Timezone string of the netcdf file. UTC follows CF-conventions.
    file_ext : str, optional
        file extension of the measurement files
    silent : bool
        If set tot True, some verbose texts are not printed to stdout/screen
    load_in_memory : {'auto', True, False}
        If 'auto' the Stokes data is only loaded to memory for small files
    kwargs : dict-like, optional
        keyword-arguments are passed to DataStore initialization

    Returns
    -------
    datastore : DataStore
        The newly created datastore.
    """

    assert 'timezone_input_files' not in kwargs, 'The silixa files are ' \
                                                 'already timezone aware'

    if isinstance(filepathlist, type(None)):
        filepathlist = sorted(glob.glob(os.path.join(directory, file_ext)))

        # Make sure that the list of files contains any files
        assert len(
            filepathlist) >= 1, 'No measurement files found in provided ' \
                                'directory: \n' + \
                                str(directory)

    # Make sure that the list of files contains any files
    assert len(
        filepathlist) >= 1, 'No measurement files found in provided ' \
                            'list/directory'

    xml_version = silixa_xml_version_check(filepathlist)

    if xml_version == 4:
        data_vars, coords, attrs = read_silixa_files_routine_v4(
            filepathlist,
            timezone_netcdf=timezone_netcdf,
            silent=silent,
            load_in_memory=load_in_memory)

    elif xml_version == 6:
        data_vars, coords, attrs = read_silixa_files_routine_v6(
            filepathlist,
            timezone_netcdf=timezone_netcdf,
            silent=silent,
            load_in_memory=load_in_memory)

    else:
        raise NotImplementedError(
            'Silixa xml version ' + '{0} not implemented'.format(xml_version))

    ds = DataStore(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)
    return ds


def read_sensornet_files(
        filepathlist=None,
        directory=None,
        file_ext='*.ddf',
        timezone_netcdf='UTC',
        timezone_input_files='UTC',
        silent=False,
        **kwargs):
    """Read a folder with measurement files. Each measurement file contains
    values for a single timestep. Remember to check which timezone
    you are working in.

    Parameters
    ----------
    filepathlist : list of str, optional
        List of paths that point the the silixa files
    directory : str, Path, optional
        Path to folder
    timezone_netcdf : str, optional
        Timezone string of the netcdf file. UTC follows CF-conventions.
    timezone_input_files : str, optional
        Timezone string of the measurement files.
        Remember to check when measurements are taken.
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

    if isinstance(filepathlist, type(None)):
        filepathlist = sorted(glob.glob(os.path.join(directory, file_ext)))

    # Make sure that the list of files contains any files
    assert len(
        filepathlist) >= 1, 'No measurement files found in provided ' \
                            'list/directory'

    data_vars, coords, attrs = read_sensornet_files_routine_v3(
        filepathlist,
        timezone_netcdf=timezone_netcdf,
        timezone_input_files=timezone_input_files,
        silent=silent)

    ds = DataStore(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)
    return ds


def plot_dask(arr, file_path=None):
    """
    For debugging the scheduling of the calculation of dask arrays. Requires
    additional libraries
    to be installed.

    Parameters
    ----------
    arr : Dask-Array
        An uncomputed dask array
    file_path : Path-like, str, optional
        Path to save graph

    Returns
    -------
    out : array-like
        The calculated array

    """
    from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler, \
        visualize

    with Profiler() as prof, ResourceProfiler(
            dt=0.25) as rprof, CacheProfiler() as cprof:
        out = arr.compute()

    if not file_path:
        arr.visualize(file_path)

    visualize([prof, rprof, cprof], show=True)

    return out

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

from dtscalibration.calibrate_utils import wls_sparse
from dtscalibration.datastore_utils import coords_time
from dtscalibration.datastore_utils import grab_data


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

    def variance_stokes(self, st_label, sections=None, use_statsmodels=False, suppress_info=False):
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

        self.check_dims([st_label], correct_dims=('x', 'time'))

        nt = self['time'].size

        len_stretch_list = []
        y_list = []  # intensities of stokes
        x_list = []  # length rel to start of section. for alpha

        for k, stretches in self.sections.items():
            for stretch in stretches:
                y_list.append(self[st_label].sel(x=stretch).data.T.reshape(-1))
                _x = self.x.sel(x=stretch).data
                _x -= _x[0]
                x_list.append(da.tile(_x, nt))
                len_stretch_list.append(_x.size)

        n_sections = len(len_stretch_list)
        n_locs = sum(len_stretch_list)

        x = da.concatenate(x_list)  # coordinates are already in memory
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
        w = y.copy()

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
            C_expand_to_sec = np.hstack([
                np.repeat(a[i * nt + 1:(i + 1) * nt + 1], leni)
                for i, leni in enumerate(len_stretch_list)
                ])
            I_est = np.exp(C_expand_to_sec) * np.exp(x * a[0])
            resid = I_est - y
            var_I = resid.std(ddof=ddof).compute() ** 2

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
            var_I = resid.std(ddof=ddof).compute() ** 2

        return var_I, resid

    def calibration_single_ended(self,
                                 sections=None,
                                 st_label=None,
                                 ast_label=None,
                                 store_c='c',
                                 store_gamma='gamma',
                                 store_alphaint='alphaint',
                                 store_alpha='alpha',
                                 store_IFW_var='IF_var',
                                 store_resid_IFW='errF',
                                 variance_suffix='_var',
                                 method='single1'):

        if sections:
            self.sections = sections

        self.check_dims(['st_label', 'ast_label'])
        pass

    def calibration_double_ended(self,
                                 sections=None,
                                 st_label=None,
                                 ast_label=None,
                                 rst_label=None,
                                 rast_label=None,
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
                                 conf_ints=None,
                                 conf_ints_size=100,
                                 ci_avg_time_flag=False
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
        st_var
        ast_var
        rst_var
        rast_var
        store_c : str, optional
            Label of where to store C
        store_gamma : str, optional
            Label of where to store gamma
        store_alphaint : str, optional
            Label of where to store alphaint
        store_alpha : str, optional
            Label of where to store alpha
        store_tmpf : str, optional
            Label of where to store the calibrated temperature of the forward direction
        store_tmpb : str, optional
            Label of where to store the calibrated temperature of the backward direction
        variance_suffix : str, optional
            String appended for storing the variance. Only used when method is wls.
        method : {'ols', 'wls'}
            Use 'ols' for ordinary least squares and 'wls' for weighted least squares

        Returns
        -------

        """

        if sections:
            self.sections = sections

        self.check_dims([st_label, ast_label, rst_label, rast_label],
                        correct_dims=('x', 'time'))

        if method == 'ols':
            nt, z, p0_ = self.calibration_double_ended_ols(
                st_label, ast_label, rst_label, rast_label)

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
            nt, z, p_sol, p_var, p_cov = self.calibration_double_ended_wls(
                st_label, ast_label, rst_label, rast_label,
                st_var, ast_var, rst_var, rast_var)

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
        tempF_data = gamma / \
            (np.log(self[st_label].data / self[ast_label].data)
             + c + alpha[:, None]) - 273.15
        self[store_tmpf] = (('x', 'time'), tempF_data)

        # deal with BW
        tempB_data = gamma / \
            (np.log(self[rst_label].data / self[rast_label].data)
             + c - alpha[:, None] + alphaint) - 273.15
        self[store_tmpb] = (('x', 'time'), tempB_data)

        if conf_ints:
            assert method == 'wls'
            p_mc = sst.multivariate_normal.rvs(mean=p_sol,
                                               cov=p_cov,
                                               size=conf_ints_size)
            self.coords['MC'] = range(conf_ints_size)
            self.coords['CI'] = conf_ints

            gamma = p_mc[:, 0]
            alphaint = p_mc[:, 1]
            c = p_mc[:, 2:nt + 2]
            alpha = p_mc[:, nt + 2:]

            self[store_gamma + '_MC'] = (('MC',), gamma)
            self[store_alphaint + '_MC'] = (('MC',), alphaint)
            self[store_alpha + '_MC'] = (('MC', 'x',), alpha)
            self[store_c + '_MC'] = (('MC', 'time',), c)

            # deal with FW
            tempF_data = self[store_gamma + '_MC'] / \
                (xr.ufuncs.log(self[st_label] / self[ast_label])
                 + self[store_c + '_MC'] + self[store_alpha + '_MC']) - 273.15
            self[store_tmpf + '_MC'] = (('MC', 'x', 'time'), tempF_data)

            tempB_data = self[store_gamma + '_MC'] / \
                (xr.ufuncs.log(self[rst_label] / self[rast_label])
                 + self[store_c + '_MC'] - self[store_alpha + '_MC'] +
                 self[store_alphaint + '_MC']) - 273.15
            self[store_tmpb + '_MC'] = (('MC', 'x', 'time'), tempB_data)

            if not ci_avg_time_flag:
                q = self[store_tmpf + '_MC'].quantile(conf_ints, dim=['MC', 'time'])
                self[store_tmpf + '_MC'] = (('CI', 'x'), q)

                q = self[store_tmpb + '_MC'].quantile(conf_ints, dim=['MC', 'time'])
                self[store_tmpb + '_MC'] = (('CI', 'x'), q)

            else:
                q = self[store_tmpf + '_MC'].quantile(conf_ints, dim='MC')
                self[store_tmpf + '_MC'] = (('CI', 'x', 'time'), q)

                q = self[store_tmpb + '_MC'].quantile(conf_ints, dim='MC')
                self[store_tmpb + '_MC'] = (('CI', 'x', 'time'), q)

            drop_var = [store_gamma + '_MC',
                        store_alphaint + '_MC',
                        store_alpha + '_MC',
                        store_c + '_MC',
                        'MC']
            for k in drop_var:
                del self[k]

        pass

    def calibration_double_ended_ols(self, st_label, ast_label, rst_label,
                                     rast_label):
        nx = 0
        z_list = []
        cal_dts_list = []
        cal_ref_list = []
        x_index_list = []
        for k, v in self.sections.items():
            for vi in v:
                nx += len(self.x.sel(x=vi))

                z_list.append(self.x.sel(x=vi).data)

                # cut out calibration sections
                cal_dts_list.append(self.sel(x=vi))

                # broadcast point measurements to calibration sections
                ref = xr.full_like(cal_dts_list[-1]['TMP'], 1.) * self[k]

                cal_ref_list.append(ref)

                x_index_list.append(
                    np.where(
                        np.logical_and(self.x > vi.start, self.x < vi.stop))[
                        0])

        z = np.concatenate(z_list)
        x_index = np.concatenate(x_index_list)

        cal_ref = xr.concat(cal_ref_list, dim='x').data

        st = da.concatenate(
            [a[st_label].data
             for a in cal_dts_list], axis=0)
        ast = da.concatenate(
            [a[ast_label].data
             for a in cal_dts_list], axis=0)
        rst = da.concatenate(
            [a[rst_label].data
             for a in cal_dts_list], axis=0)
        rast = da.concatenate(
            [a[rast_label].data
             for a in cal_dts_list], axis=0)

        if hasattr(cal_ref, 'chunks'):
            chunks_dim = (nx, cal_ref.chunks[1])

            for item in [st, ast, rst, rast]:
                item.rechunk(chunks_dim)

        nt = self[st_label].data.shape[1]
        no = self[st_label].data.shape[0]

        p0_est = np.asarray([482., 0.1] + nt * [1.4] + no * [0.])

        # Eqs for F and B temperature
        data1 = np.repeat(1 / (cal_ref.T.ravel() + 273.15), 2)  # gamma
        data2 = np.tile([0., -1.], nt * nx)  # alphaint
        data3 = np.tile([-1., -1.], nt * nx)  # C
        data5 = np.tile([-1., 1.], nt * nx)  # alpha
        # Eqs for alpha
        data6 = np.repeat(-0.5, nt * no)  # alphaint
        data9 = np.ones(nt * no, dtype=float)  # alpha
        data = np.concatenate([data1, data2, data3, data5, data6, data9])

        # (irow, icol)
        coord1row = np.arange(2 * nt * nx, dtype=int)
        coord2row = np.arange(2 * nt * nx, dtype=int)
        coord3row = np.arange(2 * nt * nx, dtype=int)
        coord5row = np.arange(2 * nt * nx, dtype=int)

        coord6row = np.arange(2 * nt * nx, 2 * nt * nx + nt * no, dtype=int)
        coord9row = np.arange(2 * nt * nx, 2 * nt * nx + nt * no, dtype=int)

        coord1col = np.zeros(2 * nt * nx, dtype=int)
        coord2col = np.ones(2 * nt * nx, dtype=int) * (2 + nt + no - 1)
        coord3col = np.repeat(np.arange(nt, dtype=int) + 2, 2 * nx)
        coord5col = np.tile(np.repeat(x_index, 2) + nt + 2, nt)

        coord6col = np.ones(nt * no, dtype=int)
        coord9col = np.tile(np.arange(no, dtype=int) + nt + 2, nt)

        rows = [coord1row, coord2row, coord3row,
                coord5row, coord6row, coord9row]
        cols = [coord1col, coord2col, coord3col,
                coord5col, coord6col, coord9col]
        coords = (np.concatenate(rows), np.concatenate(cols))

        # try scipy.sparse.bsr_matrix
        X = sp.coo_matrix(
            (data, coords),
            shape=(2 * nx * nt + nt * no, nt + 2 + no),
            dtype=float,
            copy=False)

        y1F = da.log(st / ast).T.ravel()
        y1B = da.log(rst / rast).T.ravel()
        y1 = da.stack([y1F, y1B]).T.ravel()
        y2F = np.log(self[st_label].data / self[ast_label].data).T.ravel()
        y2B = np.log(self[rst_label].data / self[rast_label].data).T.ravel()
        y2 = (y2B - y2F) / 2
        y = da.concatenate([y1, y2]).compute()
        # noinspection PyTypeChecker
        p0 = ln.lsqr(X, y, x0=p0_est, show=True, calc_var=True)

        return nt, z, p0

    def calibration_double_ended_wls(self, st_label, ast_label, rst_label,
                                     rast_label, st_var, ast_var, rst_var, rast_var,
                                     calc_cov=True):
        nx = 0
        z_list = []
        cal_dts_list = []
        cal_ref_list = []
        x_index_list = []
        for k, v in self.sections.items():
            for vi in v:
                nx += len(self.x.sel(x=vi))

                z_list.append(self.x.sel(x=vi).data)

                # cut out calibration sections
                cal_dts_list.append(self.sel(x=vi))

                # broadcast point measurements to calibration sections
                ref = xr.full_like(cal_dts_list[-1]['TMP'], 1.) * self[k]

                cal_ref_list.append(ref)

                x_index_list.append(
                    np.where(
                        np.logical_and(self.x > vi.start, self.x < vi.stop))[
                        0])

        z = np.concatenate(z_list)
        x_index = np.concatenate(x_index_list)

        cal_ref = xr.concat(cal_ref_list, dim='x').data

        st = da.concatenate(
            [a[st_label].data
             for a in cal_dts_list], axis=0)
        ast = da.concatenate(
            [a[ast_label].data
             for a in cal_dts_list], axis=0)
        rst = da.concatenate(
            [a[rst_label].data
             for a in cal_dts_list], axis=0)
        rast = da.concatenate(
            [a[rast_label].data
             for a in cal_dts_list], axis=0)

        if hasattr(cal_ref, 'chunks'):
            chunks_dim = (nx, cal_ref.chunks[1])

            for item in [st, ast, rst, rast]:
                # not sure if rechunk happens in-place
                item.rechunk(chunks_dim)

        nt = self[st_label].data.shape[1]
        no = self[st_label].data.shape[0]

        p0_est = np.asarray([482., 0.1] + nt * [1.4] + no * [0.])

        # Data for F and B temperature, nt * nx items
        data1 = np.repeat(1 / (cal_ref.T.ravel() + 273.15), 2)  # gamma
        data2 = np.tile([0., -1.], nt * nx)  # alphaint
        data3 = np.tile([-1., -1.], nt * nx)  # C
        data5 = np.tile([-1., 1.], nt * nx)  # alpha

        # Data for alpha, nt * no items
        data6 = np.repeat(-0.5, nt * no)  # alphaint
        data9 = np.ones(nt * no, dtype=float)  # alpha

        data = np.concatenate([data1, data2, data3, data5, data6, data9])

        # Coords (irow, icol)
        coord1row = np.arange(2 * nt * nx, dtype=int)  # gamma
        coord2row = np.arange(2 * nt * nx, dtype=int)  # alphaint
        coord3row = np.arange(2 * nt * nx, dtype=int)  # C
        coord5row = np.arange(2 * nt * nx, dtype=int)  # alpha

        coord6row = np.arange(2 * nt * nx, 2 * nt * nx + nt * no, dtype=int)  # alphaint
        coord9row = np.arange(2 * nt * nx, 2 * nt * nx + nt * no, dtype=int)  # alpha

        coord1col = np.zeros(2 * nt * nx, dtype=int)  # gamma
        coord2col = np.ones(2 * nt * nx, dtype=int) * (2 + nt + no - 1)  # alphaint
        coord3col = np.repeat(np.arange(nt, dtype=int) + 2, 2 * nx)  # C
        coord5col = np.tile(np.repeat(x_index, 2) + nt + 2, nt)  # alpha

        coord6col = np.ones(nt * no, dtype=int)  # * (2 + nt + no - 1)  # alphaint
        coord9col = np.tile(np.arange(no, dtype=int) + nt + 2, nt)  # alpha

        rows = [coord1row, coord2row, coord3row,
                coord5row, coord6row, coord9row]
        cols = [coord1col, coord2col, coord3col,
                coord5col, coord6col, coord9col]
        coords = (np.concatenate(rows), np.concatenate(cols))

        # try scipy.sparse.bsr_matrix
        X = sp.coo_matrix(
            (data, coords),
            shape=(2 * nx * nt + nt * no, nt + 2 + no),
            dtype=float,
            copy=False)

        # Spooky way to interleave and ravel arrays in correct order. Works!
        y1F = da.log(st / ast).T.ravel()
        y1B = da.log(rst / rast).T.ravel()
        y1 = da.stack([y1F, y1B]).T.ravel()

        y2F = np.log(self[st_label].data /
                     self[ast_label].data).T.ravel()
        y2B = np.log(self[rst_label].data /
                     self[rast_label].data).T.ravel()
        y2 = (y2B - y2F) / 2
        y = da.concatenate([y1, y2]).compute()

        # Calculate the reprocical of the variance (not std)
        w1F = (1 / st ** 2 * st_var +
               1 / ast ** 2 * ast_var
               ).T.ravel()
        w1B = (1 / rst ** 2 * rst_var +
               1 / rast ** 2 * rast_var
               ).T.ravel()
        w1 = da.stack([w1F, w1B]).T.ravel()

        w2 = (0.5 / self[st_label].data ** 2 * st_var +
              0.5 / self[ast_label].data ** 2 * ast_var +
              0.5 / self[rst_label].data ** 2 * rst_var +
              0.5 / self[rast_label].data ** 2 * rast_var
              ).T.ravel()
        w = da.concatenate([w1, w2])

        p_sol, p_var, p_cov = wls_sparse(X, y, w=w, x0=p0_est, calc_cov=calc_cov)

        if calc_cov:
            return nt, z, p_sol, p_var, p_cov
        else:
            return nt, z, p_sol, p_var

    def check_dims(self, labels, correct_dims=None):
        """
        Compare the dimensions of different labels (e.g., 'ST', 'REV-ST').
        If a calculation is performed and the dimensions do not agree, the answers don't make
        sense and the matrices are broadcasted and the memory usage will explode. If no correct
        dims provided the dimensions of the different are compared.

        Parameters
        ----------
        labels : iterable
            An iterable with labels
        correct_dims : tuple of str, optional
            The correct dimensions

        Returns
        -------

        """
        if not correct_dims:
            assert len(labels) > 1

            for li in labels[1:]:
                assert self[labels[0]].dims == self[li].dims
        else:
            for li in labels:
                assert self[li].dims == correct_dims

        pass


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

    array, timearr, meta, extra = grab_data(filepathlist)

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

# filepath = os.path.join('..', '..', 'tests', 'data')
# ds = read_xml_dir(filepath,
#                   timezone_netcdf='UTC',
#                   timezone_ultima_xml='Europe/Amsterdam',
#                   file_ext='*.xml')

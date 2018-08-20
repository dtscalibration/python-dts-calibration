import dask as da
import numpy as np
import scipy.sparse as sp
import xarray as xr
from scipy.sparse import linalg as ln


def wls_sparse(X, y, w=1., calc_cov=False, **kwargs):
    # The var returned by ln.lsqr is normalized by the variance of the error. To
    # obtain the correct variance, it needs to be scaled by the variance of the error.

    # x0=p0_est,

    w_std = np.asarray(np.sqrt(w))
    wy = np.asarray(w_std * y)

    w_std = np.broadcast_to(np.atleast_2d(np.squeeze(w_std)).T, (X.shape[0], 1))

    if not sp.issparse(X):
        wX = w_std * X
    else:
        wX = X.multiply(w_std)

    # noinspection PyTypeChecker
    out_sol = ln.lsqr(wX, wy, show=False, calc_var=True, **kwargs)

    p_sol = out_sol[0]

    # The residual degree of freedom, defined as the number of observations
    # minus the rank of the regressor matrix.
    nobs = len(y)
    rank = np.linalg.matrix_rank(X)
    degrees_of_freedom_err = nobs - rank
    wresid = wy - wX.dot(p_sol)
    err_var = np.dot(wresid, wresid) / degrees_of_freedom_err

    if calc_cov:
        if sp.issparse(wX):
            p_cov = ln.inv(wX.T.dot(wX)).todense() * err_var
            p_var = np.diagonal(p_cov)
        else:
            p_cov = np.linalg.inv(wX.T.dot(wX)) * err_var
            p_var = np.diag(p_cov)
        return p_sol, p_var, p_cov

    else:
        p_var = out_sol[-1] * err_var  # normalized covariance
        return p_sol, p_var


def wls_stats(X, y, w=1., calc_cov=False, **kwargs):
    import statsmodels.api as sm

    if sp.issparse(X):
        X = X.todense()

    mod_wls = sm.WLS(y, X, weights=w)
    res_wls = mod_wls.fit()
    # print(res_wls.summary())

    p_sol = res_wls.params
    p_cov = res_wls.cov_params()
    p_var = res_wls.bse ** 2

    if calc_cov:
        return p_sol, p_var, p_cov
    else:
        return p_sol, p_var


def calibration_double_ended_calc(self, st_label, ast_label, rst_label,
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

    chunks_dim = (nx, cal_ref.chunks[1])
    st = da.concatenate(
        [a[st_label].data
         for a in cal_dts_list], axis=0).rechunk(chunks_dim)
    ast = da.concatenate(
        [a[ast_label].data
         for a in cal_dts_list], axis=0).rechunk(chunks_dim)
    rst = da.concatenate(
        [a[rst_label].data
         for a in cal_dts_list], axis=0).rechunk(chunks_dim)
    rast = da.concatenate(
        [a[rast_label].data
         for a in cal_dts_list], axis=0).rechunk(chunks_dim)

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

    rows = [
        coord1row, coord2row, coord3row, coord5row, coord6row, coord9row
        ]
    cols = [
        coord1col, coord2col, coord3col, coord5col, coord6col, coord9col
        ]
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
    p0 = ln.lsqr(X, y, x0=p0_est, show=True, calc_var=True)
    err = (y - X.dot(p0[0]))  # .reshape((nt, nx)).T  # dims: (nx, nt)
    errFW = err[:2 * nt * nx:2].reshape((nt, nx)).T
    errBW = err[1:2 * nt * nx:2].reshape((nt, nx)).T  # dims: (nx, nt)
    ddof = nt + 2 + no
    var_lsqr = p0[-1] * err.std(ddof=ddof) ** 2
    # var_lsqr = ln.inv(X.T.dot(X)).diagonal() * err.std(ddof=ddof) ** 2

    return nt, z, p0, err, errFW, errBW, var_lsqr

# def generate_synthetic_measurements()

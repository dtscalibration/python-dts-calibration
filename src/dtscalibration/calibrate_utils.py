# coding=utf-8
import dask.array as da
import numpy as np
import scipy.sparse as sp
import scipy.stats as sst
import xarray as xr
from scipy.sparse import linalg as ln


def calibration_single_ended_ols(ds, st_label, ast_label):
    cal_ref = ds.ufunc_per_section(label=st_label,
                                   ref_temp_broadcasted=True,
                                   calc_per='all')

    st = ds.ufunc_per_section(label=st_label, calc_per='all')
    ast = ds.ufunc_per_section(label=ast_label, calc_per='all')
    z = ds.ufunc_per_section(label='x', calc_per='all')

    nx = z.size

    nt = ds[st_label].data.shape[1]

    p0_est = np.asarray([482., 0.1] + nt * [1.4])

    # Eqs for F and B temperature
    data1 = 1 / (cal_ref.T.ravel() + 273.15)  # gamma
    data2 = np.tile(-z, nt)  # dalpha
    data3 = np.tile([-1.], nt * nx)  # C
    data = np.concatenate([data1, data2, data3])

    # (irow, icol)
    coord1row = np.arange(nt * nx, dtype=int)
    coord2row = np.arange(nt * nx, dtype=int)
    coord3row = np.arange(nt * nx, dtype=int)

    coord1col = np.zeros(nt * nx, dtype=int)
    coord2col = np.ones(nt * nx, dtype=int)
    coord3col = np.repeat(np.arange(2, nt + 2, dtype=int), nx)

    rows = [coord1row, coord2row, coord3row]
    cols = [coord1col, coord2col, coord3col]
    coords = (np.concatenate(rows), np.concatenate(cols))

    # try scipy.sparse.bsr_matrix
    X = sp.coo_matrix(
        (data, coords),
        shape=(nt * nx, nt + 2),
        dtype=float,
        copy=False)

    y = da.log(st / ast).T.ravel()
    # noinspection PyTypeChecker
    p0 = ln.lsqr(X, y, x0=p0_est, show=True, calc_var=True)

    return nt, z, p0


def calibration_single_ended_wls(ds, st_label, ast_label, st_var, ast_var,
                                 calc_cov=True, solver='sparse'):
    cal_ref = ds.ufunc_per_section(label=st_label,
                                   ref_temp_broadcasted=True,
                                   calc_per='all')

    st = ds.ufunc_per_section(label=st_label, calc_per='all')
    ast = ds.ufunc_per_section(label=ast_label, calc_per='all')
    z = ds.ufunc_per_section(label='x', calc_per='all')

    nx = z.size

    nt = ds[st_label].data.shape[1]

    p0_est = np.asarray([482., 0.1] + nt * [1.4])

    # Eqs for F and B temperature
    data1 = 1 / (cal_ref.T.ravel() + 273.15)  # gamma
    data2 = np.tile(-z, nt)  # dalpha
    data3 = np.tile([-1.], nt * nx)  # C
    data = np.concatenate([data1, data2, data3])

    # (irow, icol)
    coord1row = np.arange(nt * nx, dtype=int)
    coord2row = np.arange(nt * nx, dtype=int)
    coord3row = np.arange(nt * nx, dtype=int)

    coord1col = np.zeros(nt * nx, dtype=int)
    coord2col = np.ones(nt * nx, dtype=int)
    coord3col = np.repeat(np.arange(2, nt + 2, dtype=int), nx)

    rows = [coord1row, coord2row, coord3row]
    cols = [coord1col, coord2col, coord3col]
    coords = (np.concatenate(rows), np.concatenate(cols))

    # try scipy.sparse.bsr_matrix
    X = sp.coo_matrix(
        (data, coords),
        shape=(nt * nx, nt + 2),
        dtype=float,
        copy=False)

    y = da.log(st / ast).T.ravel()

    w = (1 / st ** 2 * st_var +
         1 / ast ** 2 * ast_var
         ).T.ravel()

    if solver == 'sparse':
        p_sol, p_var, p_cov = wls_sparse(X, y, w=w, x0=p0_est, calc_cov=calc_cov)

    elif solver == 'stats':
        p_sol, p_var, p_cov = wls_stats(X, y, w=w, calc_cov=calc_cov)

    else:
        raise ValueError("Choose a valid solver")

    if calc_cov:
        return nt, z, p_sol, p_var, p_cov
    else:
        return nt, z, p_sol, p_var


def calibration_double_ended_ols(ds, st_label, ast_label, rst_label,
                                 rast_label):
    cal_ref = ds.ufunc_per_section(label=st_label,
                                   ref_temp_broadcasted=True,
                                   calc_per='all')

    st = ds.ufunc_per_section(label=st_label, calc_per='all')
    ast = ds.ufunc_per_section(label=ast_label, calc_per='all')
    rst = ds.ufunc_per_section(label=rst_label, calc_per='all')
    rast = ds.ufunc_per_section(label=rast_label, calc_per='all')
    z = ds.ufunc_per_section(label='x', calc_per='all')

    nx = z.size

    _xsorted = np.argsort(ds.x.data)
    _ypos = np.searchsorted(ds.x.data[_xsorted], z)
    x_index = _xsorted[_ypos]

    # if hasattr(cal_ref, 'chunks'):
    #     chunks_dim = (nx, cal_ref.chunks[1])
    #
    #     for item in [st, ast, rst, rast]:
    #         item.rechunk(chunks_dim)

    nt = ds[st_label].data.shape[1]
    no = ds[st_label].data.shape[0]

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
    y2F = np.log(ds[st_label].data / ds[ast_label].data).T.ravel()
    y2B = np.log(ds[rst_label].data / ds[rast_label].data).T.ravel()
    y2 = (y2B - y2F) / 2
    y = da.concatenate([y1, y2]).compute()
    # noinspection PyTypeChecker
    p0 = ln.lsqr(X, y, x0=p0_est, show=True, calc_var=True)

    return nt, z, p0


def calibration_double_ended_wls(ds, st_label, ast_label, rst_label,
                                 rast_label, st_var, ast_var, rst_var, rast_var,
                                 calc_cov=True, solver='sparse'):
    """


    Parameters
    ----------
    ds : DataStore
    st_label
    ast_label
    rst_label
    rast_label
    st_var
    ast_var
    rst_var
    rast_var
    calc_cov
    solver : {'sparse', 'stats'}

    Returns
    -------

    """

    # x_alpha_set_zero=0.,
    # set one alpha for all times to zero
    # x_alpha_set_zeroi = np.argmin(np.abs(ds.x.data - x_alpha_set_zero))
    # x_alpha_set_zeroidata = np.arange(nt) * no + x_alpha_set_zeroi

    cal_ref = ds.ufunc_per_section(label=st_label,
                                   ref_temp_broadcasted=True,
                                   calc_per='all')

    st = ds.ufunc_per_section(label=st_label, calc_per='all')
    ast = ds.ufunc_per_section(label=ast_label, calc_per='all')
    rst = ds.ufunc_per_section(label=rst_label, calc_per='all')
    rast = ds.ufunc_per_section(label=rast_label, calc_per='all')
    z = ds.ufunc_per_section(label='x', calc_per='all')

    nx = z.size

    _xsorted = np.argsort(ds.x.data)
    _ypos = np.searchsorted(ds.x.data[_xsorted], z)
    x_index = _xsorted[_ypos]

    nt = ds[st_label].data.shape[1]
    no = ds[st_label].data.shape[0]

    p0_est = np.asarray([482., 0.1] + nt * [1.4] + no * [0.])

    # Data for F and B temperature, nt * nx items
    data1 = np.repeat(1 / (cal_ref.T.ravel() + 273.15), 2)  # gamma
    data2 = np.tile([0., -1.], nt * nx)  # alphaint
    data3 = np.tile([-1., -1.], nt * nx)  # C
    data5 = np.tile([-1., 1.], nt * nx)  # alpha

    # Data for alpha, nt * no items
    data6 = np.repeat(-0.5, nt * no)  # alphaint
    data9 = np.ones(nt * no, dtype=float)  # alpha

    # alpha should start at zero. But then the sparse solver crashes
    # data9[x_alpha_set_zeroidata] = 0.

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

    y2F = np.log(ds[st_label].data /
                 ds[ast_label].data).T.ravel()
    y2B = np.log(ds[rst_label].data /
                 ds[rast_label].data).T.ravel()
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

    w2 = (0.5 / ds[st_label].data ** 2 * st_var +
          0.5 / ds[ast_label].data ** 2 * ast_var +
          0.5 / ds[rst_label].data ** 2 * rst_var +
          0.5 / ds[rast_label].data ** 2 * rast_var
          ).T.ravel()
    w = da.concatenate([w1, w2])

    if solver == 'sparse':
        p_sol, p_var, p_cov = wls_sparse(X, y, w=w, x0=p0_est, calc_cov=calc_cov)

    elif solver == 'stats':
        p_sol, p_var, p_cov = wls_stats(X, y, w=w, x0=p0_est, calc_cov=calc_cov)

    if calc_cov:
        return nt, z, p_sol, p_var, p_cov
    else:
        return nt, z, p_sol, p_var


def conf_int_single_ended(ds,
                          p_sol,
                          p_cov,
                          st_label='ST',
                          ast_label='AST',
                          st_var=None,
                          ast_var=None,
                          store_c='c',
                          store_gamma='gamma',
                          store_dalpha='dalpha',
                          store_tmpf='TMPF',
                          store_tempvar=None,
                          conf_ints=None,
                          conf_ints_size=100,
                          ci_avg_time_flag=False,
                          da_random_state=None
                          ):
    nt = ds.time.size

    p_mc = sst.multivariate_normal.rvs(mean=p_sol,
                                       cov=p_cov,
                                       size=conf_ints_size)
    ds.coords['MC'] = range(conf_ints_size)
    ds.coords['CI'] = conf_ints
    gamma = p_mc[:, 0]
    dalpha = p_mc[:, 1]
    c = p_mc[:, 2:nt + 2]
    ds[store_gamma + '_MC'] = (('MC',), gamma)
    ds[store_dalpha + '_MC'] = (('MC',), dalpha)
    ds[store_c + '_MC'] = (('MC', 'time',), c)
    rshape = (ds.MC.size, ds.x.size, ds.time.size)
    if da_random_state:
        state = da_random_state
    else:
        state = da.random.RandomState()
    r_st = state.normal(loc=0, scale=st_var ** 0.5, size=rshape, chunks=rshape)
    r_ast = state.normal(loc=0, scale=ast_var ** 0.5, size=rshape, chunks=rshape)
    ds['r_st'] = (('MC', 'x', 'time'), r_st)
    ds['r_ast'] = (('MC', 'x', 'time'), r_ast)

    tempF_data = ds[store_gamma + '_MC'] / (xr.ufuncs.log(
        (ds[st_label] + ds['r_st']) / (ds[ast_label] + ds['r_ast']))
        + ds[store_c + '_MC'] + (ds.x * ds[store_dalpha + '_MC'])) - 273.15
    ds[store_tmpf + '_MC'] = (('MC', 'x', 'time'), tempF_data)

    if ci_avg_time_flag:
        if store_tempvar:
            ds[store_tmpf + '_MC' + store_tempvar] = (ds[store_tmpf + '_MC'] - ds[
                store_tmpf]).std(dim=['MC', 'time'], ddof=1) ** 2

        q = ds[store_tmpf + '_MC'].quantile(conf_ints, dim=['MC', 'time'])
        ds[store_tmpf + '_MC'] = (('CI', 'x'), q)

    else:
        if store_tempvar:
            ds[store_tmpf + '_MC' + store_tempvar] = (ds[store_tmpf + '_MC'] - ds[
                store_tmpf]).std(dim='MC', ddof=1) ** 2

        q = ds[store_tmpf + '_MC'].quantile(conf_ints, dim='MC')
        ds[store_tmpf + '_MC'] = (('CI', 'x', 'time'), q)
    drop_var = [store_gamma + '_MC',
                store_dalpha + '_MC',
                store_c + '_MC',
                'MC',
                'r_st',
                'r_ast']
    for k in drop_var:
        del ds[k]


def conf_int_double_ended(ds,
                          p_sol,
                          p_cov,
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
                          store_tempvar=None,
                          conf_ints=None,
                          conf_ints_size=100,
                          ci_avg_time_flag=False,
                          da_random_state=None
                          ):
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
    nt = ds.time.size
    p_mc = sst.multivariate_normal.rvs(mean=p_sol,
                                       cov=p_cov,
                                       size=conf_ints_size)
    ds.coords['MC'] = range(conf_ints_size)
    ds.coords['CI'] = conf_ints
    gamma = p_mc[:, 0]
    alphaint = p_mc[:, 1]
    c = p_mc[:, 2:nt + 2]
    alpha = p_mc[:, nt + 2:]
    ds[store_gamma + '_MC'] = (('MC',), gamma)
    ds[store_alphaint + '_MC'] = (('MC',), alphaint)
    ds[store_alpha + '_MC'] = (('MC', 'x',), alpha)
    ds[store_c + '_MC'] = (('MC', 'time',), c)
    rshape = (ds.MC.size, ds.x.size, ds.time.size)

    if da_random_state:
        # In testing environments
        assert isinstance(da_random_state, da.random.RandomState)

        state = da_random_state
    else:
        state = da.random.RandomState()

    r_st = state.normal(loc=0, scale=st_var ** 0.5, size=rshape, chunks=rshape)
    r_ast = state.normal(loc=0, scale=ast_var ** 0.5, size=rshape, chunks=rshape)
    r_rst = state.normal(loc=0, scale=rst_var ** 0.5, size=rshape, chunks=rshape)
    r_rast = state.normal(loc=0, scale=rast_var ** 0.5, size=rshape, chunks=rshape)
    ds['r_st'] = (('MC', 'x', 'time'), r_st)
    ds['r_ast'] = (('MC', 'x', 'time'), r_ast)
    ds['r_rst'] = (('MC', 'x', 'time'), r_rst)
    ds['r_rast'] = (('MC', 'x', 'time'), r_rast)
    # deal with FW
    tempF_data = ds[store_gamma + '_MC'] / (xr.ufuncs.log(
        (ds[st_label] + ds['r_st']) / (ds[ast_label] + ds['r_ast']))
        + ds[store_c + '_MC'] + ds[store_alpha + '_MC']) - 273.15
    ds[store_tmpf + '_MC'] = (('MC', 'x', 'time'), tempF_data)
    tempB_data = ds[store_gamma + '_MC'] / (xr.ufuncs.log(
        (ds[rst_label] + ds['r_rst']) / (ds[rast_label] + ds['r_rast']))
        + ds[store_c + '_MC'] - ds[store_alpha + '_MC'] +
        ds[store_alphaint + '_MC']) - 273.15
    ds[store_tmpb + '_MC'] = (('MC', 'x', 'time'), tempB_data)

    if ci_avg_time_flag:
        if store_tempvar:
            ds[store_tmpf + '_MC' + store_tempvar] = (ds[store_tmpf + '_MC'] - ds[
                store_tmpf]).std(dim=['MC', 'time'], ddof=1) ** 2
            ds[store_tmpb + '_MC' + store_tempvar] = (ds[store_tmpb + '_MC'] - ds[
                store_tmpb]).std(dim=['MC', 'time'], ddof=1) ** 2

        q = ds[store_tmpf + '_MC'].quantile(conf_ints, dim=['MC', 'time'])
        ds[store_tmpf + '_MC'] = (('CI', 'x'), q)
        q = ds[store_tmpb + '_MC'].quantile(conf_ints, dim=['MC', 'time'])
        ds[store_tmpb + '_MC'] = (('CI', 'x'), q)

    else:
        if store_tempvar:
            ds[store_tmpf + '_MC' + store_tempvar] = (ds[store_tmpf + '_MC'] - ds[
                store_tmpf]).std(dim='MC', ddof=1) ** 2
            ds[store_tmpb + '_MC' + store_tempvar] = (ds[store_tmpb + '_MC'] - ds[
                store_tmpb]).std(dim='MC', ddof=1) ** 2

        q = ds[store_tmpf + '_MC'].quantile(conf_ints, dim='MC')
        ds[store_tmpf + '_MC'] = (('CI', 'x', 'time'), q)
        q = ds[store_tmpb + '_MC'].quantile(conf_ints, dim='MC')
        ds[store_tmpb + '_MC'] = (('CI', 'x', 'time'), q)
    drop_var = [store_gamma + '_MC',
                store_alphaint + '_MC',
                store_alpha + '_MC',
                store_c + '_MC',
                'MC',
                'r_st',
                'r_ast',
                'r_rst',
                'r_rast']
    for k in drop_var:
        del ds[k]


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
    npar = X.shape[1]  # ==rank

    degrees_of_freedom_err = nobs - npar
    # wresid = np.exp(wy) - np.exp(wX.dot(p_sol))  # this option is better. difference is small
    wresid = wy - wX.dot(p_sol)  # this option is done by statsmodel
    err_var = np.dot(wresid, wresid) / degrees_of_freedom_err

    if calc_cov:
        arg = wX.T.dot(wX)

        if sp.issparse(arg):
            arg = arg.todense()

        p_cov = np.linalg.inv(arg) * err_var

        p_var = np.diagonal(p_cov)
        return p_sol, p_var, p_cov

    else:
        p_var = out_sol[-1] * err_var  # normalized covariance
        return p_sol, p_var


def wls_stats(X, y, w=1., calc_cov=False):
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

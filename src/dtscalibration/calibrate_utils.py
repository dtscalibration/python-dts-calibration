# coding=utf-8
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as ln


def calibration_single_ended_ols(ds, st_label, ast_label, verbose=False):
    """

    Parameters
    ----------
    ds : DataStore
    st_label : str
    ast_label : str
    verbose : bool

    Returns
    -------

    """
    cal_ref = ds.ufunc_per_section(
        label=st_label, ref_temp_broadcasted=True, calc_per='all')

    ix_sec = ds.ufunc_per_section(x_indices=True, calc_per='all')
    ds_sec = ds.isel(x=ix_sec)

    st = ds_sec[st_label].values
    ast = ds_sec[ast_label].values
    x_sec = ds_sec['x'].values

    assert not np.any(st <= 0.), 'There is uncontrolled noise in the ST signal'
    assert not np.any(
        ast <= 0.), 'There is uncontrolled noise in the AST signal'

    nx_sec = x_sec.size
    nt = ds.time.size

    p0_est = np.asarray([482., 0.1] + nt * [1.4])

    # Eqs for F and B temperature
    data1 = 1 / (cal_ref.T.ravel() + 273.15)  # gamma
    data2 = np.tile(-x_sec, nt)  # dalpha
    data3 = np.tile([-1.], nt * nx_sec)  # C
    data = np.concatenate([data1, data2, data3])

    # (irow, icol)
    coord1row = np.arange(nt * nx_sec, dtype=int)
    coord2row = np.arange(nt * nx_sec, dtype=int)
    coord3row = np.arange(nt * nx_sec, dtype=int)

    coord1col = np.zeros(nt * nx_sec, dtype=int)
    coord2col = np.ones(nt * nx_sec, dtype=int)
    coord3col = np.repeat(np.arange(2, nt + 2, dtype=int), nx_sec)

    rows = [coord1row, coord2row, coord3row]
    cols = [coord1col, coord2col, coord3col]
    coords = (np.concatenate(rows), np.concatenate(cols))

    # try scipy.sparse.bsr_matrix
    X = sp.coo_matrix(
        (data, coords),
        shape=(nt * nx_sec, nt + 2),
        copy=False)

    y = np.log(st / ast).T.ravel()

    # noinspection PyTypeChecker
    p0 = ln.lsqr(X, y, x0=p0_est, show=verbose, calc_var=False)

    return nt, x_sec, p0[0]


def calibration_single_ended_wls(
        ds,
        st_label,
        ast_label,
        st_var,
        ast_var,
        calc_cov=True,
        solver='sparse',
        verbose=False):
    """

    Parameters
    ----------
    ds : DataStore
    st_label
    ast_label
    st_var
    ast_var
    calc_cov : bool
      whether to calculate the covariance matrix. Required for calculation of confidence
      boiundaries. But uses a lot of memory.
    solver : {'sparse', 'stats'}
      Always use sparse to save memory. The statsmodel can be used to validate sparse solver
    verbose : bool

    Returns
    -------

    """
    cal_ref = ds.ufunc_per_section(
        label=st_label, ref_temp_broadcasted=True, calc_per='all')

    ix_sec = ds.ufunc_per_section(x_indices=True, calc_per='all')
    ds_sec = ds.isel(x=ix_sec)

    st = ds_sec[st_label].values
    ast = ds_sec[ast_label].values
    x_sec = ds_sec['x'].values

    assert not np.any(st <= 0.), 'There is uncontrolled noise in the ST signal'
    assert not np.any(
        ast <= 0.), 'There is uncontrolled noise in the AST signal'

    nx_sec = x_sec.size
    nt = ds.time.size

    p0_est = np.asarray([482., 0.1] + nt * [1.4])

    # Eqs for F and B temperature
    data1 = 1 / (cal_ref.T.ravel() + 273.15)  # gamma
    data2 = np.tile(-x_sec, nt)  # dalpha
    data3 = np.tile([-1.], nt * nx_sec)  # C
    data = np.concatenate([data1, data2, data3])

    # (irow, icol)
    coord1row = np.arange(nt * nx_sec, dtype=int)
    coord2row = np.arange(nt * nx_sec, dtype=int)
    coord3row = np.arange(nt * nx_sec, dtype=int)

    coord1col = np.zeros(nt * nx_sec, dtype=int)
    coord2col = np.ones(nt * nx_sec, dtype=int)
    coord3col = np.repeat(np.arange(2, nt + 2, dtype=int), nx_sec)

    rows = [coord1row, coord2row, coord3row]
    cols = [coord1col, coord2col, coord3col]
    coords = (np.concatenate(rows), np.concatenate(cols))

    # try scipy.sparse.bsr_matrix
    X = sp.coo_matrix(
        (data, coords),
        shape=(nt * nx_sec, nt + 2),
        copy=False)

    y = np.log(st / ast).T.ravel()

    w = (1 / st**2 * st_var + 1 / ast**2 * ast_var).T.ravel()

    if solver == 'sparse':
        p_sol, p_var, p_cov = wls_sparse(
            X, y, w=w, x0=p0_est, calc_cov=calc_cov, verbose=verbose)

    elif solver == 'stats':
        p_sol, p_var, p_cov = wls_stats(
            X, y, w=w, calc_cov=calc_cov, verbose=verbose)

    elif solver == 'external':
        return X, y, w, p0_est

    else:
        raise ValueError("Choose a valid solver")

    if calc_cov:
        return nt, x_sec, p_sol, p_var, p_cov
    else:
        return nt, x_sec, p_sol, p_var


def calibration_double_ended_ols(
        ds, st_label, ast_label, rst_label, rast_label, verbose=False):
    """

    Parameters
    ----------
    ds
    st_label
    ast_label
    rst_label
    rast_label
    verbose

    Returns
    -------

    """

    def construct_sparse_X(cal_ref, nt, nx_sec):
        """In function to delete intermediate arrays"""
        # Create sparse X matrix
        XZ_T_val = np.concatenate((
            1 / cal_ref,
            1 / cal_ref), axis=1).flatten()
        XZ_T_row = np.concatenate((
            [np.arange(im * 3 * nt, 2 * nt + im * 3 * nt) for im in
             range(nx_sec)]))
        XZ_T_col = np.zeros_like(XZ_T_val)
        XZ_1_val = -np.ones(2 * nt * nx_sec, dtype=int)
        XZ_1_row = XZ_T_row
        XZ_1_col = np.tile(np.arange(1, nt + 1), 2 * nx_sec)
        XE_val = np.tile(
            np.concatenate((
                -np.ones(nt),
                np.ones(2 * nt))),
            nx_sec)
        XE_row = np.arange(3 * nt * nx_sec)
        XE_col = np.repeat(np.arange(nt + 1, 1 + nt + nx_sec), 3 * nt)
        data = np.concatenate((XZ_T_val, XZ_1_val, XE_val))
        coords = (np.concatenate((XZ_T_row, XZ_1_row, XE_row)),
                  np.concatenate((XZ_T_col, XZ_1_col, XE_col)))
        return coords, data

    no, nt = ds[st_label].data.shape

    ix_sec = ds.ufunc_per_section(x_indices=True, calc_per='all')
    ds_sec = ds.isel(x=ix_sec)

    x_sec = ds_sec['x'].values
    st = ds_sec[st_label].values
    ast = ds_sec[ast_label].values
    rst = ds_sec[rst_label].values
    rast = ds_sec[rast_label].values

    nx_sec = x_sec.size

    assert not np.any(st <= 0.), 'There is uncontrolled noise in the ST signal'
    assert not np.any(
        ast <= 0.), 'There is uncontrolled noise in the AST signal'
    assert not np.any(
        rst <= 0.), 'There is uncontrolled noise in the REV-ST signal'
    assert not np.any(
        rast <= 0.), 'There is uncontrolled noise in the REV-AST signal'

    cal_ref = ds.ufunc_per_section(
        label=st_label, ref_temp_broadcasted=True, calc_per='all') + 273.15

    E, E_var = calc_alpha_double(
        ds,
        st_label,
        ast_label,
        rst_label,
        rast_label)

    E_sec = E.isel(x=ix_sec).values
    p0_est = np.concatenate(([482.], nt * [1.4], E_sec))

    F = np.log(st / ast)
    B = np.log(rst / rast)

    # Construct system of eqns
    y = np.concatenate((F, B, (B - F) / 2), axis=1).flatten()

    coords, data = construct_sparse_X(cal_ref, nt, nx_sec)
    X = sp.coo_matrix(
        (data, coords), shape=(3 * nt * nx_sec, 1 + nt + nx_sec),
        copy=False)

    p0 = ln.lsqr(X, y, x0=p0_est, show=verbose, calc_var=False)
    p_sol = p0[0]

    # put E outside of reference section in solution
    po_sol = np.concatenate((p_sol[:nt + 1], E))
    po_sol[ix_sec + 1 + nt] = p_sol[nt + 1:]

    return nt, x_sec, po_sol


def calibration_double_ended_wls(
        ds,
        st_label,
        ast_label,
        rst_label,
        rast_label,
        st_var,
        ast_var,
        rst_var,
        rast_var,
        calc_cov=True,
        solver='sparse',
        verbose=False):

    def construct_sparse_X(cal_ref, nt, nx_sec):
        """In function to delete intermediate arrays"""
        # Create sparse X matrix
        XZ_T_val = np.concatenate((
            1 / cal_ref,
            1 / cal_ref), axis=1).flatten()
        XZ_T_row = np.concatenate((
            [np.arange(im * 3 * nt, 2 * nt + im * 3 * nt) for im in
             range(nx_sec)]))
        XZ_T_col = np.zeros_like(XZ_T_val)
        XZ_1_val = -np.ones(2 * nt * nx_sec, dtype=int)
        XZ_1_row = XZ_T_row
        XZ_1_col = np.tile(np.arange(1, nt + 1), 2 * nx_sec)
        XE_val = np.tile(
            np.concatenate((
                -np.ones(nt),
                np.ones(2 * nt))),
            nx_sec)
        XE_row = np.arange(3 * nt * nx_sec)
        XE_col = np.repeat(np.arange(nt + 1, 1 + nt + nx_sec), 3 * nt)
        data = np.concatenate((XZ_T_val, XZ_1_val, XE_val))
        coords = (np.concatenate((XZ_T_row, XZ_1_row, XE_row)),
                  np.concatenate((XZ_T_col, XZ_1_col, XE_col)))
        return coords, data

    no, nt = ds[st_label].data.shape

    ix_sec = ds.ufunc_per_section(x_indices=True, calc_per='all')
    ds_sec = ds.isel(x=ix_sec)

    x_sec = ds_sec['x']
    st = ds_sec[st_label].values
    ast = ds_sec[ast_label].values
    rst = ds_sec[rst_label].values
    rast = ds_sec[rast_label].values

    nx_sec = x_sec.size

    assert not np.any(st <= 0.), 'There is uncontrolled noise in the ST signal'
    assert not np.any(
        ast <= 0.), 'There is uncontrolled noise in the AST signal'
    assert not np.any(
        rst <= 0.), 'There is uncontrolled noise in the REV-ST signal'
    assert not np.any(
        rast <= 0.), 'There is uncontrolled noise in the REV-AST signal'

    cal_ref = ds.ufunc_per_section(
        label=st_label, ref_temp_broadcasted=True, calc_per='all') + 273.15

    E, E_var = calc_weighted_alpha_double(
        ds,
        st_label,
        ast_label,
        rst_label,
        rast_label,
        st_var,
        ast_var,
        rst_var,
        rast_var)

    E_sec = E.isel(x=ix_sec).values
    p0_est = np.concatenate(([482.], nt * [1.4], E_sec))

    F = np.log(st / ast)
    B = np.log(rst / rast)

    # Construct system of eqns
    y = np.concatenate((F, B, (B - F) / 2), axis=1).flatten()

    coords, data = construct_sparse_X(cal_ref, nt, nx_sec)
    X = sp.coo_matrix(
        (data, coords), shape=(3 * nt * nx_sec, 1 + nt + nx_sec),
        copy=False)

    # Weights: inversed variance
    i_var_sec_fw = ds_sec.i_var_fw(
        st_var,
        ast_var,
        st_label=st_label,
        ast_label=ast_label)
    i_var_sec_bw = ds_sec.i_var_bw(
        rst_var,
        rast_var,
        rst_label=rst_label,
        rast_label=rast_label)

    w = np.concatenate(
        (1/i_var_sec_fw,
         1/i_var_sec_bw,
         1/((i_var_sec_fw + i_var_sec_bw) / 2)),
        axis=1).flatten()

    if solver == 'sparse':
        p_sol, p_var, p_cov = wls_sparse(
            X, y, w=w, x0=p0_est, calc_cov=calc_cov, verbose=verbose)

    elif solver == 'stats':
        p_sol, p_var, p_cov = wls_stats(
            X, y, w=w, calc_cov=calc_cov, verbose=verbose)

    elif solver == 'external':
        p_sol, p_var, p_cov = None, None, None
        return X, y, w, p0_est

    # put E outside of reference section in solution
    po_sol = np.concatenate((p_sol[:nt + 1], E))
    po_sol[ix_sec + 1 + nt] = p_sol[nt + 1:]

    po_var = np.concatenate((p_var[:nt + 1], E_var))
    po_var[ix_sec + 1 + nt] = p_var[nt + 1:]

    if calc_cov:
        po_cov = np.zeros((1 + nt + no, 1 + nt + no))
        po_cov[nt + 1:, nt + 1:] = np.diag(E_var)

        from_i = np.concatenate((np.arange(nt + 1), nt + 1 + ix_sec))

        iox_sec1, iox_sec2 = np.meshgrid(
            from_i, from_i, indexing='ij')
        po_cov[iox_sec1, iox_sec2] = p_cov

        return nt, x_sec, po_sol, po_var, po_cov

    else:
        return nt, x_sec, po_sol, po_var


def wls_sparse(X, y, w=1., calc_cov=False, verbose=False, **kwargs):
    """

    Parameters
    ----------
    X
    y
    w
    calc_cov
    verbose
    kwargs

    Returns
    -------

    """
    # The var returned by ln.lsqr is normalized by the variance of the error. To
    # obtain the correct variance, it needs to be scaled by the variance of the error.

    w_std = np.asarray(np.sqrt(w))
    wy = np.asarray(w_std * y)

    w_std = np.broadcast_to(
        np.atleast_2d(np.squeeze(w_std)).T, (X.shape[0], 1))

    if not sp.issparse(X):
        wX = w_std * X
    else:
        wX = X.multiply(w_std)

    # noinspection PyTypeChecker
    out_sol = ln.lsqr(wX, wy, show=verbose, calc_var=True, **kwargs)

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

        p_cov = np.array(np.linalg.inv(arg) * err_var)

        p_var = np.diagonal(p_cov)
        return p_sol, p_var, p_cov

    else:
        p_var = out_sol[-1] * err_var  # normalized covariance
        return p_sol, p_var


def wls_stats(X, y, w=1., calc_cov=False, verbose=False):
    """

    Parameters
    ----------
    X
    y
    w
    calc_cov
    verbose

    Returns
    -------

    """
    import statsmodels.api as sm

    y = np.asarray(y)
    w = np.asarray(w)

    if sp.issparse(X):
        X = X.todense()

    mod_wls = sm.WLS(y, X, weights=w)
    res_wls = mod_wls.fit()

    if verbose:
        print(res_wls.summary())

    p_sol = res_wls.params
    p_cov = res_wls.cov_params()
    p_var = res_wls.bse**2

    if calc_cov:
        return p_sol, p_var, p_cov
    else:
        return p_sol, p_var


def calc_alpha_double(
        ds,
        st_label,
        ast_label,
        rst_label,
        rast_label):
    i_fw = np.log(ds[st_label] / ds[ast_label])
    i_bw = np.log(ds[rst_label] / ds[rast_label])

    E = ((i_bw - i_fw) / 2).mean(dim='time')
    E_var = ((i_bw - i_fw) / 2).std(dim='time')**2

    return E, E_var


def calc_weighted_alpha_double(
        ds,
        st_label,
        ast_label,
        rst_label,
        rast_label,
        st_var,
        ast_var,
        rst_var,
        rast_var):

    i_var_fw = ds.i_var_fw(
        st_var,
        ast_var,
        st_label=st_label,
        ast_label=ast_label)
    i_var_bw = ds.i_var_bw(
        rst_var,
        rast_var,
        rst_label=rst_label,
        rast_label=rast_label)

    A_var = (i_var_fw + i_var_bw) / 2

    i_fw = np.log(ds[st_label] / ds[ast_label])
    i_bw = np.log(ds[rst_label] / ds[rast_label])

    A = (i_bw - i_fw) / 2

    E_var = 1 / (1 / A_var).sum(dim='time')
    E = (A / A_var).sum(dim='time') * E_var

    return E, E_var

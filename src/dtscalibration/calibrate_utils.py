# coding=utf-8
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as ln


def calibration_single_ended_solver(
        ds,
        st_label,
        ast_label,
        st_var=None,
        ast_var=None,
        calc_cov=True,
        solver='sparse',
        verbose=False):
    """
    Parameters
    ----------
    ds : DataStore
    st_label : str
    ast_label : str
    st_var : float, array-like, optional
        If `None` use ols calibration. If `float` the variance of the noise
        from the Stokes detector is described with a single value. Or when the
        variance is a function of the intensity (Poisson distributed) define an
        array with shape (nx, nt), where nx are the number of calibration
        locations.
    ast_var : float, array-like, optional
        If `None` use ols calibration. If `float` the variance of the noise
        from the Stokes detector is described with a single value. Or when the
        variance is a function of the intensity (Poisson distributed) define an
        array with shape (nx, nt), where nx are the number of calibration
        locations.
    calc_cov : bool
        whether to calculate the covariance matrix. Required for calculation
        of confidence boundaries. But uses a lot of memory.
    solver : {'sparse', 'stats', 'external', 'external_split'}
        Always use sparse to save memory. The statsmodel can be used to validate
        sparse solver. `external` returns the matrices that would enter the
        matrix solver (Eq.37). `external_split` returns a dictionary with
        matrix X split in the coefficients per parameter. The use case for
        the latter is when certain parameters are fixed/combined.

    verbose : bool

    Returns
    -------

    """
    ix_sec = ds.ufunc_per_section(x_indices=True, calc_per='all')
    ds_sec = ds.isel(x=ix_sec)

    x_sec = ds_sec['x'].values
    nx = x_sec.size

    nt = ds.time.size
    p0_est = np.asarray([485., 0.1] + nt * [1.4])

    # X \gamma  # Eq.34
    cal_ref = ds.ufunc_per_section(
        label=st_label, ref_temp_broadcasted=True, calc_per='all')

    data_gamma = 1 / (cal_ref.ravel() + 273.15)  # gamma
    coord_gamma_row = np.arange(nt * nx, dtype=int)
    coord_gamma_col = np.zeros(nt * nx, dtype=int)
    X_gamma = sp.coo_matrix(
        (data_gamma, (coord_gamma_row, coord_gamma_col)),
        shape=(nt * nx, 1),
        copy=False)

    # X \Delta\alpha  # Eq.34
    data_dalpha = np.repeat(-x_sec, nt)  # dalpha
    coord_dalpha_row = np.arange(nt * nx, dtype=int)
    coord_dalpha_col = np.zeros(nt * nx, dtype=int)
    X_dalpha = sp.coo_matrix(
        (data_dalpha, (coord_dalpha_row, coord_dalpha_col)),
        shape=(nt * nx, 1),
        copy=False)

    # X C  # Eq.34
    data_c = -np.ones(nt * nx, dtype=int)
    coord_c_row = np.arange(nt * nx, dtype=int)
    coord_c_col = np.tile(np.arange(nt, dtype=int), nx)
    X_c = sp.coo_matrix(
        (data_c, (coord_c_row, coord_c_col)),
        shape=(nt * nx, nt),
        copy=False)

    # Stack all X's
    X = sp.hstack((X_gamma, X_dalpha, X_c))

    # y
    y = np.log(ds_sec[st_label] / ds_sec[ast_label]).values.ravel()

    # w
    if st_var is not None:
        w = (1 / ds_sec[st_label] ** 2 * st_var +
             1 / ds_sec[ast_label] ** 2 * ast_var).values.ravel()

    else:
        w = 1.  # unweighted

    if solver == 'sparse':
        if calc_cov:
            p_sol, p_var, p_cov = wls_sparse(
                X, y, w=w, x0=p0_est, calc_cov=calc_cov, verbose=verbose)
        else:
            p_sol, p_var = wls_sparse(
                X, y, w=w, x0=p0_est, calc_cov=calc_cov, verbose=verbose)

    elif solver == 'stats':
        if calc_cov:
            p_sol, p_var, p_cov = wls_stats(
                X, y, w=w, calc_cov=calc_cov, verbose=verbose)
        else:
            p_sol, p_var = wls_stats(
                X, y, w=w, calc_cov=calc_cov, verbose=verbose)

    elif solver == 'external':
        return X, y, w, p0_est

    elif solver == 'external_split':
        return dict(
            y=y,
            w=w,
            X_gamma=X_gamma,
            X_dalpha=X_dalpha,
            X_c=X_c,
            p0_est=p0_est)

    else:
        raise ValueError("Choose a valid solver")

    if calc_cov:
        return p_sol, p_var, p_cov
    else:
        return p_sol, p_var


def calibration_double_ended_solver(
        ds,
        st_label,
        ast_label,
        rst_label,
        rast_label,
        st_var=None,
        ast_var=None,
        rst_var=None,
        rast_var=None,
        calc_cov=True,
        solver='sparse',
        verbose=False):
    """
    Parameters
    ----------
    ds : DataStore
    st_label : str
    ast_label : str
    rst_label : str
    rast_label : str
    st_var : float, array-like, optional
        If `None` use ols calibration. If `float` the variance of the noise
        from the Stokes detector is described with a single value. Or when the
        variance is a function of the intensity (Poisson distributed) define an
        array with shape (nx, nt), where nx are the number of calibration
        locations.
    ast_var : float, array-like, optional
        If `None` use ols calibration. If `float` the variance of the noise
        from the Stokes detector is described with a single value. Or when the
        variance is a function of the intensity (Poisson distributed) define an
        array with shape (nx, nt), where nx are the number of calibration
        locations.
    rst_var : float, array-like, optional
        If `None` use ols calibration. If `float` the variance of the noise
        from the Stokes detector is described with a single value. Or when the
        variance is a function of the intensity (Poisson distributed) define an
        array with shape (nx, nt), where nx are the number of calibration
        locations.
    rast_var : float, array-like, optional
        If `None` use ols calibration. If `float` the variance of the noise
        from the Stokes detector is described with a single value. Or when the
        variance is a function of the intensity (Poisson distributed) define an
        array with shape (nx, nt), where nx are the number of calibration
        locations.
    calc_cov : bool
        whether to calculate the covariance matrix. Required for calculation
        of confidence boundaries. But uses a lot of memory.
    solver : {'sparse', 'stats', 'external', 'external_split'}
        Always use sparse to save memory. The statsmodel can be used to validate
        sparse solver. `external` returns the matrices that would enter the
        matrix solver (Eq.37). `external_split` returns a dictionary with
        matrix X split in the coefficients per parameter. The use case for
        the latter is when certain parameters are fixed/combined.

    verbose : bool

    Returns
    -------

    """

    def construct_submatrices(nt, nx, st_label, ds):
        """Wrapped in a function to reduce memory usage"""

        # Z \gamma  # Eq.47
        cal_ref = np.array(ds.ufunc_per_section(
            label=st_label, ref_temp_broadcasted=True, calc_per='all'))
        data_gamma = 1 / (cal_ref.ravel() + 273.15)  # gamma
        coord_gamma_row = np.arange(nt * nx, dtype=int)
        coord_gamma_col = np.zeros(nt * nx, dtype=int)
        Z_gamma = sp.coo_matrix(
            (data_gamma, (coord_gamma_row, coord_gamma_col)),
            shape=(nt * nx, 1),
            copy=False)
        # Z D  # Eq.47
        data_c = -np.ones(nt * nx, dtype=int)
        coord_c_row = np.arange(nt * nx, dtype=int)
        coord_c_col = np.tile(np.arange(nt, dtype=int), nx)
        Z_d = sp.coo_matrix(
            (data_c, (coord_c_row, coord_c_col)),
            shape=(nt * nx, nt),
            copy=False)
        # E  # Eq.47
        data_c = np.ones(nt * nx, dtype=int)
        coord_c_row = np.arange(nt * nx, dtype=int)
        coord_c_col = np.repeat(np.arange(nx, dtype=int), nt)
        E = sp.coo_matrix(
            (data_c, (coord_c_row, coord_c_col)),
            shape=(nt * nx, nx),
            copy=False)
        # Zero  # Eq.45
        Zero_gamma = sp.coo_matrix(
            ([], ([], [])), shape=(nt * nx, 1))
        Zero_d = sp.coo_matrix(
            ([], ([], [])), shape=(nt * nx, nt))
        return E, Z_d, Z_gamma, Zero_d, Zero_gamma

    ix_sec = ds.ufunc_per_section(x_indices=True, calc_per='all')
    ds_sec = ds.isel(x=ix_sec)

    x_sec = ds_sec['x'].values
    nx = x_sec.size
    nt = ds.time.size

    # Calculate E for outside of calibration sections and as initial estimate
    # for the E calibration.
    E_all, E_all_var = calc_alpha_double(
        ds,
        st_label,
        ast_label,
        rst_label,
        rast_label,
        st_var,
        ast_var,
        rst_var,
        rast_var)

    p0_est = np.concatenate((np.asarray([485.] + nt * [1.4]), E_all[ix_sec]))

    E, Z_d, Z_gamma, Zero_d, Zero_gamma = construct_submatrices(
        nt, nx, st_label, ds)

    # Stack all X's
    X = sp.vstack(
        (sp.hstack((Z_gamma, Z_d, -E)),
         sp.hstack((Z_gamma, Z_d, E)),
         sp.hstack((Zero_gamma, Zero_d, E))))

    # y  # Eq.41--45
    y_F = np.log(ds_sec[st_label] / ds_sec[ast_label]).values.ravel()
    y_B = np.log(ds_sec[rst_label] / ds_sec[rast_label]).values.ravel()

    y = np.concatenate((y_F, y_B, (y_B - y_F) / 2))

    # w
    if st_var is not None:
        w_F = (1 / ds_sec[st_label] ** 2 * st_var +
               1 / ds_sec[ast_label] ** 2 * ast_var).values.ravel()
        w_B = (1 / ds_sec[rst_label] ** 2 * rst_var +
               1 / ds_sec[rast_label] ** 2 * rast_var).values.ravel()

        w = np.concatenate((w_F, w_B, (w_B + w_F) / 2))

    else:
        w = 1.  # unweighted

    if solver == 'sparse':
        if calc_cov:
            p_sol, p_var, p_cov = wls_sparse(
                X, y, w=w, x0=p0_est, calc_cov=calc_cov, verbose=verbose)
        else:
            p_sol, p_var = wls_sparse(
                X, y, w=w, x0=p0_est, calc_cov=calc_cov, verbose=verbose)

    elif solver == 'stats':
        if calc_cov:
            p_sol, p_var, p_cov = wls_stats(
                X, y, w=w, calc_cov=calc_cov, verbose=verbose)
        else:
            p_sol, p_var = wls_stats(
                X, y, w=w, calc_cov=calc_cov, verbose=verbose)

    elif solver == 'external':
        return X, y, w, p0_est

    elif solver == 'external_split':
        return dict(
            y_F=y_F,
            y_B=y_B,
            w_F=w_F,
            w_B=w_B,
            Z_gamma=Z_gamma,
            Z_d=Z_d,
            E=E,
            Zero_gamma=Zero_gamma,
            Zero_d=Zero_d,
            p0_est=p0_est)

    else:
        raise ValueError("Choose a valid solver")

    # put E outside of reference section in solution
    # concatenating makes a copy of the data instead of using a pointer
    po_sol = np.concatenate((p_sol[:nt + 1], E_all))
    po_sol[ix_sec + 1 + nt] = p_sol[nt + 1:]

    po_var = np.concatenate((p_var[:nt + 1], E_all_var))
    po_var[ix_sec + 1 + nt] = p_var[nt + 1:]

    if calc_cov:
        po_cov = np.zeros((po_sol.size, po_sol.size))
        po_cov[nt + 1:, nt + 1:] = np.diag(E_all_var)

        from_i = np.concatenate((np.arange(nt + 1), nt + 1 + ix_sec))

        iox_sec1, iox_sec2 = np.meshgrid(
            from_i, from_i, indexing='ij')
        po_cov[iox_sec1, iox_sec2] = p_cov

        return po_sol, po_var, po_cov

    else:
        return po_sol, po_var


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

    if w is None:  # gracefully default to unweighted
        w = 1.

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
    # wresid = np.exp(wy) - np.exp(wX.dot(p_sol))  # this option is better.
    # difference is small
    wresid = wy - wX.dot(p_sol)  # this option is done by statsmodel
    err_var = np.dot(wresid, wresid) / degrees_of_freedom_err

    if calc_cov:
        arg = wX.T.dot(wX)

        if sp.issparse(arg):
            # arg is square of size double: 1 + nt + no; single: 2 : nt
            arg_inv = np.linalg.inv(arg.todense())
        else:
            arg_inv = np.linalg.inv(arg)

        # for tall systems pinv (approximate) is recommended above inv
        # https://vene.ro/blog/inverses-pseudoinverses-numerical-issues-spee
        # d-symmetry.html
        # but better to solve with eye
        # p_cov = np.array(np.linalg.pinv(arg) * err_var)
        # arg_inv = np.linalg.pinv(arg)
        # else:
        #     try:
        #         arg_inv = np.linalg.lstsq(arg, np.eye(nobs), rcond=None)[0]
        #
        #     except MemoryError:
        #         print('Try calc_cov = False and p_cov = np.diag(p_var); '
        #               'And neglect the covariances.')
        #         arg_inv = np.linalg.lstsq(arg, np.eye(nobs), rcond=None)[0]

        p_cov = np.array(arg_inv * err_var)

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
        rast_label,
        st_var=None,
        ast_var=None,
        rst_var=None,
        rast_var=None):
    """Eq.50 if weighted least squares"""
    time_dim = ds.get_time_dim()

    if st_var is not None:
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

        E_var = 1 / (1 / A_var).sum(dim=time_dim)
        E = (A / A_var).sum(dim=time_dim) * E_var

    else:
        i_fw = np.log(ds[st_label] / ds[ast_label])
        i_bw = np.log(ds[rst_label] / ds[rast_label])

        A = (i_bw - i_fw) / 2

        E_var = A.var(dim=time_dim)
        E = A.mean(dim=time_dim)

    return E, E_var

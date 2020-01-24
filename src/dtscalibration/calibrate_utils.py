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
        w = 1 / (
            ds_sec[st_label] ** -2 * st_var +
            ds_sec[ast_label] ** -2 * ast_var).values.ravel()

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
        matching_indices=None,
        transient_asym_att_x=None,
        verbose=False):
    """
    The construction of X differs a bit from what is presented in the
    article. The choice to divert from the article is made because
    then remaining modular is easier.
    Eq34 and Eq43 become:
    y = [F, B, (B-F)/2], F=[F_0, F_1, .., F_M], B=[B_0, B_1, .., B_M],
    where F_m and B_m contain the coefficients for all times.

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
    matching_indices : array-like
        Is an array of size (np, 2), where np is the number of paired
        locations. This array is produced by `matching_sections()`.
    transient_asym_att_x : iterable, optional
        Connectors cause assymetrical attenuation. Normal double ended
        calibration assumes symmetrical attenuation. An additional loss
        term is added in the 'shadow' of the forward and backward
        measurements. This loss term varies over time. Provide a list
        containing the x locations of the connectors along the fiber.
        Each location introduces an additional 2*nt parameters to solve
        for. Requiering either an additional calibration section or
        matching sections. If multiple locations are defined, the losses are
        added.
    verbose : bool

    Returns
    -------

    """
    def construct_submatrices(nt, nx, st_label, ds, transient_asym_att_x, x_sec):
        """Wrapped in a function to reduce memory usage.
        Constructing:
        Z_gamma (nt * nx, 1). Data: positive 1/temp
        Z_D (nt * nx, nt). Data: ones
        E (nt * nx, nx). Data: ones
        Zero_gamma (nt * nx, 1)
        zero_d (nt * nx, nt)
        Z_TA_fw (nt * nx, nta * 2 * nt) minus ones
        Z_TA_bw (nt * nx, nta * 2 * nt) minus ones
        Z_TA_E (nt * nx, nta * 2 * nt)

        I_fw = 1/Tref*gamma - D_fw - E - TA_fw
        I_bw = 1/Tref*gamma - D_bw + E - TA_bw
        (I_bw - I_fw) / 2 = D_fw/2 - D_bw/2 + E + TA_fw/2 - TA_bw/2 Eq42
        """

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
        data_c = np.ones(nt * nx, dtype=float)
        coord_c_row = np.arange(nt * nx, dtype=int)
        coord_c_col = np.tile(np.arange(nt, dtype=int), nx)
        Z_D = sp.coo_matrix(
            (data_c, (coord_c_row, coord_c_col)),
            shape=(nt * nx, nt),
            copy=False)
        Z_D_att = sp.eye(nt, format='coo')
        # E  # Eq.47
        data_c = np.ones(nt * nx, dtype=float)
        coord_c_row = np.arange(nt * nx, dtype=int)
        coord_c_col = np.repeat(np.arange(nx, dtype=int), nt)
        E = sp.coo_matrix(
            (data_c, (coord_c_row, coord_c_col)),
            shape=(nt * nx, nx),
            copy=False)
        # Zero  # Eq.45
        Zero_gamma = sp.coo_matrix(([], ([], [])), shape=(nt * nx, 1))
        Zero_d = sp.coo_matrix(([], ([], [])), shape=(nt * nx, nt))
        Zero_E = sp.coo_matrix(([], ([], [])), shape=(nt * nx, nx))
        Zero_gamma_att = sp.coo_matrix(([], ([], [])), shape=(nt, 1))
        Zero_E_att = sp.coo_matrix(([], ([], [])), shape=(nt, nx))
        if transient_asym_att_x:
            # unpublished BdT

            TA_fw_list = list()
            TA_bw_list = list()

            for transient_asym_att_xi in transient_asym_att_x:
                """For forward direction. """
                # first index on the right hand side a the difficult splice
                # Deal with connector outside of fiber
                if transient_asym_att_xi >= x_sec[-1]:
                    ix_sec_ta_ix0 = nx
                elif transient_asym_att_xi <= x_sec[0]:
                    ix_sec_ta_ix0 = 0
                else:
                    ix_sec_ta_ix0 = np.flatnonzero(
                        x_sec >= transient_asym_att_xi)[0]

                # Data is -1 for both forward and backward
                # I_fw = 1/Tref*gamma - D_fw - E - TA_fw. Eq40
                data_ta_fw = -np.ones(nt * (nx - ix_sec_ta_ix0), dtype=float)
                # skip ix_sec_ta_ix0 locations, because they are upstream of
                # the connector.
                coord_ta_fw_row = np.arange(
                    nt * ix_sec_ta_ix0, nt * nx, dtype=int)
                # nt parameters
                coord_ta_fw_col = np.tile(
                    np.arange(nt, dtype=int), nx - ix_sec_ta_ix0)
                TA_fw_list.append(sp.coo_matrix(  # TA_fw
                        (data_ta_fw, (coord_ta_fw_row, coord_ta_fw_col)),
                        shape=(nt * nx, 2 * nt),
                        copy=False))

                # I_bw = 1/Tref*gamma - D_bw + E - TA_bw. Eq41
                data_ta_bw = -np.ones(nt * ix_sec_ta_ix0, dtype=float)
                coord_ta_bw_row = np.arange(nt * ix_sec_ta_ix0, dtype=int)
                coord_ta_bw_col = np.tile(np.arange(nt, 2 * nt, dtype=int),
                                          ix_sec_ta_ix0)
                TA_bw_list.append(sp.coo_matrix(  # TA_bw
                        (data_ta_bw, (coord_ta_bw_row, coord_ta_bw_col)),
                        shape=(nt * nx, 2 * nt),
                        copy=False))
            Z_TA_fw = sp.hstack(TA_fw_list)
            Z_TA_bw = sp.hstack(TA_bw_list)

        else:
            Z_TA_fw = sp.coo_matrix(([], ([], [])), shape=(nt * nx, 0))
            Z_TA_bw = sp.coo_matrix(([], ([], [])), shape=(nt * nx, 0))

            Z_TA_att = sp.coo_matrix(([], ([], [])), shape=(nt, 0))

        # (I_bw - I_fw) / 2 = D_fw/2 - D_bw/2 + E + TA_fw/2 - TA_bw/2 Eq42
        Z_TA_E = (Z_TA_bw - Z_TA_fw) / 2

        return E, Z_D, Z_gamma, Zero_d, Zero_gamma, Z_TA_fw, Z_TA_bw, Z_TA_E,\
            Zero_E, Z_TA_att, Z_D_att, Zero_gamma_att, Zero_E_att

    ix_sec = ds.ufunc_per_section(x_indices=True, calc_per='all')
    ds_sec = ds.isel(x=ix_sec)

    x_sec = ds_sec['x'].values
    nx = x_sec.size
    nt = ds.time.size
    nta = len(transient_asym_att_x) if transient_asym_att_x else 0

    # Calculate E as initial estimate for the E calibration.
    # Does not require ta to be passed on
    E_all_guess, E_all_var_guess = calc_alpha_double(
        'guess',
        ds,
        st_label,
        ast_label,
        rst_label,
        rast_label,
        st_var,
        ast_var,
        rst_var,
        rast_var)

    p0_est = np.concatenate((np.asarray([485.] + 2 * nt * [1.4]),
                             E_all_guess[ix_sec], nta * nt * 2 * [0.]))

    E, Z_D, Z_gamma, Zero_d, Zero_gamma, Z_TA_fw, Z_TA_bw, Z_TA_E, Zero_E, \
        Z_TA_att, Z_D_att, Zero_gamma_att, Zero_E_att = construct_submatrices(
            nt, nx, st_label, ds, transient_asym_att_x, x_sec)

    # if matching_indices is not None:
    #     # The matching indices are location indices along the entire fiber.
    #     # This calibration routine deals only with measurements along the
    #     # reference sections. The coefficient matrix X is build around
    #     # ds_sec = ds.isel(x=ix_sec). X needs to become larger.
    #     # Therefore, the matching indices are first gathered for the reference
    #     # sections, after which those for outside the reference sections.
    #     # Double-ended setups mostly benefit from matching sections if there is
    #     # asymetrical attenuation, e.g., due to connectors.
    #
    #     # select the indices in refence sections
    #     hix = np.array(list(filter(
    #         lambda x: x in ix_sec, matching_indices[:, 0])))
    #     tix = np.array(list(filter(
    #         lambda x: x in ix_sec, matching_indices[:, 1])))
    #
    #     npair = hix.size
    #
    #     assert hix.size == tix.size, 'Both locations of a matching pair ' \
    #                                  'should either be used in a calibration ' \
    #                                  'section or outside the calibration ' \
    #                                  'sections'
    #     assert hix.size > 0, 'no matching sections in calibration'
    #
    #     # Convert to indices along reference sections. To index ds_sec.
    #     ixglob_to_ix_sec = lambda x: np.where(ix_sec == x)[0]
    #
    #     hix_sec = np.concatenate([ixglob_to_ix_sec(x) for x in hix])
    #     tix_sec = np.concatenate([ixglob_to_ix_sec(x) for x in tix])
    #
    #     y_mF = (F[hix_sec] + F[tix_sec]).flatten()
    #     y_mB = (B[hix_sec] + B[tix_sec]).flatten()

    # Stack all X's
    X = sp.vstack(
        (sp.hstack((Z_gamma, -Z_D, Zero_d, -E, Z_TA_fw)),
         sp.hstack((Z_gamma, Zero_d, -Z_D, E, Z_TA_bw)),
         sp.hstack((Zero_gamma, Z_D / 2, -Z_D / 2, E, Z_TA_E)),
         sp.hstack((Zero_gamma_att, Z_D_att / 2, -Z_D_att / 2, Zero_E_att,
                    Z_TA_att))))

    # y  # Eq.41--45
    y_F_ = np.log(ds_sec[st_label] / ds_sec[ast_label])
    y_B_ = np.log(ds_sec[rst_label] / ds_sec[rast_label])
    y_F = y_F_.values.ravel()
    y_B = y_B_.values.ravel()

    y_att_F0 = np.log(ds[st_label] /
                      ds[ast_label]).isel(x=0)
    y_att_FL = np.log(ds[st_label] /
                      ds[ast_label]).isel(x=-1)
    y_att_B0 = np.log(ds[rst_label] /
                      ds[rast_label]).isel(x=0)
    y_att_BL = np.log(ds[rst_label] /
                      ds[rast_label]).isel(x=-1)

    y_att1 = ((y_B_ - y_F_) / 2).values.ravel()
    y_att2 = -((y_att_F0 + y_att_FL - y_att_B0 - y_att_BL) / 4).values

    y = np.concatenate((y_F, y_B, y_att1, y_att2))

    # w
    if st_var is not None:  # WLS
        w_F = 1 / (
            ds_sec[st_label] ** -2 * st_var +
            ds_sec[ast_label] ** -2 * ast_var).values.ravel()
        w_B = 1 / (
            ds_sec[rst_label] ** -2 * rst_var +
            ds_sec[rast_label] ** -2 * rast_var).values.ravel()
        w_att1 = 1 / (
            ds_sec[st_label] ** -2 * st_var / 2 +
            ds_sec[ast_label] ** -2 * ast_var / 2 +
            ds_sec[rst_label] ** -2 * rst_var / 2 +
            ds_sec[rast_label] ** -2 * rast_var / 2).values.ravel()
        w_att2 = 1 / (
            ds[st_label].isel(x=0) ** -2 * st_var / 2 +
            ds[ast_label].isel(x=0) ** -2 * ast_var / 2 +
            ds[rst_label].isel(x=0) ** -2 * rst_var / 2 +
            ds[rast_label].isel(x=0) ** -2 * rast_var / 2 +
            ds[st_label].isel(x=-1) ** -2 * st_var / 2 +
            ds[ast_label].isel(x=-1) ** -2 * ast_var / 2 +
            ds[rst_label].isel(x=-1) ** -2 * rst_var / 2 +
            ds[rast_label].isel(x=-1) ** -2 * rast_var / 2).values

    else:  # OLS
        w_F = np.ones(nt * nx)
        w_B = np.ones(nt * nx)
        w_att1 = np.ones(nt * nx)
        w_att2 = np.ones(nt)

    w = np.concatenate((w_F, w_B, w_att1, w_att2))

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
            y_att1=y_att1,
            y_att2=y_att2,
            w_F=w_F,
            w_B=w_B,
            w_att1=w_att1,
            w_att2=w_att2,
            Z_gamma=Z_gamma,
            Zero_gamma=Zero_gamma,
            Zero_gamma_att=Zero_gamma_att,
            Z_D=Z_D,
            Zero_d=Zero_d,
            Z_D_att=Z_D_att,
            E=E,
            Zero_E_att=Zero_E_att,
            Z_TA_fw=Z_TA_fw,
            Z_TA_bw=Z_TA_bw,
            Z_TA_E=Z_TA_E,
            Z_TA_att=Z_TA_att,
            p0_est=p0_est,
            E_all_guess=E_all_guess,
            E_all_var_guess=E_all_var_guess)

    else:
        raise ValueError("Choose a valid solver")

    # p_sol contains the int diff att of all the locations within the
    # reference sections. po_sol is its expanded version that contains also
    # the int diff att for outside the reference sections.

    # put E outside of reference section in solution
    # concatenating makes a copy of the data instead of using a pointer
    ds_sub = ds[[st_label, ast_label, rst_label, rast_label]]
    ds_sub['df'] = (('time',), p_sol[1:1 + nt])
    ds_sub['df_var'] = (('time',), p_var[1:1 + nt])
    ds_sub['db'] = (('time',), p_sol[1 + nt:1 + 2 * nt])
    ds_sub['db_var'] = (('time',), p_var[1 + nt:1 + 2 * nt])
    E_all_exact, E_all_var_exact = calc_alpha_double(
        'exact',
        ds_sub,
        st_label,
        ast_label,
        rst_label,
        rast_label,
        st_var,
        ast_var,
        rst_var,
        rast_var,
        'df',
        'db',
        'df_var',
        'db_var')
    po_sol = np.concatenate((p_sol[:1 + 2 * nt],
                             E_all_exact,
                             p_sol[1 + 2 * nt + nx:]))
    po_sol[1 + 2 * nt + ix_sec] = p_sol[1 + 2 * nt:1 + 2 * nt + nx]

    po_var = np.concatenate((p_var[:1 + 2 * nt],
                             E_all_var_exact,
                             p_var[1 + 2 * nt + nx:]))
    po_var[1 + 2 * nt + ix_sec] = p_var[1 + 2 * nt:1 + 2 * nt + nx]

    if calc_cov:
        # the COV can be expensive to compute (in the least squares routine)
        po_cov = np.diag(po_var).copy()

        from_i = np.concatenate((np.arange(1 + 2 * nt),
                                 1 + 2 * nt + ix_sec,
                                 np.arange(1 + 2 * nt + nx,
                                           1 + 2 * nt + nx + nta * nt * 2)))

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
        # assert np.any()
        arg = wX.T.dot(wX)

        if sp.issparse(arg):
            # arg is square of size double: 1 + nt + no; single: 2 : nt
            # arg_inv = np.linalg.inv(arg.toarray())
            arg_inv = np.linalg.lstsq(
                arg.todense(), np.eye(npar), rcond=None)[0]
        else:
            # arg_inv = np.linalg.inv(arg)
            arg_inv = np.linalg.lstsq(
                arg, np.eye(npar), rcond=None)[0]

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

        assert np.all(p_var >= 0), 'Unable to invert the matrix' + str(p_var)

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
        X = X.toarray()

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
        mode,
        ds,
        st_label,
        ast_label,
        rst_label,
        rast_label,
        st_var=None,
        ast_var=None,
        rst_var=None,
        rast_var=None,
        D_F_label=None,
        D_B_label=None,
        D_F_var_label=None,
        D_B_var_label=None):
    """Eq.50 if weighted least squares"""
    time_dim = ds.get_time_dim()

    if st_var is not None:
        i_var_fw = ds.i_var(
            st_var,
            ast_var,
            st_label=st_label,
            ast_label=ast_label)
        i_var_bw = ds.i_var(
            rst_var,
            rast_var,
            st_label=rst_label,
            ast_label=rast_label)

        i_fw = np.log(ds[st_label] / ds[ast_label])
        i_bw = np.log(ds[rst_label] / ds[rast_label])

        if mode == 'guess':
            A_var = (i_var_fw + i_var_bw) / 2
            A = (i_bw - i_fw) / 2

        elif mode == 'exact':
            D_F = ds[D_F_label]
            D_B = ds[D_B_label]
            D_F_var = ds[D_F_var_label]
            D_B_var = ds[D_B_var_label]
            A_var = (i_var_fw + i_var_bw + D_B_var + D_F_var) / 2
            A = (i_bw - i_fw) / 2 + (D_B - D_F) / 2

        E_var = 1 / (1 / A_var).sum(dim=time_dim)
        E = (A / A_var).sum(dim=time_dim) * E_var

    else:
        i_fw = np.log(ds[st_label] / ds[ast_label])
        i_bw = np.log(ds[rst_label] / ds[rast_label])

        if mode == 'guess':
            A = (i_bw - i_fw) / 2
        elif mode == 'exact':
            D_F = ds[D_F_label]
            D_B = ds[D_B_label]
            A = (i_bw - i_fw) / 2 - (D_B - D_F) / 2

        E_var = A.var(dim=time_dim)
        E = A.mean(dim=time_dim)

    return E, E_var


def match_sections(ds, matching_sections,
                   check_pair_in_calibration_section=False,
                   check_pair_in_calibration_section_arg=None):
    """
    Matches location indices of two sections.

    Parameters
    ----------
    ds
    matching_sections : List[Tuple[slice, slice, bool]]
        Provide a list of tuples. A tuple per matching section. Each tuple
        has three items. The first two items are the slices of the sections
        that are matched. The third item is a boolean and is True if the two
        sections have a reverse direction ("J-configuration").
    check_pair_in_calibration_section : bool
        Use the sections check whether both items
        of the pair are in the calibration section. It produces a warning for
        each pair of which at least one item is outside of the calibration
        sections. The sections are set with
        `check_pair_in_calibration_section_arg`. If `None` the sections
        are obtained from `ds`.
    check_pair_in_calibration_section_arg : Dict[str, List[slice]], optional
        If `None` the sections are obtained from `ds`.

    Returns
    -------
    matching_indices : array-like
        Is an array of size (np, 2), where np is the number of paired
        locations. The array contains indices to locations along the fiber.

    """
    import warnings

    def err_msg(s1, s2):
        return """
        The current implementation requires that both locations of a matching
        pair should either be used in a calibration section or outside the
        calibration sections. x={:.3f}m is not in the calibration section
        while its matching location at x={:.3f}m is.""".format(s1, s2)

    for hslice, tslice, reverse_flag in matching_sections:
        hxs = ds.x.sel(x=hslice).size
        txs = ds.x.sel(x=tslice).size

        assert hxs == txs, 'the two sections do not have matching ' \
                           'number of items: ' + str(hslice) + 'size: ' + \
                           str(hxs) + str(tslice) + 'size: ' + str(txs)

    hix = ds.ufunc_per_section(
        sections={0: [i[0] for i in matching_sections]},
        x_indices=True,
        calc_per='all')

    tixl = []
    for _, tslice, reverse_flag in matching_sections:
        ixi = ds.ufunc_per_section(
            sections={0: [tslice]},
            x_indices=True,
            calc_per='all')

        if reverse_flag:
            tixl.append(ixi[::-1])
        else:
            tixl.append(ixi)

    tix = np.concatenate(tixl)

    ix_sec = ds.ufunc_per_section(
        x_indices=True,
        calc_per='all',
        sections=check_pair_in_calibration_section_arg)

    if not check_pair_in_calibration_section:
        return np.stack((hix, tix)).T
    else:
        # else perform checks whether both are in valid sections
        pass

    xv = ds.x.values

    hixl = []
    tixl = []

    for hii, tii in zip(hix, tix):
        if hii in ix_sec and tii in ix_sec:
            hixl.append(hii)
            tixl.append(tii)

        elif hii not in ix_sec and tii in ix_sec:
            warnings.warn(err_msg(xv[hii], xv[tii]))

        elif hii in ix_sec and tii not in ix_sec:
            warnings.warn(err_msg(xv[tii], xv[hii]))

        else:
            warnings.warn("""x={:.3f}m and x={:.3f}m are both locatated
            outside the calibration sections""".format(xv[hii], xv[tii]))

    hix_out = np.array(hixl)
    tix_out = np.array(tixl)

    err = 'Both locations of a matching pair should either be used in a ' \
          'calibration section or outside the calibration sections.'
    assert hix_out.size == tix_out.size, err
    assert hix_out.size > 0, 'no matching sections in calibration'

    return np.stack((hix_out, tix_out)).T

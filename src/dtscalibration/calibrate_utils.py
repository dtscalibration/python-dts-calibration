# coding=utf-8
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as ln


def calibration_single_ended_solver(
        ds,
        st_var=None,
        ast_var=None,
        calc_cov=True,
        solver='sparse',
        matching_indices=None,
        transient_att_x=None,
        verbose=False):
    """
    Parameters
    ----------
    ds : DataStore
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
    matching_indices : array-like
        Is an array of size (np, 2), where np is the number of paired
        locations. This array is produced by `matching_sections()`.
    transient_att_x : iterable, optional
        Splices can cause jumps in differential attenuation. Normal single
        ended calibration assumes these are not present. An additional loss term
        is added in the 'shadow' of the splice. Each location introduces an
        additional nt parameters to solve for. Requiring either an additional
        calibration section or matching sections. If multiple locations are
        defined, the losses are added.

    verbose : bool

    Returns
    -------

    """
    # get ix_sec argsort so the sections are in order of increasing x
    ix_sec = ds.ufunc_per_section(x_indices=True, calc_per='all')
    ds_sec = ds.isel(x=ix_sec)

    x_sec = ds_sec['x'].values
    x_all = ds['x'].values
    nx = x_sec.size
    nt = ds.time.size
    nta = len(transient_att_x) if transient_att_x else 0
    nm = matching_indices.shape[0] if np.any(matching_indices) else 0

    if np.any(matching_indices):
        ds_ms0 = ds.isel(x=matching_indices[:, 0])
        ds_ms1 = ds.isel(x=matching_indices[:, 1])

    p0_est = np.asarray([485., 0.1] + nt * [1.4] + nta * nt * [0.])

    # X \gamma  # Eq.34
    cal_ref = ds.ufunc_per_section(
        label='st', ref_temp_broadcasted=True, calc_per='all')
    cal_ref = cal_ref  # sort by increasing x
    data_gamma = 1 / (cal_ref.T.ravel() + 273.15)  # gamma
    coord_gamma_row = np.arange(nt * nx, dtype=int)
    coord_gamma_col = np.zeros(nt * nx, dtype=int)
    X_gamma = sp.coo_matrix(
        (data_gamma, (coord_gamma_row, coord_gamma_col)),
        shape=(nt * nx, 1),
        copy=False)

    # X \Delta\alpha  # Eq.34
    data_dalpha = np.tile(-x_sec, nt)  # dalpha
    coord_dalpha_row = np.arange(nt * nx, dtype=int)
    coord_dalpha_col = np.zeros(nt * nx, dtype=int)
    X_dalpha = sp.coo_matrix(
        (data_dalpha, (coord_dalpha_row, coord_dalpha_col)),
        shape=(nt * nx, 1),
        copy=False)

    # X C  # Eq.34
    data_c = -np.ones(nt * nx, dtype=int)
    coord_c_row = np.arange(nt * nx, dtype=int)
    coord_c_col = np.repeat(np.arange(nt, dtype=int), nx)

    X_c = sp.coo_matrix(
        (data_c, (coord_c_row, coord_c_col)),
        shape=(nt * nx, nt),
        copy=False)

    # X ta #not documented
    if transient_att_x:
        TA_list = list()

        for transient_att_xi in transient_att_x:
            # first index on the right hand side a the difficult splice
            # Deal with connector outside of fiber
            if transient_att_xi >= x_sec[-1]:
                ix_sec_ta_ix0 = nx
            elif transient_att_xi <= x_sec[0]:
                ix_sec_ta_ix0 = 0
            else:
                ix_sec_ta_ix0 = np.flatnonzero(
                    x_sec >= transient_att_xi)[0]

            # Data is -1
            # I = 1/Tref*gamma - C - da - TA
            data_ta = -np.ones(nt * (nx - ix_sec_ta_ix0), dtype=float)

            # skip ix_sec_ta_ix0 locations, because they are upstream of
            # the connector.
            coord_ta_row = (
                np.tile(np.arange(ix_sec_ta_ix0, nx), nt) +
                np.repeat(np.arange(nx * nt, step=nx), nx - ix_sec_ta_ix0)
            )

            # nt parameters
            coord_ta_col = np.repeat(
                np.arange(nt, dtype=int), nx - ix_sec_ta_ix0)

            TA_list.append(sp.coo_matrix(
                (data_ta, (coord_ta_row, coord_ta_col)),
                shape=(nt * nx, nt),
                copy=False))

        X_TA = sp.hstack(TA_list)

    else:
        X_TA = sp.coo_matrix(([], ([], [])), shape=(nt * nx, 0))

    if np.any(matching_indices):
        # first make matrix without the TA part (only diff in attentuation)
        data_ma = np.tile(
            ds_ms1['x'].values -
            ds_ms0['x'].values,
            nt
        )

        coord_ma_row = np.arange(nm * nt)

        coord_ma_col = np.ones(nt * nm)

        X_ma = sp.coo_matrix(
            (data_ma, (coord_ma_row, coord_ma_col)),
            shape=(nm * nt, 2 + nt),
            copy=False)

        # make TA matrix
        if transient_att_x:
            transient_m_data = np.zeros((nm, nta))
            for ii, row in enumerate(matching_indices):
                for jj, transient_att_xi in enumerate(transient_att_x):
                    transient_m_data[ii, jj] = np.logical_and(
                        transient_att_xi > x_all[row[0]],
                        transient_att_xi < x_all[row[1]]
                    ).astype(int)

            data_mt = np.tile(transient_m_data, (nt, 1)).flatten('F')

            coord_mt_row = (
                np.tile(np.arange(nm * nt), nta)
            )

            coord_mt_col = (
                np.tile(np.repeat(np.arange(nt), nm), nta)
                + np.repeat(np.arange(nta*nt, step=nt), nt * nm)
            )

            X_mt = sp.coo_matrix(
                (data_mt, (coord_mt_row, coord_mt_col)),
                shape=(nm * nt, nta * nt),
                copy=False)

        else:
            X_mt = sp.coo_matrix(
                ([], ([], [])),
                shape=(nm * nt, 0),
                copy=False)

        # merge the two
        X_m = sp.hstack((X_ma, X_mt))

    else:
        X_m = sp.coo_matrix(([], ([], [])), shape=(0, 2 + nt + nta * nt))

    # Stack all X's
    X = sp.vstack((
        sp.hstack((X_gamma, X_dalpha, X_c, X_TA)),
        X_m
    ))

    # y, transpose the values to arrange them correctly
    y = np.log(ds_sec.st / ds_sec.ast).values.T.ravel()

    if np.any(matching_indices):
        # y_m = I_1 - I_2
        y_m = (
            np.log(
                ds_ms0.st.values / ds_ms0.ast.values)
            - np.log(
                ds_ms1.st.values / ds_ms1.ast.values)
        ).T.ravel()

        y = np.hstack((y, y_m))

    # w
    if st_var is not None:
        w = 1 / (
            ds_sec.st ** -2 * st_var +
            ds_sec.ast ** -2 * ast_var).values.ravel()

        if np.any(matching_indices):
            w_ms = 1 / (
                (ds_ms0.st.values ** -2 * st_var) +
                (ds_ms0.ast.values ** -2 * ast_var) +
                (ds_ms1.st.values ** -2 * st_var) +
                (ds_ms1.ast.values ** -2 * ast_var)
            ).ravel()

            w = np.hstack((w, w_ms))
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
            X_m=X_m,
            X_TA=X_TA,
            p0_est=p0_est)

    else:
        raise ValueError("Choose a valid solver")

    if calc_cov:
        return p_sol, p_var, p_cov
    else:
        return p_sol, p_var


def calibration_double_ended_solver(
        ds,
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
    ix_sec = ds.ufunc_per_section(x_indices=True, calc_per='all')
    ds_sec = ds.isel(x=ix_sec)
    ix_alpha_is_zero = ix_sec[0]  # per definition of E

    x_sec = ds_sec['x'].values
    nx_sec = x_sec.size
    nt = ds.time.size
    nta = len(transient_asym_att_x) if transient_asym_att_x else 0

    # Calculate E as initial estimate for the E calibration.
    # Does not require ta to be passed on
    E_all_guess, E_all_var_guess = calc_alpha_double(
        'guess',
        ds,
        st_var,
        ast_var,
        rst_var,
        rast_var,
        ix_alpha_is_zero=ix_alpha_is_zero)
    df_est, db_est = calc_df_db_double_est(
        ds,
        ix_alpha_is_zero,
        485.)

    E, Z_D, Z_gamma, Zero_d, Z_TA_fw, Z_TA_bw, = \
        construct_submatrices(nt, nx_sec, ds, transient_asym_att_x, x_sec)

    # y  # Eq.41--45
    y_F = np.log(ds_sec.st / ds_sec.ast).values.ravel()
    y_B = np.log(ds_sec.rst / ds_sec.rast).values.ravel()

    # w
    if st_var is not None:  # WLS
        if callable(st_var):
            st_var_sec = st_var(ds_sec.st)
        else:
            st_var_sec = np.asarray(st_var)
        if callable(ast_var):
            ast_var_sec = ast_var(ds_sec.ast)
        else:
            ast_var_sec = np.asarray(ast_var)
        if callable(rst_var):
            rst_var_sec = rst_var(ds_sec.rst)
        else:
            rst_var_sec = np.asarray(rst_var)
        if callable(rast_var):
            rast_var_sec = rast_var(ds_sec.rast)
        else:
            rast_var_sec = np.asarray(rast_var)

        w_F = 1 / (
            ds_sec.st ** -2 * st_var_sec +
            ds_sec.ast ** -2 * ast_var_sec).values.ravel()
        w_B = 1 / (
            ds_sec.rst ** -2 * rst_var_sec +
            ds_sec.rast ** -2 * rast_var_sec).values.ravel()

    else:  # OLS
        w_F = np.ones(nt * nx_sec)
        w_B = np.ones(nt * nx_sec)

    if not np.any(matching_indices):
        p0_est = np.concatenate(([485.], df_est, db_est,
                                 E_all_guess[ix_sec[1:]], nta * nt * 2 * [0.]))

        # Stack all X's
        X = sp.vstack(
            (sp.hstack((Z_gamma, -Z_D, Zero_d, -E, Z_TA_fw)),
             sp.hstack((Z_gamma, Zero_d, -Z_D, E, Z_TA_bw))))

        y = np.concatenate((y_F, y_B))
        w = np.concatenate((w_F, w_B))

    else:
        E_match_F, E_match_B, E_match_no_cal, Z_TA_eq1, Z_TA_eq2, \
            Z_TA_eq3, d_no_cal, ix_from_cal_match_to_glob, ix_match_not_cal, \
            Zero_eq12_gamma, Zero_eq3_gamma, Zero_d_eq12 = \
            construct_submatrices_matching_sections(
                ds.x.values, ix_sec, matching_indices[:, 0],
                matching_indices[:, 1], nt, transient_asym_att_x)

        p0_est = np.concatenate((np.asarray([485.] + 2 * nt * [1.4]),
                                 E_all_guess[ix_from_cal_match_to_glob],
                                 nta * nt * 2 * [0.]))
        # Stack all X's
        # X_sec contains a different number of columns than X.
        X_sec = sp.vstack(
            (sp.hstack((Z_gamma, -Z_D, Zero_d, -E, Z_TA_fw)),
             sp.hstack((Z_gamma, Zero_d, -Z_D, E, Z_TA_bw))))
        X_sec2 = sp.csr_matrix(
            ([], ([], [])),
            shape=(2 * nt * nx_sec,
                   1 + 2 * nt + ds.x.size + 2 * nta * nt))

        from_i = np.concatenate((
            np.arange(1 + 2 * nt),
            1 + 2 * nt + ix_sec[1:],
            np.arange(1 + 2 * nt + ds.x.size,
                      1 + 2 * nt + ds.x.size + 2 * nta *
                      nt)))
        X_sec2[:, from_i] = X_sec
        from_i2 = np.concatenate((
            np.arange(1 + 2 * nt),
            1 + 2 * nt + ix_from_cal_match_to_glob,
            np.arange(1 + 2 * nt + ds.x.size,
                      1 + 2 * nt + ds.x.size + 2 * nta *
                      nt)))
        X = sp.vstack(
            (X_sec2[:, from_i2],
             sp.hstack((Zero_eq12_gamma, Zero_d_eq12, E_match_F, Z_TA_eq1)),
             sp.hstack((Zero_eq12_gamma, Zero_d_eq12, E_match_B, Z_TA_eq2)),
             sp.hstack((Zero_eq3_gamma, d_no_cal, E_match_no_cal, Z_TA_eq3))))

        y_F = np.log(ds_sec.st / ds_sec.ast).values.ravel()
        y_B = np.log(ds_sec.rst / ds_sec.rast).values.ravel()

        hix = matching_indices[:, 0]
        tix = matching_indices[:, 1]
        ds_hix = ds.isel(x=hix)
        ds_tix = ds.isel(x=tix)
        y_eq1 = (np.log(ds_hix.st / ds_hix.ast).values.ravel() -
                 np.log(ds_tix.st / ds_tix.ast).values.ravel())
        y_eq2 = (np.log(ds_hix.rst / ds_hix.rast).values.ravel() -
                 np.log(ds_tix.rst / ds_tix.rast).values.ravel())

        ds_mnc = ds.isel(x=ix_match_not_cal)
        y_eq3 = ((np.log(ds_mnc.rst / ds_mnc.rast) -
                 np.log(ds_mnc.st / ds_mnc.ast)) /
                 2).values.ravel()

        y = np.concatenate((y_F, y_B, y_eq1, y_eq2, y_eq3))

        if callable(st_var):
            st_var_hix = st_var(ds_hix.st)
            st_var_tix = st_var(ds_tix.st)
            st_var_mnc = st_var(ds_mnc.st)
        else:
            st_var_hix = np.asarray(st_var)
            st_var_tix = np.asarray(st_var)
            st_var_mnc = np.asarray(st_var)
        if callable(ast_var):
            ast_var_hix = ast_var(ds_hix.ast)
            ast_var_tix = ast_var(ds_tix.ast)
            ast_var_mnc = ast_var(ds_mnc.ast)
        else:
            ast_var_hix = np.asarray(ast_var)
            ast_var_tix = np.asarray(ast_var)
            ast_var_mnc = np.asarray(ast_var)
        if callable(rst_var):
            rst_var_hix = rst_var(ds_hix.rst)
            rst_var_tix = rst_var(ds_tix.rst)
            rst_var_mnc = rst_var(ds_mnc.rst)
        else:
            rst_var_hix = np.asarray(rst_var)
            rst_var_tix = np.asarray(rst_var)
            rst_var_mnc = np.asarray(rst_var)
        if callable(rast_var):
            rast_var_hix = rast_var(ds_hix.rast)
            rast_var_tix = rast_var(ds_tix.rast)
            rast_var_mnc = rast_var(ds_mnc.rast)
        else:
            rast_var_hix = np.asarray(rast_var)
            rast_var_tix = np.asarray(rast_var)
            rast_var_mnc = np.asarray(rast_var)

        w_eq1 = 1 / (
            (ds_hix.st ** -2 * st_var_hix +
             ds_hix.ast ** -2 * ast_var_hix).values.ravel() +
            (ds_tix.st ** -2 * st_var_tix +
             ds_tix.ast ** -2 * ast_var_tix).values.ravel())
        w_eq2 = 1 / (
            (ds_hix.rst ** -2 * rst_var_hix +
             ds_hix.rast ** -2 * rast_var_hix).values.ravel() +
            (ds_tix.rst ** -2 * rst_var_tix +
             ds_tix.rast ** -2 * rast_var_tix).values.ravel())
        w_eq3 = 1 / (
            ds_mnc.st ** -2 * st_var_mnc +
            ds_mnc.ast ** -2 * ast_var_mnc +
            ds_mnc.rst ** -2 * rst_var_mnc +
            ds_mnc.rast ** -2 * rast_var_mnc).values.ravel()

        w = np.concatenate((w_F, w_B, w_eq1, w_eq2, w_eq3))

    if solver == 'sparse':
        solver_fun = wls_sparse
    elif solver == 'stats':
        solver_fun = wls_stats
    elif solver == 'external':
        return X, y, w, p0_est
    elif solver == 'external_split':
        out = dict(
            y_F=y_F,
            y_B=y_B,
            w_F=w_F,
            w_B=w_B,
            Z_gamma=Z_gamma,
            Z_D=Z_D,
            Zero_d=Zero_d,
            E=E,
            Z_TA_fw=Z_TA_fw,
            Z_TA_bw=Z_TA_bw,
            p0_est=p0_est,
            E_all_guess=E_all_guess,
            E_all_var_guess=E_all_var_guess)

        if np.any(matching_indices):
            out.update(
                ix_from_cal_match_to_glob=ix_from_cal_match_to_glob,
                E_match_F=E_match_F,
                E_match_B=E_match_B,
                E_match_no_cal=E_match_no_cal,
                Zero_eq12_gamma=Zero_eq12_gamma,
                Zero_eq3_gamma=Zero_eq3_gamma,
                Zero_d_eq12=Zero_d_eq12,
                d_no_cal=d_no_cal,
                Z_TA_eq1=Z_TA_eq1,
                Z_TA_eq2=Z_TA_eq2,
                Z_TA_eq3=Z_TA_eq3,
                y_eq1=y_eq1,
                y_eq2=y_eq2,
                y_eq3=y_eq3,
                w_eq1=w_eq1,
                w_eq2=w_eq2,
                w_eq3=w_eq3)
        return out
    else:
        raise ValueError("Choose a valid solver")

    out = solver_fun(X, y, w=w, x0=p0_est, calc_cov=calc_cov,
                     verbose=verbose, return_werr=verbose)
    if calc_cov and verbose:
        p_sol, p_var, p_cov, werr = out
    elif not calc_cov and verbose:
        p_sol, p_var, werr = out
    elif calc_cov and not verbose:
        p_sol, p_var, p_cov = out
    elif not calc_cov and not verbose:
        p_sol, p_var = out

    # if verbose:
    #     from dtscalibration.plot import plot_location_residuals_double_ended
    #
    #     dv = plot_location_residuals_double_ended(ds, werr, hix, tix, ix_sec,
    #                                               ix_match_not_cal, nt)

    # p_sol contains the int diff att of all the locations within the
    # reference sections. po_sol is its expanded version that contains also
    # the int diff att for outside the reference sections.

    # calculate talpha_fw and bw for attenuation
    if transient_asym_att_x:
        if np.any(matching_indices):
            ta = p_sol[1 + 2 * nt + ix_from_cal_match_to_glob.size:
                       ].reshape((nt, 2, nta), order='F')
            ta_var = p_var[1 + 2 * nt + ix_from_cal_match_to_glob.size:
                           ].reshape((nt, 2, nta), order='F')

        else:
            ta = p_sol[2 * nt + nx_sec:].reshape((nt, 2, nta), order='F')
            ta_var = p_var[2 * nt + nx_sec:].reshape((nt, 2, nta), order='F')

        talpha_fw = ta[:, 0, :]
        talpha_bw = ta[:, 1, :]
        talpha_fw_var = ta_var[:, 0, :]
        talpha_bw_var = ta_var[:, 1, :]
    else:
        talpha_fw = None
        talpha_bw = None
        talpha_fw_var = None
        talpha_bw_var = None

    # put E outside of reference section in solution
    # concatenating makes a copy of the data instead of using a pointer
    ds_sub = ds[['st', 'ast', 'rst', 'rast']]
    time_dim = ds_sub.get_time_dim()
    ds_sub['df'] = ((time_dim,), p_sol[1:1 + nt])
    ds_sub['df_var'] = ((time_dim,), p_var[1:1 + nt])
    ds_sub['db'] = ((time_dim,), p_sol[1 + nt:1 + 2 * nt])
    ds_sub['db_var'] = ((time_dim,), p_var[1 + nt:1 + 2 * nt])
    E_all_exact, E_all_var_exact = calc_alpha_double(
        'exact',
        ds_sub,
        st_var,
        ast_var,
        rst_var,
        rast_var,
        'df',
        'db',
        'df_var',
        'db_var',
        ix_alpha_is_zero=ix_alpha_is_zero,
        transient_asym_att_x=transient_asym_att_x,
        talpha_fw=talpha_fw,
        talpha_bw=talpha_bw,
        talpha_fw_var=talpha_fw_var,
        talpha_bw_var=talpha_bw_var)

    if np.any(matching_indices):
        p_sol_size = 1 + 2 * nt + ix_from_cal_match_to_glob.size + 2 * nt * nta
    else:
        p_sol_size = 1 + 2 * nt + (nx_sec - 1) + 2 * nt * nta
    assert p_sol.size == p_sol_size
    assert p_var.size == p_sol_size

    if np.any(matching_indices):
        po_sol = np.concatenate((
            p_sol[:1 + 2 * nt],
            E_all_exact,
            p_sol[1 + 2 * nt + ix_from_cal_match_to_glob.size:]))
        po_sol[1 + 2 * nt + ix_from_cal_match_to_glob] = \
            p_sol[1 + 2 * nt:1 + 2 * nt + ix_from_cal_match_to_glob.size]
    else:
        po_sol = np.concatenate((p_sol[:1 + 2 * nt],
                                 E_all_exact,
                                 p_sol[2 * nt + nx_sec:]))
        po_sol[1 + 2 * nt + ix_sec[1:]] = p_sol[1 + 2 * nt:2 * nt + nx_sec]

    po_sol[1 + 2 * nt + ix_sec[0]] = 0.  # per definition

    if np.any(matching_indices):
        po_var = np.concatenate((
            p_var[:1 + 2 * nt],
            E_all_var_exact,
            p_var[1 + 2 * nt + ix_from_cal_match_to_glob.size:]))
        po_var[1 + 2 * nt + ix_from_cal_match_to_glob] = \
            p_var[1 + 2 * nt:1 + 2 * nt + ix_from_cal_match_to_glob.size]
    else:
        po_var = np.concatenate((p_var[:1 + 2 * nt],
                                 E_all_var_exact,
                                 p_var[2 * nt + nx_sec:]))
        po_var[1 + 2 * nt + ix_sec[1:]] = p_var[1 + 2 * nt:2 * nt + nx_sec]
    po_var[1 + 2 * nt + ix_sec[0]] = 0.  # per definition

    if calc_cov:
        # the COV can be expensive to compute (in the least squares routine)
        po_cov = np.diag(po_var).copy()

        if np.any(matching_indices):
            from_i = np.concatenate((
                np.arange(1 + 2 * nt),
                1 + 2 * nt + ix_from_cal_match_to_glob,
                np.arange(1 + 2 * nt + ix_from_cal_match_to_glob.size,
                          1 + 2 * nt + ix_from_cal_match_to_glob.size +
                          nta * nt * 2)))
        else:
            from_i = np.concatenate((np.arange(1 + 2 * nt),
                                     1 + 2 * nt + ix_sec[1:],
                                     np.arange(1 + 2 * nt + nx_sec,
                                               1 + 2 * nt + nx_sec + nta * nt * 2)))

        iox_sec1, iox_sec2 = np.meshgrid(from_i, from_i, indexing='ij')
        po_cov[iox_sec1, iox_sec2] = p_cov

        return po_sol, po_var, po_cov

    else:
        return po_sol, po_var


def matching_section_location_indices(ix_sec, hix, tix):
    # contains all indices of the entire fiber that either are used for
    # calibrating to reference temperature or for matching sections. Is sorted.
    ix_cal_match = np.unique(np.concatenate((ix_sec, hix, tix)))

    # number of locations of interest, width of the section of interest.
    nx_cal_match = ix_cal_match.size

    # indices in the section of interest. Including E0.
    ix_sec2 = np.searchsorted(ix_cal_match, ix_sec)

    # indices in the section of interest. Excluding E0
    # ix_E0 mask - to exclude E[ix_sec[0]] from the E matrices
    ix_E0_mask = np.array([ix for ix in range(nx_cal_match) if
                           ix != ix_sec2[0]])
    # contains the global coordinate indices of the E
    ix_from_cal_match_to_glob = ix_cal_match[ix_E0_mask]
    return ix_from_cal_match_to_glob


def construct_submatrices_matching_sections(
      x, ix_sec, hix, tix, nt, transient_asym_att_x):
    """
    For all matching indices, where subscript 1 refers to the indices in
    `hix` and subscript 2 refers to the indices in `tix`.
    F1 - F2 = E2 - E1 + TAF2 - TAF1  # EQ1
    B1 - B2 = E1 - E2 + TAB2 - TAB1  # EQ2

    For matching indices (`hix` and `tix`) that are outside of the reference
    sections an additional equation is needed for `E` per time step.
    (B3 - F3) / 2 = E3 + (df-db) / 2 + (TAF3 - TAB3) / 2  # EQ3
    where subscript 3 refers an a hix or a tix that is not in a reference
    section.

    Note that E[ix_sec[0]] = 0, and not included in the parameters. Dealt
    with by first assuming it is a parameter, then remove it from coefficent
    matrices. Note that indices _sec2 contain E[ix_sec[0]]

    Ordering when unpaking square matrix: nt observations for location 1 then
    nt observations for location 2.

    # ix of Observations and weights
    # ix_y_eq1_f1 = hix
    # ix_y_eq1_f2 = tix
    # ix_y_eq2_b1 = hix
    # ix_y_eq2_b2 = tix
    # ix_y_eq3 = ix_match_not_cal

    Parameters
    ----------
    x : array-like of float
      coordinates along the fiber, needed to create the matrices for
      transient attenuation.
    ix_sec : array-like of int
    hix : array-like of int
    tix : array-like of int
    nt : int

    Returns
    -------

    """
    # contains all indices of the entire fiber that either are using for
    # calibrating to reference temperature or for matching sections. Is sorted.
    ix_cal_match = np.unique(np.concatenate((ix_sec, hix, tix)))

    # subscript 3 in doc-eqns
    ix_match_not_cal = np.array([ix for ix in ix_cal_match if ix not in ix_sec])

    # number of locations of interest, width of the section of interest.
    nx_cal_match = ix_cal_match.size
    npair = len(hix)

    # indices in the section of interest.
    ix_match_not_cal_sec2 = np.searchsorted(ix_cal_match, ix_match_not_cal)

    # indices in the section of interest. Including E0.
    ix_sec2 = np.searchsorted(ix_cal_match, ix_sec)
    hix_sec2 = np.searchsorted(ix_cal_match, hix)  # subscript 1 in doc-eqns
    tix_sec2 = np.searchsorted(ix_cal_match, tix)  # subscript 2 in doc-eqns

    # indices in the section of interest. Excluding E0
    # ix_E0 mask - to exclude E[ix_sec[0]] from the E matrices
    ix_E0_mask = np.array([ix for ix in range(nx_cal_match) if
                           ix != ix_sec2[0]])
    # contains the global coordinate indices of the E
    ix_from_cal_match_to_glob = ix_cal_match[ix_E0_mask]

    # E in EQ1
    data = np.ones(nt * npair, dtype=float)
    row = np.arange(nt * npair, dtype=int)
    col1 = np.repeat(hix_sec2, nt)
    col2 = np.repeat(tix_sec2, nt)
    E_match_F = sp.coo_matrix(
        (np.concatenate((-data, data)),
         (np.concatenate((row, row)),
          np.concatenate((col1, col2)))),
        shape=(nt * npair, nx_cal_match),
        copy=False).tocsr(copy=False)[:, ix_E0_mask].tocoo()
    Zero_eq12_gamma = sp.coo_matrix(
        ([], ([], [])),
        shape=(nt * npair, 1))
    Zero_d_eq12 = sp.coo_matrix(
        ([], ([], [])),
        shape=(nt * npair, 2 * nt))

    # E in EQ2
    data = np.ones(nt * npair, dtype=float)
    row = np.arange(nt * npair, dtype=int)
    col1 = np.repeat(hix_sec2, nt)
    col2 = np.repeat(tix_sec2, nt)
    E_match_B = sp.coo_matrix(
        (np.concatenate((data, -data)),
         (np.concatenate((row, row)),
          np.concatenate((col1, col2)))),
        shape=(nt * npair, nx_cal_match),
        copy=False).tocsr(copy=False)[:, ix_E0_mask].tocoo()

    # E in EQ3
    nx_nm = ix_match_not_cal_sec2.size
    data = np.ones(nt * nx_nm, dtype=float)
    row = np.arange(nt * nx_nm, dtype=int)
    col = np.repeat(ix_match_not_cal_sec2, nt)
    E_match_no_cal = sp.coo_matrix(
        (data, (row, col)),
        shape=(nt * nx_nm, nx_cal_match),
        copy=False).tocsr(copy=False)[:, ix_E0_mask].tocoo()

    # DF and DB in EQ3
    data = np.ones(nt * nx_nm, dtype=float) / 2
    row = np.arange(nt * nx_nm, dtype=int)
    colf = np.tile(np.arange(nt, dtype=int), nx_nm)
    colb = np.tile(np.arange(nt, 2 * nt, dtype=int), nx_nm)
    d_no_cal = sp.coo_matrix(
        (np.concatenate((data, -data)),
         (np.concatenate((row, row)),
          np.concatenate((colf, colb)))),
        shape=(nt * nx_nm, 2 * nt),
        copy=False)
    Zero_eq3_gamma = sp.coo_matrix(
        ([], ([], [])),
        shape=(nt * nx_nm, 1))

    # TA
    if transient_asym_att_x:
        # unpublished BdT

        TA_eq1_list = list()
        TA_eq2_list = list()
        TA_eq3_list = list()

        for transient_asym_att_xi in transient_asym_att_x:
            """For forward direction."""
            # first index on the right hand side a the difficult splice
            # Deal with connector outside of fiber
            if transient_asym_att_xi >= x[-1]:
                ix_ta_ix0 = x.size
            elif transient_asym_att_xi <= x[0]:
                ix_ta_ix0 = 0
            else:
                ix_ta_ix0 = np.flatnonzero(
                    x >= transient_asym_att_xi)[0]

            # TAF1 and TAF2 in EQ1
            data_taf = np.repeat(
                -np.array(hix >= ix_ta_ix0, dtype=float) +
                np.array(tix >= ix_ta_ix0, dtype=float), nt)
            row_taf = np.arange(nt * npair)
            col_taf = np.tile(np.arange(nt, dtype=int), npair)
            mask_taf = data_taf.astype(bool)  # only store non-zeros in sparse m
            TA_eq1_list.append(sp.coo_matrix(
                    (data_taf[mask_taf], (row_taf[mask_taf],
                                          col_taf[mask_taf])),
                    shape=(nt * npair, 2 * nt),
                    copy=False))

            # TAB1 and TAB2 in EQ2
            data_tab = np.repeat(
                -np.array(hix < ix_ta_ix0, dtype=float) +
                np.array(tix < ix_ta_ix0, dtype=float), nt)
            row_tab = np.arange(nt * npair)
            col_tab = np.tile(np.arange(nt, 2 * nt, dtype=int), npair)
            mask_tab = data_tab.astype(bool)  # only store non-zeros in sparse m
            TA_eq2_list.append(sp.coo_matrix(
                    (data_tab[mask_tab], (row_tab[mask_tab],
                                          col_tab[mask_tab])),
                    shape=(nt * npair, 2 * nt),
                    copy=False))

            data_taf = np.repeat(
                np.array(ix_match_not_cal >= ix_ta_ix0, dtype=float) / 2, nt)
            data_tab = np.repeat(
                -np.array(ix_match_not_cal < ix_ta_ix0, dtype=float) / 2, nt)
            row_ta = np.arange(nt * nx_nm)
            col_taf = np.tile(np.arange(nt, dtype=int), nx_nm)
            col_tab = np.tile(np.arange(nt, 2 * nt, dtype=int), nx_nm)
            mask_taf = data_taf.astype(bool)  # only store non-zeros in sparse m
            mask_tab = data_tab.astype(bool)  # only store non-zeros in sparse m
            TA_eq3_list.append(sp.coo_matrix(
                    (np.concatenate((data_taf[mask_taf], data_tab[mask_tab])),
                     (np.concatenate((row_ta[mask_taf], row_ta[mask_tab])),
                      np.concatenate((col_taf[mask_taf], col_tab[mask_tab])))),
                    shape=(nt * nx_nm, 2 * nt),
                    copy=False))

        Z_TA_eq1 = sp.hstack(TA_eq1_list)
        Z_TA_eq2 = sp.hstack(TA_eq2_list)
        Z_TA_eq3 = sp.hstack(TA_eq3_list)

    else:
        Z_TA_eq1 = sp.coo_matrix(([], ([], [])), shape=(nt * npair, 0))
        Z_TA_eq2 = sp.coo_matrix(([], ([], [])), shape=(nt * npair, 0))
        Z_TA_eq3 = sp.coo_matrix(([], ([], [])), shape=(nt * nx_nm, 0))

    return (E_match_F, E_match_B, E_match_no_cal, Z_TA_eq1, Z_TA_eq2,
            Z_TA_eq3, d_no_cal, ix_from_cal_match_to_glob, ix_match_not_cal,
            Zero_eq12_gamma, Zero_eq3_gamma, Zero_d_eq12)


def construct_submatrices(nt, nx, ds, transient_asym_att_x, x_sec):
    """Wrapped in a function to reduce memory usage.
    E is zero at the first index of the reference section (ds_sec)
    Constructing:
    Z_gamma (nt * nx, 1). Data: positive 1/temp
    Z_D (nt * nx, nt). Data: ones
    E (nt * nx, nx). Data: ones
    Zero_gamma (nt * nx, 1)
    zero_d (nt * nx, nt)
    Z_TA_fw (nt * nx, nta * 2 * nt) minus ones
    Z_TA_bw (nt * nx, nta * 2 * nt) minus ones

    I_fw = 1/Tref*gamma - D_fw - E - TA_fw
    I_bw = 1/Tref*gamma - D_bw + E - TA_bw
    """

    # Z \gamma  # Eq.47
    cal_ref = np.array(ds.ufunc_per_section(
        label='st', ref_temp_broadcasted=True, calc_per='all'))
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
    # E  # Eq.47
    # E is 0 at ix=0
    data_c = np.ones(nt * (nx - 1), dtype=float)
    coord_c_row = np.arange(nt, nt * nx, dtype=int)
    coord_c_col = np.repeat(np.arange(nx - 1, dtype=int), nt)
    E = sp.coo_matrix(
        (data_c, (coord_c_row, coord_c_col)),
        shape=(nt * nx, (nx - 1)),
        copy=False)
    # Zero  # Eq.45
    Zero_d = sp.coo_matrix(([], ([], [])), shape=(nt * nx, nt))
    # Zero_E = sp.coo_matrix(([], ([], [])), shape=(nt * nx, (nx - 1)))
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

    return E, Z_D, Z_gamma, Zero_d, Z_TA_fw, Z_TA_bw


def wls_sparse(X, y, w=1., calc_cov=False, verbose=False, x0=None,
               return_werr=False, **solver_kwargs):
    """
    If some initial estimate x0 is known and if damp == 0, one could proceed as follows:
    - Compute a residual vector r0 = b - A*x0.
    - Use LSQR to solve the system A*dx = r0.
    - Add the correction dx to obtain a final solution x = x0 + dx.
    from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.
      linalg.lsqr.html

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
    # The var returned by ln.lsqr is normalized by the variance of the error.
    # To obtain the correct variance, it needs to be scaled by the variance
    # of the error.

    if x0 is not None:
        assert not np.any(~np.isfinite(x0)), 'Nan/inf in p0 initial estimate'
    else:
        raise NotImplementedError

    assert not np.any(~np.isfinite(X)), 'Nan/inf in X: check reference' +\
        ' temperatures?'
    assert not np.any(~np.isfinite(w)), 'Nan/inf in weights'
    assert not np.any(~np.isfinite(y)), 'Nan/inf in observations'

    # precision up to 10th decimal. So that the temperature is approximately
    # estimated with 8 decimal precision.
    if 'atol' not in solver_kwargs:
        solver_kwargs['atol'] = 1e-16
    if 'btol' not in solver_kwargs:
        solver_kwargs['btol'] = 1e-16

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

    if x0 is None:
        # noinspection PyTypeChecker
        out_sol = ln.lsqr(wX, wy, show=verbose, calc_var=True,
                          **solver_kwargs)
        p_sol = out_sol[0]

    else:
        wr0 = wy - wX.dot(x0)

        # noinspection PyTypeChecker
        out_sol = ln.lsqr(wX, wr0, show=verbose, calc_var=True,
                          **solver_kwargs)

        p_sol = x0 + out_sol[0]

    # The residual degree of freedom, defined as the number of observations
    # minus the rank of the regressor matrix.
    nobs = len(y)
    npar = X.shape[1]  # ==rank
    degrees_of_freedom_err = nobs - npar
    wresid = wy - wX.dot(p_sol)
    err_var = np.dot(wresid, wresid) / degrees_of_freedom_err

    if calc_cov:
        arg = wX.T.dot(wX)

        if sp.issparse(arg):
            # arg_inv = np.linalg.inv(arg.toarray())
            arg_inv = np.linalg.lstsq(
                arg.todense(), np.eye(npar), rcond=None)[0]
        else:
            # arg_inv = np.linalg.inv(arg)
            arg_inv = np.linalg.lstsq(
                arg, np.eye(npar), rcond=None)[0]

        p_cov = np.array(arg_inv * err_var)
        p_var = np.diagonal(p_cov)

        if np.any(p_var < 0):
            m = 'Unable to invert the matrix. The following parameters are ' \
                'difficult to determine:' + str(np.where(p_var < 0))
            assert np.all(p_var >= 0), m

        if return_werr:
            return p_sol, p_var, p_cov, wresid
        else:
            return p_sol, p_var, p_cov

    else:
        p_var = out_sol[-1] * err_var  # normalized covariance

        if return_werr:
            return p_sol, p_var, wresid
        else:
            return p_sol, p_var


def wls_stats(X, y, w=1., calc_cov=False, x0=None, return_werr=False,
              verbose=False):
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

    if x0 is not None:
        # Initial values not supported by statsmodels
        pass

    mod_wls = sm.WLS(y, X, weights=w)
    res_wls = mod_wls.fit()
    p_sol = res_wls.params

    if verbose:
        print(res_wls.summary())

    p_cov = res_wls.cov_params()
    p_var = res_wls.bse**2

    if return_werr:
        wy = np.sqrt(w) * y
        wX = np.sqrt(w) * X
        werr = wy - wX.dot(p_sol)

    if calc_cov:
        if return_werr:
            return p_sol, p_var, p_cov, werr
        else:
            return p_sol, p_var, p_cov
    else:
        if return_werr:
            return p_sol, p_var, werr
        else:
            return p_sol, p_var


def calc_alpha_double(
        mode,
        ds,
        st_var=None,
        ast_var=None,
        rst_var=None,
        rast_var=None,
        D_F_label=None,
        D_B_label=None,
        D_F_var_label=None,
        D_B_var_label=None,
        ix_alpha_is_zero=-1,
        transient_asym_att_x=None,
        talpha_fw=None,
        talpha_bw=None,
        talpha_fw_var=None,
        talpha_bw_var=None):
    """Eq.50 if weighted least squares"""
    assert ix_alpha_is_zero >= 0, 'Define ix_alpha_is_zero' + \
                                  str(ix_alpha_is_zero)

    time_dim = ds.get_time_dim()

    if st_var is not None:
        if callable(st_var):
            st_var_val = st_var(ds.st)
        else:
            st_var_val = np.asarray(st_var)
        if callable(ast_var):
            ast_var_val = ast_var(ds.ast)
        else:
            ast_var_val = np.asarray(ast_var)
        if callable(rst_var):
            rst_var_val = rst_var(ds.rst)
        else:
            rst_var_val = np.asarray(rst_var)
        if callable(rast_var):
            rast_var_val = rast_var(ds.rast)
        else:
            rast_var_val = np.asarray(rast_var)

        i_var_fw = ds.i_var(
            st_var_val,
            ast_var_val,
            st_label='st',
            ast_label='ast')
        i_var_bw = ds.i_var(
            rst_var_val,
            rast_var_val,
            st_label='rst',
            ast_label='rast')

        i_fw = np.log(ds.st / ds.ast)
        i_bw = np.log(ds.rst / ds.rast)

        if mode == 'guess':
            A_var = (i_var_fw + i_var_bw) / 2
            A = (i_bw - i_fw) / 2

        elif mode == 'exact':
            D_F = ds[D_F_label]
            D_B = ds[D_B_label]
            D_F_var = ds[D_F_var_label]
            D_B_var = ds[D_B_var_label]

            if transient_asym_att_x:
                # Can be improved by including covariances. That reduces the
                # uncert.

                ta_arr_fw = np.zeros((ds.x.size, ds[time_dim].size))
                ta_arr_fw_var = np.zeros((ds.x.size, ds[time_dim].size))
                for tai, taxi, tai_var in zip(talpha_fw.T, transient_asym_att_x,
                                              talpha_fw_var.T):
                    ta_arr_fw[ds.x.values >= taxi] = \
                        ta_arr_fw[ds.x.values >= taxi] + tai
                    ta_arr_fw_var[ds.x.values >= taxi] = \
                        ta_arr_fw_var[ds.x.values >= taxi] + tai_var

                ta_arr_bw = np.zeros((ds.x.size, ds[time_dim].size))
                ta_arr_bw_var = np.zeros((ds.x.size, ds[time_dim].size))
                for tai, taxi, tai_var in zip(talpha_bw.T, transient_asym_att_x,
                                              talpha_bw_var.T):
                    ta_arr_bw[ds.x.values < taxi] = \
                        ta_arr_bw[ds.x.values < taxi] + tai
                    ta_arr_bw_var[ds.x.values < taxi] = \
                        ta_arr_bw_var[ds.x.values < taxi] + tai_var

                A_var = (i_var_fw + i_var_bw + D_B_var + D_F_var +
                         ta_arr_fw_var + ta_arr_bw_var) / 2
                A = (i_bw - i_fw) / 2 + (D_B - D_F) / 2 + (ta_arr_bw -
                                                           ta_arr_fw) / 2

            else:
                A_var = (i_var_fw + i_var_bw + D_B_var + D_F_var) / 2
                A = (i_bw - i_fw) / 2 + (D_B - D_F) / 2

        E_var = 1 / (1 / A_var).sum(dim=time_dim)
        E = (A / A_var).sum(dim=time_dim) * E_var

    else:
        i_fw = np.log(ds.st / ds.ast)
        i_bw = np.log(ds.rst / ds.rast)

        if mode == 'guess':
            A = (i_bw - i_fw) / 2
        elif mode == 'exact':
            D_F = ds[D_F_label]
            D_B = ds[D_B_label]
            A = (i_bw - i_fw) / 2 + (D_B - D_F) / 2

        E_var = A.var(dim=time_dim)
        E = A.mean(dim=time_dim)

    # E is defined zero at the first index of the reference sections
    if mode == 'guess':
        E -= E.isel(x=ix_alpha_is_zero)

    # assert np.abs(E.isel(x=ix_alpha_is_zero)) < 1e-8, \
    #     'Something went wrong in the estimation of d_f and d_b: ' + str(E)

    return E, E_var


def calc_df_db_double_est(ds, ix_alpha_is_zero, gamma_est):
    Ifwx0 = np.log(ds.st.isel(x=ix_alpha_is_zero) /
                   ds.ast.isel(x=ix_alpha_is_zero)).values
    Ibwx0 = np.log(ds.rst.isel(x=ix_alpha_is_zero) /
                   ds.rast.isel(x=ix_alpha_is_zero)).values
    ref_temps_refs = ds.ufunc_per_section(
        label='st',
        ref_temp_broadcasted=True,
        calc_per='all')
    ix_sec = ds.ufunc_per_section(x_indices=True, calc_per='all')
    ref_temps_x0 = ref_temps_refs[ix_sec == ix_alpha_is_zero].flatten(
        ).compute() + 273.15
    df_est = gamma_est / ref_temps_x0 - Ifwx0
    db_est = gamma_est / ref_temps_x0 - Ibwx0
    return df_est, db_est


def match_sections(ds, matching_sections):
    """
    Matches location indices of two sections.
    Parameters
    ----------
    ds
    matching_sections : List[Tuple[slice, slice, bool]]
        Provide a list of tuples. A tuple per matching section. Each tuple
        has three items. The first two items are the slices of the sections
        that are matched. The third item is a boolean and is True if the two
        sections have a reverse direction ("J-configuration"), most common.

    Returns
    -------
    matching_indices : array-like
        Is an array of size (np, 2), where np is the number of paired
        locations. The array contains indices to locations along the fiber.
    """
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

    return np.stack((hix, tix)).T

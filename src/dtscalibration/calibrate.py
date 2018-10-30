# coding=utf-8
def calibration_multiple_fix_gamma(ds_inputs):
    """
    Calibrate multiple DataStores simulataneous to share the same gamma

    Parameters
    ----------
    ds_inputs : List[Tuple[DataStore, str, Dict[str, Union[DataStore, str, float]]]]
        A list of tuples. Each tuple contains three items. The first item is an (uncalibrated)
        DataStore. The second item is the string 'double' or 'single'. The third
        item is a dictionary containing all the arguments to call for calibration. Only weighted
        calibration is supported.

    Notes
    -----
    1.  The function that is called may be any from {
            'calibration_single_ended_wls', 'calibration_double_ended_wls'}. It should support with
            the argument solver='external', to output (X, y, w, p0_est). Where X is the big
            matrix with the unweighted parameter coefficients (may be sparse), y are the
            unweighted observations, and w are the weights (e.g., 1/variance).

    Returns
    -------

    """
    import dtscalibration.calibrate_utils as utils

    import numpy as np
    import scipy.sparse as sp

    dtype = np.float

    X_list = []
    y_list = []
    w_list = []
    p0_est_list = [[482.]]
    npar_list = []
    nobs_list = []

    p_indices = []

    # indices
    # new coefficient matrix
    index_2ndparam_list = []  # index of second param in new matrix (first param is always gamma)
    shift_2ndparam_list = []
    index_nobs_list = [
    ]  # index of second param in new matrix (first param is always gamma)
    shift_nobs_list = []
    X_shifted_row_list = []
    X_shifted_col_list = []
    X_shifted_data_list = []

    for dsi, single_double, kwargs in ds_inputs:
        assert kwargs[
            'method'] == 'wls', 'Only weighted calibration campains can be combined'

        kwargs.pop('ds', None)

        if single_double == 'double':
            func = utils.calibration_double_ended_wls
            kwargs_routine = {
                'ds': dsi,
                'st_label': kwargs['st_label'],
                'ast_label': kwargs['ast_label'],
                'rst_label': kwargs['rst_label'],
                'rast_label': kwargs['rast_label'],
                'st_var': kwargs['st_var'],
                'ast_var': kwargs['ast_var'],
                'rst_var': kwargs['rst_var'],
                'rast_var': kwargs['rast_var'],
                'solver': 'external'
            }
        else:
            func = utils.calibration_single_ended_wls
            kwargs_routine = {
                'ds': dsi,
                'st_label': kwargs['st_label'],
                'ast_label': kwargs['ast_label'],
                'st_var': kwargs['st_var'],
                'ast_var': kwargs['ast_var'],
                'solver': 'external'
            }

        X, y, w, p0_est = func(**kwargs_routine)
        X_list.append(X)
        y_list.append(y)
        w_list.append(w)
        p0_est_list.append(p0_est[1:])

        shift_2ndparam = sum(npar_list) - len(npar_list)
        shift_2ndparam_list.append(shift_2ndparam)
        index_2ndparam_list.append(shift_2ndparam + 1)

        shift_nobs = sum(nobs_list)
        shift_nobs_list.append(shift_nobs)
        index_nobs_list.append(shift_nobs)

        nobs_list.append(X.shape[0])
        npar_list.append(X.shape[1])
        p_indices.append(
            np.concatenate(([0], np.arange(shift_2ndparam + 1, X.shape[1] +
                                           shift_2ndparam))))

        # shift all the parameter indices but the gamma
        X_shifted_row_list.append(X.row + shift_nobs)
        X_shifted_data_list.append(X.data)
        X_shifted_col = X.col
        col_shift_mask = X.col > 0
        X_shifted_col[col_shift_mask] += shift_2ndparam
        X_shifted_col_list.append(X_shifted_col)

    # Do nothing with the w and y vector
    w_shifted = np.concatenate(w_list)
    y_shifted = np.concatenate(y_list)
    p0_est_shifted = np.concatenate(p0_est_list)

    # concatenate the different X's
    X_shifted_col = np.concatenate(X_shifted_col_list)
    X_shifted_row = np.concatenate(X_shifted_row_list)
    X_shifted_data = np.concatenate(X_shifted_data_list)

    X_shifted_shape = (sum(nobs_list), sum(npar_list) - len(npar_list) + 1)

    X_shifted = sp.coo_matrix(
        (X_shifted_data, (X_shifted_row, X_shifted_col)),
        shape=X_shifted_shape,
        dtype=dtype,
        copy=False)

    p_sol, p_var, p_cov = utils.wls_sparse(
        X_shifted, y_shifted, w=w_shifted, x0=p0_est_shifted, calc_cov=True)

    for p_i, (dsi, single_double, kwargs) in zip(p_indices, ds_inputs):
        p_soli = p_sol[p_i]
        p_vari = p_var[p_i]
        p_covi = p_cov[p_i, p_i]

        kwargs['nt'] = dsi.time.size
        kwargs['z'] = dsi.ufunc_per_section(label='x', calc_per='all')
        kwargs['p_sol'] = p_soli
        kwargs['p_var'] = p_vari
        kwargs['p_cov'] = p_covi
        kwargs['method'] = 'external'

        if single_double == 'double':
            dsi.calibration_double_ended(**kwargs)
        else:
            dsi.calibration_single_ended(**kwargs)

    pass

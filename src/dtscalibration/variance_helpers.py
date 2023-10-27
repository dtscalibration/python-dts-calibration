import warnings

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.sparse import linalg as ln


def variance_stokes_constant_helper(data_dict):
    def func_fit(p, xs):
        return p[:xs, None] * p[None, xs:]

    def func_cost(p, data, xs):
        fit = func_fit(p, xs)
        return np.sum((fit - data) ** 2)

    resid_list = []

    for k, v in data_dict.items():
        for vi in v:
            nxs, nt = vi.shape
            npar = nt + nxs

            p1 = np.ones(npar) * vi.mean() ** 0.5

            res = minimize(func_cost, p1, args=(vi, nxs), method="Powell")
            assert res.success, "Unable to fit. Try variance_stokes_exponential"

            fit = func_fit(res.x, nxs)
            resid_list.append(fit - vi)

    resid = np.concatenate(resid_list)

    # unbiased estimater ddof=1, originally thought it was npar
    var_I = resid.var(ddof=1)

    return var_I, resid


def variance_stokes_exponential_helper(
    nt, x, y, len_stretch_list, use_statsmodels, suppress_info
):
    n_sections = len(len_stretch_list)  # number of sections
    n_locs = sum(len_stretch_list)  # total number of locations along cable used
    # for reference.

    data1 = x
    data2 = np.ones(sum(len_stretch_list) * nt)
    data = np.concatenate([data1, data2])

    # alpha is NOT the same for all -> one column per section
    coords1row = np.arange(nt * n_locs)
    coords1col = np.hstack(
        [np.ones(in_locs * nt) * i for i, in_locs in enumerate(len_stretch_list)]
    )  # C for

    # second calibration parameter is different per section and per timestep
    coords2row = np.arange(nt * n_locs)
    coords2col = np.hstack(
        [
            np.repeat(
                np.arange(i * nt + n_sections, (i + 1) * nt + n_sections), in_locs
            )
            for i, in_locs in enumerate(len_stretch_list)
        ]
    )  # C for
    coords = (
        np.concatenate([coords1row, coords2row]),
        np.concatenate([coords1col, coords2col]),
    )

    lny = np.log(y)
    w = y.copy()  # 1/std.

    ddof = n_sections + nt * n_sections  # see numpy documentation on ddof

    if use_statsmodels:
        # returns the same answer with statsmodel
        import statsmodels.api as sm

        X = sp.coo_matrix(
            (data, coords), shape=(nt * n_locs, ddof), dtype=float, copy=False
        )

        mod_wls = sm.WLS(lny, X.toarray(), weights=w**2)
        res_wls = mod_wls.fit()
        # print(res_wls.summary())
        a = res_wls.params

    else:
        wdata = data * np.hstack((w, w))
        wX = sp.coo_matrix(
            (wdata, coords),
            shape=(nt * n_locs, n_sections + nt * n_sections),
            dtype=float,
            copy=False,
        )

        wlny = lny * w

        p0_est = np.asarray(n_sections * [0.0] + nt * n_sections * [8])
        # noinspection PyTypeChecker
        a = ln.lsqr(wX, wlny, x0=p0_est, show=not suppress_info, calc_var=False)[0]

    beta = a[:n_sections]
    beta_expand_to_sec = np.hstack(
        [
            np.repeat(float(beta[i]), leni * nt)
            for i, leni in enumerate(len_stretch_list)
        ]
    )
    G = np.asarray(a[n_sections:])
    G_expand_to_sec = np.hstack(
        [
            np.repeat(G[i * nt : (i + 1) * nt], leni)
            for i, leni in enumerate(len_stretch_list)
        ]
    )

    I_est = np.exp(G_expand_to_sec) * np.exp(x * beta_expand_to_sec)
    resid = I_est - y
    var_I = resid.var(ddof=1)
    return var_I, resid


def variance_stokes_linear_helper(st_sec, resid_sec, nbin, through_zero):
    # Adjust nbin silently to fit residuals in
    # rectangular matrix and use numpy for computation
    nbin_ = nbin
    while st_sec.size % nbin_:
        nbin_ -= 1

    if nbin_ != nbin:
        print(f"Adjusting nbin to: {nbin_} to fit residuals in ")
        nbin = nbin_

    isort = np.argsort(st_sec)
    st_sort_mean = st_sec[isort].reshape((nbin, -1)).mean(axis=1)
    st_sort_var = resid_sec[isort].reshape((nbin, -1)).var(axis=1)

    if through_zero:
        # VAR(Stokes) = slope * Stokes
        offset = 0.0
        slope = np.linalg.lstsq(st_sort_mean[:, None], st_sort_var, rcond=None)[0]

    else:
        # VAR(Stokes) = slope * Stokes + offset
        slope, offset = np.linalg.lstsq(
            np.hstack((st_sort_mean[:, None], np.ones((nbin, 1)))),
            st_sort_var,
            rcond=None,
        )[0]

        if offset < 0:
            warnings.warn(
                "Warning! Offset of variance_stokes_linear() "
                "is negative. This is phisically "
                "not possible. Most likely, your Stokes intensities do "
                "not vary enough to fit a linear curve. Either "
                "use `through_zero` option or use "
                "`variance_stokes_constant()`. Another reason "
                "could be that your sections are defined to be "
                "wider than they actually are."
            )

    def var_fun(stokes):
        return slope * stokes + offset

    return slope, offset, st_sort_mean, st_sort_var, resid_sec, var_fun


def check_allclose_acquisitiontime(acquisitiontime, eps: float = 0.05) -> None:
    """
    Check if all acquisition times are of equal duration. For now it is not possible to calibrate
    over timesteps if the acquisition time of timesteps varies, as the Stokes variance
    would change over time.

    The acquisition time is stored for single ended measurements in userAcquisitionTime,
    for double ended measurements in userAcquisitionTimeFW and userAcquisitionTimeBW.

    Parameters
    ----------
    ds : DataStore
    eps : float
        Default accepts 1% of relative variation between min and max acquisition time.

    Returns
    -------
    """
    dtmin = acquisitiontime.min()
    dtmax = acquisitiontime.max()
    dtavg = (dtmin + dtmax) / 2
    assert (
        dtmax - dtmin
    ) / dtavg < eps, "Acquisition time is Forward channel not equal for all time steps"
    pass

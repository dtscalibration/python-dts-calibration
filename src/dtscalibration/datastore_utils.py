from typing import TYPE_CHECKING
from typing import Optional
from typing import Union

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr


def check_dims(
    ds: xr.Dataset,
    labels: Union[list[str], tuple[str]],
    correct_dims: Optional[tuple[str]] = None,
) -> None:
    """
    Compare the dimensions of different labels. For example: ['st', 'rst'].
    If a calculation is performed and the dimensions do not agree, the answers do not
    make sense and the matrices are broadcasted and the memory usage will explode.
    If no correct dims provided the dimensions of the different are compared.

    Parameters
    ----------
    ds :
        The DataStore object to check.
    labels :
        An iterable with labels.
    correct_dims :
        The correct dimensions.

    Returns
    -------

    """
    if not correct_dims:
        assert len(labels) > 1, "Define the correct dimensions"

        for li in labels[1:]:
            assert (
                ds[labels[0]].dims == ds[li].dims
            ), li + " does not have the correct dimensions." " Should be " + str(
                ds[labels[0]].dims
            )
    else:
        for li in labels:
            assert (
                ds[li].dims == correct_dims
            ), li + " does not have the correct dimensions. " "Should be " + str(
                correct_dims
            )


class ParameterIndexDoubleEnded:
    """
    npar = 1 + 2 * nt + nx + 2 * nt * nta
    assert pval.size == npar
    assert p_var.size == npar
    if calc_cov:
        assert p_cov.shape == (npar, npar)
    gamma = pval[0]
    d_fw = pval[1:nt + 1]
    d_bw = pval[1 + nt:1 + 2 * nt]
    alpha = pval[1 + 2 * nt:1 + 2 * nt + nx]
    # store calibration parameters in DataStore
    self["gamma"] = (tuple(), gamma)
    self["alpha"] = (('x',), alpha)
    self["df"] = (('time',), d_fw)
    self["db"] = (('time',), d_bw)
    if nta > 0:
        ta = pval[1 + 2 * nt + nx:].reshape((nt, 2, nta), order='F')
        self['talpha_fw'] = (('time', 'trans_att'), ta[:, 0, :])
        self['talpha_bw'] = (('time', 'trans_att'), ta[:, 1, :])
    """

    def __init__(self, nt, nx, nta, fix_gamma=False, fix_alpha=False):
        self.nt = nt
        self.nx = nx
        self.nta = nta
        self.fix_gamma = fix_gamma
        self.fix_alpha = fix_alpha

    @property
    def all(self):
        return np.concatenate(
            (self.gamma, self.df, self.db, self.alpha, self.ta.flatten(order="F"))
        )

    @property
    def npar(self):
        if not self.fix_gamma and not self.fix_alpha:
            return 1 + 2 * self.nt + self.nx + 2 * self.nt * self.nta
        elif self.fix_gamma and not self.fix_alpha:
            return 2 * self.nt + self.nx + 2 * self.nt * self.nta
        elif not self.fix_gamma and self.fix_alpha:
            return 1 + 2 * self.nt + 2 * self.nt * self.nta
        elif self.fix_gamma and self.fix_alpha:
            return 2 * self.nt + 2 * self.nt * self.nta

    @property
    def gamma(self):
        if self.fix_gamma:
            return []
        else:
            return [0]

    @property
    def df(self):
        if self.fix_gamma:
            return list(range(self.nt))
        else:
            return list(range(1, self.nt + 1))

    @property
    def db(self):
        if self.fix_gamma:
            return list(range(self.nt, 2 * self.nt))
        else:
            return list(range(1 + self.nt, 1 + 2 * self.nt))

    @property
    def alpha(self):
        if self.fix_alpha:
            return []
        elif self.fix_gamma:
            return list(range(2 * self.nt, 1 + 2 * self.nt + self.nx))
        elif not self.fix_gamma:
            return list(range(1 + 2 * self.nt, 1 + 2 * self.nt + self.nx))

    @property
    def ta(self):
        if self.nta == 0:
            return np.zeros((self.nt, 2, 0))
        elif not self.fix_gamma and not self.fix_alpha:
            arr = np.arange(1 + 2 * self.nt + self.nx, self.npar)
        elif self.fix_gamma and not self.fix_alpha:
            arr = np.arange(2 * self.nt + self.nx, self.npar)
        elif not self.fix_gamma and self.fix_alpha:
            arr = np.arange(1 + 2 * self.nt, self.npar)
        elif self.fix_gamma and self.fix_alpha:
            arr = np.arange(2 * self.nt, self.npar)

        return arr.reshape((self.nt, 2, self.nta), order="F")

    @property
    def taf(self):
        """
        Use `.reshape((nt, nta))` to convert array to (time-dim and transatt-dim). Order is the default C order.
        ta = pval[1 + 2 * nt + nx:].reshape((nt, 2, nta), order='F')
        self['talpha_fw'] = (('time', 'trans_att'), ta[:, 0, :])
        """
        return self.ta[:, 0, :].flatten(order="C")

    @property
    def tab(self):
        """
        Use `.reshape((nt, nta))` to convert array to (time-dim and transatt-dim). Order is the default C order.
        ta = pval[1 + 2 * nt + nx:].reshape((nt, 2, nta), order='F')
        self['talpha_bw'] = (('time', 'trans_att'), ta[:, 1, :])
        """
        return self.ta[:, 1, :].flatten(order="C")

    def get_ta_pars(self, pval):
        if self.nta > 0:
            if pval.ndim == 1:
                return np.take_along_axis(pval[None, None], self.ta, axis=-1)

            else:
                # assume shape is (a, npar) and returns shape (nt, 2, nta, a)
                assert pval.shape[1] == self.npar and pval.ndim == 2
                return np.stack([self.get_ta_pars(v) for v in pval], axis=-1)

        else:
            return np.zeros(shape=(self.nt, 2, 0))

    def get_taf_pars(self, pval):
        """returns taf parameters of shape (nt, nta) or (nt, nta, a)"""
        return self.get_ta_pars(pval=pval)[:, 0, :]

    def get_tab_pars(self, pval):
        """returns taf parameters of shape (nt, nta) or (nt, nta, a)"""
        return self.get_ta_pars(pval=pval)[:, 1, :]

    def get_taf_values(self, pval, x, trans_att, axis=""):
        """returns taf parameters of shape (nx, nt)"""
        pval = np.atleast_2d(pval)

        assert pval.ndim == 2 and pval.shape[1] == self.npar

        arr_out = np.zeros((self.nx, self.nt))

        if self.nta == 0:
            pass

        elif axis == "":
            assert pval.shape[0] == 1

            arr = np.transpose(self.get_taf_pars(pval), axes=(1, 2, 0))  # (nta, 1, nt)

            for tai, taxi in zip(arr, trans_att):
                arr_out[x >= taxi] += tai

        elif axis == "x":
            assert pval.shape[0] == self.nx

            arr = np.transpose(self.get_taf_pars(pval), axes=(1, 2, 0))  # (nta, nx, nt)

            for tai, taxi in zip(arr, trans_att):
                # loop over nta, tai has shape (x, t)
                arr_out[x >= taxi] += tai[x >= taxi]

        elif axis == "time":
            assert pval.shape[0] == self.nt

            # arr (nt, nta, nt) to have shape (nta, nt, nt)
            arr = np.transpose(self.get_taf_pars(pval), axes=(1, 2, 0))

            for tai, taxi in zip(arr, trans_att):
                # loop over nta, tai has shape (t, t)
                arr_out[x >= taxi] += np.diag(tai)[None]

        return arr_out

    def get_tab_values(self, pval, x, trans_att, axis=""):
        """returns tab parameters of shape (nx, nt)"""
        assert pval.shape[-1] == self.npar

        arr_out = np.zeros((self.nx, self.nt))

        if self.nta == 0:
            pass

        elif axis == "":
            pval = np.squeeze(pval)
            assert pval.ndim == 1
            arr = np.transpose(self.get_tab_pars(pval), axes=(1, 0))

            for tai, taxi in zip(arr, trans_att):
                arr_out[x < taxi] += tai

        elif axis == "x":
            assert pval.ndim == 2 and pval.shape[0] == self.nx

            arr = np.transpose(self.get_tab_pars(pval), axes=(1, 2, 0))

            for tai, taxi in zip(arr, trans_att):
                # loop over nta, tai has shape (x, t)
                arr_out[x < taxi] += tai[x < taxi]

        elif axis == "time":
            assert pval.ndim == 2 and pval.shape[0] == self.nt

            # arr (nt, nta, nt) to have shape (nta, nt, nt)
            arr = np.transpose(self.get_tab_pars(pval), axes=(1, 2, 0))

            for tai, taxi in zip(arr, trans_att):
                # loop over nta, tai has shape (t, t)
                arr_out[x < taxi] += np.diag(tai)

        return arr_out


class ParameterIndexSingleEnded:
    """
    if parameter fixed, they are not in
    npar = 1 + 1 + nt + nta * nt
    """

    def __init__(self, nt, nx, nta, includes_alpha=False, includes_dalpha=True):
        assert not (
            includes_alpha and includes_dalpha
        ), "Cannot hold both dalpha and alpha"
        self.nt = nt
        self.nx = nx
        self.nta = nta
        self.includes_alpha = includes_alpha
        self.includes_dalpha = includes_dalpha

    @property
    def all(self):
        return np.concatenate(
            (self.gamma, self.dalpha, self.alpha, self.c, self.ta.flatten(order="F"))
        )

    @property
    def npar(self):
        if self.includes_alpha:
            return 1 + self.nx + self.nt + self.nta * self.nt
        elif self.includes_dalpha:
            return 1 + 1 + self.nt + self.nta * self.nt
        else:
            return 1 + self.nt + self.nta * self.nt

    @property
    def gamma(self):
        return [0]

    @property
    def dalpha(self):
        if self.includes_dalpha:
            return [1]
        else:
            return []

    @property
    def alpha(self):
        if self.includes_alpha:
            return list(range(1, 1 + self.nx))
        else:
            return []

    @property
    def c(self):
        if self.includes_alpha:
            return list(range(1 + self.nx, 1 + self.nx + self.nt))
        elif self.includes_dalpha:
            return list(range(1 + 1, 1 + 1 + self.nt))
        else:
            return list(range(1, 1 + self.nt))

    @property
    def taf(self):
        """returns taf parameters of shape (nt, nta) or (nt, nta, a)"""
        # ta = p_val[-nt * nta:].reshape((nt, nta), order='F')
        # self["talpha"] = (('time', 'trans_att'), ta[:, :])
        if self.includes_alpha:
            return np.arange(
                1 + self.nx + self.nt, 1 + self.nx + self.nt + self.nt * self.nta
            ).reshape((self.nt, self.nta), order="F")
        elif self.includes_dalpha:
            return np.arange(
                1 + 1 + self.nt, 1 + 1 + self.nt + self.nt * self.nta
            ).reshape((self.nt, self.nta), order="F")
        else:
            return np.arange(1 + self.nt, 1 + self.nt + self.nt * self.nta).reshape(
                (self.nt, self.nta), order="F"
            )

    def get_taf_pars(self, pval):
        if self.nta > 0:
            if pval.ndim == 1:
                # returns shape (nta, nt)
                assert len(pval) == self.npar, "Length of pval is incorrect"
                return np.stack([pval[tafi] for tafi in self.taf.T])

            else:
                # assume shape is (a, npar) and returns shape (nta, nt, a)
                assert pval.shape[1] == self.npar and pval.ndim == 2
                return np.stack([self.get_taf_pars(v) for v in pval], axis=-1)

        else:
            return np.zeros(shape=(self.nt, 0))

    def get_taf_values(self, pval, x, trans_att, axis=""):
        """returns taf parameters of shape (nx, nt)"""
        # assert pval.ndim == 2 and pval.shape[1] == self.npar

        arr_out = np.zeros((self.nx, self.nt))

        if self.nta == 0:
            pass

        elif axis == "":
            pval = pval.flatten()
            assert pval.shape == (self.npar,)
            arr = self.get_taf_pars(pval)
            assert arr.shape == (
                self.nta,
                self.nt,
            )

            for tai, taxi in zip(arr, trans_att):
                arr_out[x >= taxi] += tai

        elif axis == "x":
            assert pval.shape == (self.nx, self.npar)
            arr = self.get_taf_pars(pval)
            assert arr.shape == (self.nta, self.nx, self.nt)

            for tai, taxi in zip(arr, trans_att):
                # loop over nta, tai has shape (x, t)
                arr_out[x >= taxi] += tai[x >= taxi]

        elif axis == "time":
            assert pval.shape == (self.nt, self.npar)
            arr = self.get_taf_pars(pval)
            assert arr.shape == (self.nta, self.nt, self.nt)

            for tai, taxi in zip(arr, trans_att):
                # loop over nta, tai has shape (t, t)
                arr_out[x >= taxi] += np.diag(tai)[None]

        return arr_out


def check_deprecated_kwargs(kwargs):
    """
    Internal function that parses the `kwargs` for depreciated keyword
    arguments.

    Depreciated keywords raise an error, pending to be depreciated do not.
    But this requires that the code currently deals with those arguments.

    Parameters
    ----------
    kwargs : Dict
        A dictionary with keyword arguments.

    Returns
    -------

    """
    msg = """Previously, it was possible to manually set the label from
    which the Stokes and anti-Stokes were read within the DataStore
    object. To reduce the clutter in the code base and be able to
    maintain it, this option was removed.
    See: https://github.com/dtscalibration/python-dts-calibration/issues/81

    The new **fixed** names are: st, ast, rst, rast.

    It is still possible to use the previous defaults, for example when
    reading stored measurements from netCDF, by renaming the labels. The
    old default labels were ST, AST, REV-ST, REV-AST.

    ```
    ds = open_datastore(path_to_old_file)
    ds = ds.rename_labels()
    ds.calibration_double_ended(
        st_var=1.5,
        ast_var=1.5,
        rst_var=1.,
        rast_var=1.,
        method='wls')
    ```

    ds.tmpw.plot()
    """
    list_of_depr = [
        "st_label",
        "ast_label",
        "rst_label",
        "rast_label",
        "transient_asym_att_x",
        "transient_att_x",
    ]
    list_of_pending_depr = []

    kwargs = {k: v for k, v in kwargs.items() if k not in list_of_pending_depr}

    for k in kwargs:
        if k in list_of_depr:
            raise NotImplementedError(msg)

    if len(kwargs) != 0:
        raise NotImplementedError(
            "The following keywords are not " + "supported: " + ", ".join(kwargs.keys())
        )

    pass


def get_netcdf_encoding(
    ds: xr.Dataset, zlib: bool = True, complevel: int = 5, **kwargs
) -> dict:
    """Get default netcdf compression parameters. The same for each data variable.

    TODO: Truncate precision to XML precision per data variable


    Parameters
    ----------
    zlib
    complevel
    ds : DataStore

    Returns
    -------
    encoding:
        Encoding dictionary.
    """
    comp = dict(zlib=zlib, complevel=complevel)
    comp.update(kwargs)
    encoding = {var: comp for var in ds.data_vars}

    return encoding


def get_params_from_pval_double_ended(ip, coords, p_val=None, p_cov=None):
    if p_val is not None:
        assert len(p_val) == ip.npar, "Length of p_val is incorrect"

        params = xr.Dataset(coords=coords)

        # save estimates and variances to datastore, skip covariances
        params["gamma"] = (tuple(), p_val[ip.gamma].item())
        params["alpha"] = (("x",), p_val[ip.alpha])
        params["df"] = (("time",), p_val[ip.df])
        params["db"] = (("time",), p_val[ip.db])

        if ip.nta:
            params["talpha_fw"] = (
                ("time", "trans_att"),
                p_val[ip.taf].reshape((ip.nt, ip.nta), order="C"),
            )
            params["talpha_bw"] = (
                ("time", "trans_att"),
                p_val[ip.tab].reshape((ip.nt, ip.nta), order="C"),
            )
        else:
            params["talpha_fw"] = (("time", "trans_att"), np.zeros((ip.nt, 0)))
            params["talpha_bw"] = (("time", "trans_att"), np.zeros((ip.nt, 0)))

        params["talpha_fw_full"] = (
            ("x", "time"),
            ip.get_taf_values(
                pval=p_val,
                x=params.x.values,
                trans_att=params.trans_att.values,
                axis="",
            ),
        )
        params["talpha_bw_full"] = (
            ("x", "time"),
            ip.get_tab_values(
                pval=p_val,
                x=params.x.values,
                trans_att=params.trans_att.values,
                axis="",
            ),
        )
    if p_cov is not None:
        assert p_cov.shape == (ip.npar, ip.npar), "Shape of p_cov is incorrect"

        # extract covariances and ensure broadcastable to (nx, nt)
        params["gamma_df"] = (("time",), p_cov[np.ix_(ip.gamma, ip.df)][0])
        params["gamma_db"] = (("time",), p_cov[np.ix_(ip.gamma, ip.db)][0])
        params["gamma_alpha"] = (("x",), p_cov[np.ix_(ip.alpha, ip.gamma)][:, 0])
        params["df_db"] = (
            ("time",),
            p_cov[ip.df, ip.db],
        )
        params["alpha_df"] = (
            (
                "x",
                "time",
            ),
            p_cov[np.ix_(ip.alpha, ip.df)],
        )
        params["alpha_db"] = (
            (
                "x",
                "time",
            ),
            p_cov[np.ix_(ip.alpha, ip.db)],
        )
        params["tafw_gamma"] = (
            (
                "x",
                "time",
            ),
            ip.get_taf_values(
                pval=p_cov[ip.gamma],
                x=params["x"].values,
                trans_att=params["trans_att"].values,
                axis="",
            ),
        )
        params["tabw_gamma"] = (
            (
                "x",
                "time",
            ),
            ip.get_tab_values(
                pval=p_cov[ip.gamma],
                x=params["x"].values,
                trans_att=params["trans_att"].values,
                axis="",
            ),
        )
        params["tafw_alpha"] = (
            (
                "x",
                "time",
            ),
            ip.get_taf_values(
                pval=p_cov[ip.alpha],
                x=params["x"].values,
                trans_att=params["trans_att"].values,
                axis="x",
            ),
        )
        params["tabw_alpha"] = (
            (
                "x",
                "time",
            ),
            ip.get_tab_values(
                pval=p_cov[ip.alpha],
                x=params["x"].values,
                trans_att=params["trans_att"].values,
                axis="x",
            ),
        )
        params["tafw_df"] = (
            (
                "x",
                "time",
            ),
            ip.get_taf_values(
                pval=p_cov[ip.df],
                x=params["x"].values,
                trans_att=params["trans_att"].values,
                axis="time",
            ),
        )
        params["tafw_db"] = (
            (
                "x",
                "time",
            ),
            ip.get_taf_values(
                pval=p_cov[ip.db],
                x=params["x"].values,
                trans_att=params["trans_att"].values,
                axis="time",
            ),
        )
        params["tabw_db"] = (
            (
                "x",
                "time",
            ),
            ip.get_tab_values(
                pval=p_cov[ip.db],
                x=params["x"].values,
                trans_att=params["trans_att"].values,
                axis="time",
            ),
        )
        params["tabw_df"] = (
            (
                "x",
                "time",
            ),
            ip.get_tab_values(
                pval=p_cov[ip.df],
                x=params["x"].values,
                trans_att=params["trans_att"].values,
                axis="time",
            ),
        )
        # sigma2_tafw_tabw
    return params


def get_params_from_pval_single_ended(
    ip, coords, p_val=None, p_var=None, p_cov=None, fix_alpha=None
):
    if p_val is not None:
        assert len(p_val) == ip.npar, "Length of p_val is incorrect"

    if p_var is not None:
        assert len(p_var) == ip.npar, "Length of p_var is incorrect"

    if p_cov is not None:
        assert p_cov.shape == (ip.npar, ip.npar), "Shape of p_cov is incorrect"

    params = xr.Dataset(coords=coords)
    param_covs = xr.Dataset(coords=coords)

    params["gamma"] = (tuple(), p_val[ip.gamma].item())
    param_covs["gamma"] = (tuple(), p_var[ip.gamma].item())

    if ip.nta > 0:
        params["talpha_fw"] = (
            ("trans_att", "time"),
            ip.get_taf_pars(p_val),
        )
        param_covs["talpha_fw"] = (
            ("trans_att", "time"),
            ip.get_taf_pars(p_var),
        )
    else:
        params["talpha_fw"] = (("trans_att", "time"), np.zeros((0, ip.nt)))
        param_covs["talpha_fw"] = (("trans_att", "time"), np.zeros((0, ip.nt)))

    params["c"] = (("time",), p_val[ip.c])
    param_covs["c"] = (("time",), p_var[ip.c])

    if fix_alpha is not None:
        params["alpha"] = (("x",), fix_alpha[0])
        param_covs["alpha"] = (("x",), fix_alpha[1])

    else:
        params["dalpha"] = (tuple(), p_val[ip.dalpha].item())
        param_covs["dalpha"] = (tuple(), p_var[ip.dalpha].item())

        params["alpha"] = params["dalpha"] * params["x"]
        param_covs["alpha"] = param_covs["dalpha"] * params["x"] ** 2

        param_covs["gamma_dalpha"] = (tuple(), p_cov[np.ix_(ip.dalpha, ip.gamma)][0, 0])
        param_covs["dalpha_c"] = (("time",), p_cov[np.ix_(ip.dalpha, ip.c)][0, :])
        param_covs["tafw_dalpha"] = (
            ("x", "time"),
            ip.get_taf_values(
                pval=p_cov[ip.dalpha],
                x=params["x"].values,
                trans_att=params["trans_att"].values,
                axis="",
            ),
        )

    params["talpha_fw_full"] = (
        ("x", "time"),
        ip.get_taf_values(
            pval=p_val,
            x=params["x"].values,
            trans_att=params["trans_att"].values,
            axis="",
        ),
    )
    param_covs["talpha_fw_full"] = (
        ("x", "time"),
        ip.get_taf_values(
            pval=p_var,
            x=params["x"].values,
            trans_att=params["trans_att"].values,
            axis="",
        ),
    )
    param_covs["gamma_c"] = (("time",), p_cov[np.ix_(ip.gamma, ip.c)][0, :])
    param_covs["tafw_gamma"] = (
        ("x", "time"),
        ip.get_taf_values(
            pval=p_cov[ip.gamma],
            x=params["x"].values,
            trans_att=params["trans_att"].values,
            axis="",
        ),
    )
    param_covs["tafw_c"] = (
        ("x", "time"),
        ip.get_taf_values(
            pval=p_cov[ip.c],
            x=params["x"].values,
            trans_att=params["trans_att"].values,
            axis="time",
        ),
    )
    return params, param_covs


def merge_double_ended(
    ds_fw: xr.Dataset,
    ds_bw: xr.Dataset,
    cable_length: float,
    plot_result: bool = True,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Some measurements are not set up on the DTS-device as double-ended
    meausurements. This means that the two channels have to be merged manually.

    This function can merge two single-ended DataStore objects into a single
    double-ended DataStore. There is no interpolation, the two arrays are
    flipped and overlayed, based on the entered cable length. This can
    introduce spatial inaccuracies with extremely long cables.

    Parameters
    ----------
    ds_fw : DataSore object
        DataStore object representing the forward measurement channel
    ds_bw : DataSore object
        DataStore object representing the backward measurement channel
    cable_length : float
        Manually estimated cable length to base alignment on
    plot_result : bool
        Plot the aligned Stokes of the forward and backward channels
    verbose : bool

    Returns
    -------
    ds : DataStore object
        With the two channels merged
    """
    assert (
        ds_fw.attrs["isDoubleEnded"] == "0" and ds_bw.attrs["isDoubleEnded"] == "0"
    ), "(one of the) input DataStores is already double ended"

    ds_fw, ds_bw = merge_double_ended_times(ds_fw, ds_bw, verbose=verbose)

    ds = ds_fw.copy()
    ds_bw = ds_bw.copy()

    ds_bw["x"] = cable_length - ds_bw["x"].values

    # TODO: check if reindexing matters, and should be used.
    # one way to do it is performed below, but this could create artifacts
    x_resolution = ds["x"].values[1] - ds["x"].values[0]
    ds_bw = ds_bw.reindex(
        {"x": ds["x"]}, method="nearest", tolerance=0.99 * x_resolution
    )

    ds_bw = ds_bw.sortby("x")

    ds["rst"] = (["x", "time"], ds_bw["st"].values)
    ds["rast"] = (["x", "time"], ds_bw["ast"].values)

    ds = ds.dropna(dim="x")

    ds.attrs["isDoubleEnded"] = "1"
    ds["userAcquisitionTimeBW"] = ("time", ds_bw["userAcquisitionTimeFW"].values)

    if plot_result:
        _, ax = plt.subplots()
        ds["st"].isel(time=0).plot(ax=ax, label="Stokes forward")
        ds["rst"].isel(time=0).plot(ax=ax, label="Stokes backward")
        ax.legend()

    return ds


def merge_double_ended_times(
    ds_fw: xr.Dataset,
    ds_bw: xr.Dataset,
    verify_timedeltas: bool = True,
    verbose: bool = True,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Helper for `merge_double_ended()` to deal with missing measurements. The
    number of measurements of the forward and backward channels might get out
    of sync if the device shuts down before the measurement of the last channel
    is complete. This skips all measurements that are not accompanied by a partner
    channel.

    Provides little protection against swapping fw and bw.

    If all measurements are recorded: fw_t0, bw_t0, fw_t1, bw_t1, fw_t2, bw_t2, ..
        > all are passed

    If some are missing the accompanying measurement is skipped:
     - fw_t0, bw_t0, bw_t1, fw_t2, bw_t2, .. > fw_t0, bw_t0, fw_t2, bw_t2, ..
     - fw_t0, bw_t0, fw_t1, fw_t2, bw_t2, .. > fw_t0, bw_t0, fw_t2, bw_t2, ..
     - fw_t0, bw_t0, bw_t1, fw_t2, fw_t3, bw_t3, .. > fw_t0, bw_t0, fw_t3, bw_t3,

    Mixing forward and backward channels can be problematic when there is a pause
    after measuring all channels. This function is not perfect as the following
    situation is not caught:
     - fw_t0, bw_t0, fw_t1, bw_t2, fw_t3, bw_t3, ..
        > fw_t0, bw_t0, fw_t1, bw_t2, fw_t3, bw_t3, ..

    This routine checks that the lowest channel
    number is measured first (aka the forward channel), but it doesn't catch the
    last case as it doesn't know that fw_t1 and bw_t2 belong to different cycles.

    Parameters
    ----------
    ds_fw : DataSore object
        DataStore object representing the forward measurement channel
    ds_bw : DataSore object
        DataStore object representing the backward measurement channel
    verify_timedeltas : bool
        Check whether times between forward and backward measurements are similar to
        those of neighboring measurements
    verbose : bool
        Print additional information to screen

    Returns
    -------
    ds_fw_sel : DataSore object
        DataStore object representing the forward measurement channel with
        only times for which there is also a ds_bw measurement
    ds_bw_sel : DataSore object
        DataStore object representing the backward measurement channel with
        only times for which there is also a ds_fw measurement
    """
    if "forward channel" in ds_fw.attrs and "forward channel" in ds_bw.attrs:
        assert (
            ds_fw.attrs["forward channel"] < ds_bw.attrs["forward channel"]
        ), "ds_fw and ds_bw are swapped"
    elif (
        "forwardMeasurementChannel" in ds_fw.attrs
        and "forwardMeasurementChannel" in ds_bw.attrs
    ):
        assert (
            ds_fw.attrs["forwardMeasurementChannel"]
            < ds_bw.attrs["forwardMeasurementChannel"]
        ), "ds_fw and ds_bw are swapped"

    # Are all dt's within 1.5 seconds from one another?
    if (ds_bw.time.size == ds_fw.time.size) and np.all(
        ds_bw.time.values > ds_fw.time.values
    ):
        if verify_timedeltas:
            dt_ori = (ds_bw.time.values - ds_fw.time.values) / np.array(
                1, dtype="timedelta64[s]"
            )
            dt_all_close = np.allclose(dt_ori, dt_ori[0], atol=1.5, rtol=0.0)
        else:
            dt_all_close = True

        if dt_all_close:
            return ds_fw, ds_bw

    iuse_chfw = list()
    iuse_chbw = list()

    times_fw = {k: ("fw", i) for i, k in enumerate(ds_fw.time.values)}
    times_bw = {k: ("bw", i) for i, k in enumerate(ds_bw.time.values)}
    times_all = dict(sorted(({**times_fw, **times_bw}).items()))
    times_all_val = list(times_all.values())

    for (direction, ind), (direction_next, ind_next) in zip(
        times_all_val[:-1], times_all_val[1:]
    ):
        if direction == "fw" and direction_next == "bw":
            iuse_chfw.append(ind)
            iuse_chbw.append(ind_next)

        elif direction == "bw" and direction_next == "fw":
            pass

        elif direction == "fw" and direction_next == "fw":
            if verbose:
                print(
                    f"Missing backward measurement beween {ds_fw.time.values[ind]} and {ds_fw.time.values[ind_next]}"
                )

        elif direction == "bw" and direction_next == "bw":
            if verbose:
                print(
                    f"Missing forward measurement beween {ds_bw.time.values[ind]} and {ds_bw.time.values[ind_next]}"
                )

    # throw out is dt differs from its neighbors
    if verify_timedeltas:
        dt = (
            ds_bw.isel(time=iuse_chbw).time.values
            - ds_fw.isel(time=iuse_chfw).time.values
        ) / np.timedelta64(1, "s")
        leaveout = np.zeros_like(dt, dtype=bool)
        leaveout[1:-1] = np.isclose(dt[:-2], dt[2:], atol=1.5, rtol=0.0) * ~np.isclose(
            dt[:-2], dt[1:-1], atol=1.5, rtol=0.0
        )
        iuse_chfw2 = np.array(iuse_chfw)[~leaveout]
        iuse_chbw2 = np.array(iuse_chbw)[~leaveout]

        if verbose:
            for itfw, itbw in zip(
                np.array(iuse_chfw)[leaveout], np.array(iuse_chbw)[leaveout]
            ):
                print(
                    "The following measurements do not belong together, as the time difference\n"
                    "between the\forward and backward measurements is more than 1.5 seconds\n"
                    "larger than the neighboring measurements.\n"
                    f"FW: {ds_fw.isel(time=itfw).time.values} and BW: {ds_bw.isel(time=itbw).time.values}"
                )
        return ds_fw.isel(time=iuse_chfw2), ds_bw.isel(time=iuse_chbw2)

    return ds_fw.isel(time=iuse_chfw), ds_bw.isel(time=iuse_chbw)


def shift_double_ended(
    ds: xr.Dataset, i_shift: int, verbose: bool = True
) -> xr.Dataset:
    """
    The cable length was initially configured during the DTS measurement. For double ended
    measurements it is important to enter the correct length so that the forward channel and the
    backward channel are aligned.

    This function can be used to shift the backward channel so that the forward channel and the
    backward channel are aligned. The backward channel is shifted per index
    along the x dimension.

    The DataStore object that is returned only has values for the backscatter that are measured
    for both the forward channel and the backward channel in the shifted object

    There is no interpolation, as this would alter the accuracy.


    Parameters
    ----------
    ds : DataSore object
        DataStore object that needs to be shifted
    i_shift : int
        if i_shift < 0, the cable was configured to be too long and there is too much data
        recorded. If i_shift > 0, the cable was configured to be too short and part of the cable is
        not measured
    verbose: bool
        If True, the function will inform the user which variables are
        dropped. If False, the function will silently drop the variables.

    Returns
    -------
    ds2 : DataStore object
        With a shifted x-axis
    """
    assert isinstance(i_shift, (int, np.integer))

    nx = ds.x.size
    nx2 = nx - i_shift
    if i_shift < 0:
        # The cable was configured to be too long.
        # There is too much data recorded.
        st = ds.st.data[:i_shift]
        ast = ds.ast.data[:i_shift]
        rst = ds.rst.data[-i_shift:]
        rast = ds.rast.data[-i_shift:]
        x2 = ds.x.data[:i_shift]
        # TMP2 = ds.tmp.data[:i_shift]

    else:
        # The cable was configured to be too short.
        # Part of the cable is not measured.
        st = ds["st"].data[i_shift:]
        ast = ds["ast"].data[i_shift:]
        rst = ds["rst"].data[:nx2]
        rast = ds["rast"].data[:nx2]
        x2 = ds["x"].data[i_shift:]
        # TMP2 = ds.tmp.data[i_shift:]

    d2_coords = dict(ds.coords)
    d2_coords["x"] = xr.DataArray(data=x2, dims=("x",), attrs=ds["x"].attrs)

    d2_data = dict(ds.data_vars)
    for k in ds.data_vars:
        if "x" in ds[k].dims and k in d2_data:
            del d2_data[k]

    new_data = (("st", st), ("ast", ast), ("rst", rst), ("rast", rast))

    for k, v in new_data:
        d2_data[k] = xr.DataArray(data=v, dims=ds[k].dims, attrs=ds[k].attrs)

    not_included = [k for k in ds.data_vars if k not in d2_data]
    if not_included and verbose:
        print("I dont know what to do with the following data", not_included)

    return xr.Dataset(data_vars=d2_data, coords=d2_coords, attrs=ds.attrs)


def suggest_cable_shift_double_ended(
    ds: xr.Dataset,
    irange: npt.NDArray[np.int_],
    plot_result: bool = True,
    **fig_kwargs,
) -> tuple[int, int]:
    """The cable length was initially configured during the DTS measurement.
    For double ended measurements it is important to enter the correct length
    so that the forward channel and the backward channel are aligned.

    This function can be used to find the shift of the backward channel so
    that the forward channel and the backward channel are aligned. The shift
    index refers to the x-dimension.

    The attenuation should be approximately a straight line with jumps at the
    splices. Temperature independent and invariant over time. The following
    objective functions seems to do the job at determining the best shift for
    which the attenuation is most straight.

    Err1 sums the first derivative. Is a proxy for the length of the
    attenuation line. Err2 sums the second derivative. Is a proxy for the
    wiggelyness of the line.

    The top plot shows the origional Stokes and the origional and shifted
    anti-Stokes The bottom plot is generated that shows the two objective
    functions


    Parameters
    ----------
    ds : Xarray Dataset
    irange : array-like
        a numpy array with data of type int. Containing all the shift index
        that are tested.
        Example: np.arange(-250, 200, 1, dtype=int). It shifts the return
        scattering with 250 indices. Calculates err1 and err2. Then shifts
        the return scattering with 249 indices. Calculates err1 and err2. The
        lowest err1 and err2 are suggested as best shift options.
    plot_result : bool
        Plot the summed error as a function of the shift.

    Returns
    -------
    ishift1: int
        Suggested shift based on Err1
    ishift2: int
        Suggested shift based on Err2
    """
    err1 = []
    err2 = []

    for shift in irange:
        i_shift = int(shift)  # int() because iterating irange does not narrow type.
        nx = ds["x"].size
        nx2 = nx - i_shift
        if i_shift < 0:
            # The cable was configured to be too long. There is too much data recorded.
            st = ds["st"].data[:i_shift]
            ast = ds["ast"].data[:i_shift]
            rst = ds["rst"].data[-i_shift:]
            rast = ds["rast"].data[-i_shift:]
            x2 = ds["x"].data[:i_shift]
        else:
            # The cable was configured to be too short. Part of the cable is not measured.
            st = ds["st"].data[i_shift:]
            ast = ds["ast"].data[i_shift:]
            rst = ds["rst"].data[:nx2]
            rast = ds["rast"].data[:nx2]
            x2 = ds["x"].data[i_shift:]

        i_f = np.log(st / ast)
        i_b = np.log(rst / rast)

        att = (i_b - i_f) / 2  # varianble E in article

        att_dif1 = np.diff(att, n=1, axis=0)
        att_x_dif1 = 0.5 * x2[1:] + 0.5 * x2[:-1]
        err1_mask = np.logical_and(att_x_dif1 > 1.0, att_x_dif1 < 150.0)
        err1.append(np.nansum(np.abs(att_dif1[err1_mask])))

        att_dif2 = np.diff(att, n=2, axis=0)
        att_x_dif2 = x2[1:-1]
        err2_mask = np.logical_and(att_x_dif2 > 1.0, att_x_dif2 < 150.0)
        err2.append(np.nansum(np.abs(att_dif2[err2_mask])))

    # int() is required for typing.
    ishift1 = int(irange[np.argmin(err1, axis=0)])
    ishift2 = int(irange[np.argmin(err2, axis=0)])

    if plot_result:
        if fig_kwargs is None:
            fig_kwargs = {}

        f, (ax0, ax1) = plt.subplots(2, 1, sharex=False, **fig_kwargs)
        f.suptitle(f"best shift is {ishift1} or {ishift2}")

        dt = ds.isel(time=0)
        x = dt["x"].data
        y = dt["st"].data
        ax0.plot(x, y, label="ST original")
        y = dt["rst"].data
        ax0.plot(x, y, label="REV-ST original")

        dtsh1 = shift_double_ended(dt, ishift1)
        dtsh2 = shift_double_ended(dt, ishift2)
        x1 = dtsh1["x"].data
        x2 = dtsh2["x"].data
        y1 = dtsh1["rst"].data
        y2 = dtsh2["rst"].data
        ax0.plot(x1, y1, label=f"ST i_shift={ishift1}")
        ax0.plot(x2, y2, label=f"ST i_shift={ishift2}")
        ax0.set_xlabel("x (m)")
        ax0.legend()

        ax2 = ax1.twinx()
        ax1.plot(irange, err1, c="red", label="1 deriv")
        ax2.plot(irange, err2, c="blue", label="2 deriv")
        ax1.axvline(
            ishift1, c="red", linewidth=0.8, label=f"1 deriv. i_shift={ishift1}"
        )
        ax2.axvline(
            ishift2, c="blue", linewidth=0.8, label=f"2 deriv. i_shift={ishift1}"
        )
        ax1.set_xlabel("i_shift")
        ax1.legend(loc=2)  # left axis
        ax2.legend(loc=1)  # right axis

        plt.tight_layout()

    return ishift1, ishift2


def ufunc_per_section_helper(
    sections=None,
    func=None,
    x_coords=None,
    reference_dataset=None,
    dataarray=None,
    subtract_from_dataarray=None,
    subtract_reference_from_dataarray=False,
    ref_temp_broadcasted=False,
    calc_per="stretch",
    **func_kwargs,
):
    """
    User function applied to parts of the cable. Super useful,
    many options and slightly
    complicated.

    The function `func` is taken over all the timesteps and calculated
    per `calc_per`. This
    is returned as a dictionary

    Parameters
    ----------
    sections : Dict[str, List[slice]], optional
        If `None` is supplied, `ds.sections` is used. Define calibration
        sections. Each section requires a reference temperature time series,
        such as the temperature measured by an external temperature sensor.
        They should already be part of the DataStore object. `sections`
        is defined with a dictionary with its keywords of the
        names of the reference temperature time series. Its values are
        lists of slice objects, where each slice object is a fiber stretch
        that has the reference temperature. Afterwards, `sections` is stored
        under `ds.sections`.
    func : callable, str
        A numpy function, or lambda function to apply to each 'calc_per'.
    x_coords : xarray.DataArray, optional
        x-coordinates, stored as ds.x. If supplied, returns the x-indices of
        the reference sections.
    reference_dataset : xarray.Dataset or Dict, optional
        Contains the reference temperature timeseries refered to in `sections`.
        Not required if `x_indices`.
    dataarray : xarray.DataArray, optional
        Pass your DataArray of which you want to compute the statistics. Has an
        (x,) dimension or (x, time) dimensions.
    subtract_from_dataarray : xarray.DataArray, optional
        Pass your DataArray of which you want to subtract from `dataarray` before you
        compute the statistics. Has an (x,) dimension or (x, time) dimensions.
    subtract_reference_from_dataarray : bool
        If True the reference temperature according to sections is subtracted from
        dataarray before computing statistics
    ref_temp_broadcasted : bool
        Use if you want to return the reference temperature of shape of the reference
        sections
    calc_per : {'all', 'section', 'stretch'}
    func_kwargs : dict
        Dictionary with options that are passed to func

    Returns
    -------

    Examples
    --------

    1. Calculate the variance of the residuals in the along ALL the\
    reference sections wrt the temperature of the water baths

    >>> tmpf_var = ufunc_per_section_helper(
    >>>     func='var',
    >>>     calc_per='all',
    >>>     dataarray=d['tmpf'],
    >>>     subtract_reference_from_dataarray=True)

    2. Calculate the variance of the residuals in the along PER\
    reference section wrt the temperature of the water baths

    >>> tmpf_var = ufunc_per_section_helper
    >>>     func='var',
    >>>     calc_per='stretch',
    >>>     dataarray=d['tmpf'],
    >>>     subtract_reference_from_dataarray=True)

    3. Calculate the variance of the residuals in the along PER\
    water bath wrt the temperature of the water baths

    >>> tmpf_var = ufunc_per_section_helper(
    >>>     func='var',
    >>>     calc_per='section',
    >>>     dataarray=d['tmpf'],
    >>>     subtract_reference_from_dataarray=True)

    4. Obtain the coordinates of the measurements per section

    >>> locs = ufunc_per_section_helper(
    >>>     func=None,
    >>>     dataarray=d.x,
    >>>     subtract_reference_from_dataarray=False,
    >>>     ref_temp_broadcasted=False,
    >>>     calc_per='stretch')

    5. Number of observations per stretch

    >>> nlocs = ufunc_per_section_helper(
    >>>     func=len,
    >>>     dataarray=d.x,
    >>>     subtract_reference_from_dataarray=False,
    >>>     ref_temp_broadcasted=False,
    >>>     calc_per='stretch')

    6. broadcast the temperature of the reference sections to\
    stretch/section/all dimensions. The value of the reference\
    temperature (a timeseries) is broadcasted to the shape of self[\
    label]. The dataarray is not used for anything else.

    >>> temp_ref = ufunc_per_section_helper(
    >>>     dataarray=d["st"],
    >>>     ref_temp_broadcasted=True,
    >>>     calc_per='all')

    7. x-coordinate index

    >>> ix_loc = ufunc_per_section_helper(x_coords=d.x)


    Note
    ----
    If `dataarray` or `subtract_from_dataarray` is a Dask array, a Dask
    array is returned else a numpy array is returned
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

    elif isinstance(func, str) and func == "var":

        def func(a):
            """

            Parameters
            ----------
            a

            Returns
            -------

            """
            return np.var(a, ddof=1)

    else:
        assert callable(func)

    assert calc_per in ["all", "section", "stretch"]
    assert "x_indices" not in func_kwargs, "pass x_coords arg instead"

    if x_coords is None and (
        (dataarray is not None and hasattr(dataarray.data, "chunks"))
        or (subtract_from_dataarray and hasattr(subtract_from_dataarray.data, "chunks"))
    ):
        concat = da.concatenate
    else:
        concat = np.concatenate

    out = dict()

    for k, section in sections.items():
        out[k] = []
        for stretch in section:
            if x_coords is not None:
                # get indices from stretches
                assert subtract_from_dataarray is None
                assert not subtract_reference_from_dataarray
                assert not ref_temp_broadcasted
                assert not func_kwargs, "Unsupported kwargs"

                # so it is slicable with x-indices
                _x_indices = x_coords.astype(int) * 0 + np.arange(x_coords.size)
                arg1 = _x_indices.sel(x=stretch).data
                out[k].append(arg1)

            elif (
                subtract_from_dataarray is not None
                and not subtract_reference_from_dataarray
                and not ref_temp_broadcasted
            ):
                # calculate std wrt other series
                arg1 = dataarray.sel(x=stretch).data
                arg2 = subtract_from_dataarray.sel(x=stretch).data
                out[k].append(arg1 - arg2)

            elif (
                subtract_from_dataarray is None
                and subtract_reference_from_dataarray
                and not ref_temp_broadcasted
            ):
                # calculate std wrt reference temperature of the corresponding bath
                arg1 = dataarray.sel(x=stretch).data
                arg2 = reference_dataset[k].data
                out[k].append(arg1 - arg2)

            elif (
                subtract_from_dataarray is None
                and not subtract_reference_from_dataarray
                and ref_temp_broadcasted
            ):
                # Broadcast the reference temperature to the length of the stretch
                arg1 = dataarray.sel(x=stretch).data
                arg2 = da.broadcast_to(reference_dataset[k].data, arg1.shape)
                out[k].append(arg2)

            elif (
                subtract_from_dataarray is None
                and not subtract_reference_from_dataarray
                and not ref_temp_broadcasted
            ):
                # calculate std wrt mean value
                arg1 = dataarray.sel(x=stretch).data
                out[k].append(arg1)

        if calc_per == "stretch":
            out[k] = [func(argi, **func_kwargs) for argi in out[k]]

        elif calc_per == "section":
            # flatten the out_dict to sort them
            start = [i.start for i in section]
            i_sorted = np.argsort(start)
            out_flat_sort = [out[k][i] for i in i_sorted]
            out[k] = func(concat(out_flat_sort), **func_kwargs)

        elif calc_per == "all":
            pass

    if calc_per == "all":
        # flatten the out_dict to sort them
        start = [item.start for sublist in sections.values() for item in sublist]
        i_sorted = np.argsort(start)
        out_flat = [item for sublist in out.values() for item in sublist]
        out_flat_sort = [out_flat[i] for i in i_sorted]
        out = func(concat(out_flat_sort, axis=0), **func_kwargs)

        if hasattr(out, "chunks") and len(out.chunks) > 0 and "x" in dataarray.dims:
            # also sum the chunksize in the x dimension
            # first find out where the x dim is
            ixdim = dataarray.dims.index("x")
            c_old = out.chunks
            c_new = list(c_old)
            c_new[ixdim] = sum(c_old[ixdim])
            out = out.rechunk(c_new)

    return out

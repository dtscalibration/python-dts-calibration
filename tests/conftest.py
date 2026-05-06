"""Shared pytest fixtures and helpers for the dtscalibration test suite.

This module is auto-loaded by pytest. Test modules can either rely on
pytest's fixture discovery (no import needed) or import helpers explicitly
via ``from conftest import assert_almost_equal_verbose``.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from xarray import Dataset


def assert_almost_equal_verbose(actual, desired, verbose=False, **kwargs):
    """Assert two arrays are almost equal and report the achieved precision.

    Parameters
    ----------
    actual : array_like
        Array obtained from the computation under test.
    desired : array_like
        Reference values. Broadcast to the shape of ``actual``.
    verbose : bool, optional
        If True, print the achieved precision (decimals) to stdout.
    **kwargs
        Forwarded to ``numpy.testing.assert_almost_equal`` (e.g. ``decimal``).

    Notes
    -----
    On exact match, the assertion is delegated to
    ``numpy.testing.assert_array_equal`` and the reported precision is
    ``decimal=inf``. Previously, the helper coerced ``decimal=NaN`` to a
    constant ``18.0`` which silently masked the difference between
    machine-exact and ordinary default-precision asserts.
    """
    actual_arr = np.asarray(actual)
    desired_arr = np.broadcast_to(desired, actual_arr.shape)
    err = np.abs(actual_arr - desired_arr).max()

    if err == 0:
        if verbose:
            print("\n>>>>>The actual precision is: inf")
        np.testing.assert_array_equal(actual_arr, desired_arr)
        return

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log10")
        dec = -np.ceil(np.log10(err))

    if not np.isfinite(dec):
        dec = float("inf")

    m = "\n>>>>>The actual precision is: " + str(float(dec))

    if verbose:
        print(m)

    np.testing.assert_almost_equal(actual_arr, desired_arr, err_msg=m, **kwargs)


def _build_synthetic_double_ended(
    nx=100,
    nt=50,
    cable_len=100.0,
    ts_cold=4.0,
    ts_warm=20.0,
    noise_st=0.0,
    noise_ast=0.0,
    noise_rst=0.0,
    noise_rast=0.0,
    seed=0,
):
    """Build a synthetic double-ended DTS dataset with known truth.

    Parameters
    ----------
    nx, nt : int
        Number of points along the cable and number of time steps.
    cable_len : float
        Total cable length [m].
    ts_cold, ts_warm : float
        Reference-bath temperatures [degC].
    noise_st, noise_ast, noise_rst, noise_rast : float
        Standard deviation of additive Gaussian noise on each Stokes channel.
        Set all to ``0.0`` (default) for a noise-free dataset.
    seed : int
        Seed used to build the noise generator.

    Returns
    -------
    SimpleNamespace
        Bundle with ``ds`` (the xarray Dataset), ``sections`` (a sections
        dict), and the ground-truth values (``gamma``, ``C_p``, ``C_m``,
        ``dalpha_r``, ``dalpha_p``, ``dalpha_m``, ``temp_real_C``,
        ``temp_real_K``, ``alpha``, ``x``, ``time``).
    """
    rng = np.random.default_rng(seed)
    time = np.arange(nt)
    x = np.linspace(0.0, cable_len, nx)
    ts_cold_arr = np.ones(nt) * ts_cold
    ts_warm_arr = np.ones(nt) * ts_warm

    C_p = 15246.0
    C_m = 2400.0
    dalpha_r = 0.0005284
    dalpha_m = 0.0004961
    dalpha_p = 0.0005607
    gamma = 482.6

    cold_mask = x < 0.5 * cable_len
    warm_mask = ~cold_mask
    temp_real_K = np.ones((len(x), nt))
    temp_real_K[cold_mask] *= ts_cold_arr + 273.15
    temp_real_K[warm_mask] *= ts_warm_arr + 273.15

    st = (
        C_p
        * np.exp(-(dalpha_r + dalpha_p) * x[:, None])
        * np.exp(gamma / temp_real_K)
        / (np.exp(gamma / temp_real_K) - 1)
    )
    ast = (
        C_m
        * np.exp(-(dalpha_r + dalpha_m) * x[:, None])
        / (np.exp(gamma / temp_real_K) - 1)
    )
    rst = (
        C_p
        * np.exp(-(dalpha_r + dalpha_p) * (cable_len - x[:, None]))
        * np.exp(gamma / temp_real_K)
        / (np.exp(gamma / temp_real_K) - 1)
    )
    rast = (
        C_m
        * np.exp(-(dalpha_r + dalpha_m) * (cable_len - x[:, None]))
        / (np.exp(gamma / temp_real_K) - 1)
    )

    if noise_st > 0:
        st = st + rng.standard_normal(st.shape) * noise_st
    if noise_ast > 0:
        ast = ast + rng.standard_normal(ast.shape) * noise_ast
    if noise_rst > 0:
        rst = rst + rng.standard_normal(rst.shape) * noise_rst
    if noise_rast > 0:
        rast = rast + rng.standard_normal(rast.shape) * noise_rast

    alpha = np.mean(np.log(rst / rast) - np.log(st / ast), axis=1) / 2
    alpha -= alpha[0]

    ds = Dataset(
        {
            "st": (["x", "time"], st),
            "ast": (["x", "time"], ast),
            "rst": (["x", "time"], rst),
            "rast": (["x", "time"], rast),
            "userAcquisitionTimeFW": (["time"], np.ones(nt)),
            "userAcquisitionTimeBW": (["time"], np.ones(nt)),
            "cold": (["time"], ts_cold_arr),
            "warm": (["time"], ts_warm_arr),
        },
        coords={"x": x, "time": time},
        attrs={"isDoubleEnded": "1"},
    )

    sections = {
        "cold": [slice(0.0, 0.4 * cable_len)],
        "warm": [slice(0.65 * cable_len, cable_len)],
    }

    return SimpleNamespace(
        ds=ds,
        sections=sections,
        gamma=gamma,
        C_p=C_p,
        C_m=C_m,
        dalpha_r=dalpha_r,
        dalpha_p=dalpha_p,
        dalpha_m=dalpha_m,
        temp_real_C=temp_real_K - 273.15,
        temp_real_K=temp_real_K,
        alpha=alpha,
        x=x,
        time=time,
        cable_len=cable_len,
    )


@pytest.fixture
def synthetic_double_ended():
    """Factory fixture: call to build a synthetic double-ended dataset.

    Examples
    --------
    >>> def test_something(synthetic_double_ended):
    ...     bundle = synthetic_double_ended(nx=20, nt=10)
    ...     ds = bundle.ds
    ...     truth = bundle.gamma
    """
    return _build_synthetic_double_ended

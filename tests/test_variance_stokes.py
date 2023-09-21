import os

import numpy as np
import pytest
from scipy import stats

from dtscalibration import DataStore
from dtscalibration import read_silixa_files
from dtscalibration.variance_stokes import variance_stokes_exponential
from dtscalibration.variance_stokes import variance_stokes_constant
from dtscalibration.variance_stokes import variance_stokes_linear

np.random.seed(0)

fn = [
    "channel 1_20170921112245510.xml",
    "channel 1_20170921112746818.xml",
    "channel 1_20170921112746818.xml",
]
fn_single = [
    "channel 2_20180504132202074.xml",
    "channel 2_20180504132232903.xml",
    "channel 2_20180504132303723.xml",
]

if 1:
    # working dir is tests
    wd = os.path.dirname(os.path.abspath(__file__))
    data_dir_single_ended = os.path.join(wd, "data", "single_ended")
    data_dir_double_ended = os.path.join(wd, "data", "double_ended")
    data_dir_double_ended2 = os.path.join(wd, "data", "double_ended2")

else:
    # working dir is src
    data_dir_single_ended = os.path.join("..", "..", "tests", "data", "single_ended")
    data_dir_double_ended = os.path.join("..", "..", "tests", "data", "double_ended")
    data_dir_double_ended2 = os.path.join("..", "..", "tests", "data", "double_ended2")


def assert_almost_equal_verbose(actual, desired, verbose=False, **kwargs):
    """Print the actual precision decimals"""
    err = np.abs(actual - desired).max()
    dec = -np.ceil(np.log10(err))

    if not (np.isfinite(dec)):
        dec = 18.0

    m = "\n>>>>>The actual precision is: " + str(float(dec))

    if verbose:
        print(m)

    desired2 = np.broadcast_to(desired, actual.shape)
    np.testing.assert_almost_equal(actual, desired2, err_msg=m, **kwargs)
    pass


@pytest.mark.skip(reason="Not enough measurements in time. Use exponential instead.")
def test_variance_of_stokes():
    correct_var = 9.045
    filepath = data_dir_double_ended2
    ds = read_silixa_files(directory=filepath, timezone_netcdf="UTC", file_ext="*.xml")
    sections = {
        "probe1Temperature": [slice(7.5, 17.0), slice(70.0, 80.0)],  # cold bath
        "probe2Temperature": [slice(24.0, 34.0), slice(85.0, 95.0)],  # warm bath
    }

    I_var, _ = variance_stokes_constant(st=ds["st"], sections=sections)
    assert_almost_equal_verbose(I_var, correct_var, decimal=1)

    ds_dask = ds.chunk(chunks={})
    I_var, _ = variance_stokes_constant(st=ds_dask["st"], sections=sections)
    assert_almost_equal_verbose(I_var, correct_var, decimal=1)

    pass


def test_variance_of_stokes_synthetic():
    """
    Produces a synthetic Stokes measurement with a known noise distribution. Check if same
    variance is obtained.

    Returns
    -------

    """
    yvar = 5.0

    nx = 500
    x = np.linspace(0.0, 20.0, nx)

    nt = 200
    G = np.linspace(3000, 4000, nt)[None]

    y = G * np.exp(-0.001 * x[:, None])

    y += stats.norm.rvs(size=y.size, scale=yvar**0.5).reshape(y.shape)

    ds = DataStore(
        {
            "st": (["x", "time"], y),
            "probe1Temperature": (["time"], range(nt)),
            "userAcquisitionTimeFW": (["time"], np.ones(nt)),
        },
        coords={"x": x, "time": range(nt)},
        attrs={"isDoubleEnded": "0"},
    )

    sections = {"probe1Temperature": [slice(0.0, 20.0)]}
    test_st_var, _ = variance_stokes_constant(st=ds["st"], sections=sections)

    assert_almost_equal_verbose(test_st_var, yvar, decimal=1)
    pass


@pytest.mark.slow  # Execution time ~20 seconds
def test_variance_of_stokes_linear_synthetic():
    """
    Produces a synthetic Stokes measurement with a known noise distribution.
    Check if same variance is obtained.

    Returns
    -------

    """
    var_slope = 0.01

    nx = 500
    x = np.linspace(0.0, 20.0, nx)

    nt = 200
    G = np.linspace(500, 4000, nt)[None]
    c_no_noise = G * np.exp(-0.001 * x[:, None])

    c_lin_var_through_zero = stats.norm.rvs(
        loc=c_no_noise,
        # size=y.size,
        scale=(var_slope * c_no_noise) ** 0.5,
    )
    ds = DataStore(
        {
            "st": (["x", "time"], c_no_noise),
            "c_lin_var_through_zero": (["x", "time"], c_lin_var_through_zero),
            "probe1Temperature": (["time"], range(nt)),
            "userAcquisitionTimeFW": (["time"], np.ones(nt)),
        },
        coords={"x": x, "time": range(nt)},
        attrs={"isDoubleEnded": "0"},
    )

    sections = {"probe1Temperature": [slice(0.0, 20.0)]}
    test_st_var, _ = variance_stokes_constant(st=ds["st"], sections=sections)

    # If fit is forced through zero. Only Poisson distributed noise
    (
        slope,
        offset,
        st_sort_mean,
        st_sort_var,
        resid,
        var_fun,
    ) = variance_stokes_linear(
        st=ds["c_lin_var_through_zero"],
        sections=sections,
        nbin=10,
        through_zero=True,
        plot_fit=False,
    )
    assert_almost_equal_verbose(slope, var_slope, decimal=3)

    # Fit accounts for Poisson noise plus white noise
    (
        slope,
        offset,
        st_sort_mean,
        st_sort_var,
        resid,
        var_fun,
    ) = variance_stokes_linear(
        st=ds["c_lin_var_through_zero"], sections=sections, nbin=100, through_zero=False
    )
    assert_almost_equal_verbose(slope, var_slope, decimal=3)
    assert_almost_equal_verbose(offset, 0.0, decimal=0)

    pass


@pytest.mark.slow  # Execution time ~20 seconds
def test_exponential_variance_of_stokes():
    correct_var = 11.86535
    filepath = data_dir_double_ended2
    ds = read_silixa_files(directory=filepath, timezone_netcdf="UTC", file_ext="*.xml")
    sections = {
        "probe1Temperature": [slice(7.5, 17.0), slice(70.0, 80.0)],  # cold bath
        "probe2Temperature": [slice(24.0, 34.0), slice(85.0, 95.0)],  # warm bath
    }

    I_var, _ = variance_stokes_exponential(st=ds["st"], sections=sections)
    assert_almost_equal_verbose(I_var, correct_var, decimal=5)

    ds_dask = ds.chunk(chunks={})
    I_var, _ = variance_stokes_exponential(st=ds_dask["st"], sections=sections)
    assert_almost_equal_verbose(I_var, correct_var, decimal=5)

    pass


def test_exponential_variance_of_stokes_synthetic():
    """
    Produces a synthetic Stokes measurement with a known noise distribution. Check if same
    variance is obtained.

    Returns
    -------

    """
    yvar = 5.0

    nx = 500
    x = np.linspace(0.0, 20.0, nx)

    nt = 200
    beta = np.linspace(3000, 4000, nt)[None]

    y = beta * np.exp(-0.001 * x[:, None])

    y += stats.norm.rvs(size=y.size, scale=yvar**0.5).reshape(y.shape)

    ds = DataStore(
        {
            "st": (["x", "time"], y),
            "probe1Temperature": (["time"], range(nt)),
            "userAcquisitionTimeFW": (["time"], np.ones(nt)),
        },
        coords={"x": x, "time": range(nt)},
        attrs={"isDoubleEnded": "0"},
    )

    sections = {"probe1Temperature": [slice(0.0, 20.0)]}
    test_st_var, _ = variance_stokes_exponential(st=ds["st"], sections=sections)

    assert_almost_equal_verbose(test_st_var, yvar, decimal=1)
    pass

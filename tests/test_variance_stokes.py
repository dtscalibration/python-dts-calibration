import os

import numpy as np
import pytest
from scipy import stats
import xarray as xr
from xarray import Dataset

from dtscalibration import read_silixa_files
from dtscalibration.datastore_accessor import DtsAccessor  # noqa: F401
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

@pytest.mark.slow  # Execution time ~20 seconds
def test_variance_input_types_single():
    import dask.array as da

    state = da.random.RandomState(0)

    stokes_m_var = 40.0
    cable_len = 100.0
    nt = 500
    time = np.arange(nt)
    x = np.linspace(0.0, cable_len, 100)
    ts_cold = np.ones(nt) * 4.0
    ts_warm = np.ones(nt) * 20.0

    C_p = 15246
    C_m = 2400.0
    dalpha_r = 0.005284
    dalpha_m = 0.004961
    dalpha_p = 0.005607
    gamma = 482.6
    cold_mask = x < 0.5 * cable_len
    warm_mask = np.invert(cold_mask)  # == False
    temp_real = np.ones((len(x), nt))
    temp_real[cold_mask] *= ts_cold + 273.15
    temp_real[warm_mask] *= ts_warm + 273.15

    st = (
        C_p
        * np.exp(-dalpha_r * x[:, None])
        * np.exp(-dalpha_p * x[:, None])
        * np.exp(-gamma / temp_real)
        / (1 - np.exp(-gamma / temp_real))
    )
    ast = (
        C_m
        * np.exp(-dalpha_r * x[:, None])
        * np.exp(-dalpha_m * x[:, None])
        / (1 - np.exp(-gamma / temp_real))
    )

    st_m = st + stats.norm.rvs(size=st.shape, scale=stokes_m_var**0.5)
    ast_m = ast + stats.norm.rvs(size=ast.shape, scale=1.1 * stokes_m_var**0.5)

    print("alphaint", cable_len * (dalpha_p - dalpha_m))
    print("alpha", dalpha_p - dalpha_m)
    print("C", np.log(C_p / C_m))
    print("x0", x.max())

    ds = Dataset(
        {
            "st": (["x", "time"], st_m),
            "ast": (["x", "time"], ast_m),
            "userAcquisitionTimeFW": (["time"], np.ones(nt)),
            "cold": (["time"], ts_cold),
            "warm": (["time"], ts_warm),
        },
        coords={"x": x, "time": time},
        attrs={"isDoubleEnded": "0"},
    )

    sections = {
        "cold": [slice(0.0, 0.4 * cable_len)],
        "warm": [slice(0.6 * cable_len, cable_len)],
    }

    # Test float input
    st_var = 5.0

    out = ds.dts.calibrate_single_ended(
        sections=sections, st_var=st_var, ast_var=st_var, method="wls", solver="sparse"
    )

    out2 = ds.dts.monte_carlo_single_ended(
        result=out,
        st_var=st_var,
        ast_var=st_var,
        mc_sample_size=100,
        da_random_state=state,
        mc_remove_set_flag=False,
    )

    assert_almost_equal_verbose(
        out2["tmpf_mc_var"].sel(x=slice(0, 10)).mean(), 0.044361, decimal=2
    )
    assert_almost_equal_verbose(
        out2["tmpf_mc_var"].sel(x=slice(90, 100)).mean(), 0.242028, decimal=2
    )

    # Test callable input
    def callable_st_var(stokes):
        slope = 0.01
        offset = 0
        return slope * stokes + offset

    out = ds.dts.calibrate_single_ended(
        sections=sections,
        st_var=callable_st_var,
        ast_var=callable_st_var,
        method="wls",
        solver="sparse",
    )

    out2 = ds.dts.monte_carlo_single_ended(
        result=out,
        st_var=callable_st_var,
        ast_var=callable_st_var,
        mc_sample_size=100,
        da_random_state=state,
    )

    assert_almost_equal_verbose(
        out2["tmpf_mc_var"].sel(x=slice(0, 10)).mean(), 0.184753, decimal=2
    )
    assert_almost_equal_verbose(
        out2["tmpf_mc_var"].sel(x=slice(90, 100)).mean(), 0.545186, decimal=2
    )

    # Test input with shape of (ntime, nx)
    st_var = ds.st.values * 0 + 20.0
    out = ds.dts.calibrate_single_ended(
        sections=sections, st_var=st_var, ast_var=st_var, method="wls", solver="sparse"
    )

    out2 = ds.dts.monte_carlo_single_ended(result=out,
        st_var=st_var, ast_var=st_var, mc_sample_size=100, da_random_state=state
    )

    assert_almost_equal_verbose(out2["tmpf_mc_var"].mean(), 0.418098, decimal=2)

    # Test input with shape (nx, 1)
    st_var = np.vstack(
        ds.st.mean(dim="time").values * 0 + np.linspace(10, 50, num=ds.st.x.size)
    )

    out = ds.dts.calibrate_single_ended(
        sections=sections, st_var=st_var, ast_var=st_var, method="wls", solver="sparse"
    )

    out2 = ds.dts.monte_carlo_single_ended(result=out,
        st_var=st_var, ast_var=st_var, mc_sample_size=100, da_random_state=state
    )

    assert_almost_equal_verbose(
        out2["tmpf_mc_var"].sel(x=slice(0, 50)).mean().values, 0.2377, decimal=2
    )
    assert_almost_equal_verbose(
        out2["tmpf_mc_var"].sel(x=slice(50, 100)).mean().values, 1.3203, decimal=2
    )

    # Test input with shape (ntime)
    st_var = ds.st.mean(dim="x").values * 0 + np.linspace(5, 200, num=nt)

    out = ds.dts.calibrate_single_ended(
        sections=sections, st_var=st_var, ast_var=st_var, method="wls", solver="sparse"
    )

    out2 = ds.dts.monte_carlo_single_ended(result=out,
        st_var=st_var, ast_var=st_var, mc_sample_size=100, da_random_state=state
    )

    assert_almost_equal_verbose(
        out2["tmpf_mc_var"].sel(time=slice(0, nt // 2)).mean().values, 1.0908, decimal=2
    )
    assert_almost_equal_verbose(
        out2["tmpf_mc_var"].sel(time=slice(nt // 2, None)).mean().values, 3.0759, decimal=2
    )

    pass


@pytest.mark.slow  # Execution time ~0.5 minute
def test_variance_input_types_double():
    import dask.array as da

    state = da.random.RandomState(0)

    stokes_m_var = 40.0
    cable_len = 100.0
    nt = 500
    time = np.arange(nt)
    x = np.linspace(0.0, cable_len, 100)
    ts_cold = np.ones(nt) * 4.0
    ts_warm = np.ones(nt) * 20.0

    C_p = 15246
    C_m = 2400.0
    dalpha_r = 0.005284
    dalpha_m = 0.004961
    dalpha_p = 0.005607
    gamma = 482.6
    cold_mask = x < 0.5 * cable_len
    warm_mask = np.invert(cold_mask)  # == False
    temp_real = np.ones((len(x), nt))
    temp_real[cold_mask] *= ts_cold + 273.15
    temp_real[warm_mask] *= ts_warm + 273.15

    st = (
        C_p
        * np.exp(-dalpha_r * x[:, None])
        * np.exp(-dalpha_p * x[:, None])
        * np.exp(-gamma / temp_real)
        / (1 - np.exp(-gamma / temp_real))
    )
    ast = (
        C_m
        * np.exp(-dalpha_r * x[:, None])
        * np.exp(-dalpha_m * x[:, None])
        / (1 - np.exp(-gamma / temp_real))
    )
    rst = (
        C_p
        * np.exp(-dalpha_r * (-x[:, None] + 100))
        * np.exp(-dalpha_p * (-x[:, None] + 100))
        * np.exp(-gamma / temp_real)
        / (1 - np.exp(-gamma / temp_real))
    )
    rast = (
        C_m
        * np.exp(-dalpha_r * (-x[:, None] + 100))
        * np.exp(-dalpha_m * (-x[:, None] + 100))
        / (1 - np.exp(-gamma / temp_real))
    )

    st_m = st + stats.norm.rvs(size=st.shape, scale=stokes_m_var**0.5)
    ast_m = ast + stats.norm.rvs(size=ast.shape, scale=1.1 * stokes_m_var**0.5)
    rst_m = rst + stats.norm.rvs(size=rst.shape, scale=0.9 * stokes_m_var**0.5)
    rast_m = rast + stats.norm.rvs(size=rast.shape, scale=0.8 * stokes_m_var**0.5)

    print("alphaint", cable_len * (dalpha_p - dalpha_m))
    print("alpha", dalpha_p - dalpha_m)
    print("C", np.log(C_p / C_m))
    print("x0", x.max())

    ds = Dataset(
        {
            "st": (["x", "time"], st_m),
            "ast": (["x", "time"], ast_m),
            "rst": (["x", "time"], rst_m),
            "rast": (["x", "time"], rast_m),
            "userAcquisitionTimeFW": (["time"], np.ones(nt)),
            "userAcquisitionTimeBW": (["time"], np.ones(nt)),
            "cold": (["time"], ts_cold),
            "warm": (["time"], ts_warm),
        },
        coords={"x": x, "time": time},
        attrs={"isDoubleEnded": "1"},
    )

    sections = {
        "cold": [slice(0.0, 0.4 * cable_len)],
        "warm": [slice(0.6 * cable_len, cable_len)],
    }

    # Test float input
    st_var = 5.0

    out = ds.dts.calibration_double_ended(
        sections=sections,
        st_var=st_var,
        ast_var=st_var,
        rst_var=st_var,
        rast_var=st_var,
        method="wls",
        solver="sparse",
    )

    out2 = ds.dts.monte_carlo_double_ended(
        result=out,
        st_var=st_var,
        ast_var=st_var,
        rst_var=st_var,
        rast_var=st_var,
        conf_ints=[],
        mc_sample_size=100,
        da_random_state=state,
    )

    assert_almost_equal_verbose(
        out2["tmpf_mc_var"].sel(x=slice(0, 10)).mean(), 0.03584935, decimal=2
    )
    assert_almost_equal_verbose(
        out2["tmpf_mc_var"].sel(x=slice(90, 100)).mean(), 0.22982146, decimal=2
    )

    # Test callable input
    def st_var_callable(stokes):
        slope = 0.01
        offset = 0
        return slope * stokes + offset

    out = ds.dts.calibration_double_ended(
        sections=sections,
        st_var=st_var_callable,
        ast_var=st_var_callable,
        rst_var=st_var_callable,
        rast_var=st_var_callable,
        method="wls",
        solver="sparse",
    )

    out2 = ds.dts.monte_carlo_double_ended(
        result=out,
        st_var=st_var_callable,
        ast_var=st_var_callable,
        rst_var=st_var_callable,
        rast_var=st_var_callable,
        conf_ints=[],
        mc_sample_size=100,
        da_random_state=state,
    )

    assert_almost_equal_verbose(
        out2["tmpf_mc_var"].sel(x=slice(0, 10)).mean(), 0.18058514, decimal=2
    )
    assert_almost_equal_verbose(
        out2["tmpf_mc_var"].sel(x=slice(90, 100)).mean(), 0.53862813, decimal=2
    )

    # Test input with shape of (ntime, nx)
    st_var = ds.st.values * 0 + 20.0

    out = ds.dts.calibration_double_ended(
        sections=sections,
        st_var=st_var,
        ast_var=st_var,
        rst_var=st_var,
        rast_var=st_var,
        method="wls",
        solver="sparse",
    )

    out2 = ds.dts.monte_carlo_double_ended(
        result=out,
        st_var=st_var,
        ast_var=st_var,
        rst_var=st_var,
        rast_var=st_var,
        conf_ints=[],
        mc_sample_size=100,
        da_random_state=state,
    )

    assert_almost_equal_verbose(out2["tmpf_mc_var"].mean(), 0.40725674, decimal=2)

    # Test input with shape (nx, 1)
    st_var = np.vstack(
        ds.st.mean(dim="time").values * 0 + np.linspace(10, 50, num=ds.st.x.size)
    )

    out = ds.dts.calibration_double_ended(
        sections=sections,
        st_var=st_var,
        ast_var=st_var,
        rst_var=st_var,
        rast_var=st_var,
        method="wls",
        solver="sparse",
    )

    out2 = ds.dts.monte_carlo_double_ended(
        result=out,
        st_var=st_var,
        ast_var=st_var,
        rst_var=st_var,
        rast_var=st_var,
        conf_ints=[],
        mc_sample_size=100,
        da_random_state=state,
    )

    assert_almost_equal_verbose(
        out2["tmpf_mc_var"].sel(x=slice(0, 50)).mean().values, 0.21163704, decimal=2
    )
    assert_almost_equal_verbose(
        out2["tmpf_mc_var"].sel(x=slice(50, 100)).mean().values, 1.28247762, decimal=2
    )

    # Test input with shape (ntime)
    st_var = ds.st.mean(dim="x").values * 0 + np.linspace(5, 200, num=nt)

    out = ds.dts.calibration_double_ended(
        sections=sections,
        st_var=st_var,
        ast_var=st_var,
        rst_var=st_var,
        rast_var=st_var,
        method="wls",
        solver="sparse",
    )

    out2 = ds.dts.monte_carlo_double_ended(
        result=out,
        st_var=st_var,
        ast_var=st_var,
        rst_var=st_var,
        rast_var=st_var,
        conf_ints=[],
        mc_sample_size=100,
        da_random_state=state,
    )

    assert_almost_equal_verbose(
        out2["tmpf_mc_var"].sel(time=slice(0, nt // 2)).mean().values, 1.090, decimal=2
    )
    assert_almost_equal_verbose(
        out2["tmpf_mc_var"].sel(time=slice(nt // 2, None)).mean().values, 3.06, decimal=2
    )

    pass


@pytest.mark.slow  # Execution time ~0.5 minute
def test_double_ended_variance_estimate_synthetic():
    import dask.array as da

    state = da.random.RandomState(0)

    stokes_m_var = 40.0
    cable_len = 100.0
    nt = 500
    time = np.arange(nt)
    x = np.linspace(0.0, cable_len, 100)
    ts_cold = np.ones(nt) * 4.0
    ts_warm = np.ones(nt) * 20.0

    C_p = 15246
    C_m = 2400.0
    dalpha_r = 0.0005284
    dalpha_m = 0.0004961
    dalpha_p = 0.0005607
    gamma = 482.6
    cold_mask = x < 0.5 * cable_len
    warm_mask = np.invert(cold_mask)  # == False
    temp_real = np.ones((len(x), nt))
    temp_real[cold_mask] *= ts_cold + 273.15
    temp_real[warm_mask] *= ts_warm + 273.15

    st = (
        C_p
        * np.exp(-dalpha_r * x[:, None])
        * np.exp(-dalpha_p * x[:, None])
        * np.exp(-gamma / temp_real)
        / (1 - np.exp(-gamma / temp_real))
    )
    ast = (
        C_m
        * np.exp(-dalpha_r * x[:, None])
        * np.exp(-dalpha_m * x[:, None])
        / (1 - np.exp(-gamma / temp_real))
    )
    rst = (
        C_p
        * np.exp(-dalpha_r * (-x[:, None] + 100))
        * np.exp(-dalpha_p * (-x[:, None] + 100))
        * np.exp(-gamma / temp_real)
        / (1 - np.exp(-gamma / temp_real))
    )
    rast = (
        C_m
        * np.exp(-dalpha_r * (-x[:, None] + 100))
        * np.exp(-dalpha_m * (-x[:, None] + 100))
        / (1 - np.exp(-gamma / temp_real))
    )

    st_m = st + stats.norm.rvs(size=st.shape, scale=stokes_m_var**0.5)
    ast_m = ast + stats.norm.rvs(size=ast.shape, scale=1.1 * stokes_m_var**0.5)
    rst_m = rst + stats.norm.rvs(size=rst.shape, scale=0.9 * stokes_m_var**0.5)
    rast_m = rast + stats.norm.rvs(size=rast.shape, scale=0.8 * stokes_m_var**0.5)

    print("alphaint", cable_len * (dalpha_p - dalpha_m))
    print("alpha", dalpha_p - dalpha_m)
    print("C", np.log(C_p / C_m))
    print("x0", x.max())

    ds = Dataset(
        {
            "st": (["x", "time"], st_m),
            "ast": (["x", "time"], ast_m),
            "rst": (["x", "time"], rst_m),
            "rast": (["x", "time"], rast_m),
            "userAcquisitionTimeFW": (["time"], np.ones(nt)),
            "userAcquisitionTimeBW": (["time"], np.ones(nt)),
            "cold": (["time"], ts_cold),
            "warm": (["time"], ts_warm),
        },
        coords={"x": x, "time": time},
        attrs={"isDoubleEnded": "1"},
    )

    sections = {
        "cold": [slice(0.0, 0.5 * cable_len)],
        "warm": [slice(0.5 * cable_len, cable_len)],
    }

    mst_var, _ = variance_stokes_constant(ds.dts.st, sections, ds.dts.acquisitiontime_fw, reshape_residuals=False)
    mast_var, _ = variance_stokes_constant(ds.dts.ast, sections, ds.dts.acquisitiontime_fw, reshape_residuals=False)
    mrst_var, _ = variance_stokes_constant(ds.dts.rst, sections, ds.dts.acquisitiontime_bw, reshape_residuals=False)
    mrast_var, _ = variance_stokes_constant(ds.dts.rast, sections, ds.dts.acquisitiontime_bw, reshape_residuals=False)

    mst_var = float(mst_var)
    mast_var = float(mast_var)
    mrst_var = float(mrst_var)
    mrast_var = float(mrast_var)

    # MC variance
    out = ds.dts.calibration_double_ended(
        sections=sections,
        st_var=mst_var,
        ast_var=mast_var,
        rst_var=mrst_var,
        rast_var=mrast_var,
        method="wls",
        solver="sparse",
    )

    assert_almost_equal_verbose(out["tmpf"].mean(), 12.0, decimal=2)
    assert_almost_equal_verbose(out["tmpb"].mean(), 12.0, decimal=3)
    
    # Calibrated variance
    stdsf1 = out.dts.ufunc_per_section(
        sections=sections, label="tmpf", func=np.std, temp_err=True, calc_per="stretch"
    )
    stdsb1 = out.dts.ufunc_per_section(
        sections=sections, label="tmpb", func=np.std, temp_err=True, calc_per="stretch"
    )

    out2 = ds.dts.monte_carlo_double_ended(
        sections=sections,
        p_val="p_val",
        p_cov="p_cov",
        st_var=mst_var,
        ast_var=mast_var,
        rst_var=mrst_var,
        rast_var=mrast_var,
        conf_ints=[2.5, 50.0, 97.5],
        mc_sample_size=100,
        da_random_state=state,
    )

    # Use a single timestep to better check if the parameter uncertainties propagate
    ds1 = out2.isel(time=1)
    # Estimated VAR
    stdsf2 = ds1.dts.ufunc_per_section(
        sections=sections,
        label="tmpf_mc_var",
        func=np.mean,
        temp_err=False,
        calc_per="stretch",
    )
    stdsb2 = ds1.dts.ufunc_per_section(
        sections=sections,
        label="tmpb_mc_var",
        func=np.mean,
        temp_err=False,
        calc_per="stretch",
    )

    for (_, v1), (_, v2) in zip(stdsf1.items(), stdsf2.items()):
        for v1i, v2i in zip(v1, v2):
            print("Real VAR: ", v1i**2, "Estimated VAR: ", float(v2i))
            assert_almost_equal_verbose(v1i**2, v2i, decimal=2)

    for (_, v1), (_, v2) in zip(stdsb1.items(), stdsb2.items()):
        for v1i, v2i in zip(v1, v2):
            print("Real VAR: ", v1i**2, "Estimated VAR: ", float(v2i))
            assert_almost_equal_verbose(v1i**2, v2i, decimal=2)

    pass


def test_single_ended_variance_estimate_synthetic():
    import dask.array as da

    state = da.random.RandomState(0)

    stokes_m_var = 40.0
    astokes_m_var = 60.0
    cable_len = 100.0
    nt = 50
    time = np.arange(nt)
    x = np.linspace(0.0, cable_len, 500)
    ts_cold = np.ones(nt) * 4.0
    ts_warm = np.ones(nt) * 20.0

    C_p = 15246
    C_m = 2400.0
    dalpha_r = 0.0005284
    dalpha_m = 0.0004961
    dalpha_p = 0.0005607
    gamma = 482.6
    cold_mask = x < 0.5 * cable_len
    warm_mask = np.invert(cold_mask)  # == False
    temp_real = np.ones((len(x), nt))
    temp_real[cold_mask] *= ts_cold + 273.15
    temp_real[warm_mask] *= ts_warm + 273.15

    st = (
        C_p
        * np.exp(-dalpha_r * x[:, None])
        * np.exp(-dalpha_p * x[:, None])
        * np.exp(-gamma / temp_real)
        / (1 - np.exp(-gamma / temp_real))
    )
    ast = (
        C_m
        * np.exp(-dalpha_r * x[:, None])
        * np.exp(-dalpha_m * x[:, None])
        / (1 - np.exp(-gamma / temp_real))
    )
    st_m = st + stats.norm.rvs(size=st.shape, scale=stokes_m_var**0.5)
    ast_m = ast + stats.norm.rvs(size=ast.shape, scale=astokes_m_var**0.5)

    print("alphaint", cable_len * (dalpha_p - dalpha_m))
    print("alpha", dalpha_p - dalpha_m)
    print("C", np.log(C_p / C_m))
    print("x0", x.max())

    ds = Dataset(
        {
            "st": (["x", "time"], st_m),
            "ast": (["x", "time"], ast_m),
            "userAcquisitionTimeFW": (["time"], np.ones(nt)),
            "cold": (["time"], ts_cold),
            "warm": (["time"], ts_warm),
        },
        coords={"x": x, "time": time},
        attrs={"isDoubleEnded": "0"},
    )

    sections = {
        "cold": [slice(0.0, 0.5 * cable_len)],
        "warm": [slice(0.5 * cable_len, cable_len)],
    }

    st_label = "st"
    ast_label = "ast"

    mst_var, _ = ds.variance_stokes(st_label=st_label, sections=sections)
    mast_var, _ = ds.variance_stokes(st_label=ast_label, sections=sections)
    mst_var = float(mst_var)
    mast_var = float(mast_var)

    # MC variqnce
    out = ds.dts.calibrate_single_ended(
        sections=sections,
        st_var=mst_var,
        ast_var=mast_var,
        method="wls",
        solver="sparse",
    )

    out2 = ds.dts.monte_carlo_single_ended(
        result=out,
        st_var=mst_var,
        ast_var=mast_var,
        conf_ints=[2.5, 50.0, 97.5],
        mc_sample_size=50,
        da_random_state=state,
    )

    # Calibrated variance
    stdsf1 = out.dts.ufunc_per_section(
        sections=sections,
        label="tmpf",
        func=np.std,
        temp_err=True,
        calc_per="stretch",
        ddof=1,
    )

    # Use a single timestep to better check if the parameter uncertainties propagate
    ds1 = out2.isel(time=1)
    # Estimated VAR
    stdsf2 = ds1.dts.ufunc_per_section(
        sections=sections,
        label="tmpf_mc_var",
        func=np.mean,
        temp_err=False,
        calc_per="stretch",
    )

    for (_, v1), (_, v2) in zip(stdsf1.items(), stdsf2.items()):
        for v1i, v2i in zip(v1, v2):
            print("Real VAR: ", v1i**2, "Estimated VAR: ", float(v2i))
            assert_almost_equal_verbose(v1i**2, v2i, decimal=2)

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

    ds = xr.Dataset(
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

    ds = Dataset(
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

import os

import numpy as np

from dtscalibration import read_silixa_files
from dtscalibration.dts_accessor import DtsAccessor  # noqa: F401

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


def test_average_measurements_single_ended():
    filepath = data_dir_single_ended

    ds_ = read_silixa_files(directory=filepath, timezone_netcdf="UTC", file_ext="*.xml")

    ds = ds_.sel(x=slice(0, 100))  # only calibrate parts of the fiber
    sections = {"probe2Temperature": [slice(6.0, 14.0)]}  # warm bath

    st_var, ast_var = 5.0, 5.0

    out = ds.dts.calibrate_single_ended(
        sections=sections, st_var=st_var, ast_var=ast_var, method="wls", solver="sparse"
    )
    ds.dts.average_monte_carlo_single_ended(
        result=out,
        st_var=st_var,
        ast_var=ast_var,
        conf_ints=[2.5, 97.5],
        mc_sample_size=50,  # <- choose a much larger sample size
        ci_avg_x_flag1=True,
        ci_avg_x_sel=slice(6.0, 14.0),
    )

    def get_section_indices(x_da, sec):
        """Returns the x-indices of the section. `sec` is a slice."""
        xis = x_da.astype(int) * 0 + np.arange(x_da.size, dtype=int)
        return xis.sel(x=sec).values

    ix = get_section_indices(ds.x, slice(6, 14))
    ds.dts.average_monte_carlo_single_ended(
        result=out,
        st_var=st_var,
        ast_var=ast_var,
        conf_ints=[2.5, 97.5],
        mc_sample_size=50,  # <- choose a much larger sample size
        ci_avg_x_flag2=True,
        ci_avg_x_isel=ix,
    )
    sl = slice(
        np.datetime64("2018-05-04T12:22:17.710000000"),
        np.datetime64("2018-05-04T12:22:47.702000000"),
    )
    ds.dts.average_monte_carlo_single_ended(
        result=out,
        st_var=st_var,
        ast_var=ast_var,
        conf_ints=[2.5, 97.5],
        mc_sample_size=50,  # <- choose a much larger sample size
        ci_avg_time_flag1=True,
        ci_avg_time_flag2=False,
        ci_avg_time_sel=sl,
    )
    ds.dts.average_monte_carlo_single_ended(
        result=out,
        st_var=st_var,
        ast_var=ast_var,
        conf_ints=[2.5, 97.5],
        mc_sample_size=50,  # <- choose a much larger sample size
        ci_avg_time_flag1=False,
        ci_avg_time_flag2=True,
        ci_avg_time_isel=range(3),
    )
    pass


def test_average_measurements_double_ended():
    filepath = data_dir_double_ended2

    ds_ = read_silixa_files(directory=filepath, timezone_netcdf="UTC", file_ext="*.xml")

    ds = ds_.sel(x=slice(0, 100))  # only calibrate parts of the fiber
    sections = {
        "probe1Temperature": [slice(7.5, 17.0), slice(70.0, 80.0)],  # cold bath
        "probe2Temperature": [slice(24.0, 34.0), slice(85.0, 95.0)],  # warm bath
    }

    st_var, ast_var, rst_var, rast_var = 5.0, 5.0, 5.0, 5.0

    out = ds.dts.calibrate_double_ended(
        sections=sections,
        st_var=st_var,
        ast_var=ast_var,
        rst_var=rst_var,
        rast_var=rast_var,
        method="wls",
        solver="sparse",
    )
    ds.dts.average_monte_carlo_double_ended(
        result=out,
        st_var=st_var,
        ast_var=ast_var,
        rst_var=rst_var,
        rast_var=rast_var,
        conf_ints=[2.5, 97.5],
        mc_sample_size=50,  # <- choose a much larger sample size
        ci_avg_x_flag1=True,
        ci_avg_x_sel=slice(6, 10),
    )

    def get_section_indices(x_da, sec):
        """Returns the x-indices of the section. `sec` is a slice."""
        xis = x_da.astype(int) * 0 + np.arange(x_da.size, dtype=int)
        return xis.sel(x=sec).values

    ix = get_section_indices(ds.x, slice(6, 10))
    ds.dts.average_monte_carlo_double_ended(
        result=out,
        st_var=st_var,
        ast_var=ast_var,
        rst_var=rst_var,
        rast_var=rast_var,
        conf_ints=[2.5, 97.5],
        mc_sample_size=50,  # <- choose a much larger sample size
        ci_avg_x_flag2=True,
        ci_avg_x_isel=ix,
    )
    sl = slice(
        np.datetime64("2018-03-28T00:40:54.097000000"),
        np.datetime64("2018-03-28T00:41:12.084000000"),
    )
    ds.dts.average_monte_carlo_double_ended(
        result=out,
        st_var=st_var,
        ast_var=ast_var,
        rst_var=rst_var,
        rast_var=rast_var,
        conf_ints=[2.5, 97.5],
        mc_sample_size=50,  # <- choose a much larger sample size
        ci_avg_time_flag1=True,
        ci_avg_time_flag2=False,
        ci_avg_time_sel=sl,
    )
    ds.dts.average_monte_carlo_double_ended(
        result=out,
        st_var=st_var,
        ast_var=ast_var,
        rst_var=rst_var,
        rast_var=rast_var,
        conf_ints=[2.5, 97.5],
        mc_sample_size=50,  # <- choose a much larger sample size
        ci_avg_time_flag1=False,
        ci_avg_time_flag2=True,
        ci_avg_time_isel=range(3),
    )
    pass

import hashlib
import os
import tempfile
import warnings
from zipfile import ZipFile

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from xarray import Dataset

from dtscalibration import read_apsensing_files
from dtscalibration import read_sensornet_files
from dtscalibration import read_sensortran_files
from dtscalibration import read_silixa_files
from dtscalibration.calibration.section_utils import set_sections
from dtscalibration.dts_accessor_utils import merge_double_ended
from dtscalibration.dts_accessor_utils import shift_double_ended
from dtscalibration.dts_accessor_utils import suggest_cable_shift_double_ended

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

wd = os.path.dirname(os.path.abspath(__file__))
data_dir_single_ended = os.path.join(wd, "data", "single_ended")
data_dir_double_ended = os.path.join(wd, "data", "double_ended")
data_dir_double_ended2 = os.path.join(wd, "data", "double_ended2")
data_dir_silixa_long = os.path.join(wd, "data", "double_single_ended", "channel_1")
data_dir_sensornet_single_ended = os.path.join(wd, "data", "sensornet_oryx_v3.7")
data_dir_sensornet_halo_double_ended = os.path.join(wd, "data", "sensornet_halo_v1.0")
data_dir_sensornet_oryx_double_ended = os.path.join(
    wd, "data", "sensornet_oryx_v3.7_double"
)
data_dir_sensornet_sentinel_double_ended = os.path.join(
    wd, "data", "sensornet_sentinel_v5.1_double"
)
data_dir_single_silixa_v45 = os.path.join(wd, "data", "silixa_v4.5")
data_dir_single_silixa_v7 = os.path.join(wd, "data", "silixa_v7.0")
data_dir_single_silixa_v8 = os.path.join(wd, "data", "silixa_v8.1")
data_dir_ap_sensing = os.path.join(wd, "data", "ap_sensing")
data_dir_sensortran_binary = os.path.join(wd, "data", "sensortran_binary")
data_dir_double_single_ch1 = os.path.join(
    wd, "data", "double_single_ended", "channel_1"
)
data_dir_double_single_ch2 = os.path.join(
    wd, "data", "double_single_ended", "channel_2"
)

# zips
data_dir_zipped_single_ended = os.path.join(
    wd, "data", "zipped data", "single_ended.zip"
)
data_dir_zipped_double_ended = os.path.join(
    wd, "data", "zipped data", "double_ended.zip"
)
data_dir_zipped_double_ended2 = os.path.join(
    wd, "data", "zipped data", "double_ended2.zip"
)
data_dir_zipped_silixa_long = os.path.join(
    wd, "data", "zipped data", "double_single_ended.zip"
)
data_dir_zipped_sensornet_single_ended = os.path.join(
    wd, "data", "zipped data", "sensornet_oryx_v3.7.zip"
)


def test_read_data_from_single_file_double_ended():
    """
    Check if read data from file is correct
    :return:
    """
    fp0 = os.path.join(data_dir_double_ended, fn[0])
    data = read_data_from_fp_numpy(fp0)

    nx, ncols = data.shape

    err_msg = "Not all points along the cable are read from file"
    np.testing.assert_equal(nx, 2330, err_msg=err_msg)

    err_msg = "Not all columns are read from file"
    np.testing.assert_equal(ncols, 6, err_msg=err_msg)

    actual_hash = hashlib.sha1(data).hexdigest()
    desired_hash = "51b94dedd77c83c6cdd9dd132f379a39f742edae"

    assert actual_hash == desired_hash, "The data is not read correctly"


def test_read_data_from_single_file_single_ended():
    """
    Check if read data from file is correct
    :return:
    """
    fp0 = os.path.join(data_dir_single_ended, fn_single[0])
    data = read_data_from_fp_numpy(fp0)

    nx, ncols = data.shape

    err_msg = "Not all points along the cable are read from file"
    np.testing.assert_equal(nx, 1461, err_msg=err_msg)

    err_msg = "Not all columns are read from file"
    np.testing.assert_equal(ncols, 4, err_msg=err_msg)

    actual_hash = hashlib.sha1(data).hexdigest()
    desired_hash = "58103e2d79f777f98bf279442eea138065883062"

    assert actual_hash == desired_hash, "The data is not read correctly"


def test_repr():
    ds = Dataset(
        {
            "st": (["x", "time"], np.ones((100, 5))),
            "ast": (["x", "time"], np.ones((100, 5))),
            "probe1Temperature": (["time"], range(5)),
            "probe2Temperature": (["time"], range(5)),
        },
        coords={"x": range(100), "time": range(5)},
    )

    assert "dtscalibration" in str(ds.dts)
    assert "Sections" in str(ds.dts)


def test_sections_property():
    ds = Dataset(
        {
            "st": (["x", "time"], np.ones((100, 5))),
            "ast": (["x", "time"], np.ones((100, 5))),
            "probe1Temperature": (["time"], range(5)),
            "probe2Temperature": (["time"], range(5)),
        },
        coords={"x": range(100), "time": range(5)},
    )

    sections1 = {
        "probe1Temperature": [slice(7.5, 17.0), slice(70.0, 80.0)],  # cold bath
        "probe2Temperature": [slice(24.0, 34.0), slice(85.0, 95.0)],  # warm bath
    }
    sections2 = {
        "probe1Temperature": [slice(0.0, 17.0), slice(70.0, 80.0)],  # cold bath
        "probe2Temperature": [slice(24.0, 34.0), slice(85.0, 95.0)],  # warm bath
    }
    set_sections(ds, sections1)

    assert isinstance(ds.attrs["_sections"], str)

    assert ds.dts.sections == sections1
    assert ds.dts.sections != sections2

    # test if accepts singleton numpy arrays
    set_sections(
        ds,
        {
            "probe1Temperature": [
                slice(np.array(0.0), np.array(17.0)),
                slice(70.0, 80.0),
            ]
        },
    )


def test_io_sections_property():
    ds = Dataset(
        {
            "st": (["x", "time"], np.ones((100, 5))),
            "ast": (["x", "time"], np.ones((100, 5))),
            "probe1Temperature": (["time"], range(5)),
            "probe2Temperature": (["time"], range(5)),
        },
        coords={"x": ("x", range(100), {"units": "m"}), "time": range(5)},
    )

    sections = {
        "probe1Temperature": [slice(7.5, 17.0), slice(70.0, 80.0)],  # cold bath
        "probe2Temperature": [slice(24.0, 34.0), slice(85.0, 95.0)],  # warm bath
    }
    ds["x"].attrs["units"] = "m"

    set_sections(ds, sections)

    # Create a temporary file to write data to.
    # 'with' method is used so the file is closed by tempfile
    # and free to be overwritten.
    with tempfile.NamedTemporaryFile("w") as tmp:
        temppath = tmp.name

    # Write the datastore to the temp file
    ds.to_netcdf(path=temppath)

    try:
        ds2 = xr.open_dataset(temppath)
    except ValueError as e:
        if str(e) != "cannot guess the engine, try passing one explicitly":
            raise
        warnings.warn("Could not guess engine, defaulted to netcdf4")
        ds2 = xr.open_datastore(temppath, engine="netcdf4")

    assert ds.dts.sections == ds2.dts.sections

    # Close the datastore so the temp file can be removed
    ds2.close()
    ds2 = None

    # Remove the temp file once the test is done
    if os.path.exists(temppath):
        os.remove(temppath)


def test_read_silixa_files_single_ended():
    filepath = data_dir_single_ended
    ds = read_silixa_files(directory=filepath, timezone_netcdf="UTC", file_ext="*.xml")

    np.testing.assert_almost_equal(ds.st.sum(), 11387947.857, decimal=3)


def test_read_silixa_files_double_ended():
    filepath = data_dir_double_ended
    ds = read_silixa_files(directory=filepath, timezone_netcdf="UTC", file_ext="*.xml")

    np.testing.assert_almost_equal(ds.st.sum(), 19613502.2617, decimal=3)


def test_read_silixa_files_single_loadinmemory():
    filepath = data_dir_single_ended

    # False
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf="UTC",
        file_ext="*.xml",
        load_in_memory=False,
    )
    for k in ["st", "ast"]:
        assert isinstance(ds[k].data, da.Array)

    # auto -> True
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf="UTC",
        file_ext="*.xml",
        load_in_memory="auto",
    )
    for k in ["st", "ast"]:
        assert isinstance(ds[k].data, np.ndarray)

    # True
    ds = read_silixa_files(
        directory=filepath, timezone_netcdf="UTC", file_ext="*.xml", load_in_memory=True
    )
    for k in ["st", "ast"]:
        assert isinstance(ds[k].data, np.ndarray)


def test_read_silixa_files_double_loadinmemory():
    filepath = data_dir_double_ended

    # False
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf="UTC",
        file_ext="*.xml",
        load_in_memory=False,
    )
    for k in ["st", "ast", "rst", "rast"]:
        assert isinstance(ds[k].data, da.Array)

    # auto -> True Because small amount of data
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf="UTC",
        file_ext="*.xml",
        load_in_memory="auto",
    )
    for k in ["st", "ast", "rst", "rast"]:
        assert isinstance(ds[k].data, np.ndarray)

    # True
    ds = read_silixa_files(
        directory=filepath, timezone_netcdf="UTC", file_ext="*.xml", load_in_memory=True
    )
    for k in ["st", "ast", "rst", "rast"]:
        assert isinstance(ds[k].data, np.ndarray)


def test_read_single_silixa_v45():
    filepath = data_dir_single_silixa_v45
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf="UTC",
        file_ext="*.xml",
        load_in_memory=False,
    )

    for k in ["st", "ast"]:
        assert isinstance(ds[k].data, da.Array)

    ds = read_silixa_files(
        directory=filepath, timezone_netcdf="UTC", file_ext="*.xml", load_in_memory=True
    )

    for k in ["st", "ast"]:
        assert isinstance(ds[k].data, np.ndarray)


def test_read_single_silixa_v7():
    filepath = data_dir_single_silixa_v7
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf="UTC",
        file_ext="*.xml",
        load_in_memory=False,
    )

    for k in ["st", "ast"]:
        assert isinstance(ds[k].data, da.Array)

    ds = read_silixa_files(
        directory=filepath, timezone_netcdf="UTC", file_ext="*.xml", load_in_memory=True
    )

    for k in ["st", "ast"]:
        assert isinstance(ds[k].data, np.ndarray)


def test_read_single_silixa_v8():
    filepath = data_dir_single_silixa_v8
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf="UTC",
        file_ext="*.xml",
        load_in_memory=False,
    )

    for k in ["st", "ast"]:
        assert isinstance(ds[k].data, da.Array)

    ds = read_silixa_files(
        directory=filepath, timezone_netcdf="UTC", file_ext="*.xml", load_in_memory=True
    )

    for k in ["st", "ast"]:
        assert isinstance(ds[k].data, np.ndarray)


@pytest.mark.skip(
    reason="Randomly fails. Has to do with delayed reading" "out of zips with dask."
)
def test_read_silixa_zipped():
    files = [
        (data_dir_zipped_single_ended, 11387947.857184),
        (data_dir_zipped_double_ended, 19613502.26171),
        (data_dir_zipped_double_ended2, 28092965.5188),
        (data_dir_zipped_silixa_long, 2.88763942e08),
    ]

    for file, stsum in files:
        with ZipFile(file) as fh:
            ds = read_silixa_files(
                zip_handle=fh,
                timezone_netcdf="UTC",
                file_ext="*.xml",
                load_in_memory=True,
            )
            np.testing.assert_almost_equal(ds.st.sum(), stsum, decimal=0)
            ds.close()


def test_read_long_silixa_files():
    filepath = data_dir_silixa_long
    ds = read_silixa_files(directory=filepath, timezone_netcdf="UTC", file_ext="*.xml")
    np.testing.assert_almost_equal(ds.st.sum(), 133223729.17096, decimal=0)


def test_read_sensornet_files_single_ended():
    filepath = data_dir_sensornet_single_ended
    ds = read_sensornet_files(
        directory=filepath,
        timezone_netcdf="UTC",
        timezone_input_files="UTC",
        add_internal_fiber_length=50.0,
        fiber_length=None,
        file_ext="*.ddf",
    )
    np.testing.assert_almost_equal(ds.st.sum(), 2955105.679, decimal=2)


def test_read_sensornet_halo_files_double_ended():
    filepath = data_dir_sensornet_halo_double_ended
    ds = read_sensornet_files(
        directory=filepath,
        timezone_netcdf="UTC",
        timezone_input_files="UTC",
        add_internal_fiber_length=50.0,
        fiber_length=1253.3,
        file_ext="*.ddf",
    )

    np.testing.assert_almost_equal(ds.st.sum(), 2835988.114, decimal=2)


def test_read_sensornet_oryx_files_double_ended():
    filepath = data_dir_sensornet_oryx_double_ended
    ds = read_sensornet_files(
        directory=filepath,
        timezone_netcdf="UTC",
        timezone_input_files="UTC",
        add_internal_fiber_length=50.0,
        fiber_length=187.0,
        file_ext="*.ddf",
    )

    np.testing.assert_almost_equal(ds.st.sum(), 2301637.154, decimal=2)
    np.testing.assert_almost_equal(ds.rst.sum(), 1835770.651, decimal=2)


def test_read_sensornet_sentinel_files_double_ended():
    filepath = data_dir_sensornet_sentinel_double_ended
    ds = read_sensornet_files(
        directory=filepath,
        timezone_netcdf="UTC",
        timezone_input_files="UTC",
        add_internal_fiber_length=50.0,
        fiber_length=2100.0,
        file_ext="*.ddf",
    )

    np.testing.assert_almost_equal(ds.st.sum(), 16531426.023, decimal=2)
    np.testing.assert_almost_equal(ds.rst.sum(), 15545880.215, decimal=2)


def test_read_apsensing_files():
    filepath = data_dir_ap_sensing
    ds = read_apsensing_files(
        directory=filepath,
        timezone_netcdf="UTC",
        timezone_input_files="UTC",
        file_ext="*.xml",
    )
    np.testing.assert_almost_equal(ds.st.sum(), 10415.2837, decimal=2)


def test_read_apsensing_files_loadinmemory():
    filepath = data_dir_ap_sensing

    # False
    ds = read_apsensing_files(
        directory=filepath,
        timezone_netcdf="UTC",
        file_ext="*.xml",
        load_in_memory=False,
    )
    for k in ["st", "ast"]:
        assert isinstance(ds[k].data, da.Array)

    # auto -> True Because small amount of data
    ds = read_apsensing_files(
        directory=filepath,
        timezone_netcdf="UTC",
        file_ext="*.xml",
        load_in_memory="auto",
    )
    for k in ["st", "ast"]:
        assert isinstance(ds[k].data, np.ndarray)

    # True
    ds = read_apsensing_files(
        directory=filepath, timezone_netcdf="UTC", file_ext="*.xml", load_in_memory=True
    )
    for k in ["st", "ast"]:
        assert isinstance(ds[k].data, np.ndarray)


def test_read_sensortran_files():
    filepath = data_dir_sensortran_binary
    ds = read_sensortran_files(directory=filepath, timezone_netcdf="UTC")
    np.testing.assert_approx_equal(
        ds.st.values.astype(np.int64).sum(), np.int64(1432441254828), significant=12
    )


def read_data_from_fp_numpy(fp):
    """
    Read the data from a single Silixa xml file. Using a simple approach

    Parameters
    ----------
    fp : file, str, or pathlib.Path
        File path

    Returns
    -------
    data : ndarray
        The data of the file as numpy array of shape (nx, ncols)

    Notes
    -----
    calculating i_first and i_last is fast compared to the rest
    """

    with open(fp) as fh:
        s = fh.readlines()

    s = [si.strip() for si in s]  # remove xml hierarchy spacing

    i_first = s.index("<data>")
    i_last = len(s) - s[::-1].index("</data>") - 1

    lssl = slice(i_first + 1, i_last, 3)  # list of strings slice

    return np.loadtxt(s[lssl], delimiter=",", dtype=float)


def test_resample_datastore():
    filepath = data_dir_single_ended
    ds = read_silixa_files(directory=filepath, timezone_netcdf="UTC", file_ext="*.xml")
    assert ds.time.size == 3

    ds_resampled = Dataset(ds.resample(time="47S").mean())

    assert ds_resampled.time.size == 2
    assert ds_resampled.st.dims == ("x", "time"), (
        "The dimension have to "
        "be manually transposed "
        "after resampling. To "
        "guarantee the order"
    )


def test_timeseries_keys():
    filepath = data_dir_single_ended
    ds = read_silixa_files(directory=filepath, timezone_netcdf="UTC", file_ext="*.xml")

    k = ds.dts.get_timeseries_keys()

    # no false positive
    for ki in k:
        assert ds[ki].dims == ("time",)

    # no false negatives
    k_not = [ki for ki in ds.data_vars if ki not in k]
    for kni in k_not:
        assert ds[kni].dims != ("time",)


def test_shift_double_ended_shift_backforward():
    # shifting it back and forward, should result in the same
    filepath = data_dir_double_ended
    ds = read_silixa_files(directory=filepath, timezone_netcdf="UTC", file_ext="*.xml")

    dsmin1 = shift_double_ended(ds, -1)
    ds2 = shift_double_ended(dsmin1, 1)

    np.testing.assert_allclose(ds.x[1:-1], ds2.x)

    for k in ds2:
        if "x" not in ds2[k].dims:
            continue

        old = ds[k].isel(x=slice(1, -1))
        new = ds2[k]

        np.testing.assert_allclose(old, new)


def test_suggest_cable_shift_double_ended():
    # need more measurements for proper testing. Therefore only checking if
    # no errors occur

    filepath = data_dir_double_ended
    ds = read_silixa_files(directory=filepath, timezone_netcdf="UTC", file_ext="*.xml")

    irange = np.arange(-4, 4)
    suggest_cable_shift_double_ended(ds, irange, plot_result=True)


def test_merge_double_ended():
    # Checking if alignment keeps working as designed and if the expected
    # result changed
    filepath_fw = data_dir_double_single_ch1
    filepath_bw = data_dir_double_single_ch2

    ds_fw = read_silixa_files(directory=filepath_fw)

    ds_bw = read_silixa_files(directory=filepath_bw)

    cable_length = 2017.7
    ds = merge_double_ended(ds_fw, ds_bw, cable_length=cable_length, plot_result=True)

    result = (ds.isel(time=0).st - ds.isel(time=0).rst).sum().values

    np.testing.assert_approx_equal(result, -3712866.0382, significant=10)


@pytest.mark.parametrize(
    "inotinfw, inotinbw, inotinout",
    [
        ([], [], []),
        ([1], [], [1]),
        ([], [1], [1]),
        ([1], [1], [1]),
        ([1, 2], [], [1, 2]),
        ([], [1, 2], [1, 2]),
        ([1], [2], [1, 2]),
        ([2], [1], [1, 2]),
    ],
)
def test_merge_double_ended_times(inotinfw, inotinbw, inotinout):
    """
    Arguments are the indices not included in resp fw measurements, bw measurements,
    and merged output.

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
    Any ideas are welcome.
    """
    filepath_fw = data_dir_double_single_ch1
    filepath_bw = data_dir_double_single_ch2
    cable_length = 2017.7

    ds_fw = read_silixa_files(directory=filepath_fw)
    ds_bw = read_silixa_files(directory=filepath_bw)

    # set stokes to varify proper time alignment
    ds_fw.st.values = np.tile(np.arange(ds_fw.time.size)[None], (ds_fw.x.size, 1))
    ds_bw.st.values = np.tile(np.arange(ds_bw.time.size)[None], (ds_bw.x.size, 1))

    # transform time index that is not included in: fw, bw, merged, to the ones included
    ifw = list(i for i in range(6) if i not in inotinfw)
    ibw = list(i for i in range(6) if i not in inotinbw)
    iout = list(i for i in range(6) if i not in inotinout)

    ds = merge_double_ended(
        ds_fw.isel(time=ifw),
        ds_bw.isel(time=ibw),
        cable_length=cable_length,
        plot_result=False,
        verbose=False,
    )
    assert (
        ds.time.size == len(iout) and np.all(ds.st.isel(x=0) == iout)
    ), f"FW:{ifw} & BW:{ibw} should lead to {iout} but instead leads to {ds.st.isel(x=0).values}"

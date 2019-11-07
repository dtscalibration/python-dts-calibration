# coding=utf-8
import hashlib
import os
import tempfile
import time
from zipfile import ZipFile as zipf

import dask.array as da
import numpy as np
import pytest

from dtscalibration import DataStore
from dtscalibration import open_datastore
from dtscalibration import open_mf_datastore
from dtscalibration import read_sensornet_files
from dtscalibration import read_silixa_files
from dtscalibration.datastore_utils import shift_double_ended
from dtscalibration.datastore_utils import suggest_cable_shift_double_ended

np.random.seed(0)

fn = [
    "channel 1_20170921112245510.xml", "channel 1_20170921112746818.xml",
    "channel 1_20170921112746818.xml"]
fn_single = [
    "channel 2_20180504132202074.xml", "channel 2_20180504132232903.xml",
    "channel 2_20180504132303723.xml"]

if 1:
    # working dir is tests
    wd = os.path.dirname(os.path.abspath(__file__))
    data_dir_single_ended = os.path.join(wd, 'data', 'single_ended')
    data_dir_double_ended = os.path.join(wd, 'data', 'double_ended')
    data_dir_double_ended2 = os.path.join(wd, 'data', 'double_ended2')
    data_dir_silixa_long = os.path.join(
        wd, 'data', 'double_single_ended', 'channel_1')
    data_dir_sensornet_single_ended = os.path.join(
        wd, 'data', 'sensornet_oryx_v3.7')
    data_dir_sensornet_double_ended = os.path.join(
        wd, 'data', 'sensornet_halo_v1.0')
    data_dir_single_silixa_v45 = os.path.join(wd, 'data', 'silixa_v4.5')
    data_dir_single_silixa_v7 = os.path.join(wd, 'data', 'silixa_v7.0')

    # zips
    data_dir_zipped_single_ended = os.path.join(
        wd, 'data', 'zipped data', 'single_ended.zip')
    data_dir_zipped_double_ended = os.path.join(
        wd, 'data', 'zipped data', 'double_ended.zip')
    data_dir_zipped_double_ended2 = os.path.join(
        wd, 'data', 'zipped data', 'double_ended2.zip')
    data_dir_zipped_silixa_long = os.path.join(
        wd, 'data', 'zipped data', 'double_single_ended.zip')
    data_dir_zipped_sensornet_single_ended = os.path.join(
        wd, 'data', 'zipped data', 'sensornet_oryx_v3.7.zip')

else:
    # working dir is src
    data_dir_single_ended = os.path.join(
        '..', '..', 'tests', 'data', 'single_ended')
    data_dir_double_ended = os.path.join(
        '..', '..', 'tests', 'data', 'double_ended')
    data_dir_double_ended2 = os.path.join(
        '..', '..', 'tests', 'data', 'double_ended2')
    data_dir_silixa_long = os.path.join(
        '..', '..', 'tests', 'data', 'double_single_ended', 'channel_1')
    data_dir_sensornet_single_ended = os.path.join(
        '..', '..', 'tests', 'data', 'sensornet_oryx_v3.7')
    data_dir_sensornet_double_ended = os.path.join(
        '..', '..', 'tests', 'data', 'sensornet_halo_v1.0')


def test_read_data_from_single_file_double_ended():
    """
    Check if read data from file is correct
    :return:
    """
    fp0 = os.path.join(data_dir_double_ended, fn[0])
    data = read_data_from_fp_numpy(fp0)

    nx, ncols = data.shape

    err_msg = 'Not all points along the cable are read from file'
    np.testing.assert_equal(nx, 2330, err_msg=err_msg)

    err_msg = 'Not all columns are read from file'
    np.testing.assert_equal(ncols, 6, err_msg=err_msg)

    actual_hash = hashlib.sha1(data).hexdigest()
    desired_hash = '51b94dedd77c83c6cdd9dd132f379a39f742edae'

    assert actual_hash == desired_hash, 'The data is not read correctly'
    pass


def test_read_data_from_single_file_single_ended():
    """
    Check if read data from file is correct
    :return:
    """
    fp0 = os.path.join(data_dir_single_ended, fn_single[0])
    data = read_data_from_fp_numpy(fp0)

    nx, ncols = data.shape

    err_msg = 'Not all points along the cable are read from file'
    np.testing.assert_equal(nx, 1461, err_msg=err_msg)

    err_msg = 'Not all columns are read from file'
    np.testing.assert_equal(ncols, 4, err_msg=err_msg)

    actual_hash = hashlib.sha1(data).hexdigest()
    desired_hash = '58103e2d79f777f98bf279442eea138065883062'

    assert actual_hash == desired_hash, 'The data is not read correctly'
    pass


def test_empty_construction():
    ds = DataStore()  # noqa: F841
    pass


def test_repr():
    ds = DataStore()
    assert str(ds).find('dtscalibration') != -1
    assert str(ds).find('Sections') != -1
    pass


def test_has_sectionattr_upon_creation():
    ds = DataStore()
    assert hasattr(ds, '_sections')
    assert isinstance(ds._sections, str)
    pass


def test_sections_property():
    ds = DataStore(
        {
            'st': (['x', 'time'], np.ones((100, 5))),
            'ast': (['x', 'time'], np.ones((100, 5))),
            'probe1Temperature': (['time'], range(5)),
            'probe2Temperature': (['time'], range(5))},
        coords={
            'x': range(100),
            'time': range(5)})

    sections1 = {
        'probe1Temperature': [slice(7.5, 17.),
                              slice(70., 80.)],  # cold bath
        'probe2Temperature': [slice(24., 34.),
                              slice(85., 95.)],  # warm bath
    }
    sections2 = {
        'probe1Temperature': [slice(0., 17.), slice(70., 80.)],  # cold bath
        'probe2Temperature': [slice(24., 34.),
                              slice(85., 95.)],  # warm bath
    }
    ds.sections = sections1

    assert isinstance(ds._sections, str)

    assert ds.sections == sections1
    assert ds.sections != sections2

    # test if accepts singleton numpy arrays
    ds.sections = {
        'probe1Temperature': [
            slice(np.array(0.), np.array(17.)), slice(70., 80.)]}

    # delete property
    del ds.sections
    assert ds.sections is None

    pass


def test_io_sections_property():
    ds = DataStore(
        {
            'st': (['x', 'time'], np.ones((100, 5))),
            'ast': (['x', 'time'], np.ones((100, 5))),
            'probe1Temperature': (['time'], range(5)),
            'probe2Temperature': (['time'], range(5))},
        coords={
            'x': ('x', range(100), {'units': 'm'}),
            'time': range(5)})

    sections = {
        'probe1Temperature': [slice(7.5, 17.),
                              slice(70., 80.)],  # cold bath
        'probe2Temperature': [slice(24., 34.),
                              slice(85., 95.)],  # warm bath
    }
    ds['x'].attrs['units'] = 'm'

    ds.sections = sections

    # Create a temporary file to write data to.
    # 'with' method is used so the file is closed by tempfile
    # and free to be overwritten.
    with tempfile.NamedTemporaryFile('w') as tmp:
        temppath = tmp.name

    # Write the datastore to the temp file
    ds.to_netcdf(path=temppath)

    ds2 = open_datastore(temppath)

    assert ds.sections == ds2.sections

    # Close the datastore so the temp file can be removed
    ds2.close()
    ds2 = None

    # Remove the temp file once the test is done
    if os.path.exists(temppath):
        os.remove(temppath)

    pass


def test_read_silixa_files_single_ended():
    filepath = data_dir_single_ended
    ds = read_silixa_files(
        directory=filepath, timezone_netcdf='UTC', file_ext='*.xml')

    np.testing.assert_almost_equal(ds.ST.sum(), 11387947.857, decimal=3)

    pass


def test_read_silixa_files_double_ended():
    filepath = data_dir_double_ended
    ds = read_silixa_files(
        directory=filepath, timezone_netcdf='UTC', file_ext='*.xml')

    np.testing.assert_almost_equal(ds.ST.sum(), 19613502.2617, decimal=3)

    pass


def test_read_silixa_files_single_loadinmemory():
    filepath = data_dir_single_ended

    # False
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf='UTC',
        file_ext='*.xml',
        load_in_memory=False)
    for k in ['ST', 'AST']:
        assert isinstance(ds[k].data, da.Array)

    # auto -> True
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf='UTC',
        file_ext='*.xml',
        load_in_memory='auto')
    for k in ['ST', 'AST']:
        assert isinstance(ds[k].data, np.ndarray)

    # True
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf='UTC',
        file_ext='*.xml',
        load_in_memory=True)
    for k in ['ST', 'AST']:
        assert isinstance(ds[k].data, np.ndarray)

    pass


def test_read_silixa_files_double_loadinmemory():
    filepath = data_dir_double_ended

    # False
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf='UTC',
        file_ext='*.xml',
        load_in_memory=False)
    for k in ['ST', 'AST', 'REV-ST', 'REV-AST']:
        assert isinstance(ds[k].data, da.Array)

    # auto -> True Because small amount of data
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf='UTC',
        file_ext='*.xml',
        load_in_memory='auto')
    for k in ['ST', 'AST', 'REV-ST', 'REV-AST']:
        assert isinstance(ds[k].data, np.ndarray)

    # True
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf='UTC',
        file_ext='*.xml',
        load_in_memory=True)
    for k in ['ST', 'AST', 'REV-ST', 'REV-AST']:
        assert isinstance(ds[k].data, np.ndarray)

    pass


def test_read_single_silixa_v45():
    filepath = data_dir_single_silixa_v45
    ds = read_silixa_files(
        directory=filepath, timezone_netcdf='UTC', file_ext='*.xml',
        load_in_memory=False)

    for k in ['ST', 'AST']:
        assert isinstance(ds[k].data, da.Array)

    ds = read_silixa_files(
        directory=filepath, timezone_netcdf='UTC', file_ext='*.xml',
        load_in_memory=True)

    for k in ['ST', 'AST']:
        assert isinstance(ds[k].data, np.ndarray)

    pass


def test_read_single_silixa_v7():
    filepath = data_dir_single_silixa_v7
    ds = read_silixa_files(
        directory=filepath, timezone_netcdf='UTC', file_ext='*.xml',
        load_in_memory=False)

    for k in ['ST', 'AST']:
        assert isinstance(ds[k].data, da.Array)

    ds = read_silixa_files(
        directory=filepath, timezone_netcdf='UTC', file_ext='*.xml',
        load_in_memory=True)

    for k in ['ST', 'AST']:
        assert isinstance(ds[k].data, np.ndarray)

    pass

@pytest.mark.skip(reason="Randomly fails. Has to do with delayed reading out of"
                         "zips with dask.")
def test_read_silixa_zipped():
    files = [
        (data_dir_zipped_single_ended, 11387947.857184),
        (data_dir_zipped_double_ended, 19613502.26171),
        (data_dir_zipped_double_ended2, 28092965.5188),
        (data_dir_zipped_silixa_long, 2.88763942e+08)]

    for file, stsum in files:
        with zipf(file) as fh:
            ds = read_silixa_files(
                zip_handle=fh,
                timezone_netcdf='UTC',
                file_ext='*.xml',
                load_in_memory=True)
            np.testing.assert_almost_equal(ds.ST.sum(), stsum, decimal=0)
            ds.close()
    pass


def test_read_long_silixa_files():
    filepath = data_dir_silixa_long
    ds = read_silixa_files(
        directory=filepath, timezone_netcdf='UTC', file_ext='*.xml')
    np.testing.assert_almost_equal(ds.ST.sum(), 133223729.17096, decimal=0)
    pass


def test_read_sensornet_files_single_ended():
    filepath = data_dir_sensornet_single_ended
    ds = read_sensornet_files(
        directory=filepath,
        timezone_netcdf='UTC',
        timezone_input_files='UTC',
        file_ext='*.ddf')
    np.testing.assert_almost_equal(ds.ST.sum(), 3015991.361, decimal=2)
    pass


def test_read_sensornet_files_double_ended():
    filepath = data_dir_sensornet_double_ended
    ds = read_sensornet_files(
        directory=filepath,
        timezone_netcdf='UTC',
        timezone_input_files='UTC',
        file_ext='*.ddf')

    np.testing.assert_almost_equal(ds.ST.sum(), 2832389.888, decimal=2)
    pass


def test_to_mf_netcdf_open_mf_datastore():
    filepath = data_dir_single_ended
    ds = read_silixa_files(directory=filepath, file_ext='*.xml')

    with tempfile.TemporaryDirectory() as tmpdirname:
        print('created temporary directory', tmpdirname)

        # work around the effects of deafault encoding.
        path = os.path.join(tmpdirname, 'ds_merged.nc')
        ds.to_netcdf(path)
        ds.close()
        time.sleep(2)  # to ensure all is written on Windows and file released
        ds1 = open_datastore(path, load_in_memory=True)

        # Test saving
        ds1 = ds1.chunk({'time': 1})
        ds1.to_mf_netcdf(folder_path=tmpdirname, filename_preamble='file_',
                         filename_extension='.nc')
        correct_val = float(ds1.ST.sum())
        ds1.close()
        time.sleep(2)  # to ensure all is written on Windows and file released

        # Test loading
        path = os.path.join(tmpdirname, 'file_*.nc')
        ds2 = open_mf_datastore(path=path, load_in_memory=True)
        test_val = float(ds1.ST.sum())

        np.testing.assert_equal(correct_val, test_val)
        ds2.close()

    pass


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

    i_first = s.index('<data>')
    i_last = len(s) - s[::-1].index('</data>') - 1

    lssl = slice(i_first + 1, i_last, 3)  # list of strings slice

    data = np.loadtxt(s[lssl], delimiter=',', dtype=float)

    return data


def test_resample_datastore():
    filepath = data_dir_single_ended
    ds = read_silixa_files(
        directory=filepath, timezone_netcdf='UTC', file_ext='*.xml')
    assert ds.time.size == 3

    ds_resampled = ds.resample_datastore(how='mean', time="47S")

    assert ds_resampled.time.size == 2
    assert ds_resampled.ST.dims == ('x', 'time'), 'The dimension have to be ' \
                                                  'manually transposed after ' \
                                                  'resampling. To guarantee ' \
                                                  'the order'

    pass


def test_timeseries_keys():
    filepath = data_dir_single_ended
    ds = read_silixa_files(
        directory=filepath, timezone_netcdf='UTC', file_ext='*.xml')

    k = ds.timeseries_keys

    # no false positive
    for ki in k:
        assert ds[ki].dims == ('time',)

    # no false negatives
    k_not = [ki for ki in ds.data_vars if ki not in k]
    for kni in k_not:
        assert ds[kni].dims != ('time',)

    pass


def test_shift_double_ended_shift_backforward():
    # shifting it back and forward, should result in the same
    filepath = data_dir_double_ended
    ds = read_silixa_files(
        directory=filepath, timezone_netcdf='UTC', file_ext='*.xml')

    dsmin1 = shift_double_ended(ds, -1)
    ds2 = shift_double_ended(dsmin1, 1)

    np.testing.assert_allclose(ds.x[1:-1], ds2.x)

    for k in ds2:
        if 'x' not in ds2[k].dims:
            continue

        old = ds[k].isel(x=slice(1, -1))
        new = ds2[k]

        np.testing.assert_allclose(old, new)

    pass


def test_suggest_cable_shift_double_ended():
    # need more measurements for proper testing. Therefore only checking if
    # no errors occur

    filepath = data_dir_double_ended
    ds = read_silixa_files(
        directory=filepath, timezone_netcdf='UTC', file_ext='*.xml')

    irange = np.arange(-4, 4)
    suggest_cable_shift_double_ended(ds, irange, plot_result=True)

    pass

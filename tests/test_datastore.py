# coding=utf-8
import hashlib
import os
import tempfile

import numpy as np

from dtscalibration import DataStore
from dtscalibration import open_datastore
from dtscalibration import read_silixa_files
from dtscalibration.datastore_utils import read_data_from_fp_numpy

np.random.seed(0)

fn = ["channel 1_20170921112245510.xml",
      "channel 1_20170921112746818.xml",
      "channel 1_20170921112746818.xml"]
fn_single = ["channel 2_20180504132202074.xml",
             "channel 2_20180504132232903.xml",
             "channel 2_20180504132303723.xml"]

if 1:
    # working dir is tests
    wd = os.path.dirname(os.path.abspath(__file__))
    data_dir_single_ended = os.path.join(wd, 'data', 'single_ended')
    data_dir_double_ended = os.path.join(wd, 'data', 'double_ended')
    data_dir_double_ended2 = os.path.join(wd, 'data', 'double_ended2')

else:
    # working dir is src
    data_dir_single_ended = os.path.join('..', '..', 'tests', 'data', 'single_ended')
    data_dir_double_ended = os.path.join('..', '..', 'tests', 'data', 'double_ended')
    data_dir_double_ended2 = os.path.join('..', '..', 'tests', 'data', 'double_ended2')


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
    ds = DataStore()
    assert ds._initialized, 'Empty obj in not initialized'
    pass


def test_has_sectionattr_upon_creation():
    ds = DataStore()
    assert hasattr(ds, '_sections')
    assert isinstance(ds._sections, str)
    pass


def test_sections_property():
    ds = DataStore({
        'st':    (['x', 'time'], np.ones((5, 5))),
        'ast':   (['x', 'time'], np.ones((5, 5))),
        'probe1Temperature':  (['time'], range(5)),
        'probe2Temperature':  (['time'], range(5))
        },
        coords={
            'x':    range(5),
            'time': range(5)})

    sections1 = {
        'probe1Temperature': [slice(7.5, 17.), slice(70., 80.)],  # cold bath
        'probe2Temperature': [slice(24., 34.), slice(85., 95.)],  # warm bath
        }
    sections2 = {
        'probe1Temperature': [slice(0., 17.), slice(70., 80.)],  # cold bath
        'probe2Temperature': [slice(24., 34.), slice(85., 95.)],  # warm bath
        }
    ds.sections = sections1

    assert isinstance(ds._sections, str)

    assert ds.sections == sections1
    assert ds.sections != sections2

    # delete property
    del ds.sections
    assert ds.sections is None

    pass


def test_io_sections_property():
    ds = DataStore({
        'st':    (['x', 'time'], np.ones((5, 5))),
        'ast':   (['x', 'time'], np.ones((5, 5))),
        'probe1Temperature':  (['time'], range(5)),
        'probe2Temperature':  (['time'], range(5))
        },
        coords={
            'x':    range(5),
            'time': range(5)})

    sections = {
        'probe1Temperature': [slice(7.5, 17.), slice(70., 80.)],  # cold bath
        'probe2Temperature': [slice(24., 34.), slice(85., 95.)],  # warm bath
        }

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
        directory=filepath,
        timezone_netcdf='UTC',
        timezone_ultima_xml='Europe/Amsterdam',
        file_ext='*.xml')

    assert ds._initialized

    pass


def test_read_silixa_files_double_ended():
    filepath = data_dir_double_ended
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf='UTC',
        timezone_ultima_xml='Europe/Amsterdam',
        file_ext='*.xml')

    assert ds._initialized

    pass

# coding=utf-8
import hashlib
import os
import tempfile

import numpy as np

from dtscalibration import DataStore
from dtscalibration import open_datastore
from dtscalibration import read_xml_dir
from dtscalibration.datastore_utils import read_data_from_fp_numpy


fn = ["channel 1_20170921112245510.xml",
      "channel 1_20170921112746818.xml",
      "channel 1_20170921112746818.xml"]

if 1:
    # working dir is tests
    wd = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(wd, 'data')

else:
    # working dir is src
    data_dir = os.path.join('..', '..', 'tests', 'data')


def test_read_data_from_single_file():
    """
    Check if read data from file is correct
    :return:
    """
    fp0 = os.path.join(data_dir, fn[0])
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
    ds = DataStore()
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
    ds = DataStore()
    sections = {
        'probe1Temperature': [slice(7.5, 17.), slice(70., 80.)],  # cold bath
        'probe2Temperature': [slice(24., 34.), slice(85., 95.)],  # warm bath
        }

    ds.sections = sections
    with tempfile.NamedTemporaryFile() as tmp:
        ds.to_netcdf(path=tmp.name)

        ds2 = open_datastore(tmp.name)

        assert ds.sections == ds2.sections

    pass


def test_read_xml_dir():
    filepath = data_dir
    ds = read_xml_dir(filepath,
                timezone_netcdf='UTC',
                timezone_ultima_xml='Europe/Amsterdam',
                file_ext='*.xml')

    assert ds._initialized

    pass

# coding=utf-8
import hashlib
import os
import tempfile

import numpy as np
import scipy.sparse as sp
from scipy import stats

from dtscalibration import DataStore
from dtscalibration import open_datastore
from dtscalibration import read_xml_dir
from dtscalibration.calibrate_utils import wls_sparse
from dtscalibration.calibrate_utils import wls_stats
from dtscalibration.datastore_utils import read_data_from_fp_numpy

np.random.seed(0)

fn = ["channel 1_20170921112245510.xml",
      "channel 1_20170921112746818.xml",
      "channel 1_20170921112746818.xml"]

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


def test_read_xml_dir_single_ended():
    filepath = data_dir_single_ended
    ds = read_xml_dir(filepath,
                      timezone_netcdf='UTC',
                      timezone_ultima_xml='Europe/Amsterdam',
                      file_ext='*.xml')

    assert ds._initialized

    pass


def test_read_xml_dir_double_ended():
    filepath = data_dir_double_ended
    ds = read_xml_dir(filepath,
                      timezone_netcdf='UTC',
                      timezone_ultima_xml='Europe/Amsterdam',
                      file_ext='*.xml')

    assert ds._initialized

    pass


def test_variance_of_stokes():
    correct_var = 40.16
    filepath = data_dir_double_ended2
    ds = read_xml_dir(filepath,
                      timezone_netcdf='UTC',
                      timezone_ultima_xml='Europe/Amsterdam',
                      file_ext='*.xml')
    sections = {
        'probe1Temperature': [slice(7.5, 17.), slice(70., 80.)],  # cold bath
        'probe2Temperature': [slice(24., 34.), slice(85., 95.)],  # warm bath
        }

    I_var, _ = ds.variance_stokes(st_label='ST',
                                  sections=sections,
                                  use_statsmodels=True)
    np.testing.assert_almost_equal(I_var,
                                   correct_var,
                                   decimal=1)

    I_var, _ = ds.variance_stokes(st_label='ST',
                                  sections=sections,
                                  use_statsmodels=False)
    np.testing.assert_almost_equal(I_var,
                                   correct_var,
                                   decimal=1)

    pass


def test_variance_of_stokes_synthetic():
    """
    Produces a synthetic Stokes measurement with a known noise distribution. Check if same
    variance is obtained.

    Returns
    -------

    """
    yvar = 5.

    nx = 50
    x = np.linspace(0., 20., nx)

    nt = 1000
    beta = np.linspace(3000, 4000, nt)[None]

    y = beta * np.exp(-0.001 * x[:, None])

    y += stats.norm.rvs(size=y.size,
                        scale=yvar ** 0.5).reshape(y.shape)

    ds = DataStore({'test_ST': (['x', 'time'], y)},
                   coords={'x':    x,
                           'time': range(nt)})

    sections = {'placeholder': [slice(0., 20.), ]}
    test_ST_var, _ = ds.variance_stokes(st_label='test_ST',
                                        sections=sections,
                                        suppress_info=True)

    np.testing.assert_almost_equal(test_ST_var, yvar,
                                   decimal=1)


def test_calibration_ols():
    """Testing ordinary least squares procedure. And compare with device calibrated temperature.
    The measurements were calibrated by the device using only section 8--17.m. Those temperatures
    are compared up to 2 decimals. Silixa only uses a single calibration constant (I think they
    fix gamma).
    """
    filepath = data_dir_double_ended2
    ds = read_xml_dir(filepath,
                      timezone_netcdf='UTC',
                      timezone_ultima_xml='Europe/Amsterdam',
                      file_ext='*.xml')
    ds100 = ds.sel(x=slice(0, 100))
    sections_ultima = {
        'probe1Temperature': [slice(8., 17.)],  # cold bath
        }

    st_label = 'ST'
    ast_label = 'AST'
    rst_label = 'REV-ST'
    rast_label = 'REV-AST'
    ds100.calibration_double_ended(sections=sections_ultima,
                                   st_label=st_label,
                                   ast_label=ast_label,
                                   rst_label=rst_label,
                                   rast_label=rast_label,
                                   method='ols')

    ds100['TMPAVG'] = (ds100.TMPF + ds100.TMPB) / 2
    np.testing.assert_array_almost_equal(ds100.TMPAVG.data,
                                         ds100.TMP.data,
                                         decimal=1)

    ds009 = ds100.sel(x=sections_ultima['probe1Temperature'][0])
    np.testing.assert_array_almost_equal(ds009.TMPAVG.data,
                                         ds009.TMP.data,
                                         decimal=2)
    pass


def test_calibrate_wls_procedures():
    x = np.linspace(0, 10, 25 * 4)
    np.random.shuffle(x)

    X = x.reshape((25, 4))
    beta = np.array([1, 0.1, 10, 5])
    beta_w = np.concatenate((np.ones(10), np.ones(15) * 1.0))
    beta_0 = np.array([1, 1, 1, 1])
    y = np.dot(X, beta)
    y_meas = y + np.random.normal(size=y.size)

    # first check unweighted convergence
    beta_numpy = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_array_almost_equal(beta, beta_numpy, decimal=8)

    ps_sol, ps_var = wls_stats(X, y, w=1, calc_cov=0, x0=beta_0)
    p_sol, p_var = wls_sparse(X, y, w=1, calc_cov=0, x0=beta_0)

    np.testing.assert_array_almost_equal(beta, ps_sol, decimal=8)
    np.testing.assert_array_almost_equal(beta, p_sol, decimal=8)

    # now with weights
    dec = 8
    ps_sol, ps_var, ps_cov = wls_stats(X, y_meas, w=beta_w, calc_cov=True, x0=beta_0)
    p_sol, p_var, p_cov = wls_sparse(X, y_meas, w=beta_w, calc_cov=True, x0=beta_0)

    np.testing.assert_array_almost_equal(p_sol, ps_sol, decimal=dec)
    np.testing.assert_array_almost_equal(p_var, ps_var, decimal=dec)
    np.testing.assert_array_almost_equal(p_cov, ps_cov, decimal=dec)

    # Test array sparse
    Xsp = sp.coo_matrix(X)
    psp_sol, psp_var, psp_cov = wls_stats(Xsp, y_meas, w=beta_w, calc_cov=True, x0=beta_0)

    np.testing.assert_array_almost_equal(p_sol, psp_sol, decimal=dec)
    np.testing.assert_array_almost_equal(p_var, psp_var, decimal=dec)
    np.testing.assert_array_almost_equal(p_cov, psp_cov, decimal=dec)

    pass

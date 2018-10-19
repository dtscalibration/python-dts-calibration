# coding=utf-8
import os

import numpy as np
import scipy.sparse as sp
from scipy import stats

from dtscalibration import DataStore
from dtscalibration import read_silixa_files
from dtscalibration.calibrate_utils import wls_sparse
from dtscalibration.calibrate_utils import wls_stats
from dtscalibration.cli import main

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


def test_main():
    assert main([]) == 0


def test_double_ended_variance_estimate_synthetic():
    import dask.array as da
    from dtscalibration import DataStore
    import numpy as np
    from scipy import stats

    np.random.seed(0)
    state = da.random.RandomState(0)

    stokes_m_var = 40.
    cable_len = 100.
    nt = 500
    time = np.arange(nt)
    x = np.linspace(0., cable_len, 100)
    ts_cold = np.ones(nt) * 4.
    ts_warm = np.ones(nt) * 20.

    C_p = 15246
    C_m = 2400.
    dalpha_r = 0.0005284
    dalpha_m = 0.0004961
    dalpha_p = 0.0005607
    gamma = 482.6
    cold_mask = x < 0.5 * cable_len
    warm_mask = np.invert(cold_mask)  # == False
    temp_real = np.ones((len(x), nt))
    temp_real[cold_mask] *= ts_cold + 273.15
    temp_real[warm_mask] *= ts_warm + 273.15

    st = C_p * np.exp(-dalpha_r * x[:, None]) * np.exp(-dalpha_p * x[:, None]) * np.exp(
        -gamma / temp_real) / (1 - np.exp(-gamma / temp_real))
    ast = C_m * np.exp(-dalpha_r * x[:, None]) * np.exp(-dalpha_m * x[:, None]) / (
        1 - np.exp(-gamma / temp_real))
    rst = C_p * np.exp(-dalpha_r * (-x[:, None] + 100)) * np.exp(
        -dalpha_p * (-x[:, None] + 100)) * np.exp(-gamma / temp_real) / (
              1 - np.exp(-gamma / temp_real))
    rast = C_m * np.exp(-dalpha_r * (-x[:, None] + 100)) * np.exp(
        -dalpha_m * (-x[:, None] + 100)) / (1 - np.exp(-gamma / temp_real))

    st_m = st + stats.norm.rvs(size=st.shape, scale=stokes_m_var ** 0.5)
    ast_m = ast + stats.norm.rvs(size=ast.shape, scale=1.1 * stokes_m_var ** 0.5)
    rst_m = rst + stats.norm.rvs(size=rst.shape, scale=0.9 * stokes_m_var ** 0.5)
    rast_m = rast + stats.norm.rvs(size=rast.shape, scale=0.8 * stokes_m_var ** 0.5)

    print('alphaint', cable_len * (dalpha_p - dalpha_m))
    print('alpha', dalpha_p - dalpha_m)
    print('C', np.log(C_p / C_m))
    print('x0', x.max())

    ds = DataStore({
        'st':    (['x', 'time'], st),
        'ast':   (['x', 'time'], ast),
        'rst':   (['x', 'time'], rst),
        'rast':  (['x', 'time'], rast),
        'mst':   (['x', 'time'], st_m),
        'mast':  (['x', 'time'], ast_m),
        'mrst':  (['x', 'time'], rst_m),
        'mrast': (['x', 'time'], rast_m),
        'userAcquisitionTimeFW': (['time'], np.ones(nt)),
        'userAcquisitionTimeBW': (['time'], np.ones(nt)),
        'cold':  (['time'], ts_cold),
        'warm':  (['time'], ts_warm)
        },
        coords={
            'x':    x,
            'time': time},
        attrs={
            'customData:isDoubleEnded': '1'})

    sections = {
        'cold': [slice(0., 0.5 * cable_len)],
        'warm': [slice(0.5 * cable_len, cable_len)]}

    mst_var, _ = ds.variance_stokes(st_label='mst',
                                    sections=sections,
                                    suppress_info=True)
    mast_var, _ = ds.variance_stokes(st_label='mast',
                                     sections=sections,
                                     suppress_info=True)
    mrst_var, _ = ds.variance_stokes(st_label='mrst',
                                     sections=sections,
                                     suppress_info=True)
    mrast_var, _ = ds.variance_stokes(st_label='mrast',
                                      sections=sections,
                                      suppress_info=True)

    st_label = 'mst'
    ast_label = 'mast'
    rst_label = 'mrst'
    rast_label = 'mrast'

    # MC variqnce
    ds.calibration_double_ended(sections=sections,
                                st_label=st_label,
                                ast_label=ast_label,
                                rst_label=rst_label,
                                rast_label=rast_label,
                                st_var=mst_var,
                                ast_var=mast_var,
                                rst_var=mrst_var,
                                rast_var=mrast_var,
                                method='wls',
                                # conf_ints=[0.00135, 0.025, 0.15865, 0.5, 0.84135, 0.975, 0.99865],
                                conf_ints=[0.025, 0.5, 0.975],
                                ci_avg_time_flag=0,
                                store_tempvar='_var',
                                conf_ints_size=500,
                                solver='sparse',
                                da_random_state=state)

    # Calibrated variance
    stdsf1 = ds.ufunc_per_section(label='TMPF',
                                  func=np.std,
                                  temp_err=True,
                                  calc_per='stretch')
    stdsb1 = ds.ufunc_per_section(label='TMPB',
                                  func=np.std,
                                  temp_err=True,
                                  calc_per='stretch')

    # Use a single timestep to better check if the parameter uncertainties propagate
    ds1 = ds.isel(time=1)
    # Estimated VAR
    stdsf2 = ds1.ufunc_per_section(label='TMPF_MC_var',
                                   func=np.mean,
                                   temp_err=False,
                                   calc_per='stretch')
    stdsb2 = ds1.ufunc_per_section(label='TMPB_MC_var',
                                   func=np.mean,
                                   temp_err=False,
                                   calc_per='stretch')

    for (_, v1), (_, v2) in zip(stdsf1.items(), stdsf2.items()):
        for v1i, v2i in zip(v1, v2):
            print('Real VAR: ', v1i ** 2, 'Estimated VAR: ', v2i)
            np.testing.assert_almost_equal(v1i ** 2, v2i, decimal=2)

    for (_, v1), (_, v2) in zip(stdsb1.items(), stdsb2.items()):
        for v1i, v2i in zip(v1, v2):
            print('Real VAR: ', v1i ** 2, 'Estimated VAR: ', v2i)
            np.testing.assert_almost_equal(v1i ** 2, v2i, decimal=2)

    pass


def test_single_ended_variance_estimate_synthetic():
    import dask.array as da
    from dtscalibration import DataStore
    import numpy as np
    from scipy import stats

    np.random.seed(0)
    state = da.random.RandomState(0)

    stokes_m_var = 40.
    astokes_m_var = 60.
    cable_len = 100.
    nt = 50
    time = np.arange(nt)
    x = np.linspace(0., cable_len, 500)
    ts_cold = np.ones(nt) * 4.
    ts_warm = np.ones(nt) * 20.

    C_p = 15246
    C_m = 2400.
    dalpha_r = 0.0005284
    dalpha_m = 0.0004961
    dalpha_p = 0.0005607
    gamma = 482.6
    cold_mask = x < 0.5 * cable_len
    warm_mask = np.invert(cold_mask)  # == False
    temp_real = np.ones((len(x), nt))
    temp_real[cold_mask] *= ts_cold + 273.15
    temp_real[warm_mask] *= ts_warm + 273.15

    st = C_p * np.exp(-dalpha_r * x[:, None]) * np.exp(-dalpha_p * x[:, None]) * np.exp(
        -gamma / temp_real) / (1 - np.exp(-gamma / temp_real))
    ast = C_m * np.exp(-dalpha_r * x[:, None]) * np.exp(-dalpha_m * x[:, None]) / (
        1 - np.exp(-gamma / temp_real))
    st_m = st + stats.norm.rvs(size=st.shape, scale=stokes_m_var ** 0.5)
    ast_m = ast + stats.norm.rvs(size=ast.shape, scale=astokes_m_var ** 0.5)

    print('alphaint', cable_len * (dalpha_p - dalpha_m))
    print('alpha', dalpha_p - dalpha_m)
    print('C', np.log(C_p / C_m))
    print('x0', x.max())

    ds = DataStore({
        'st':    (['x', 'time'], st),
        'ast':   (['x', 'time'], ast),
        'mst':   (['x', 'time'], st_m),
        'mast':  (['x', 'time'], ast_m),
        'userAcquisitionTimeFW': (['time'], np.ones(nt)),
        'cold':  (['time'], ts_cold),
        'warm':  (['time'], ts_warm)
        },
        coords={
            'x':    x,
            'time': time},
        attrs={
            'customData:isDoubleEnded': '0'})

    sections = {
        'cold': [slice(0., 0.5 * cable_len)],
        'warm': [slice(0.5 * cable_len, cable_len)]}

    st_label = 'mst'
    ast_label = 'mast'

    mst_var, _ = ds.variance_stokes(st_label=st_label,
                                    sections=sections,
                                    suppress_info=True)
    mast_var, _ = ds.variance_stokes(st_label=ast_label,
                                     sections=sections,
                                     suppress_info=True)

    # MC variqnce
    ds.calibration_single_ended(sections=sections,
                                st_label=st_label,
                                ast_label=ast_label,
                                st_var=mst_var,
                                ast_var=mast_var,
                                method='wls',
                                # conf_ints=[0.00135, 0.025, 0.15865, 0.5, 0.84135, 0.975, 0.99865],
                                conf_ints=[0.025, 0.5, 0.975],
                                ci_avg_time_flag=0,
                                store_tempvar='_var',
                                conf_ints_size=500,
                                solver='sparse',
                                da_random_state=state)

    # Calibrated variance
    stdsf1 = ds.ufunc_per_section(label='TMPF',
                                  func=np.std,
                                  temp_err=True,
                                  calc_per='stretch',
                                  ddof=1)

    # Use a single timestep to better check if the parameter uncertainties propagate
    ds1 = ds.isel(time=1)
    # Estimated VAR
    stdsf2 = ds1.ufunc_per_section(label='TMPF_MC_var',
                                   func=np.mean,
                                   temp_err=False,
                                   calc_per='stretch')

    for (_, v1), (_, v2) in zip(stdsf1.items(), stdsf2.items()):
        for v1i, v2i in zip(v1, v2):
            print('Real VAR: ', v1i ** 2, 'Estimated VAR: ', v2i)
            np.testing.assert_almost_equal(v1i ** 2, v2i, decimal=2)

    pass


def test_variance_of_stokes():
    correct_var = 40.16
    filepath = data_dir_double_ended2
    ds = read_silixa_files(
        directory=filepath,
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
    np.testing.assert_almost_equal(I_var, correct_var, decimal=1)

    I_var, _ = ds.variance_stokes(st_label='ST',
                                  sections=sections,
                                  use_statsmodels=False)
    np.testing.assert_almost_equal(I_var, correct_var, decimal=1)

    ds_dask = ds.chunk(chunks={})
    I_var, _ = ds_dask.variance_stokes(
        st_label='ST',
        sections=sections,
        use_statsmodels=False)
    np.testing.assert_almost_equal(I_var, correct_var, decimal=1)

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

    ds = DataStore({
        'test_ST': (['x', 'time'], y),
        'probe1Temperature':  (['time'], range(nt)),
        'userAcquisitionTimeFW': (['time'], np.ones(nt)),
        },
        coords={
            'x':    x,
            'time': range(nt)},
        attrs={'customData:isDoubleEnded': '0'})

    sections = {'probe1Temperature': [slice(0., 20.), ]}
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
    ds = read_silixa_files(
        directory=filepath,
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

    ps_sol, ps_var = wls_stats(X, y, w=1, calc_cov=0)
    p_sol, p_var = wls_sparse(X, y, w=1, calc_cov=0, x0=beta_0)

    np.testing.assert_array_almost_equal(beta, ps_sol, decimal=8)
    np.testing.assert_array_almost_equal(beta, p_sol, decimal=8)

    # now with weights
    dec = 8
    ps_sol, ps_var, ps_cov = wls_stats(X, y_meas, w=beta_w, calc_cov=True)
    p_sol, p_var, p_cov = wls_sparse(X, y_meas, w=beta_w, calc_cov=True, x0=beta_0)

    np.testing.assert_array_almost_equal(p_sol, ps_sol, decimal=dec)
    np.testing.assert_array_almost_equal(p_var, ps_var, decimal=dec)
    np.testing.assert_array_almost_equal(p_cov, ps_cov, decimal=dec)

    # Test array sparse
    Xsp = sp.coo_matrix(X)
    psp_sol, psp_var, psp_cov = wls_stats(Xsp, y_meas, w=beta_w, calc_cov=True)

    np.testing.assert_array_almost_equal(p_sol, psp_sol, decimal=dec)
    np.testing.assert_array_almost_equal(p_var, psp_var, decimal=dec)
    np.testing.assert_array_almost_equal(p_cov, psp_cov, decimal=dec)

    pass

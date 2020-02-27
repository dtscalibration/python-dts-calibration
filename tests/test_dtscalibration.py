# coding=utf-8
import os

import numpy as np
import pytest
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


def assert_almost_equal_verbose(actual, desired, verbose=False, **kwargs):
    """Print the actual precision decimals"""
    err = np.abs(actual - desired).max()
    dec = -np.ceil(np.log10(err))

    if not (np.isfinite(dec)):
        dec = 18.

    if verbose:
        print(dec)

    m = "\n>>>>>The actual precision is: " + str(float(dec))

    # assert int(dec) == kwargs['decimal'], \
    #     'The actual precision is different: ' + str(dec)
    desired2 = np.broadcast_to(desired, actual.shape)
    np.testing.assert_almost_equal(actual, desired2, err_msg=m, **kwargs)
    pass


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
            'isDoubleEnded': '1'})

    sections = {
        'cold': [slice(0., 0.5 * cable_len)],
        'warm': [slice(0.5 * cable_len, cable_len)]}

    mst_var, _ = ds.variance_stokes(st_label='mst',
                                    sections=sections)
    mast_var, _ = ds.variance_stokes(st_label='mast',
                                     sections=sections)
    mrst_var, _ = ds.variance_stokes(st_label='mrst',
                                     sections=sections)
    mrast_var, _ = ds.variance_stokes(st_label='mrast',
                                      sections=sections)

    st_label = 'mst'
    ast_label = 'mast'
    rst_label = 'mrst'
    rast_label = 'mrast'

    mst_var = float(mst_var)
    mast_var = float(mast_var)
    mrst_var = float(mrst_var)
    mrast_var = float(mrast_var)

    # MC variance
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
                                solver='sparse')

    assert_almost_equal_verbose(ds.TMPF.mean(), 12., decimal=2)
    assert_almost_equal_verbose(ds.TMPB.mean(), 12., decimal=3)

    ds.conf_int_double_ended(
        p_val='p_val',
        p_cov='p_cov',
        st_label=st_label,
        ast_label=ast_label,
        rst_label=rst_label,
        rast_label=rast_label,
        st_var=mst_var,
        ast_var=mast_var,
        rst_var=mrst_var,
        rast_var=mrast_var,
        store_tmpf='TMPF',
        store_tmpb='TMPB',
        store_tmpw='TMPW',
        store_tempvar='_var',
        conf_ints=[2.5, 50., 97.5],
        mc_sample_size=100,
        ci_avg_time_flag=False,
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
            print('Real VAR: ', v1i ** 2, 'Estimated VAR: ', float(v2i))
            assert_almost_equal_verbose(v1i ** 2, v2i, decimal=2)

    for (_, v1), (_, v2) in zip(stdsb1.items(), stdsb2.items()):
        for v1i, v2i in zip(v1, v2):
            print('Real VAR: ', v1i ** 2, 'Estimated VAR: ', float(v2i))
            assert_almost_equal_verbose(v1i ** 2, v2i, decimal=2)

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
            'isDoubleEnded': '0'})

    sections = {
        'cold': [slice(0., 0.5 * cable_len)],
        'warm': [slice(0.5 * cable_len, cable_len)]}

    st_label = 'mst'
    ast_label = 'mast'

    mst_var, _ = ds.variance_stokes(st_label=st_label,
                                    sections=sections)
    mast_var, _ = ds.variance_stokes(st_label=ast_label,
                                     sections=sections)
    mst_var = float(mst_var)
    mast_var = float(mast_var)

    # MC variqnce
    ds.calibration_single_ended(sections=sections,
                                st_label=st_label,
                                ast_label=ast_label,
                                st_var=mst_var,
                                ast_var=mast_var,
                                method='wls',
                                solver='sparse')

    ds.conf_int_single_ended(
        p_val='p_val',
        p_cov='p_cov',
        st_label=st_label,
        ast_label=ast_label,
        st_var=mst_var,
        ast_var=mast_var,
        store_tmpf='TMPF',
        store_tempvar='_var',
        conf_ints=[2.5, 50., 97.5],
        mc_sample_size=50,
        ci_avg_time_flag=False,
        da_random_state=state
        )

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
            print('Real VAR: ', v1i ** 2, 'Estimated VAR: ', float(v2i))
            assert_almost_equal_verbose(v1i ** 2, v2i, decimal=2)

    pass


@pytest.mark.skip(reason="Not enough measurements in time. Use exponential "
                         "instead.")
def test_variance_of_stokes():
    np.random.seed(0)

    correct_var = 9.045
    filepath = data_dir_double_ended2
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf='UTC',
        file_ext='*.xml')
    sections = {
        'probe1Temperature': [slice(7.5, 17.), slice(70., 80.)],  # cold bath
        'probe2Temperature': [slice(24., 34.), slice(85., 95.)],  # warm bath
        }

    I_var, _ = ds.variance_stokes(st_label='ST',
                                  sections=sections)
    assert_almost_equal_verbose(I_var, correct_var, decimal=1)

    ds_dask = ds.chunk(chunks={})
    I_var, _ = ds_dask.variance_stokes(
        st_label='ST',
        sections=sections)
    assert_almost_equal_verbose(I_var, correct_var, decimal=1)

    pass


def test_variance_of_stokes_synthetic():
    """
    Produces a synthetic Stokes measurement with a known noise distribution. Check if same
    variance is obtained.

    Returns
    -------

    """
    np.random.seed(0)

    yvar = 5.

    nx = 500
    x = np.linspace(0., 20., nx)

    nt = 200
    G = np.linspace(3000, 4000, nt)[None]

    y = G * np.exp(-0.001 * x[:, None])

    y += stats.norm.rvs(size=y.size,
                        scale=yvar ** 0.5).reshape(y.shape)

    ds = DataStore({
        'ST': (['x', 'time'], y),
        'probe1Temperature':  (['time'], range(nt)),
        'userAcquisitionTimeFW': (['time'], np.ones(nt)),
        },
        coords={
            'x':    x,
            'time': range(nt)},
        attrs={'isDoubleEnded': '0'})

    sections = {'probe1Temperature': [slice(0., 20.), ]}
    test_ST_var, _ = ds.variance_stokes(st_label='ST',
                                        sections=sections)

    assert_almost_equal_verbose(test_ST_var, yvar, decimal=1)

    test_ST_var, _ = ds.variance_stokes(st_label='ST',
                                        sections=sections)

    assert_almost_equal_verbose(test_ST_var, yvar, decimal=1)
    pass


def test_variance_of_stokes_linear_synthetic():
    """
    Produces a synthetic Stokes measurement with a known noise distribution.
    Check if same variance is obtained.

    Returns
    -------

    """
    np.random.seed(0)

    var_slope = 0.01

    nx = 500
    x = np.linspace(0., 20., nx)

    nt = 200
    G = np.linspace(500, 4000, nt)[None]
    c_no_noise = G * np.exp(-0.001 * x[:, None])

    c_lin_var_through_zero = stats.norm.rvs(
        loc=c_no_noise,
        # size=y.size,
        scale=(var_slope * c_no_noise) ** 0.5)
    ds = DataStore({
        'ST':                     (['x', 'time'], c_no_noise),
        'c_lin_var_through_zero': (['x', 'time'], c_lin_var_through_zero),
        'probe1Temperature':      (['time'], range(nt)),
        'userAcquisitionTimeFW':  (['time'], np.ones(nt)),
        },
        coords={
            'x':    x,
            'time': range(nt)},
        attrs={'isDoubleEnded': '0'})

    sections = {'probe1Temperature': [slice(0., 20.), ]}
    test_ST_var, _ = ds.variance_stokes(st_label='ST',
                                        sections=sections)

    # If fit is forced through zero. Only Poisson distributed noise
    slope, offset, st_sort_mean, st_sort_var, resid, var_fun = \
        ds.variance_stokes_linear(
            'c_lin_var_through_zero', nbin=10, through_zero=True,
            plot_fit=False)
    assert_almost_equal_verbose(slope, var_slope, decimal=3)

    # Fit accounts for Poisson noise plus white noise
    slope, offset, st_sort_mean, st_sort_var, resid, var_fun = \
        ds.variance_stokes_linear(
            'c_lin_var_through_zero', nbin=100, through_zero=False)
    assert_almost_equal_verbose(slope, var_slope, decimal=3)
    assert_almost_equal_verbose(offset, 0., decimal=0)

    pass


def test_exponential_variance_of_stokes():
    correct_var = 11.86535
    filepath = data_dir_double_ended2
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf='UTC',
        file_ext='*.xml')
    sections = {
        'probe1Temperature': [slice(7.5, 17.), slice(70., 80.)],  # cold bath
        'probe2Temperature': [slice(24., 34.), slice(85., 95.)],  # warm bath
        }

    I_var, _ = ds.variance_stokes_exponential(
        st_label='ST', sections=sections)
    assert_almost_equal_verbose(I_var, correct_var, decimal=5)

    ds_dask = ds.chunk(chunks={})
    I_var, _ = ds_dask.variance_stokes_exponential(
        st_label='ST',
        sections=sections)
    assert_almost_equal_verbose(I_var, correct_var, decimal=5)

    pass


def test_exponential_variance_of_stokes_synthetic():
    """
    Produces a synthetic Stokes measurement with a known noise distribution. Check if same
    variance is obtained.

    Returns
    -------

    """
    yvar = 5.

    nx = 500
    x = np.linspace(0., 20., nx)

    nt = 200
    beta = np.linspace(3000, 4000, nt)[None]

    y = beta * np.exp(-0.001 * x[:, None])

    y += stats.norm.rvs(size=y.size,
                        scale=yvar ** 0.5).reshape(y.shape)

    ds = DataStore({
        'ST': (['x', 'time'], y),
        'probe1Temperature':  (['time'], range(nt)),
        'userAcquisitionTimeFW': (['time'], np.ones(nt)),
        },
        coords={
            'x':    x,
            'time': range(nt)},
        attrs={'isDoubleEnded': '0'})

    sections = {'probe1Temperature': [slice(0., 20.), ]}
    test_ST_var, _ = ds.variance_stokes_exponential(
        st_label='ST', sections=sections)

    assert_almost_equal_verbose(test_ST_var, yvar, decimal=1)
    pass


def test_double_ended_ols_wls_estimate_synthetic():
    """Checks whether the coefficients are correctly defined by creating a
    synthetic measurement set, and derive the parameters from this set.
    Without variance.
    They should be the same as the parameters used to create the synthetic
    measurment set"""
    from dtscalibration import DataStore
    import numpy as np

    np.random.seed(0)

    cable_len = 100.
    nt = 50
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

    st = C_p * np.exp(-(dalpha_r + dalpha_p) * x[:, None]) * \
        np.exp(gamma / temp_real) / (np.exp(gamma / temp_real) - 1)
    ast = C_m * np.exp(-(dalpha_r + dalpha_m) * x[:, None]) / \
        (np.exp(gamma / temp_real) - 1)
    rst = C_p * np.exp(-(dalpha_r + dalpha_p) * (-x[:, None] + cable_len)) * \
        np.exp(gamma / temp_real) / (np.exp(gamma / temp_real) - 1)
    rast = C_m * np.exp(-(dalpha_r + dalpha_m) * (-x[:, None] + cable_len)) / \
        (np.exp(gamma / temp_real) - 1)

    alpha = np.mean(np.log(rst / rast) - np.log(st / ast), axis=1) / 2
    alpha -= alpha[0]  # the first x-index is where to start counting

    ds = DataStore({
        'st':                    (['x', 'time'], st),
        'ast':                   (['x', 'time'], ast),
        'rst':                   (['x', 'time'], rst),
        'rast':                  (['x', 'time'], rast),
        'userAcquisitionTimeFW': (['time'], np.ones(nt)),
        'userAcquisitionTimeBW': (['time'], np.ones(nt)),
        'cold':                  (['time'], ts_cold),
        'warm':                  (['time'], ts_warm)
        },
        coords={
            'x':    x,
            'time': time},
        attrs={
            'isDoubleEnded': '1'})

    sections = {
        'cold': [slice(0., 0.4 * cable_len)],
        'warm': [slice(0.65 * cable_len, cable_len)]}

    # OLS
    ds.calibration_double_ended(sections=sections,
                                st_label='st',
                                ast_label='ast',
                                rst_label='rst',
                                rast_label='rast',
                                method='ols',
                                solver='sparse')

    assert_almost_equal_verbose(
        ds.gamma.values, gamma, decimal=11)
    assert_almost_equal_verbose(
        ds.alpha.values, alpha, decimal=12)  # 13 in 64-bit
    assert_almost_equal_verbose(
        ds.TMPF.values, temp_real - 273.15, decimal=10)
    assert_almost_equal_verbose(
        ds.TMPB.values, temp_real - 273.15, decimal=10)
    assert_almost_equal_verbose(
        ds.TMPW.values, temp_real - 273.15, decimal=11)

    # WLS
    ds.calibration_double_ended(sections=sections,
                                st_label='st',
                                ast_label='ast',
                                rst_label='rst',
                                rast_label='rast',
                                st_var=1e-7,
                                ast_var=1e-7,
                                rst_var=1e-7,
                                rast_var=1e-7,
                                method='wls',
                                solver='sparse',
                                tmpw_mc_size=5)

    assert_almost_equal_verbose(
        ds.gamma.values, gamma, decimal=10)
    assert_almost_equal_verbose(
        ds.alpha.values, alpha, decimal=8)
    assert_almost_equal_verbose(
        ds.TMPF.values, temp_real - 273.15, decimal=6)
    assert_almost_equal_verbose(
        ds.TMPB.values, temp_real - 273.15, decimal=6)
    assert_almost_equal_verbose(
        ds.TMPW.values, temp_real - 273.15, decimal=6)


def test_double_ended_ols_wls_estimate_synthetic_df_and_db_are_different():
    """Checks whether the coefficients are correctly defined by creating a
    synthetic measurement set, and derive the parameters from this set.
    Without variance.
    They should be the same as the parameters used to create the synthetic
    measurment set. This one has a different D for the forward channel than
    for the backward channel."""
    from dtscalibration import DataStore
    import numpy as np

    np.random.seed(0)

    cable_len = 100.
    nt = 3
    time = np.arange(nt)
    x = np.linspace(0., cable_len, 8)
    ts_cold = np.ones(nt) * 4. + np.cos(time) * 4
    ts_warm = np.ones(nt) * 20. + -np.sin(time) * 4

    C_p = 1324  # 1/2 * E0 * v * K_+/lam_+^4
    eta_pf = np.cos(time) / 10 + 1  # eta_+ (gain factor forward channel)
    eta_pb = np.sin(time) / 10 + 1  # eta_- (gain factor backward channel)
    C_m = 5000.
    eta_mf = np.cos(time + np.pi / 8) / 10 + 1
    eta_mb = np.sin(time + np.pi / 8) / 10 + 1
    dalpha_r = 0.005284
    dalpha_m = 0.004961
    dalpha_p = 0.005607
    gamma = 482.6

    temp_real_kelvin = np.zeros((len(x), nt)) + 273.15
    temp_real_kelvin[x < 0.2 * cable_len] += ts_cold[None]
    temp_real_kelvin[x > 0.85 * cable_len] += ts_warm[None]
    temp_real_celsius = temp_real_kelvin - 273.15

    st = eta_pf[None] * C_p * np.exp(-dalpha_r * x[:, None]) * \
        np.exp(-dalpha_p * x[:, None]) * np.exp(gamma / temp_real_kelvin) / \
        (np.exp(gamma / temp_real_kelvin) - 1)
    ast = eta_mf[None] * C_m * np.exp(-dalpha_r * x[:, None]) * \
        np.exp(-dalpha_m * x[:, None]) / (np.exp(gamma / temp_real_kelvin) - 1)
    rst = eta_pb[None] * C_p * np.exp(-dalpha_r * (-x[:, None] + cable_len)) * \
        np.exp(-dalpha_p * (-x[:, None] + cable_len)) * \
        np.exp(gamma / temp_real_kelvin) / (
        np.exp(gamma / temp_real_kelvin) - 1)
    rast = eta_mb[None] * C_m * np.exp(
        -dalpha_r * (-x[:, None] + cable_len)) * np.exp(
        -dalpha_m * (-x[:, None] + cable_len)) / \
        (np.exp(gamma / temp_real_kelvin) - 1)

    c_f = np.log(eta_mf * C_m / (eta_pf * C_p))
    c_b = np.log(eta_mb * C_m / (eta_pb * C_p))

    dalpha = dalpha_p - dalpha_m  # \Delta\alpha
    alpha_int = cable_len * dalpha

    df = c_f  # reference section starts at first x-index
    db = c_b + alpha_int
    i_fw = np.log(st / ast)
    i_bw = np.log(rst / rast)

    E_real = (i_bw - i_fw) / 2 + (db - df) / 2

    ds = DataStore({
        'st':                    (['x', 'time'], st),
        'ast':                   (['x', 'time'], ast),
        'rst':                   (['x', 'time'], rst),
        'rast':                  (['x', 'time'], rast),
        'userAcquisitionTimeFW': (['time'], np.ones(nt)),
        'userAcquisitionTimeBW': (['time'], np.ones(nt)),
        'cold':                  (['time'], ts_cold),
        'warm':                  (['time'], ts_warm)
        },
        coords={
            'x':    x,
            'time': time},
        attrs={
            'isDoubleEnded': '1'})

    ds.sections = {
        'cold': [slice(0., 0.09 * cable_len)],
        'warm': [slice(0.9 * cable_len, cable_len)]}

    real_ans2 = np.concatenate(([gamma], df, db, E_real[:, 0]))

    ds.calibration_double_ended(
        st_label='st',
        ast_label='ast',
        rst_label='rst',
        rast_label='rast',
        st_var=1.5,
        ast_var=1.5,
        rst_var=1.,
        rast_var=1.,
        method='wls',
        solver='sparse',
        tmpw_mc_size=1000,
        fix_gamma=(gamma, 0.),
        remove_mc_set_flag=True)

    assert_almost_equal_verbose(df, ds.df.values, decimal=14)
    assert_almost_equal_verbose(db, ds.db.values, decimal=13)
    assert_almost_equal_verbose(x * (dalpha_p - dalpha_m),
                                ds.alpha.values - ds.alpha.values[0],
                                decimal=13)
    assert np.all(np.abs(real_ans2 - ds.p_val.values) < 1e-10)
    assert_almost_equal_verbose(temp_real_celsius, ds.TMPF.values, decimal=10)
    assert_almost_equal_verbose(temp_real_celsius, ds.TMPB.values, decimal=10)
    assert_almost_equal_verbose(temp_real_celsius, ds.TMPW.values, decimal=10)
    pass


def test_double_ended_asymmetrical_attenuation():
    from dtscalibration import DataStore
    import numpy as np

    np.random.seed(0)

    cable_len = 100.
    nt = 3
    time = np.arange(nt)
    nx_per_sec = 1
    nx = nx_per_sec * 8
    x = np.linspace(0., cable_len, nx)
    ts_cold = 4. + np.cos(time) * 4
    ts_warm = 20. + -np.sin(time) * 4
    ts_ground = 6.

    C_p = 1324  # 1/2 * E0 * v * K_+/lam_+^4
    eta_pf = np.cos(time) / 10 + 1  # eta_+ (gain factor forward channel)
    eta_pb = np.sin(time) / 10 + 1  # eta_- (gain factor backward channel)
    C_m = 5000.
    eta_mf = np.cos(time + np.pi / 8) / 10 + 1
    eta_mb = np.sin(time + np.pi / 8) / 10 + 1
    dalpha_r = 0.005284
    dalpha_m = 0.004961
    dalpha_p = 0.005607
    gamma = 482.6
    talph_fw = 0.9
    talph_bw = 0.8

    temp_real_kelvin = np.zeros((len(x), nt)) + 273.15
    temp_real_kelvin[:nx_per_sec] += ts_cold[None]
    temp_real_kelvin[nx_per_sec:2 * nx_per_sec] += ts_warm[None]
    temp_real_kelvin[-nx_per_sec:] += ts_cold[None]
    temp_real_kelvin[-2 * nx_per_sec:-nx_per_sec] += ts_warm[None]
    temp_real_kelvin[2 * nx_per_sec:-2 * nx_per_sec] += ts_ground
    temp_real_celsius = temp_real_kelvin - 273.15

    st = eta_pf[None] * C_p * np.exp(-dalpha_r * x[:, None]) * \
        np.exp(-dalpha_p * x[:, None]) * np.exp(gamma / temp_real_kelvin) / \
        (np.exp(gamma / temp_real_kelvin) - 1)
    st[4 * nx_per_sec:] *= talph_fw
    ast = eta_mf[None] * C_m * np.exp(-dalpha_r * x[:, None]) * \
        np.exp(-dalpha_m * x[:, None]) / (
        np.exp(gamma / temp_real_kelvin) - 1)
    rst = eta_pb[None] * C_p * np.exp(-dalpha_r * (-x[:, None] + cable_len)) * \
        np.exp(-dalpha_p * (-x[:, None] + cable_len)) * \
        np.exp(gamma / temp_real_kelvin) / (
        np.exp(gamma / temp_real_kelvin) - 1)
    rst[:4 * nx_per_sec] *= talph_bw
    rast = eta_mb[None] * C_m * np.exp(
        -dalpha_r * (-x[:, None] + cable_len)) * np.exp(
        -dalpha_m * (-x[:, None] + cable_len)) / \
        (np.exp(gamma / temp_real_kelvin) - 1)

    ds = DataStore({
        'TMPR':                  (['x', 'time'], temp_real_celsius),
        'st':                    (['x', 'time'], st),
        'ast':                   (['x', 'time'], ast),
        'rst':                   (['x', 'time'], rst),
        'rast':                  (['x', 'time'], rast),
        'userAcquisitionTimeFW': (['time'], np.ones(nt)),
        'userAcquisitionTimeBW': (['time'], np.ones(nt)),
        'cold':                  (['time'], ts_cold),
        'warm':                  (['time'], ts_warm)
        },
        coords={
            'x':    x,
            'time': time},
        attrs={
            'isDoubleEnded': '1'})

    ds.sections = {
        'cold': [slice(0., x[nx_per_sec - 1]),
                 slice(x[-nx_per_sec], x[-1])],
        'warm': [slice(x[nx_per_sec], x[2 * nx_per_sec - 1]),
                 slice(x[-2 * nx_per_sec], x[-1 * nx_per_sec - 1])]}

    ds.calibration_double_ended(
        st_label='st',
        ast_label='ast',
        rst_label='rst',
        rast_label='rast',
        st_var=1.5,
        ast_var=1.5,
        rst_var=1.,
        rast_var=1.,
        method='wls',
        solver='sparse',
        tmpw_mc_size=1000,
        remove_mc_set_flag=True,
        transient_asym_att_x=[50.])

    assert_almost_equal_verbose(temp_real_celsius, ds.TMPF.values, decimal=7)
    assert_almost_equal_verbose(temp_real_celsius, ds.TMPB.values, decimal=7)
    assert_almost_equal_verbose(temp_real_celsius, ds.TMPW.values, decimal=7)
    pass


def test_double_ended_matching_sections_and_single_asym_att():
    from dtscalibration import DataStore
    import numpy as np

    np.random.seed(0)

    cable_len = 100.
    nt = 1
    time = np.arange(nt)
    nx_per_sec = 2
    nx = nx_per_sec * 8
    x = np.linspace(0., cable_len, nx)
    ts_cold = 4. + np.cos(time) * 4
    ts_warm = 20. + -np.sin(time) * 4
    ts_ground = 6.

    C_p = 1324  # 1/2 * E0 * v * K_+/lam_+^4
    eta_pf = np.cos(time) / 10 + 1  # eta_+ (gain factor forward channel)
    eta_pb = np.sin(time) / 10 + 1  # eta_- (gain factor backward channel)
    C_m = 5000.
    eta_mf = np.cos(time + np.pi / 8) / 10 + 1
    eta_mb = np.sin(time + np.pi / 8) / 10 + 1
    dalpha_r = 0.005284
    dalpha_m = 0.004961
    dalpha_p = 0.005607
    gamma = 482.6
    talph_fw = 0.9
    talph_bw = 0.8

    temp_real_kelvin = np.zeros((len(x), nt)) + 273.15
    temp_real_kelvin[:nx_per_sec] += ts_cold[None]
    temp_real_kelvin[nx_per_sec:2 * nx_per_sec] += ts_warm[None]
    temp_real_kelvin[-nx_per_sec:] += ts_cold[None]
    temp_real_kelvin[-2 * nx_per_sec:-nx_per_sec] += ts_warm[None]
    temp_real_kelvin[2 * nx_per_sec:-2 * nx_per_sec] += ts_ground
    temp_real_celsius = temp_real_kelvin - 273.15

    st = eta_pf[None] * C_p * np.exp(-dalpha_r * x[:, None]) * \
        np.exp(-dalpha_p * x[:, None]) * np.exp(gamma / temp_real_kelvin) / \
        (np.exp(gamma / temp_real_kelvin) - 1)
    st[4 * nx_per_sec:] *= talph_fw
    ast = eta_mf[None] * C_m * np.exp(-dalpha_r * x[:, None]) * \
        np.exp(-dalpha_m * x[:, None]) / (
        np.exp(gamma / temp_real_kelvin) - 1)
    rst = eta_pb[None] * C_p * np.exp(-dalpha_r * (-x[:, None] + cable_len)) * \
        np.exp(-dalpha_p * (-x[:, None] + cable_len)) * \
        np.exp(gamma / temp_real_kelvin) / (
        np.exp(gamma / temp_real_kelvin) - 1)
    rst[:4 * nx_per_sec] *= talph_bw
    rast = eta_mb[None] * C_m * np.exp(
        -dalpha_r * (-x[:, None] + cable_len)) * np.exp(
        -dalpha_m * (-x[:, None] + cable_len)) / \
        (np.exp(gamma / temp_real_kelvin) - 1)

    ds = DataStore({
        'TMPR':                  (['x', 'time'], temp_real_celsius),
        'st':                    (['x', 'time'], st),
        'ast':                   (['x', 'time'], ast),
        'rst':                   (['x', 'time'], rst),
        'rast':                  (['x', 'time'], rast),
        'userAcquisitionTimeFW': (['time'], np.ones(nt)),
        'userAcquisitionTimeBW': (['time'], np.ones(nt)),
        'cold':                  (['time'], ts_cold),
        'warm':                  (['time'], ts_warm)
        },
        coords={
            'x':    x,
            'time': time},
        attrs={
            'isDoubleEnded': '1'})

    ds.sections = {
        'cold': [slice(0., x[nx_per_sec - 1])],
        'warm': [slice(x[nx_per_sec], x[2 * nx_per_sec - 1])]}

    ds.calibration_double_ended(
        st_label='st',
        ast_label='ast',
        rst_label='rst',
        rast_label='rast',
        st_var=1.5,
        ast_var=1.5,
        rst_var=1.,
        rast_var=1.,
        method='wls',
        solver='sparse',
        tmpw_mc_size=3,
        remove_mc_set_flag=True,
        transient_asym_att_x=[50.],
        matching_sections=[(slice(x[3 * nx_per_sec], x[4 * nx_per_sec - 1]),
                            slice(x[4 * nx_per_sec], x[5 * nx_per_sec - 1]),
                            True)])

    assert_almost_equal_verbose(temp_real_celsius, ds.TMPF.values, decimal=7)
    assert_almost_equal_verbose(temp_real_celsius, ds.TMPB.values, decimal=7)
    assert_almost_equal_verbose(temp_real_celsius, ds.TMPW.values, decimal=7)


def test_double_ended_ols_wls_fix_gamma_estimate_synthetic():
    """Checks whether the coefficients are correctly defined by creating a
    synthetic measurement set, and derive the parameters from this set.
    Without variance.
    They should be the same as the parameters used to create the synthetic
    measurment set"""
    from dtscalibration import DataStore
    import numpy as np

    np.random.seed(0)

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

    st = C_p * np.exp(-(dalpha_r + dalpha_p) * x[:, None]) * \
        np.exp(gamma / temp_real) / (np.exp(gamma / temp_real) - 1)
    ast = C_m * np.exp(-(dalpha_r + dalpha_m) * x[:, None]) / \
        (np.exp(gamma / temp_real) - 1)
    rst = C_p * np.exp(-(dalpha_r + dalpha_p) * (-x[:, None] + cable_len)) * \
        np.exp(gamma / temp_real) / (np.exp(gamma / temp_real) - 1)
    rast = C_m * np.exp(-(dalpha_r + dalpha_m) * (-x[:, None] + cable_len)) / \
        (np.exp(gamma / temp_real) - 1)

    alpha = np.mean(np.log(rst / rast) - np.log(st / ast), axis=1) / 2
    alpha -= alpha[0]  # the first x-index is where to start counting
    dalpha = dalpha_p - dalpha_m
    alpha2 = x * dalpha

    # to ensure the st, rst, ast, rast were correctly defined.
    np.testing.assert_allclose(alpha2, alpha, atol=1e-15, rtol=0)

    ds = DataStore({
        'st':                    (['x', 'time'], st),
        'ast':                   (['x', 'time'], ast),
        'rst':                   (['x', 'time'], rst),
        'rast':                  (['x', 'time'], rast),
        'userAcquisitionTimeFW': (['time'], np.ones(nt)),
        'userAcquisitionTimeBW': (['time'], np.ones(nt)),
        'cold':                  (['time'], ts_cold),
        'warm':                  (['time'], ts_warm)
        },
        coords={
            'x':    x,
            'time': time},
        attrs={
            'isDoubleEnded': '1'})

    sections = {
        'cold': [slice(0., 0.35 * cable_len)],
        'warm': [slice(0.67 * cable_len, cable_len)]}

    # OLS
    ds.calibration_double_ended(sections=sections,
                                st_label='st',
                                ast_label='ast',
                                rst_label='rst',
                                rast_label='rast',
                                method='ols',
                                solver='sparse',
                                fix_gamma=(gamma, 0.))

    assert_almost_equal_verbose(
        ds.gamma.values, gamma, decimal=18)
    assert_almost_equal_verbose(
        ds.alpha.values, alpha, decimal=10)  # 11 in 64-bit
    assert_almost_equal_verbose(
        ds.TMPF.values, temp_real - 273.15, decimal=8)
    assert_almost_equal_verbose(
        ds.TMPB.values, temp_real - 273.15, decimal=8)
    assert_almost_equal_verbose(
        ds.TMPW.values, temp_real - 273.15, decimal=8)

    # WLS
    ds.calibration_double_ended(sections=sections,
                                st_label='st',
                                ast_label='ast',
                                rst_label='rst',
                                rast_label='rast',
                                st_var=1e-12,
                                ast_var=1e-12,
                                rst_var=1e-12,
                                rast_var=1e-12,
                                method='wls',
                                solver='sparse',
                                tmpw_mc_size=5,
                                fix_gamma=(gamma, 0.))

    assert_almost_equal_verbose(
        ds.gamma.values, gamma, decimal=18)
    assert_almost_equal_verbose(
        ds.alpha.values, alpha, decimal=9)
    assert_almost_equal_verbose(
        ds.TMPF.values, temp_real - 273.15, decimal=6)
    assert_almost_equal_verbose(
        ds.TMPB.values, temp_real - 273.15, decimal=6)
    assert_almost_equal_verbose(
        ds.TMPW.values, temp_real - 273.15, decimal=6)

    pass


def test_double_ended_ols_wls_fix_alpha_estimate_synthetic():
    """Checks whether the coefficients are correctly defined by creating a
    synthetic measurement set, and derive the parameters from this set.
    Without variance.
    They should be the same as the parameters used to create the synthetic
    measurment set"""
    from dtscalibration import DataStore
    import numpy as np

    np.random.seed(0)

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

    st = C_p * np.exp(-(dalpha_r + dalpha_p) * x[:, None]) * \
        np.exp(gamma / temp_real) / (np.exp(gamma / temp_real) - 1)
    ast = C_m * np.exp(-(dalpha_r + dalpha_m) * x[:, None]) / \
        (np.exp(gamma / temp_real) - 1)
    rst = C_p * np.exp(-(dalpha_r + dalpha_p) * (-x[:, None] + cable_len)) * \
        np.exp(gamma / temp_real) / (np.exp(gamma / temp_real) - 1)
    rast = C_m * np.exp(-(dalpha_r + dalpha_m) * (-x[:, None] + cable_len)) / \
        (np.exp(gamma / temp_real) - 1)

    alpha = np.mean(np.log(rst / rast) - np.log(st / ast), axis=1) / 2
    alpha -= alpha[0]  # the first x-index is where to start counting

    ds = DataStore({
        'st':                    (['x', 'time'], st),
        'ast':                   (['x', 'time'], ast),
        'rst':                   (['x', 'time'], rst),
        'rast':                  (['x', 'time'], rast),
        'userAcquisitionTimeFW': (['time'], np.ones(nt)),
        'userAcquisitionTimeBW': (['time'], np.ones(nt)),
        'cold':                  (['time'], ts_cold),
        'warm':                  (['time'], ts_warm)
        },
        coords={
            'x':    x,
            'time': time},
        attrs={
            'isDoubleEnded': '1'})

    sections = {
        'cold': [slice(0., 0.4 * cable_len)],
        'warm': [slice(0.78 * cable_len, cable_len)]}

    # OLS
    ds.calibration_double_ended(sections=sections,
                                st_label='st',
                                ast_label='ast',
                                rst_label='rst',
                                rast_label='rast',
                                method='ols',
                                solver='sparse',
                                fix_alpha=(alpha, np.zeros_like(alpha)))

    assert_almost_equal_verbose(
        ds.gamma.values, gamma, decimal=9)
    assert_almost_equal_verbose(
        ds.alpha.values, alpha, decimal=18)
    assert_almost_equal_verbose(
        ds.TMPF.values, temp_real - 273.15, decimal=8)  # 9 on 64-bit
    assert_almost_equal_verbose(
        ds.TMPB.values, temp_real - 273.15, decimal=8)  # 9 on 64-bit
    assert_almost_equal_verbose(
        ds.TMPW.values, temp_real - 273.15, decimal=7)  # 11 on 64-bit

    # WLS
    ds.calibration_double_ended(sections=sections,
                                st_label='st',
                                ast_label='ast',
                                rst_label='rst',
                                rast_label='rast',
                                st_var=1e-7,
                                ast_var=1e-7,
                                rst_var=1e-7,
                                rast_var=1e-7,
                                method='wls',
                                solver='sparse',
                                tmpw_mc_size=5,
                                fix_alpha=(alpha, np.zeros_like(alpha)))

    assert_almost_equal_verbose(
        ds.gamma.values, gamma, decimal=8)
    assert_almost_equal_verbose(
        ds.alpha.values, alpha, decimal=18)
    assert_almost_equal_verbose(
        ds.TMPF.values, temp_real - 273.15, decimal=7)
    assert_almost_equal_verbose(
        ds.TMPB.values, temp_real - 273.15, decimal=7)
    assert_almost_equal_verbose(
        ds.TMPW.values, temp_real - 273.15, decimal=7)

    pass


def test_double_ended_ols_wls_fix_alpha_fix_gamma_estimate_synthetic():
    """Checks whether the coefficients are correctly defined by creating a
    synthetic measurement set, and derive the parameters from this set.
    Without variance.
    They should be the same as the parameters used to create the synthetic
    measurment set"""
    from dtscalibration import DataStore
    import numpy as np

    np.random.seed(0)

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

    st = C_p * np.exp(-(dalpha_r + dalpha_p) * x[:, None]) * \
        np.exp(gamma / temp_real) / (np.exp(gamma / temp_real) - 1)
    ast = C_m * np.exp(-(dalpha_r + dalpha_m) * x[:, None]) / \
        (np.exp(gamma / temp_real) - 1)
    rst = C_p * np.exp(-(dalpha_r + dalpha_p) * (-x[:, None] + cable_len)) * \
        np.exp(gamma / temp_real) / (np.exp(gamma / temp_real) - 1)
    rast = C_m * np.exp(-(dalpha_r + dalpha_m) * (-x[:, None] + cable_len)) / \
        (np.exp(gamma / temp_real) - 1)

    alpha = np.mean(np.log(rst / rast) - np.log(st / ast), axis=1) / 2
    alpha -= alpha[0]  # the first x-index is where to start counting

    ds = DataStore({
        'st':                    (['x', 'time'], st),
        'ast':                   (['x', 'time'], ast),
        'rst':                   (['x', 'time'], rst),
        'rast':                  (['x', 'time'], rast),
        'userAcquisitionTimeFW': (['time'], np.ones(nt)),
        'userAcquisitionTimeBW': (['time'], np.ones(nt)),
        'cold':                  (['time'], ts_cold),
        'warm':                  (['time'], ts_warm)
        },
        coords={
            'x':    x,
            'time': time},
        attrs={
            'isDoubleEnded': '1'})

    sections = {
        'cold': [slice(0., 0.5 * cable_len)],
        'warm': [slice(0.5 * cable_len, cable_len)]}

    # OLS
    ds.calibration_double_ended(sections=sections,
                                st_label='st',
                                ast_label='ast',
                                rst_label='rst',
                                rast_label='rast',
                                method='ols',
                                solver='sparse',
                                fix_gamma=(gamma, 0.),
                                fix_alpha=(alpha, np.zeros_like(alpha)))

    assert_almost_equal_verbose(
        ds.gamma.values, gamma, decimal=18)
    assert_almost_equal_verbose(
        ds.alpha.values, alpha, decimal=18)
    assert_almost_equal_verbose(
        ds.TMPF.values, temp_real - 273.15, decimal=9)
    assert_almost_equal_verbose(
        ds.TMPB.values, temp_real - 273.15, decimal=9)
    assert_almost_equal_verbose(
        ds.TMPW.values, temp_real - 273.15, decimal=9)

    # WLS
    ds.calibration_double_ended(sections=sections,
                                st_label='st',
                                ast_label='ast',
                                rst_label='rst',
                                rast_label='rast',
                                st_var=1e-7,
                                ast_var=1e-7,
                                rst_var=1e-7,
                                rast_var=1e-7,
                                method='wls',
                                solver='sparse',
                                tmpw_mc_size=5,
                                fix_gamma=(gamma, 0.),
                                fix_alpha=(alpha, np.zeros_like(alpha)))

    assert_almost_equal_verbose(
        ds.gamma.values, gamma, decimal=18)
    assert_almost_equal_verbose(
        ds.alpha.values, alpha, decimal=18)
    assert_almost_equal_verbose(
        ds.TMPF.values, temp_real - 273.15, decimal=11)
    assert_almost_equal_verbose(
        ds.TMPB.values, temp_real - 273.15, decimal=11)
    assert_almost_equal_verbose(
        ds.TMPW.values, temp_real - 273.15, decimal=11)

    pass


@pytest.mark.skip(reason="Superseeded by "
                         "test_estimate_variance_of_temperature_estimate")
def test_double_ended_exponential_variance_estimate_synthetic():
    import dask.array as da
    from dtscalibration import DataStore
    import numpy as np
    from scipy import stats

    np.random.seed(0)
    state = da.random.RandomState(0)

    stokes_m_var = 4.
    cable_len = 100.
    nt = 5
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

    st = C_p * np.exp(-(dalpha_r + dalpha_p) * x[:, None]) * \
        np.exp(gamma / temp_real) / (np.exp(gamma / temp_real) - 1)
    ast = C_m * np.exp(-(dalpha_r + dalpha_m) * x[:, None]) / \
        (np.exp(gamma / temp_real) - 1)
    rst = C_p * np.exp(-(dalpha_r + dalpha_p) * (-x[:, None] + cable_len)) * \
        np.exp(gamma / temp_real) / (np.exp(gamma / temp_real) - 1)
    rast = C_m * np.exp(-(dalpha_r + dalpha_m) * (-x[:, None] + cable_len)) / \
        (np.exp(gamma / temp_real) - 1)

    mst_var = 1. * stokes_m_var
    mast_var = 1.5 * stokes_m_var
    mrst_var = 0.8 * stokes_m_var
    mrast_var = 0.5 * stokes_m_var

    st_m = st + stats.norm.rvs(size=st.shape, scale=mst_var ** 0.5)
    ast_m = ast + stats.norm.rvs(size=ast.shape, scale=mast_var ** 0.5)
    rst_m = rst + stats.norm.rvs(size=rst.shape, scale=mrst_var ** 0.5)
    rast_m = rast + stats.norm.rvs(size=rast.shape, scale=mrast_var ** 0.5)

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
            'isDoubleEnded': '1'})

    sections = {
        'cold': [slice(0., 0.5 * cable_len)],
        'warm': [slice(0.5 * cable_len, cable_len)]}

    st_label = 'mst'
    ast_label = 'mast'
    rst_label = 'mrst'
    rast_label = 'mrast'

    # MC variance
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
                                solver='sparse')

    ds.conf_int_double_ended(
        p_val='p_val',
        p_cov='p_cov',
        st_label=st_label,
        ast_label=ast_label,
        rst_label=rst_label,
        rast_label=rast_label,
        st_var=mst_var,
        ast_var=mast_var,
        rst_var=mrst_var,
        rast_var=mrast_var,
        store_tmpf='TMPF',
        store_tmpb='TMPB',
        store_tmpw='TMPW',
        store_tempvar='_var',
        conf_ints=[2.5, 50., 97.5],
        mc_sample_size=100,
        ci_avg_time_flag=False,
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
            assert_almost_equal_verbose(v1i ** 2, v2i, decimal=1)

    for (_, v1), (_, v2) in zip(stdsb1.items(), stdsb2.items()):
        for v1i, v2i in zip(v1, v2):
            print('Real VAR: ', v1i ** 2, 'Estimated VAR: ', v2i)
            assert_almost_equal_verbose(v1i ** 2, v2i, decimal=1)

    pass


def test_estimate_variance_of_temperature_estimate():
    import dask.array as da
    from dtscalibration import DataStore
    import numpy as np
    from scipy import stats

    np.random.seed(0)
    state = da.random.RandomState(0)

    stokes_m_var = 0.1
    cable_len = 10.
    nt = 150
    time = np.arange(nt)
    nmc = 201
    x = np.linspace(0., cable_len, 64)
    ts_cold = np.ones(nt) * 4.
    ts_warm = np.ones(nt) * 20.

    C_p = 1524.
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
    # alpha_int = cable_len * (dalpha_p - dalpha_m)
    # alpha = x * (dalpha_p - dalpha_m)

    st = C_p * np.exp(-(dalpha_r + dalpha_p) * x[:, None]) * \
        np.exp(gamma / temp_real) / (np.exp(gamma / temp_real) - 1)
    ast = C_m * np.exp(-(dalpha_r + dalpha_m) * x[:, None]) / \
        (np.exp(gamma / temp_real) - 1)
    rst = C_p * np.exp(-(dalpha_r + dalpha_p) * (-x[:, None] + cable_len)) * \
        np.exp(gamma / temp_real) / (np.exp(gamma / temp_real) - 1)
    rast = C_m * np.exp(-(dalpha_r + dalpha_m) * (-x[:, None] + cable_len)) / \
        (np.exp(gamma / temp_real) - 1)

    mst_var = 1. * stokes_m_var
    mast_var = 1.5 * stokes_m_var
    mrst_var = 0.8 * stokes_m_var
    mrast_var = 0.5 * stokes_m_var

    st_m = st + stats.norm.rvs(size=st.shape, scale=mst_var ** 0.5)
    ast_m = ast + stats.norm.rvs(size=ast.shape, scale=mast_var ** 0.5)
    rst_m = rst + stats.norm.rvs(size=rst.shape, scale=mrst_var ** 0.5)
    rast_m = rast + stats.norm.rvs(size=rast.shape, scale=mrast_var ** 0.5)

    print('alphaint', cable_len * (dalpha_p - dalpha_m))
    print('alpha', dalpha_p - dalpha_m)
    print('C', np.log(C_p / C_m))
    print('x0', x.max())

    ds = DataStore({
        'st':                    (['x', 'time'], st),
        'ast':                   (['x', 'time'], ast),
        'rst':                   (['x', 'time'], rst),
        'rast':                  (['x', 'time'], rast),
        'mst':                   (['x', 'time'], st_m),
        'mast':                  (['x', 'time'], ast_m),
        'mrst':                  (['x', 'time'], rst_m),
        'mrast':                 (['x', 'time'], rast_m),
        'userAcquisitionTimeFW': (['time'], np.ones(nt)),
        'userAcquisitionTimeBW': (['time'], np.ones(nt)),
        'cold':                  (['time'], ts_cold),
        'warm':                  (['time'], ts_warm)
        },
        coords={
            'x':    x,
            'time': time},
        attrs={
            'isDoubleEnded': '1'})

    sections = {
        'cold': [slice(0., 0.25 * cable_len)],
        'warm': [slice(0.5 * cable_len, 0.75 * cable_len)]}

    st_label = 'mst'
    ast_label = 'mast'
    rst_label = 'mrst'
    rast_label = 'mrast'

    # MC variance
    ds.calibration_double_ended(sections=sections,
                                st_label=st_label,
                                ast_label=ast_label,
                                rst_label=rst_label,
                                rast_label=rast_label,
                                st_var=mst_var,
                                ast_var=mast_var,
                                rst_var=mrst_var,
                                rast_var=mrast_var,
                                # fix_gamma=(gamma, 0.),
                                # fix_alpha=(alpha, 0. * alpha),
                                method='wls',
                                solver='stats',
                                tmpw_mc_size=nmc)

    ds.conf_int_double_ended(
        p_val='p_val',
        p_cov='p_cov',
        st_label=st_label,
        ast_label=ast_label,
        rst_label=rst_label,
        rast_label=rast_label,
        st_var=mst_var,
        ast_var=mast_var,
        rst_var=mrst_var,
        rast_var=mrast_var,
        store_tmpf='TMPF',
        store_tmpb='TMPB',
        store_tmpw='TMPW',
        store_tempvar='_var',
        conf_ints=[20., 80.],
        mc_sample_size=nmc,
        ci_avg_time_flag=True,
        da_random_state=state,
        remove_mc_set_flag=False,
        reduce_memory_usage=1)

    assert_almost_equal_verbose(
        (ds.r_st - ds[st_label]).var(dim=['MC', 'time']), mst_var, decimal=2)
    assert_almost_equal_verbose(
        (ds.r_ast - ds[ast_label]).var(dim=['MC', 'time']), mast_var, decimal=2)
    assert_almost_equal_verbose(
        (ds.r_rst - ds[rst_label]).var(dim=['MC', 'time']), mrst_var, decimal=2)
    assert_almost_equal_verbose(
        (ds.r_rast - ds[rast_label]).var(dim=['MC', 'time']), mrast_var,
        decimal=3)

    assert_almost_equal_verbose(
        ds.gamma_MC.var(dim='MC'), 0., decimal=2)
    assert_almost_equal_verbose(
        ds.alpha_MC.var(dim='MC'), 0., decimal=8)
    assert_almost_equal_verbose(
        ds.df_MC.var(dim='MC'), ds.df_var, decimal=7)
    assert_almost_equal_verbose(
        ds.db_MC.var(dim='MC'), ds.db_var, decimal=8)

    # TMPF
    temp_real2 = temp_real[:, 0] - 273.15
    actual = (np.square(ds.TMPF - temp_real2[:, None]).sum(dim='time') /
              ds.time.size)
    desire = ds.TMPF_MC_var.values

    # Validate on sections that were not used for calibration.
    assert_almost_equal_verbose(actual[16:32].mean(), desire[16:32].mean(),
                                decimal=3)
    assert_almost_equal_verbose(actual[48:].mean(), desire[48:].mean(),
                                decimal=3)

    # TMPB
    actual = (np.square(ds.TMPB - temp_real2[:, None]).sum(dim='time') /
              ds.time.size)
    desire = ds.TMPB_MC_var.values

    # Validate on sections that were not used for calibration.
    assert_almost_equal_verbose(actual[16:32].mean(), desire[16:32].mean(),
                                decimal=4)
    assert_almost_equal_verbose(actual[48:].mean(), desire[48:].mean(),
                                decimal=4)
    pass


def test_single_ended_ols_wls_estimate_synthetic():
    """Checks whether the coefficients are correctly defined by creating a
    synthetic measurement set, and derive the parameters from this set.
    Without variance.
    They should be the same as the parameters used to create the synthetic
    measurment set"""

    from dtscalibration import DataStore
    import numpy as np

    np.random.seed(0)

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

    st = C_p * np.exp(-dalpha_r * x[:, None]) * \
        np.exp(-dalpha_p * x[:, None]) * \
        np.exp(gamma / temp_real) / (np.exp(gamma / temp_real) - 1)
    ast = C_m * np.exp(-dalpha_r * x[:, None]) * \
        np.exp(-dalpha_m * x[:, None]) / (np.exp(gamma / temp_real) - 1)

    print('alphaint', cable_len * (dalpha_p - dalpha_m))
    print('alpha', dalpha_p - dalpha_m)
    print('C', np.log(C_p / C_m))
    print('x0', x.max())

    ds = DataStore({
        'st':    (['x', 'time'], st),
        'ast':   (['x', 'time'], ast),
        'userAcquisitionTimeFW': (['time'], np.ones(nt)),
        'cold':  (['time'], ts_cold),
        'warm':  (['time'], ts_warm)
        },
        coords={
            'x':    x,
            'time': time},
        attrs={
            'isDoubleEnded': '0'})

    sections = {
        'cold': [slice(0., 0.5 * cable_len)],
        'warm': [slice(0.5 * cable_len, cable_len)]}

    # OLS
    ds.calibration_single_ended(sections=sections,
                                st_label='st',
                                ast_label='ast',
                                method='ols',
                                solver='sparse')

    assert_almost_equal_verbose(
        ds.gamma.values, gamma, decimal=6)
    assert_almost_equal_verbose(
        ds.dalpha.values, dalpha_p - dalpha_m, decimal=8)
    assert_almost_equal_verbose(
        ds.TMPF.values, temp_real - 273.15, decimal=4)

    # WLS
    ds.calibration_single_ended(sections=sections,
                                st_label='st',
                                ast_label='ast',
                                st_var=1.,
                                ast_var=1.,
                                method='wls',
                                solver='sparse')

    assert_almost_equal_verbose(
        ds.gamma.values, gamma, decimal=6)
    assert_almost_equal_verbose(
        ds.dalpha.values, dalpha_p - dalpha_m, decimal=8)
    assert_almost_equal_verbose(
        ds.TMPF.values, temp_real - 273.15, decimal=4)

    pass


def test_single_ended_ols_wls_fix_dalpha_synthetic():
    """Checks whether the coefficients are correctly defined by creating a
    synthetic measurement set, and derive the parameters from this set.
    Without variance.
    They should be the same as the parameters used to create the synthetic
    measurment set"""

    from dtscalibration import DataStore
    import numpy as np

    np.random.seed(0)

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

    st = C_p * np.exp(-dalpha_r * x[:, None]) * \
        np.exp(-dalpha_p * x[:, None]) * \
        np.exp(gamma / temp_real) / (np.exp(gamma / temp_real) - 1)
    ast = C_m * np.exp(-dalpha_r * x[:, None]) * \
        np.exp(-dalpha_m * x[:, None]) / (np.exp(gamma / temp_real) - 1)

    print('alphaint', cable_len * (dalpha_p - dalpha_m))
    print('alpha', dalpha_p - dalpha_m)
    print('C', np.log(C_p / C_m))
    print('x0', x.max())

    ds = DataStore({
        'st':    (['x', 'time'], st),
        'ast':   (['x', 'time'], ast),
        'userAcquisitionTimeFW': (['time'], np.ones(nt)),
        'cold':  (['time'], ts_cold),
        'warm':  (['time'], ts_warm)
        },
        coords={
            'x':    x,
            'time': time},
        attrs={
            'isDoubleEnded': '0'})

    sections = {
        'cold': [slice(0., 0.5 * cable_len)],
        'warm': [slice(0.5 * cable_len, cable_len)]}

    # OLS
    ds.calibration_single_ended(sections=sections,
                                st_label='st',
                                ast_label='ast',
                                method='ols',
                                solver='sparse',
                                fix_dalpha=(dalpha_p - dalpha_m, 0.))

    assert_almost_equal_verbose(
        ds.gamma.values, gamma, decimal=11)
    assert_almost_equal_verbose(
        ds.dalpha.values, dalpha_p - dalpha_m, decimal=18)
    assert_almost_equal_verbose(
        ds.TMPF.values, temp_real - 273.15, decimal=12)

    # WLS
    ds.calibration_single_ended(sections=sections,
                                st_label='st',
                                ast_label='ast',
                                st_var=1.,
                                ast_var=1.,
                                method='wls',
                                solver='sparse',
                                fix_dalpha=(dalpha_p - dalpha_m, 0.))

    assert_almost_equal_verbose(
        ds.gamma.values, gamma, decimal=12)
    assert_almost_equal_verbose(
        ds.dalpha.values, dalpha_p - dalpha_m, decimal=18)
    assert_almost_equal_verbose(
        ds.TMPF.values, temp_real - 273.15, decimal=12)

    pass


def test_single_ended_ols_wls_fix_gamma_synthetic():
    """Checks whether the coefficients are correctly defined by creating a
    synthetic measurement set, and derive the parameters from this set.
    Without variance.
    They should be the same as the parameters used to create the synthetic
    measurment set"""

    from dtscalibration import DataStore
    import numpy as np

    np.random.seed(0)

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

    st = C_p * np.exp(-dalpha_r * x[:, None]) * \
        np.exp(-dalpha_p * x[:, None]) * \
        np.exp(gamma / temp_real) / (np.exp(gamma / temp_real) - 1)
    ast = C_m * np.exp(-dalpha_r * x[:, None]) * \
        np.exp(-dalpha_m * x[:, None]) / (np.exp(gamma / temp_real) - 1)

    print('alphaint', cable_len * (dalpha_p - dalpha_m))
    print('alpha', dalpha_p - dalpha_m)
    print('C', np.log(C_p / C_m))
    print('x0', x.max())

    ds = DataStore({
        'st':    (['x', 'time'], st),
        'ast':   (['x', 'time'], ast),
        'userAcquisitionTimeFW': (['time'], np.ones(nt)),
        'cold':  (['time'], ts_cold),
        'warm':  (['time'], ts_warm)
        },
        coords={
            'x':    x,
            'time': time},
        attrs={
            'isDoubleEnded': '0'})

    sections = {
        'cold': [slice(0., 0.5 * cable_len)],
        'warm': [slice(0.5 * cable_len, cable_len)]}

    # OLS
    ds.calibration_single_ended(sections=sections,
                                st_label='st',
                                ast_label='ast',
                                method='ols',
                                solver='sparse',
                                fix_gamma=(gamma, 0.))

    assert_almost_equal_verbose(
        ds.gamma.values, gamma, decimal=18)
    assert_almost_equal_verbose(
        ds.dalpha.values, dalpha_p - dalpha_m, decimal=8)
    assert_almost_equal_verbose(
        ds.TMPF.values, temp_real - 273.15, decimal=4)

    # WLS
    ds.calibration_single_ended(sections=sections,
                                st_label='st',
                                ast_label='ast',
                                st_var=1.,
                                ast_var=1.,
                                method='wls',
                                solver='sparse',
                                fix_gamma=(gamma, 0.))

    assert_almost_equal_verbose(
        ds.gamma.values, gamma, decimal=18)
    assert_almost_equal_verbose(
        ds.dalpha.values, dalpha_p - dalpha_m, decimal=8)
    assert_almost_equal_verbose(
        ds.TMPF.values, temp_real - 273.15, decimal=4)

    pass


def test_single_ended_ols_wls_fix_gamma_fix_dalpha_synthetic():
    """Checks whether the coefficients are correctly defined by creating a
    synthetic measurement set, and derive the parameters from this set.
    Without variance.
    They should be the same as the parameters used to create the synthetic
    measurment set"""

    from dtscalibration import DataStore
    import numpy as np

    np.random.seed(0)

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

    st = C_p * np.exp(-dalpha_r * x[:, None]) * \
        np.exp(-dalpha_p * x[:, None]) * \
        np.exp(gamma / temp_real) / (np.exp(gamma / temp_real) - 1)
    ast = C_m * np.exp(-dalpha_r * x[:, None]) * \
        np.exp(-dalpha_m * x[:, None]) / (np.exp(gamma / temp_real) - 1)

    print('alphaint', cable_len * (dalpha_p - dalpha_m))
    print('alpha', dalpha_p - dalpha_m)
    print('C', np.log(C_p / C_m))
    print('x0', x.max())

    ds = DataStore({
        'st':    (['x', 'time'], st),
        'ast':   (['x', 'time'], ast),
        'userAcquisitionTimeFW': (['time'], np.ones(nt)),
        'cold':  (['time'], ts_cold),
        'warm':  (['time'], ts_warm)
        },
        coords={
            'x':    x,
            'time': time},
        attrs={
            'isDoubleEnded': '0'})

    sections = {
        'cold': [slice(0., 0.5 * cable_len)],
        'warm': [slice(0.5 * cable_len, cable_len)]}

    # OLS
    ds.calibration_single_ended(sections=sections,
                                st_label='st',
                                ast_label='ast',
                                method='ols',
                                solver='sparse',
                                fix_gamma=(gamma, 0.),
                                fix_dalpha=(dalpha_p - dalpha_m, 0.))

    assert_almost_equal_verbose(
        ds.gamma.values, gamma, decimal=18)
    assert_almost_equal_verbose(
        ds.dalpha.values, dalpha_p - dalpha_m, decimal=18)
    assert_almost_equal_verbose(
        ds.TMPF.values, temp_real - 273.15, decimal=10)  # 11 on 64-bit

    # WLS
    ds.calibration_single_ended(sections=sections,
                                st_label='st',
                                ast_label='ast',
                                st_var=1.,
                                ast_var=1.,
                                method='wls',
                                solver='sparse',
                                fix_gamma=(gamma, 0.),
                                fix_dalpha=(dalpha_p - dalpha_m, 0.))

    assert_almost_equal_verbose(
        ds.gamma.values, gamma, decimal=18)
    assert_almost_equal_verbose(
        ds.dalpha.values, dalpha_p - dalpha_m, decimal=18)
    assert_almost_equal_verbose(
        ds.TMPF.values, temp_real - 273.15, decimal=11)

    pass


def test_single_ended_exponential_variance_estimate_synthetic():
    """Checks whether the coefficients are correctly defined by creating a
    synthetic measurement set, and derive the parameters from this set.
    With variance.
    They should be the same as the parameters used to create the synthetic
    measurment set"""
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

    st = C_p * np.exp(-dalpha_r * x[:, None]) * \
        np.exp(-dalpha_p * x[:, None]) * \
        np.exp(gamma / temp_real) / (np.exp(gamma / temp_real) - 1)
    ast = C_m * np.exp(-dalpha_r * x[:, None]) * \
        np.exp(-dalpha_m * x[:, None]) / (np.exp(gamma / temp_real) - 1)
    st_m = st + stats.norm.rvs(size=st.shape, scale=stokes_m_var ** 0.5)
    ast_m = ast + stats.norm.rvs(size=ast.shape, scale=astokes_m_var ** 0.5)

    # print('alphaint', cable_len * (dalpha_p - dalpha_m))
    # print('alpha', dalpha_p - dalpha_m)
    # print('C', np.log(C_p / C_m))
    # print('x0', x.max())

    ds = DataStore({
        'st':                    (['x', 'time'], st),
        'ast':                   (['x', 'time'], ast),
        'mst':                   (['x', 'time'], st_m),
        'mast':                  (['x', 'time'], ast_m),
        'userAcquisitionTimeFW': (['time'], np.ones(nt)),
        'cold':                  (['time'], ts_cold),
        'warm':                  (['time'], ts_warm)
        },
        coords={
            'x':    x,
            'time': time},
        attrs={
            'isDoubleEnded': '0'})

    sections = {
        'cold': [slice(0., 0.5 * cable_len)],
        'warm': [slice(0.5 * cable_len, cable_len)]}

    st_label = 'mst'
    ast_label = 'mast'

    mst_var, _ = ds.variance_stokes_exponential(
        st_label=st_label, sections=sections)
    mast_var, _ = ds.variance_stokes_exponential(
        st_label=ast_label, sections=sections)

    # MC variqnce
    ds.calibration_single_ended(sections=sections,
                                st_label=st_label,
                                ast_label=ast_label,
                                st_var=mst_var,
                                ast_var=mast_var,
                                method='wls',
                                solver='sparse')

    ds.conf_int_single_ended(
        p_val='p_val',
        p_cov='p_cov',
        st_label=st_label,
        ast_label=ast_label,
        st_var=mst_var,
        ast_var=mast_var,
        store_tmpf='TMPF',
        store_tempvar='_var',
        conf_ints=[2.5, 50., 97.5],
        mc_sample_size=50,
        ci_avg_time_flag=False,
        da_random_state=state
        )

    # Calibrated variance
    stdsf1 = ds.ufunc_per_section(label='TMPF',
                                  func=np.var,
                                  temp_err=True,
                                  calc_per='stretch',
                                  ddof=1)

    # Use a single timestep to better check if the parameter uncertainties
    # propagate
    ds1 = ds.isel(time=1)
    # Estimated VAR
    stdsf2 = ds1.ufunc_per_section(label='TMPF_MC_var',
                                   func=np.mean,
                                   temp_err=False,
                                   calc_per='stretch')

    for (_, v1), (_, v2) in zip(stdsf1.items(), stdsf2.items()):
        for v1i, v2i in zip(v1, v2):
            v2i_c = float(v2i)
            print('Real VAR: ', v1i, 'Estimated VAR: ', v2i_c)
            assert_almost_equal_verbose(v1i, v2i_c, decimal=1)

    pass
    print('hoi')


def test_calibration_ols():
    """Testing ordinary least squares procedure. And compare with device calibrated temperature.
    The measurements were calibrated by the device using only section 8--17.m. Those temperatures
    are compared up to 2 decimals. Silixa only uses a single calibration constant (I think they
    fix gamma), or a different formulation, see Shell primer.
    """
    filepath = data_dir_double_ended2
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf='UTC',
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
                                   store_tmpw='TMPW',
                                   method='ols')

    np.testing.assert_array_almost_equal(ds100['TMPW'].data,
                                         ds100.TMP.data,
                                         decimal=1)

    ds009 = ds100.sel(x=sections_ultima['probe1Temperature'][0])
    np.testing.assert_array_almost_equal(ds009['TMPW'].data,
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

# coding=utf-8
from dtscalibration.cli import main


def test_main():
    assert main([]) == 0


def test_double_ended_variance_estimate_synthetic():
    import dask.array as da
    from dtscalibration import DataStore
    import numpy as np
    from scipy import stats

    np.random.seed(0)
    state = da.random.RandomState(0)
    # from dtscalibration.calibrate_utils import

    stokes_m_var = 40.
    cable_len = 100.
    time = np.arange(500)
    x = np.linspace(0., cable_len, 100)
    ts_cold = np.ones(time.size) * 4.
    ts_warm = np.ones(time.size) * 20.

    C_p = 15246
    C_m = 2400.
    dalpha_r = 0.0005284
    dalpha_m = 0.0004961
    dalpha_p = 0.0005607
    gamma = 482.6
    cold_mask = x < 0.5 * cable_len
    warm_mask = np.invert(cold_mask)  # == False
    temp_real = np.ones((len(x), time.size))
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
        'cold':  (['time'], ts_cold),
        'warm':  (['time'], ts_warm)
        },
        coords={
            'x':    x,
            'time': time})

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

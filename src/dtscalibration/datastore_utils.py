# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np


def check_dims(ds, labels, correct_dims=None):
    """
    Compare the dimensions of different labels (e.g., 'st', 'rst').
    If a calculation is performed and the dimensions do not agree, the answers don't make
    sense and the matrices are broadcasted and the memory usage will explode. If no correct
    dims provided the dimensions of the different are compared.

    Parameters
    ----------
    ds
    labels : iterable
        An iterable with labels
    correct_dims : tuple of str, optional
        The correct dimensions

    Returns
    -------

    """
    if not correct_dims:
        assert len(labels) > 1, 'Define the correct dimensions'

        for li in labels[1:]:
            assert ds[labels[0]].dims == ds[li].dims, li + ' doesnot have the correct dimensions.' \
                                                           ' Should be ' + str(ds[labels[0]].dims)
    else:
        for li in labels:
            assert ds[li].dims == correct_dims, li + ' doesnot have the correct dimensions. ' \
                                                     'Should be ' + str(correct_dims)


def get_netcdf_encoding(ds, zlib=True, complevel=5, **kwargs):
    """Get default netcdf compression parameters. The same for each data variable.

    TODO: Truncate precision to XML precision per data variable


    Parameters
    ----------
    zlib
    complevel
    ds : DataStore

    Returns
    -------

    """
    comp = dict(zlib=zlib, complevel=complevel)
    comp.update(kwargs)
    encoding = {var: comp for var in ds.data_vars}

    return encoding


def check_timestep_allclose(ds, eps=0.01):
    """
    Check if all timesteps are of equal size. For now it is not possible to calibrate over timesteps
    if the acquisition time of timesteps varies, as the Stokes variance would change over time.

    The acquisition time is stored for single ended
    measurements in userAcquisitionTime, for doubleended measurements in userAcquisitionTimeFW
    and userAcquisitionTimeBW
    Parameters
    ----------
    ds : DataStore
    eps : float
        Default accepts 1% of relative variation between min and max acquisition time.

    Returns
    -------

    """
    dim = ds.channel_configuration['chfw']['acquisitiontime_label']
    dt = ds[dim].data
    dtmin = dt.min()
    dtmax = dt.max()
    dtavg = (dtmin + dtmax) / 2
    assert (dtmax - dtmin) / dtavg < eps, 'Acquisition time is Forward channel not equal for all ' \
                                          'time steps'

    if ds.is_double_ended:
        dim = ds.channel_configuration['chbw']['acquisitiontime_label']
        dt = ds[dim].data
        dtmin = dt.min()
        dtmax = dt.max()
        dtavg = (dtmin + dtmax) / 2
        assert (dtmax - dtmin) / dtavg < eps, 'Acquisition time Backward channel is not equal ' \
                                              'for all time steps'


def merge_double_ended(ds_fw, ds_bw, cable_length, plot_result=True):
    """
    Some measurements are not set up on the DTS-device as double-ended
    meausurements. This means that the two channels have to be merged manually.

    This function can merge two single-ended DataStore objects into a single
    double-ended DataStore. There is no interpolation, the two arrays are
    flipped and overlayed, based on the entered cable length. This can
    introduce spatial inaccuracies with extremely long cables.

    Parameters
    ----------
    ds_fw : DataSore object
        DataStore object representing the forward measurement channel
    ds_bw : DataSore object
        DataStore object representing the backward measurement channel
    cable_length : float
        Manually estimated cable length to base alignment on
    plot_result : bool
        Plot the aligned Stokes of the forward and backward channels

    Returns
    -------
    ds : DataStore object
        With the two channels merged
    """
    assert (ds_fw.attrs['isDoubleEnded'] == '0'
            and ds_bw.attrs['isDoubleEnded'] == '0'), \
        "(one of the) input DataStores is already double ended"

    assert (ds_fw.time.size == ds_bw.time.size), \
        "The two input DataStore objects are not of the same size in the " +\
        "time dimension."

    ds = ds_fw.copy()
    ds_bw = ds_bw.copy()

    ds_bw['x'] = cable_length - ds_bw.x.values

    # TODO: check if reindexing matters, and should be used.
    # one way to do it is performed below, but this could create artifacts
    x_resolution = ds.x.values[1] - ds.x.values[0]
    ds_bw = ds_bw.reindex(
        {'x': ds.x}, method='nearest', tolerance=0.99 * x_resolution)

    ds_bw = ds_bw.sortby('x')

    ds['rst'] = (['x', 'time'], ds_bw.st.values)
    ds['rast'] = (['x', 'time'], ds_bw.ast.values)

    ds = ds.dropna(dim='x')

    ds.attrs['isDoubleEnded'] = '1'
    ds['userAcquisitionTimeBW'] = (
        'time', ds_bw['userAcquisitionTimeFW'].values)

    if plot_result:
        _, ax = plt.subplots()
        ds['st'].isel(time=0).plot(ax=ax, label='Stokes forward')
        ds['rst'].isel(time=0).plot(ax=ax, label='Stokes backward')
        ax.legend()

    return ds


# pylint: disable=too-many-locals
def shift_double_ended(ds, i_shift, verbose=True):
    """
    The cable length was initially configured during the DTS measurement. For double ended
    measurements it is important to enter the correct length so that the forward channel and the
    backward channel are aligned.

    This function can be used to shift the backward channel so that the forward channel and the
    backward channel are aligned. The backward channel is shifted per index
    along the x dimension.

    The DataStore object that is returned only has values for the backscatter that are measured
    for both the forward channel and the backward channel in the shifted object

    There is no interpolation, as this would alter the accuracy.


    Parameters
    ----------
    ds : DataSore object
        DataStore object that needs to be shifted
    i_shift : int
        if i_shift < 0, the cable was configured to be too long and there is too much data
        recorded. If i_shift > 0, the cable was configured to be too short and part of the cable is
        not measured
    verbose: bool
        If True, the function will inform the user which variables are
        dropped. If False, the function will silently drop the variables.

    Returns
    -------
    ds2 : DataStore oobject
        With a shifted x-axis
    """
    # pylint: disable=import-outside-toplevel
    from . import DataStore

    assert isinstance(i_shift, (int, np.integer))

    nx = ds.x.size
    nx2 = nx - i_shift
    if i_shift < 0:
        # The cable was configured to be too long.
        # There is too much data recorded.
        st = ds.st.data[:i_shift]
        ast = ds.ast.data[:i_shift]
        rst = ds.rst.data[-i_shift:]
        rast = ds.rast.data[-i_shift:]
        x2 = ds.x.data[:i_shift]
        # TMP2 = ds.tmp.data[:i_shift]

    else:
        # The cable was configured to be too short.
        # Part of the cable is not measured.
        st = ds.st.data[i_shift:]
        ast = ds.ast.data[i_shift:]
        rst = ds.rst.data[:nx2]
        rast = ds.rast.data[:nx2]
        x2 = ds.x.data[i_shift:]
        # TMP2 = ds.tmp.data[i_shift:]

    d2_coords = dict(ds.coords)
    d2_coords['x'] = (('x',), x2, ds.x.attrs)

    d2_data = dict(ds.data_vars)
    for k in ds.data_vars:
        if 'x' in ds[k].dims and k in d2_data:
            del d2_data[k]

    new_data = (('st', st), ('ast', ast), ('rst', rst), ('rast', rast))

    for k, v in new_data:
        d2_data[k] = (ds[k].dims, v, ds[k].attrs)

    not_included = [k for k in ds.data_vars if k not in d2_data]
    if (not_included and verbose):
        print('I dont know what to do with the following data', not_included)

    return DataStore(data_vars=d2_data, coords=d2_coords, attrs=ds.attrs)


# pylint: disable=too-many-locals
def suggest_cable_shift_double_ended(
        ds, irange, plot_result=True, **fig_kwargs):
    """The cable length was initially configured during the DTS measurement.
    For double ended measurements it is important to enter the correct length
    so that the forward channel and the backward channel are aligned.

    This function can be used to find the shift of the backward channel so
    that the forward channel and the backward channel are aligned. The shift
    index refers to the x-dimension.

    The attenuation should be approximately a straight line with jumps at the
    splices. Temperature independent and invariant over time. The following
    objective functions seems to do the job at determining the best shift for
    which the attenuation is most straight.

    Err1 sums the first derivative. Is a proxy for the length of the
    attenuation line. Err2 sums the second derivative. Is a proxy for the
    wiggelyness of the line.

    The top plot shows the origional Stokes and the origional and shifted
    anti-Stokes The bottom plot is generated that shows the two objective
    functions


    Parameters
    ----------
    ds : DataSore object
        DataStore object that needs to be shifted
    irange : array-like
        a numpy array with data of type int. Containing all the shift index
        that are tested.
        Example: np.arange(-250, 200, 1, dtype=int). It shifts the return
        scattering with 250 indices. Calculates err1 and err2. Then shifts
        the return scattering with 249 indices. Calculates err1 and err2. The
        lowest err1 and err2 are suggested as best shift options.
    plot_result : bool
        Plot the summed error as a function of the shift.

    Returns
    -------
    ishift1: int
        Suggested shift based on Err1
    ishift2: int
        Suggested shift based on Err2
    """
    err1 = []
    err2 = []

    for i_shift in irange:
        nx = ds.x.size
        nx2 = nx - i_shift
        if i_shift < 0:
            # The cable was configured to be too long. There is too much data recorded.
            st = ds.st.data[:i_shift]
            ast = ds.ast.data[:i_shift]
            rst = ds.rst.data[-i_shift:]
            rast = ds.rast.data[-i_shift:]
            x2 = ds.x.data[:i_shift]
        else:
            # The cable was configured to be too short. Part of the cable is not measured.
            st = ds.st.data[i_shift:]
            ast = ds.ast.data[i_shift:]
            rst = ds.rst.data[:nx2]
            rast = ds.rast.data[:nx2]
            x2 = ds.x.data[i_shift:]

        i_f = np.log(st / ast)
        i_b = np.log(rst / rast)

        att = (i_b - i_f) / 2  # varianble E in article

        att_dif1 = np.diff(att, n=1, axis=0)
        att_x_dif1 = 0.5 * x2[1:] + 0.5 * x2[:-1]
        err1_mask = np.logical_and(att_x_dif1 > 1., att_x_dif1 < 150.)
        err1.append(np.nansum(np.abs(att_dif1[err1_mask])))

        att_dif2 = np.diff(att, n=2, axis=0)
        att_x_dif2 = x2[1:-1]
        err2_mask = np.logical_and(att_x_dif2 > 1., att_x_dif2 < 150.)
        err2.append(np.nansum(np.abs(att_dif2[err2_mask])))

    ishift1 = irange[np.argmin(err1, axis=0)]
    ishift2 = irange[np.argmin(err2, axis=0)]

    if plot_result:
        if fig_kwargs is None:
            fig_kwargs = {}

        f, (ax0, ax1) = plt.subplots(2, 1, sharex=False, **fig_kwargs)
        f.suptitle(f'best shift is {ishift1} or {ishift2}')

        dt = ds.isel(time=0)
        x = dt.x.data
        y = dt.st.data
        ax0.plot(x, y, label='ST original')
        y = dt.rst.data
        ax0.plot(x, y, label='REV-ST original')

        dtsh1 = shift_double_ended(dt, ishift1)
        dtsh2 = shift_double_ended(dt, ishift2)
        x1 = dtsh1.x.data
        x2 = dtsh2.x.data
        y1 = dtsh1.rst.data
        y2 = dtsh2.rst.data
        ax0.plot(x1, y1, label=f'ST i_shift={ishift1}')
        ax0.plot(x2, y2, label=f'ST i_shift={ishift2}')
        ax0.set_xlabel('x (m)')
        ax0.legend()

        ax2 = ax1.twinx()
        ax1.plot(irange, err1, c='red', label='1 deriv')
        ax2.plot(irange, err2, c='blue', label='2 deriv')
        ax1.axvline(
            ishift1, c='red', linewidth=0.8, label=f'1 deriv. i_shift={ishift1}')
        ax2.axvline(
            ishift2, c='blue', linewidth=0.8, label=f'2 deriv. i_shift={ishift1}')
        ax1.set_xlabel('i_shift')
        ax1.legend(loc=2)  # left axis
        ax2.legend(loc=1)  # right axis

        plt.tight_layout()

    return ishift1, ishift2

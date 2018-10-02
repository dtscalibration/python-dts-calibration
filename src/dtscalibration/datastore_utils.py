# coding=utf-8
import os
from pathlib import Path
from xml.etree import ElementTree

import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xmltodict


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


def metakey(meta, dic_to_parse, prefix, sep):
    for key in dic_to_parse:
        if prefix == "":
            prefix_parse = key.replace('@', '')
        else:
            prefix_parse = sep.join([prefix, key.replace('@', '')])

        if prefix_parse == sep.join(('logData', 'data')):  # u'logData:data':
            continue

        if hasattr(dic_to_parse[key], 'keys'):
            meta.update(metakey(meta, dic_to_parse[key], prefix_parse, sep))

        elif isinstance(dic_to_parse[key], list):
            for ival, val in enumerate(dic_to_parse[key]):
                num_key = prefix_parse + '_' + str(ival)
                meta.update(metakey(meta, val, num_key, sep))
        else:

            meta[prefix_parse] = dic_to_parse[key]

    return meta


# In[]: Define functions
def read_meta(filename, sep):
    with open(filename) as fh:
        doc_ = xmltodict.parse(fh.read())

    if u'wellLogs' in doc_.keys():
        doc = doc_[u'wellLogs'][u'wellLog']
    else:
        doc = doc_[u'logs'][u'log']

    return metakey(dict(), doc, '', sep)


def clear_output_dir(fp, fn='*.nc'):
    for p in Path(fp).glob(fn):
        p.unlink()


def find_dimensions(filename):
    # Returns nitem and nx
    with open(filename) as fh:
        doc_ = xmltodict.parse(fh.read())

    if u'wellLogs' in doc_.keys():
        doc = doc_[u'wellLogs'][u'wellLog']
    else:
        doc = doc_[u'logs'][u'log']

    nitem = len(doc[u'logCurveInfo'])
    nx = len(doc[u'logData']['data'])

    return nitem, nx


# def grab_data(filelist, sep=':'):
#     ns = {
#         's': 'http://www.witsml.org/schemas/1series'}
#
#     # Obtain meta data from the first file
#     meta = read_meta(filelist[0], sep)
#
#     nitem, nx = find_dimensions(filelist[0])
#     ntime = len(filelist)
#
#     double_ended_flag = bool(int(meta['customData:isDoubleEnded']))
#     chFW = int(meta['customData:forwardMeasurementChannel']) - 1  # zero-based
#
#     extra2 = [('log', 'customData', 'acquisitionTime'),
#               ('log', 'customData', 'referenceTemperature'),
#               ('log', 'customData', 'probe1Temperature'),
#               ('log', 'customData', 'probe2Temperature'),
#               ('log', 'customData', 'referenceProbeVoltage'),
#               ('log', 'customData', 'probe1Voltage'),
#               ('log', 'customData', 'probe2Voltage')
#               ]
#
#     if not double_ended_flag:
#         extra2.append(('log', 'customData', 'UserConfiguration',
#                        'ChannelConfiguration', 'AcquisitionConfiguration',
#                        'AcquisitionTime', 'userAcquisitionTime'))
#
#     else:
#         extra2.append(('log', 'customData', 'UserConfiguration',
#                        'ChannelConfiguration', 'AcquisitionConfiguration',
#                        'AcquisitionTime', 'userAcquisitionTimeFW'))
#         extra2.append(('log', 'customData', 'UserConfiguration',
#                        'ChannelConfiguration', 'AcquisitionConfiguration',
#                        'AcquisitionTime', 'userAcquisitionTimeBW'))
#
#     extra = {
#         item[-1]: dict(path=item, array=np.zeros(ntime, dtype=np.float32))
#         for item in extra2
#         }
#     arr_path = 's:' + '/s:'.join(['log', 'logData', 'data'])
#
#     '''allocate and add units'''
#     for key, item in extra.items():
#         item['array'] = np.zeros(ntime, dtype=np.float32)
#
#         if f'customData:{key}:uom' in meta:
#             item['uom'] = meta[f'customData:{key}:uom']
#
#         else:
#             item['uom'] = ''
#
#     # Lookup recorded item names
#     ddict = []
#     for iitem in range(nitem):
#         key = u'logCurveInfo_' + str(iitem) + sep + 'mnemonic'
#         ddict.append((str(meta[key]), np.float32))
#
#     # Allocate data and time arrays
#     ddtype = np.dtype(ddict)
#     array = np.zeros((nx, ntime), dtype=ddtype)
#     timearr = np.zeros(
#         ntime,
#         dtype=[('minDateTimeIndex', '<U29'), ('maxDateTimeIndex', '<U29'),
#                ('filename_tstamp', (np.unicode_, 17))])  # 'S17'
#
#     # print summary
#     print('%s files were found, each representing a single timestep' % ntime)
#     print('%s recorded vars were found: ' % nitem + ', '.join(ddtype.names))
#     print('Recorded at %s points along the cable' % nx)
#
#     # grab data from all *.xml files in filelist
#     for it, file_ in enumerate(filelist):
#         if it % 10 == 0:
#             print('processing file ' + str(it + 1) + ' out of ' + str(ntime))
#
#         with open(file_, 'r') as fh:
#             tree = ElementTree.parse(fh)
#
#             arr_el = tree.findall(arr_path, namespaces=ns)
#             arr_str = [
#                 tuple(arr_eli.text.strip().split(',')) for arr_eli in arr_el
#                 ]
#             assert len(arr_str) == nx, 'This file has a different nr of measurement points: ' + \
#                                        file_
#             array[:, it] = arr_str
#
#             for key, item in extra.items():
#                 if 'ChannelConfiguration' in item['path']:
#                     path1 = 's:' + '/s:'.join(item['path'][:4])
#                     val1 = tree.findall(path1, namespaces=ns)
#                     path2 = 's:' + '/s:'.join(item['path'][4:6])
#                     val2 = val1[chFW].find(path2, namespaces=ns)
#                     item['array'][it] = val2.text
#
#                 else:
#                     path = 's:' + '/s:'.join(item['path'])
#                     val = tree.find(path, namespaces=ns)
#                     item['array'][it] = val.text
#
#             timearr[it]['minDateTimeIndex'] = tree.find(
#                 's:log/s:startDateTimeIndex', namespaces=ns).text
#             timearr[it]['maxDateTimeIndex'] = tree.find(
#                 's:log/s:endDateTimeIndex', namespaces=ns).text
#
#             file_name = os.path.split(file_)[1]
#             timearr[it]['filename_tstamp'] = file_name[10:-4]  # hack
#
#     return array, timearr, meta, extra


def grab_data2(filelist, sep=':'):
    ns = {'s': 'http://www.witsml.org/schemas/1series'}

    # Obtain meta data from the first file
    meta = read_meta(filelist[0], sep)

    nitem, nx = find_dimensions(filelist[0])
    ntime = len(filelist)

    double_ended_flag = bool(int(meta['customData:isDoubleEnded']))
    chFW = int(meta['customData:forwardMeasurementChannel']) - 1  # zero-based

    extra2 = [('log', 'customData', 'acquisitionTime'),
              ('log', 'customData', 'referenceTemperature'),
              ('log', 'customData', 'probe1Temperature'),
              ('log', 'customData', 'probe2Temperature'),
              ('log', 'customData', 'referenceProbeVoltage'),
              ('log', 'customData', 'probe1Voltage'),
              ('log', 'customData', 'probe2Voltage')
              ]

    if not double_ended_flag:
        extra2.append(('log', 'customData', 'UserConfiguration',
                       'ChannelConfiguration', 'AcquisitionConfiguration',
                       'AcquisitionTime', 'userAcquisitionTimeFW'))

    else:
        extra2.append(('log', 'customData', 'UserConfiguration',
                       'ChannelConfiguration', 'AcquisitionConfiguration',
                       'AcquisitionTime', 'userAcquisitionTimeFW'))
        extra2.append(('log', 'customData', 'UserConfiguration',
                       'ChannelConfiguration', 'AcquisitionConfiguration',
                       'AcquisitionTime', 'userAcquisitionTimeBW'))

    extra = {
        item[-1]: dict(path=item, array=np.zeros(ntime, dtype=np.float32))
        for item in extra2
        }
    arr_path = 's:' + '/s:'.join(['log', 'logData', 'data'])

    '''allocate and add units'''
    for key, item in extra.items():
        item['array'] = np.zeros(ntime, dtype=np.float32)

        if f'customData:{key}:uom' in meta:
            item['uom'] = meta[f'customData:{key}:uom']

        else:
            item['uom'] = ''

    # Lookup recorded item names
    ddict = []
    for iitem in range(nitem):
        key = u'logCurveInfo_' + str(iitem) + sep + 'mnemonic'
        ddict.append((str(meta[key]), np.float32))

    # Allocate data and time arrays
    ddtype = np.dtype(ddict)

    # print summary
    print('%s files were found, each representing a single timestep' % ntime)
    print('%s recorded vars were found: ' % nitem + ', '.join(ddtype.names))
    print('Recorded at %s points along the cable' % nx)

    @dask.delayed
    def grab_data_per_file(file_handle, array_path, namespace, data_dtype):
        with open(file_handle, 'r') as f_h:
            eltree = ElementTree.parse(f_h)
            arr_el = eltree.findall(array_path, namespaces=namespace)
            arr_str = [
                tuple(arr_eli.text.strip().split(',')) for arr_eli in arr_el
                ]

        return np.array(arr_str, dtype=data_dtype)

    data_arr = []

    for it, file_ in enumerate(filelist):
        if it % 25 == 0 or it == (ntime - 1):
            print('Dask: Setting up handle for delayed readout. ' +
                  str(it + 1) + ' out of ' + str(ntime))

        data_arr.append(grab_data_per_file(file_, arr_path, ns, ddtype))

    data_arr = [da.from_delayed(x, shape=(nx,), dtype=ddtype) for x in data_arr]
    data_arr = da.stack(data_arr).T

    # read time data and extra data not lazy, as they are needed as coordinates
    time_dtype = [('filename_tstamp', np.int64),
                  ('minDateTimeIndex', '<U29'),
                  ('maxDateTimeIndex', '<U29')]

    extra_dtype = []
    extra_path = []
    for key, item in extra.items():
        extra_path.append(item['path'])
        extra_dtype.append((key, float))

    time_arr = np.zeros((ntime,), dtype=time_dtype)
    extra_arr = np.zeros((ntime,), dtype=extra_dtype)

    for it, file_ in enumerate(filelist):
        if it % 25 == 0 or it == (ntime - 1):
            print('Directly reading time and extra info from xml files. ' +
                  str(it + 1) + ' out of ' + str(ntime))

        with open(file_, 'r') as fh:
            tree = ElementTree.parse(fh)

            for extra_pathi, extra_dtypei in zip(extra_path, extra_dtype):
                if 'ChannelConfiguration' in extra_pathi:
                    path1 = 's:' + '/s:'.join(extra_pathi[:4])
                    val1 = tree.findall(path1, namespaces=ns)
                    path2 = 's:' + '/s:'.join(extra_pathi[4:6])
                    val2 = val1[chFW].find(path2, namespaces=ns)
                    extra_arr[it][extra_dtypei[0]] = val2.text

                else:
                    path = 's:' + '/s:'.join(extra_pathi)
                    val = tree.find(path, namespaces=ns)
                    extra_arr[it][extra_dtypei[0]] = val.text

            startDateTimeIndex = tree.find(
                's:log/s:startDateTimeIndex', namespaces=ns).text
            endDateTimeIndex = tree.find(
                's:log/s:endDateTimeIndex', namespaces=ns).text

            file_name = os.path.split(file_)[1]
            tstamp = np.int64(file_name[10:-4])
            time_arr[it] = (tstamp, startDateTimeIndex, endDateTimeIndex)

    for key, item in extra.items():
        item['array'] = extra_arr[key]

    return data_arr, time_arr, meta, extra


def coords_time(extra, timearr, timezone_netcdf, timezone_ultima_xml):
    time_attrs = {
        'time':        {
            'description': 'time halfway the measurement',
            'timezone':    str(timezone_netcdf)},
        'timestart':   {
            'description': 'time start of the measurement',
            'timezone':    str(timezone_netcdf)},
        'timeend':     {
            'description': 'time end of the measurement',
            'timezone':    str(timezone_netcdf)},
        'timeFW':      {
            'description': 'time halfway the forward channel measurement',
            'timezone':    str(timezone_netcdf)},
        'timeFWstart': {
            'description': 'time start of the forward channel measurement',
            'timezone':    str(timezone_netcdf)},
        'timeFWend':   {
            'description': 'time end of the forward channel measurement',
            'timezone':    str(timezone_netcdf)},
        'timeBW':      {
            'description': 'time halfway the backward channel measurement',
            'timezone':    str(timezone_netcdf)},
        'timeBWstart': {
            'description': 'time start of the backward channel measurement',
            'timezone':    str(timezone_netcdf)},
        'timeBWend':   {
            'description': 'time end of the backward channel measurement',
            'timezone':    str(timezone_netcdf)},
        }

    double_ended_flag = 'userAcquisitionTimeBW' in extra

    maxTimeIndex = pd.DatetimeIndex(timearr['maxDateTimeIndex'])

    if not double_ended_flag:
        # single ended measurement
        dt1 = extra['userAcquisitionTimeFW']['array'].astype('timedelta64[s]')

        # start of the forward measurement
        index_time_FWstart = maxTimeIndex - dt1

        # end of the forward measurement
        index_time_FWend = maxTimeIndex

        # center of forward measurement
        index_time_FWmean = maxTimeIndex - dt1 / 2

        coords_zip = [('timestart', index_time_FWstart),
                      ('timeend', index_time_FWend),
                      ('time', index_time_FWmean)]

    else:
        # double ended measurement
        dt1 = extra['userAcquisitionTimeFW']['array'].astype('timedelta64[s]')
        dt2 = extra['userAcquisitionTimeBW']['array'].astype('timedelta64[s]')

        # start of the forward measurement
        index_time_FWstart = maxTimeIndex - dt1

        # end of the forward measurement
        index_time_FWend = maxTimeIndex

        # center of forward measurement
        index_time_FWmean = maxTimeIndex - dt1 / 2

        # start of the backward measurement
        index_time_BWstart = index_time_FWend.copy()

        # end of the backward measurement
        index_time_BWend = maxTimeIndex + dt2

        # center of backward measurement
        index_time_BWmean = maxTimeIndex + dt2 / 2

        coords_zip = [('timeFWstart', index_time_FWstart),
                      ('timeFWend', index_time_FWend),
                      ('timeFW', index_time_FWmean),
                      ('timeBWstart', index_time_BWstart),
                      ('timeBWend', index_time_BWend),
                      ('timeBW', index_time_BWmean),
                      ('timestart', index_time_FWstart),
                      ('timeend', index_time_BWend),
                      ('time', index_time_FWend)]

    coords = {k: (
        'time',
        pd.DatetimeIndex(v).tz_localize(
            tz=timezone_ultima_xml).tz_convert(
            timezone_netcdf).astype('datetime64[ns]'),
        time_attrs[k]) for k, v in coords_zip
        }

    return coords


def check_dims(ds, labels, correct_dims=None):
    """
    Compare the dimensions of different labels (e.g., 'ST', 'REV-ST').
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

    pass


def get_netcdf_encoding(ds, zlib=True, complevel=5, **kwargs):
    """Get default netcdf compression parameters. The same for each data variable.

    TODO: Truncate precision to XML precision per data variable


    Parameters
    ----------
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
    dim : str
        Label of the acquisition timeseries
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


def shift_double_ended(ds, i_shift):
    """
    The cable length was initially configured during the DTS measurement. For double ended
    measurements it is important to enter the correct length so that the forward channel and the
    backward channel are aligned.

    This function can be used to shift the backward channel so that the forward channel and the
    backward channel are aligned. The backward channel is shifted by index along the x dimension.

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
        not measured.

    Returns
    -------
    ds2 : DataStore oobject
        With a shifted x-axis
    """
    from dtscalibration import DataStore

    assert isinstance(i_shift, (int, np.integer))

    nx = ds.x.size
    nx2 = nx - i_shift
    if i_shift < 0:
        # The cable was configured to be too long. There is too much data recorded.
        st = ds.ST.data[:i_shift]
        ast = ds.AST.data[:i_shift]
        rst = ds['REV-ST'].data[-i_shift:]
        rast = ds['REV-AST'].data[-i_shift:]
        x2 = ds.x.data[:i_shift]
        # TMP2 = ds.TMP.data[:i_shift]

    else:
        # The cable was configured to be too short. Part of the cable is not measured.
        st = ds.ST.data[i_shift:]
        ast = ds.AST.data[i_shift:]
        rst = ds['REV-ST'].data[:nx2]
        rast = ds['REV-AST'].data[:nx2]
        x2 = ds.x.data[i_shift:]
        # TMP2 = ds.TMP.data[i_shift:]

    d2_coords = dict(ds.coords)
    d2_coords['x'] = (('x',), x2, ds.x.attrs)

    d2_data = dict(ds.data_vars)
    for k in ds.data_vars:
        if 'x' in ds[k].dims and k in d2_data:
            del d2_data[k]

    new_data = (('ST', st),
                ('AST', ast),
                ('REV-ST', rst),
                ('REV-AST', rast))

    for k, v in new_data:
        d2_data[k] = (ds[k].dims, v, ds[k].attrs)

    not_included = [k for k in ds.data_vars if k not in d2_data]
    print('I dont know what to do with the following data', not_included)

    return DataStore(data_vars=d2_data, coords=d2_coords, attrs=ds.attrs)


def suggest_cable_shift_double_ended(ds, irange):
    """The cable length was initially configured during the DTS measurement. For double ended
    measurements it is important to enter the correct length so that the forward channel and the
    backward channel are aligned.

    This function can be used to find the shift of the backward channel so that the forward
    channel and the backward channel are aligned. The shift index refers to the x-dimension.

    The attenuation should be approximately a straight line with jumps at the splices.
    Temperature independent and invariant over time. The following objective functions seems to
    do the job at determining the best shift for which the attenuation is most straight.

    Err1 sums the first derivative. Is a proxy for the length of the attenuation line.
    Err2 sums the second derivative. Is a proxy for the wiggelyness of the line.

    The top plot shows the origional Stokes and the origional and shifted anti-Stokes
    The bottom plot is generated that shows the two objective functions


    Parameters
    ----------
    ds : DataSore object
        DataStore object that needs to be shifted
    irange : array-like
        a numpy array with data of type int. Containing all the shift index that are tested.
        Example: np.arange(-250, 200, 1, dtype=int). It shifts the return scattering with
        250 indices. Calculates err1 and err2. Then shifts the return scattering with
        249 indices. Calculates err1 and err2. The lowest err1 and err2 are suggested as best
        shift options.

    Returns
    -------
    ds2 : DataStore oobject
        With a shifted x-axis
    """
    err1 = []
    err2 = []

    for i_shift in irange:
        nx = ds.x.size
        nx2 = nx - i_shift
        if i_shift < 0:
            # The cable was configured to be too long. There is too much data recorded.
            st = ds.ST.data[:i_shift]
            ast = ds.AST.data[:i_shift]
            rst = ds['REV-ST'].data[-i_shift:]
            rast = ds['REV-AST'].data[-i_shift:]
            x2 = ds.x.data[:i_shift]
        else:
            # The cable was configured to be too short. Part of the cable is not measured.
            st = ds.ST.data[i_shift:]
            ast = ds.AST.data[i_shift:]
            rst = ds['REV-ST'].data[:nx2]
            rast = ds['REV-AST'].data[:nx2]
            x2 = ds.x.data[i_shift:]

        i_f = np.log(st / ast)
        i_b = np.log(rst / rast)

        att = (i_b - i_f) / 2

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

    f, (ax0, ax1) = plt.subplots(2, 1, sharex=False)
    f.suptitle('best shift is {} or {}'.format(ishift1, ishift2))

    dt = ds.isel(time=0)
    x = dt.x.data
    y = dt.ST.data
    ax0.plot(x, y, label='ST original')
    y = dt['REV-ST'].data
    ax0.plot(x, y, label='REV-ST original')

    dtsh1 = shift_double_ended(dt, ishift1)
    dtsh2 = shift_double_ended(dt, ishift2)
    x1 = dtsh1.x.data
    x2 = dtsh2.x.data
    y1 = dtsh1['REV-ST'].data
    y2 = dtsh2['REV-ST'].data
    ax0.plot(x1, y1, label='ST i_shift={}'.format(ishift1))
    ax0.plot(x2, y2, label='ST i_shift={}'.format(ishift2))
    ax0.set_xlabel('x (m)')
    ax0.legend()

    ax2 = ax1.twinx()
    ax1.plot(irange, err1, c='red', label='1 deriv')
    ax2.plot(irange, err2, c='blue', label='2 deriv')
    ax1.axvline(ishift1, c='red', linewidth=0.8, label='1 deriv. i_shift={}'.format(ishift1))
    ax2.axvline(ishift2, c='blue', linewidth=0.8, label='2 deriv. i_shift={}'.format(ishift1))
    ax1.set_xlabel('i_shift')
    ax1.legend(loc=2)  # left axis
    ax2.legend(loc=1)  # right axis

    plt.tight_layout()

    return ishift1, ishift2

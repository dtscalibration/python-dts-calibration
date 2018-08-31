# coding=utf-8
import os
from pathlib import Path
from xml.etree import ElementTree

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xmltodict


def read_data_from_fp_numpy(fp):
    """
    Read the data from a single Silixa xml file. Using a simple approach

    :param fp: File path
    :param i_first: Index of the first <data>. Use this if reading multiple files.
    :param i_last:  Index of the last </data>. Use this if reading multiple files.
    :return: The data of the file as numpy array of shape (nx, ncols)

    :note: calculating i_first and i_last is fast compared to the rest
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


def grab_data(filelist, sep=':'):
    ns = {
        's': 'http://www.witsml.org/schemas/1series'}

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
                       'AcquisitionTime', 'userAcquisitionTime'))

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
    array = np.zeros((nx, ntime), dtype=ddtype)
    timearr = np.zeros(
        ntime,
        dtype=[('minDateTimeIndex', '<U29'), ('maxDateTimeIndex', '<U29'),
               ('filename_tstamp', (np.unicode_, 17))])  # 'S17'

    # print summary
    print('%s files were found, each representing a single timestep' % ntime)
    print('%s recorded vars were found: ' % nitem + ', '.join(ddtype.names))
    print('Recorded at %s points along the cable' % nx)

    # grab data from all *.xml files in filelist
    for it, file_ in enumerate(filelist):
        if it % 10 == 0:
            print('processing file ' + str(it + 1) + ' out of ' + str(ntime))

        with open(file_, 'r') as fh:
            tree = ElementTree.parse(fh)

            arr_el = tree.findall(arr_path, namespaces=ns)
            arr_str = [
                tuple(arr_eli.text.strip().split(',')) for arr_eli in arr_el
                ]
            assert len(arr_str) == nx, 'This file has a different nr of measurement points: ' + \
                                       file_
            array[:, it] = arr_str

            for key, item in extra.items():
                if 'ChannelConfiguration' in item['path']:
                    path1 = 's:' + '/s:'.join(item['path'][:4])
                    val1 = tree.findall(path1, namespaces=ns)
                    path2 = 's:' + '/s:'.join(item['path'][4:6])
                    val2 = val1[chFW].find(path2, namespaces=ns)
                    item['array'][it] = val2.text

                else:
                    path = 's:' + '/s:'.join(item['path'])
                    val = tree.find(path, namespaces=ns)
                    item['array'][it] = val.text

            timearr[it]['minDateTimeIndex'] = tree.find(
                's:log/s:startDateTimeIndex', namespaces=ns).text
            timearr[it]['maxDateTimeIndex'] = tree.find(
                's:log/s:endDateTimeIndex', namespaces=ns).text

            file_name = os.path.split(file_)[1]
            timearr[it]['filename_tstamp'] = file_name[10:-4]  # hack

    return array, timearr, meta, extra


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
                       'AcquisitionTime', 'userAcquisitionTime'))

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
    def grab_data_per_file(file_, array_path, ns, data_dtype):
        with open(file_, 'r') as fh:
            tree = ElementTree.parse(fh)
            arr_el = tree.findall(array_path, namespaces=ns)
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
    time_dtype = [('filename_tstamp', int),
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
            tstamp = int(file_name[10:-4])
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
        dt1 = extra['userAcquisitionTime']['array'].astype('timedelta64[s]')

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
    labels : iterable
        An iterable with labels
    correct_dims : tuple of str, optional
        The correct dimensions

    Returns
    -------

    """
    if not correct_dims:
        assert len(labels) > 1

        for li in labels[1:]:
            assert ds[labels[0]].dims == ds[li].dims
    else:
        for li in labels:
            assert ds[li].dims == correct_dims

    pass

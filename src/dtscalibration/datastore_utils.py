# coding=utf-8
import os
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import xmltodict

from dtscalibration.datastore import DataStore


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

    return (len(doc[u'logCurveInfo']), len(doc[u'logData']['data']))


def file_to_array(file, nx, dtype):
    with open(file, 'r') as fh:
        s = fh.readlines()

    i_start = s.index('   <data>\n')
    i_stop = i_start - 1 + 3 * nx

    str_array = s[slice(i_start + 1, i_stop, 3)]

    return np.loadtxt(str_array, delimiter=',', dtype=dtype)


def items_per_chunk(filelisti, maxfilesize=1000):
    from math import ceil
    nitem, nx = find_dimensions(filelisti)
    return int(ceil(maxfilesize * 1e6 / (nitem * nx * 8)))


def split_filelist(filelist, maxfilesize=1000):
    maxitems = items_per_chunk(filelist[0], maxfilesize=maxfilesize)

    return [
        filelist[i:i + maxitems] for i in range(0, len(filelist), maxitems)
        ]


def grab_data(filelist, sep=':'):
    ns = {
        's': 'http://www.witsml.org/schemas/1series'}

    # Obtain meta data from the first file
    meta = read_meta(filelist[0], sep)

    nitem, nx = find_dimensions(filelist[0])
    ntime = len(filelist)
    '''save other data'''
    # extra = {
    #     'acquisitionTime': dict(iloc=(18, 0)),
    #     'referenceTemperature': dict(iloc=(18, 1)),
    #     'probe1Temperature': dict(iloc=(18, 2)),
    #     'probe2Temperature': dict(iloc=(18, 3)),
    #     'referenceProbeVoltage': dict(iloc=(18, 6)),
    #     'probe1Voltage': dict(iloc=(18, 7)),
    #     'probe2Voltage': dict(iloc=(18, 8)),
    #     'userAcquisitionTimeFW': dict(iloc=(-1, -1, 1, 4, 2)),
    #     'userAcquisitionTimeBW': dict(iloc=(-1, -1, 2, 4, 2))
    # }

    double_ended_flag = bool(int(meta['customData:isDoubleEnded']))
    chFW = int(meta['customData:forwardMeasurementChannel']) - 1  # zero-based

    extra2 = [('log', 'customData', 'acquisitionTime'),
              ('log', 'customData',
               'referenceTemperature'), ('log', 'customData',
                                         'probe1Temperature'),
              ('log', 'customData',
               'probe2Temperature'), ('log', 'customData',
                                      'referenceProbeVoltage'),
              ('log', 'customData', 'probe1Voltage'), ('log', 'customData',
                                                       'probe2Voltage'),
              ('log', 'customData', 'UserConfiguration',
               'ChannelConfiguration', 'AcquisitionConfiguration',
               'AcquisitionTime', 'userAcquisitionTimeFW')]
    if double_ended_flag:
        extra2.append(('log', 'customData', 'UserConfiguration',
                       'ChannelConfiguration', 'AcquisitionConfiguration',
                       'AcquisitionTime', 'userAcquisitionTimeBW'))

    # 'userAcquisitionTimeFW': dict(iloc=(0, -1, -1, 1, 4, 2)),
    # 'userAcquisitionTimeBW': dict(iloc=(0, -1, -1, 2, 4, 2))

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
        (ntime),
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


def process_files(filelisti, timezone_ultima_xml, timezone_netCDF):
    """
    In double-ended measurements:
    - 'time' is the timestamp between the forward and backward measurement
    - 'timeFW is the timestamp in the middle of the FW measurement.
        representative for the forward measurement
    - 'timeBW' is the timestamp in the middle of the BW measuremnt.
        representative for the backward measurement


    :param filelisti:
    :param timezone_ultima_xml:
    :param timezone_netCDF:
    :return:
    """
    array, timearr, meta, extra = grab_data(filelisti)

    double_ended_flag = 'REV-ST' in array.dtype.names

    # assert double_ended_flag, 'Only double ended is supported'

    dt1 = pd.to_timedelta(extra['userAcquisitionTimeFW']['array'], 's').values

    index_time_FWstart = pd.to_datetime(
        (pd.DatetimeIndex(timearr['maxDateTimeIndex']) - dt1).tz_localize(
            tz=timezone_ultima_xml).tz_convert(timezone_netCDF))
    index_time_FWend = pd.to_datetime(
        pd.DatetimeIndex(timearr['maxDateTimeIndex']).tz_localize(
            tz=timezone_ultima_xml).tz_convert(timezone_netCDF))
    index_time_FWmean = pd.to_datetime(
        (pd.DatetimeIndex(timearr['maxDateTimeIndex']) - dt1 / 2).tz_localize(
            tz=timezone_ultima_xml).tz_convert(timezone_netCDF))

    if double_ended_flag:
        dt2 = pd.to_timedelta(extra['userAcquisitionTimeBW']['array'],
                              's').values
        # index_time_BWstart = index_time_FWend.copy()
        # index_time_BWend = pd.to_datetime(
        #     (pd.DatetimeIndex(timearr['maxDateTimeIndex']) + dt2).tz_localize(
        #         tz=timezone_ultima_xml).tz_convert(timezone_netCDF))
        index_time_BWmean = pd.to_datetime(
            (pd.DatetimeIndex(timearr['maxDateTimeIndex']) +
             dt2 / 2).tz_localize(
                tz=timezone_ultima_xml).tz_convert(timezone_netCDF))

    dataset_dict = {}
    for name in array.dtype.names:
        if name in ['ST', 'AST']:
            dataset_dict[name] = (['x', 'timeFW'], array[name])

        elif name in ['REV-ST', 'REV-AST']:
            dataset_dict[name] = (['x', 'timeBW'], array[name])

        elif name == 'TMP':
            dataset_dict[name] = (['x', 'time'], array[name])

        elif name == 'LAF':
            continue

        else:
            assert 0

    for key, item in extra.items():
        dataset_dict[key] = (['time'], item['array'])

    coords = {
        'x': array['LAF'][:, 0],
        'time': index_time_FWend.astype('datetime64[ns]'),  # in UTC
        'timeFW': index_time_FWmean.astype('datetime64[ns]'),  # in UTC
        'timestartmeas': index_time_FWstart.astype('datetime64[ns]')  # in UTC
        }
    if double_ended_flag:
        coords['timeBW'] = index_time_BWmean.astype('datetime64[ns]')  # in UTC

    return DataStore(dataset_dict, coords=coords, attrs=meta)  # xr.Dataset

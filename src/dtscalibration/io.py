# coding=utf-8
import os
import struct
from contextlib import contextmanager
from xml.etree import ElementTree

import dask
import dask.array as da
import numpy as np
import pandas as pd

# Returns a dictionary with the attributes to the dimensions.
#  The keys refer to the naming used in the raw files.
_dim_attrs = {
    ('x', 'distance'):
        dict(
            name='distance',
            description='Length along fiber',
            long_description='Starting at connector of forward channel',
            units='m'),
    ('tmp', 'temperature'):
        dict(
            name='tmp',
            description='Temperature calibrated by device',
            units='degC'),
    ('st',): dict(name='st', description='Stokes intensity', units='-'),
    ('ast',): dict(name='ast', description='anti-Stokes intensity', units='-'),
    ('rst',):
        dict(name='rst', description='reverse Stokes intensity', units='-'),
    ('rast',):
        dict(
            name='rast',
            description='reverse anti-Stokes intensity',
            units='-'),
    ('acquisitionTime',):
        dict(
            name='acquisitionTime',
            description='Measurement duration of forward channel',
            long_description='Actual measurement duration of forward '
            'channel',
            units='seconds'),
    ('userAcquisitionTimeFW',):
        dict(
            name='userAcquisitionTimeFW',
            description='Measurement duration of forward channel',
            long_description='Desired measurement duration of forward '
            'channel',
            units='seconds'),
    ('userAcquisitionTimeBW',):
        dict(
            name='userAcquisitionTimeBW',
            description='Measurement duration of backward channel',
            long_description='Desired measurement duration of backward '
            'channel',
            units='seconds'),
    ('trans_att',):
        dict(
            name='Locations introducing transient directional differential '
                 'attenuation',
            description='Locations along the x-dimension introducing transient '
                        'directional differential attenuation',
            long_description='Connectors introduce additional differential '
                             'attenuation that is different for the forward '
                             'and backward direction, and varies over time.',
            units='m')}

# Because variations in the names exist between the different file formats. The
#   tuple as key contains the possible keys, which is expanded below.
dim_attrs = {k: v for kl, v in _dim_attrs.items() for k in kl}

dim_attrs_apsensing = dict(dim_attrs)
dim_attrs_apsensing['TEMP'] = dim_attrs_apsensing.pop('tmp')
dim_attrs_apsensing['TEMP']['name'] = 'TEMP'
dim_attrs_apsensing.pop('acquisitionTime')
dim_attrs_apsensing.pop('userAcquisitionTimeFW')
dim_attrs_apsensing.pop('userAcquisitionTimeBW')


@contextmanager
def open_file(path, **kwargs):
    if isinstance(path, tuple):
        # print('\nabout to open zipfile', path[0], '. from', path[1])
        # assert isinstance(path[1], zip)
        the_file = path[1].open(path[0], **kwargs)

    else:
        the_file = open(path, **kwargs)

    yield the_file
    the_file.close()


def silixa_xml_version_check(filepathlist):
    """Function which tests which version of xml files have to be read.

    Parameters
    ----------
    filepathlist

    Returns
    -------

    """

    sep = ':'
    attrs = read_silixa_attrs_singlefile(filepathlist[0], sep)

    version_string = attrs['customData:SystemSettings:softwareVersion']

    # Get major version from string. Tested for Ultima v4, v6, v7 XT-DTS v6
    major_version = int(version_string.replace(' ', '').split(':')[-1][0])

    return major_version


def apsensing_xml_version_check(filepathlist):
    """Function which tests which version of xml files are read.

    Parameters
    ----------
    filepathlist

    Returns
    -------

    """

    sep = ':'
    attrs, _ = read_apsensing_attrs_singlefile(filepathlist[0], sep)

    return attrs['wellbore:uid']


def sensornet_ddf_version_check(filepathlist):
    """Function which checks and returns the .ddf file version

    Parameters
    ----------
    filepathlist

    Returns
    -------
    ddf_version

    """

    # Obtain metadata fro mthe first file
    _, meta = read_sensornet_single(filepathlist[0])

    if 'Software version number' in meta.keys():
        version_string = meta['Software version number']
    else:
        raise ValueError(
            'Software version number could not be detected in .ddf file'
            + 'Either file is corrupted or not supported')

    ddf_version = version_string.replace(',', '.')

    return ddf_version


def sensortran_binary_version_check(filepathlist):
    """Function which tests which version the sensortran binaries are.

    Parameters
    ----------
    filepathlist

    Returns
    -------

    """
    fname = filepathlist[0]

    with open(fname, 'rb') as f:
        f.read(2)
        version = struct.unpack('<h', f.read(2))[0]

    return version


# pylint: disable=too-many-locals
def read_silixa_files_routine_v6(  # noqa: MC0001
        filepathlist,
        xml_version=6,
        timezone_netcdf='UTC',
        silent=False,
        load_in_memory='auto'):
    """
    Internal routine that reads Silixa files.
    Use dtscalibration.read_silixa_files function instead.

    The silixa files are already timezone aware

    Parameters
    ----------
    load_in_memory
    filepathlist
    timezone_netcdf
    silent

    Returns
    -------

    """

    # translate names
    tld = {
        'ST': 'st',
        'AST': 'ast',
        'REV-ST': 'rst',
        'REV-AST': 'rast',
        'TMP': 'tmp'}

    # Open the first xml file using ET, get the name space and amount of data

    with open_file(filepathlist[0]) as fh:
        xml_tree = ElementTree.parse(fh)

    namespace = get_xml_namespace(xml_tree.getroot())

    logtree = xml_tree.find(f'./{namespace}log')
    logdata_tree = logtree.find(f'./{namespace}logData')

    # Amount of datapoints is the size of the logdata tree, corrected for
    #  the mnemonic list and unit list
    nx = len(logdata_tree) - 2

    sep = ':'
    ns = {'s': namespace[1:-1]}

    # Obtain metadata from the first file
    attrs = read_silixa_attrs_singlefile(filepathlist[0], sep)

    # Add standardised required attributes
    attrs['isDoubleEnded'] = attrs['customData:isDoubleEnded']
    double_ended_flag = bool(int(attrs['isDoubleEnded']))

    attrs['forwardMeasurementChannel'] = attrs[
        'customData:forwardMeasurementChannel']
    if double_ended_flag:
        attrs['backwardMeasurementChannel'] = attrs[
            'customData:reverseMeasurementChannel']
    else:
        attrs['backwardMeasurementChannel'] = 'N/A'

    # obtain basic data info
    data_item_names = attrs['logData:mnemonicList'].replace(
        " ", "").strip(' ').split(',')
    nitem = len(data_item_names)

    ntime = len(filepathlist)

    double_ended_flag = bool(int(attrs['isDoubleEnded']))
    chFW = int(attrs['forwardMeasurementChannel']) - 1  # zero-based
    if double_ended_flag:
        chBW = int(attrs['backwardMeasurementChannel']) - 1  # zero-based
    else:
        # no backward channel is negative value. writes better to netcdf
        chBW = -1

    # print summary
    if not silent:
        print(f'{ntime} files were found, each representing a single timestep')
        print(f'{nitem} recorded vars were found: ' +
              ', '.join(data_item_names))
        print(f'Recorded at {nx} points along the cable')

        if double_ended_flag:
            print('The measurement is double ended')
        else:
            print('The measurement is single ended')

    # obtain timeseries from data
    timeseries_loc_in_hierarchy = [
        ('log', 'customData', 'acquisitionTime'),
        ('log', 'customData', 'referenceTemperature'),
        ('log', 'customData', 'probe1Temperature'),
        ('log', 'customData', 'probe2Temperature'),
        ('log', 'customData', 'referenceProbeVoltage'),
        ('log', 'customData', 'probe1Voltage'),
        ('log', 'customData', 'probe2Voltage'),
        (
            'log', 'customData', 'UserConfiguration', 'ChannelConfiguration',
            'AcquisitionConfiguration', 'AcquisitionTime',
            'userAcquisitionTimeFW')]

    if double_ended_flag:
        timeseries_loc_in_hierarchy.append(
            (
                'log', 'customData', 'UserConfiguration',
                'ChannelConfiguration', 'AcquisitionConfiguration',
                'AcquisitionTime', 'userAcquisitionTimeBW'))

    timeseries = {
        item[-1]: dict(loc=item, array=np.zeros(ntime, dtype=np.float32))
        for item in timeseries_loc_in_hierarchy}

    # add units to timeseries (unit of measurement)
    for key, item in timeseries.items():
        if f'customData:{key}:uom' in attrs:
            item['uom'] = attrs[f'customData:{key}:uom']
        else:
            item['uom'] = ''

    # Gather data
    arr_path = 's:' + '/s:'.join(['log', 'logData', 'data'])

    @dask.delayed
    def grab_data_per_file(file_handle):
        """

        Parameters
        ----------
        file_handle

        Returns
        -------

        """
        with open_file(file_handle, mode='r') as f_h:
            eltree = ElementTree.parse(f_h)
            arr_el = eltree.findall(arr_path, namespaces=ns)

            if not len(arr_el) == nx:
                raise ValueError(
                    'Inconsistent length of x-dimension'
                    + '\nCheck if files are mixed up, or if the number of '
                    + 'data points vary per file.')

            # remove the breaks on both sides of the string
            # split the string on the comma
            arr_str = [arr_eli.text[1:-1].split(',') for arr_eli in arr_el]

        return np.array(arr_str, dtype=float)

    data_lst_dly = [grab_data_per_file(fp) for fp in filepathlist]
    data_lst = [
        da.from_delayed(x, shape=(nx, nitem), dtype=float)
        for x in data_lst_dly]
    data_arr = da.stack(data_lst).T  # .compute()

    # Check whether to compute data_arr (if possible 25% faster)
    data_arr_cnk = data_arr.rechunk({0: -1, 1: -1, 2: 'auto'})
    if load_in_memory == 'auto' and data_arr_cnk.npartitions <= 5:
        if not silent:
            print('Reading the data from disk')
        data_arr = data_arr_cnk.compute()
    elif load_in_memory:
        if not silent:
            print('Reading the data from disk')
        data_arr = data_arr_cnk.compute()
    else:
        if not silent:
            print('Not reading the data from disk')
        data_arr = data_arr_cnk

    data_vars = {}
    for name, data_arri in zip(data_item_names, data_arr):
        if name == 'LAF':
            continue
        if tld[name] in dim_attrs:
            data_vars[tld[name]] = (
                ['x', 'time'], data_arri, dim_attrs[tld[name]])
        else:
            raise ValueError(
                f'Dont know what to do with the {name} data column'
            )

    # Obtaining the timeseries data (reference temperature etc)
    _ts_dtype = [(k, np.float32) for k in timeseries]
    _time_dtype = [
        ('filename_tstamp', np.int64), ('minDateTimeIndex', '<U29'),
        ('maxDateTimeIndex', '<U29')]
    ts_dtype = np.dtype(_ts_dtype + _time_dtype)

    @dask.delayed
    def grab_timeseries_per_file(file_handle, xml_version):
        """

        Parameters
        ----------
        file_handle

        Returns
        -------

        """
        with open_file(file_handle, mode='r') as f_h:
            eltree = ElementTree.parse(f_h)

            out = []
            for _, v in timeseries.items():
                # Get all the timeseries data
                if 'userAcquisitionTimeFW' in v['loc']:
                    # requires two namespace searches
                    path1 = 's:' + '/s:'.join(v['loc'][:4])
                    val1 = eltree.findall(path1, namespaces=ns)
                    path2 = 's:' + '/s:'.join(v['loc'][4:6])
                    val2 = val1[chFW].find(path2, namespaces=ns)
                    out.append(val2.text)

                elif 'userAcquisitionTimeBW' in v['loc']:
                    # requires two namespace searches
                    path1 = 's:' + '/s:'.join(v['loc'][:4])
                    val1 = eltree.findall(path1, namespaces=ns)
                    path2 = 's:' + '/s:'.join(v['loc'][4:6])
                    val2 = val1[chBW].find(path2, namespaces=ns)
                    out.append(val2.text)

                else:
                    path = 's:' + '/s:'.join(v['loc'])
                    val = eltree.find(path, namespaces=ns)
                    out.append(val.text)

            # get all the time related data
            startDateTimeIndex = eltree.find(
                's:log/s:startDateTimeIndex', namespaces=ns).text
            endDateTimeIndex = eltree.find(
                's:log/s:endDateTimeIndex', namespaces=ns).text

            if isinstance(file_handle, tuple):
                file_name = os.path.split(file_handle[0])[-1]
            else:
                file_name = os.path.split(file_handle)[-1]

            if xml_version == 6:
                tstamp = np.int64(file_name[10:27])
            elif xml_version == 7:
                tstamp = np.int64(file_name[15:27])
            elif xml_version == 8:
                tstamp = np.int64(file_name[-23:-4].replace('.', ''))
            else:
                raise ValueError(f'Unknown version number: {xml_version}')

            out += [tstamp, startDateTimeIndex, endDateTimeIndex]
        return np.array(tuple(out), dtype=ts_dtype)

    ts_lst_dly = [
        grab_timeseries_per_file(fp, xml_version) for fp in filepathlist]
    ts_lst = [
        da.from_delayed(x, shape=tuple(), dtype=ts_dtype) for x in ts_lst_dly]
    ts_arr = da.stack(ts_lst).compute()

    for name in timeseries:
        if name in dim_attrs:
            data_vars[name] = (('time',), ts_arr[name], dim_attrs[name])

        else:
            data_vars[name] = (('time',), ts_arr[name])

    # construct the coordinate dictionary
    if isinstance(filepathlist[0], tuple):
        filename_list = [os.path.split(f)[-1] for f, n2 in filepathlist]
    else:
        filename_list = [os.path.split(f)[-1] for f in filepathlist]

    coords = {
        'x': ('x', data_arr[0, :, 0], dim_attrs['x']),
        'filename': ('time', filename_list),
        'filename_tstamp': ('time', ts_arr['filename_tstamp'])}

    maxTimeIndex = pd.DatetimeIndex(ts_arr['maxDateTimeIndex'])
    dtFW = ts_arr['userAcquisitionTimeFW'].astype('timedelta64[s]')

    if not double_ended_flag:
        tcoords = coords_time(
            maxTimeIndex,
            timezone_netcdf=timezone_netcdf,
            dtFW=dtFW,
            double_ended_flag=double_ended_flag)
    else:
        dtBW = ts_arr['userAcquisitionTimeBW'].astype('timedelta64[s]')
        tcoords = coords_time(
            maxTimeIndex,
            timezone_netcdf=timezone_netcdf,
            dtFW=dtFW,
            dtBW=dtBW,
            double_ended_flag=double_ended_flag)

    coords.update(tcoords)

    return data_vars, coords, attrs


def read_silixa_files_routine_v4(  # noqa: MC0001
        filepathlist,
        timezone_netcdf='UTC',
        silent=False,
        load_in_memory='auto'):
    """
    Internal routine that reads Silixa files.
    Use dtscalibration.read_silixa_files function instead.

    The silixa files are already timezone aware

    Parameters
    ----------
    load_in_memory
    filepathlist
    timezone_netcdf
    silent

    Returns
    -------

    """
    # translate names
    tld = {
        'ST': 'st',
        'AST': 'ast',
        'REV-ST': 'rst',
        'REV-AST': 'rast',
        'TMP': 'tmp'}

    # Open the first xml file using ET, get the name space and amount of data
    xml_tree = ElementTree.parse(filepathlist[0])
    namespace = get_xml_namespace(xml_tree.getroot())

    logtree = xml_tree.find(f'./{namespace}wellLog')
    logdata_tree = logtree.find(f'./{namespace}logData')

    # Amount of datapoints is the size of the logdata tree
    nx = len(logdata_tree)

    sep = ':'
    ns = {'s': namespace[1:-1]}

    # Obtain metadata from the first file
    attrs = read_silixa_attrs_singlefile(filepathlist[0], sep)

    # Add standardised required attributes
    attrs['isDoubleEnded'] = attrs['customData:isDoubleEnded']
    double_ended_flag = bool(int(attrs['isDoubleEnded']))

    attrs['forwardMeasurementChannel'] = attrs[
        'customData:forwardMeasurementChannel']
    if double_ended_flag:
        attrs['backwardMeasurementChannel'] = attrs[
            'customData:reverseMeasurementChannel']
    else:
        attrs['backwardMeasurementChannel'] = 'N/A'

    chFW = int(attrs['forwardMeasurementChannel']) - 1  # zero-based
    if double_ended_flag:
        chBW = int(attrs['backwardMeasurementChannel']) - 1  # zero-based
    else:
        # no backward channel is negative value. writes better to netcdf
        chBW = -1

    # obtain basic data info
    if double_ended_flag:
        data_item_names = [
            attrs[f'logCurveInfo_{x}:mnemonic'] for x in range(6)]
    else:
        data_item_names = [
            attrs[f'logCurveInfo_{x}:mnemonic'] for x in range(4)]

    nitem = len(data_item_names)

    ntime = len(filepathlist)

    # print summary
    if not silent:
        print(f'{ntime} files were found, each representing a single timestep')
        print(f'{nitem} recorded vars were found: ' +
              ', '.join(data_item_names))
        print(f'Recorded at {nx} points along the cable')

        if double_ended_flag:
            print('The measurement is double ended')
        else:
            print('The measurement is single ended')

    # obtain timeseries from data
    timeseries_loc_in_hierarchy = [
        ('wellLog', 'customData', 'acquisitionTime'),
        ('wellLog', 'customData', 'referenceTemperature'),
        ('wellLog', 'customData', 'probe1Temperature'),
        ('wellLog', 'customData', 'probe2Temperature'),
        ('wellLog', 'customData', 'referenceProbeVoltage'),
        ('wellLog', 'customData', 'probe1Voltage'),
        ('wellLog', 'customData', 'probe2Voltage'),
        (
            'wellLog', 'customData', 'UserConfiguration',
            'ChannelConfiguration', 'AcquisitionConfiguration',
            'AcquisitionTime', 'userAcquisitionTimeFW')]

    if double_ended_flag:
        timeseries_loc_in_hierarchy.append(
            (
                'wellLog', 'customData', 'UserConfiguration',
                'ChannelConfiguration', 'AcquisitionConfiguration',
                'AcquisitionTime', 'userAcquisitionTimeBW'))

    timeseries = {
        item[-1]: dict(loc=item, array=np.zeros(ntime, dtype=np.float32))
        for item in timeseries_loc_in_hierarchy}

    # add units to timeseries (unit of measurement)
    for key, item in timeseries.items():
        if f'customData:{key}:uom' in attrs:
            item['uom'] = attrs[f'customData:{key}:uom']
        else:
            item['uom'] = ''

    # Gather data
    arr_path = 's:' + '/s:'.join(['wellLog', 'logData', 'data'])

    @dask.delayed
    def grab_data_per_file(file_handle):
        """

        Parameters
        ----------
        file_handle

        Returns
        -------

        """
        with open_file(file_handle, mode='r') as f_h:
            eltree = ElementTree.parse(f_h)
            arr_el = eltree.findall(arr_path, namespaces=ns)

            if not len(arr_el) == nx:
                raise ValueError(
                    'Inconsistent length of x-dimension'
                    + '\nCheck if files are mixed up, or if the number of '
                    + 'data points vary per file.')

            # remove the breaks on both sides of the string
            # split the string on the comma
            arr_str = [arr_eli.text.split(',') for arr_eli in arr_el]
        return np.array(arr_str, dtype=float)

    data_lst_dly = [grab_data_per_file(fp) for fp in filepathlist]
    data_lst = [
        da.from_delayed(x, shape=(nx, nitem), dtype=float)
        for x in data_lst_dly]
    data_arr = da.stack(data_lst).T  # .compute()

    # Check whether to compute data_arr (if possible 25% faster)
    data_arr_cnk = data_arr.rechunk({0: -1, 1: -1, 2: 'auto'})
    if load_in_memory == 'auto' and data_arr_cnk.npartitions <= 5:
        if not silent:
            print('Reading the data from disk')
        data_arr = data_arr_cnk.compute()
    elif load_in_memory:
        if not silent:
            print('Reading the data from disk')
        data_arr = data_arr_cnk.compute()
    else:
        if not silent:
            print('Not reading the data from disk')
        data_arr = data_arr_cnk

    data_vars = {}
    for name, data_arri in zip(data_item_names, data_arr):
        if name == 'LAF':
            continue

        if tld[name] in dim_attrs:
            data_vars[tld[name]] = (
                ['x', 'time'], data_arri, dim_attrs[tld[name]])

        else:
            raise ValueError(
                'Dont know what to do with the'
                ' {name} data column')

    # Obtaining the timeseries data (reference temperature etc)
    _ts_dtype = [(k, np.float32) for k in timeseries]
    _time_dtype = [
        ('filename_tstamp', np.int64), ('minDateTimeIndex', '<U29'),
        ('maxDateTimeIndex', '<U29')]
    ts_dtype = np.dtype(_ts_dtype + _time_dtype)

    @dask.delayed
    def grab_timeseries_per_file(file_handle):
        """

        Parameters
        ----------
        file_handle

        Returns
        -------

        """
        with open_file(file_handle, mode='r') as f_h:
            eltree = ElementTree.parse(f_h)

            out = []
            for k, v in timeseries.items():
                # Get all the timeseries data
                if 'userAcquisitionTimeFW' in v['loc']:
                    # requires two namespace searches
                    path1 = 's:' + '/s:'.join(v['loc'][:4])
                    val1 = eltree.findall(path1, namespaces=ns)
                    path2 = 's:' + '/s:'.join(v['loc'][4:6])
                    val2 = val1[chFW].find(path2, namespaces=ns)
                    out.append(val2.text)

                elif 'userAcquisitionTimeBW' in v['loc']:
                    # requires two namespace searches
                    path1 = 's:' + '/s:'.join(v['loc'][:4])
                    val1 = eltree.findall(path1, namespaces=ns)
                    path2 = 's:' + '/s:'.join(v['loc'][4:6])
                    val2 = val1[chBW].find(path2, namespaces=ns)
                    out.append(val2.text)

                else:
                    path = 's:' + '/s:'.join(v['loc'])
                    val = eltree.find(path, namespaces=ns)
                    out.append(val.text)

            # get all the time related data
            startDateTimeIndex = eltree.find(
                's:wellLog/s:minDateTimeIndex', namespaces=ns).text
            endDateTimeIndex = eltree.find(
                's:wellLog/s:maxDateTimeIndex', namespaces=ns).text

            if isinstance(file_handle, tuple):
                file_name = os.path.split(file_handle[0])[-1]
            else:
                file_name = os.path.split(file_handle)[-1]

            tstamp = np.int64(file_name[10:-4])

            out += [tstamp, startDateTimeIndex, endDateTimeIndex]
        return np.array(tuple(out), dtype=ts_dtype)

    ts_lst_dly = [grab_timeseries_per_file(fp) for fp in filepathlist]
    ts_lst = [
        da.from_delayed(x, shape=tuple(), dtype=ts_dtype) for x in ts_lst_dly]
    ts_arr = da.stack(ts_lst).compute()

    for name in timeseries:
        if name in dim_attrs:
            data_vars[name] = (('time',), ts_arr[name], dim_attrs[name])

        else:
            data_vars[name] = (('time',), ts_arr[name])

    # construct the coordinate dictionary
    coords = {
        'x': ('x', data_arr[0, :, 0], dim_attrs['x']),
        'filename': ('time', [os.path.split(f)[1] for f in filepathlist]),
        'filename_tstamp': ('time', ts_arr['filename_tstamp'])}

    maxTimeIndex = pd.DatetimeIndex(ts_arr['maxDateTimeIndex'])
    dtFW = ts_arr['userAcquisitionTimeFW'].astype('timedelta64[s]')

    if not double_ended_flag:
        tcoords = coords_time(
            maxTimeIndex,
            timezone_netcdf=timezone_netcdf,
            dtFW=dtFW,
            double_ended_flag=double_ended_flag)
    else:
        dtBW = ts_arr['userAcquisitionTimeBW'].astype('timedelta64[s]')
        tcoords = coords_time(
            maxTimeIndex,
            timezone_netcdf=timezone_netcdf,
            dtFW=dtFW,
            dtBW=dtBW,
            double_ended_flag=double_ended_flag)

    coords.update(tcoords)

    return data_vars, coords, attrs


def read_sensornet_files_routine_v3(
        filepathlist,
        timezone_netcdf='UTC',
        timezone_input_files='UTC',
        silent=False,
        add_internal_fiber_length=50.,
        fiber_length=None,
        flip_reverse_measurements=False):
    """
    Internal routine that reads Sensor files.
    Use dtscalibration.read_sensornet_files function instead.

    Parameters
    ----------
    filepathlist
    timezone_netcdf
    timezone_input_files
    silent
    add_internal_fiber_length : float
        Set to zero if only the measurements of the fiber connected to the DTS
        system of interest. Set to 50 if you also want to keep the internal
        reference section.
    fiber_length : float
        It is the fiber length between the two connector entering the DTS
        device.

    Returns
    -------

    """

    # Obtain metadata from the first file
    data, meta = read_sensornet_single(filepathlist[0])

    # Pop keys from the meta dict which are variable over time
    popkeys = (
        'T ext. ref 1 (°C)', 'T ext. ref 2 (°C)', 'T internal ref (°C)',
        'date', 'time', 'gamma', 'k internal', 'k external')
    [meta.pop(key) for key in popkeys]
    attrs = meta

    # Add standardised required attributes
    if meta['differential loss correction'] == 'single-ended':
        attrs['isDoubleEnded'] = '0'
    elif meta['differential loss correction'] == 'combined':
        attrs['isDoubleEnded'] = '1'

    double_ended_flag = bool(int(attrs['isDoubleEnded']))

    attrs['forwardMeasurementChannel'] = meta['forward channel'][-1]
    if double_ended_flag:
        attrs['backwardMeasurementChannel'] = 'N/A'
    else:
        attrs['backwardMeasurementChannel'] = meta['reverse channel'][-1]

    # obtain basic data info
    nx = data['x'].size

    ntime = len(filepathlist)

    # chFW = int(attrs['forwardMeasurementChannel']) - 1  # zero-based
    # if double_ended_flag:
    #     chBW = int(attrs['backwardMeasurementChannel']) - 1  # zero-based
    # else:
    #     # no backward channel is negative value. writes better to netcdf
    #     chBW = -1

    # print summary
    if not silent:
        print(
            '%s files were found,' % ntime
            + ' each representing a single timestep')
        print('Recorded at %s points along the cable' % nx)

        if double_ended_flag:
            print('The measurement is double ended')
        else:
            print('The measurement is single ended')

    #   Gather data
    # x has already been read. should not change over time
    xraw = data['x']

    # Define all variables
    referenceTemperature = np.zeros(ntime)
    probe1temperature = np.zeros(ntime)
    probe2temperature = np.zeros(ntime)
    gamma_ddf = np.zeros(ntime)
    k_internal = np.zeros(ntime)
    k_external = np.zeros(ntime)
    acquisitiontimeFW = np.zeros(ntime)
    acquisitiontimeBW = np.zeros(ntime)

    timestamp = [''] * ntime
    ST = np.zeros((nx, ntime))
    AST = np.zeros((nx, ntime))
    TMP = np.zeros((nx, ntime))

    if double_ended_flag:
        REV_ST = np.zeros((nx, ntime))
        REV_AST = np.zeros((nx, ntime))

    for ii in range(ntime):
        data, meta = read_sensornet_single(filepathlist[ii])

        timestamp[ii] = pd.DatetimeIndex([
            meta['date'] + ' ' + meta['time']])[0]
        probe1temperature[ii] = float(meta['T ext. ref 1 (°C)'])
        probe2temperature[ii] = float(meta['T ext. ref 2 (°C)'])
        referenceTemperature[ii] = float(meta['T internal ref (°C)'])
        gamma_ddf[ii] = float(meta['gamma'])
        k_internal[ii] = float(meta['k internal'])
        k_external[ii] = float(meta['k external'])
        acquisitiontimeFW[ii] = float(meta['forward acquisition time'])
        acquisitiontimeBW[ii] = float(meta['reverse acquisition time'])

        ST[:, ii] = data['st']
        AST[:, ii] = data['ast']
        TMP[:, ii] = data['tmp']

        if double_ended_flag:
            REV_ST[:, ii] = data['rst']
            REV_AST[:, ii] = data['rast']

    if fiber_length is None and double_ended_flag:
        fiber_length = np.max([0., xraw[-1] - add_internal_fiber_length])
    elif fiber_length is None and not double_ended_flag:
        fiber_length = xraw[-1]
    else:
        pass

    assert fiber_length > 0., '`fiber_length` is not defined. Use key' \
                              'word argument in read function.' + \
                              str(fiber_length)

    fiber_start_index = (np.abs(xraw + add_internal_fiber_length)).argmin()
    fiber_0_index = np.abs(xraw).argmin()
    fiber_1_index = (np.abs(xraw - fiber_length)).argmin()
    fiber_n_indices = fiber_1_index - fiber_0_index
    fiber_n_indices_internal = fiber_0_index - fiber_start_index
    if double_ended_flag:
        fiber_end_index = np.min(
            [xraw.size, fiber_1_index + fiber_n_indices_internal])
    else:
        fiber_end_index = fiber_1_index

    if double_ended_flag:
        if not flip_reverse_measurements:
            # fiber length how the backward channel is aligned
            fiber_length_raw = float(meta['fibre end'])
            fiber_bw_1_index = np.abs(xraw - fiber_length_raw).argmin()
            fiber_bw_end_index = np.min(
                [
                    xraw.size, fiber_bw_1_index +
                    (fiber_end_index - fiber_1_index)])
            fiber_bw_start_index = np.max(
                [
                    0, fiber_bw_1_index - fiber_n_indices
                    - fiber_n_indices_internal])

            REV_ST = REV_ST[fiber_bw_start_index:fiber_bw_end_index]
            REV_AST = REV_AST[fiber_bw_start_index:fiber_bw_end_index]

        else:
            # Use the fiber indices from the forward channel
            n_indices_internal_left = fiber_0_index - fiber_start_index
            n_indices_internal_right = np.max(
                [0, fiber_end_index - fiber_1_index])
            n_indices_internal_shortest = np.min(
                [n_indices_internal_left, n_indices_internal_right])
            fiber_start_index = fiber_0_index - n_indices_internal_shortest
            fiber_end_index = fiber_0_index + fiber_n_indices + \
                n_indices_internal_shortest
            REV_ST = REV_ST[fiber_end_index:fiber_start_index:-1]
            REV_AST = REV_AST[fiber_end_index:fiber_start_index:-1]

    x = xraw[fiber_start_index:fiber_end_index]
    TMP = TMP[fiber_start_index:fiber_end_index]
    ST = ST[fiber_start_index:fiber_end_index]
    AST = AST[fiber_start_index:fiber_end_index]

    data_vars = {
        'st': (['x', 'time'], ST, dim_attrs['st']),
        'ast': (['x', 'time'], AST, dim_attrs['ast']),
        'tmp': (['x', 'time'], TMP, dim_attrs['tmp']),
        'probe1Temperature':
            (
                'time', probe1temperature, {
                    'name': 'Probe 1 temperature',
                    'description': 'reference probe 1 '
                                   'temperature',
                    'units': 'degC'}),
        'probe2Temperature':
            (
                'time', probe2temperature, {
                    'name': 'Probe 2 temperature',
                    'description': 'reference probe 2 '
                                   'temperature',
                    'units': 'degC'}),
        'referenceTemperature':
            (
                'time', referenceTemperature, {
                    'name': 'reference temperature',
                    'description': 'Internal reference '
                                   'temperature',
                    'units': 'degC'}),
        'gamma_ddf':
            (
                'time', gamma_ddf, {
                    'name': 'gamma ddf',
                    'description': 'machine '
                                   'calibrated gamma',
                    'units': '-'}),
        'k_internal':
            (
                'time', k_internal, {
                    'name': 'k internal',
                    'description': 'machine calibrated '
                                   'internal k',
                    'units': '-'}),
        'k_external':
            (
                'time', k_external, {
                    'name': 'reference temperature',
                    'description': 'machine calibrated '
                                   'external k',
                    'units': '-'}),
        'userAcquisitionTimeFW':
            ('time', acquisitiontimeFW, dim_attrs['userAcquisitionTimeFW']),
        'userAcquisitionTimeBW':
            ('time', acquisitiontimeBW, dim_attrs['userAcquisitionTimeBW'])}

    if double_ended_flag:
        data_vars['rst'] = (['x', 'time'], REV_ST, dim_attrs['rst'])
        data_vars['rast'] = (['x', 'time'], REV_AST, dim_attrs['rast'])

    filenamelist = [os.path.split(f)[-1] for f in filepathlist]

    coords = {
        'x': ('x', x, dim_attrs['x']),
        'filename': ('time', filenamelist)}

    dtFW = data_vars['userAcquisitionTimeFW'][1].astype('timedelta64[s]')
    dtBW = data_vars['userAcquisitionTimeBW'][1].astype('timedelta64[s]')
    if not double_ended_flag:
        tcoords = coords_time(
            np.array(timestamp).astype('datetime64[ns]'),
            timezone_netcdf=timezone_netcdf,
            timezone_input_files=timezone_input_files,
            dtFW=dtFW,
            double_ended_flag=double_ended_flag)
    else:
        tcoords = coords_time(
            np.array(timestamp).astype('datetime64[ns]'),
            timezone_netcdf=timezone_netcdf,
            timezone_input_files=timezone_input_files,
            dtFW=dtFW,
            dtBW=dtBW,
            double_ended_flag=double_ended_flag)

    coords.update(tcoords)

    return data_vars, coords, attrs


def read_silixa_attrs_singlefile(filename, sep):
    """

    Parameters
    ----------
    filename
    sep

    Returns
    -------

    """
    import xmltodict

    def metakey(meta, dict_to_parse, prefix):
        """
        Fills the metadata dictionairy with data from dict_to_parse.
        The dict_to_parse is the raw data from a silixa xml-file.
        dict_to_parse is a nested dictionary to represent the
        different levels of hierarchy. For example,
        toplevel = {lowlevel: {key: value}}.
        This function returns {'toplevel:lowlevel:key': value}.
        Where prefix is the flattened hierarchy.

        Parameters
        ----------
        meta : dict
            the output dictionairy with prcessed metadata
        dict_to_parse : dict
        prefix

        Returns
        -------

        """

        for key in dict_to_parse:
            if prefix == "":

                prefix_parse = key.replace('@', '')
            else:
                prefix_parse = sep.join([prefix, key.replace('@', '')])

            if prefix_parse == sep.join(('logData', 'data')):
                # skip the LAF , ST data
                continue

            if hasattr(dict_to_parse[key], 'keys'):
                # Nested dictionaries, flatten hierarchy.
                meta.update(metakey(meta, dict_to_parse[key], prefix_parse))

            elif isinstance(dict_to_parse[key], list):
                # if the key has values for the multiple channels
                for ival, val in enumerate(dict_to_parse[key]):
                    num_key = prefix_parse + '_' + str(ival)
                    meta.update(metakey(meta, val, num_key))
            else:

                meta[prefix_parse] = dict_to_parse[key]

        return meta

    with open_file(filename) as fh:
        doc_ = xmltodict.parse(fh.read())

    if 'wellLogs' in doc_.keys():
        doc = doc_['wellLogs']['wellLog']
    else:
        doc = doc_['logs']['log']

    return metakey({}, doc, '')


def read_sensortran_files_routine(
        filepathlist_dts,
        filepathlist_temp,
        timezone_netcdf='UTC',
        silent=False):
    """
    Internal routine that reads sensortran files.
    Use dtscalibration.read_sensortran_files function instead.

    The sensortran files are in UTC time

    Parameters
    ----------
    filepathlist_dts
    filepathlist_temp
    timezone_netcdf
    silent

    Returns
    -------

    """

    # Obtain metadata from the first file
    data_dts, meta_dts = read_sensortran_single(filepathlist_dts[0])
    data_temp, meta_temp = read_sensortran_single(filepathlist_temp[0])

    attrs = meta_dts

    # Add standardised required attributes
    attrs['isDoubleEnded'] = '0'

    attrs['forwardMeasurementChannel'] = meta_dts['channel_id'] - 1
    attrs['backwardMeasurementChannel'] = 'N/A'

    # obtain basic data info
    nx = meta_temp['num_points']

    ntime = len(filepathlist_dts)

    # print summary
    if not silent:
        print(
            '%s files were found,' % ntime
            + ' each representing a single timestep')
        print('Recorded at %s points along the cable' % nx)

        print('The measurement is single ended')

    #   Gather data
    # x has already been read. should not change over time
    x = data_temp['x']

    # Define all variables
    referenceTemperature = np.zeros(ntime)
    acquisitiontimeFW = np.ones(ntime)

    timestamp = [''] * ntime
    ST = np.zeros((nx, ntime), dtype=np.int32)
    AST = np.zeros((nx, ntime), dtype=np.int32)
    TMP = np.zeros((nx, ntime))

    ST_zero = np.zeros((ntime))
    AST_zero = np.zeros((ntime))

    for ii in range(ntime):
        data_dts, meta_dts = read_sensortran_single(filepathlist_dts[ii])
        data_temp, meta_temp = read_sensortran_single(filepathlist_temp[ii])

        timestamp[ii] = data_dts['time']

        referenceTemperature[ii] = data_temp['reference_temperature'] - 273.15

        ST[:, ii] = data_dts['st'][:nx]
        AST[:, ii] = data_dts['ast'][:nx]
        # The TMP can vary by 1 or 2 datapoints, dynamically assign the values
        TMP[:meta_temp['num_points'], ii] = data_temp['tmp'][:nx]

        zero_index = (meta_dts['num_points'] - nx) // 2
        ST_zero[ii] = np.mean(data_dts['st'][nx + zero_index:])
        AST_zero[ii] = np.mean(data_dts['ast'][nx + zero_index:])

    data_vars = {
        'st': (['x', 'time'], ST, dim_attrs['st']),
        'ast': (['x', 'time'], AST, dim_attrs['ast']),
        'tmp':
            (
                ['x', 'time'], TMP, {
                    'name': 'tmp',
                    'description': 'Temperature calibrated by device',
                    'units': meta_temp['y_units']}),
        'referenceTemperature':
            (
                'time', referenceTemperature, {
                    'name': 'reference temperature',
                    'description': 'Internal reference '
                                   'temperature',
                    'units': 'degC'}),
        'st_zero':
            (
                ['time'], ST_zero, {
                    'name': 'ST_zero',
                    'description': 'Stokes zero count',
                    'units': meta_dts['y_units']}),
        'ast_zero':
            (
                ['time'], AST_zero, {
                    'name': 'AST_zero',
                    'description': 'anit-Stokes zero count',
                    'units': meta_dts['y_units']}),
        'userAcquisitionTimeFW':
            ('time', acquisitiontimeFW, dim_attrs['userAcquisitionTimeFW'])}

    filenamelist_dts = [os.path.split(f)[-1] for f in filepathlist_dts]
    filenamelist_temp = [os.path.split(f)[-1] for f in filepathlist_temp]

    coords = {
        'x':
            (
                'x', x, {
                    'name': 'distance',
                    'description': 'Length along fiber',
                    'long_description':
                        'Starting at connector ' + 'of forward channel',
                    'units': 'm'}),
        'filename': ('time', filenamelist_dts),
        'filename_temp': ('time', filenamelist_temp)}

    dtFW = data_vars['userAcquisitionTimeFW'][1].astype('timedelta64[s]')

    tcoords = coords_time(
        np.array(timestamp).astype('datetime64[ns]'),
        timezone_netcdf=timezone_netcdf,
        timezone_input_files='UTC',
        dtFW=dtFW,
        double_ended_flag=False)

    coords.update(tcoords)

    return data_vars, coords, attrs


def read_sensortran_single(fname):
    """
    Internal routine that reads a single sensortran file.
    Use dtscalibration.read_sensortran_files function instead.

    Parameters
    ----------
    fname

    Returns
    -------
    data, metadata
    """
    import struct
    from datetime import datetime

    meta = {}
    data = {}
    with open(fname, 'rb') as f:
        meta['survey_type'] = struct.unpack('<h', f.read(2))[0]
        meta['hdr_version'] = struct.unpack('<h', f.read(2))[0]
        meta['x_units'] = struct.unpack('<i', f.read(4))[0]
        meta['y_units'] = struct.unpack('<i', f.read(4))[0]
        meta['num_points'] = struct.unpack('<i', f.read(4))[0]
        meta['num_pulses'] = struct.unpack('<i', f.read(4))[0]
        meta['channel_id'] = struct.unpack('<i', f.read(4))[0]
        meta['num_subtraces'] = struct.unpack('<i', f.read(4))[0]
        meta['num_skipped'] = struct.unpack('<i', f.read(4))[0]

        data['reference_temperature'] = struct.unpack('<f', f.read(4))[0]
        data['time'] = datetime.fromtimestamp(
            struct.unpack('<i', f.read(4))[0])

        meta['probe_name'] = f.read(128).decode('utf-16').split('\x00')[0]

        meta['hdr_size'] = struct.unpack('<i', f.read(4))[0]
        meta['hw_config'] = struct.unpack('<i', f.read(4))[0]

        data_1 = f.read(meta['num_points'] * 4)
        data_2 = f.read(meta['num_points'] * 4)

        if meta['survey_type'] == 0:
            distance = np.frombuffer(data_1, dtype=np.float32)
            temperature = np.frombuffer(data_2, dtype=np.float32)
            data['x'] = distance
            data['tmp'] = temperature

        if meta['survey_type'] == 2:
            ST = np.frombuffer(data_1, dtype=np.int32)
            AST = np.frombuffer(data_2, dtype=np.int32)
            data['st'] = ST
            data['ast'] = AST

    x_units_map = {0: 'm', 1: 'ft', 2: 'n/a'}
    meta['x_units'] = x_units_map[meta['x_units']]
    y_units_map = {0: 'K', 1: 'degC', 2: 'degF', 3: 'counts'}
    meta['y_units'] = y_units_map[meta['y_units']]

    return data, meta


def read_apsensing_files_routine(
        filepathlist,
        timezone_netcdf='UTC',
        silent=False,
        load_in_memory='auto'):
    """
    Internal routine that reads AP Sensing files.
    Use dtscalibration.read_apsensing_files function instead.

    The AP sensing files are not timezone aware

    Parameters
    ----------
    filepathlist
    timezone_netcdf
    silent
    load_in_memory

    Returns
    -------

    """
    from xml.etree import ElementTree

    import dask

    # translate names
    tld = {
        'ST': 'st',
        'AST': 'ast',
        'REV-ST': 'rst',
        'REV-AST': 'rast',
        'TEMP': 'tmp'}

    # Open the first xml file using ET, get the name space and amount of data
    xml_tree = ElementTree.parse(filepathlist[0])
    namespace = get_xml_namespace(xml_tree.getroot())

    logtree = xml_tree.find(
        (
            '{0}wellSet/{0}well/{0}wellboreSet/{0}wellbore'
            + '/{0}wellLogSet/{0}wellLog').format(namespace))
    logdata_tree = logtree.find('./{0}logData'.format(namespace))

    # Amount of datapoints is the size of the logdata tree
    nx = len(logdata_tree)

    sep = ':'
    ns = {'s': namespace[1:-1]}

    # Obtain metadata from the first file
    attrs, skip_chars = read_apsensing_attrs_singlefile(filepathlist[0], sep)

    # Add standardised required attributes
    # No example of DE file available
    attrs['isDoubleEnded'] = '0'
    double_ended_flag = bool(int(attrs['isDoubleEnded']))

    attrs['forwardMeasurementChannel'] = attrs[
        'wellbore:dtsMeasurementSet:dtsMeasurement:connectedToFiber:uidRef']
    attrs['backwardMeasurementChannel'] = 'N/A'

    data_item_names = [
        attrs['wellbore:wellLogSet:wellLog:logCurveInfo_{0}:mnemonic'.format(
            x)] for x in range(0, 4)]

    nitem = len(data_item_names)

    ntime = len(filepathlist)

    # print summary
    if not silent:
        print(
            '%s files were found, each representing a single timestep' % ntime)
        print(
            '%s recorded vars were found: ' % nitem
            + ', '.join(data_item_names))
        print('Recorded at %s points along the cable' % nx)

        if double_ended_flag:
            print('The measurement is double ended')
        else:
            print('The measurement is single ended')

    # Gather data
    arr_path = 's:' + '/s:'.join(
        [
            'wellSet', 'well', 'wellboreSet', 'wellbore', 'wellLogSet',
            'wellLog', 'logData', 'data'])

    @dask.delayed
    def grab_data_per_file(file_handle):
        """

        Parameters
        ----------
        file_handle

        Returns
        -------

        """

        with open_file(file_handle, mode='r') as f_h:
            if skip_chars:
                f_h.read(3)
            eltree = ElementTree.parse(f_h)
            arr_el = eltree.findall(arr_path, namespaces=ns)

            if not len(arr_el) == nx:
                raise ValueError(
                    'Inconsistent length of x-dimension'
                    + '\nCheck if files are mixed up, or if the number of '
                    + 'data points vary per file.')

            # remove the breaks on both sides of the string
            # split the string on the comma
            arr_str = [arr_eli.text.split(',') for arr_eli in arr_el]
        return np.array(arr_str, dtype=float)

    data_lst_dly = [grab_data_per_file(fp) for fp in filepathlist]

    data_lst = [
        da.from_delayed(x, shape=(nx, nitem), dtype=float)
        for x in data_lst_dly]
    data_arr = da.stack(data_lst).T  # .compute()

    # Check whether to compute data_arr (if possible 25% faster)
    data_arr_cnk = data_arr.rechunk({0: -1, 1: -1, 2: 'auto'})
    if load_in_memory == 'auto' and data_arr_cnk.npartitions <= 5:
        if not silent:
            print('Reading the data from disk')
        data_arr = data_arr_cnk.compute()
    elif load_in_memory:
        if not silent:
            print('Reading the data from disk')
        data_arr = data_arr_cnk.compute()
    else:
        if not silent:
            print('Not reading the data from disk')
        data_arr = data_arr_cnk

    data_vars = {}
    for name, data_arri in zip(data_item_names, data_arr):
        if name == 'LAF':
            continue

        if tld[name] in dim_attrs_apsensing:
            data_vars[tld[name]] = (
                ['x', 'time'], data_arri, dim_attrs_apsensing[tld[name]])
        elif name in dim_attrs_apsensing:
            data_vars[tld[name]] = (
                ['x', 'time'], data_arri, dim_attrs_apsensing[name])
        else:
            raise ValueError(
                'Dont know what to do with the'
                + ' {} data column'.format(name))

    _time_dtype = [('filename_tstamp', np.int64), ('acquisitionTime', '<U29')]
    ts_dtype = np.dtype(_time_dtype)

    @dask.delayed
    def grab_timeseries_per_file(file_handle):
        """

        Parameters
        ----------
        file_handle

        Returns
        -------

        """
        with open_file(file_handle, mode='r') as f_h:
            if skip_chars:
                f_h.read(3)
            eltree = ElementTree.parse(f_h)

            out = []

            # get all the time related data
            creationDate = eltree.find(
                (
                    '{0}wellSet/{0}well/{0}wellboreSet'
                    + '/{0}wellbore/{0}wellLogSet'
                    + '/{0}wellLog/{0}creationDate').format(namespace)).text

            if isinstance(file_handle, tuple):
                file_name = os.path.split(file_handle[0])[-1]
            else:
                file_name = os.path.split(file_handle)[-1]

            tstamp = np.int64(file_name[-20:-4])

            out += [tstamp, creationDate]
        return np.array(tuple(out), dtype=ts_dtype)

    ts_lst_dly = [grab_timeseries_per_file(fp) for fp in filepathlist]
    ts_lst = [
        da.from_delayed(x, shape=tuple(), dtype=ts_dtype) for x in ts_lst_dly]
    ts_arr = da.stack(ts_lst).compute()

    data_vars['creationDate'] = (
        ('time',), [pd.Timestamp(str(item[1])) for item in ts_arr])

    # construct the coordinate dictionary
    coords = {
        'x': ('x', data_arr[0, :, 0], dim_attrs_apsensing['x']),
        'filename': ('time', [os.path.split(f)[1] for f in filepathlist]),
        'time': data_vars['creationDate']}

    return data_vars, coords, attrs


def read_apsensing_attrs_singlefile(filename, sep):
    """

    Parameters
    ----------
    filename
    sep

    Returns
    -------

    """
    from xml.parsers.expat import ExpatError

    import xmltodict

    def metakey(meta, dict_to_parse, prefix):
        """
        Fills the metadata dictionairy with data from dict_to_parse.
        The dict_to_parse is the raw data from a silixa xml-file.
        dict_to_parse is a nested dictionary to represent the
        different levels of hierarchy. For example,
        toplevel = {lowlevel: {key: value}}.
        This function returns {'toplevel:lowlevel:key': value}.
        Where prefix is the flattened hierarchy.

        Parameters
        ----------
        meta : dict
            the output dictionairy with prcessed metadata
        dict_to_parse : dict
        prefix

        Returns
        -------

        """

        for key in dict_to_parse:
            if prefix == "":

                prefix_parse = key.replace('@', '')
            else:
                prefix_parse = sep.join([prefix, key.replace('@', '')])

            items = ('wellbore', 'wellLogSet', 'wellLog', 'logData', 'data')
            if prefix_parse == sep.join(items):
                continue

            if hasattr(dict_to_parse[key], 'keys'):
                # Nested dictionaries, flatten hierarchy.
                meta.update(metakey(meta, dict_to_parse[key], prefix_parse))

            elif isinstance(dict_to_parse[key], list):
                # if the key has values for the multiple channels
                for ival, val in enumerate(dict_to_parse[key]):
                    num_key = prefix_parse + '_' + str(ival)
                    meta.update(metakey(meta, val, num_key))
            else:

                meta[prefix_parse] = dict_to_parse[key]

        return meta

    with open_file(filename) as fh:
        data = fh.read()
        try:
            doc_ = xmltodict.parse(data)
            skip_chars = False
        # the first 3 characters can be weird, skip them
        except ExpatError:
            doc_ = xmltodict.parse(data[3:])
            skip_chars = True

    doc = doc_['WITSMLComposite']['wellSet']['well']['wellboreSet']

    return metakey({}, doc, ''), skip_chars


def read_sensornet_single(filename):
    """

    Parameters
    ----------
    filename

    Returns
    -------

    """
    headerlength = 26

    # The $\circ$ Celsius symbol is unreadable in utf8
    with open_file(filename, encoding='windows-1252') as fileobject:
        filelength = sum([1 for _ in fileobject])
    datalength = filelength - headerlength

    meta = {}
    with open_file(filename, encoding='windows-1252') as fileobject:
        for ii in range(0, 4):
            fileline = fileobject.readline().split(':\t')
            meta[fileline[0]] = fileline[1].replace('\n', '')

        for ii in range(4, headerlength - 1):
            fileline = fileobject.readline().split('\t')
            meta[fileline[0]] = fileline[1].replace('\n', '').replace(',', '.')

        # data_names =
        fileobject.readline().split('\t')

        if meta['differential loss correction'] == 'single-ended':
            data = {
                'x': np.zeros(datalength),
                'tmp': np.zeros(datalength),
                'st': np.zeros(datalength),
                'ast': np.zeros(datalength)}

            for ii in range(0, datalength):
                fileline = fileobject.readline().replace(',', '.').split('\t')

                data['x'][ii] = float(fileline[0])
                data['tmp'][ii] = float(fileline[1])
                data['st'][ii] = float(fileline[2])
                data['ast'][ii] = float(fileline[3])

        elif meta['differential loss correction'] == 'combined':
            data = {
                'x': np.zeros(datalength),
                'tmp': np.zeros(datalength),
                'st': np.zeros(datalength),
                'ast': np.zeros(datalength),
                'rst': np.zeros(datalength),
                'rast': np.zeros(datalength)}

            for ii in range(0, datalength):
                fileline = fileobject.readline().replace(',', '.').split('\t')

                data['x'][ii] = float(fileline[0])
                data['tmp'][ii] = float(fileline[1])
                data['st'][ii] = float(fileline[2])
                data['ast'][ii] = float(fileline[3])
                data['rst'][ii] = float(fileline[4])
                data['rast'][ii] = float(fileline[5])

        else:
            raise ValueError(
                'unknown differential loss correction: "'
                + meta['differential loss correction'] + '"')

    meta['default loss term dB per km'] = meta['default loss term (dB/km)']
    del meta['default loss term (dB/km)']

    return data, meta


def get_xml_namespace(element):
    """

    Parameters
    ----------
    element

    Returns
    -------

    """
    import re
    m = re.match('\\{.*\\}', element.tag)
    return m.group(0) if m else ''


def coords_time(
        maxTimeIndex,
        timezone_input_files=None,
        timezone_netcdf='UTC',
        dtFW=None,
        dtBW=None,
        double_ended_flag=False):
    """
    Prepares the time coordinates for the construction of DataStore
    instances with metadata

    Parameters
    ----------
    maxTimeIndex : array-like (1-dimensional)
        Is an array with 'datetime64[ns]' timestamps of the end of the
        forward channel. If single ended this is the end of the measurement.
        If double ended this is halfway the double ended measurement.
    timezone_input_files : string, pytz.timezone, dateutil.tz.tzfile or None
        A string of a timezone that is understood by pandas of maxTimeIndex.
        If None, it is assumed that the input files are already timezone aware
    timezone_netcdf : string, pytz.timezone, dateutil.tz.tzfile or None
        A string of a timezone that is understood by pandas to write the
        netCDF to. Using UTC as default, according to CF conventions.
    dtFW : array-like (1-dimensional) of float
        The acquisition time of the Forward channel in seconds
    dtBW : array-like (1-dimensional) of float
        The acquisition time of the Backward channel in seconds
    double_ended_flag : bool
        A flag whether the measurement is double ended

    Returns
    -------

    """
    time_attrs = {
        'time':
            {
                'description': 'time halfway the measurement',
                'timezone': str(timezone_netcdf)},
        'timestart':
            {
                'description': 'time start of the measurement',
                'timezone': str(timezone_netcdf)},
        'timeend':
            {
                'description': 'time end of the measurement',
                'timezone': str(timezone_netcdf)},
        'timeFW':
            {
                'description': 'time halfway the forward channel measurement',
                'timezone': str(timezone_netcdf)},
        'timeFWstart':
            {
                'description': 'time start of the forward channel measurement',
                'timezone': str(timezone_netcdf)},
        'timeFWend':
            {
                'description': 'time end of the forward channel measurement',
                'timezone': str(timezone_netcdf)},
        'timeBW':
            {
                'description': 'time halfway the backward channel measurement',
                'timezone': str(timezone_netcdf)},
        'timeBWstart':
            {
                'description': 'time start of the backward channel measurement',
                'timezone': str(timezone_netcdf)},
        'timeBWend':
            {
                'description': 'time end of the backward channel measurement',
                'timezone': str(timezone_netcdf)}}

    if not double_ended_flag:
        # single ended measurement
        dt1 = dtFW.astype('timedelta64[s]')

        # start of the forward measurement
        index_time_FWstart = maxTimeIndex - dt1

        # end of the forward measurement
        index_time_FWend = maxTimeIndex

        # center of forward measurement
        index_time_FWmean = maxTimeIndex - dt1 / 2

        coords_zip = [
            ('timestart', index_time_FWstart), ('timeend', index_time_FWend),
            ('time', index_time_FWmean)]

    else:
        # double ended measurement
        dt1 = dtFW.astype('timedelta64[s]')
        dt2 = dtBW.astype('timedelta64[s]')

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

        coords_zip = [
            ('timeFWstart', index_time_FWstart),
            ('timeFWend', index_time_FWend), ('timeFW', index_time_FWmean),
            ('timeBWstart', index_time_BWstart),
            ('timeBWend', index_time_BWend), ('timeBW', index_time_BWmean),
            ('timestart', index_time_FWstart), ('timeend', index_time_BWend),
            ('time', index_time_FWend)]

    if timezone_input_files is not None:
        coords = {
            k: (
                'time', pd.DatetimeIndex(v).tz_localize(
                    tz=timezone_input_files).tz_convert(
                        timezone_netcdf).astype('datetime64[ns]'),
                time_attrs[k])
            for k, v in coords_zip}
    else:
        coords = {
            k: (
                'time', pd.DatetimeIndex(v).tz_convert(timezone_netcdf).astype(
                    'datetime64[ns]'), time_attrs[k])
            for k, v in coords_zip}

    # The units are already stored in the dtype
    coords['acquisitiontimeFW'] = (
        'time', dt1, {
            'description': 'Acquisition time of the forward measurement'})

    if double_ended_flag:
        # The units are already stored in the dtype
        coords['acquisitiontimeBW'] = (
            'time', dt2, {
                'description': 'Acquisition time of the backward measurement'})

    return coords


def ziphandle_to_filepathlist(fh=None, extension=None):
    fnl_ = sorted(fh.namelist())

    fnl = []
    for name in fnl_:
        if name[:1] == '_':
            # private POSIX
            continue

        if fh.getinfo(name).is_dir():
            continue

        if not name.endswith(extension.strip('*')):
            continue

        fnl.append((name, fh))

    return fnl

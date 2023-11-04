import os
from pathlib import Path
from xml.etree import ElementTree

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from dtscalibration.io.utils import coords_time
from dtscalibration.io.utils import dim_attrs
from dtscalibration.io.utils import get_xml_namespace
from dtscalibration.io.utils import open_file
from dtscalibration.io.utils import ziphandle_to_filepathlist


def read_silixa_files(
    filepathlist=None,
    directory=None,
    zip_handle=None,
    file_ext="*.xml",
    timezone_netcdf="UTC",
    silent=False,
    load_in_memory="auto",
    **kwargs,
):
    """Read a folder with measurement files from a device of the Silixa brand. Each measurement file contains
    values for a
    single timestep. Remember to check which timezone you are working in.

    The silixa files are already timezone aware

    Parameters
    ----------
    filepathlist : list of str, optional
        List of paths that point the the silixa files
    directory : str, Path, optional
        Path to folder
    timezone_netcdf : str, optional
        Timezone string of the netcdf file. UTC follows CF-conventions.
    file_ext : str, optional
        file extension of the measurement files
    silent : bool
        If set tot True, some verbose texts are not printed to stdout/screen
    load_in_memory : {'auto', True, False}
        If 'auto' the Stokes data is only loaded to memory for small files
    kwargs : dict-like, optional
        keyword-arguments are passed to DataStore initialization

    Returns:
    -------
    datastore : DataStore
        The newly created datastore.
    """
    assert "timezone_input_files" not in kwargs, (
        "The silixa files are " "already timezone aware"
    )

    if filepathlist is None and zip_handle is None:
        filepathlist = sorted(Path(directory).glob(file_ext))

        # Make sure that the list of files contains any files
        assert (
            len(filepathlist) >= 1
        ), "No measurement files found in provided " "directory: \n" + str(directory)

    elif filepathlist is None and zip_handle:
        filepathlist = ziphandle_to_filepathlist(fh=zip_handle, extension=file_ext)

    # Make sure that the list of files contains any files
    assert len(filepathlist) >= 1, (
        "No measurement files found in provided " "list/directory"
    )

    xml_version = silixa_xml_version_check(filepathlist)

    if xml_version == 4:
        data_vars, coords, attrs = read_silixa_files_routine_v4(
            filepathlist,
            timezone_netcdf=timezone_netcdf,
            silent=silent,
            load_in_memory=load_in_memory,
        )

    elif xml_version in (6, 7, 8):
        data_vars, coords, attrs = read_silixa_files_routine_v6(
            filepathlist,
            xml_version=xml_version,
            timezone_netcdf=timezone_netcdf,
            silent=silent,
            load_in_memory=load_in_memory,
        )

    else:
        raise NotImplementedError(
            "Silixa xml version " + f"{xml_version} not implemented"
        )

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)
    return ds


def silixa_xml_version_check(filepathlist):
    """Function which tests which version of xml files have to be read.

    Parameters
    ----------
    filepathlist

    Returns:
    -------

    """
    sep = ":"
    attrs = read_silixa_attrs_singlefile(filepathlist[0], sep)

    version_string = attrs["customData:SystemSettings:softwareVersion"]

    # Get major version from string. Tested for Ultima v4, v6, v7 XT-DTS v6
    major_version = int(version_string.replace(" ", "").split(":")[-1][0])

    return major_version


def read_silixa_attrs_singlefile(filename, sep):
    """Parameters
    ----------
    filename
    sep

    Returns:
    -------

    """
    import xmltodict

    def metakey(meta, dict_to_parse, prefix):
        """Fills the metadata dictionairy with data from dict_to_parse.
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

        Returns:
        -------

        """
        for key in dict_to_parse:
            if prefix == "":
                prefix_parse = key.replace("@", "")
            else:
                prefix_parse = sep.join([prefix, key.replace("@", "")])

            if prefix_parse == sep.join(("logData", "data")):
                # skip the LAF , ST data
                continue

            if hasattr(dict_to_parse[key], "keys"):
                # Nested dictionaries, flatten hierarchy.
                meta.update(metakey(meta, dict_to_parse[key], prefix_parse))

            elif isinstance(dict_to_parse[key], list):
                # if the key has values for the multiple channels
                for ival, val in enumerate(dict_to_parse[key]):
                    num_key = prefix_parse + "_" + str(ival)
                    meta.update(metakey(meta, val, num_key))
            else:
                meta[prefix_parse] = dict_to_parse[key]

        return meta

    with open_file(filename) as fh:
        doc_ = xmltodict.parse(fh.read())

    if "wellLogs" in doc_.keys():
        doc = doc_["wellLogs"]["wellLog"]
    else:
        doc = doc_["logs"]["log"]

    return metakey({}, doc, "")


def read_silixa_files_routine_v6(  # noqa: MC0001
    filepathlist,
    xml_version=6,
    timezone_netcdf="UTC",
    silent=False,
    load_in_memory="auto",
):
    """Internal routine that reads Silixa files.
    Use dtscalibration.read_silixa_files function instead.

    The silixa files are already timezone aware

    Parameters
    ----------
    load_in_memory
    filepathlist
    timezone_netcdf
    silent

    Returns:
    -------

    """
    # translate names
    tld = {"ST": "st", "AST": "ast", "REV-ST": "rst", "REV-AST": "rast", "TMP": "tmp"}

    # Open the first xml file using ET, get the name space and amount of data

    with open_file(filepathlist[0]) as fh:
        xml_tree = ElementTree.parse(fh)

    namespace = get_xml_namespace(xml_tree.getroot())

    logtree = xml_tree.find(f"./{namespace}log")
    logdata_tree = logtree.find(f"./{namespace}logData")

    # Amount of datapoints is the size of the logdata tree, corrected for
    #  the mnemonic list and unit list
    nx = len(logdata_tree) - 2

    sep = ":"
    ns = {"s": namespace[1:-1]}

    # Obtain metadata from the first file
    attrs = read_silixa_attrs_singlefile(filepathlist[0], sep)

    # Add standardised required attributes
    attrs["isDoubleEnded"] = attrs["customData:isDoubleEnded"]
    double_ended_flag = bool(int(attrs["isDoubleEnded"]))

    attrs["forwardMeasurementChannel"] = attrs["customData:forwardMeasurementChannel"]
    if double_ended_flag:
        attrs["backwardMeasurementChannel"] = attrs[
            "customData:reverseMeasurementChannel"
        ]
    else:
        attrs["backwardMeasurementChannel"] = "N/A"

    # obtain basic data info
    data_item_names = (
        attrs["logData:mnemonicList"].replace(" ", "").strip(" ").split(",")
    )
    nitem = len(data_item_names)

    ntime = len(filepathlist)

    double_ended_flag = bool(int(attrs["isDoubleEnded"]))
    chFW = int(attrs["forwardMeasurementChannel"]) - 1  # zero-based
    if double_ended_flag:
        chBW = int(attrs["backwardMeasurementChannel"]) - 1  # zero-based
    else:
        # no backward channel is negative value. writes better to netcdf
        chBW = -1

    # print summary
    if not silent:
        print(f"{ntime} files were found, each representing a single timestep")
        print(f"{nitem} recorded vars were found: " + ", ".join(data_item_names))
        print(f"Recorded at {nx} points along the cable")

        if double_ended_flag:
            print("The measurement is double ended")
        else:
            print("The measurement is single ended")

    # obtain timeseries from data
    timeseries_loc_in_hierarchy = [
        ("log", "customData", "acquisitionTime"),
        ("log", "customData", "referenceTemperature"),
        ("log", "customData", "probe1Temperature"),
        ("log", "customData", "probe2Temperature"),
        ("log", "customData", "referenceProbeVoltage"),
        ("log", "customData", "probe1Voltage"),
        ("log", "customData", "probe2Voltage"),
        (
            "log",
            "customData",
            "UserConfiguration",
            "ChannelConfiguration",
            "AcquisitionConfiguration",
            "AcquisitionTime",
            "userAcquisitionTimeFW",
        ),
    ]

    if double_ended_flag:
        timeseries_loc_in_hierarchy.append(
            (
                "log",
                "customData",
                "UserConfiguration",
                "ChannelConfiguration",
                "AcquisitionConfiguration",
                "AcquisitionTime",
                "userAcquisitionTimeBW",
            )
        )

    timeseries = {
        item[-1]: dict(loc=item, array=np.zeros(ntime, dtype=np.float32))
        for item in timeseries_loc_in_hierarchy
    }

    # add units to timeseries (unit of measurement)
    for key, item in timeseries.items():
        if f"customData:{key}:uom" in attrs:
            item["uom"] = attrs[f"customData:{key}:uom"]
        else:
            item["uom"] = ""

    # Gather data
    arr_path = "s:" + "/s:".join(["log", "logData", "data"])

    @dask.delayed
    def grab_data_per_file(file_handle):
        """Parameters
        ----------
        file_handle

        Returns:
        -------

        """
        with open_file(file_handle, mode="r") as f_h:
            eltree = ElementTree.parse(f_h)
            arr_el = eltree.findall(arr_path, namespaces=ns)

            if not len(arr_el) == nx:
                raise ValueError(
                    "Inconsistent length of x-dimension"
                    + "\nCheck if files are mixed up, or if the number of "
                    + "data points vary per file."
                )

            # remove the breaks on both sides of the string
            # split the string on the comma
            arr_str = [arr_eli.text[1:-1].split(",") for arr_eli in arr_el]

        return np.array(arr_str, dtype=float)

    data_lst_dly = [grab_data_per_file(fp) for fp in filepathlist]
    data_lst = [
        da.from_delayed(x, shape=(nx, nitem), dtype=float) for x in data_lst_dly
    ]
    data_arr = da.stack(data_lst).T  # .compute()

    # Check whether to compute data_arr (if possible 25% faster)
    data_arr_cnk = data_arr.rechunk({0: -1, 1: -1, 2: "auto"})
    if load_in_memory == "auto" and data_arr_cnk.npartitions <= 5:
        if not silent:
            print("Reading the data from disk")
        data_arr = data_arr_cnk.compute()
    elif load_in_memory:
        if not silent:
            print("Reading the data from disk")
        data_arr = data_arr_cnk.compute()
    else:
        if not silent:
            print("Not reading the data from disk")
        data_arr = data_arr_cnk

    data_vars = {}
    for name, data_arri in zip(data_item_names, data_arr):
        if name == "LAF":
            continue
        if tld[name] in dim_attrs:
            data_vars[tld[name]] = (["x", "time"], data_arri, dim_attrs[tld[name]])
        else:
            raise ValueError(f"Dont know what to do with the {name} data column")

    # Obtaining the timeseries data (reference temperature etc)
    _ts_dtype = [(k, np.float32) for k in timeseries]
    _time_dtype = [
        ("filename_tstamp", np.int64),
        ("minDateTimeIndex", "<U29"),
        ("maxDateTimeIndex", "<U29"),
    ]
    ts_dtype = np.dtype(_ts_dtype + _time_dtype)

    @dask.delayed
    def grab_timeseries_per_file(file_handle, xml_version):
        """Parameters
        ----------
        file_handle

        Returns:
        -------

        """
        with open_file(file_handle, mode="r") as f_h:
            eltree = ElementTree.parse(f_h)

            out = []
            for _, v in timeseries.items():
                # Get all the timeseries data
                if "userAcquisitionTimeFW" in v["loc"]:
                    # requires two namespace searches
                    path1 = "s:" + "/s:".join(v["loc"][:4])
                    val1 = eltree.findall(path1, namespaces=ns)
                    path2 = "s:" + "/s:".join(v["loc"][4:6])
                    val2 = val1[chFW].find(path2, namespaces=ns)
                    out.append(val2.text)

                elif "userAcquisitionTimeBW" in v["loc"]:
                    # requires two namespace searches
                    path1 = "s:" + "/s:".join(v["loc"][:4])
                    val1 = eltree.findall(path1, namespaces=ns)
                    path2 = "s:" + "/s:".join(v["loc"][4:6])
                    val2 = val1[chBW].find(path2, namespaces=ns)
                    out.append(val2.text)

                else:
                    path = "s:" + "/s:".join(v["loc"])
                    val = eltree.find(path, namespaces=ns)
                    out.append(val.text)

            # get all the time related data
            startDateTimeIndex = eltree.find(
                "s:log/s:startDateTimeIndex", namespaces=ns
            ).text
            endDateTimeIndex = eltree.find(
                "s:log/s:endDateTimeIndex", namespaces=ns
            ).text

            if isinstance(file_handle, tuple):
                file_name = os.path.split(file_handle[0])[-1]
            else:
                file_name = os.path.split(file_handle)[-1]

            if xml_version == 6:
                tstamp = np.int64(file_name[10:27])
            elif xml_version == 7:
                tstamp = np.int64(file_name[15:27])
            elif xml_version == 8:
                tstamp = np.int64(file_name[-23:-4].replace(".", ""))
            else:
                raise ValueError(f"Unknown version number: {xml_version}")

            out += [tstamp, startDateTimeIndex, endDateTimeIndex]
        return np.array(tuple(out), dtype=ts_dtype)

    ts_lst_dly = [grab_timeseries_per_file(fp, xml_version) for fp in filepathlist]
    ts_lst = [da.from_delayed(x, shape=tuple(), dtype=ts_dtype) for x in ts_lst_dly]
    ts_arr = da.stack(ts_lst).compute()

    for name in timeseries:
        if name in dim_attrs:
            data_vars[name] = (("time",), ts_arr[name], dim_attrs[name])

        else:
            data_vars[name] = (("time",), ts_arr[name])

    # construct the coordinate dictionary
    if isinstance(filepathlist[0], tuple):
        filename_list = [os.path.split(f)[-1] for f, n2 in filepathlist]
    else:
        filename_list = [os.path.split(f)[-1] for f in filepathlist]

    coords = {
        "x": ("x", data_arr[0, :, 0], dim_attrs["x"]),
        "filename": ("time", filename_list),
        "filename_tstamp": ("time", ts_arr["filename_tstamp"]),
    }

    maxTimeIndex = pd.DatetimeIndex(ts_arr["maxDateTimeIndex"])
    dtFW = ts_arr["userAcquisitionTimeFW"].astype("timedelta64[s]")

    if not double_ended_flag:
        tcoords = coords_time(
            maxTimeIndex,
            timezone_netcdf=timezone_netcdf,
            dtFW=dtFW,
            double_ended_flag=double_ended_flag,
        )
    else:
        dtBW = ts_arr["userAcquisitionTimeBW"].astype("timedelta64[s]")
        tcoords = coords_time(
            maxTimeIndex,
            timezone_netcdf=timezone_netcdf,
            dtFW=dtFW,
            dtBW=dtBW,
            double_ended_flag=double_ended_flag,
        )

    coords.update(tcoords)

    return data_vars, coords, attrs


def read_silixa_files_routine_v4(  # noqa: MC0001
    filepathlist, timezone_netcdf="UTC", silent=False, load_in_memory="auto"
):
    """Internal routine that reads Silixa files.
    Use dtscalibration.read_silixa_files function instead.

    The silixa files are already timezone aware

    Parameters
    ----------
    load_in_memory
    filepathlist
    timezone_netcdf
    silent

    Returns:
    -------

    """
    # translate names
    tld = {"ST": "st", "AST": "ast", "REV-ST": "rst", "REV-AST": "rast", "TMP": "tmp"}

    # Open the first xml file using ET, get the name space and amount of data
    xml_tree = ElementTree.parse(filepathlist[0])
    namespace = get_xml_namespace(xml_tree.getroot())

    logtree = xml_tree.find(f"./{namespace}wellLog")
    logdata_tree = logtree.find(f"./{namespace}logData")

    # Amount of datapoints is the size of the logdata tree
    nx = len(logdata_tree)

    sep = ":"
    ns = {"s": namespace[1:-1]}

    # Obtain metadata from the first file
    attrs = read_silixa_attrs_singlefile(filepathlist[0], sep)

    # Add standardised required attributes
    attrs["isDoubleEnded"] = attrs["customData:isDoubleEnded"]
    double_ended_flag = bool(int(attrs["isDoubleEnded"]))

    attrs["forwardMeasurementChannel"] = attrs["customData:forwardMeasurementChannel"]
    if double_ended_flag:
        attrs["backwardMeasurementChannel"] = attrs[
            "customData:reverseMeasurementChannel"
        ]
    else:
        attrs["backwardMeasurementChannel"] = "N/A"

    chFW = int(attrs["forwardMeasurementChannel"]) - 1  # zero-based
    if double_ended_flag:
        chBW = int(attrs["backwardMeasurementChannel"]) - 1  # zero-based
    else:
        # no backward channel is negative value. writes better to netcdf
        chBW = -1

    # obtain basic data info
    if double_ended_flag:
        data_item_names = [attrs[f"logCurveInfo_{x}:mnemonic"] for x in range(6)]
    else:
        data_item_names = [attrs[f"logCurveInfo_{x}:mnemonic"] for x in range(4)]

    nitem = len(data_item_names)

    ntime = len(filepathlist)

    # print summary
    if not silent:
        print(f"{ntime} files were found, each representing a single timestep")
        print(f"{nitem} recorded vars were found: " + ", ".join(data_item_names))
        print(f"Recorded at {nx} points along the cable")

        if double_ended_flag:
            print("The measurement is double ended")
        else:
            print("The measurement is single ended")

    # obtain timeseries from data
    timeseries_loc_in_hierarchy = [
        ("wellLog", "customData", "acquisitionTime"),
        ("wellLog", "customData", "referenceTemperature"),
        ("wellLog", "customData", "probe1Temperature"),
        ("wellLog", "customData", "probe2Temperature"),
        ("wellLog", "customData", "referenceProbeVoltage"),
        ("wellLog", "customData", "probe1Voltage"),
        ("wellLog", "customData", "probe2Voltage"),
        (
            "wellLog",
            "customData",
            "UserConfiguration",
            "ChannelConfiguration",
            "AcquisitionConfiguration",
            "AcquisitionTime",
            "userAcquisitionTimeFW",
        ),
    ]

    if double_ended_flag:
        timeseries_loc_in_hierarchy.append(
            (
                "wellLog",
                "customData",
                "UserConfiguration",
                "ChannelConfiguration",
                "AcquisitionConfiguration",
                "AcquisitionTime",
                "userAcquisitionTimeBW",
            )
        )

    timeseries = {
        item[-1]: dict(loc=item, array=np.zeros(ntime, dtype=np.float32))
        for item in timeseries_loc_in_hierarchy
    }

    # add units to timeseries (unit of measurement)
    for key, item in timeseries.items():
        if f"customData:{key}:uom" in attrs:
            item["uom"] = attrs[f"customData:{key}:uom"]
        else:
            item["uom"] = ""

    # Gather data
    arr_path = "s:" + "/s:".join(["wellLog", "logData", "data"])

    @dask.delayed
    def grab_data_per_file(file_handle):
        """Parameters
        ----------
        file_handle

        Returns:
        -------

        """
        with open_file(file_handle, mode="r") as f_h:
            eltree = ElementTree.parse(f_h)
            arr_el = eltree.findall(arr_path, namespaces=ns)

            if not len(arr_el) == nx:
                raise ValueError(
                    "Inconsistent length of x-dimension"
                    + "\nCheck if files are mixed up, or if the number of "
                    + "data points vary per file."
                )

            # remove the breaks on both sides of the string
            # split the string on the comma
            arr_str = [arr_eli.text.split(",") for arr_eli in arr_el]
        return np.array(arr_str, dtype=float)

    data_lst_dly = [grab_data_per_file(fp) for fp in filepathlist]
    data_lst = [
        da.from_delayed(x, shape=(nx, nitem), dtype=float) for x in data_lst_dly
    ]
    data_arr = da.stack(data_lst).T  # .compute()

    # Check whether to compute data_arr (if possible 25% faster)
    data_arr_cnk = data_arr.rechunk({0: -1, 1: -1, 2: "auto"})
    if load_in_memory == "auto" and data_arr_cnk.npartitions <= 5:
        if not silent:
            print("Reading the data from disk")
        data_arr = data_arr_cnk.compute()
    elif load_in_memory:
        if not silent:
            print("Reading the data from disk")
        data_arr = data_arr_cnk.compute()
    else:
        if not silent:
            print("Not reading the data from disk")
        data_arr = data_arr_cnk

    data_vars = {}
    for name, data_arri in zip(data_item_names, data_arr):
        if name == "LAF":
            continue

        if tld[name] in dim_attrs:
            data_vars[tld[name]] = (["x", "time"], data_arri, dim_attrs[tld[name]])

        else:
            raise ValueError("Dont know what to do with the" " {name} data column")

    # Obtaining the timeseries data (reference temperature etc)
    _ts_dtype = [(k, np.float32) for k in timeseries]
    _time_dtype = [
        ("filename_tstamp", np.int64),
        ("minDateTimeIndex", "<U29"),
        ("maxDateTimeIndex", "<U29"),
    ]
    ts_dtype = np.dtype(_ts_dtype + _time_dtype)

    @dask.delayed
    def grab_timeseries_per_file(file_handle):
        """Parameters
        ----------
        file_handle

        Returns:
        -------

        """
        with open_file(file_handle, mode="r") as f_h:
            eltree = ElementTree.parse(f_h)

            out = []
            for k, v in timeseries.items():
                # Get all the timeseries data
                if "userAcquisitionTimeFW" in v["loc"]:
                    # requires two namespace searches
                    path1 = "s:" + "/s:".join(v["loc"][:4])
                    val1 = eltree.findall(path1, namespaces=ns)
                    path2 = "s:" + "/s:".join(v["loc"][4:6])
                    val2 = val1[chFW].find(path2, namespaces=ns)
                    out.append(val2.text)

                elif "userAcquisitionTimeBW" in v["loc"]:
                    # requires two namespace searches
                    path1 = "s:" + "/s:".join(v["loc"][:4])
                    val1 = eltree.findall(path1, namespaces=ns)
                    path2 = "s:" + "/s:".join(v["loc"][4:6])
                    val2 = val1[chBW].find(path2, namespaces=ns)
                    out.append(val2.text)

                else:
                    path = "s:" + "/s:".join(v["loc"])
                    val = eltree.find(path, namespaces=ns)
                    out.append(val.text)

            # get all the time related data
            startDateTimeIndex = eltree.find(
                "s:wellLog/s:minDateTimeIndex", namespaces=ns
            ).text
            endDateTimeIndex = eltree.find(
                "s:wellLog/s:maxDateTimeIndex", namespaces=ns
            ).text

            if isinstance(file_handle, tuple):
                file_name = os.path.split(file_handle[0])[-1]
            else:
                file_name = os.path.split(file_handle)[-1]

            tstamp = np.int64(file_name[10:-4])

            out += [tstamp, startDateTimeIndex, endDateTimeIndex]
        return np.array(tuple(out), dtype=ts_dtype)

    ts_lst_dly = [grab_timeseries_per_file(fp) for fp in filepathlist]
    ts_lst = [da.from_delayed(x, shape=tuple(), dtype=ts_dtype) for x in ts_lst_dly]
    ts_arr = da.stack(ts_lst).compute()

    for name in timeseries:
        if name in dim_attrs:
            data_vars[name] = (("time",), ts_arr[name], dim_attrs[name])

        else:
            data_vars[name] = (("time",), ts_arr[name])

    # construct the coordinate dictionary
    coords = {
        "x": ("x", data_arr[0, :, 0], dim_attrs["x"]),
        "filename": ("time", [os.path.split(f)[1] for f in filepathlist]),
        "filename_tstamp": ("time", ts_arr["filename_tstamp"]),
    }

    maxTimeIndex = pd.DatetimeIndex(ts_arr["maxDateTimeIndex"])
    dtFW = ts_arr["userAcquisitionTimeFW"].astype("timedelta64[s]")

    if not double_ended_flag:
        tcoords = coords_time(
            maxTimeIndex,
            timezone_netcdf=timezone_netcdf,
            dtFW=dtFW,
            double_ended_flag=double_ended_flag,
        )
    else:
        dtBW = ts_arr["userAcquisitionTimeBW"].astype("timedelta64[s]")
        tcoords = coords_time(
            maxTimeIndex,
            timezone_netcdf=timezone_netcdf,
            dtFW=dtFW,
            dtBW=dtBW,
            double_ended_flag=double_ended_flag,
        )

    coords.update(tcoords)

    return data_vars, coords, attrs

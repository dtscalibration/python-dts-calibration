import os
import re
import warnings
from pathlib import Path
from xml.etree import ElementTree

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from dtscalibration.io.utils import dim_attrs
from dtscalibration.io.utils import get_xml_namespace
from dtscalibration.io.utils import open_file

dim_attrs_apsensing = dict(dim_attrs)
dim_attrs_apsensing["TEMP"] = dim_attrs_apsensing.pop("tmp")
dim_attrs_apsensing["TEMP"]["name"] = "TEMP"
dim_attrs_apsensing.pop("acquisitionTime")
dim_attrs_apsensing.pop("userAcquisitionTimeFW")
dim_attrs_apsensing.pop("userAcquisitionTimeBW")


def read_apsensing_files(
    filepathlist=None,
    directory=None,
    file_ext="*.xml",
    timezone_input_files="UTC",
    timezone_netcdf="UTC",
    silent=False,
    load_in_memory="auto",
    **kwargs,
):
    """Read a folder with measurement files from a device of the Sensortran
    brand. Each measurement file contains values for a single timestep.
    Remember to check which timezone you are working in.

    Parameters
    ----------
    filepathlist : list of str, optional
        List of paths that point the the silixa files
    directory : str, Path, optional
        Path to folder
    timezone_input_files : str, optional
        Timezone string of the measurement files.
        Remember to check when measurements are taken.
        Also if summertime is used.
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

    Notes:
    ------
    Only XML files are supported for now

    Returns:
    --------
    datastore : DataStore
        The newly created datastore.
    """
    if not file_ext == "*.xml":
        raise NotImplementedError("Only .xml files are supported for now")

    if filepathlist is None:
        filepathlist = sorted(Path(directory).glob(file_ext))

        # Make sure that the list of files contains any files
        assert len(filepathlist) >= 1, (
            "No measurement files found in provided " "directory: \n" + str(directory)
        )

    # Make sure that the list of files contains any files
    assert len(filepathlist) >= 1, (
        "No measurement files found in provided " "list/directory"
    )

    device = apsensing_xml_version_check(filepathlist)

    valid_devices = ["N4386B"]

    if device in valid_devices:
        pass

    else:
        warnings.warn(
            "AP sensing device {device}"
            " has not been tested.\nPlease open an issue on github"
            " and provide an example file"
        )

    data_vars, coords, attrs = read_apsensing_files_routine(
        filepathlist,
        timezone_netcdf=timezone_netcdf,
        timezone_input_files=timezone_input_files,
        silent=silent,
        load_in_memory=load_in_memory,
    )

    # add .tra data if it is available
    tra_exists, tra_filepathlist = check_if_tra_exists(filepathlist)
    if tra_exists:
        print(".tra files exist and will be read")
        data_dict_list = []
        for _, tra_file in enumerate(tra_filepathlist):
            data_dict = read_single_tra_file(tra_file)
            data_dict_list.append(data_dict)
        data_vars = append_to_data_vars_structure(data_vars, data_dict_list)

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)
    return ds


def apsensing_xml_version_check(filepathlist):
    """Function which tests which version of xml files are read.

    Parameters
    ----------
    filepathlist

    Returns:
    --------

    """
    sep = ":"
    attrs, _ = read_apsensing_attrs_singlefile(filepathlist[0], sep)
    deviceid_serialnb = attrs["wellbore:dtsInstalledSystemSet:dtsInstalledSystem:uid"]
    deviceid = deviceid_serialnb.split("-")[0]

    return deviceid


def read_apsensing_files_routine(
    filepathlist,
    timezone_input_files="UTC",
    timezone_netcdf="UTC",
    silent=False,
    load_in_memory="auto",
):
    """Internal routine that reads AP Sensing files.
    Use dtscalibration.read_apsensing_files function instead.

    The AP sensing files are not timezone aware

    Parameters
    ----------
    filepathlist
    timezone_input_files
    timezone_netcdf
    silent
    load_in_memory

    Returns:
    --------

    """
    assert (
        timezone_input_files == "UTC" and timezone_netcdf == "UTC"
    ), "Only UTC timezones supported"

    # translate names
    tld = {"ST": "st", "AST": "ast", "REV-ST": "rst", "REV-AST": "rast", "TEMP": "tmp"}

    # Open the first xml file using ET, get the name space and amount of data
    xml_tree = ElementTree.parse(filepathlist[0])
    namespace = get_xml_namespace(xml_tree.getroot())

    logtree = xml_tree.find(
        (
            "{0}wellSet/{0}well/{0}wellboreSet/{0}wellbore"
            + "/{0}wellLogSet/{0}wellLog"
        ).format(namespace)
    )
    logdata_tree = logtree.find(f"./{namespace}logData")

    # Amount of datapoints is the size of the logdata tree
    nx = len(logdata_tree)

    sep = ":"
    ns = {"s": namespace[1:-1]}

    # Obtain metadata from the first file
    attrs, skip_chars = read_apsensing_attrs_singlefile(filepathlist[0], sep)

    # Add standardised required attributes
    # No example of DE file available
    attrs["isDoubleEnded"] = "0"
    double_ended_flag = bool(int(attrs["isDoubleEnded"]))

    attrs["forwardMeasurementChannel"] = attrs[
        "wellbore:dtsMeasurementSet:dtsMeasurement:connectedToFiber:uidRef"
    ]
    attrs["backwardMeasurementChannel"] = "N/A"

    data_item_names = [
        attrs[f"wellbore:wellLogSet:wellLog:logCurveInfo_{x}:mnemonic"]
        for x in range(0, 4)
    ]

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

    # Gather data
    arr_path = "s:" + "/s:".join(
        [
            "wellSet",
            "well",
            "wellboreSet",
            "wellbore",
            "wellLogSet",
            "wellLog",
            "logData",
            "data",
        ]
    )

    @dask.delayed
    def grab_data_per_file(file_handle):
        """Parameters
        ----------
        file_handle

        Returns:
        --------

        """
        with open_file(file_handle, mode="r") as f_h:
            if skip_chars:
                f_h.read(3)
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
    if load_in_memory == "auto" and data_arr_cnk.npartitions <= 5 or load_in_memory:
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

        if tld[name] in dim_attrs_apsensing:
            data_vars[tld[name]] = (
                ["x", "time"],
                data_arri,
                dim_attrs_apsensing[tld[name]],
            )
        elif name in dim_attrs_apsensing:
            data_vars[tld[name]] = (["x", "time"], data_arri, dim_attrs_apsensing[name])
        else:
            raise ValueError("Dont know what to do with the" + f" {name} data column")

    _time_dtype = [("filename_tstamp", np.int64), ("acquisitionTime", "<U29")]
    ts_dtype = np.dtype(_time_dtype)

    @dask.delayed
    def grab_timeseries_per_file(file_handle):
        """Parameters
        ----------
        file_handle

        Returns:
        --------

        """
        with open_file(file_handle, mode="r") as f_h:
            if skip_chars:
                f_h.read(3)
            eltree = ElementTree.parse(f_h)

            out = []

            # get all the time related data
            creationDate = eltree.find(
                (
                    "{0}wellSet/{0}well/{0}wellboreSet"
                    + "/{0}wellbore/{0}wellLogSet"
                    + "/{0}wellLog/{0}creationDate"
                ).format(namespace)
            ).text

            if isinstance(file_handle, tuple):
                file_name = os.path.split(file_handle[0])[-1]
            else:
                file_name = os.path.split(file_handle)[-1]

            tstamp = np.int64(file_name[-20:-4])

            out += [tstamp, creationDate]
        return np.array(tuple(out), dtype=ts_dtype)

    ts_lst_dly = [grab_timeseries_per_file(fp) for fp in filepathlist]
    ts_lst = [da.from_delayed(x, shape=tuple(), dtype=ts_dtype) for x in ts_lst_dly]
    ts_arr = da.stack(ts_lst).compute()

    data_vars["creationDate"] = (
        ("time",),
        [pd.Timestamp(str(item[1])) for item in ts_arr],
    )

    # construct the coordinate dictionary
    coords = {
        "x": ("x", data_arr[0, :, 0], dim_attrs_apsensing["x"]),
        "filename": ("time", [os.path.split(f)[1] for f in filepathlist]),
        "time": data_vars["creationDate"],
    }

    return data_vars, coords, attrs


def read_apsensing_attrs_singlefile(filename, sep):
    """Parameters
    ----------
    filename
    sep

    Returns:
    --------

    """
    from xml.parsers.expat import ExpatError

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
        --------

        """
        for key in dict_to_parse:
            if prefix == "":
                prefix_parse = key.replace("@", "")
            else:
                prefix_parse = sep.join([prefix, key.replace("@", "")])

            items = ("wellbore", "wellLogSet", "wellLog", "logData", "data")
            if prefix_parse == sep.join(items):
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
        data = fh.read()
        try:
            doc_ = xmltodict.parse(data)
            skip_chars = False
        # the first 3 characters can be weird, skip them
        except ExpatError:
            doc_ = xmltodict.parse(data[3:])
            skip_chars = True

    doc = doc_["WITSMLComposite"]["wellSet"]["well"]["wellboreSet"]

    return metakey({}, doc, ""), skip_chars


def check_if_tra_exists(filepathlist):
    """
    Using AP Sensing N4386B both POSC (.xml) export and trace (.tra) export can be used to log measurements.
    This function checks, whether both export options were turned on simultaneously. All files .xml and .tra
    must be placed in the same directory.

    Parameters
    ----------
    filepathlist : list of str
        List of paths that point the the .xml files

    Notes:
    ------
    All files .xml and .tra must be placed in the same directory.

    Returns:
    --------
    tra_available : boolean,
        True only, when all .xml files have a corresponding .tra file
    ordered_tra_filepathlist . list of str
        if tra_available is True: This list contains a list of filepaths for the
        .tra file. The list is ordered the same as the input .xml filepath list.
    """

    
    directory = Path(filepathlist[0]).parent # create list of .tra files in directory
    sorted_tra_filepathlist = sorted(directory.glob("*.tra"))
    tra_files = "\n".join([file.name for file in sorted_tra_filepathlist]) # make it one big string
    tra_timestamps = set(re.findall(r"(\d{14}).tra", tra_files))  # find 14 digits followed by .tra

    xml_timestamps = "\n".join([file.name for file in filepathlist])
    xml_timestamps = set(re.findall(r"(\d{14}).xml", xml_timestamps)) # note that these are sets now

    diff = xml_timestamps - tra_timestamps
    if len(diff) == len(xml_timestamps): # No tra data - that may be intended --> warning.
        msg = f"Not all .xml files have a matching .tra file.\n Missing are time following timestamps {diff}.  Not loading .tra data."
        warnings.warn(msg)
        return False, []
    
    elif len(diff) > 0: 
        msg = f"Not all .xml files have a matching .tra file.\n Missing are time following timestamps {diff}."
        raise ValueError(msg)
    
    diff = tra_timestamps - xml_timestamps
    if len(diff) > 0:
        msg = f"Not all .tra files have a matching .xml file.\n Missing are time following timestamps {diff}."
        raise ValueError(msg)

    return True, sorted_tra_filepathlist


def parse_tra_numbers(val: str):
    """parsing helper function used by function read_single_tra_file() to determine correct datatype of read string.

    Parameters
    ----------
    val : str
        String value of tra file item

    Returns:
    --------
    val : Value in correct datatype (boolean, int, float and string supported),
    """
    if val == "True":
        return True
    if val == "False":
        return False
    if val.isdigit():
        return int(val)
    try:  # a bit ugly, but sadly there is no string.isfloat() method...
        return float(val)
    except ValueError:
        return val


def read_single_tra_file(tra_filepath):
    """
    Using AP Sensing N4386B both POSC (.xml) export and trace (.tra) export can be used to log measurements.
    This function reads the .tra data and appends it to the dask array, which was read from the POSC export (.xml) file.

    .tra files contain different data then the .xml files from POSC export
        - more metadata
        - log_ratio and loss(attenuation) calculated by device
        - PT100 sensor data (optional only if sensors are connnected to device)


    Parameters
    ----------
    tra_filepathlist : list of str
        List of paths that point the the .tra files
    Notes:
    ------
    more metadata could be read from the .tra file and stored in the dask array

    Returns:
    --------
    data_dict : dict containing time series measured fibre data by distance
                                PT100 reference as float
                                timestamp data
                                other metadata
    """

    with open(tra_filepath) as f:
        file = f.readlines()

    data = [line for line in file if line != "\n"]  # drops out empty lines

    data_dict = {}

    current_section = None
    for line_with_break in data:
        line = line_with_break.replace("\n", "")  # drops out linebreaks
        if line.startswith("["):  # detects new section and sets it as current section
            current_section = line.replace("[", "").replace("]", "")
            data_dict[current_section] = {}
        else:
            content = line.split(";")
            content = [parse_tra_numbers(val) for val in content]
            content_name = content[0]
            if (
                len(content) == 2
            ):  # = metadata & data after trace data (optional sensors and time stamp)
                data_dict[current_section][content_name] = content[1]
            else:  # == trace data containing distance, temperature, logratio, attenuation
                data_dict[current_section][content_name] = tuple(content[1:])

    trace_key = [key for key in data_dict if "Trace." in key][
        0
    ]  # find key of trace n in "Trace.n" is unknown
    data_dict["trace_key"] = trace_key

    return data_dict


def append_to_data_vars_structure(data_vars, data_dict_list):
    """
    append data from .tra files to data_vars structure.
    (The data_vars structure is later on used to initialize the xarray dataset).


    Parameters
    ----------
    data_vars : dictionary containing *.xml data
    data_dict_list: list of dictionaries
                each dictionary in the list contains the data of one .tra file

    Returns:
    --------
    data_vars : dictionary containing *.xml data and *.tra data

    """
    # compose array of format [[value(x1t1).. value(x1tm)]
    #                           ....
    #                           [value(xnt1).. value(xntm)]]
    for idx, data_dict in enumerate(data_dict_list):
        # first get distance, t_by_dts, log_ratio and loss as list from dictionary
        tr_key = data_dict["trace_key"]
        [distance_list, t_by_dts_list, log_ratio_list, loss_list] = [[], [], [], []]
        [
            [
                distance_list.append(data_dict[tr_key][key][0]),
                t_by_dts_list.append(data_dict[tr_key][key][1]),
                log_ratio_list.append(data_dict[tr_key][key][2]),
                loss_list.append(data_dict[tr_key][key][3]),
            ]
            for key in data_dict[tr_key]
            if isinstance(key, int)
        ]

        if idx == 0:
            # initialize numpy arrays
            distance = np.column_stack(np.column_stack(np.array(distance_list)))
            t_by_dts = np.column_stack(np.column_stack(np.array(t_by_dts_list)))
            log_ratio = np.column_stack(np.column_stack(np.array(log_ratio_list)))
            loss = np.column_stack(np.column_stack(np.array(loss_list)))
        else:
            distance = np.concatenate(
                (distance, np.column_stack(np.column_stack(np.array(distance_list)))),
                axis=1,
            )
            t_by_dts = np.concatenate(
                (t_by_dts, np.column_stack(np.column_stack(np.array(t_by_dts_list)))),
                axis=1,
            )
            log_ratio = np.concatenate(
                (log_ratio, np.column_stack(np.column_stack(np.array(log_ratio_list)))),
                axis=1,
            )
            loss = np.concatenate(
                (loss, np.column_stack(np.column_stack(np.array(loss_list)))), axis=1
            )

    # add log_ratio and attenaution to data_vars
    data_vars["log_ratio_by_dts"] = (("x", "time"), log_ratio)
    data_vars["loss_by_dts"] = (("x", "time"), loss)

    # add reference temp data, if they exist
    for idx_ref_temp in range(1, 5):
        if f"Ref.Temperature.Sensor.{idx_ref_temp}" in data_dict[tr_key]:
            ref_temps = []
            for _, data_dict in enumerate(data_dict_list):
                tr_key = data_dict["trace_key"]
                ref_temps.append(
                    data_dict[tr_key][f"Ref.Temperature.Sensor.{idx_ref_temp}"]
                )
            data_vars[f"probe{idx_ref_temp}Temperature"] = (("time",), ref_temps)

    # check if files match by comparing timestamps and dts temperature
    for idx_t in range(0, len(data_dict_list)):
        # check timestamps
        data_dict = data_dict_list[idx_t]
        tr_key = data_dict["trace_key"]
        dd_ts = pd.Timestamp(
            int(data_dict[tr_key]["Date.Year"]),
            int(data_dict[tr_key]["Date.Month"]),
            int(data_dict[tr_key]["Date.Day"]),
            int(data_dict[tr_key]["Time.Hour"]),
            int(data_dict[tr_key]["Time.Minute"]),
            int(data_dict[tr_key]["Time.Second"]),
        )

        err_msg = f"fatal error in allocation of .xml and .tra data.\nxml file {data_vars['creationDate'][1][idx_t]}\ntra file {str(dd_ts)}\n\n"
        if not data_vars["creationDate"][1][idx_t] == dd_ts:
            raise Exception(err_msg)

        # check dts temperature
        for idx_x in [0, 2, 5]:
            if not data_vars["tmp"][1][idx_x][idx_t] == t_by_dts[idx_x][idx_t]:
                # fatal error in allocation of .tra and .xml data
                raise Exception(err_msg)
    return data_vars

import os
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
    -----
    Only XML files are supported for now

    Returns:
    -------
    datastore : DataStore
        The newly created datastore.
    """
    if not file_ext == "*.xml":
        raise NotImplementedError("Only .xml files are supported for now")

    if filepathlist is None:
        filepathlist = sorted(Path(directory).glob(file_ext))

        # Make sure that the list of files contains any files
        assert (
            len(filepathlist) >= 1
        ), "No measurement files found in provided " "directory: \n" + str(directory)

    # Make sure that the list of files contains any files
    assert len(filepathlist) >= 1, (
        "No measurement files found in provided " "list/directory"
    )

    device = apsensing_xml_version_check(filepathlist)

    valid_devices = ["CP320"]

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

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)
    return ds


def apsensing_xml_version_check(filepathlist):
    """Function which tests which version of xml files are read.

    Parameters
    ----------
    filepathlist

    Returns:
    -------

    """
    sep = ":"
    attrs, _ = read_apsensing_attrs_singlefile(filepathlist[0], sep)

    return attrs["wellbore:uid"]


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
    -------

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
        print("%s files were found, each representing a single timestep" % ntime)
        print("%s recorded vars were found: " % nitem + ", ".join(data_item_names))
        print("Recorded at %s points along the cable" % nx)

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
        -------

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
        -------

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
    -------

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
        -------

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

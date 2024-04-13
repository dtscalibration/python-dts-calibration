import fnmatch
import os
import warnings
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr

from dtscalibration.io.utils import coords_time
from dtscalibration.io.utils import dim_attrs
from dtscalibration.io.utils import open_file


def read_sensornet_files(
    filepathlist=None,
    directory=None,
    file_ext="*.ddf",
    timezone_input_files="UTC",
    timezone_netcdf="UTC",
    silent=False,
    add_internal_fiber_length=50.0,
    fiber_length=None,
    **kwargs,
):
    """Read a folder with measurement files. Each measurement file contains
    values for a single timestep. Remember to check which timezone
    you are working in.

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
    add_internal_fiber_length : float
        Set to zero if only the measurements of the fiber connected to the DTS
        system of interest. Set to 50 if you also want to keep the internal
        reference section.
    fiber_length : float
        It is the fiber length between the two connector entering the DTS
        device. If left to `None`, it is approximated with
        `x[-1] - add_internal_fiber_length`.
    kwargs : dict-like, optional
        keyword-arguments are passed to DataStore initialization

    Notes:
    ------
    Compressed sensornet files can not be directly decoded
    because the files are encoded with encoding='windows-1252' instead of
    UTF-8.

    Returns:
    --------
    datastore : DataStore
        The newly created datastore.
    """
    if filepathlist is None:
        # Also look for files in sub-folders
        filepathlist_unsorted = glob(
            os.path.join(directory, "**", file_ext), recursive=True
        )

        # Make sure that the list of files contains any files
        msg = "No measurement files found in provided directory: \n" + str(directory)
        assert len(filepathlist_unsorted) >= 1, msg

        # sort based on dates in filesname. A simple sorted() is not sufficient
        # as month folders do not sort well
        basenames = [os.path.basename(fp) for fp in filepathlist_unsorted]
        dates = ["".join(bn.split(" ")[2:4]) for bn in basenames]
        i_sort = np.argsort(dates)
        filepathlist = [filepathlist_unsorted[i] for i in i_sort]

        # Check measurements are all from same channel
        chno = [bn.split(" ")[1] for bn in basenames]
        assert (
            len(set(chno)) == 1
        ), "Folder contains measurements from multiple channels"

    # Make sure that the list of files contains any files
    assert len(filepathlist) >= 1, (
        "No measurement files found in provided " "list/directory"
    )

    ddf_version = sensornet_ddf_version_check(filepathlist)

    valid_versions = [
        "Halo DTS v1*",
        "ORYX F/W v1.02 Oryx Data Collector v3*",
        "ORYX F/W v4.00 Oryx Data Collector v3*",
        "Sentinel DTS v5*",
    ]

    valid = any([fnmatch.fnmatch(ddf_version, v_) for v_ in valid_versions])

    if valid and (
        fnmatch.fnmatch(ddf_version, "Halo DTS v1*")
        or fnmatch.fnmatch(ddf_version, "Sentinel DTS v5*")
    ):
        flip_reverse_measurements = True
    elif fnmatch.fnmatch(ddf_version, "ORYX F/W v4*"):
        flip_reverse_measurements = False
    else:
        flip_reverse_measurements = False
        warnings.warn(
            f"\n    Sensornet .dff version {ddf_version}"
            " has not been tested.\n    Please open an issue on github"
            " and provide an example file"
        )

    data_vars, coords, attrs = read_sensornet_files_routine_v3(
        filepathlist,
        timezone_netcdf=timezone_netcdf,
        timezone_input_files=timezone_input_files,
        silent=silent,
        add_internal_fiber_length=add_internal_fiber_length,
        fiber_length=fiber_length,
        flip_reverse_measurements=flip_reverse_measurements,
    )

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)
    ds = ds.transpose("x", "time", ...)
    return ds


def sensornet_ddf_version_check(filepathlist):
    """Function which checks and returns the .ddf file version.

    Parameters
    ----------
    filepathlist

    Returns:
    --------
    ddf_version

    """
    # Obtain metadata fro mthe first file
    _, meta = read_sensornet_single(filepathlist[0])

    if "Software version number" in meta:
        version_string = meta["Software version number"]
    else:
        raise ValueError(
            "Software version number could not be detected in .ddf file"
            + "Either file is corrupted or not supported"
        )

    ddf_version = version_string.replace(",", ".")

    return ddf_version


def read_sensornet_single(filename):
    """Parameters
    ----------
    filename

    Returns:
    --------

    """
    headerlength = 26

    # The $\circ$ Celsius symbol is unreadable in utf8
    with open_file(filename, encoding="windows-1252") as fileobject:
        filelength = sum([1 for _ in fileobject])
    datalength = filelength - headerlength

    meta = {}
    with open_file(filename, encoding="windows-1252") as fileobject:
        for ii in range(0, 4):
            fileline = fileobject.readline().split(":\t")
            meta[fileline[0]] = fileline[1].replace("\n", "")

        for ii in range(4, headerlength - 1):
            fileline = fileobject.readline().split("\t")
            meta[fileline[0]] = fileline[1].replace("\n", "").replace(",", ".")

        # data_names =
        fileobject.readline().split("\t")

        if meta["differential loss correction"] == "single-ended":
            data = {
                "x": np.zeros(datalength),
                "tmp": np.zeros(datalength),
                "st": np.zeros(datalength),
                "ast": np.zeros(datalength),
            }

            for ii in range(0, datalength):
                fileline = fileobject.readline().replace(",", ".").split("\t")

                data["x"][ii] = float(fileline[0])
                data["tmp"][ii] = float(fileline[1])
                data["st"][ii] = float(fileline[2])
                data["ast"][ii] = float(fileline[3])

        elif meta["differential loss correction"] == "combined":
            data = {
                "x": np.zeros(datalength),
                "tmp": np.zeros(datalength),
                "st": np.zeros(datalength),
                "ast": np.zeros(datalength),
                "rst": np.zeros(datalength),
                "rast": np.zeros(datalength),
            }

            for ii in range(0, datalength):
                fileline = fileobject.readline().replace(",", ".").split("\t")

                data["x"][ii] = float(fileline[0])
                data["tmp"][ii] = float(fileline[1])
                data["st"][ii] = float(fileline[2])
                data["ast"][ii] = float(fileline[3])
                data["rst"][ii] = float(fileline[4])
                data["rast"][ii] = float(fileline[5])

        else:
            raise ValueError(
                'unknown differential loss correction: "'
                + meta["differential loss correction"]
                + '"'
            )

    meta["default loss term dB per km"] = meta["default loss term (dB/km)"]
    del meta["default loss term (dB/km)"]

    return data, meta


def read_sensornet_files_routine_v3(
    filepathlist,
    timezone_netcdf="UTC",
    timezone_input_files="UTC",
    silent=False,
    add_internal_fiber_length=50.0,
    fiber_length=None,
    flip_reverse_measurements=False,
):
    """Internal routine that reads Sensor files.
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

    Returns:
    --------

    """
    # Obtain metadata from the first file
    data, meta = read_sensornet_single(filepathlist[0])

    # Pop keys from the meta dict which are variable over time
    popkeys = (
        "T ext. ref 1 (°C)",
        "T ext. ref 2 (°C)",
        "T internal ref (°C)",
        "date",
        "time",
        "gamma",
        "k internal",
        "k external",
    )
    [meta.pop(key) for key in popkeys]
    attrs = meta

    # Add standardised required attributes
    if meta["differential loss correction"] == "single-ended":
        attrs["isDoubleEnded"] = "0"
    elif meta["differential loss correction"] == "combined":
        attrs["isDoubleEnded"] = "1"

    double_ended_flag = bool(int(attrs["isDoubleEnded"]))

    attrs["forwardMeasurementChannel"] = meta["forward channel"][-1]
    if double_ended_flag:
        attrs["backwardMeasurementChannel"] = "N/A"
    else:
        attrs["backwardMeasurementChannel"] = meta["reverse channel"][-1]

    # obtain basic data info
    nx = data["x"].size

    ntime = len(filepathlist)

    # chFW = int(attrs['forwardMeasurementChannel']) - 1  # zero-based
    # if double_ended_flag:
    #     chBW = int(attrs['backwardMeasurementChannel']) - 1  # zero-based
    # else:
    #     # no backward channel is negative value. writes better to netcdf
    #     chBW = -1

    # print summary
    if not silent:
        print("%s files were found," % ntime + " each representing a single timestep")
        print("Recorded at %s points along the cable" % nx)

        if double_ended_flag:
            print("The measurement is double ended")
        else:
            print("The measurement is single ended")

    #   Gather data
    # x has already been read. should not change over time
    xraw = data["x"]

    # Define all variables
    referenceTemperature = np.zeros(ntime)
    probe1temperature = np.zeros(ntime)
    probe2temperature = np.zeros(ntime)
    gamma_ddf = np.zeros(ntime)
    k_internal = np.zeros(ntime)
    k_external = np.zeros(ntime)
    acquisitiontimeFW = np.zeros(ntime)
    acquisitiontimeBW = np.zeros(ntime)

    timestamp = [""] * ntime
    ST = np.zeros((nx, ntime))
    AST = np.zeros((nx, ntime))
    TMP = np.zeros((nx, ntime))

    if double_ended_flag:
        REV_ST = np.zeros((nx, ntime))
        REV_AST = np.zeros((nx, ntime))

    for ii in range(ntime):
        data, meta = read_sensornet_single(filepathlist[ii])

        timestamp[ii] = pd.DatetimeIndex([meta["date"] + " " + meta["time"]])[0]
        probe1temperature[ii] = float(meta["T ext. ref 1 (°C)"])
        probe2temperature[ii] = float(meta["T ext. ref 2 (°C)"])
        referenceTemperature[ii] = float(meta["T internal ref (°C)"])
        gamma_ddf[ii] = float(meta["gamma"])
        k_internal[ii] = float(meta["k internal"])
        k_external[ii] = float(meta["k external"])
        acquisitiontimeFW[ii] = float(meta["forward acquisition time"])
        acquisitiontimeBW[ii] = float(meta["reverse acquisition time"])

        ST[:, ii] = data["st"]
        AST[:, ii] = data["ast"]
        TMP[:, ii] = data["tmp"]

        if double_ended_flag:
            REV_ST[:, ii] = data["rst"]
            REV_AST[:, ii] = data["rast"]

    if fiber_length is None and double_ended_flag:
        fiber_length = np.max([0.0, xraw[-1] - add_internal_fiber_length])
    elif fiber_length is None and not double_ended_flag:
        fiber_length = xraw[-1]
    else:
        pass

    assert fiber_length > 0.0, (
        "`fiber_length` is not defined. Use key"
        "word argument in read function." + str(fiber_length)
    )

    fiber_start_index = (np.abs(xraw + add_internal_fiber_length)).argmin()
    fiber_0_index = np.abs(xraw).argmin()
    fiber_1_index = (np.abs(xraw - fiber_length)).argmin()
    fiber_n_indices = fiber_1_index - fiber_0_index
    fiber_n_indices_internal = fiber_0_index - fiber_start_index
    if double_ended_flag:
        fiber_end_index = np.min([xraw.size, fiber_1_index + fiber_n_indices_internal])
    else:
        fiber_end_index = fiber_1_index

    if double_ended_flag:
        if not flip_reverse_measurements:
            # fiber length how the backward channel is aligned
            fiber_length_raw = float(meta["fibre end"])
            fiber_bw_1_index = np.abs(xraw - fiber_length_raw).argmin()
            fiber_bw_end_index = np.min(
                [xraw.size, fiber_bw_1_index + (fiber_end_index - fiber_1_index)]
            )
            fiber_bw_start_index = np.max(
                [0, fiber_bw_1_index - fiber_n_indices - fiber_n_indices_internal]
            )

            if (fiber_end_index - fiber_start_index) == (
                fiber_bw_end_index - fiber_bw_start_index
            ):
                REV_ST = REV_ST[fiber_bw_start_index:fiber_bw_end_index]
                REV_AST = REV_AST[fiber_bw_start_index:fiber_bw_end_index]
            else:
                REV_ST = REV_ST[fiber_start_index:fiber_end_index]
                REV_AST = REV_AST[fiber_start_index:fiber_end_index]

        else:
            # Use the fiber indices from the forward channel
            n_indices_internal_left = fiber_0_index - fiber_start_index
            n_indices_internal_right = np.max([0, fiber_end_index - fiber_1_index])
            n_indices_internal_shortest = np.min(
                [n_indices_internal_left, n_indices_internal_right]
            )
            fiber_start_index = fiber_0_index - n_indices_internal_shortest
            fiber_end_index = (
                fiber_0_index + fiber_n_indices + n_indices_internal_shortest
            )
            REV_ST = REV_ST[fiber_end_index:fiber_start_index:-1]
            REV_AST = REV_AST[fiber_end_index:fiber_start_index:-1]

    x = xraw[fiber_start_index:fiber_end_index]
    TMP = TMP[fiber_start_index:fiber_end_index]
    ST = ST[fiber_start_index:fiber_end_index]
    AST = AST[fiber_start_index:fiber_end_index]

    data_vars = {
        "st": (["x", "time"], ST, dim_attrs["st"]),
        "ast": (["x", "time"], AST, dim_attrs["ast"]),
        "tmp": (["x", "time"], TMP, dim_attrs["tmp"]),
        "probe1Temperature": (
            "time",
            probe1temperature,
            {
                "name": "Probe 1 temperature",
                "description": "reference probe 1 " "temperature",
                "units": r"$^\circ$C",
            },
        ),
        "probe2Temperature": (
            "time",
            probe2temperature,
            {
                "name": "Probe 2 temperature",
                "description": "reference probe 2 " "temperature",
                "units": r"$^\circ$C",
            },
        ),
        "referenceTemperature": (
            "time",
            referenceTemperature,
            {
                "name": "reference temperature",
                "description": "Internal reference " "temperature",
                "units": r"$^\circ$C",
            },
        ),
        "gamma_ddf": (
            "time",
            gamma_ddf,
            {
                "name": "gamma ddf",
                "description": "machine " "calibrated gamma",
                "units": "-",
            },
        ),
        "k_internal": (
            "time",
            k_internal,
            {
                "name": "k internal",
                "description": "machine calibrated " "internal k",
                "units": "-",
            },
        ),
        "k_external": (
            "time",
            k_external,
            {
                "name": "reference temperature",
                "description": "machine calibrated " "external k",
                "units": "-",
            },
        ),
        "userAcquisitionTimeFW": (
            "time",
            acquisitiontimeFW,
            dim_attrs["userAcquisitionTimeFW"],
        ),
        "userAcquisitionTimeBW": (
            "time",
            acquisitiontimeBW,
            dim_attrs["userAcquisitionTimeBW"],
        ),
    }

    if double_ended_flag:
        data_vars["rst"] = (["x", "time"], REV_ST, dim_attrs["rst"])
        data_vars["rast"] = (["x", "time"], REV_AST, dim_attrs["rast"])

    filenamelist = [os.path.split(f)[-1] for f in filepathlist]

    coords = {"x": ("x", x, dim_attrs["x"]), "filename": ("time", filenamelist)}

    dtFW = data_vars["userAcquisitionTimeFW"][1].astype("timedelta64[s]")
    dtBW = data_vars["userAcquisitionTimeBW"][1].astype("timedelta64[s]")
    if not double_ended_flag:
        tcoords = coords_time(
            np.array(timestamp).astype("datetime64[ns]"),
            timezone_netcdf=timezone_netcdf,
            timezone_input_files=timezone_input_files,
            dtFW=dtFW,
            double_ended_flag=double_ended_flag,
        )
    else:
        tcoords = coords_time(
            np.array(timestamp).astype("datetime64[ns]"),
            timezone_netcdf=timezone_netcdf,
            timezone_input_files=timezone_input_files,
            dtFW=dtFW,
            dtBW=dtBW,
            double_ended_flag=double_ended_flag,
        )

    coords.update(tcoords)

    return data_vars, coords, attrs

import struct
from pathlib import Path
from typing import Any
from typing import Union

import numpy as np
import xarray as xr

from dtscalibration.io.utils import coords_time
from dtscalibration.io.utils import dim_attrs


def read_sensortran_files(
    directory: Union[str, Path],
    timezone_input_files: str = "UTC",
    timezone_netcdf: str = "UTC",
    silent: bool = False,
    **kwargs,
) -> xr.Dataset:
    """Read a folder with measurement files from a device of the Sensortran
    brand. Each measurement file contains values for a single timestep. Remember
    to check which timezone you are working in.

    The sensortran files are already timezone aware

    Parameters
    ----------
    directory : str, Path
        Path to folder containing BinaryRawDTS and BinaryTemp files
    timezone_input_files : str, optional
        Timezone string of the measurement files.
        Remember to check when measurements are taken.
        Also if summertime is used.
    timezone_netcdf : str, optional
        Timezone string of the netcdf file. UTC follows CF-conventions.
    silent : bool
        If set tot True, some verbose texts are not printed to stdout/screen
    kwargs : dict-like, optional
        keyword-arguments are passed to DataStore initialization

    Returns
    -------
    DataStore
        The newly created datastore.
    """

    filepathlist_dts = sorted(Path(directory).glob("*BinaryRawDTS.dat"))

    # Make sure that the list of files contains any files
    assert (
        len(filepathlist_dts) >= 1
    ), "No RawDTS measurement files found " "in provided directory: \n" + str(directory)

    filepathlist_temp = [
        Path(str(f).replace("RawDTS", "Temp")) for f in filepathlist_dts
    ]

    for ii, fname in enumerate(filepathlist_dts):
        # Check if corresponding temperature file exists
        if not Path(filepathlist_temp[ii]).is_file():
            raise FileNotFoundError(
                "Could not find BinaryTemp " + f"file corresponding to {fname}"
            )

    version = sensortran_binary_version_check(filepathlist_dts)

    if version == 3:
        data_vars, coords, attrs = read_sensortran_files_routine(
            filepathlist_dts,
            filepathlist_temp,
            timezone_input_files=timezone_input_files,
            timezone_netcdf=timezone_netcdf,
            silent=silent,
        )
    else:
        raise NotImplementedError(
            "Sensortran binary version " + f"{version} not implemented"
        )

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)
    return ds


def sensortran_binary_version_check(filepathlist: list[Path]):
    """Function which tests which version the sensortran binaries are.

    Parameters
    ----------
    filepathlist

    Returns
    -------

    """
    fname = filepathlist[0]

    with fname.open(mode="rb") as f:
        f.read(2)
        version = struct.unpack("<h", f.read(2))[0]

    return version


def read_sensortran_files_routine(
    filepathlist_dts: list[Path],
    filepathlist_temp: list[Path],
    timezone_input_files: str = "UTC",
    timezone_netcdf: str = "UTC",
    silent: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict]:
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
    assert timezone_input_files == "UTC", "The sensortran files are always in UTC time."

    # Obtain metadata from the first file
    data_dts, meta_dts = read_sensortran_single(filepathlist_dts[0])
    data_temp, meta_temp = read_sensortran_single(filepathlist_temp[0])

    attrs = meta_dts

    # Add standardised required attributes
    attrs["isDoubleEnded"] = "0"

    attrs["forwardMeasurementChannel"] = meta_dts["channel_id"] - 1
    attrs["backwardMeasurementChannel"] = "N/A"

    # obtain basic data info
    nx = meta_temp["num_points"]

    ntime = len(filepathlist_dts)

    # print summary
    if not silent:
        print("%s files were found," % ntime + " each representing a single timestep")
        print("Recorded at %s points along the cable" % nx)

        print("The measurement is single ended")

    #   Gather data
    # x has already been read. should not change over time
    x = data_temp["x"]

    # Define all variables
    referenceTemperature = np.zeros(ntime)
    acquisitiontimeFW = np.ones(ntime)

    timestamp = [""] * ntime
    ST = np.zeros((nx, ntime), dtype=np.int32)
    AST = np.zeros((nx, ntime), dtype=np.int32)
    TMP = np.zeros((nx, ntime))

    ST_zero = np.zeros(ntime)
    AST_zero = np.zeros(ntime)

    for ii in range(ntime):
        data_dts, meta_dts = read_sensortran_single(filepathlist_dts[ii])
        data_temp, meta_temp = read_sensortran_single(filepathlist_temp[ii])

        timestamp[ii] = data_dts["time"]

        referenceTemperature[ii] = data_temp["reference_temperature"] - 273.15

        ST[:, ii] = data_dts["st"][:nx]
        AST[:, ii] = data_dts["ast"][:nx]
        # The TMP can vary by 1 or 2 datapoints, dynamically assign the values
        TMP[: meta_temp["num_points"], ii] = data_temp["tmp"][:nx]

        zero_index = (meta_dts["num_points"] - nx) // 2
        ST_zero[ii] = np.mean(data_dts["st"][nx + zero_index :])
        AST_zero[ii] = np.mean(data_dts["ast"][nx + zero_index :])

    data_vars = {
        "st": (["x", "time"], ST, dim_attrs["st"]),
        "ast": (["x", "time"], AST, dim_attrs["ast"]),
        "tmp": (
            ["x", "time"],
            TMP,
            {
                "name": "tmp",
                "description": "Temperature calibrated by device",
                "units": meta_temp["y_units"],
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
        "st_zero": (
            ["time"],
            ST_zero,
            {
                "name": "ST_zero",
                "description": "Stokes zero count",
                "units": meta_dts["y_units"],
            },
        ),
        "ast_zero": (
            ["time"],
            AST_zero,
            {
                "name": "AST_zero",
                "description": "anit-Stokes zero count",
                "units": meta_dts["y_units"],
            },
        ),
        "userAcquisitionTimeFW": (
            "time",
            acquisitiontimeFW,
            dim_attrs["userAcquisitionTimeFW"],
        ),
    }

    coords = {
        "x": (
            "x",
            x,
            {
                "name": "distance",
                "description": "Length along fiber",
                "long_description": "Starting at connector " + "of forward channel",
                "units": "m",
            },
        ),
        "filename": ("time", [f.name for f in filepathlist_dts]),
        "filename_temp": ("time", [f.name for f in filepathlist_temp]),
    }

    dtFW = data_vars["userAcquisitionTimeFW"][1].astype("timedelta64[s]")  # type: ignore

    tcoords = coords_time(
        np.array(timestamp).astype("datetime64[ns]"),
        timezone_netcdf=timezone_netcdf,
        timezone_input_files="UTC",
        dtFW=dtFW,
        double_ended_flag=False,
    )

    coords.update(tcoords)

    return data_vars, coords, attrs


def read_sensortran_single(file: Path) -> tuple[dict, dict]:
    """
    Internal routine that reads a single sensortran file.
    Use dtscalibration.read_sensortran_files function instead.

    Parameters
    ----------
    file

    Returns
    -------
    data, metadata
    """
    import struct
    from datetime import datetime

    meta = {}
    data = {}
    with file.open(mode="rb") as f:
        meta["survey_type"] = struct.unpack("<h", f.read(2))[0]
        meta["hdr_version"] = struct.unpack("<h", f.read(2))[0]
        meta["x_units"] = struct.unpack("<i", f.read(4))[0]
        meta["y_units"] = struct.unpack("<i", f.read(4))[0]
        meta["num_points"] = struct.unpack("<i", f.read(4))[0]
        meta["num_pulses"] = struct.unpack("<i", f.read(4))[0]
        meta["channel_id"] = struct.unpack("<i", f.read(4))[0]
        meta["num_subtraces"] = struct.unpack("<i", f.read(4))[0]
        meta["num_skipped"] = struct.unpack("<i", f.read(4))[0]

        data["reference_temperature"] = struct.unpack("<f", f.read(4))[0]
        data["time"] = datetime.fromtimestamp(struct.unpack("<i", f.read(4))[0])

        meta["probe_name"] = f.read(128).decode("utf-16").split("\x00")[0]

        meta["hdr_size"] = struct.unpack("<i", f.read(4))[0]
        meta["hw_config"] = struct.unpack("<i", f.read(4))[0]

        data_1 = f.read(meta["num_points"] * 4)
        data_2 = f.read(meta["num_points"] * 4)

        if meta["survey_type"] == 0:
            distance = np.frombuffer(data_1, dtype=np.float32)
            temperature = np.frombuffer(data_2, dtype=np.float32)
            data["x"] = distance
            data["tmp"] = temperature

        if meta["survey_type"] == 2:
            ST = np.frombuffer(data_1, dtype=np.int32)
            AST = np.frombuffer(data_2, dtype=np.int32)
            data["st"] = ST
            data["ast"] = AST

    x_units_map = {0: "m", 1: "ft", 2: "n/a"}
    meta["x_units"] = x_units_map[meta["x_units"]]
    y_units_map = {0: "K", 1: "degC", 2: "degF", 3: "counts"}
    meta["y_units"] = y_units_map[meta["y_units"]]

    return data, meta

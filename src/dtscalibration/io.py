import fnmatch
import glob
import inspect
import os
import warnings

import numpy as np
import xarray as xr
from xarray import Dataset

from dtscalibration.io_utils import apsensing_xml_version_check
from dtscalibration.io_utils import read_apsensing_files_routine
from dtscalibration.io_utils import read_sensornet_files_routine_v3
from dtscalibration.io_utils import read_sensortran_files_routine
from dtscalibration.io_utils import read_silixa_files_routine_v4
from dtscalibration.io_utils import read_silixa_files_routine_v6
from dtscalibration.io_utils import sensornet_ddf_version_check
from dtscalibration.io_utils import sensortran_binary_version_check
from dtscalibration.io_utils import silixa_xml_version_check
from dtscalibration.io_utils import ziphandle_to_filepathlist


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
        keyword-arguments are passed to Dataset initialization

    Returns
    -------
    datastore : Dataset
        The newly created datastore.
    """

    assert "timezone_input_files" not in kwargs, (
        "The silixa files are " "already timezone aware"
    )

    if filepathlist is None and zip_handle is None:
        filepathlist = sorted(glob.glob(os.path.join(directory, file_ext)))

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

    ds = Dataset(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)
    return ds


def read_sensortran_files(
    directory, timezone_input_files="UTC", timezone_netcdf="UTC", silent=False, **kwargs
):
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
        keyword-arguments are passed to Dataset initialization

    Returns
    -------
    datastore : Dataset
        The newly created datastore.
    """

    filepathlist_dts = sorted(glob.glob(os.path.join(directory, "*BinaryRawDTS.dat")))

    # Make sure that the list of files contains any files
    assert (
        len(filepathlist_dts) >= 1
    ), "No RawDTS measurement files found " "in provided directory: \n" + str(directory)

    filepathlist_temp = [f.replace("RawDTS", "Temp") for f in filepathlist_dts]

    for ii, fname in enumerate(filepathlist_dts):
        # Check if corresponding temperature file exists
        if not os.path.isfile(filepathlist_temp[ii]):
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

    ds = Dataset(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)
    return ds


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
        keyword-arguments are passed to Dataset initialization

    Notes
    -----
    Only XML files are supported for now

    Returns
    -------
    datastore : Dataset
        The newly created datastore.
    """
    if not file_ext == "*.xml":
        raise NotImplementedError("Only .xml files are supported for now")

    if filepathlist is None:
        filepathlist = sorted(glob.glob(os.path.join(directory, file_ext)))

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

    ds = Dataset(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)
    return ds


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
        keyword-arguments are passed to Dataset initialization

    Notes
    -----
    Compressed sensornet files can not be directly decoded,
    because the files are encoded with encoding='windows-1252' instead of
    UTF-8.

    Returns
    -------
    datastore : Dataset
        The newly created datastore.
    """
    if filepathlist is None:
        # Also look for files in sub-folders
        filepathlist_unsorted = glob.glob(
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

    if valid:
        if fnmatch.fnmatch(ddf_version, "Halo DTS v1*"):
            flip_reverse_measurements = True
        elif fnmatch.fnmatch(ddf_version, "Sentinel DTS v5*"):
            flip_reverse_measurements = True
        else:
            flip_reverse_measurements = False

    else:
        flip_reverse_measurements = False
        warnings.warn(
            f"Sensornet .dff version {ddf_version}"
            " has not been tested.\nPlease open an issue on github"
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

    ds = Dataset(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)
    return ds

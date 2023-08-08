import fnmatch
import glob
import inspect
import os
import warnings

import numpy as np
import xarray as xr

from dtscalibration import DataStore
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


def open_datastore(
    filename_or_obj,
    group=None,
    decode_cf=True,
    mask_and_scale=None,
    decode_times=True,
    concat_characters=True,
    decode_coords=True,
    engine=None,
    chunks=None,
    lock=None,
    cache=None,
    drop_variables=None,
    backend_kwargs=None,
    load_in_memory=False,
    **kwargs,
):
    """Load and decode a datastore from a file or file-like object. Most
    arguments are passed to xarray.open_dataset().

    Parameters
    ----------
    filename_or_obj : str, Path, file or xarray.backends.*DataStore
        Strings and Path objects are interpreted as a path to a netCDF file
        or an OpenDAP URL and opened with python-netCDF4, unless the filename
        ends with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). File-like objects are opened
        with scipy.io.netcdf (only netCDF3 supported).
    group : str, optional
        Path to the netCDF4 group in the given file to open (only works for
        netCDF4 files).
    decode_cf : bool, optional
        Whether to decode these variables, assuming they were saved according
        to CF conventions.
    mask_and_scale : bool, optional
        If True, replace array values equal to `_FillValue` with NA and scale
        values according to the formula `original_values * scale_factor +
        add_offset`, where `_FillValue`, `scale_factor` and `add_offset` are
        taken from variable attributes (if they exist).  If the `_FillValue` or
        `missing_value` attribute contains multiple values a warning will be
        issued and all array values matching one of the multiple values will
        be replaced by NA. mask_and_scale defaults to True except for the
        pseudonetcdf backend.
    decode_times : bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
    decode_coords : bool, optional
        If True, decode the 'coordinates' attribute to identify coordinates in
        the resulting dataset.
    engine : {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'pynio',
    'pseudonetcdf'}, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        'netcdf4'.
    chunks : int or dict, optional
        If chunks is provided, it used to load the new dataset into dask
        arrays. ``chunks={}`` loads the dataset with dask using a single
        chunk for all arrays.
    lock : False, True or threading.Lock, optional
        If chunks is provided, this argument is passed on to
        :py:func:`dask.array.from_array`. By default, a global lock is
        used when reading data from netCDF files with the netcdf4 and h5netcdf
        engines to avoid issues with concurrent access when using dask's
        multithreaded backend.
    cache : bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False. Does not
        change the behavior of coordinates corresponding to dimensions, which
        always load their data from disk into a ``pandas.Index``.
    drop_variables: string or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    backend_kwargs: dictionary, optional
        A dictionary of keyword arguments to pass on to the backend. This
        may be useful when backend options would improve performance or
        allow user control of dataset processing.

    Returns
    -------
    dataset : Dataset
        The newly created dataset.

    See Also
    --------
    xarray.open_dataset
    xarray.load_dataset
    """

    xr_kws = inspect.signature(xr.open_dataset).parameters.keys()

    ds_kwargs = {k: v for k, v in kwargs.items() if k not in xr_kws}

    if chunks is None:
        chunks = {}

    with xr.open_dataset(
        filename_or_obj,
        group=group,
        decode_cf=decode_cf,
        mask_and_scale=mask_and_scale,
        decode_times=decode_times,
        concat_characters=concat_characters,
        decode_coords=decode_coords,
        engine=engine,
        chunks=chunks,
        lock=lock,
        cache=cache,
        drop_variables=drop_variables,
        backend_kwargs=backend_kwargs,
    ) as ds_xr:
        ds = DataStore(
            data_vars=ds_xr.data_vars,
            coords=ds_xr.coords,
            attrs=ds_xr.attrs,
            **ds_kwargs,
        )

        # to support deprecated st_labels
        ds = ds.rename_labels(assertion=False)

        if load_in_memory:
            if "cache" in kwargs:
                raise TypeError("cache has no effect in this context")
            return ds.load()

        else:
            return ds


def open_mf_datastore(
    path=None, paths=None, combine="by_coords", load_in_memory=False, **kwargs
):
    """
    Open a datastore from multiple netCDF files. This script assumes the
    datastore was split along the time dimension. But only variables with a
    time dimension should be concatenated in the time dimension. Other
    options from xarray do not support this.

    Parameters
    ----------
    combine : {'by_coords', 'nested'}, optional
        Leave it at by_coords
    path : str
        A file path to the stored netcdf files with an asterisk in the
        filename to list all. Ensure you have leading zeros in the file
        numbering.
    paths : list
        Define you own list of file paths.
    Returns
    -------
    dataset : Dataset
        The newly created dataset.
    """
    from xarray.backends.api import open_mfdataset

    if paths is None:
        paths = sorted(glob.glob(path))
        assert paths, "No files match found with: " + path

    with open_mfdataset(paths=paths, combine=combine, **kwargs) as xds:
        ds = DataStore(data_vars=xds.data_vars, coords=xds.coords, attrs=xds.attrs)

        # to support deprecated st_labels
        ds = ds.rename_labels(assertion=False)

        if load_in_memory:
            if "cache" in kwargs:
                raise TypeError("cache has no effect in this context")
            return ds.load()

        else:
            return ds


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

    Returns
    -------
    datastore : DataStore
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
            "Silixa xml version " + "{0} not implemented".format(xml_version)
        )

    ds = DataStore(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)
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
        keyword-arguments are passed to DataStore initialization

    Returns
    -------
    datastore : DataStore
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
                "Could not find BinaryTemp " + "file corresponding to {}".format(fname)
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
            "Sensortran binary version " + "{0} not implemented".format(version)
        )

    ds = DataStore(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)
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
        keyword-arguments are passed to DataStore initialization

    Notes
    -----
    Only XML files are supported for now

    Returns
    -------
    datastore : DataStore
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
            "AP sensing device "
            '"{0}"'.format(device)
            + " has not been tested.\nPlease open an issue on github"
            + " and provide an example file"
        )

    data_vars, coords, attrs = read_apsensing_files_routine(
        filepathlist,
        timezone_netcdf=timezone_netcdf,
        timezone_input_files=timezone_input_files,
        silent=silent,
        load_in_memory=load_in_memory,
    )

    ds = DataStore(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)
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
        keyword-arguments are passed to DataStore initialization

    Notes
    -----
    Compressed sensornet files can not be directly decoded,
    because the files are encoded with encoding='windows-1252' instead of
    UTF-8.

    Returns
    -------
    datastore : DataStore
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
            "Sensornet .dff version "
            '"{0}"'.format(ddf_version)
            + " has not been tested.\nPlease open an issue on github"
            + " and provide an example file"
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

    ds = DataStore(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)
    return ds

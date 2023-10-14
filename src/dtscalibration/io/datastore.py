import glob
import inspect

import xarray as xr

from dtscalibration import DataStore


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

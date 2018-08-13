# coding=utf-8
import glob
import inspect
import os

import numpy as np
import xarray as xr
import yaml

from dtscalibration.datastore_utils import coords_time
from dtscalibration.datastore_utils import grab_data


class DataStore(xr.Dataset):
    """To load data from a file or file-like object, use the `open_datastore`
        function.

        Parameters
        ----------
        data_vars : dict-like, optional
            A mapping from variable names to :py:class:`~xarray.DataArray`
            objects, :py:class:`~xarray.Variable` objects or tuples of the
            form ``(dims, data[, attrs])`` which can be used as arguments to
            create a new ``Variable``. Each dimension must have the same length
            in all variables in which it appears.
        coords : dict-like, optional
            Another mapping in the same form as the `variables` argument,
            except the each item is saved on the datastore as a "coordinate".
            These variables have an associated meaning: they describe
            constant/fixed/independent quantities, unlike the
            varying/measured/dependent quantities that belong in `variables`.
            Coordinates values may be given by 1-dimensional arrays or scalars,
            in which case `dims` do not need to be supplied: 1D arrays will be
            assumed to give index values along the dimension with the same
            name.
        attrs : dict-like, optional
            Global attributes to save on this datastore.
        sections : dict, optional
            Sections for calibration. The dictionary should contain key-var couples
            in which the key is the name of the calibration temp time series. And
            the var is a list of slice objects as 'slice(start, stop)'; start and
            stop in meter (float).
        compat : {'broadcast_equals', 'equals', 'identical'}, optional
            String indicating how to compare variables of the same name for
            potential conflicts when initializing this datastore:

            - 'broadcast_equals': all values must be equal when variables are
              broadcast against each other to ensure common dimensions.
            - 'equals': all values and dimensions must be the same.
            - 'identical': all values, dimensions and attributes must be the
              same.
        """

    def __init__(self, *args, **kwargs):
        super(DataStore, self).__init__(*args, **kwargs)

        if '_sections' not in self.attrs:
            self.attrs['_sections'] = yaml.dump(None)

        if 'sections' in kwargs:
            self.sections = kwargs['sections']

    @property
    def sections(self):
        assert hasattr(self, '_sections'), 'first set the sections'
        return yaml.load(self.attrs['_sections'])

    @sections.setter
    def sections(self, sections):
        if sections:
            assert isinstance(sections, dict)

            for k, v in sections.items():
                for vi in v:
                    assert isinstance(vi, slice)

        self.attrs['_sections'] = yaml.dump(sections)
        pass

    @sections.deleter
    def sections(self):
        self.sections = None
        pass

    def calibration_single_ended(self,
                                 sections=None,
                                 st_label=None,
                                 ast_label=None,
                                 store_c='c',
                                 store_gamma='gamma',
                                 store_alphaint='alphaint',
                                 store_alpha='alpha',
                                 store_IFW_var='IF_var',
                                 store_resid_IFW='errF',
                                 variance_suffix='_var',
                                 method='single1'):

        if sections:
            self.sections = sections

        self.check_dims(['st_label', 'ast_label'])
        pass

    def calibration_double_ended(self,
                                 sections=None,
                                 st_label=None,
                                 ast_label=None,
                                 rst_label=None,
                                 rast_label=None,
                                 store_c='c',
                                 store_gamma='gamma',
                                 store_alphaint='alphaint',
                                 store_alpha='alpha',
                                 store_IFW_var='IF_var',
                                 store_IBW_var='IB_var',
                                 store_resid_IFW='errF',
                                 store_resid_IBW='errB',
                                 variance_suffix='_var',
                                 method='double1'):

        if sections:
            self.sections = sections

        self.check_dims(['st_label', 'ast_label', 'rst_label', 'rast_label'])

        if method == 'double1':
            nt, z, p0_, err, errFW, errBW, var = self.calibration_double_ended_calc(
                st_label, ast_label, rst_label, rast_label)

            p0 = p0_[0]
            gamma = p0[0]
            alphaint = p0[1]
            c = p0[2:nt + 2]
            alpha = p0[nt + 2:]

            no = len(alpha)
            ddof = nt + 2 + no

            # Estimate of the standard error - sqrt(diag of the COV matrix) - is not squared
            p0var = var
            gammavar = p0var[0]
            alphaintvar = p0var[1]
            cvar = p0var[2:nt + 2]
            alphavar = p0var[nt + 2:]

        else:
            raise ValueError('Choose a valid method')

        # store calibration parameters in DataStore
        self[store_gamma] = (tuple(), gamma)
        self[store_alphaint] = (tuple(), alphaint)
        self[store_alpha] = (('x',), alpha)
        self[store_c] = (('time',), c)

        # store variances in DataStore
        self[store_gamma + variance_suffix] = (tuple(), gammavar)
        self[store_alphaint + variance_suffix] = (tuple(), alphaintvar)
        self[store_alpha + variance_suffix] = (('x',), alphavar)
        self[store_c + variance_suffix] = (('time',), cvar)

        # deal with FW
        to_dim = self[st_label].dims

        tempF_data = gamma / \
            (np.log(self[st_label].data / self[ast_label].data)
             + c + alpha[:, None]) - 273.15
        self['TMPF'] = (to_dim, tempF_data)

        # deal with BW
        tempB_data = gamma / \
            (np.log(self[rst_label].data / self[rast_label].data)
             + c - alpha[:, None] + alphaint) - 273.15
        self['TMPB'] = (to_dim, tempB_data)

        iz = np.argsort(z)  # What does this do?
        self.coords['errz'] = z[iz]
        self[store_resid_IFW] = (('errz', to_dim), errFW[iz])
        self[store_IFW_var] = (tuple(), errFW.std(ddof=ddof) ** 2)
        self[store_resid_IBW] = (('errz', to_dim), errBW[iz])
        self[store_IBW_var] = (tuple(), errBW.std(ddof=ddof) ** 2)

        self.Tvar_double_ended(store_IBW_var, store_IFW_var, store_alpha,
                               store_c, store_gamma)

        pass

    def check_dims(self, labels):
        """
        Compare the dimensions of different labels (e.g., 'ST', 'REV-ST').
        If a calculation is performed and the dimensions do not agree, the answers don't make
        sense and the matrices are broadcasted and the memory usage will explode.

        Parameters
        ----------
        labels : iterable
            An iterable with labels

        Returns
        -------

        """
        assert len(labels) > 1

        for li in labels[1:]:
            assert self[labels[0]].dims == self[li].dims

        pass


def open_datastore(filename_or_obj, **kwargs):
    """Load and decode a datastore from a file or file-like object
    into a DataStore object.

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
    autoclose : bool, optional
        If True, automatically close files to avoid OS Error of too many files
        being open.  However, this option doesn't work with streams, e.g.,
        BytesIO.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
    decode_coords : bool, optional
        If True, decode the 'coordinates' attribute to identify coordinates in
        the resulting datastore.
    engine : {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'pynio', 'pseudonetcdf'}, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        'netcdf4'.
    chunks : int or dict, optional
        If chunks is provided, it used to load the new datastore into dask
        arrays. ``chunks={}`` loads the datastore with dask using a single
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
        datastore. This may be useful to drop variables with problems or
        inconsistent values.
    backend_kwargs: dictionary, optional
        A dictionary of keyword arguments to pass on to the backend. This
        may be useful when backend options would improve performance or
        allow user control of datastore processing.
    sections : dict, optional
        Sections for calibration. The dictionary should contain key-var couples
        in which the key is the name of the calibration temp time series. And
        the var is a list of slice objects as 'slice(start, stop)'; start and
        stop in meter (float).
    Returns
    -------
    datastore : DataStore
        The newly created datastore.
    See Also
    --------
    read_xml_dir
    """

    xr_kws = inspect.signature(xr.open_dataset).parameters.keys()

    xr_kwargs = {k: v for k, v in kwargs.items() if k in xr_kws}
    ds_kwargs = {k: v for k, v in kwargs.items() if k not in xr_kws}

    ds_xr = xr.open_dataset(filename_or_obj, **xr_kwargs)

    ds = DataStore(data_vars=ds_xr.data_vars,
                   coords=ds_xr.coords,
                   attrs=ds_xr.attrs,
                   **ds_kwargs)
    return ds


def read_xml_dir(filepath,
                 timezone_netcdf='UTC',
                 timezone_ultima_xml='Europe/Amsterdam',
                 file_ext='*.xml',
                 **kwargs):
    """Read a folder with measurement files. Each measurement file contains values for a
    single timestep. Remember to check which timezone you are working in.

    Parameters
    ----------
    filepath : str, Path
        Path to folder
    timezone_netcdf : str, optional
        Timezone string of the netcdf file. UTC follows CF-conventions.
    timezone_ultima_xml : str, optional
        Timezone string of the measurement files. Remember to check when measurements are taken.
        Also if summertime is used.
    file_ext : str, optional
        file extension of the measurement files
    kwargs : dict-like, optional
        keyword-arguments are passed to DataStore initialization

    Returns
    -------
    datastore : DataStore
        The newly created datastore.
    """
    log_attrs = {
        'x':                     {
            'description':      'Length along fiber',
            'long_describtion': 'Starting at connector of forward channel',
            'units':            'm'},
        'TMP':                   {
            'description': 'temperature calibrated by device',
            'units':       'degC'},
        'ST':                    {
            'description': 'Stokes intensity',
            'units':       '?'},
        'AST':                   {
            'description': 'anti-Stokes intensity',
            'units':       '?'},
        'REV-ST':                {
            'description': 'reverse Stokes intensity',
            'units':       '?'},
        'REV-AST':               {
            'description': 'reverse anti-Stokes intensity',
            'units':       '?'},
        'acquisitionTime':       {
            'description':      'Measurement duration of forward channel',
            'long_describtion': 'Actual measurement duration of forward channel',
            'units':            'seconds'},
        'userAcquisitionTimeFW': {
            'description':      'Measurement duration of forward channel',
            'long_describtion': 'Desired measurement duration of forward channel',
            'units':            'seconds'},
        'userAcquisitionTimeBW': {
            'description':      'Measurement duration of backward channel',
            'long_describtion': 'Desired measurement duration of backward channel',
            'units':            'seconds'},
        }

    filepathlist = sorted(glob.glob(os.path.join(filepath, file_ext)))
    filenamelist = [os.path.basename(path) for path in filepathlist]
    assert len(filepathlist) >= 1, 'No measurement files with extension {} found in {}'.format(
        file_ext, filepath)

    array, timearr, meta, extra = grab_data(filepathlist)

    coords = {
        'x':        ('x', array['LAF'][:, 0], log_attrs['x']),
        'filename': ('time', filenamelist)}
    tcoords = coords_time(extra, timearr,
                          timezone_netcdf=timezone_netcdf,
                          timezone_ultima_xml=timezone_ultima_xml)
    coords.update(tcoords)

    dataset_dict = {}
    for name in array.dtype.names:
        if name in ['TMP', 'ST', 'AST', 'REV-ST', 'REV-AST']:
            dataset_dict[name] = (['x', 'time'], array[name], log_attrs[name])

        elif name == 'LAF':
            continue

        else:
            print(name)
            assert 0

    for key, item in extra.items():
        if key in log_attrs:
            dataset_dict[key] = (['time'], item['array'], log_attrs[key])

        else:
            dataset_dict[key] = (['time'], item['array'])

    ds = DataStore(data_vars=dataset_dict,
                   coords=coords,
                   attrs=meta,
                   **kwargs)

    return ds

# filepath = os.path.join('..', '..', 'tests', 'data')
# ds = read_xml_dir(filepath,
#                   timezone_netcdf='UTC',
#                   timezone_ultima_xml='Europe/Amsterdam',
#                   file_ext='*.xml')

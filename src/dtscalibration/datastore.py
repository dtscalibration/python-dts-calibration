import fnmatch
import glob
import inspect
import os
import warnings
from typing import Dict
from typing import List

import dask
import dask.array as da
import numpy as np
import scipy.sparse as sp
import scipy.stats as sst
import xarray as xr
import yaml
from scipy.optimize import minimize
from scipy.sparse import linalg as ln

from .calibrate_utils import calc_alpha_double
from .calibrate_utils import calibration_double_ended_solver
from .calibrate_utils import calibration_single_ended_solver
from .calibrate_utils import match_sections
from .calibrate_utils import wls_sparse
from .calibrate_utils import wls_stats
from .datastore_utils import check_timestep_allclose
from .io import _dim_attrs
from .io import apsensing_xml_version_check
from .io import read_apsensing_files_routine
from .io import read_sensornet_files_routine_v3
from .io import read_sensortran_files_routine
from .io import read_silixa_files_routine_v4
from .io import read_silixa_files_routine_v6
from .io import sensornet_ddf_version_check
from .io import sensortran_binary_version_check
from .io import silixa_xml_version_check
from .io import ziphandle_to_filepathlist

dtsattr_namelist = ['double_ended_flag']
dim_attrs = {k: v for kl, v in _dim_attrs.items() for k in kl}
warnings.filterwarnings(
    'ignore',
    message='xarray subclass DataStore should explicitly define __slots__')


# pylint: disable=W605
class DataStore(xr.Dataset):
    """The data class that stores the measurements, contains calibration
    methods to relate Stokes and anti-Stokes to temperature. The user should
    never initiate this class directly, but use read_xml_dir or open_datastore
    functions instead.

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
        sections : Dict[str, List[slice]], optional
            Sections for calibration. The dictionary should contain key-var
            couples in which the key is the name of the calibration temp time
            series. And the var is a list of slice objects as 'slice(start,
            stop)'; start and stop in meter (float).
        compat : {'broadcast_equals', 'equals', 'identical'}, optional
            String indicating how to compare variables of the same name for
            potential conflicts when initializing this datastore:
            - 'broadcast_equals': all values must be equal when variables are
              broadcast against each other to ensure common dimensions.
            - 'equals': all values and dimensions must be the same.
            - 'identical': all values, dimensions and attributes must be the
              same.

        See Also
        --------
        dtscalibration.read_xml_dir : Load measurements stored in XML-files
        dtscalibration.open_datastore : Load (calibrated) measurements from
        netCDF-like file
        """
    def __init__(self, *args, autofill_dim_attrs=True, **kwargs):
        super().__init__(*args, **kwargs)

        # check order of the dimensions of the data_vars
        # first 'x' (if in initiated DataStore), then 'time', then the rest
        ideal_dim = []  # perfect order dims
        all_dim = list(self.dims)

        if all_dim:
            if 'x' in all_dim:
                ideal_dim.append('x')
                all_dim.pop(all_dim.index('x'))

            time_dim = self.get_time_dim()
            if time_dim:
                if time_dim in all_dim:
                    ideal_dim.append(time_dim)
                    all_dim.pop(all_dim.index(time_dim))

                ideal_dim += all_dim

                for name, var in self._variables.items():
                    var_dims = tuple(
                        dim for dim in ideal_dim if dim in (var.dims + (...,)))
                    self._variables[name] = var.transpose(*var_dims)

        if 'trans_att' not in self.coords:
            self.set_trans_att(trans_att=[])

        # Get attributes from dataset
        for arg in args:
            if isinstance(arg, xr.Dataset):
                self.attrs = arg.attrs

        # Add attributes to loaded dimensions
        if autofill_dim_attrs:
            for name, data_arri in self.coords.items():
                if name in dim_attrs and not self.coords[name].attrs:
                    self.coords[name].attrs = dim_attrs[name]

        if '_sections' not in self.attrs:
            self.attrs['_sections'] = yaml.dump(None)

        if 'sections' in kwargs:
            self.sections = kwargs['sections']

        pass

    def __repr__(self):
        # __repr__ from xarray is used and edited.
        #   'xarray' is prepended. so we remove it and add 'dtscalibration'
        s = xr.core.formatting.dataset_repr(self)
        name_module = type(self).__name__
        preamble_new = u'<dtscalibration.%s>' % name_module

        # Add sections to new preamble
        preamble_new += '\nSections:'
        if hasattr(self, '_sections') and self.sections:
            preamble_new += '\n'

            if 'units' in self.x:
                unit = self.x.units
            else:
                unit = ''

            for k, v in self.sections.items():
                preamble_new += '    {0: <23}'.format(k)

                # Compute statistics reference section timeseries
                sec_stat = '({0:6.2f}'.format(float(self[k].mean()))
                sec_stat += ' +/-{0:5.2f}'.format(float(self[k].std()))
                sec_stat += u'\N{DEGREE SIGN}C)\t'
                preamble_new += sec_stat

                # print sections
                vl = [
                    '{0:.2f}{2} - {1:.2f}{2}'.format(vi.start, vi.stop, unit)
                    for vi in v]
                preamble_new += ' and '.join(vl) + '\n'

        else:
            preamble_new += 18 * ' ' + '()\n'

        # add new preamble to the remainder of the former __repr__
        len_preamble_old = 8 + len(name_module) + 2

        # untill the attribute listing
        attr_index = s.find('Attributes:')

        # abbreviate attribute listing
        attr_list_all = s[attr_index:].split(sep='\n')
        if len(attr_list_all) > 10:
            s_too_many = ['\n.. and many more attributes. See: ds.attrs']
            attr_list = attr_list_all[:10] + s_too_many
        else:
            attr_list = attr_list_all

        s_out = (
            preamble_new + s[len_preamble_old:attr_index]
            + '\n'.join(attr_list))

        # return new __repr__
        return s_out

    # noinspection PyIncorrectDocstring
    @property
    def sections(self):
        """
        Define calibration sections. Each section requires a reference
        temperature time series, such as the temperature measured by an
        external temperature sensor. They should already be part of the
        DataStore object.

        Please look at the example notebook on `sections` if you encounter
        difficulties.

        Parameters
        ----------
        sections : Dict[str, List[slice]]
            Sections are defined in a dictionary with its keywords of the
            names of the reference
            temperature time series. Its values are lists of slice objects,
            where each slice object
            is a stretch.
        Returns
        -------

        """
        if '_sections' not in self.attrs:
            self.attrs['_sections'] = yaml.dump(None)
        return yaml.load(self.attrs['_sections'], Loader=yaml.UnsafeLoader)

    @sections.setter
    def sections(self, sections: Dict[str, List[slice]]):
        sections_fix_slice_fixed = None

        if sections:
            assert isinstance(sections, dict)

            # be less restrictive for capitalized labels
            # find lower cases label
            labels = np.reshape(
                [[s.lower(), s] for s in self.data_vars.keys()],
                (-1,)).tolist()

            sections_fix = dict()
            for k, v in sections.items():
                if k.lower() in labels:
                    i_lower_case = labels.index(k.lower())
                    i_normal_case = i_lower_case + 1
                    k_normal_case = labels[i_normal_case]
                    sections_fix[k_normal_case] = v
                else:
                    assert k in self.data_vars, 'The keys of the ' \
                                                'sections-dictionary should ' \
                                                'refer to a valid timeserie ' \
                                                'already stored in ' \
                                                'ds.data_vars '

            sections_fix_slice_fixed = dict()

            for k, v in sections_fix.items():
                assert isinstance(v, (list, tuple)), \
                    'The values of the sections-dictionary ' \
                    'should be lists of slice objects.'

                for vi in v:
                    assert isinstance(vi, slice), \
                        'The values of the sections-dictionary should ' \
                        'be lists of slice objects.'

                    assert self.x.sel(x=vi).size > 0, \
                        f'Better define the {k} section. You tried {vi}, ' \
                        'which is out of reach'

                # sorted stretches
                stretch_unsort = [
                    slice(float(vi.start), float(vi.stop)) for vi in v]
                stretch_start = [i.start for i in stretch_unsort]
                stretch_i_sorted = np.argsort(stretch_start)
                sections_fix_slice_fixed[k] = [
                    stretch_unsort[i] for i in stretch_i_sorted]

            # Prevent overlapping slices
            ix_sec = self.ufunc_per_section(
                sections=sections_fix_slice_fixed,
                x_indices=True,
                calc_per='all')
            assert np.unique(ix_sec).size == ix_sec.size, \
                "The sections are overlapping"

        self.attrs['_sections'] = yaml.dump(sections_fix_slice_fixed)
        pass

    @sections.deleter
    def sections(self):
        self.sections = None
        pass

    @property
    def is_double_ended(self):
        """
        Whether or not the data is loaded from a double-ended setup.

        Returns
        -------

        """
        if 'isDoubleEnded' in self.attrs:
            return bool(int(self.attrs['isDoubleEnded']))
        elif 'customData:isDoubleEnded' in self.attrs:
            # backward compatible to when only silixa files were supported
            return bool(int(self.attrs['customData:isDoubleEnded']))
        else:
            assert 0

    @is_double_ended.setter
    def is_double_ended(self, flag: bool):
        self.attrs['isDoubleEnded'] = flag
        pass

    @property
    def chfw(self):
        """
        Zero based channel index of the forward measurements

        Returns
        -------

        """
        return int(self.attrs['forwardMeasurementChannel']) - 1  # zero-based

    @property
    def chbw(self):
        """
        Zero based channel index of the backward measurements

        Returns
        -------

        """
        if self.is_double_ended:
            return int(
                self.attrs['reverseMeasurementChannel']) - 1  # zero-based
        else:
            return None

    @property
    def channel_configuration(self):
        """
        Renaming conversion dictionary

        Returns
        -------

        """
        d = {
            'chfw':
                {
                    'st_label': 'st',
                    'ast_label': 'ast',
                    'acquisitiontime_label': 'userAcquisitionTimeFW',
                    'time_start_label': 'timeFWstart',
                    'time_label': 'timeFW',
                    'time_end_label': 'timeFWend'},
            'chbw':
                {
                    'st_label': 'rst',
                    'ast_label': 'rast',
                    'acquisitiontime_label': 'userAcquisitionTimeBW',
                    'time_start_label': 'timeBWstart',
                    'time_label': 'timeBW',
                    'time_end_label': 'timeBWend'}}
        return d

    @property
    def timeseries_keys(self):
        """
        Returns the keys of all timeseires that can be used for calibration.
        """
        time_dim = self.get_time_dim()
        return [k for k, v in self.data_vars.items() if v.dims == (time_dim,)]

    def resample_datastore(
            self,
            how,
            freq=None,
            dim=None,
            skipna=None,
            closed=None,
            label=None,
            origin='start_day',
            offset=None,
            keep_attrs=True,
            **indexer):
        """Returns a resampled DataStore. Always define the how.
        Handles both downsampling and upsampling. If any intervals contain no
        values from the original object, they will be given the value ``NaN``.
        Parameters
        ----------
        freq
        dim
        how : str
            Any function that is available via groupby. E.g., 'mean'
            http://pandas.pydata.org/pandas-docs/stable/groupby.html#groupby
            -dispatch
        skipna : bool, optional
            Whether to skip missing values when aggregating in downsampling.
        closed : 'left' or 'right', optional
            Side of each interval to treat as closed.
        label : 'left or 'right', optional
            Side of each interval to use for labeling.
        base : int, optional
            For frequencies that evenly subdivide 1 day, the "origin" of the
            aggregated intervals. For example, for '24H' frequency, base could
            range from 0 through 23.
        keep_attrs : bool, optional
            If True, the object's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **indexer : {dim: freq}
            Dictionary with a key indicating the dimension name to resample
            over and a value corresponding to the resampling frequency.
        Returns
        -------
        resampled : same type as caller
            This object resampled.
            """
        import pandas as pd
        from xarray.core.dataarray import DataArray

        RESAMPLE_DIM = '__resample_dim__'

        if (freq and indexer) or (dim and indexer):
            raise TypeError(
                "If passing an 'indexer' then 'dim' "
                "and 'freq' should not be used")

        if indexer:
            dim, freq = indexer.popitem()

        if isinstance(dim, str):
            dim = self[dim]
        else:
            raise TypeError(
                "Dimension name should be a string; "
                "was passed %r" % dim)

        if how is None:
            how = 'mean'

        group = DataArray(dim.data, [(dim.dims, dim.data)], name=RESAMPLE_DIM)
        grouper = pd.Grouper(
            freq=freq,
            how=how,
            closed=closed,
            label=label,
            origin=origin,
            offset=offset)
        gb = self._groupby_cls(self, group, grouper=grouper)
        if isinstance(how, str):
            f = getattr(gb, how)
            if how in ['first', 'last']:
                result = f(skipna=skipna, keep_attrs=False)
            elif how == 'count':
                result = f(dim=dim.name, keep_attrs=False)
            else:
                result = f(dim=dim.name, skipna=skipna, keep_attrs=False)
        else:
            result = gb.reduce(how, dim=dim.name, keep_attrs=False)
        result = result.rename({RESAMPLE_DIM: dim.name})

        attrs = self.attrs if keep_attrs else None
        return DataStore(
            data_vars=result.data_vars, coords=result.coords, attrs=attrs)

    def to_netcdf(
            self,
            path=None,
            mode='w',
            format=None,
            group=None,
            engine=None,
            encoding=None,
            unlimited_dims=None,
            compute=True):
        """Write datastore contents to a netCDF file.

        Parameters
        ----------
        path : str, Path or file-like object, optional
            Path to which to save this dataset. File-like objects are only
            supported by the scipy engine. If no path is provided, this
            function returns the resulting netCDF file as bytes; in this case,
            we need to use scipy, which does not support netCDF version 4 (the
            default format becomes NETCDF3_64BIT).
        mode : {'w', 'a'}, optional
            Write ('w') or append ('a') mode. If mode='w', any existing file at
            this location will be overwritten. If mode='a', existing variables
            will be overwritten.
        format : {'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT',
        'NETCDF3_CLASSIC'}, optional
            File format for the resulting netCDF file:
            * NETCDF4: Data is stored in an HDF5 file, using netCDF4 API
              features.
            * NETCDF4_CLASSIC: Data is stored in an HDF5 file, using only
              netCDF 3 compatible API features.
            * NETCDF3_64BIT: 64-bit offset version of the netCDF 3 file format,
              which fully supports 2+ GB files, but is only compatible with
              clients linked against netCDF version 3.6.0 or later.
            * NETCDF3_CLASSIC: The classic netCDF 3 file format. It does not
              handle 2+ GB files very well.
            All formats are supported by the netCDF4-python library.
            scipy.io.netcdf only supports the last two formats.
            The default format is NETCDF4 if you are saving a file to disk and
            have the netCDF4-python library available. Otherwise, xarray falls
            back to using scipy to write netCDF files and defaults to the
            NETCDF3_64BIT format (scipy does not support netCDF4).
        group : str, optional
            Path to the netCDF4 group in the given file to open (only works for
            format='NETCDF4'). The group(s) will be created if necessary.
        engine : {'netcdf4', 'scipy', 'h5netcdf'}, optional
            Engine to use when writing netCDF files. If not provided, the
            default engine is chosen based on available dependencies, with a
            preference for 'netcdf4' if writing to a file on disk.
        encoding : dict, optional
            defaults to reasonable compression. Use encoding={} to disable
            encoding.
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{'my_variable': {'dtype': 'int16', 'scale_factor': 0.1,
                               'zlib': True}, ...}``
            The `h5netcdf` engine supports both the NetCDF4-style compression
            encoding parameters ``{'zlib': True, 'complevel': 9}`` and the h5py
            ones ``{'compression': 'gzip', 'compression_opts': 9}``.
            This allows using any compression plugin installed in the HDF5
            library, e.g. LZF.
        unlimited_dims : sequence of str, optional
            Dimension(s) that should be serialized as unlimited dimensions.
            By default, no dimensions are treated as unlimited dimensions.
            Note that unlimited_dims may also be set via
            ``dataset.encoding['unlimited_dims']``.
        compute: boolean
            If true compute immediately, otherwise return a
            ``dask.delayed.Delayed`` object that can be computed later.
        """
        if encoding is None:
            encoding = self.get_default_encoding()

        if engine is None:
            engine = 'netcdf4'

        # Fix Bart Schilperoort: netCDF doesn't like None's
        for attribute, value in self.attrs.items():
            if value is None:
                self.attrs[attribute] = ''

        return super(DataStore, self).to_netcdf(
            path,
            mode,
            format=format,
            group=group,
            engine=engine,
            encoding=encoding,
            unlimited_dims=unlimited_dims,
            compute=compute)

    def to_mf_netcdf(
            self,
            folder_path=None,
            filename_preamble='file_',
            filename_extension='.nc',
            format='netCDF4',
            engine='netcdf4',
            encoding=None,
            mode='w',
            compute=True,
            time_chunks_from_key='st'):
        """Write DataStore to multiple to multiple netCDF files.

        Splits the DataStore along the time dimension using the chunks. It
        first checks if all chunks in `ds` are time aligned. If this is not
        the case, calculate optimal chunk sizes using the
        `time_chunks_from_key` array. The files are written per time-chunk to
        disk.

        Almost similar to xarray.save_mfdataset,

        Parameters
        ----------
        folder_path : str, Path
            Folder to place the files
        filename_preamble : str
            Filename is `filename_preamble + '0000' + filename_extension
        filename_extension : str
            Filename is `filename_preamble + '0000' + filename_extension
        mode : {'w', 'a'}, optional
            Write ('w') or append ('a') mode. If mode='w', any existing file at
            these locations will be overwritten.
        format : {'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT',
                  'NETCDF3_CLASSIC'}, optional
            File format for the resulting netCDF file:
            * NETCDF4: Data is stored in an HDF5 file, using netCDF4 API
              features.
            * NETCDF4_CLASSIC: Data is stored in an HDF5 file, using only
              netCDF 3 compatible API features.
            * NETCDF3_64BIT: 64-bit offset version of the netCDF 3 file format,
              which fully supports 2+ GB files, but is only compatible with
              clients linked against netCDF version 3.6.0 or later.
            * NETCDF3_CLASSIC: The classic netCDF 3 file format. It does not
              handle 2+ GB files very well.
            All formats are supported by the netCDF4-python library.
            scipy.io.netcdf only supports the last two formats.
            The default format is NETCDF4 if you are saving a file to disk and
            have the netCDF4-python library available. Otherwise, xarray falls
            back to using scipy to write netCDF files and defaults to the
            NETCDF3_64BIT format (scipy does not support netCDF4).
        engine : {'netcdf4', 'scipy', 'h5netcdf'}, optional
            Engine to use when writing netCDF files. If not provided, the
            default engine is chosen based on available dependencies, with a
            preference for 'netcdf4' if writing to a file on disk.
            See `Dataset.to_netcdf` for additional information.
        encoding : list of dict, optional
            Defaults to reasonable compression/encoding.
            If you want to define your own encoding, you first needs to know the
            time-chunk sizes this routine will write to disk. After which you
            need to provide a list with the encoding specified for each chunk.
            Use a list of empty dicts to disable encoding.
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{'my_variable': {'dtype': 'int16', 'scale_factor': 0.1,
                               'zlib': True}, ...}``
            The `h5netcdf` engine supports both the NetCDF4-style compression
            encoding parameters ``{'zlib': True, 'complevel': 9}`` and the h5py
            ones ``{'compression': 'gzip', 'compression_opts': 9}``.
            This allows using any compression plugin installed in the HDF5
            library, e.g. LZF.
        compute: boolean
            If true compute immediately, otherwise return a
            ``dask.delayed.Delayed`` object that can be computed later.
        time_chunks_from_key: str

        Examples
        --------
        ds.to_mf_netcdf(folder_path='.')

        See Also
        --------
        dtscalibration.open_mf_datastore
        xarray.save_mfdataset

        """

        try:
            # This fails if not all chunks of the data_vars are time aligned.
            # In case we let Dask estimate an optimal chunk size.
            t_chunks = self.chunks['time']

        except:  # noqa: E722
            if self[time_chunks_from_key].dims == ('x', 'time'):
                _, t_chunks = da.ones(
                    self[time_chunks_from_key].shape,
                    chunks=(-1, 'auto'),
                    dtype='float64').chunks

            elif self[time_chunks_from_key].dims == ('time', 'x'):
                _, t_chunks = da.ones(
                    self[time_chunks_from_key].shape,
                    chunks=('auto', -1),
                    dtype='float64').chunks
            else:
                assert 0, 'something went wrong with your Stokes dimensions'

        bnds = np.cumsum((0,) + t_chunks)
        x = [range(bu, bd) for bu, bd in zip(bnds[:-1], bnds[1:])]

        datasets = [self.isel(time=xi) for xi in x]
        paths = [
            os.path.join(
                folder_path,
                filename_preamble + "{:04d}".format(ix) + filename_extension)
            for ix in range(len(x))]

        encodings = []
        for ids, ds in enumerate(datasets):
            if encoding is None:
                encodings.append(
                    ds.get_default_encoding(
                        time_chunks_from_key=time_chunks_from_key))

            else:
                encodings.append(encoding[ids])

        writers, stores = zip(
            *[
                xr.backends.api.to_netcdf(
                    ds,
                    path,
                    mode,
                    format,
                    None,
                    engine,
                    compute=compute,
                    multifile=True,
                    encoding=enc)
                for ds, path, enc in zip(datasets, paths, encodings)])

        try:
            writes = [w.sync(compute=compute) for w in writers]
        finally:
            if compute:
                for store in stores:
                    store.close()

        if not compute:

            def _finalize_store(write, store):
                """ Finalize this store by explicitly syncing and closing"""
                del write  # ensure writing is done first
                store.close()
                pass

            return dask.delayed(
                [
                    dask.delayed(_finalize_store)(w, s)
                    for w, s in zip(writes, stores)])

        pass

    def get_default_encoding(self, time_chunks_from_key=None):
        """
        Returns a dictionary with sensible compression setting for writing
        netCDF files.

        Returns
        -------

        """
        # The following variables are stored with a sufficiently large
        # precision in 32 bit
        float32l = [
            'st', 'ast', 'rst', 'rast', 'time', 'timestart', 'tmp', 'timeend',
            'acquisitionTime', 'x']
        int32l = [
            'filename_tstamp', 'acquisitiontimeFW', 'acquisitiontimeBW',
            'userAcquisitionTimeFW', 'userAcquisitionTimeBW']

        # default variable compression
        compdata = dict(
            zlib=True, complevel=6,
            shuffle=False)  # , least_significant_digit=None

        # default coordinate compression
        compcoords = dict(zlib=True, complevel=4)

        # construct encoding dict
        encoding = {var: compdata.copy() for var in self.data_vars}
        encoding.update({var: compcoords.copy() for var in self.coords})

        for k, v in encoding.items():
            if k in float32l:
                v['dtype'] = 'float32'

            if k in int32l:
                v['dtype'] = 'int32'
                # v['_FillValue'] = -9999  # Int does not support NaN

        if time_chunks_from_key is not None:
            # obtain optimal chunk sizes in time and x dim
            if self[time_chunks_from_key].dims == ('x', 'time'):
                x_chunk, t_chunk = da.ones(
                    self[time_chunks_from_key].shape,
                    chunks=(-1, 'auto'),
                    dtype='float64').chunks

            elif self[time_chunks_from_key].dims == ('time', 'x'):
                x_chunk, t_chunk = da.ones(
                    self[time_chunks_from_key].shape,
                    chunks=('auto', -1),
                    dtype='float64').chunks
            else:
                assert 0, 'something went wrong with your Stokes dimensions'

            for k, v in encoding.items():
                # By writing and compressing the data in chunks, some sort of
                # parallism is possible.
                if self[k].dims == ('x', 'time'):
                    chunks = (x_chunk[0], t_chunk[0])

                elif self[k].dims == ('time', 'x'):
                    chunks = (t_chunk[0], x_chunk[0])

                elif self[k].dims == ('x',):
                    chunks = (x_chunk[0],)

                elif self[k].dims == ('time',):
                    chunks = (t_chunk[0],)

                else:
                    continue

                v['chunksizes'] = chunks

        return encoding

    def get_time_dim(self, data_var_key=None):
        """
        Find relevant time dimension. by educative guessing

        Parameters
        ----------
        data_var_key : str
            The data variable key that contains a relevant time dimension. If
            None, 'st' is used.

        Returns
        -------

        """
        options = [
            'date', 'time', 'day', 'days', 'hour', 'hours', 'minute',
            'minutes', 'second', 'seconds']
        if data_var_key is None:
            if 'st' in self.data_vars:
                data_var_key = 'st'
            elif 'st' in self.data_vars:
                data_var_key = 'st'
            else:
                return 'time'

        dims = self[data_var_key].dims
        # find all dims in options
        in_opt = [next(filter(lambda s: s == d, options), None) for d in dims]

        if in_opt and in_opt != [None]:
            # exclude Nones from list
            return next(filter(None, in_opt))

        else:
            # there is no time dimension
            return None

    def get_section_indices(self, sec):
        """Returns the x-indices of the section. `sec` is a slice."""
        xis = self.x.astype(int) * 0 + np.arange(self.x.size, dtype=int)
        return xis.sel(x=sec).values

    def check_deprecated_kwargs(self, kwargs):
        """
        Internal function that parses the `kwargs` for depreciated keyword
        arguments.

        Depreciated keywords raise an error, pending to be depreciated do not.
        But this requires that the code currently deals with those arguments.

        Parameters
        ----------
        kwargs : Dict
            A dictionary with keyword arguments.

        Returns
        -------

        """
        msg = """Previously, it was possible to manually set the label from
        which the Stokes and anti-Stokes were read within the DataStore
        object. To reduce the clutter in the code base and be able to
        maintain it, this option was removed.
        See: https://github.com/dtscalibration/python-dts-calibration/issues/81

        The new **fixed** names are: st, ast, rst, rast.

        It is still possible to use the previous defaults, for example when
        reading stored measurements from netCDF, by renaming the labels. The
        old default labels were ST, AST, REV-ST, REV-AST.

        ```
        ds = open_datastore(path_to_old_file)
        ds = ds.rename_labels()
        ds.calibration_double_ended(
            st_var=1.5,
            ast_var=1.5,
            rst_var=1.,
            rast_var=1.,
            method='wls')
        ```

        ds.tmpw.plot()
        """
        list_of_depr = ['st_label', 'ast_label', 'rst_label', 'rast_label']
        list_of_pending_depr = ['transient_asym_att_x', 'transient_att_x']

        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in list_of_pending_depr}

        for k in kwargs:
            if k in list_of_depr:
                raise NotImplementedError(msg)

        if len(kwargs) != 0:
            raise NotImplementedError(
                'The following keywords are not ' + 'supported: '
                + ', '.join(kwargs.keys()))

        pass

    def rename_labels(self, assertion=True):
        """
        Renames the `ST` DataArrays (old convention) to `st` (new convention).
        The new naming convention simplifies the notation of the reverse Stokes
        `ds['REV-ST']` becomes `ds.rst`. Plus the parameter-naming convention in
        Python in lowercase.

        Parameters
        ----------
        assertion : bool
            If set to `True`, raises an error if complications occur.

        Returns
        -------

        """
        re_dict = {
            'ST': 'st',
            'AST': 'ast',
            'REV-ST': 'rst',
            'REV-AST': 'rast',
            'TMP': 'tmp',
            'TMPF': 'tmpf',
            'TMPB': 'tmpb',
            'TMPW': 'tmpw'}

        re_dict_err = {
            k: v
            for k, v in re_dict.items()
            if k in self.data_vars and v in self.data_vars}

        msg = (
            'Unable to rename the st_labels automagically. \n'
            'Please manually rename ST->st and REV-ST->rst. The \n'
            f'parameters {re_dict_err.values()} were already present')

        if assertion:
            assert len(re_dict_err) == 0, msg
        elif len(re_dict_err) != 0:
            print(msg)
            for v in re_dict_err.values():
                print(f'Variable {v} was not renamed')

        re_dict2 = {
            k: v
            for k, v in re_dict.items()
            if k in self.data_vars and v not in self.data_vars}

        return self.rename(re_dict2)

    def variance_stokes(self, *args, **kwargs):
        """Backwards compatibility. See `ds.variance_stokes_constant()`
        """
        return self.variance_stokes_constant(*args, **kwargs)

    def variance_stokes_constant(
            self, st_label, sections=None, reshape_residuals=True):
        """
        Approximate the variance of the noise in Stokes intensity measurements
        with one value, suitable for small setups.

        * `ds.variance_stokes_constant()` for small setups with small variations in\
        intensity. Variance of the Stokes measurements is assumed to be the same\
        along the entire fiber.

        * `ds.variance_stokes_exponential()` for small setups with very few time\
        steps. Too many degrees of freedom results in an under estimation of the\
        noise variance. Almost never the case, but use when calibrating pre time\
        step.

        * `ds.variance_stokes_linear()` for larger setups with more time steps.\
            Assumes Poisson distributed noise with the following model::

                st_var = a * ds.st + b


            where `a` and `b` are constants. Requires reference sections at
            beginning and end of the fiber, to have residuals at high and low
            intensity measurements.

        The Stokes and anti-Stokes intensities are measured with detectors,
        which inherently introduce noise to the measurements. Knowledge of the
        distribution of the measurement noise is needed for a calibration with
        weighted observations (Sections 5 and 6 of [1]_)
        and to project the associated uncertainty to the temperature confidence
        intervals (Section 7 of [1]_). Two sources dominate the noise
        in the Stokes and anti-Stokes intensity measurements
        (Hartog, 2017, p.125). Close to the laser, noise from the conversion of
        backscatter to electricity dominates the measurement noise. The
        detecting component, an avalanche photodiode, produces Poisson-
        distributed noise with a variance that increases linearly with the
        intensity. The Stokes and anti-Stokes intensities are commonly much
        larger than the standard deviation of the noise, so that the Poisson
        distribution can be approximated with a Normal distribution with a mean
        of zero and a variance that increases linearly with the intensity. At
        the far-end of the fiber, noise from the electrical circuit dominates
        the measurement noise. It produces Normal-distributed noise with a mean
        of zero and a variance that is independent of the intensity.

        Calculates the variance between the measurements and a best fit
        at each reference section. This fits a function to the nt * nx
        measurements with ns * nt + nx parameters, where nx are the total
        number of reference locations along all sections. The temperature is
        constant along the reference sections, so the expression of the
        Stokes power can be split in a time series per reference section and
        a constant per observation location.

        Idea from Discussion at page 127 in Richter, P. H. (1995). Estimating
        errors in least-squares fitting.

        The timeseries and the constant are, of course, highly correlated
        (Equations 20 and 21 in [1]_), but that is not relevant here as only the
        product is of interest. The residuals between the fitted product and the
        Stokes intensity measurements are attributed to the
        noise from the detector. The variance of the residuals is used as a
        proxy for the variance of the noise in the Stokes and anti-Stokes
        intensity measurements. A non-uniform temperature of
        the reference sections results in an over estimation of the noise
        variance estimate because all temperature variation is attributed to
        the noise.

        Parameters
        ----------
        reshape_residuals
        st_label : str
            label of the Stokes, anti-Stokes measurement.
            E.g., st, ast, rst, rast
        sections : Dict[str, List[slice]], optional
            If `None` is supplied, `ds.sections` is used. Define calibration
            sections. Each section requires a reference temperature time series,
            such as the temperature measured by an external temperature sensor.
            They should already be part of the DataStore object. `sections`
            is defined with a dictionary with its keywords of the
            names of the reference temperature time series. Its values are
            lists of slice objects, where each slice object is a fiber stretch
            that has the reference temperature. Afterwards, `sections` is stored
            under `ds.sections`.

        Returns
        -------
        I_var : float
            Variance of the residuals between measured and best fit
        resid : array_like
            Residuals between measured and best fit

        Notes
        -----

        * Because there are a large number of unknowns, spend time on\
        calculating an initial estimate. Can be turned off by setting to False.

        * It is often not needed to use measurements from all time steps. If\
        your variance estimate does not change when including measurements from\
        more time steps, you have included enough measurements.

        References
        ----------
        .. [1] des Tombe, B., Schilperoort, B., & Bakker, M. (2020). Estimation
            of Temperature and Associated Uncertainty from Fiber-Optic Raman-
            Spectrum Distributed Temperature Sensing. Sensors, 20(8), 2235.
            https://doi.org/10.3390/s20082235

        Examples
        --------
        - `Example notebook 4: Calculate variance Stokes intensity measurements\
        <https://github.com/\
        dtscalibration/python-dts-calibration/blob/master/examples/notebooks/\
        04Calculate_variance_Stokes.ipynb>`_
        """
        if sections:
            self.sections = sections
        else:
            assert self.sections, 'sections are not defined'

        assert self[st_label].dims[0] == 'x', 'Stokes are transposed'

        check_timestep_allclose(self, eps=0.01)

        data_dict = da.compute(
            self.ufunc_per_section(label=st_label, calc_per='stretch'))[
                0]  # should maybe be per section. But then residuals
        # seem to be correlated between stretches. I don't know why.. BdT.
        resid_list = []

        for k, v in data_dict.items():
            for vi in v:
                nxs, nt = vi.shape
                npar = nt + nxs

                p1 = np.ones(npar) * vi.mean()**0.5

                res = minimize(func_cost, p1, args=(vi, nxs), method='Powell')
                assert res.success, 'Unable to fit. Try variance_stokes_exponential'

                fit = func_fit(res.x, nxs)
                resid_list.append(fit - vi)

        resid = np.concatenate(resid_list)

        # unbiased estimater ddof=1, originally thought it was npar
        var_I = resid.var(ddof=1)

        if not reshape_residuals:
            return var_I, resid

        else:
            ix_resid = self.ufunc_per_section(x_indices=True, calc_per='all')

            resid_sorted = np.full(
                shape=self[st_label].shape, fill_value=np.nan)
            resid_sorted[ix_resid, :] = resid
            resid_da = xr.DataArray(
                data=resid_sorted, coords=self[st_label].coords)

            return var_I, resid_da

    def variance_stokes_exponential(
            self,
            st_label,
            sections=None,
            use_statsmodels=False,
            suppress_info=True,
            reshape_residuals=True):
        """
        Approximate the variance of the noise in Stokes intensity measurements
        with one value, suitable for small setups with measurements from only
        a few times.

        * `ds.variance_stokes_constant()` for small setups with small variations in\
        intensity. Variance of the Stokes measurements is assumed to be the same\
        along the entire fiber.

        * `ds.variance_stokes_exponential()` for small setups with very few time\
        steps. Too many degrees of freedom results in an under estimation of the\
        noise variance. Almost never the case, but use when calibrating pre time\
        step.

        * `ds.variance_stokes_linear()` for larger setups with more time steps.\
            Assumes Poisson distributed noise with the following model::

                st_var = a * ds.st + b


            where `a` and `b` are constants. Requires reference sections at
            beginning and end of the fiber, to have residuals at high and low
            intensity measurements.

        The Stokes and anti-Stokes intensities are measured with detectors,
        which inherently introduce noise to the measurements. Knowledge of the
        distribution of the measurement noise is needed for a calibration with
        weighted observations (Sections 5 and 6 of [1]_)
        and to project the associated uncertainty to the temperature confidence
        intervals (Section 7 of [1]_). Two sources dominate the noise
        in the Stokes and anti-Stokes intensity measurements
        (Hartog, 2017, p.125). Close to the laser, noise from the conversion of
        backscatter to electricity dominates the measurement noise. The
        detecting component, an avalanche photodiode, produces Poisson-
        distributed noise with a variance that increases linearly with the
        intensity. The Stokes and anti-Stokes intensities are commonly much
        larger than the standard deviation of the noise, so that the Poisson
        distribution can be approximated with a Normal distribution with a mean
        of zero and a variance that increases linearly with the intensity. At
        the far-end of the fiber, noise from the electrical circuit dominates
        the measurement noise. It produces Normal-distributed noise with a mean
        of zero and a variance that is independent of the intensity.

        Calculates the variance between the measurements and a best fit
        at each reference section. This fits a function to the nt * nx
        measurements with ns * nt + nx parameters, where nx are the total
        number of reference locations along all sections. The temperature is
        constant along the reference sections. This fits a two-parameter
        exponential to the stokes measurements. The temperature is constant
        and there are no splices/sharp bends in each reference section.
        Therefore all signal decrease is due to differential attenuation,
        which is the same for each reference section. The scale of the
        exponential does differ per reference section.

        Assumptions: 1) the temperature is the same along a reference
        section. 2) no sharp bends and splices in the reference sections. 3)
        Same type of optical cable in each reference section.

        Idea from discussion at page 127 in Richter, P. H. (1995). Estimating
        errors in least-squares fitting. For weights used error propagation:
        w^2 = 1/sigma(lny)^2 = y^2/sigma(y)^2 = y^2

        The timeseries and the constant are, of course, highly correlated
        (Equations 20 and 21 in [1]_), but that is not relevant here as only the
        product is of interest. The residuals between the fitted product and the
        Stokes intensity measurements are attributed to the
        noise from the detector. The variance of the residuals is used as a
        proxy for the variance of the noise in the Stokes and anti-Stokes
        intensity measurements. A non-uniform temperature of
        the reference sections results in an over estimation of the noise
        variance estimate because all temperature variation is attributed to
        the noise.

        Parameters
        ----------
        reshape_residuals
        st_label : str
            label of the Stokes, anti-Stokes measurement.
            E.g., st, ast, rst, rast
        sections : Dict[str, List[slice]], optional
            If `None` is supplied, `ds.sections` is used. Define calibration
            sections. Each section requires a reference temperature time series,
            such as the temperature measured by an external temperature sensor.
            They should already be part of the DataStore object. `sections`
            is defined with a dictionary with its keywords of the
            names of the reference temperature time series. Its values are
            lists of slice objects, where each slice object is a fiber stretch
            that has the reference temperature. Afterwards, `sections` is stored
            under `ds.sections`.

        Returns
        -------
        I_var : float
            Variance of the residuals between measured and best fit
        resid : array_like
            Residuals between measured and best fit

        Notes
        -----

        * Because there are a large number of unknowns, spend time on\
        calculating an initial estimate. Can be turned off by setting to False.

        * It is often not needed to use measurements from all time steps. If\
        your variance estimate does not change when including measurements from\
        more time steps, you have included enough measurements.

        References
        ----------
        .. [1] des Tombe, B., Schilperoort, B., & Bakker, M. (2020). Estimation
            of Temperature and Associated Uncertainty from Fiber-Optic Raman-
            Spectrum Distributed Temperature Sensing. Sensors, 20(8), 2235.
            https://doi.org/10.3390/s20082235

        Examples
        --------
        - `Example notebook 4: Calculate variance Stokes intensity measurements\
        <https://github.com/\
        dtscalibration/python-dts-calibration/blob/master/examples/notebooks/\
        04Calculate_variance_Stokes.ipynb>`_
        """
        if sections:
            self.sections = sections
        else:
            assert self.sections, 'sections are not defined'

        assert self[st_label].dims[0] == 'x', 'Stokes are transposed'

        check_timestep_allclose(self, eps=0.01)

        nt = self.time.size

        len_stretch_list = []  # number of reference points per section (
        # spatial)
        y_list = []  # intensities of stokes
        x_list = []  # length rel to start of section. for alpha

        for k, stretches in self.sections.items():
            for stretch in stretches:
                y_list.append(self[st_label].sel(x=stretch).data.T.reshape(-1))
                _x = self.x.sel(x=stretch).data.copy()
                _x -= _x[0]
                x_list.append(da.tile(_x, nt))
                len_stretch_list.append(_x.size)

        n_sections = len(len_stretch_list)  # number of sections
        n_locs = sum(
            len_stretch_list)  # total number of locations along cable used
        # for reference.

        x = np.concatenate(x_list)  # coordinates are already in memory
        y = np.concatenate(y_list)

        data1 = x
        data2 = np.ones(sum(len_stretch_list) * nt)
        data = np.concatenate([data1, data2])

        # alpha is NOT the same for all -> one column per section
        coords1row = np.arange(nt * n_locs)
        coords1col = np.hstack(
            [
                np.ones(in_locs * nt) * i
                for i, in_locs in enumerate(len_stretch_list)])  # C for

        # second calibration parameter is different per section and per timestep
        coords2row = np.arange(nt * n_locs)
        coords2col = np.hstack(
            [
                np.repeat(
                    np.arange(i * nt + n_sections, (i + 1) * nt + n_sections),
                    in_locs)
                for i, in_locs in enumerate(len_stretch_list)])  # C for
        coords = (
            np.concatenate([coords1row, coords2row]),
            np.concatenate([coords1col, coords2col]))

        lny = np.log(y)
        w = y.copy()  # 1/std.

        ddof = n_sections + nt * n_sections  # see numpy documentation on ddof

        if use_statsmodels:
            # returns the same answer with statsmodel
            import statsmodels.api as sm

            X = sp.coo_matrix(
                (data, coords),
                shape=(nt * n_locs, ddof),
                dtype=float,
                copy=False)

            mod_wls = sm.WLS(lny, X.toarray(), weights=w**2)
            res_wls = mod_wls.fit()
            # print(res_wls.summary())
            a = res_wls.params

        else:
            wdata = data * np.hstack((w, w))
            wX = sp.coo_matrix(
                (wdata, coords),
                shape=(nt * n_locs, n_sections + nt * n_sections),
                dtype=float,
                copy=False)

            wlny = (lny * w)

            p0_est = np.asarray(n_sections * [0.] + nt * n_sections * [8])
            # noinspection PyTypeChecker
            a = ln.lsqr(
                wX, wlny, x0=p0_est, show=not suppress_info, calc_var=False)[0]

        beta = a[:n_sections]
        beta_expand_to_sec = np.hstack(
            [
                np.repeat(float(beta[i]), leni * nt)
                for i, leni in enumerate(len_stretch_list)])
        G = np.asarray(a[n_sections:])
        G_expand_to_sec = np.hstack(
            [
                np.repeat(G[i * nt:(i + 1) * nt], leni)
                for i, leni in enumerate(len_stretch_list)])

        I_est = np.exp(G_expand_to_sec) * np.exp(x * beta_expand_to_sec)
        resid = I_est - y
        var_I = resid.var(ddof=1)

        if not reshape_residuals:
            return var_I, resid
        else:
            # restructure the residuals, such that they can be plotted and
            # added to ds
            resid_res = []
            for leni, lenis, lenie in zip(
                    len_stretch_list,
                    nt * np.cumsum([0] + len_stretch_list[:-1]),
                    nt * np.cumsum(len_stretch_list)):
                try:
                    resid_res.append(
                        resid[lenis:lenie].reshape((leni, nt), order='F'))
                except:  # noqa: E722
                    # Dask array does not support order
                    resid_res.append(
                        resid[lenis:lenie].T.reshape((nt, leni)).T)

            _resid = np.concatenate(resid_res)
            _resid_x = self.ufunc_per_section(label='x', calc_per='all')
            isort = np.argsort(_resid_x)
            resid_x = _resid_x[isort]  # get indices from ufunc directly
            resid = _resid[isort, :]

            ix_resid = np.array(
                [np.argmin(np.abs(ai - self.x.data)) for ai in resid_x])

            resid_sorted = np.full(
                shape=self[st_label].shape, fill_value=np.nan)
            resid_sorted[ix_resid, :] = resid
            resid_da = xr.DataArray(
                data=resid_sorted, coords=self[st_label].coords)

            return var_I, resid_da

    def variance_stokes_linear(
            self,
            st_label,
            sections=None,
            nbin=50,
            through_zero=True,
            plot_fit=False):
        """
        Approximate the variance of the noise in Stokes intensity measurements
        with a linear function of the intensity, suitable for large setups.

        * `ds.variance_stokes_constant()` for small setups with small variations in\
        intensity. Variance of the Stokes measurements is assumed to be the same\
        along the entire fiber.

        * `ds.variance_stokes_exponential()` for small setups with very few time\
        steps. Too many degrees of freedom results in an under estimation of the\
        noise variance. Almost never the case, but use when calibrating pre time\
        step.

        * `ds.variance_stokes_linear()` for larger setups with more time steps.\
            Assumes Poisson distributed noise with the following model::

                st_var = a * ds.st + b


            where `a` and `b` are constants. Requires reference sections at
            beginning and end of the fiber, to have residuals at high and low
            intensity measurements.

        The Stokes and anti-Stokes intensities are measured with detectors,
        which inherently introduce noise to the measurements. Knowledge of the
        distribution of the measurement noise is needed for a calibration with
        weighted observations (Sections 5 and 6 of [1]_)
        and to project the associated uncertainty to the temperature confidence
        intervals (Section 7 of [1]_). Two sources dominate the noise
        in the Stokes and anti-Stokes intensity measurements
        (Hartog, 2017, p.125). Close to the laser, noise from the conversion of
        backscatter to electricity dominates the measurement noise. The
        detecting component, an avalanche photodiode, produces Poisson-
        distributed noise with a variance that increases linearly with the
        intensity. The Stokes and anti-Stokes intensities are commonly much
        larger than the standard deviation of the noise, so that the Poisson
        distribution can be approximated with a Normal distribution with a mean
        of zero and a variance that increases linearly with the intensity. At
        the far-end of the fiber, noise from the electrical circuit dominates
        the measurement noise. It produces Normal-distributed noise with a mean
        of zero and a variance that is independent of the intensity.

        Calculates the variance between the measurements and a best fit
        at each reference section. This fits a function to the nt * nx
        measurements with ns * nt + nx parameters, where nx are the total
        number of reference locations along all sections. The temperature is
        constant along the reference sections, so the expression of the
        Stokes power can be split in a time series per reference section and
        a constant per observation location.

        Idea from Discussion at page 127 in Richter, P. H. (1995). Estimating
        errors in least-squares fitting.

        The timeseries and the constant are, of course, highly correlated
        (Equations 20 and 21 in [1]_), but that is not relevant here as only the
        product is of interest. The residuals between the fitted product and the
        Stokes intensity measurements are attributed to the
        noise from the detector. The variance of the residuals is used as a
        proxy for the variance of the noise in the Stokes and anti-Stokes
        intensity measurements. A non-uniform temperature of
        the reference sections results in an over estimation of the noise
        variance estimate because all temperature variation is attributed to
        the noise.

        Notes
        -----

        * Because there are a large number of unknowns, spend time on\
        calculating an initial estimate. Can be turned off by setting to False.

        * It is often not needed to use measurements from all time steps. If\
        your variance estimate does not change when including measurements \
        from more time steps, you have included enough measurements.

        References
        ----------
        .. [1] des Tombe, B., Schilperoort, B., & Bakker, M. (2020). Estimation
            of Temperature and Associated Uncertainty from Fiber-Optic Raman-
            Spectrum Distributed Temperature Sensing. Sensors, 20(8), 2235.
            https://doi.org/10.3390/s20082235

        Examples
        --------
        - `Example notebook 4: Calculate variance Stokes intensity \
        measurements <https://github.com/\
        dtscalibration/python-dts-calibration/blob/master/examples/notebooks/\
        04Calculate_variance_Stokes.ipynb>`_

        Parameters
        ----------
        st_label : str
            Key under which the Stokes DataArray is stored. E.g., 'st', 'rst'
        sections : dict, optional
            Define sections. See documentation
        nbin : int
            Number of bins to compute the variance for, through which the
            linear function is fitted. Make sure that that are at least 50
            residuals per bin to compute the variance from.
        through_zero : bool
            If True, the variance is computed as: VAR(Stokes) = slope * Stokes
            If False, VAR(Stokes) = slope * Stokes + offset.
            From what we can tell from our inital trails, is that the offset
            seems relatively small, so that True seems a better option for
            setups where a reference section with very low Stokes intensities
            is missing. If data with low Stokes intensities available, it is
            better to not fit through zero, but determine the offset from
            the data.
        plot_fit : bool
            If True plot the variances for each bin and plot the fitted
            linear function
        """
        import matplotlib.pyplot as plt

        if sections:
            self.sections = sections
        else:
            assert self.sections, 'sections are not defined'

        assert self[st_label].dims[0] == 'x', 'Stokes are transposed'
        _, resid = self.variance_stokes(st_label=st_label)

        ix_sec = self.ufunc_per_section(x_indices=True, calc_per='all')
        st = self.isel(x=ix_sec)[st_label].values.ravel()
        diff_st = resid.isel(x=ix_sec).values.ravel()

        # Adjust nbin silently to fit residuals in
        # rectangular matrix and use numpy for computation
        nbin_ = nbin
        while st.size % nbin_:
            nbin_ -= 1

        if nbin_ != nbin:
            print(
                'Estimation of linear variance of', st_label,
                'Adjusting nbin to:', nbin_)
            nbin = nbin_

        isort = np.argsort(st)
        st_sort_mean = st[isort].reshape((nbin, -1)).mean(axis=1)
        st_sort_var = diff_st[isort].reshape((nbin, -1)).var(axis=1)

        if through_zero:
            # VAR(Stokes) = slope * Stokes
            offset = 0.
            slope = np.linalg.lstsq(
                st_sort_mean[:, None], st_sort_var, rcond=None)[0]
        else:
            # VAR(Stokes) = slope * Stokes + offset
            slope, offset = np.linalg.lstsq(
                np.hstack((st_sort_mean[:, None], np.ones((nbin, 1)))),
                st_sort_var,
                rcond=None)[0]

            if offset < 0:
                warnings.warn(
                    f"Warning! Offset of variance_stokes_linear() "
                    f"of {st_label} is negative. This is phisically "
                    f"not possible. Most likely, your {st_label} do "
                    f"not vary enough to fit a linear curve. Either "
                    f"use `through_zero` option or use "
                    f"`ds.variance_stokes_constant()`")

        def var_fun(stokes):
            return slope * stokes + offset

        if plot_fit:
            plt.figure()
            plt.scatter(st_sort_mean, st_sort_var, marker='.', c='black')
            plt.plot(
                [0., st_sort_mean[-1]],
                [var_fun(0.), var_fun(st_sort_mean[-1])],
                c='white',
                lw=1.3)
            plt.plot(
                [0., st_sort_mean[-1]],
                [var_fun(0.), var_fun(st_sort_mean[-1])],
                c='black',
                lw=0.8)
            plt.xlabel(st_label + ' intensity')
            plt.ylabel(st_label + ' intensity variance')

        return slope, offset, st_sort_mean, st_sort_var, resid, var_fun

    def i_var(self, st_var, ast_var, st_label='st', ast_label='ast'):
        """
        Compute the variance of an observation given the stokes and anti-Stokes
        intensities and their variance.
        The variance, :math:`\sigma^2_{I_{m,n}}`, of the distribution of the
        noise in the observation at location :math:`m`, time :math:`n`, is a
        function of the variance of the noise in the Stokes and anti-Stokes
        intensity measurements (:math:`\sigma_{P_+}^2` and
        :math:`\sigma_{P_-}^2`), and is approximated with (Ku et al., 1966):

        .. math::

            \sigma^2_{I_{m,n}} \\approx \left[\\frac{\partial I_{m,n}}{\partial\
            P_{m,n+}}\\right]^2\sigma^2_{P_{+}} + \left[\\frac{\partial\
            I_{m,n}}{\partial\
            P_{m,n-}}\\right]^2\sigma^2_{P_{-}}

        .. math::

            \sigma^2_{I_{m,n}} \\approx \\frac{1}{P_{m,n+}^2}\sigma^2_{P_{+}} +\
            \\frac{1}{P_{m,n-}^2}\sigma^2_{P_{-}}

        The variance of the noise in the Stokes and anti-Stokes intensity
        measurements is estimated directly from Stokes and anti-Stokes intensity
        measurements using the steps outlined in Section 4.

        Parameters
        ----------
        st_var, ast_var : float, callable, array-like, optional
            The variance of the measurement noise of the Stokes signals in the
            forward direction. If `float` the variance of the noise from the
            Stokes detector is described with a single value.
            If `callable` the variance of the noise from the Stokes detector is
            a function of the intensity, as defined in the callable function.
            Or manually define a variance with a DataArray of the shape
            `ds.st.shape`, where the variance can be a function of time and/or
            x.
        st_label : {'st', 'rst'}
        ast_label : {'ast', 'rast'}

        Returns
        -------

        """
        st = self[st_label]
        ast = self[ast_label]

        if callable(st_var):
            st_var = st_var(self[st_label]).values
        else:
            st_var = np.asarray(st_var, dtype=float)

        if callable(ast_var):
            ast_var = ast_var(self[ast_label]).values
        else:
            ast_var = np.asarray(ast_var, dtype=float)

        return st**-2 * st_var + ast**-2 * ast_var

    def inverse_variance_weighted_mean(
            self,
            tmp1='tmpf',
            tmp2='tmpb',
            tmp1_var='tmpf_mc_var',
            tmp2_var='tmpb_mc_var',
            tmpw_store='tmpw',
            tmpw_var_store='tmpw_var'):
        """
        Average two temperature datasets with the inverse of the variance as
        weights. The two
        temperature datasets `tmp1` and `tmp2` with their variances
        `tmp1_var` and `tmp2_var`,
        respectively. Are averaged and stored in the DataStore.

        Parameters
        ----------
        tmp1 : str
            The label of the first temperature dataset that is averaged
        tmp2 : str
            The label of the second temperature dataset that is averaged
        tmp1_var : str
            The variance of tmp1
        tmp2_var : str
            The variance of tmp2
        tmpw_store : str
            The label of the averaged temperature dataset
        tmpw_var_store : str
            The label of the variance of the averaged temperature dataset

        Returns
        -------

        """

        self[tmpw_var_store] = 1 / (1 / self[tmp1_var] + 1 / self[tmp2_var])

        self[tmpw_store] = (
            self[tmp1] / self[tmp1_var]
            + self[tmp2] / self[tmp2_var]) * self[tmpw_var_store]

        pass

    def inverse_variance_weighted_mean_array(
            self,
            tmp_label='tmpf',
            tmp_var_label='tmpf_mc_var',
            tmpw_store='tmpw',
            tmpw_var_store='tmpw_var',
            dim='time'):
        """
        Calculates the weighted average across a dimension.

        Parameters
        ----------

        Returns
        -------

        See Also
        --------
        - https://en.wikipedia.org/wiki/Inverse-variance_weighting

        """
        self[tmpw_var_store] = 1 / (1 / self[tmp_var_label]).sum(dim=dim)

        self[tmpw_store] = (self[tmp_label] / self[tmp_var_label]).sum(
            dim=dim) / (1 / self[tmp_var_label]).sum(dim=dim)

        pass

    def in_confidence_interval(self, ci_label, conf_ints=None, sections=None):
        """
        Returns an array with bools wether the temperature of the reference
        sections are within the confidence intervals

        Parameters
        ----------
        sections : Dict[str, List[slice]]
        ci_label : str
            The label of the data containing the confidence intervals.
        conf_ints : Tuple
            A tuple containing two floats between 0 and 1, representing the
            levels between which the reference temperature should lay.

        Returns
        -------

        """
        if sections is None:
            sections = self.sections

        if conf_ints is None:
            conf_ints = self[ci_label].values

        assert len(conf_ints) == 2, 'Please define conf_ints'

        tmp_dn = self[ci_label].sel(CI=conf_ints[0], method='nearest')
        tmp_up = self[ci_label].sel(CI=conf_ints[1], method='nearest')

        ref = self.ufunc_per_section(
            sections=sections,
            label='st',
            ref_temp_broadcasted=True,
            calc_per='all')
        ix_resid = self.ufunc_per_section(
            sections=sections, x_indices=True, calc_per='all')
        ref_sorted = np.full(shape=tmp_dn.shape, fill_value=np.nan)
        ref_sorted[ix_resid, :] = ref
        ref_da = xr.DataArray(data=ref_sorted, coords=tmp_dn.coords)

        mask_dn = ref_da >= tmp_dn
        mask_up = ref_da <= tmp_up

        return np.logical_and(mask_dn, mask_up)

    def set_trans_att(self, trans_att=None, **kwargs):
        """Gracefully set the locations that introduce directional differential
        attenuation

        Parameters
        ----------
        trans_att : iterable, optional
            Splices can cause jumps in differential attenuation. Normal single
            ended calibration assumes these are not present. An additional loss
            term is added in the 'shadow' of the splice. Each location
            introduces an additional nt parameters to solve for. Requiring
            either an additional calibration section or matching sections.
            If multiple locations are defined, the losses are added.

        """

        if 'transient_att_x' in kwargs:
            warnings.warn(
                "transient_att_x argument will be deprecated in version 2, "
                "use trans_att", DeprecationWarning)
            trans_att = kwargs['transient_att_x']

        if 'transient_asym_att_x' in kwargs:
            warnings.warn(
                "transient_asym_att_x arg will be deprecated in version 2, "
                "use trans_att", DeprecationWarning)
            trans_att = kwargs['transient_asym_att_x']

        if 'trans_att' in self.coords and self.trans_att.size > 0:
            raise_warning = 0

            del_keys = []
            for k, v in self.data_vars.items():
                if 'trans_att' in v.dims:
                    del_keys.append(k)

            for del_key in del_keys:
                del self[del_key]

            if raise_warning:
                m = 'trans_att was set before. All `data_vars` that make use ' \
                    'of the `trans_att` coordinates were deleted: ' + \
                    str(del_keys)
                warnings.warn(m)

        if trans_att is None:
            trans_att = []

        self['trans_att'] = trans_att
        self.trans_att.attrs = dim_attrs['trans_att']
        pass

    def check_reference_section_values(self):
        """
        Checks if the values of the used sections are of the right datatype
        (floats), if there are finite number (no NaN/inf), and if the time
        dimension corresponds with the time dimension of the st/ast data.

        Parameters
        ----------

        Returns
        -------

        """
        time_dim = self.get_time_dim()

        for key in self.sections.keys():
            if not np.issubdtype(self[key].dtype, np.floating):
                raise ValueError(
                    'Data of reference temperature "' + key
                    + '" does not have a float data type. Please ensure that '
                    'the data is of a valid type (e.g. np.float32)')

            if np.any(~np.isfinite(self[key].values)):
                raise ValueError(
                    'NaN/inf value(s) found in reference temperature "' + key
                    + '"')

            if self[key].dims != (time_dim,):
                raise ValueError(
                    'Time dimension of the reference temperature timeseries '
                    + key + 'is not the same as the time dimension'
                    + ' of the Stokes measurement. See examples/notebooks/09'
                    + 'Import_timeseries.ipynb for more info')

    def calibration_single_ended(
            self,
            sections=None,
            st_var=None,
            ast_var=None,
            store_c='c',
            store_gamma='gamma',
            store_dalpha='dalpha',
            store_alpha='alpha',
            store_ta='talpha',
            store_tmpf='tmpf',
            store_p_cov='p_cov',
            store_p_val='p_val',
            variance_suffix='_var',
            method='wls',
            solver='sparse',
            p_val=None,
            p_var=None,
            p_cov=None,
            matching_sections=None,
            trans_att=None,
            fix_gamma=None,
            fix_dalpha=None,
            fix_alpha=None,
            **kwargs):
        """
        Calibrate the Stokes (`ds.st`) and anti-Stokes (`ds.ast`) data to
        temperature using fiber sections with a known temperature
        (`ds.sections`) for single-ended setups. The calibrated temperature is
        stored under `ds.tmpf` and its variance under `ds.tmpf_var`.

        In single-ended setups, Stokes and anti-Stokes intensity is measured
        from a single end of the fiber. The differential attenuation is assumed
        constant along the fiber so that the integrated differential attenuation
        may be written as (Hausner et al, 2011):

        .. math::

            \int_0^x{\Delta\\alpha(x')\,\mathrm{d}x'} \\approx \Delta\\alpha x

        The temperature can now be written from Equation 10 [1]_ as:

        .. math::

            T(x,t)  \\approx \\frac{\gamma}{I(x,t) + C(t) + \Delta\\alpha x}

        where

        .. math::

            I(x,t) = \ln{\left(\\frac{P_+(x,t)}{P_-(x,t)}\\right)}


        .. math::

            C(t) = \ln{\left(\\frac{\eta_-(t)K_-/\lambda_-^4}{\eta_+(t)K_+/\lambda_+^4}\\right)}

        where :math:`C` is the lumped effect of the difference in gain at
        :math:`x=0` between Stokes and anti-Stokes intensity measurements and
        the dependence of the scattering intensity on the wavelength. The
        parameters :math:`P_+` and :math:`P_-` are the Stokes and anti-Stokes
        intensity measurements, respectively.
        The parameters :math:`\gamma`, :math:`C(t)`, and :math:`\Delta\\alpha`
        must be estimated from calibration to reference sections, as discussed
        in Section 5 [1]_. The parameter :math:`C` must be estimated
        for each time and is constant along the fiber. :math:`T` in the listed
        equations is in Kelvin, but is converted to Celsius after calibration.

        Parameters
        ----------
        store_p_cov : str
            Key to store the covariance matrix of the calibrated parameters
        store_p_val : str
            Key to store the values of the calibrated parameters
        p_val : array-like, optional
            Define `p_val`, `p_var`, `p_cov` if you used an external function
            for calibration. Has size 2 + `nt`. First value is :math:`\gamma`,
            second is :math:`\Delta \\alpha`, others are :math:`C` for each
            timestep.
        p_var : array-like, optional
            Define `p_val`, `p_var`, `p_cov` if you used an external function
            for calibration. Has size 2 + `nt`. First value is :math:`\gamma`,
            second is :math:`\Delta \\alpha`, others are :math:`C` for each
            timestep.
        p_cov : array-like, optional
            The covariances of `p_val`.
            If set to False, no uncertainty in the parameters is propagated
            into the confidence intervals. Similar to the spec sheets of the DTS
            manufacturers. And similar to passing an array filled with zeros.
        sections : Dict[str, List[slice]], optional
            If `None` is supplied, `ds.sections` is used. Define calibration
            sections. Each section requires a reference temperature time series,
            such as the temperature measured by an external temperature sensor.
            They should already be part of the DataStore object. `sections`
            is defined with a dictionary with its keywords of the
            names of the reference temperature time series. Its values are
            lists of slice objects, where each slice object is a fiber stretch
            that has the reference temperature. Afterwards, `sections` is stored
            under `ds.sections`.
        st_var, ast_var : float, callable, array-like, optional
            The variance of the measurement noise of the Stokes signals in the
            forward direction. If `float` the variance of the noise from the
            Stokes detector is described with a single value.
            If `callable` the variance of the noise from the Stokes detector is
            a function of the intensity, as defined in the callable function.
            Or manually define a variance with a DataArray of the shape
            `ds.st.shape`, where the variance can be a function of time and/or
            x. Required if method is wls.
        store_c : str
            Label of where to store C
        store_gamma : str
            Label of where to store gamma
        store_dalpha : str
            Label of where to store dalpha; the spatial derivative of alpha.
        store_alpha : str
            Label of where to store alpha; The integrated differential
            attenuation.
            alpha(x=0) = 0
        store_ta : str
            Label of where to store transient alpha's
        store_tmpf : str
            Label of where to store the calibrated temperature of the forward
            direction
        variance_suffix : str
            String appended for storing the variance. Only used when method
            is wls.
        method : {'ols', 'wls'}
            Use `'ols'` for ordinary least squares and `'wls'` for weighted least
            squares. `'wls'` is the default, and there is currently no reason to
            use `'ols'`.
        solver : {'sparse', 'stats'}
            Either use the homemade weighted sparse solver or the weighted
            dense matrix solver of statsmodels. The sparse solver uses much less
            memory, is faster, and gives the same result as the statsmodels
            solver. The statsmodels solver is mostly used to check the sparse
            solver. `'stats'` is the default.
        matching_sections : List[Tuple[slice, slice, bool]], optional
            Provide a list of tuples. A tuple per matching section. Each tuple
            has three items. The first two items are the slices of the sections
            that are matched. The third item is a boolean and is True if the two
            sections have a reverse direction ("J-configuration").
        transient_att_x, transient_asym_att_x : iterable, optional
            Depreciated. See trans_att
        trans_att : iterable, optional
            Splices can cause jumps in differential attenuation. Normal single
            ended calibration assumes these are not present. An additional loss
            term is added in the 'shadow' of the splice. Each location
            introduces an additional nt parameters to solve for. Requiring
            either an additional calibration section or matching sections.
            If multiple locations are defined, the losses are added.
        fix_gamma : Tuple[float, float], optional
            A tuple containing two floats. The first float is the value of
            gamma, and the second item is the variance of the estimate of gamma.
            Covariances between gamma and other parameters are not accounted
            for.
        fix_dalpha : Tuple[float, float], optional
            A tuple containing two floats. The first float is the value of
            dalpha (:math:`\Delta \\alpha` in [1]_), and the second item is the
            variance of the estimate of dalpha.
            Covariances between alpha and other parameters are not accounted
            for.

        Returns
        -------

        References
        ----------
        .. [1] des Tombe, B., Schilperoort, B., & Bakker, M. (2020). Estimation
            of Temperature and Associated Uncertainty from Fiber-Optic Raman-
            Spectrum Distributed Temperature Sensing. Sensors, 20(8), 2235.
            https://doi.org/10.3390/s20082235

        Examples
        --------
        - `Example notebook 7: Calibrate single ended <https://github.com/\
dtscalibration/python-dts-calibration/blob/master/examples/notebooks/\
07Calibrate_single_wls.ipynb>`_

        """
        self.check_deprecated_kwargs(kwargs)
        self.set_trans_att(trans_att=trans_att, **kwargs)

        if sections:
            self.sections = sections
        else:
            assert self.sections, 'sections are not defined'

        if method == 'wls':
            assert st_var is not None and ast_var is not None, 'Set `st_var`'

        self.check_reference_section_values()

        nx = self.x.size
        time_dim = self.get_time_dim()
        nt = self[time_dim].size
        nta = self.trans_att.size

        assert self.st.dims[0] == 'x', 'Stokes are transposed'
        assert self.ast.dims[0] == 'x', 'Stokes are transposed'

        if matching_sections:
            matching_indices = match_sections(self, matching_sections)
        else:
            matching_indices = None

        ix_sec = self.ufunc_per_section(x_indices=True, calc_per='all')
        assert not np.any(
            self.st.isel(x=ix_sec) <= 0.), \
            'There is uncontrolled noise in the ST signal. Are your sections' \
            'correctly defined?'
        assert not np.any(
            self.ast.isel(x=ix_sec) <= 0.), \
            'There is uncontrolled noise in the AST signal. Are your sections' \
            'correctly defined?'

        if method == 'ols' or method == 'wls':
            if method == 'ols':
                assert st_var is None and ast_var is None, ''
                st_var = None  # ols
                ast_var = None  # ols
                calc_cov = False
            else:
                for input_item in [st_var, ast_var]:
                    assert input_item is not None, 'For wls define all ' \
                                                   'variances (`st_var`, ' \
                                                   '`ast_var`) '

                calc_cov = True

            split = calibration_single_ended_solver(
                self,
                st_var,
                ast_var,
                calc_cov=calc_cov,
                solver='external_split',
                matching_indices=matching_indices)

            y = split['y']
            w = split['w']

            # Stack all X's
            if fix_alpha:
                assert not fix_dalpha, 'Use either `fix_dalpha` or `fix_alpha`'
                assert fix_alpha[0].size == self.x.size, 'fix_alpha also needs to be defined outside the reference ' \
                                                         'sections'
                assert fix_alpha[1].size == self.x.size, 'fix_alpha also needs to be defined outside the reference ' \
                                                         'sections'
                p_val = split['p0_est_alpha'].copy()

                if np.any(matching_indices):
                    raise NotImplementedError(
                        "Configuring fix_alpha and matching sections requires extra code"
                    )

                X = sp.hstack(
                    (
                        split['X_gamma'], split['X_alpha'], split['X_c'],
                        split['X_TA'])).tocsr()
                ip_use = list(range(1 + nx + nt + nta * nt))

            else:
                X = sp.vstack(
                    (
                        sp.hstack(
                            (
                                split['X_gamma'], split['X_dalpha'],
                                split['X_c'], split['X_TA'])),
                        split['X_m'])).tocsr()
                p_val = split['p0_est_dalpha'].copy()
                ip_use = list(range(1 + 1 + nt + nta * nt))

            p_var = np.zeros_like(p_val)
            p_cov = np.zeros((p_val.size, p_val.size), dtype=np.float)

            if fix_gamma is not None:
                ip_remove = [0]
                ip_use = [i for i in ip_use if i not in ip_remove]
                p_val[ip_remove] = fix_gamma[0]
                p_var[ip_remove] = fix_gamma[1]

                X_gamma = sp.vstack(
                    (split['X_gamma'],
                     split['X_m'].tocsr()[:, 0].tocoo())).toarray().flatten()

                y -= fix_gamma[0] * X_gamma
                w = 1 / (1 / w + fix_gamma[1] * X_gamma)

            if fix_alpha is not None:
                ip_remove = list(range(1, nx + 1))
                ip_use = [i for i in ip_use if i not in ip_remove]
                p_val[ip_remove] = fix_alpha[0]
                p_var[ip_remove] = fix_alpha[1]

                # X_alpha needs to be vertically extended to support matching sections
                y -= split['X_alpha'].dot(fix_alpha[0])
                w = 1 / (1 / w + split['X_alpha'].dot(fix_alpha[1]))

            if fix_dalpha is not None:
                ip_remove = [1]
                ip_use = [i for i in ip_use if i not in ip_remove]
                p_val[ip_remove] = fix_dalpha[0]
                p_var[ip_remove] = fix_dalpha[1]

                y -= np.hstack(
                    (
                        fix_dalpha[0] * split['X_dalpha'].toarray().flatten(),
                        (
                            fix_dalpha[0] * split['X_m'].tocsr()
                            [:, 1].tocoo().toarray().flatten())))
                w = 1 / (
                    1 / w + np.hstack(
                        (
                            fix_dalpha[1]
                            * split['X_dalpha'].toarray().flatten(), (
                                fix_dalpha[1] * split['X_m'].tocsr()
                                [:, 1].tocoo().toarray().flatten()))))

            if solver == 'sparse':
                out = wls_sparse(
                    X[:, ip_use],
                    y,
                    w=w,
                    x0=p_val[ip_use],
                    calc_cov=calc_cov,
                    verbose=False)

            elif solver == 'stats':
                out = wls_stats(
                    X[:, ip_use], y, w=w, calc_cov=calc_cov, verbose=False)

            p_val[ip_use] = out[0]
            p_var[ip_use] = out[1]

            if calc_cov:
                np.fill_diagonal(
                    p_cov, p_var)  # set variance of all fixed params
                p_cov[np.ix_(ip_use, ip_use)] = out[2]

        elif method == 'external':
            for input_item in [p_val, p_var, p_cov]:
                assert input_item is not None, \
                    'Define p_val, p_var, p_cov when using an external solver'

        elif method == 'external_split':
            raise ValueError('Not implemented yet')

        else:
            raise ValueError('Choose a valid method')

        # store calibration parameters in DataStore
        self[store_gamma] = (tuple(), p_val[0])

        if method == 'wls' or method == 'external':
            self[store_gamma + variance_suffix] = (tuple(), p_var[0])

        if nta > 0:
            ta = p_val[-nt * nta:].reshape((nt, nta), order='F')
            self[store_ta] = ((time_dim, 'trans_att'), ta[:, :])

            if method == 'wls' or method == 'external':
                tavar = p_var[-nt * nta:].reshape((nt, nta), order='F')
                self[store_ta + variance_suffix] = (
                    (time_dim, 'trans_att'), tavar[:, :])

        if fix_alpha:
            ic_start = 1 + nx
            self[store_c] = ((time_dim,), p_val[ic_start:nt + ic_start])
            self[store_alpha] = (('x',), fix_alpha[0])

            if method == 'wls' or method == 'external':
                self[store_c + variance_suffix] = (
                    (time_dim,), p_var[ic_start:nt + ic_start])
                self[store_alpha + variance_suffix] = (('x',), fix_alpha[1])
        else:
            self[store_c] = ((time_dim,), p_val[2:nt + 2])
            dalpha = p_val[1]
            self[store_dalpha] = (tuple(), dalpha)
            self[store_alpha] = (('x',), dalpha * self.x.data)

            if method == 'wls' or method == 'external':
                self[store_c
                     + variance_suffix] = ((time_dim,), p_var[2:nt + 2])
                dalpha_var = p_var[1]
                self[store_dalpha + variance_suffix] = (tuple(), dalpha_var)
                self[store_alpha
                     + variance_suffix] = (('x',), dalpha_var * self.x.data)

        # deal with FW
        if store_tmpf:
            ta_arr = np.zeros((nx, nt))
            if nta > 0:
                for tai, taxi in zip(self[store_ta].values.T,
                                     self.trans_att.values):
                    ta_arr[self.x.values >= taxi] = \
                        ta_arr[self.x.values >= taxi] + tai

            tempF_data = self.gamma.data / (
                (
                    np.log(self.st.data) - np.log(self.ast.data) +
                    (self.c.data[None, :] + ta_arr)) +
                (self.alpha.data[:, None])) - 273.15
            self[store_tmpf] = (('x', time_dim), tempF_data)

        if store_p_val:
            drop_vars = [
                k for k, v in self.items()
                if {'params1', 'params2'}.intersection(v.dims)]

            for k in drop_vars:
                del self[k]

            self[store_p_val] = (('params1',), p_val)

            if method == 'wls' or method == 'external':
                assert store_p_cov, 'Might as well store the covariance matrix. Already computed.'
                self[store_p_cov] = (('params1', 'params2'), p_cov)

        pass

    def calibration_double_ended(
            self,
            sections=None,
            st_var=None,
            ast_var=None,
            rst_var=None,
            rast_var=None,
            store_df='df',
            store_db='db',
            store_gamma='gamma',
            store_alpha='alpha',
            store_ta='talpha',
            store_tmpf='tmpf',
            store_tmpb='tmpb',
            store_tmpw='tmpw',
            tmpw_mc_size=50,
            store_p_cov='p_cov',
            store_p_val='p_val',
            variance_suffix='_var',
            method='wls',
            solver='sparse',
            p_val=None,
            p_var=None,
            p_cov=None,
            remove_mc_set_flag=True,
            reduce_memory_usage=False,
            trans_att=None,
            fix_gamma=None,
            fix_alpha=None,
            matching_sections=None,
            matching_indices=None,
            verbose=False,
            **kwargs):
        """
        Calibrate the Stokes (`ds.st`) and anti-Stokes (`ds.ast`) of the forward
        channel and from the backward channel (`ds.rst`, `ds.rast`) data to
        temperature using fiber sections with a known temperature
        (`ds.sections`) for double-ended setups. The calibrated temperature of
        the forward channel is stored under `ds.tmpf` and its variance under
        `ds.tmpf_var`, and that of the the backward channel under `ds.tmpb` and
        `ds.tmpb_var`. The inverse-variance weighted average of the forward and
        backward channel is stored under `ds.tmpw` and `ds.tmpw_var`.

        In double-ended setups, Stokes and anti-Stokes intensity is measured in
        two directions from both ends of the fiber. The forward-channel
        measurements are denoted with subscript F, and the backward-channel
        measurements are denoted with subscript B. Both measurement channels
        start at a different end of the fiber and have opposite directions, and
        therefore have different spatial coordinates. The first processing step
        with double-ended measurements is to align the measurements of the two
        measurement channels so that they have the same spatial coordinates. The
        spatial coordinate :math:`x` (m) is defined here positive in the forward
        direction, starting at 0 where the fiber is connected to the forward
        channel of the DTS system; the length of the fiber is :math:`L`.
        Consequently, the backward-channel measurements are flipped and shifted
        to align with the forward-channel measurements. Alignment of the
        measurements of the two channels is prone to error because it requires
        the exact fiber length (McDaniel et al., 2018). Depending on the DTS system
        used, the forward channel and backward channel are measured one after
        another by making use of an optical switch, so that only a single
        detector is needed. However, it is assumed in this paper that the
        forward channel and backward channel are measured simultaneously, so
        that the temperature of both measurements is the same. This assumption
        holds better for short acquisition times with respect to the time scale
        of the temperature variation, and when there is no systematic difference
        in temperature between the two channels. The temperature may be computed
        from the forward-channel measurements (Equation 10 [1]_) with:

        .. math::

            T_\mathrm{F} (x,t)  = \\frac{\gamma}{I_\mathrm{F}(x,t) + \
C_\mathrm{F}(t) + \int_0^x{\Delta\\alpha(x')\,\mathrm{d}x'}}

        and from the backward-channel measurements with:

        .. math::
            T_\mathrm{B} (x,t)  = \\frac{\gamma}{I_\mathrm{B}(x,t) + \
C_\mathrm{B}(t) + \int_x^L{\Delta\\alpha(x')\,\mathrm{d}x'}}

        with

        .. math::

            I(x,t) = \ln{\left(\\frac{P_+(x,t)}{P_-(x,t)}\\right)}


        .. math::

            C(t) = \ln{\left(\\frac{\eta_-(t)K_-/\lambda_-^4}{\eta_+(t)K_+/\lambda_+^4}\\right)}


        where :math:`C` is the lumped effect of the difference in gain at
        :math:`x=0` between Stokes and anti-Stokes intensity measurements and
        the dependence of the scattering intensity on the wavelength. The
        parameters :math:`P_+` and :math:`P_-` are the Stokes and anti-Stokes
        intensity measurements, respectively.
        :math:`C_\mathrm{F}(t)` and :math:`C_\mathrm{B}(t)` are the
        parameter :math:`C(t)` for the forward-channel and backward-channel
        measurements, respectively. :math:`C_\mathrm{B}(t)` may be different
        from :math:`C_\mathrm{F}(t)` due to differences in gain, and difference
        in the attenuation between the detectors and the point the fiber end is
        connected to the DTS system (:math:`\eta_+` and :math:`\eta_-` in
        Equation~\\ref{eqn:c}). :math:`T` in the listed
        equations is in Kelvin, but is converted to Celsius after calibration.
        The calibration procedure presented in van de
        Giesen et al. 2012 approximates :math:`C(t)` to be
        the same for the forward and backward-channel measurements, but this
        approximation is not made here.

        Parameter :math:`A(x)` (`ds.alpha`) is introduced to simplify the notation of the
        double-ended calibration procedure and represents the integrated
        differential attenuation between locations :math:`x_1` and :math:`x`
        along the fiber. Location :math:`x_1` is the first reference section
        location (the smallest x-value of all used reference sections).

        .. math::
            A(x) = \\int_{x_1}^x{\\Delta\\alpha(x')\\,\\mathrm{d}x'}

        so that the expressions for temperature may be written as:

        .. math::
            T_\mathrm{F} (x,t) = \\frac{\gamma}{I_\mathrm{F}(x,t) + D_\mathrm{F}(t) + A(x)},
            T_\mathrm{B} (x,t) = \\frac{\gamma}{I_\mathrm{B}(x,t) + D_\mathrm{B}(t) - A(x)}

        where

        .. math::
            D_{\mathrm{F}}(t) = C_{\mathrm{F}}(t) + \int_0^{x_1}{\Delta\\alpha(x')\,\mathrm{d}x'},
            D_{\mathrm{B}}(t) = C_{\mathrm{B}}(t) + \int_{x_1}^L{\Delta\\alpha(x')\,\mathrm{d}x'}

        Parameters :math:`D_\mathrm{F}` (`ds.df`) and :math:`D_\mathrm{B}`
        (`ds.db`) must be estimated for each time and are constant along the fiber, and parameter
        :math:`A` must be estimated for each location and is constant over time.
        The calibration procedure is discussed in Section 6.
        :math:`T_\mathrm{F}` (`ds.tmpf`) and :math:`T_\mathrm{B}` (`ds.tmpb`)
        are separate
        approximations of the same temperature at the same time. The estimated
        :math:`T_\mathrm{F}` is more accurate near :math:`x=0` because that is
        where the signal is strongest. Similarly, the estimated
        :math:`T_\mathrm{B}` is more accurate near :math:`x=L`. A single best
        estimate of the temperature is obtained from the weighted average of
        :math:`T_\mathrm{F}` and :math:`T_\mathrm{B}` as discussed in
        Section 7.2 [1]_ .

        Parameters
        ----------
        store_p_cov : str
            Key to store the covariance matrix of the calibrated parameters
        store_p_val : str
            Key to store the values of the calibrated parameters
        p_val : array-like, optional
            Define `p_val`, `p_var`, `p_cov` if you used an external function
            for calibration. Has size `1 + 2 * nt + nx + 2 * nt * nta`.
            First value is :math:`\gamma`, then `nt` times
            :math:`D_\mathrm{F}`, then `nt` times
            :math:`D_\mathrm{B}`, then for each location :math:`D_\mathrm{B}`,
            then for each connector that introduces directional attenuation two
            parameters per time step.
        p_var : array-like, optional
            Define `p_val`, `p_var`, `p_cov` if you used an external function
            for calibration. Has size `1 + 2 * nt + nx + 2 * nt * nta`.
            Is the variance of `p_val`.
        p_cov : array-like, optional
            The covariances of `p_val`. Square matrix.
            If set to False, no uncertainty in the parameters is propagated
            into the confidence intervals. Similar to the spec sheets of the DTS
            manufacturers. And similar to passing an array filled with zeros.
        sections : Dict[str, List[slice]], optional
            If `None` is supplied, `ds.sections` is used. Define calibration
            sections. Each section requires a reference temperature time series,
            such as the temperature measured by an external temperature sensor.
            They should already be part of the DataStore object. `sections`
            is defined with a dictionary with its keywords of the
            names of the reference temperature time series. Its values are
            lists of slice objects, where each slice object is a fiber stretch
            that has the reference temperature. Afterwards, `sections` is stored
            under `ds.sections`.
        st_var, ast_var, rst_var, rast_var : float, callable, array-like, optional
            The variance of the measurement noise of the Stokes signals in the
            forward direction. If `float` the variance of the noise from the
            Stokes detector is described with a single value.
            If `callable` the variance of the noise from the Stokes detector is
            a function of the intensity, as defined in the callable function.
            Or manually define a variance with a DataArray of the shape
            `ds.st.shape`, where the variance can be a function of time and/or
            x. Required if method is wls.
        store_df, store_db : str
            Label of where to store D. D is different for the forward channel
            and the backward channel
        store_gamma : str
            Label of where to store gamma
        store_alpha : str
            Label of where to store alpha
        store_ta : str
            Label of where to store transient alpha's
        store_tmpf : str
            Label of where to store the calibrated temperature of the forward
            direction
        store_tmpb : str
            Label of where to store the calibrated temperature of the
            backward direction
        store_tmpw : str
            Label of where to store the inverse-variance weighted average
            temperature of the forward and backward channel measurements.
        tmpw_mc_size : int
            The number of Monte Carlo samples drawn used to estimate the
            variance of the forward and backward channel temperature estimates
            and estimate the inverse-variance weighted average temperature.
        variance_suffix : str, optional
            String appended for storing the variance. Only used when method
            is wls.
        method : {'ols', 'wls', 'external'}
            Use `'ols'` for ordinary least squares and `'wls'` for weighted least
            squares. `'wls'` is the default, and there is currently no reason to
            use `'ols'`.
        solver : {'sparse', 'stats'}
            Either use the homemade weighted sparse solver or the weighted
            dense matrix solver of statsmodels. The sparse solver uses much less
            memory, is faster, and gives the same result as the statsmodels
            solver. The statsmodels solver is mostly used to check the sparse
            solver. `'stats'` is the default.
        transient_att_x, transient_asym_att_x : iterable, optional
            Depreciated. See trans_att
        trans_att : iterable, optional
            Splices can cause jumps in differential attenuation. Normal single
            ended calibration assumes these are not present. An additional loss
            term is added in the 'shadow' of the splice. Each location
            introduces an additional nt parameters to solve for. Requiring
            either an additional calibration section or matching sections.
            If multiple locations are defined, the losses are added.
        fix_gamma : Tuple[float, float], optional
            A tuple containing two floats. The first float is the value of
            gamma, and the second item is the variance of the estimate of gamma.
            Covariances between gamma and other parameters are not accounted
            for.
        fix_alpha : Tuple[array-like, array-like], optional
            A tuple containing two arrays. The first array contains the
            values of integrated differential att (:math:`A` in paper), and the
            second array contains the variance of the estimate of alpha.
            Covariances (in-) between alpha and other parameters are not
            accounted for.
        matching_sections : List[Tuple[slice, slice, bool]]
            Provide a list of tuples. A tuple per matching section. Each tuple
            has three items. The first two items are the slices of the sections
            that are matched. The third item is a boolean and is True if the two
            sections have a reverse direction ("J-configuration").
        matching_indices : array
            Provide an array of x-indices of size (npair, 2), where each pair
            has the same temperature. Used to improve the estimate of the
            integrated differential attenuation.
        verbose : bool
            Show additional calibration information


        Returns
        -------

        References
        ----------
        .. [1] des Tombe, B., Schilperoort, B., & Bakker, M. (2020). Estimation
            of Temperature and Associated Uncertainty from Fiber-Optic Raman-
            Spectrum Distributed Temperature Sensing. Sensors, 20(8), 2235.
            https://doi.org/10.3390/s20082235

        Examples
        --------
        - `Example notebook 8: Calibrate double ended <https://github.com/\
dtscalibration/python-dts-calibration/blob/master/examples/notebooks/\
08Calibrate_double_wls.ipynb>`_

        """
        self.check_deprecated_kwargs(kwargs)

        self.set_trans_att(trans_att=trans_att, **kwargs)

        if sections:
            self.sections = sections
        else:
            assert self.sections, 'sections are not defined'

        if method == 'wls':
            assert st_var is not None and ast_var is not None and rst_var is not None and rast_var is not None, 'Configure `st_var`'

        self.check_reference_section_values()

        nx = self.x.size
        time_dim = self.get_time_dim()
        nt = self[time_dim].size
        nta = self.trans_att.size
        ix_sec = self.ufunc_per_section(x_indices=True, calc_per='all')
        nx_sec = ix_sec.size

        assert self.st.dims[0] == 'x', 'Stokes are transposed'
        assert self.ast.dims[0] == 'x', 'Stokes are transposed'
        assert self.rst.dims[0] == 'x', 'Stokes are transposed'
        assert self.rast.dims[0] == 'x', 'Stokes are transposed'

        assert not np.any(
            self.st.isel(x=ix_sec) <= 0.), \
            'There is uncontrolled noise in the ST signal. Are your sections' \
            'correctly defined?'
        assert not np.any(
            self.ast.isel(x=ix_sec) <= 0.), \
            'There is uncontrolled noise in the AST signal. Are your sections' \
            'correctly defined?'
        assert not np.any(
            self.rst.isel(x=ix_sec) <= 0.), \
            'There is uncontrolled noise in the REV-ST signal. Are your ' \
            'sections correctly defined?'
        assert not np.any(
            self.rast.isel(x=ix_sec) <= 0.), \
            'There is uncontrolled noise in the REV-AST signal. Are your ' \
            'sections correctly defined?'

        if method == 'wls':
            for input_item in [st_var, ast_var, rst_var, rast_var]:
                assert input_item is not None, \
                    'For wls define all variances (`st_var`, `ast_var`,' +\
                    ' `rst_var`, `rast_var`)'

        if np.any(matching_indices):
            assert not matching_sections, \
                'Either define `matching_sections` or `matching_indices`'

        if matching_sections:
            assert not matching_indices, \
                'Either define `matching_sections` or `matching_indices'
            matching_indices = match_sections(self, matching_sections)

        if method == 'ols' or method == 'wls':
            if method == 'ols':
                calc_cov = False
            else:
                calc_cov = True

            if fix_alpha or fix_gamma:
                split = calibration_double_ended_solver(
                    self,
                    st_var,
                    ast_var,
                    rst_var,
                    rast_var,
                    calc_cov=calc_cov,
                    solver='external_split',
                    matching_indices=matching_indices,
                    verbose=verbose)
            else:
                out = calibration_double_ended_solver(
                    self,
                    st_var,
                    ast_var,
                    rst_var,
                    rast_var,
                    calc_cov=calc_cov,
                    solver=solver,
                    matching_indices=matching_indices,
                    verbose=verbose)

                if calc_cov:
                    p_val, p_var, p_cov = out
                else:
                    p_val, p_var = out

            # adjust split to fix parameters
            """Wrapped in a function to reduce memory usage.
            Constructing:
            Z_gamma (nt * nx, 1). Data: positive 1/temp
            Z_D (nt * nx, nt). Data: ones
            E (nt * nx, nx). Data: ones
            Zero_gamma (nt * nx, 1)
            zero_d (nt * nx, nt)
            Z_TA_fw (nt * nx, nta * 2 * nt) minus ones
            Z_TA_bw (nt * nx, nta * 2 * nt) minus ones
            Z_TA_E (nt * nx, nta * 2 * nt)

            I_fw = 1/Tref*gamma - D_fw - E - TA_fw
            I_bw = 1/Tref*gamma - D_bw + E - TA_bw
            (I_bw - I_fw) / 2 = D_fw/2 - D_bw/2 + E + TA_fw/2 - TA_bw/2 Eq42
            """
            if fix_alpha and fix_gamma:
                assert np.size(fix_alpha[0]) == self.x.size, \
                    'define alpha for each location'
                assert np.size(fix_alpha[1]) == self.x.size, \
                    'define var alpha for each location'
                m = 'The integrated differential attenuation is zero at the ' \
                    'first index of the reference sections.'
                assert np.abs(fix_alpha[0][ix_sec[0]]) < 1e-8, m
                # The array with the integrated differential att is termed E

                if np.any(matching_indices):
                    n_E_in_cal = split['ix_from_cal_match_to_glob'].size
                    p0_est = np.concatenate(
                        (
                            split['p0_est'][1:1 + 2 * nt],
                            split['p0_est'][1 + 2 * nt + n_E_in_cal:]))
                    X_E1 = sp.csr_matrix(
                        ([], ([], [])), shape=(nt * nx_sec, self.x.size))
                    X_E1[:, ix_sec[1:]] = split['E']
                    X_E2 = X_E1[:, split['ix_from_cal_match_to_glob']]
                    X_E = sp.vstack(
                        (
                            -X_E2, X_E2, split['E_match_F'],
                            split['E_match_B'], split['E_match_no_cal']))

                    X_gamma = sp.vstack(
                        (
                            split['Z_gamma'], split['Z_gamma'],
                            split['Zero_eq12_gamma'], split['Zero_eq12_gamma'],
                            split['Zero_eq3_gamma'])).toarray().flatten()

                    X = sp.vstack(
                        (
                            sp.hstack(
                                (
                                    -split['Z_D'], split['Zero_d'],
                                    split['Z_TA_fw'])),
                            sp.hstack(
                                (
                                    split['Zero_d'], -split['Z_D'],
                                    split['Z_TA_bw'])),
                            sp.hstack(
                                (split['Zero_d_eq12'], split['Z_TA_eq1'])),
                            sp.hstack(
                                (split['Zero_d_eq12'], split['Z_TA_eq2'])),
                            sp.hstack((split['d_no_cal'], split['Z_TA_eq3']))))

                    y = np.concatenate(
                        (
                            split['y_F'], split['y_B'], split['y_eq1'],
                            split['y_eq2'], split['y_eq3']))
                    y -= X_E.dot(
                        fix_alpha[0][split['ix_from_cal_match_to_glob']])
                    y -= fix_gamma[0] * X_gamma

                    # variances are added. weight is the inverse of the variance
                    # of the observations
                    if method == 'wls':
                        w_ = np.concatenate(
                            (
                                split['w_F'], split['w_B'], split['w_eq1'],
                                split['w_eq2'], split['w_eq3']))
                        w = 1 / (
                            1 / w_ + X_E.dot(
                                fix_alpha[1][split['ix_from_cal_match_to_glob']])
                            + fix_gamma[1] * X_gamma)

                    else:
                        w = 1.

                else:
                    # X_gamma
                    X_E = sp.vstack((-split['E'], split['E']))
                    X_gamma = sp.vstack(
                        (split['Z_gamma'],
                         split['Z_gamma'])).toarray().flatten()
                    # Use only the remaining coefficients
                    # Stack all X's
                    X = sp.vstack(
                        (
                            sp.hstack(
                                (
                                    -split['Z_D'], split['Zero_d'],
                                    split['Z_TA_fw'])),
                            sp.hstack(
                                (
                                    split['Zero_d'], -split['Z_D'],
                                    split['Z_TA_bw']))))

                    # Move the coefficients times the fixed gamma to the
                    # observations
                    y = np.concatenate((split['y_F'], split['y_B']))
                    y -= X_E.dot(fix_alpha[0][ix_sec[1:]])
                    y -= fix_gamma[0] * X_gamma
                    # variances are added. weight is the inverse of the variance
                    # of the observations
                    if method == 'wls':
                        w_ = np.concatenate((split['w_F'], split['w_B']))
                        w = 1 / (
                            1 / w_ + X_E.dot(fix_alpha[1][ix_sec[1:]])
                            + fix_gamma[1] * X_gamma)

                    else:
                        w = 1.

                    # [C_1, C_2, .., C_nt, TA_fw_a_1, TA_fw_a_2, TA_fw_a_nt,
                    # TA_bw_a_1, TA_bw_a_2, TA_bw_a_nt] Then continues with
                    # TA for connector b.
                    p0_est = np.concatenate(
                        (
                            split['p0_est'][1:1 + 2 * nt],
                            split['p0_est'][1 + 2 * nt + nx_sec - 1:]))

                if solver == 'sparse':
                    out = wls_sparse(
                        X, y, w=w, x0=p0_est, calc_cov=calc_cov, verbose=False)

                elif solver == 'stats':
                    out = wls_stats(
                        X, y, w=w, calc_cov=calc_cov, verbose=False)

                # Added fixed gamma and its variance to the solution
                p_val = np.concatenate(
                    (
                        [fix_gamma[0]], out[0][:2 * nt], fix_alpha[0],
                        out[0][2 * nt:]))
                p_var = np.concatenate(
                    (
                        [fix_gamma[1]], out[1][:2 * nt], fix_alpha[1],
                        out[1][2 * nt:]))

                if calc_cov:
                    # whether it returns a copy or a view depends on what
                    # version of numpy you are using
                    p_cov = np.diag(p_var).copy()
                    from_i = np.concatenate(
                        (
                            np.arange(1, 2 * nt + 1),
                            np.arange(
                                1 + 2 * nt + nx,
                                1 + 2 * nt + nx + nta * nt * 2)))
                    iox_sec1, iox_sec2 = np.meshgrid(
                        from_i, from_i, indexing='ij')
                    p_cov[iox_sec1, iox_sec2] = out[2]

            elif fix_gamma:
                if np.any(matching_indices):
                    # n_E_in_cal = split['ix_from_cal_match_to_glob'].size
                    p0_est = split['p0_est'][1:]
                    X_E1 = sp.csr_matrix(
                        ([], ([], [])), shape=(nt * nx_sec, self.x.size))
                    from_i = ix_sec[1:]
                    X_E1[:, from_i] = split['E']
                    X_E2 = X_E1[:, split['ix_from_cal_match_to_glob']]
                    X = sp.vstack(
                        (
                            sp.hstack(
                                (
                                    -split['Z_D'], split['Zero_d'], -X_E2,
                                    split['Z_TA_fw'])),
                            sp.hstack(
                                (
                                    split['Zero_d'], -split['Z_D'], X_E2,
                                    split['Z_TA_bw'])),
                            sp.hstack(
                                (
                                    split['Zero_d_eq12'], split['E_match_F'],
                                    split['Z_TA_eq1'])),
                            sp.hstack(
                                (
                                    split['Zero_d_eq12'], split['E_match_B'],
                                    split['Z_TA_eq2'])),
                            sp.hstack(
                                (
                                    split['d_no_cal'], split['E_match_no_cal'],
                                    split['Z_TA_eq3']))))
                    X_gamma = sp.vstack(
                        (
                            split['Z_gamma'], split['Z_gamma'],
                            split['Zero_eq12_gamma'], split['Zero_eq12_gamma'],
                            split['Zero_eq3_gamma'])).toarray().flatten()

                    y = np.concatenate(
                        (
                            split['y_F'], split['y_B'], split['y_eq1'],
                            split['y_eq2'], split['y_eq3']))
                    y -= fix_gamma[0] * X_gamma

                    # variances are added. weight is the inverse of the variance
                    # of the observations
                    if method == 'wls':
                        w_ = np.concatenate(
                            (
                                split['w_F'], split['w_B'], split['w_eq1'],
                                split['w_eq2'], split['w_eq3']))
                        w = 1 / (1 / w_ + fix_gamma[1] * X_gamma)

                    else:
                        w = 1.

                else:
                    X_gamma = sp.vstack(
                        (split['Z_gamma'],
                         split['Z_gamma'])).toarray().flatten()
                    # Use only the remaining coefficients
                    X = sp.vstack(
                        (
                            sp.hstack(
                                (
                                    -split['Z_D'], split['Zero_d'],
                                    -split['E'], split['Z_TA_fw'])),
                            sp.hstack(
                                (
                                    split['Zero_d'], -split['Z_D'], split['E'],
                                    split['Z_TA_bw']))))
                    # Move the coefficients times the fixed gamma to the
                    # observations
                    y = np.concatenate((split['y_F'], split['y_B']))
                    y -= fix_gamma[0] * X_gamma
                    # variances are added. weight is the inverse of the variance
                    # of the observations
                    if method == 'wls':
                        w_ = np.concatenate((split['w_F'], split['w_B']))
                        w = 1 / (1 / w_ + fix_gamma[1] * X_gamma)

                    else:
                        w = 1.
                    p0_est = split['p0_est'][1:]

                if solver == 'sparse':
                    out = wls_sparse(
                        X, y, w=w, x0=p0_est, calc_cov=calc_cov, verbose=False)

                elif solver == 'stats':
                    out = wls_stats(
                        X, y, w=w, calc_cov=calc_cov, verbose=False)

                # put E outside of reference section in solution
                # concatenating makes a copy of the data instead of using a
                # pointer
                ds_sub = self[['st', 'ast', 'rst', 'rast', 'trans_att']]
                ds_sub['df'] = (('time',), out[0][:nt])
                ds_sub['df_var'] = (('time',), out[1][:nt])
                ds_sub['db'] = (('time',), out[0][nt:2 * nt])
                ds_sub['db_var'] = (('time',), out[1][nt:2 * nt])

                if nta > 0:
                    if np.any(matching_indices):
                        n_E_in_cal = split['ix_from_cal_match_to_glob'].size
                        ta = out[0][2 * nt + n_E_in_cal:].reshape(
                            (nt, 2, nta), order='F')
                        ta_var = out[1][2 * nt + n_E_in_cal:].reshape(
                            (nt, 2, nta), order='F')

                    else:
                        ta = out[0][2 * nt + nx_sec - 1:].reshape(
                            (nt, 2, nta), order='F')
                        ta_var = out[1][2 * nt + nx_sec - 1:].reshape(
                            (nt, 2, nta), order='F')

                    talpha_fw = ta[:, 0, :]
                    talpha_bw = ta[:, 1, :]
                    talpha_fw_var = ta_var[:, 0, :]
                    talpha_bw_var = ta_var[:, 1, :]
                else:
                    talpha_fw = None
                    talpha_bw = None
                    talpha_fw_var = None
                    talpha_bw_var = None

                E_all_exact, E_all_var_exact = calc_alpha_double(
                    'exact',
                    ds_sub,
                    st_var,
                    ast_var,
                    rst_var,
                    rast_var,
                    'df',
                    'db',
                    'df_var',
                    'db_var',
                    ix_alpha_is_zero=ix_sec[0],
                    talpha_fw=talpha_fw,
                    talpha_bw=talpha_bw,
                    talpha_fw_var=talpha_fw_var,
                    talpha_bw_var=talpha_bw_var)

                if not np.any(matching_indices):
                    # Added fixed gamma and its variance to the solution. And
                    # expand to include locations outside reference sections.
                    p_val = np.concatenate(
                        (
                            [fix_gamma[0]], out[0][:2 * nt], E_all_exact,
                            out[0][2 * nt + nx_sec - 1:]))
                    p_val[1 + 2 * nt + ix_sec[1:]] = out[0][2 * nt:2 * nt
                                                            + nx_sec - 1]
                    p_val[1 + 2 * nt + ix_sec[0]] = 0.
                    p_var = np.concatenate(
                        (
                            [fix_gamma[1]], out[1][:2 * nt], E_all_var_exact,
                            out[1][2 * nt + nx_sec - 1:]))
                    p_var[1 + 2 * nt + ix_sec[1:]] = out[1][2 * nt:2 * nt
                                                            + nx_sec - 1]
                else:
                    n_E_in_cal = split['ix_from_cal_match_to_glob'].size

                    # Added fixed gamma and its variance to the solution. And
                    # expand to include locations outside reference sections.
                    p_val = np.concatenate(
                        (
                            [fix_gamma[0]], out[0][:2 * nt], E_all_exact,
                            out[0][2 * nt + n_E_in_cal:]))
                    p_val[1 + 2 * nt + split['ix_from_cal_match_to_glob']] = \
                        out[0][2 * nt:2 * nt + n_E_in_cal]
                    p_val[1 + 2 * nt + ix_sec[0]] = 0.
                    p_var = np.concatenate(
                        (
                            [fix_gamma[1]], out[1][:2 * nt], E_all_var_exact,
                            out[1][2 * nt + n_E_in_cal:]))
                    p_var[1 + 2 * nt + split['ix_from_cal_match_to_glob']] = \
                        out[1][2 * nt:2 * nt + n_E_in_cal]

                if calc_cov:
                    p_cov = np.diag(p_var).copy()

                    if not np.any(matching_indices):
                        from_i = np.concatenate(
                            (
                                np.arange(1,
                                          2 * nt + 1), 2 * nt + 1 + ix_sec[1:],
                                np.arange(
                                    1 + 2 * nt + nx,
                                    1 + 2 * nt + nx + nta * nt * 2)))
                    else:
                        from_i = np.concatenate(
                            (
                                np.arange(1, 2 * nt + 1), 2 * nt + 1
                                + split['ix_from_cal_match_to_glob'],
                                np.arange(
                                    1 + 2 * nt + nx,
                                    1 + 2 * nt + nx + nta * nt * 2)))

                    iox_sec1, iox_sec2 = np.meshgrid(
                        from_i, from_i, indexing='ij')
                    p_cov[iox_sec1, iox_sec2] = out[2]

            elif fix_alpha:
                assert np.size(fix_alpha[0]) == self.x.size, \
                    'define alpha for each location'
                assert np.size(fix_alpha[1]) == self.x.size, \
                    'define var alpha for each location'
                m = 'The integrated differential attenuation is zero at the ' \
                    'first index of the reference sections.'
                assert np.abs(fix_alpha[0][ix_sec[0]]) < 1e-6, m
                # The array with the integrated differential att is termed E

                if not np.any(matching_indices):
                    # X_gamma
                    X_E = sp.vstack((-split['E'], split['E']))
                    # Use only the remaining coefficients
                    # Stack all X's
                    X = sp.vstack(
                        (
                            sp.hstack(
                                (
                                    split['Z_gamma'], -split['Z_D'],
                                    split['Zero_d'], split['Z_TA_fw'])),
                            sp.hstack(
                                (
                                    split['Z_gamma'], split['Zero_d'],
                                    -split['Z_D'], split['Z_TA_bw']))))
                    # Move the coefficients times the fixed gamma to the
                    # observations
                    y = np.concatenate((split['y_F'], split['y_B']))
                    y -= X_E.dot(fix_alpha[0][ix_sec[1:]])

                    # variances are added. weight is the inverse of the variance
                    # of the observations
                    if method == 'wls':
                        w_ = np.concatenate((split['w_F'], split['w_B']))
                        w = 1 / (1 / w_ + X_E.dot(fix_alpha[1][ix_sec[1:]]))

                    else:
                        w = 1.

                    p0_est = np.concatenate(
                        (
                            split['p0_est'][:1 + 2 * nt],
                            split['p0_est'][1 + 2 * nt + nx_sec - 1:]))

                else:
                    n_E_in_cal = split['ix_from_cal_match_to_glob'].size
                    p0_est = np.concatenate(
                        (
                            split['p0_est'][:1 + 2 * nt],
                            split['p0_est'][1 + 2 * nt + n_E_in_cal:]))
                    X_E1 = sp.csr_matrix(
                        ([], ([], [])), shape=(nt * nx_sec, self.x.size))
                    X_E1[:, ix_sec[1:]] = split['E']
                    X_E2 = X_E1[:, split['ix_from_cal_match_to_glob']]
                    X_E = sp.vstack(
                        (
                            -X_E2, X_E2, split['E_match_F'],
                            split['E_match_B'], split['E_match_no_cal']))

                    X = sp.vstack(
                        (
                            sp.hstack(
                                (
                                    split['Z_gamma'], -split['Z_D'],
                                    split['Zero_d'], split['Z_TA_fw'])),
                            sp.hstack(
                                (
                                    split['Z_gamma'], split['Zero_d'],
                                    -split['Z_D'], split['Z_TA_bw'])),
                            sp.hstack(
                                (
                                    split['Zero_eq12_gamma'],
                                    split['Zero_d_eq12'], split['Z_TA_eq1'])),
                            sp.hstack(
                                (
                                    split['Zero_eq12_gamma'],
                                    split['Zero_d_eq12'], split['Z_TA_eq2'])),
                            sp.hstack(
                                (
                                    split['Zero_eq3_gamma'], split['d_no_cal'],
                                    split['Z_TA_eq3']))))

                    y = np.concatenate(
                        (
                            split['y_F'], split['y_B'], split['y_eq1'],
                            split['y_eq2'], split['y_eq3']))
                    y -= X_E.dot(
                        fix_alpha[0][split['ix_from_cal_match_to_glob']])

                    # variances are added. weight is the inverse of the variance
                    # of the observations
                    if method == 'wls':
                        w_ = np.concatenate(
                            (
                                split['w_F'], split['w_B'], split['w_eq1'],
                                split['w_eq2'], split['w_eq3']))
                        w = 1 / (
                            1 / w_ + X_E.dot(
                                fix_alpha[1][
                                    split['ix_from_cal_match_to_glob']]))

                    else:
                        w = 1.

                if solver == 'sparse':
                    out = wls_sparse(
                        X, y, w=w, x0=p0_est, calc_cov=calc_cov, verbose=False)

                elif solver == 'stats':
                    out = wls_stats(
                        X, y, w=w, calc_cov=calc_cov, verbose=False)

                # Added fixed gamma and its variance to the solution
                p_val = np.concatenate(
                    (out[0][:1 + 2 * nt], fix_alpha[0], out[0][1 + 2 * nt:]))
                p_var = np.concatenate(
                    (out[1][:1 + 2 * nt], fix_alpha[1], out[1][1 + 2 * nt:]))

                if calc_cov:
                    p_cov = np.diag(p_var).copy()

                    from_i = np.concatenate(
                        (
                            np.arange(1 + 2 * nt),
                            np.arange(
                                1 + 2 * nt + nx,
                                1 + 2 * nt + nx + nta * nt * 2)))

                    iox_sec1, iox_sec2 = np.meshgrid(
                        from_i, from_i, indexing='ij')
                    p_cov[iox_sec1, iox_sec2] = out[2]

            else:
                pass

        elif method == 'external':
            for input_item in [p_val, p_var, p_cov]:
                assert input_item is not None

            calc_cov = True

        elif method == 'external_split':
            raise ValueError('Not implemented yet')

        else:
            raise ValueError('Choose a valid method')

        # all below require the following solution sizes
        npar = 1 + 2 * nt + nx + 2 * nt * nta
        assert p_val.size == npar
        assert p_var.size == npar
        if calc_cov:
            assert p_cov.shape == (npar, npar)

        gamma = p_val[0]
        d_fw = p_val[1:nt + 1]
        d_bw = p_val[1 + nt:1 + 2 * nt]
        alpha = p_val[1 + 2 * nt:1 + 2 * nt + nx]

        # store calibration parameters in DataStore
        self[store_gamma] = (tuple(), gamma)
        self[store_alpha] = (('x',), alpha)
        self[store_df] = ((time_dim,), d_fw)
        self[store_db] = ((time_dim,), d_bw)

        if nta > 0:
            ta = p_val[1 + 2 * nt + nx:].reshape((nt, 2, nta), order='F')
            self[store_ta + '_fw'] = ((time_dim, 'trans_att'), ta[:, 0, :])
            self[store_ta + '_bw'] = ((time_dim, 'trans_att'), ta[:, 1, :])

        # store variances in DataStore
        if method == 'wls' or method == 'external':
            # the variances only have meaning if the observations are weighted
            gammavar = p_var[0]
            dfvar = p_var[1:nt + 1]
            dbvar = p_var[1 + nt:1 + 2 * nt]
            alphavar = p_var[2 * nt + 1:2 * nt + 1 + nx]

            self[store_gamma + variance_suffix] = (tuple(), gammavar)
            self[store_alpha + variance_suffix] = (('x',), alphavar)
            self[store_df + variance_suffix] = ((time_dim,), dfvar)
            self[store_db + variance_suffix] = ((time_dim,), dbvar)

            if nta > 0:
                # neglecting the covariances. Better include them
                tavar = p_var[2 * nt + 1 + nx:].reshape(
                    (nt, 2, nta), order='F')
                self[store_ta + '_fw' + variance_suffix] = (
                    (time_dim, 'trans_att'), tavar[:, 0, :])
                self[store_ta + '_bw' + variance_suffix] = (
                    (time_dim, 'trans_att'), tavar[:, 1, :])

        # deal with FW
        if store_tmpf or (store_tmpw and method == 'ols'):
            ta_arr = np.zeros((nx, nt))
            if nta > 0:
                for tai, taxi in zip(self[store_ta + '_fw'].values.T,
                                     self.trans_att.values):
                    ta_arr[self.x.values >= taxi] = \
                        ta_arr[self.x.values >= taxi] + tai

            tempF_data = gamma / (
                np.log(self.st.data / self.ast.data) + d_fw + alpha[:, None]
                + ta_arr) - 273.15
            self[store_tmpf] = (('x', time_dim), tempF_data)

        # deal with BW
        if store_tmpb or (store_tmpw and method == 'ols'):
            ta_arr = np.zeros((nx, nt))
            if nta > 0:
                for tai, taxi in zip(self[store_ta + '_bw'].values.T,
                                     self.trans_att.values):
                    ta_arr[self.x.values < taxi] = \
                        ta_arr[self.x.values < taxi] + tai
            tempB_data = gamma / (
                np.log(self.rst.data / self.rast.data) + d_bw - alpha[:, None]
                + ta_arr) - 273.15
            self[store_tmpb] = (('x', time_dim), tempB_data)

        if store_tmpw and (method == 'wls' or method == 'external'):
            self.conf_int_double_ended(
                p_val=p_val,
                p_cov=p_cov,
                store_ta=store_ta if self.trans_att.size > 0 else None,
                st_var=st_var,
                ast_var=ast_var,
                rst_var=rst_var,
                rast_var=rast_var,
                store_tmpf='',
                store_tmpb='',
                store_tmpw=store_tmpw,
                store_tempvar=variance_suffix,
                conf_ints=[],
                mc_sample_size=tmpw_mc_size,
                da_random_state=None,
                remove_mc_set_flag=remove_mc_set_flag,
                reduce_memory_usage=reduce_memory_usage)

        elif store_tmpw and method == 'ols':
            self[store_tmpw] = (self[store_tmpf] + self[store_tmpb]) / 2
        else:
            pass

        if store_p_val:
            drop_vars = [
                k for k, v in self.items()
                if {'params1', 'params2'}.intersection(v.dims)]

            for k in drop_vars:
                del self[k]

            self[store_p_val] = (('params1',), p_val)

            if method == 'wls' or method == 'external':
                assert store_p_cov, 'Might as well store the covariance matrix. Already computed.'
                self[store_p_cov] = (('params1', 'params2'), p_cov)

        pass

    def conf_int_single_ended(
            self,
            p_val='p_val',
            p_cov='p_cov',
            st_var=None,
            ast_var=None,
            store_tmpf='tmpf',
            store_tempvar='_var',
            conf_ints=None,
            mc_sample_size=100,
            da_random_state=None,
            remove_mc_set_flag=True,
            reduce_memory_usage=False,
            **kwargs):
        """
        Estimation of the confidence intervals for the temperatures measured
        with a single-ended setup. It consists of five steps. First, the variances
        of the Stokes and anti-Stokes intensity measurements are estimated
        following the steps in Section 4 [1]_. A Normal
        distribution is assigned to each intensity measurement that is centered
        at the measurement and using the estimated variance. Second, a multi-
        variate Normal distribution is assigned to the estimated parameters
        using the covariance matrix from the calibration procedure presented in
        Section 5 [1]_. Third, the distributions are sampled, and the
        temperature is computed with Equation 12 [1]_. Fourth, step
        three is repeated, e.g., 10,000 times for each location and for each
        time. The resulting 10,000 realizations of the temperatures
        approximate the probability density functions of the estimated
        temperature at that location and time. Fifth, the standard uncertainties
        are computed with the standard deviations of the realizations of the
        temperatures, and the 95\% confidence intervals are computed from the
        2.5\% and 97.5\% percentiles of the realizations of the temperatures.


        Parameters
        ----------
        p_val : array-like, optional
            Define `p_val`, `p_var`, `p_cov` if you used an external function
            for calibration. Has size 2 + `nt`. First value is :math:`\gamma`,
            second is :math:`\Delta \\alpha`, others are :math:`C` for each
            timestep.
            If set to False, no uncertainty in the parameters is propagated
            into the confidence intervals. Similar to the spec sheets of the DTS
            manufacturers. And similar to passing an array filled with zeros
        p_cov : array-like, optional
            The covariances of `p_val`.
        st_var, ast_var : float, callable, array-like, optional
            The variance of the measurement noise of the Stokes signals in the
            forward direction. If `float` the variance of the noise from the
            Stokes detector is described with a single value.
            If `callable` the variance of the noise from the Stokes detector is
            a function of the intensity, as defined in the callable function.
            Or manually define a variance with a DataArray of the shape
            `ds.st.shape`, where the variance can be a function of time and/or
            x. Required if method is wls.
        store_tmpf : str
            Key of how to store the Forward calculated temperature. Is
            calculated using the
            forward Stokes and anti-Stokes observations.
        store_tempvar : str
            a string that is appended to the store_tmp_ keys. and the
            variance is calculated
            for those store_tmp_ keys
        conf_ints : iterable object of float
            A list with the confidence boundaries that are calculated. Valid
            values are between
            [0, 1].
        mc_sample_size : int
            Size of the monte carlo parameter set used to calculate the
            confidence interval
        da_random_state
            For testing purposes. Similar to random seed. The seed for dask.
            Makes random not so random. To produce reproducable results for
            testing environments.
        remove_mc_set_flag : bool
            Remove the monte carlo data set, from which the CI and the
            variance are calculated.
        reduce_memory_usage : bool
            Use less memory but at the expense of longer computation time


        References
        ----------
        .. [1] des Tombe, B., Schilperoort, B., & Bakker, M. (2020). Estimation
            of Temperature and Associated Uncertainty from Fiber-Optic Raman-
            Spectrum Distributed Temperature Sensing. Sensors, 20(8), 2235.
            https://doi.org/10.3390/s20082235
        """
        self.check_deprecated_kwargs(kwargs)

        if da_random_state:
            state = da_random_state
        else:
            state = da.random.RandomState()

        time_dim = self.get_time_dim(data_var_key='st')

        no, nt = self.st.data.shape
        if 'trans_att' in self.keys():
            nta = self.trans_att.size
        else:
            nta = 0

        assert isinstance(p_val, (str, np.ndarray, np.generic))
        if isinstance(p_val, str):
            p_val = self[p_val].data

        npar = p_val.size

        # number of parameters
        if npar == nt + 2 + nt * nta:
            fixed_alpha = False
        elif npar == 1 + no + nt + nt * nta:
            fixed_alpha = True
        else:
            raise Exception('The size of `p_val` is not what I expected')

        self.coords['mc'] = range(mc_sample_size)

        if conf_ints:
            self.coords['CI'] = conf_ints

        # WLS
        if isinstance(p_cov, str):
            p_cov = self[p_cov].data
        assert p_cov.shape == (npar, npar)

        p_mc = sst.multivariate_normal.rvs(
            mean=p_val, cov=p_cov, size=mc_sample_size)

        if fixed_alpha:
            self['alpha_mc'] = (('mc', 'x'), p_mc[:, 1:no + 1])
            self['c_mc'] = (('mc', time_dim), p_mc[:, 1 + no:1 + no + nt])
        else:
            self['dalpha_mc'] = (('mc',), p_mc[:, 1])
            self['c_mc'] = (('mc', time_dim), p_mc[:, 2:nt + 2])

        self['gamma_mc'] = (('mc',), p_mc[:, 0])
        if nta:
            self['ta_mc'] = (
                ('mc', 'trans_att', time_dim),
                np.reshape(p_mc[:, -nt * nta:], (mc_sample_size, nta, nt)))

        rsize = (self.mc.size, self.x.size, self.time.size)

        if reduce_memory_usage:
            memchunk = da.ones(
                (mc_sample_size, no, nt), chunks={
                    0: -1,
                    1: 1,
                    2: 'auto'}).chunks
        else:
            memchunk = da.ones(
                (mc_sample_size, no, nt), chunks={
                    0: -1,
                    1: 'auto',
                    2: 'auto'}).chunks

        # Draw from the normal distributions for the Stokes intensities
        for k, st_labeli, st_vari in zip(['r_st', 'r_ast'], ['st', 'ast'],
                                         [st_var, ast_var]):

            # Load the mean as chunked Dask array, otherwise eats memory
            if type(self[st_labeli].data) == da.core.Array:
                loc = da.asarray(self[st_labeli].data, chunks=memchunk[1:])
            else:
                loc = da.from_array(self[st_labeli].data, chunks=memchunk[1:])

            # Make sure variance is of size (no, nt)
            if np.size(st_vari) > 1:
                if st_vari.shape == self[st_labeli].shape:
                    pass
                else:
                    st_vari = np.broadcast_to(st_vari, (no, nt))
            else:
                pass

            # Load variance as chunked Dask array, otherwise eats memory
            if type(st_vari) == da.core.Array:
                st_vari_da = da.asarray(st_vari, chunks=memchunk[1:])

            elif (callable(st_vari) and
                  type(self[st_labeli].data) == da.core.Array):
                st_vari_da = da.asarray(
                    st_vari(self[st_labeli]).data, chunks=memchunk[1:])

            elif (callable(st_vari) and
                  type(self[st_labeli].data) != da.core.Array):
                st_vari_da = da.from_array(
                    st_vari(self[st_labeli]).data, chunks=memchunk[1:])

            else:
                st_vari_da = da.from_array(st_vari, chunks=memchunk[1:])

            self[k] = (
                ('mc', 'x', time_dim),
                state.normal(
                    loc=loc,  # has chunks=memchunk[1:]
                    scale=st_vari_da**0.5,
                    size=rsize,
                    chunks=memchunk))

        ta_arr = np.zeros((mc_sample_size, no, nt))

        if nta:
            for ii, ta in enumerate(self['ta_mc']):
                for tai, taxi in zip(ta.values, self.trans_att.values):
                    ta_arr[ii, self.x.values >= taxi] = \
                        ta_arr[ii, self.x.values >= taxi] + tai
        self['ta_mc_arr'] = (('mc', 'x', time_dim), ta_arr)

        if fixed_alpha:
            self[store_tmpf + '_mc_set'] = self['gamma_mc'] / (
                (
                    np.log(self['r_st']) - np.log(self['r_ast']) +
                    (self['c_mc'] + self['ta_mc_arr']))
                + self['alpha_mc']) - 273.15
        else:
            self[store_tmpf + '_mc_set'] = self['gamma_mc'] / (
                (
                    np.log(self['r_st']) - np.log(self['r_ast']) +
                    (self['c_mc'] + self['ta_mc_arr'])) +
                (self['dalpha_mc'] * self.x)) - 273.15

        avg_dims = ['mc']

        avg_axis = self[store_tmpf + '_mc_set'].get_axis_num(avg_dims)

        self[store_tmpf + '_mc' + store_tempvar] = (
            self[store_tmpf + '_mc_set'] - self[store_tmpf]).var(
                dim=avg_dims, ddof=1)

        if conf_ints:
            new_chunks = (
                (len(conf_ints),),) + self[store_tmpf + '_mc_set'].chunks[1:]

            qq = self[store_tmpf + '_mc_set']

            q = qq.data.map_blocks(
                lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
                chunks=new_chunks,  #
                drop_axis=avg_axis,  # avg dimesnions are dropped from input arr
                new_axis=0)  # The new CI dimension is added as first axis

            self[store_tmpf + '_mc'] = (('CI', 'x', time_dim), q)

        if remove_mc_set_flag:
            drop_var = [
                'gamma_mc', 'dalpha_mc', 'alpha_mc', 'c_mc', 'mc', 'r_st',
                'r_ast', store_tmpf + '_mc_set', 'ta_mc_arr']
            for k in drop_var:
                if k in self:
                    del self[k]

        pass

    def average_single_ended(
            self,
            p_val='p_val',
            p_cov='p_cov',
            st_var=None,
            ast_var=None,
            store_tmpf='tmpf',
            store_tempvar='_var',
            conf_ints=None,
            mc_sample_size=100,
            ci_avg_time_flag1=False,
            ci_avg_time_flag2=False,
            ci_avg_time_sel=None,
            ci_avg_time_isel=None,
            ci_avg_x_flag1=False,
            ci_avg_x_flag2=False,
            ci_avg_x_sel=None,
            ci_avg_x_isel=None,
            var_only_sections=None,
            da_random_state=None,
            remove_mc_set_flag=True,
            reduce_memory_usage=False,
            **kwargs):
        """
        Average temperatures from single-ended setups.

        Four types of averaging are implemented. Please see Example Notebook 16.


        Parameters
        ----------
        p_val : array-like, optional
            Define `p_val`, `p_var`, `p_cov` if you used an external function
            for calibration. Has size 2 + `nt`. First value is :math:`\gamma`,
            second is :math:`\Delta \\alpha`, others are :math:`C` for each
            timestep.
            If set to False, no uncertainty in the parameters is propagated
            into the confidence intervals. Similar to the spec sheets of the DTS
            manufacturers. And similar to passing an array filled with zeros
        p_cov : array-like, optional
            The covariances of `p_val`.
        st_var, ast_var : float, callable, array-like, optional
            The variance of the measurement noise of the Stokes signals in the
            forward direction. If `float` the variance of the noise from the
            Stokes detector is described with a single value.
            If `callable` the variance of the noise from the Stokes detector is
            a function of the intensity, as defined in the callable function.
            Or manually define a variance with a DataArray of the shape
            `ds.st.shape`, where the variance can be a function of time and/or
            x. Required if method is wls.
        store_tmpf : str
            Key of how to store the Forward calculated temperature. Is
            calculated using the
            forward Stokes and anti-Stokes observations.
        store_tempvar : str
            a string that is appended to the store_tmp_ keys. and the
            variance is calculated
            for those store_tmp_ keys
        store_ta : str
            Key of how transient attenuation parameters are stored. Default
            is `talpha`. `_fw` and `_bw` is appended to for the forward and
            backward parameters. The `transient_asym_att_x` is derived from
            the `coords` of this DataArray. The `coords` of `ds[store_ta +
            '_fw']` should be ('time', 'trans_att').
        conf_ints : iterable object of float
            A list with the confidence boundaries that are calculated. Valid
            values are between
            [0, 1].
        mc_sample_size : int
            Size of the monte carlo parameter set used to calculate the
            confidence interval
        ci_avg_time_flag1 : bool
            The confidence intervals differ each time step. Assumes the
            temperature varies during the measurement period. Computes the
            arithmic temporal mean. If you would like to know the confidence
            interfal of:
            (1) a single additional measurement. So you can state "if another
            measurement were to be taken, it would have this ci"
            (2) all measurements. So you can state "The temperature remained
            during the entire measurement period between these ci bounds".
            Adds store_tmpw + '_avg1' and store_tmpw + '_mc_avg1_var' to the
            DataStore. If `conf_ints` are set, also the confidence intervals
            `_mc_avg1` are added to the DataStore. Works independently of the
            ci_avg_time_flag2 and ci_avg_x_flag.
        ci_avg_time_flag2 : bool
            The confidence intervals differ each time step. Assumes the
            temperature remains constant during the measurement period.
            Computes the inverse-variance-weighted-temporal-mean temperature
            and its uncertainty.
            If you would like to know the confidence interfal of:
            (1) I want to estimate a background temperature with confidence
            intervals. I hereby assume the temperature does not change over
            time and average all measurements to get a better estimate of the
            background temperature.
            Adds store_tmpw + '_avg2' and store_tmpw + '_mc_avg2_var' to the
            DataStore. If `conf_ints` are set, also the confidence intervals
            `_mc_avg2` are added to the DataStore. Works independently of the
            ci_avg_time_flag1 and ci_avg_x_flag.
        ci_avg_time_sel : slice
            Compute ci_avg_time_flag1 and ci_avg_time_flag2 using only a
            selection of the data
        ci_avg_time_isel : iterable of int
            Compute ci_avg_time_flag1 and ci_avg_time_flag2 using only a
            selection of the data
        ci_avg_x_flag1 : bool
            The confidence intervals differ at each location. Assumes the
            temperature varies over `x` and over time. Computes the
            arithmic spatial mean. If you would like to know the confidence
            interfal of:
            (1) a single additional measurement location. So you can state "if
            another measurement location were to be taken,
            it would have this ci"
            (2) all measurement locations. So you can state "The temperature
            along the fiber remained between these ci bounds".
            Adds store_tmpw + '_avgx1' and store_tmpw + '_mc_avgx1_var' to the
            DataStore. If `conf_ints` are set, also the confidence intervals
            `_mc_avgx1` are added to the DataStore. Works independently of the
            ci_avg_time_flag1, ci_avg_time_flag2 and ci_avg_x2_flag.
        ci_avg_x_flag2 : bool
            The confidence intervals differ at each location. Assumes the
            temperature is the same at each location but varies over time.
            Computes the inverse-variance-weighted-spatial-mean temperature
            and its uncertainty.
            If you would like to know the confidence interfal of:
            (1) I have put a lot of fiber in water, and I know that the
            temperature variation in the water is much smaller than along
            other parts of the fiber. And I would like to average the
            measurements from multiple locations to improve the estimated
            temperature.
            Adds store_tmpw + '_avg2' and store_tmpw + '_mc_avg2_var' to the
            DataStore. If `conf_ints` are set, also the confidence intervals
            `_mc_avg2` are added to the DataStore. Works independently of the
            ci_avg_time_flag1 and ci_avg_x_flag.
        ci_avg_x_sel : slice
            Compute ci_avg_time_flag1 and ci_avg_time_flag2 using only a
            selection of the data
        ci_avg_x_isel : iterable of int
            Compute ci_avg_time_flag1 and ci_avg_time_flag2 using only a
            selection of the data
        var_only_sections : bool
            useful if using the ci_avg_x_flag. Only calculates the var over the
            sections, so that the values can be compared with accuracy along the
            reference sections. Where the accuracy is the variance of the
            residuals between the estimated temperature and temperature of the
            water baths.
        da_random_state
            For testing purposes. Similar to random seed. The seed for dask.
            Makes random not so random. To produce reproducable results for
            testing environments.
        remove_mc_set_flag : bool
            Remove the monte carlo data set, from which the CI and the
            variance are calculated.
        reduce_memory_usage : bool
            Use less memory but at the expense of longer computation time

        Returns
        -------

        """
        self.check_deprecated_kwargs(kwargs)

        if var_only_sections is not None:
            raise NotImplementedError()

        self.conf_int_single_ended(
            p_val=p_val,
            p_cov=p_cov,
            st_var=st_var,
            ast_var=ast_var,
            store_tmpf=store_tmpf,
            store_tempvar=store_tempvar,
            conf_ints=None,
            mc_sample_size=mc_sample_size,
            da_random_state=da_random_state,
            remove_mc_set_flag=False,
            reduce_memory_usage=reduce_memory_usage,
            **kwargs)

        time_dim = self.get_time_dim(data_var_key='st')

        if ci_avg_time_sel is not None:
            time_dim2 = time_dim + '_avg'
            x_dim2 = 'x'
            self.coords[time_dim2] = (
                (time_dim2,),
                self[time_dim].sel(**{
                    time_dim: ci_avg_time_sel}).data)
            self[store_tmpf + '_avgsec'] = (
                ('x', time_dim2),
                self[store_tmpf].sel(**{
                    time_dim: ci_avg_time_sel}).data)
            self[store_tmpf + '_mc_set'] = (
                ('mc', 'x', time_dim2),
                self[store_tmpf
                     + '_mc_set'].sel(**{
                         time_dim: ci_avg_time_sel}).data)

        elif ci_avg_time_isel is not None:
            time_dim2 = time_dim + '_avg'
            x_dim2 = 'x'
            self.coords[time_dim2] = (
                (time_dim2,),
                self[time_dim].isel(**{
                    time_dim: ci_avg_time_isel}).data)
            self[store_tmpf + '_avgsec'] = (
                ('x', time_dim2),
                self[store_tmpf].isel(**{
                    time_dim: ci_avg_time_isel}).data)
            self[store_tmpf + '_mc_set'] = (
                ('mc', 'x', time_dim2),
                self[store_tmpf
                     + '_mc_set'].isel(**{
                         time_dim: ci_avg_time_isel}).data)

        elif ci_avg_x_sel is not None:
            time_dim2 = time_dim
            x_dim2 = 'x_avg'
            self.coords[x_dim2] = ((x_dim2,), self.x.sel(x=ci_avg_x_sel).data)
            self[store_tmpf + '_avgsec'] = (
                (x_dim2, time_dim), self[store_tmpf].sel(x=ci_avg_x_sel).data)
            self[store_tmpf + '_mc_set'] = (
                ('mc', x_dim2, time_dim),
                self[store_tmpf + '_mc_set'].sel(x=ci_avg_x_sel).data)

        elif ci_avg_x_isel is not None:
            time_dim2 = time_dim
            x_dim2 = 'x_avg'
            self.coords[x_dim2] = (
                (x_dim2,), self.x.isel(x=ci_avg_x_isel).data)
            self[store_tmpf + '_avgsec'] = (
                (x_dim2, time_dim2),
                self[store_tmpf].isel(x=ci_avg_x_isel).data)
            self[store_tmpf + '_mc_set'] = (
                ('mc', x_dim2, time_dim2),
                self[store_tmpf + '_mc_set'].isel(x=ci_avg_x_isel).data)
        else:
            self[store_tmpf + '_avgsec'] = self[store_tmpf]
            x_dim2 = 'x'
            time_dim2 = time_dim

        # subtract the mean temperature
        q = self[store_tmpf + '_mc_set'] - self[store_tmpf + '_avgsec']
        self[store_tmpf + '_mc' + '_avgsec' + store_tempvar] = (
            q.var(dim='mc', ddof=1))

        if ci_avg_x_flag1:
            # unweighted mean
            self[store_tmpf + '_avgx1'] = self[store_tmpf
                                               + '_avgsec'].mean(dim=x_dim2)

            q = self[store_tmpf + '_mc_set'] - self[store_tmpf + '_avgsec']
            qvar = q.var(dim=['mc', x_dim2], ddof=1)
            self[store_tmpf + '_mc_avgx1' + store_tempvar] = qvar

            if conf_ints:
                new_chunks = (
                    len(conf_ints), self[store_tmpf + '_mc_set'].chunks[2])
                avg_axis = self[store_tmpf + '_mc_set'].get_axis_num(
                    ['mc', x_dim2])
                q = self[store_tmpf + '_mc_set'].data.map_blocks(
                    lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
                    chunks=new_chunks,  #
                    drop_axis=avg_axis,
                    # avg dimensions are dropped from input arr
                    new_axis=0)  # The new CI dim is added as firsaxis

                self[store_tmpf + '_mc_avgx1'] = (('CI', time_dim2), q)

        if ci_avg_x_flag2:
            q = self[store_tmpf + '_mc_set'] - self[store_tmpf + '_avgsec']

            qvar = q.var(dim=['mc'], ddof=1)

            # Inverse-variance weighting
            avg_x_var = 1 / (1 / qvar).sum(dim=x_dim2)

            self[store_tmpf + '_mc_avgx2' + store_tempvar] = avg_x_var

            self[store_tmpf
                 + '_mc_avgx2_set'] = (self[store_tmpf + '_mc_set']
                                       / qvar).sum(dim=x_dim2) * avg_x_var
            self[store_tmpf
                 + '_avgx2'] = self[store_tmpf
                                    + '_mc_avgx2_set'].mean(dim='mc')

            if conf_ints:
                new_chunks = (
                    len(conf_ints), self[store_tmpf + '_mc_set'].chunks[2])
                avg_axis_avgx = self[store_tmpf + '_mc_set'].get_axis_num('mc')

                qq = self[store_tmpf + '_mc_avgx2_set'].data.map_blocks(
                    lambda x: np.percentile(
                        x, q=conf_ints, axis=avg_axis_avgx),
                    chunks=new_chunks,  #
                    drop_axis=avg_axis_avgx,
                    # avg dimensions are dropped from input arr
                    new_axis=0,
                    dtype=float)  # The new CI dimension is added as
                # firsaxis
                self[store_tmpf + '_mc_avgx2'] = (('CI', time_dim2), qq)

        if ci_avg_time_flag1 is not None:
            # unweighted mean
            self[store_tmpf + '_avg1'] = self[store_tmpf
                                              + '_avgsec'].mean(dim=time_dim2)

            q = self[store_tmpf + '_mc_set'] - self[store_tmpf + '_avgsec']
            qvar = q.var(dim=['mc', time_dim2], ddof=1)
            self[store_tmpf + '_mc_avg1' + store_tempvar] = qvar

            if conf_ints:
                new_chunks = (
                    len(conf_ints), self[store_tmpf + '_mc_set'].chunks[1])
                avg_axis = self[store_tmpf + '_mc_set'].get_axis_num(
                    ['mc', time_dim2])
                q = self[store_tmpf + '_mc_set'].data.map_blocks(
                    lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
                    chunks=new_chunks,  #
                    drop_axis=avg_axis,
                    # avg dimensions are dropped from input arr
                    new_axis=0)  # The new CI dim is added as firsaxis

                self[store_tmpf + '_mc_avg1'] = (('CI', x_dim2), q)

        if ci_avg_time_flag2:
            q = self[store_tmpf + '_mc_set'] - self[store_tmpf + '_avgsec']

            qvar = q.var(dim=['mc'], ddof=1)

            # Inverse-variance weighting
            avg_time_var = 1 / (1 / qvar).sum(dim=time_dim2)

            self[store_tmpf + '_mc_avg2' + store_tempvar] = avg_time_var

            self[store_tmpf
                 + '_mc_avg2_set'] = (self[store_tmpf + '_mc_set'] / qvar).sum(
                     dim=time_dim2) * avg_time_var
            self[store_tmpf + '_avg2'] = self[store_tmpf
                                              + '_mc_avg2_set'].mean(dim='mc')

            if conf_ints:
                new_chunks = (
                    len(conf_ints), self[store_tmpf + '_mc_set'].chunks[1])
                avg_axis_avg2 = self[store_tmpf + '_mc_set'].get_axis_num('mc')

                qq = self[store_tmpf + '_mc_avg2_set'].data.map_blocks(
                    lambda x: np.percentile(
                        x, q=conf_ints, axis=avg_axis_avg2),
                    chunks=new_chunks,  #
                    drop_axis=avg_axis_avg2,
                    # avg dimensions are dropped from input arr
                    new_axis=0,
                    dtype=float)  # The new CI dimension is added as
                # firsaxis
                self[store_tmpf + '_mc_avg2'] = (('CI', x_dim2), qq)
        # Clean up the garbage. All arrays with a Monte Carlo dimension.
        if remove_mc_set_flag:
            remove_mc_set = [
                'r_st', 'r_ast', 'gamma_mc', 'dalpha_mc', 'c_mc', 'x_avg',
                'time_avg', 'mc', 'ta_mc_arr']
            remove_mc_set.append(store_tmpf + '_avgsec')
            remove_mc_set.append(store_tmpf + '_mc_set')
            remove_mc_set.append(store_tmpf + '_mc_avg2_set')
            remove_mc_set.append(store_tmpf + '_mc_avgx2_set')
            remove_mc_set.append(store_tmpf + '_mc_avgsec' + store_tempvar)

            for k in remove_mc_set:
                if k in self:
                    del self[k]
        pass

    def conf_int_double_ended(
            self,
            p_val='p_val',
            p_cov='p_cov',
            store_ta=None,
            st_var=None,
            ast_var=None,
            rst_var=None,
            rast_var=None,
            store_tmpf='tmpf',
            store_tmpb='tmpb',
            store_tmpw='tmpw',
            store_tempvar='_var',
            conf_ints=None,
            mc_sample_size=100,
            var_only_sections=False,
            da_random_state=None,
            remove_mc_set_flag=True,
            reduce_memory_usage=False,
            **kwargs):
        """
        Estimation of the confidence intervals for the temperatures measured
        with a double-ended setup.
        Double-ended setups require four additional steps to estimate the
        confidence intervals for the temperature. First, the variances of the
        Stokes and anti-Stokes intensity measurements of the forward and
        backward channels are estimated following the steps in
        Section 4 [1]_. See `ds.variance_stokes_constant()`.
        A Normal distribution is assigned to each
        intensity measurement that is centered at the measurement and using the
        estimated variance. Second, a multi-variate Normal distribution is
        assigned to the estimated parameters using the covariance matrix from
        the calibration procedure presented in Section 6 [1]_ (`p_cov`). Third,
        Normal distributions are assigned for :math:`A` (`ds.alpha`)
        for each location
        outside of the reference sections. These distributions are centered
        around :math:`A_p` and have variance :math:`\sigma^2\left[A_p\\right]`
        given by Equations 44 and 45. Fourth, the distributions are sampled
        and :math:`T_{\mathrm{F},m,n}` and :math:`T_{\mathrm{B},m,n}` are
        computed with Equations 16 and 17, respectively. Fifth, step four is repeated to
        compute, e.g., 10,000 realizations (`mc_sample_size`) of :math:`T_{\mathrm{F},m,n}` and
        :math:`T_{\mathrm{B},m,n}` to approximate their probability density
        functions. Sixth, the standard uncertainties of
        :math:`T_{\mathrm{F},m,n}` and :math:`T_{\mathrm{B},m,n}`
        (:math:`\sigma\left[T_{\mathrm{F},m,n}\\right]` and
        :math:`\sigma\left[T_{\mathrm{B},m,n}\\right]`) are estimated with the
        standard deviation of their realizations. Seventh, for each realization
        :math:`i` the temperature :math:`T_{m,n,i}` is computed as the weighted
        average of :math:`T_{\mathrm{F},m,n,i}` and
        :math:`T_{\mathrm{B},m,n,i}`:

        .. math::

            T_{m,n,i} =\
            \sigma^2\left[T_{m,n}\\right]\left({\\frac{T_{\mathrm{F},m,n,i}}{\
            \sigma^2\left[T_{\mathrm{F},m,n}\\right]} +\
            \\frac{T_{\mathrm{B},m,n,i}}{\
            \sigma^2\left[T_{\mathrm{B},m,n}\\right]}}\\right)

        where

        .. math::

            \sigma^2\left[T_{m,n}\\right] = \\frac{1}{1 /\
            \sigma^2\left[T_{\mathrm{F},m,n}\\right] + 1 /\
            \sigma^2\left[T_{\mathrm{B},m,n}\\right]}

        The best estimate of the temperature :math:`T_{m,n}` is computed
        directly from the best estimates of :math:`T_{\mathrm{F},m,n}` and
        :math:`T_{\mathrm{B},m,n}` as:

        .. math::
            T_{m,n} =\
            \sigma^2\left[T_{m,n}\\right]\left({\\frac{T_{\mathrm{F},m,n}}{\
            \sigma^2\left[T_{\mathrm{F},m,n}\\right]} + \\frac{T_{\mathrm{B},m,n}}{\
            \sigma^2\left[T_{\mathrm{B},m,n}\\right]}}\\right)

        Alternatively, the best estimate of :math:`T_{m,n}` can be approximated
        with the mean of the :math:`T_{m,n,i}` values. Finally, the 95\%
        confidence interval for :math:`T_{m,n}` are estimated with the 2.5\% and
        97.5\% percentiles of :math:`T_{m,n,i}`.

        Parameters
        ----------
        p_val : array-like, optional
            Define `p_val`, `p_var`, `p_cov` if you used an external function
            for calibration. Has size `1 + 2 * nt + nx + 2 * nt * nta`.
            First value is :math:`\gamma`, then `nt` times
            :math:`D_\mathrm{F}`, then `nt` times
            :math:`D_\mathrm{B}`, then for each location :math:`D_\mathrm{B}`,
            then for each connector that introduces directional attenuation two
            parameters per time step.
        p_cov : array-like, optional
            The covariances of `p_val`. Square matrix.
            If set to False, no uncertainty in the parameters is propagated
            into the confidence intervals. Similar to the spec sheets of the DTS
            manufacturers. And similar to passing an array filled with zeros.
        st_var, ast_var, rst_var, rast_var : float, callable, array-like, optional
            The variance of the measurement noise of the Stokes signals in the
            forward direction. If `float` the variance of the noise from the
            Stokes detector is described with a single value.
            If `callable` the variance of the noise from the Stokes detector is
            a function of the intensity, as defined in the callable function.
            Or manually define a variance with a DataArray of the shape
            `ds.st.shape`, where the variance can be a function of time and/or
            x. Required if method is wls.
        store_tmpf : str
            Key of how to store the Forward calculated temperature. Is
            calculated using the
            forward Stokes and anti-Stokes observations.
        store_tmpb : str
            Key of how to store the Backward calculated temperature. Is
            calculated using the
            backward Stokes and anti-Stokes observations.
        store_tmpw : str
            Key of how to store the forward-backward-weighted temperature.
            First, the variance of
            tmpf and tmpb are calculated. The Monte Carlo set of tmpf and
            tmpb are averaged,
            weighted by their variance. The median of this set is thought to
            be the a reasonable
            estimate of the temperature
        store_tempvar : str
            a string that is appended to the store_tmp_ keys. and the
            variance is calculated
            for those store_tmp_ keys
        store_ta : str
            Key of how transient attenuation parameters are stored. Default
            is `talpha`. `_fw` and `_bw` is appended to for the forward and
            backward parameters. The `transient_asym_att_x` is derived from
            the `coords` of this DataArray. The `coords` of `ds[store_ta +
            '_fw']` should be ('time', 'trans_att').
        conf_ints : iterable object of float
            A list with the confidence boundaries that are calculated. Valid
            values are between [0, 1].
        mc_sample_size : int
            Size of the monte carlo parameter set used to calculate the
            confidence interval
        var_only_sections : bool
            useful if using the ci_avg_x_flag. Only calculates the var over the
            sections, so that the values can be compared with accuracy along the
            reference sections. Where the accuracy is the variance of the
            residuals between the estimated temperature and temperature of the
            water baths.
        da_random_state
            For testing purposes. Similar to random seed. The seed for dask.
            Makes random not so random. To produce reproducable results for
            testing environments.
        remove_mc_set_flag : bool
            Remove the monte carlo data set, from which the CI and the
            variance are calculated.
        reduce_memory_usage : bool
            Use less memory but at the expense of longer computation time

        Returns
        -------

        References
        ----------
        .. [1] des Tombe, B., Schilperoort, B., & Bakker, M. (2020). Estimation
            of Temperature and Associated Uncertainty from Fiber-Optic Raman-
            Spectrum Distributed Temperature Sensing. Sensors, 20(8), 2235.
            https://doi.org/10.3390/s20082235

        """
        def create_da_ta2(no, i_splice, direction='fw', chunks=None):
            """create mask array mc, o, nt"""

            if direction == 'fw':
                arr = da.concatenate(
                    (
                        da.zeros(
                            (1, i_splice, 1),
                            chunks=((1, i_splice, 1)),
                            dtype=bool),
                        da.ones(
                            (1, no - i_splice, 1),
                            chunks=(1, no - i_splice, 1),
                            dtype=bool)),
                    axis=1).rechunk((1, chunks[1], 1))
            else:
                arr = da.concatenate(
                    (
                        da.ones(
                            (1, i_splice, 1),
                            chunks=(1, i_splice, 1),
                            dtype=bool),
                        da.zeros(
                            (1, no - i_splice, 1),
                            chunks=((1, no - i_splice, 1)),
                            dtype=bool)),
                    axis=1).rechunk((1, chunks[1], 1))
            return arr

        self.check_deprecated_kwargs(kwargs)

        if da_random_state:
            # In testing environments
            assert isinstance(da_random_state, da.random.RandomState)
            state = da_random_state
        else:
            state = da.random.RandomState()

        time_dim = self.get_time_dim(data_var_key='st')

        del_tmpf_after, del_tmpb_after = False, False

        if store_tmpw and not store_tmpf:
            if store_tmpf in self:
                del_tmpf_after = True
            store_tmpf = 'tmpf'
        if store_tmpw and not store_tmpb:
            if store_tmpb in self:
                del_tmpb_after = True
            store_tmpb = 'tmpb'

        if conf_ints:
            assert store_tmpw, 'Current implementation requires you to ' \
                               'define store_tmpw when istimating confidence ' \
                               'intervals'

        no, nt = self.st.shape
        npar = 1 + 2 * nt + no  # number of parameters

        if store_ta:
            ta_dim = [
                i for i in self[store_ta + '_fw'].dims if i != time_dim][0]
            tax = self[ta_dim].values
            nta = tax.size
            npar += nt * 2 * nta
        else:
            nta = 0

        rsize = (mc_sample_size, no, nt)

        if reduce_memory_usage:
            memchunk = da.ones(
                (mc_sample_size, no, nt), chunks={
                    0: -1,
                    1: 1,
                    2: 'auto'}).chunks
        else:
            memchunk = da.ones(
                (mc_sample_size, no, nt), chunks={
                    0: -1,
                    1: 'auto',
                    2: 'auto'}).chunks

        self.coords['mc'] = range(mc_sample_size)
        if conf_ints:
            self.coords['CI'] = conf_ints

        assert isinstance(p_val, (str, np.ndarray, np.generic))
        if isinstance(p_val, str):
            p_val = self[p_val].values
        assert p_val.shape == (npar,), "Did you set `store_ta='talpha'` as " \
                                       "keyword argument of the " \
                                       "conf_int_double_ended() function?"

        assert isinstance(p_cov, (str, np.ndarray, np.generic, bool))

        if isinstance(p_cov, bool) and not p_cov:
            # Exclude parameter uncertainty if p_cov == False
            gamma = p_val[0]
            d_fw = p_val[1:nt + 1]
            d_bw = p_val[1 + nt:2 * nt + 1]
            alpha = p_val[2 * nt + 1:2 * nt + 1 + no]

            self['gamma_mc'] = (tuple(), gamma)
            self['alpha_mc'] = (('x',), alpha)
            self['df_mc'] = ((time_dim,), d_fw)
            self['db_mc'] = ((time_dim,), d_bw)

            if store_ta:
                ta = p_val[2 * nt + 1 + no:].reshape((nt, 2, nta), order='F')
                ta_fw = ta[:, 0, :]
                ta_bw = ta[:, 1, :]

                ta_fw_arr = np.zeros((no, nt))
                for tai, taxi in zip(ta_fw.T, self.coords[ta_dim].values):
                    ta_fw_arr[self.x.values >= taxi] = \
                        ta_fw_arr[self.x.values >= taxi] + tai

                ta_bw_arr = np.zeros((no, nt))
                for tai, taxi in zip(ta_bw.T, self.coords[ta_dim].values):
                    ta_bw_arr[self.x.values < taxi] = \
                        ta_bw_arr[self.x.values < taxi] + tai

                self[store_ta + '_fw_mc'] = (('x', time_dim), ta_fw_arr)
                self[store_ta + '_bw_mc'] = (('x', time_dim), ta_bw_arr)

        elif isinstance(p_cov, bool) and p_cov:
            raise NotImplementedError(
                'Not an implemented option. Check p_cov argument')

        else:
            # WLS
            if isinstance(p_cov, str):
                p_cov = self[p_cov].values
            assert p_cov.shape == (npar, npar)

            ix_sec = self.ufunc_per_section(x_indices=True, calc_per='all')
            nx_sec = ix_sec.size
            from_i = np.concatenate(
                (
                    np.arange(1 + 2 * nt), 1 + 2 * nt + ix_sec,
                    np.arange(1 + 2 * nt + no,
                              1 + 2 * nt + no + nt * 2 * nta)))
            iox_sec1, iox_sec2 = np.meshgrid(from_i, from_i, indexing='ij')
            po_val = p_val[from_i]
            po_cov = p_cov[iox_sec1, iox_sec2]

            po_mc = sst.multivariate_normal.rvs(
                mean=po_val, cov=po_cov, size=mc_sample_size)

            gamma = po_mc[:, 0]
            d_fw = po_mc[:, 1:nt + 1]
            d_bw = po_mc[:, 1 + nt:2 * nt + 1]

            self['gamma_mc'] = (('mc',), gamma)
            self['df_mc'] = (('mc', time_dim), d_fw)
            self['db_mc'] = (('mc', time_dim), d_bw)

            # calculate alpha seperately
            alpha = np.zeros((mc_sample_size, no), dtype=float)
            alpha[:, ix_sec] = po_mc[:, 1 + 2 * nt:1 + 2 * nt + nx_sec]

            not_ix_sec = np.array([i for i in range(no) if i not in ix_sec])

            if np.any(not_ix_sec):
                not_alpha_val = p_val[2 * nt + 1 + not_ix_sec]
                not_alpha_var = p_cov[2 * nt + 1 + not_ix_sec,
                                      2 * nt + 1 + not_ix_sec]

                not_alpha_mc = np.random.normal(
                    loc=not_alpha_val,
                    scale=not_alpha_var**0.5,
                    size=(mc_sample_size, not_alpha_val.size))

                alpha[:, not_ix_sec] = not_alpha_mc

            self['alpha_mc'] = (('mc', 'x'), alpha)

            if store_ta:
                ta = po_mc[:, 2 * nt + 1 + nx_sec:].reshape(
                    (mc_sample_size, nt, 2, nta), order='F')
                ta_fw = ta[:, :, 0, :]
                ta_bw = ta[:, :, 1, :]

                ta_fw_arr = da.zeros(
                    (mc_sample_size, no, nt), chunks=memchunk, dtype=float)
                for tai, taxi in zip(ta_fw.swapaxes(0, 2),
                                     self.coords[ta_dim].values):
                    # iterate over the splices
                    i_splice = sum(self.x.values < taxi)
                    mask = create_da_ta2(
                        no, i_splice, direction='fw', chunks=memchunk)

                    ta_fw_arr += mask * tai.T[:, None, :]

                ta_bw_arr = da.zeros(
                    (mc_sample_size, no, nt), chunks=memchunk, dtype=float)
                for tai, taxi in zip(ta_bw.swapaxes(0, 2),
                                     self.coords[ta_dim].values):
                    i_splice = sum(self.x.values < taxi)
                    mask = create_da_ta2(
                        no, i_splice, direction='bw', chunks=memchunk)

                    ta_bw_arr += mask * tai.T[:, None, :]

                self[store_ta + '_fw_mc'] = (('mc', 'x', time_dim), ta_fw_arr)
                self[store_ta + '_bw_mc'] = (('mc', 'x', time_dim), ta_bw_arr)

        # Draw from the normal distributions for the Stokes intensities
        for k, st_labeli, st_vari in zip(['r_st', 'r_ast', 'r_rst', 'r_rast'],
                                         ['st', 'ast', 'rst', 'rast'],
                                         [st_var, ast_var, rst_var, rast_var]):

            # Load the mean as chunked Dask array, otherwise eats memory
            if type(self[st_labeli].data) == da.core.Array:
                loc = da.asarray(self[st_labeli].data, chunks=memchunk[1:])
            else:
                loc = da.from_array(self[st_labeli].data, chunks=memchunk[1:])

            # Make sure variance is of size (no, nt)
            if np.size(st_vari) > 1:
                if st_vari.shape == self[st_labeli].shape:
                    pass
                else:
                    st_vari = np.broadcast_to(st_vari, (no, nt))
            else:
                pass

            # Load variance as chunked Dask array, otherwise eats memory
            if type(st_vari) == da.core.Array:
                st_vari_da = da.asarray(st_vari, chunks=memchunk[1:])

            elif (callable(st_vari) and
                  type(self[st_labeli].data) == da.core.Array):
                st_vari_da = da.asarray(
                    st_vari(self[st_labeli]).data, chunks=memchunk[1:])

            elif (callable(st_vari) and
                  type(self[st_labeli].data) != da.core.Array):
                st_vari_da = da.from_array(
                    st_vari(self[st_labeli]).data, chunks=memchunk[1:])

            else:
                st_vari_da = da.from_array(st_vari, chunks=memchunk[1:])

            self[k] = (
                ('mc', 'x', time_dim),
                state.normal(
                    loc=loc,  # has chunks=memchunk[1:]
                    scale=st_vari_da**0.5,
                    size=rsize,
                    chunks=memchunk))

        for label, del_label in zip([store_tmpf, store_tmpb],
                                    [del_tmpf_after, del_tmpb_after]):
            if store_tmpw or label:
                if label == store_tmpf:
                    if store_ta:
                        self[store_tmpf + '_mc_set'] = self['gamma_mc'] / (
                            np.log(self['r_st'] / self['r_ast'])
                            + self['df_mc'] + self['alpha_mc']
                            + self[store_ta + '_fw_mc']) - 273.15
                    else:
                        self[store_tmpf + '_mc_set'] = self['gamma_mc'] / (
                            np.log(self['r_st'] / self['r_ast'])
                            + self['df_mc'] + self['alpha_mc']) - 273.15
                else:
                    if store_ta:
                        self[store_tmpb + '_mc_set'] = self['gamma_mc'] / (
                            np.log(self['r_rst'] / self['r_rast'])
                            + self['db_mc'] - self['alpha_mc']
                            + self[store_ta + '_bw_mc']) - 273.15
                    else:
                        self[store_tmpb + '_mc_set'] = self['gamma_mc'] / (
                            np.log(self['r_rst'] / self['r_rast'])
                            + self['db_mc'] - self['alpha_mc']) - 273.15

                if var_only_sections:
                    # sets the values outside the reference sections to NaN
                    xi = self.ufunc_per_section(x_indices=True, calc_per='all')
                    x_mask_ = [
                        True if ix in xi else False
                        for ix in range(self.x.size)]
                    x_mask = np.reshape(x_mask_, (1, -1, 1))
                    self[label + '_mc_set'] = self[label
                                                   + '_mc_set'].where(x_mask)

                # subtract the mean temperature
                q = self[label + '_mc_set'] - self[label]
                self[label + '_mc' + store_tempvar] = (q.var(dim='mc', ddof=1))

                if conf_ints and not del_label:
                    new_chunks = list(self[label + '_mc_set'].chunks)
                    new_chunks[0] = (len(conf_ints),)
                    avg_axis = self[label + '_mc_set'].get_axis_num('mc')
                    q = self[label + '_mc_set'].data.map_blocks(
                        lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
                        chunks=new_chunks,  #
                        drop_axis=avg_axis,
                        # avg dimensions are dropped from input arr
                        new_axis=0)  # The new CI dimension is added as firsaxis

                    self[label + '_mc'] = (('CI', 'x', time_dim), q)

        # Weighted mean of the forward and backward
        tmpw_var = 1 / (
            1 / self[store_tmpf + '_mc' + store_tempvar]
            + 1 / self[store_tmpb + '_mc' + store_tempvar])

        q = (
            self[store_tmpf + '_mc_set']
            / self[store_tmpf + '_mc' + store_tempvar]
            + self[store_tmpb + '_mc_set']
            / self[store_tmpb + '_mc' + store_tempvar]) * tmpw_var

        self[store_tmpw + '_mc_set'] = q  #

        self[store_tmpw] = \
            (self[store_tmpf] /
             self[store_tmpf + '_mc' + store_tempvar] +
             self[store_tmpb] /
             self[store_tmpb + '_mc' + store_tempvar]
             ) * tmpw_var

        q = self[store_tmpw + '_mc_set'] - self[store_tmpw]
        self[store_tmpw + '_mc' + store_tempvar] = q.var(dim='mc', ddof=1)

        # Calculate the CI of the weighted MC_set
        if conf_ints:
            new_chunks_weighted = ((len(conf_ints),),) + memchunk[1:]
            avg_axis = self[store_tmpw + '_mc_set'].get_axis_num('mc')
            q2 = self[store_tmpw + '_mc_set'].data.map_blocks(
                lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
                chunks=new_chunks_weighted,  # Explicitly define output chunks
                drop_axis=avg_axis,  # avg dimensions are dropped
                new_axis=0,
                dtype=float)  # The new CI dimension is added as first axis
            self[store_tmpw + '_mc'] = (('CI', 'x', time_dim), q2)

        # Clean up the garbage. All arrays with a Monte Carlo dimension.
        if remove_mc_set_flag:
            remove_mc_set = [
                'r_st', 'r_ast', 'r_rst', 'r_rast', 'gamma_mc', 'alpha_mc',
                'df_mc', 'db_mc']

            for i in [store_tmpf, store_tmpb, store_tmpw]:
                remove_mc_set.append(i + '_mc_set')

            if store_ta:
                remove_mc_set.append(store_ta + '_fw_mc')
                remove_mc_set.append(store_ta + '_bw_mc')

            for k in remove_mc_set:
                if k in self:
                    del self[k]

        if del_tmpf_after:
            del self['tmpf']
        if del_tmpb_after:
            del self['tmpb']
        pass

    def average_double_ended(
            self,
            p_val='p_val',
            p_cov='p_cov',
            store_ta=None,
            st_var=None,
            ast_var=None,
            rst_var=None,
            rast_var=None,
            store_tmpf='tmpf',
            store_tmpb='tmpb',
            store_tmpw='tmpw',
            store_tempvar='_var',
            conf_ints=None,
            mc_sample_size=100,
            ci_avg_time_flag1=False,
            ci_avg_time_flag2=False,
            ci_avg_time_sel=None,
            ci_avg_time_isel=None,
            ci_avg_x_flag1=False,
            ci_avg_x_flag2=False,
            ci_avg_x_sel=None,
            ci_avg_x_isel=None,
            da_random_state=None,
            remove_mc_set_flag=True,
            reduce_memory_usage=False,
            **kwargs):
        """
        Average temperatures from double-ended setups.

        Four types of averaging are implemented. Please see Example Notebook 16.

        Parameters
        ----------
        p_val : array-like, optional
            Define `p_val`, `p_var`, `p_cov` if you used an external function
            for calibration. Has size 2 + `nt`. First value is :math:`\gamma`,
            second is :math:`\Delta \\alpha`, others are :math:`C` for each
            timestep.
            If set to False, no uncertainty in the parameters is propagated
            into the confidence intervals. Similar to the spec sheets of the DTS
            manufacturers. And similar to passing an array filled with zeros
        p_cov : array-like, optional
            The covariances of `p_val`.
        st_var, ast_var, rst_var, rast_var : float, callable, array-like, optional
            The variance of the measurement noise of the Stokes signals in the
            forward direction. If `float` the variance of the noise from the
            Stokes detector is described with a single value.
            If `callable` the variance of the noise from the Stokes detector is
            a function of the intensity, as defined in the callable function.
            Or manually define a variance with a DataArray of the shape
            `ds.st.shape`, where the variance can be a function of time and/or
            x. Required if method is wls.
        store_tmpf : str
            Key of how to store the Forward calculated temperature. Is
            calculated using the
            forward Stokes and anti-Stokes observations.
        store_tmpb : str
            Key of how to store the Backward calculated temperature. Is
            calculated using the
            backward Stokes and anti-Stokes observations.
        store_tmpw : str
            Key of how to store the forward-backward-weighted temperature.
            First, the variance of
            tmpf and tmpb are calculated. The Monte Carlo set of tmpf and
            tmpb are averaged,
            weighted by their variance. The median of this set is thought to
            be the a reasonable
            estimate of the temperature
        store_tempvar : str
            a string that is appended to the store_tmp_ keys. and the
            variance is calculated
            for those store_tmp_ keys
        store_ta : str
            Key of how transient attenuation parameters are stored. Default
            is `talpha`. `_fw` and `_bw` is appended to for the forward and
            backward parameters. The `transient_asym_att_x` is derived from
            the `coords` of this DataArray. The `coords` of `ds[store_ta +
            '_fw']` should be ('time', 'trans_att').
        conf_ints : iterable object of float
            A list with the confidence boundaries that are calculated. Valid
            values are between
            [0, 1].
        mc_sample_size : int
            Size of the monte carlo parameter set used to calculate the
            confidence interval
        ci_avg_time_flag1 : bool
            The confidence intervals differ each time step. Assumes the
            temperature varies during the measurement period. Computes the
            arithmic temporal mean. If you would like to know the confidence
            interfal of:
            (1) a single additional measurement. So you can state "if another
            measurement were to be taken, it would have this ci"
            (2) all measurements. So you can state "The temperature remained
            during the entire measurement period between these ci bounds".
            Adds store_tmpw + '_avg1' and store_tmpw + '_mc_avg1_var' to the
            DataStore. If `conf_ints` are set, also the confidence intervals
            `_mc_avg1` are added to the DataStore. Works independently of the
            ci_avg_time_flag2 and ci_avg_x_flag.
        ci_avg_time_flag2 : bool
            The confidence intervals differ each time step. Assumes the
            temperature remains constant during the measurement period.
            Computes the inverse-variance-weighted-temporal-mean temperature
            and its uncertainty.
            If you would like to know the confidence interfal of:
            (1) I want to estimate a background temperature with confidence
            intervals. I hereby assume the temperature does not change over
            time and average all measurements to get a better estimate of the
            background temperature.
            Adds store_tmpw + '_avg2' and store_tmpw + '_mc_avg2_var' to the
            DataStore. If `conf_ints` are set, also the confidence intervals
            `_mc_avg2` are added to the DataStore. Works independently of the
            ci_avg_time_flag1 and ci_avg_x_flag.
        ci_avg_time_sel : slice
            Compute ci_avg_time_flag1 and ci_avg_time_flag2 using only a
            selection of the data
        ci_avg_time_isel : iterable of int
            Compute ci_avg_time_flag1 and ci_avg_time_flag2 using only a
            selection of the data
        ci_avg_x_flag1 : bool
            The confidence intervals differ at each location. Assumes the
            temperature varies over `x` and over time. Computes the
            arithmic spatial mean. If you would like to know the confidence
            interfal of:
            (1) a single additional measurement location. So you can state "if
            another measurement location were to be taken,
            it would have this ci"
            (2) all measurement locations. So you can state "The temperature
            along the fiber remained between these ci bounds".
            Adds store_tmpw + '_avgx1' and store_tmpw + '_mc_avgx1_var' to the
            DataStore. If `conf_ints` are set, also the confidence intervals
            `_mc_avgx1` are added to the DataStore. Works independently of the
            ci_avg_time_flag1, ci_avg_time_flag2 and ci_avg_x2_flag.
        ci_avg_x_flag2 : bool
            The confidence intervals differ at each location. Assumes the
            temperature is the same at each location but varies over time.
            Computes the inverse-variance-weighted-spatial-mean temperature
            and its uncertainty.
            If you would like to know the confidence interfal of:
            (1) I have put a lot of fiber in water, and I know that the
            temperature variation in the water is much smaller than along
            other parts of the fiber. And I would like to average the
            measurements from multiple locations to improve the estimated
            temperature.
            Adds store_tmpw + '_avg2' and store_tmpw + '_mc_avg2_var' to the
            DataStore. If `conf_ints` are set, also the confidence intervals
            `_mc_avg2` are added to the DataStore. Works independently of the
            ci_avg_time_flag1 and ci_avg_x_flag.
        ci_avg_x_sel : slice
            Compute ci_avg_time_flag1 and ci_avg_time_flag2 using only a
            selection of the data
        ci_avg_x_isel : iterable of int
            Compute ci_avg_time_flag1 and ci_avg_time_flag2 using only a
            selection of the data
        da_random_state
            For testing purposes. Similar to random seed. The seed for dask.
            Makes random not so random. To produce reproducable results for
            testing environments.
        remove_mc_set_flag : bool
            Remove the monte carlo data set, from which the CI and the
            variance are calculated.
        reduce_memory_usage : bool
            Use less memory but at the expense of longer computation time

        Returns
        -------

        """
        def create_da_ta2(no, i_splice, direction='fw', chunks=None):
            """create mask array mc, o, nt"""

            if direction == 'fw':
                arr = da.concatenate(
                    (
                        da.zeros(
                            (1, i_splice, 1),
                            chunks=((1, i_splice, 1)),
                            dtype=bool),
                        da.ones(
                            (1, no - i_splice, 1),
                            chunks=(1, no - i_splice, 1),
                            dtype=bool)),
                    axis=1).rechunk((1, chunks[1], 1))
            else:
                arr = da.concatenate(
                    (
                        da.ones(
                            (1, i_splice, 1),
                            chunks=(1, i_splice, 1),
                            dtype=bool),
                        da.zeros(
                            (1, no - i_splice, 1),
                            chunks=((1, no - i_splice, 1)),
                            dtype=bool)),
                    axis=1).rechunk((1, chunks[1], 1))
            return arr

        self.check_deprecated_kwargs(kwargs)

        if (ci_avg_x_flag1 or ci_avg_x_flag2) and (ci_avg_time_flag1 or
                                                   ci_avg_time_flag2):
            raise NotImplementedError(
                'Incompatible flags. Can not pick '
                'the right chunks')

        elif not (ci_avg_x_flag1 or ci_avg_x_flag2 or ci_avg_time_flag1 or
                  ci_avg_time_flag2):
            raise NotImplementedError('Pick one of the averaging options')

        else:
            pass

        self.conf_int_double_ended(
            p_val=p_val,
            p_cov=p_cov,
            store_ta=store_ta,
            st_var=st_var,
            ast_var=ast_var,
            rst_var=rst_var,
            rast_var=rast_var,
            store_tmpf=store_tmpf,
            store_tmpb=store_tmpb,
            store_tmpw=store_tmpw,
            store_tempvar=store_tempvar,
            conf_ints=None,
            mc_sample_size=mc_sample_size,
            da_random_state=da_random_state,
            remove_mc_set_flag=False,
            reduce_memory_usage=reduce_memory_usage,
            **kwargs)

        time_dim = self.get_time_dim(data_var_key='st')

        for label in [store_tmpf, store_tmpb]:
            if ci_avg_time_sel is not None:
                time_dim2 = time_dim + '_avg'
                x_dim2 = 'x'
                self.coords[time_dim2] = (
                    (time_dim2,),
                    self[time_dim].sel(**{
                        time_dim: ci_avg_time_sel}).data)
                self[label + '_avgsec'] = (
                    ('x', time_dim2),
                    self[label].sel(**{
                        time_dim: ci_avg_time_sel}).data)
                self[label + '_mc_set'] = (
                    ('mc', 'x', time_dim2),
                    self[label + '_mc_set'].sel(**{
                        time_dim: ci_avg_time_sel}).data)

            elif ci_avg_time_isel is not None:
                time_dim2 = time_dim + '_avg'
                x_dim2 = 'x'
                self.coords[time_dim2] = (
                    (time_dim2,),
                    self[time_dim].isel(**{
                        time_dim: ci_avg_time_isel}).data)
                self[label + '_avgsec'] = (
                    ('x', time_dim2),
                    self[label].isel(**{
                        time_dim: ci_avg_time_isel}).data)
                self[label + '_mc_set'] = (
                    ('mc', 'x', time_dim2),
                    self[label
                         + '_mc_set'].isel(**{
                             time_dim: ci_avg_time_isel}).data)

            elif ci_avg_x_sel is not None:
                time_dim2 = time_dim
                x_dim2 = 'x_avg'
                self.coords[x_dim2] = (
                    (x_dim2,), self.x.sel(x=ci_avg_x_sel).data)
                self[label + '_avgsec'] = (
                    (x_dim2, time_dim), self[label].sel(x=ci_avg_x_sel).data)
                self[label + '_mc_set'] = (
                    ('mc', x_dim2, time_dim),
                    self[label + '_mc_set'].sel(x=ci_avg_x_sel).data)

            elif ci_avg_x_isel is not None:
                time_dim2 = time_dim
                x_dim2 = 'x_avg'
                self.coords[x_dim2] = (
                    (x_dim2,), self.x.isel(x=ci_avg_x_isel).data)
                self[label + '_avgsec'] = (
                    (x_dim2, time_dim2),
                    self[label].isel(x=ci_avg_x_isel).data)
                self[label + '_mc_set'] = (
                    ('mc', x_dim2, time_dim2),
                    self[label + '_mc_set'].isel(x=ci_avg_x_isel).data)
            else:
                self[label + '_avgsec'] = self[label]
                x_dim2 = 'x'
                time_dim2 = time_dim

            memchunk = self[label + '_mc_set'].chunks

            # subtract the mean temperature
            q = self[label + '_mc_set'] - self[label + '_avgsec']
            self[label + '_mc' + '_avgsec' + store_tempvar] = (
                q.var(dim='mc', ddof=1))

            if ci_avg_x_flag1:
                # unweighted mean
                self[label + '_avgx1'] = self[label
                                              + '_avgsec'].mean(dim=x_dim2)

                q = self[label + '_mc_set'] - self[label + '_avgsec']
                qvar = q.var(dim=['mc', x_dim2], ddof=1)
                self[label + '_mc_avgx1' + store_tempvar] = qvar

                if conf_ints:
                    new_chunks = (
                        len(conf_ints), self[label + '_mc_set'].chunks[2])
                    avg_axis = self[label + '_mc_set'].get_axis_num(
                        ['mc', x_dim2])
                    q = self[label + '_mc_set'].data.map_blocks(
                        lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
                        chunks=new_chunks,  #
                        drop_axis=avg_axis,
                        # avg dimensions are dropped from input arr
                        new_axis=0)  # The new CI dim is added as firsaxis

                    self[label + '_mc_avgx1'] = (('CI', time_dim2), q)

            if ci_avg_x_flag2:
                q = self[label + '_mc_set'] - self[label + '_avgsec']

                qvar = q.var(dim=['mc'], ddof=1)

                # Inverse-variance weighting
                avg_x_var = 1 / (1 / qvar).sum(dim=x_dim2)

                self[label + '_mc_avgx2' + store_tempvar] = avg_x_var

                self[label
                     + '_mc_avgx2_set'] = (self[label + '_mc_set']
                                           / qvar).sum(dim=x_dim2) * avg_x_var
                self[label + '_avgx2'] = self[label + '_mc_avgx2_set'].mean(
                    dim='mc')

                if conf_ints:
                    new_chunks = (
                        len(conf_ints), self[label + '_mc_set'].chunks[2])
                    avg_axis_avgx = self[label + '_mc_set'].get_axis_num('mc')

                    qq = self[label + '_mc_avgx2_set'].data.map_blocks(
                        lambda x: np.percentile(
                            x, q=conf_ints, axis=avg_axis_avgx),
                        chunks=new_chunks,  #
                        drop_axis=avg_axis_avgx,
                        # avg dimensions are dropped from input arr
                        new_axis=0,
                        dtype=float)  # The new CI dimension is added as
                    # firsaxis
                    self[label + '_mc_avgx2'] = (('CI', time_dim2), qq)

            if ci_avg_time_flag1 is not None:
                # unweighted mean
                self[label + '_avg1'] = self[label
                                             + '_avgsec'].mean(dim=time_dim2)

                q = self[label + '_mc_set'] - self[label + '_avgsec']
                qvar = q.var(dim=['mc', time_dim2], ddof=1)
                self[label + '_mc_avg1' + store_tempvar] = qvar

                if conf_ints:
                    new_chunks = (
                        len(conf_ints), self[label + '_mc_set'].chunks[1])
                    avg_axis = self[label + '_mc_set'].get_axis_num(
                        ['mc', time_dim2])
                    q = self[label + '_mc_set'].data.map_blocks(
                        lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
                        chunks=new_chunks,  #
                        drop_axis=avg_axis,
                        # avg dimensions are dropped from input arr
                        new_axis=0)  # The new CI dim is added as firsaxis

                    self[label + '_mc_avg1'] = (('CI', x_dim2), q)

            if ci_avg_time_flag2:
                q = self[label + '_mc_set'] - self[label + '_avgsec']

                qvar = q.var(dim=['mc'], ddof=1)

                # Inverse-variance weighting
                avg_time_var = 1 / (1 / qvar).sum(dim=time_dim2)

                self[label + '_mc_avg2' + store_tempvar] = avg_time_var

                self[label
                     + '_mc_avg2_set'] = (self[label + '_mc_set'] / qvar).sum(
                         dim=time_dim2) * avg_time_var
                self[label + '_avg2'] = self[label
                                             + '_mc_avg2_set'].mean(dim='mc')

                if conf_ints:
                    new_chunks = (
                        len(conf_ints), self[label + '_mc_set'].chunks[1])
                    avg_axis_avg2 = self[label + '_mc_set'].get_axis_num('mc')

                    qq = self[label + '_mc_avg2_set'].data.map_blocks(
                        lambda x: np.percentile(
                            x, q=conf_ints, axis=avg_axis_avg2),
                        chunks=new_chunks,  #
                        drop_axis=avg_axis_avg2,
                        # avg dimensions are dropped from input arr
                        new_axis=0,
                        dtype=float)  # The new CI dimension is added as
                    # firsaxis
                    self[label + '_mc_avg2'] = (('CI', x_dim2), qq)

        # Weighted mean of the forward and backward
        tmpw_var = 1 / (
            1 / self[store_tmpf + '_mc' + '_avgsec' + store_tempvar]
            + 1 / self[store_tmpb + '_mc' + '_avgsec' + store_tempvar])

        q = (
            self[store_tmpf + '_mc_set']
            / self[store_tmpf + '_mc' + '_avgsec' + store_tempvar]
            + self[store_tmpb + '_mc_set']
            / self[store_tmpb + '_mc' + '_avgsec' + store_tempvar]) * tmpw_var

        self[store_tmpw + '_mc_set'] = q  #

        # self[store_tmpw] = self[store_tmpw + '_mc_set'].mean(dim='mc')
        self[store_tmpw + '_avgsec'] = \
            (self[store_tmpf + '_avgsec'] /
             self[store_tmpf + '_mc' + '_avgsec' + store_tempvar] +
             self[store_tmpb + '_avgsec'] /
             self[store_tmpb + '_mc' + '_avgsec' + store_tempvar]
             ) * tmpw_var

        q = self[store_tmpw + '_mc_set'] - self[store_tmpw + '_avgsec']
        self[store_tmpw + '_mc' + '_avgsec' + store_tempvar] = q.var(
            dim='mc', ddof=1)

        if ci_avg_time_flag1:
            self[store_tmpw + '_avg1'] = \
                self[store_tmpw + '_avgsec'].mean(dim=time_dim2)

            self[store_tmpw + '_mc_avg1' + store_tempvar] = \
                self[store_tmpw + '_mc_set'].var(dim=['mc', time_dim2])

            if conf_ints:
                new_chunks_weighted = ((len(conf_ints),),) + (memchunk[1],)
                avg_axis = self[store_tmpw + '_mc_set'].get_axis_num(
                    ['mc', time_dim2])
                q2 = self[store_tmpw + '_mc_set'].data.map_blocks(
                    lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
                    chunks=new_chunks_weighted,
                    # Explicitly define output chunks
                    drop_axis=avg_axis,  # avg dimensions are dropped
                    new_axis=0,
                    dtype=float)  # The new CI dimension is added as
                # first axis
                self[store_tmpw + '_mc_avg1'] = (('CI', x_dim2), q2)

        if ci_avg_time_flag2:
            tmpw_var_avg2 = 1 / (
                1 / self[store_tmpf + '_mc_avg2' + store_tempvar]
                + 1 / self[store_tmpb + '_mc_avg2' + store_tempvar])

            q = (self[store_tmpf + '_mc_avg2_set'] /
                 self[store_tmpf + '_mc_avg2' + store_tempvar] +
                 self[store_tmpb + '_mc_avg2_set'] /
                 self[store_tmpb + '_mc_avg2' + store_tempvar]) * \
                tmpw_var_avg2

            self[store_tmpw + '_mc_avg2_set'] = q  #

            self[store_tmpw + '_avg2'] = \
                (self[store_tmpf + '_avg2'] /
                 self[store_tmpf + '_mc_avg2' + store_tempvar] +
                 self[store_tmpb + '_avg2'] /
                 self[store_tmpb + '_mc_avg2' + store_tempvar]
                 ) * tmpw_var_avg2

            self[store_tmpw + '_mc_avg2' + store_tempvar] = \
                tmpw_var_avg2

            if conf_ints:
                # We first need to know the x-dim-chunk-size
                new_chunks_weighted = ((len(conf_ints),),) + (memchunk[1],)
                avg_axis_avg2 = self[store_tmpw
                                     + '_mc_avg2_set'].get_axis_num('mc')
                q2 = self[store_tmpw + '_mc_avg2_set'].data.map_blocks(
                    lambda x: np.percentile(
                        x, q=conf_ints, axis=avg_axis_avg2),
                    chunks=new_chunks_weighted,
                    # Explicitly define output chunks
                    drop_axis=avg_axis_avg2,  # avg dimensions are dropped
                    new_axis=0,
                    dtype=float)  # The new CI dimension is added as firstax
                self[store_tmpw + '_mc_avg2'] = (('CI', x_dim2), q2)

        if ci_avg_x_flag1:
            self[store_tmpw + '_avgx1'] = \
                self[store_tmpw + '_avgsec'].mean(dim=x_dim2)

            self[store_tmpw + '_mc_avgx1' + store_tempvar] = \
                self[store_tmpw + '_mc_set'].var(dim=x_dim2)

            if conf_ints:
                new_chunks_weighted = ((len(conf_ints),),) + (memchunk[2],)
                avg_axis = self[store_tmpw + '_mc_set'].get_axis_num(
                    ['mc', x_dim2])
                q2 = self[store_tmpw + '_mc_set'].data.map_blocks(
                    lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
                    chunks=new_chunks_weighted,
                    # Explicitly define output chunks
                    drop_axis=avg_axis,  # avg dimensions are dropped
                    new_axis=0,
                    dtype=float)  # The new CI dimension is added as
                # first axis
                self[store_tmpw + '_mc_avgx1'] = (('CI', time_dim2), q2)

        if ci_avg_x_flag2:
            tmpw_var_avgx2 = 1 / (
                1 / self[store_tmpf + '_mc_avgx2' + store_tempvar]
                + 1 / self[store_tmpb + '_mc_avgx2' + store_tempvar])

            q = (self[store_tmpf + '_mc_avgx2_set'] /
                 self[store_tmpf + '_mc_avgx2' + store_tempvar] +
                 self[store_tmpb + '_mc_avgx2_set'] /
                 self[store_tmpb + '_mc_avgx2' + store_tempvar]) * \
                tmpw_var_avgx2

            self[store_tmpw + '_mc_avgx2_set'] = q  #

            self[store_tmpw + '_avgx2'] = \
                (self[store_tmpf + '_avgx2'] /
                 self[store_tmpf + '_mc_avgx2' + store_tempvar] +
                 self[store_tmpb + '_avgx2'] /
                 self[store_tmpb + '_mc_avgx2' + store_tempvar]
                 ) * tmpw_var_avgx2

            self[store_tmpw + '_mc_avgx2' + store_tempvar] = \
                tmpw_var_avgx2

            if conf_ints:
                # We first need to know the x-dim-chunk-size
                new_chunks_weighted = ((len(conf_ints),),) + (memchunk[2],)
                avg_axis_avgx2 = self[store_tmpw
                                      + '_mc_avgx2_set'].get_axis_num('mc')
                q2 = self[store_tmpw + '_mc_avgx2_set'].data.map_blocks(
                    lambda x: np.percentile(
                        x, q=conf_ints, axis=avg_axis_avgx2),
                    chunks=new_chunks_weighted,
                    # Explicitly define output chunks
                    drop_axis=avg_axis_avgx2,  # avg dimensions are dropped
                    new_axis=0,
                    dtype=float)  # The new CI dimension is added as firstax
                self[store_tmpw + '_mc_avgx2'] = (('CI', time_dim2), q2)

        # Clean up the garbage. All arrays with a Monte Carlo dimension.
        if remove_mc_set_flag:
            remove_mc_set = [
                'r_st', 'r_ast', 'r_rst', 'r_rast', 'gamma_mc', 'alpha_mc',
                'df_mc', 'db_mc', 'x_avg', 'time_avg', 'mc']

            for i in [store_tmpf, store_tmpb, store_tmpw]:
                remove_mc_set.append(i + '_avgsec')
                remove_mc_set.append(i + '_mc_set')
                remove_mc_set.append(i + '_mc_avg2_set')
                remove_mc_set.append(i + '_mc_avgx2_set')
                remove_mc_set.append(i + '_mc_avgsec' + store_tempvar)

            if store_ta:
                remove_mc_set.append(store_ta + '_fw_mc')
                remove_mc_set.append(store_ta + '_bw_mc')

            for k in remove_mc_set:
                if k in self:
                    del self[k]
        pass

    def temperature_residuals(self, label=None, sections=None):
        """
        Compute the temperature residuals, between the known temperature of the
        reference sections and the DTS temperature.

        Parameters
        ----------
        label : str
            The key of the temperature DataArray
        sections : Dict[str, List[slice]], optional
            If `None` is supplied, `ds.sections` is used. Define calibration
            sections. Each section requires a reference temperature time series,
            such as the temperature measured by an external temperature sensor.
            They should already be part of the DataStore object. `sections`
            is defined with a dictionary with its keywords of the
            names of the reference temperature time series. Its values are
            lists of slice objects, where each slice object is a fiber stretch
            that has the reference temperature. Afterwards, `sections` is stored
            under `ds.sections`.

        Returns
        -------
        resid_da : xarray.DataArray
            The residuals as DataArray
        """
        time_dim = self.get_time_dim(data_var_key=label)

        resid_temp = self.ufunc_per_section(
            sections=sections, label=label, temp_err=True, calc_per='all')
        resid_x = self.ufunc_per_section(
            sections=sections, label='x', calc_per='all')

        resid_ix = np.array(
            [np.argmin(np.abs(ai - self.x.data)) for ai in resid_x])

        resid_sorted = np.full(shape=self[label].shape, fill_value=np.nan)
        resid_sorted[resid_ix, :] = resid_temp
        resid_da = xr.DataArray(
            data=resid_sorted,
            dims=('x', time_dim),
            coords={
                'x': self.x,
                time_dim: self.time})
        return resid_da

    def ufunc_per_section(
            self,
            sections=None,
            func=None,
            label=None,
            subtract_from_label=None,
            temp_err=False,
            x_indices=False,
            ref_temp_broadcasted=False,
            calc_per='stretch',
            **func_kwargs):
        """
        User function applied to parts of the cable. Super useful,
        many options and slightly
        complicated.

        The function `func` is taken over all the timesteps and calculated
        per `calc_per`. This
        is returned as a dictionary

        Parameters
        ----------
        sections : Dict[str, List[slice]], optional
            If `None` is supplied, `ds.sections` is used. Define calibration
            sections. Each section requires a reference temperature time series,
            such as the temperature measured by an external temperature sensor.
            They should already be part of the DataStore object. `sections`
            is defined with a dictionary with its keywords of the
            names of the reference temperature time series. Its values are
            lists of slice objects, where each slice object is a fiber stretch
            that has the reference temperature. Afterwards, `sections` is stored
            under `ds.sections`.
        func : callable, str
            A numpy function, or lambda function to apple to each 'calc_per'.
        label
        subtract_from_label
        temp_err : bool
            The argument of the function is label minus the reference
            temperature.
        x_indices : bool
            To retreive an integer array with the indices of the
            x-coordinates in the section/stretch. The indices are sorted.
        ref_temp_broadcasted : bool
        calc_per : {'all', 'section', 'stretch'}
        func_kwargs : dict
            Dictionary with options that are passed to func

        TODO: Spend time on creating a slice instead of appendng everything\
        to a list and concatenating after.


        Returns
        -------

        Examples
        --------

        1. Calculate the variance of the residuals in the along ALL the\
        reference sections wrt the temperature of the water baths

        >>> tmpf_var = d.ufunc_per_section(
        >>>     func='var',
        >>>     calc_per='all',
        >>>     label='tmpf',
        >>>     temp_err=True)

        2. Calculate the variance of the residuals in the along PER\
        reference section wrt the temperature of the water baths

        >>> tmpf_var = d.ufunc_per_section(
        >>>     func='var',
        >>>     calc_per='stretch',
        >>>     label='tmpf',
        >>>     temp_err=True)

        3. Calculate the variance of the residuals in the along PER\
        water bath wrt the temperature of the water baths

        >>> tmpf_var = d.ufunc_per_section(
        >>>     func='var',
        >>>     calc_per='section',
        >>>     label='tmpf',
        >>>     temp_err=True)

        4. Obtain the coordinates of the measurements per section

        >>> locs = d.ufunc_per_section(
        >>>     func=None,
        >>>     label='x',
        >>>     temp_err=False,
        >>>     ref_temp_broadcasted=False,
        >>>     calc_per='stretch')

        5. Number of observations per stretch

        >>> nlocs = d.ufunc_per_section(
        >>>     func=len,
        >>>     label='x',
        >>>     temp_err=False,
        >>>     ref_temp_broadcasted=False,
        >>>     calc_per='stretch')

        6. broadcast the temperature of the reference sections to\
        stretch/section/all dimensions. The value of the reference\
        temperature (a timeseries) is broadcasted to the shape of self[\
        label]. The self[label] is not used for anything else.

        >>> temp_ref = d.ufunc_per_section(
        >>>     label='st',
        >>>     ref_temp_broadcasted=True,
        >>>     calc_per='all')

        7. x-coordinate index

        >>> ix_loc = d.ufunc_per_section(x_indices=True)


        Note
        ----
        If `self[label]` or `self[subtract_from_label]` is a Dask array, a Dask
        array is returned else a numpy array is returned
        """
        if sections is None:
            sections = self.sections

        if not func:

            def func(a):
                """

                Parameters
                ----------
                a

                Returns
                -------

                """
                return a

        elif isinstance(func, str) and func == 'var':

            def func(a):
                """

                Parameters
                ----------
                a

                Returns
                -------

                """
                return np.var(a, ddof=1)

        else:
            assert callable(func)

        assert calc_per in ['all', 'section', 'stretch']

        if not x_indices and \
            ((label and hasattr(self[label].data, 'chunks')) or
             (subtract_from_label and hasattr(self[subtract_from_label].data,
                                              'chunks'))):
            concat = da.concatenate
        else:
            concat = np.concatenate

        out = dict()

        for k, section in sections.items():
            out[k] = []
            for stretch in section:
                if x_indices:
                    assert not subtract_from_label
                    assert not temp_err
                    assert not ref_temp_broadcasted
                    # so it is slicable with x-indices
                    self['_x_indices'] = self.x.astype(int) * 0 + \
                        np.arange(self.x.size)
                    arg1 = self['_x_indices'].sel(x=stretch).data
                    del self['_x_indices']

                else:
                    arg1 = self[label].sel(x=stretch).data

                if subtract_from_label:
                    # calculate std wrt other series
                    # check_dims(self, [subtract_from_label],
                    #            correct_dims=('x', time_dim))

                    assert not temp_err

                    arg2 = self[subtract_from_label].sel(x=stretch).data
                    out[k].append(arg1 - arg2)

                elif temp_err:
                    # calculate std wrt reference temperature of the
                    # corresponding bath
                    arg2 = self[k].data
                    out[k].append(arg1 - arg2)

                elif ref_temp_broadcasted:
                    assert not temp_err
                    assert not subtract_from_label

                    arg2 = da.broadcast_to(self[k].data, arg1.shape)
                    out[k].append(arg2)

                else:
                    # calculate std wrt mean value
                    out[k].append(arg1)

            if calc_per == 'stretch':
                out[k] = [func(argi, **func_kwargs) for argi in out[k]]

            elif calc_per == 'section':
                # flatten the out_dict to sort them
                start = [i.start for i in section]
                i_sorted = np.argsort(start)
                out_flat_sort = [out[k][i] for i in i_sorted]
                out[k] = func(concat(out_flat_sort), **func_kwargs)

        if calc_per == 'all':
            # flatten the out_dict to sort them
            start = [
                item.start
                for sublist in sections.values()
                for item in sublist]
            i_sorted = np.argsort(start)
            out_flat = [item for sublist in out.values() for item in sublist]
            out_flat_sort = [out_flat[i] for i in i_sorted]
            out = func(concat(out_flat_sort, axis=0), **func_kwargs)

            if (hasattr(out, 'chunks') and len(out.chunks) > 0 and
                    'x' in self[label].dims):
                # also sum the chunksize in the x dimension
                # first find out where the x dim is
                ixdim = self[label].dims.index('x')
                c_old = out.chunks
                c_new = list(c_old)
                c_new[ixdim] = sum(c_old[ixdim])
                out = out.rechunk(c_new)

        return out


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
        **kwargs):
    """Load and decode a datastore from a file or file-like object.
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
            filename_or_obj, group=group, decode_cf=decode_cf,
            mask_and_scale=mask_and_scale, decode_times=decode_times,
            concat_characters=concat_characters, decode_coords=decode_coords,
            engine=engine, chunks=chunks, lock=lock, cache=cache,
            drop_variables=drop_variables,
            backend_kwargs=backend_kwargs) as ds_xr:
        ds = DataStore(
            data_vars=ds_xr.data_vars,
            coords=ds_xr.coords,
            attrs=ds_xr.attrs,
            **ds_kwargs)

        # to support deprecated st_labels
        ds = ds.rename_labels(assertion=False)

        if load_in_memory:
            if "cache" in kwargs:
                raise TypeError("cache has no effect in this context")
            return ds.load()

        else:
            return ds


def open_mf_datastore(
        path=None,
        paths=None,
        combine='by_coords',
        load_in_memory=False,
        **kwargs):
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
        assert paths, 'No files match found with: ' + path

    with open_mfdataset(paths=paths, combine=combine, **kwargs) as xds:
        ds = DataStore(
            data_vars=xds.data_vars, coords=xds.coords, attrs=xds.attrs)

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
        file_ext='*.xml',
        timezone_netcdf='UTC',
        silent=False,
        load_in_memory='auto',
        **kwargs):
    """Read a folder with measurement files. Each measurement file contains
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

    assert 'timezone_input_files' not in kwargs, 'The silixa files are ' \
                                                 'already timezone aware'

    if filepathlist is None and zip_handle is None:
        filepathlist = sorted(glob.glob(os.path.join(directory, file_ext)))

        # Make sure that the list of files contains any files
        assert len(
            filepathlist) >= 1, 'No measurement files found in provided ' \
                                'directory: \n' + \
                                str(directory)

    elif filepathlist is None and zip_handle:
        filepathlist = ziphandle_to_filepathlist(
            fh=zip_handle, extension=file_ext)

    # Make sure that the list of files contains any files
    assert len(
        filepathlist) >= 1, 'No measurement files found in provided ' \
                            'list/directory'

    xml_version = silixa_xml_version_check(filepathlist)

    if xml_version == 4:
        data_vars, coords, attrs = read_silixa_files_routine_v4(
            filepathlist,
            timezone_netcdf=timezone_netcdf,
            silent=silent,
            load_in_memory=load_in_memory)

    elif xml_version in (6, 7, 8):
        data_vars, coords, attrs = read_silixa_files_routine_v6(
            filepathlist,
            xml_version=xml_version,
            timezone_netcdf=timezone_netcdf,
            silent=silent,
            load_in_memory=load_in_memory)

    else:
        raise NotImplementedError(
            'Silixa xml version ' + '{0} not implemented'.format(xml_version))

    ds = DataStore(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)
    return ds


def read_sensortran_files(
        directory, timezone_netcdf='UTC', silent=False, **kwargs):
    """Read a folder with measurement files. Each measurement file contains
    values for a
    single timestep. Remember to check which timezone you are working in.

    The sensortran files are already timezone aware

    Parameters
    ----------
    directory : str, Path
        Path to folder containing BinaryRawDTS and BinaryTemp files
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

    filepathlist_dts = sorted(
        glob.glob(os.path.join(directory, '*BinaryRawDTS.dat')))

    # Make sure that the list of files contains any files
    assert len(
        filepathlist_dts) >= 1, 'No RawDTS measurement files found ' \
                                'in provided directory: \n' + \
                                str(directory)

    filepathlist_temp = [f.replace('RawDTS', 'Temp') for f in filepathlist_dts]

    for ii, fname in enumerate(filepathlist_dts):
        # Check if corresponding temperature file exists
        if not os.path.isfile(filepathlist_temp[ii]):
            raise FileNotFoundError(
                'Could not find BinaryTemp '
                + 'file corresponding to {}'.format(fname))

    version = sensortran_binary_version_check(filepathlist_dts)

    if version == 3:
        data_vars, coords, attrs = read_sensortran_files_routine(
            filepathlist_dts,
            filepathlist_temp,
            timezone_netcdf=timezone_netcdf,
            silent=silent)
    else:
        raise NotImplementedError(
            'Sensortran binary version '
            + '{0} not implemented'.format(version))

    ds = DataStore(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)
    return ds


def read_apsensing_files(
        filepathlist=None,
        directory=None,
        file_ext='*.xml',
        timezone_netcdf='UTC',
        timezone_input_files='UTC',
        silent=False,
        load_in_memory='auto',
        **kwargs):
    """Read a folder with measurement files. Each measurement file contains
    values for a single timestep. Remember to check which timezone
    you are working in.

    Parameters
    ----------
    filepathlist : list of str, optional
        List of paths that point the the silixa files
    directory : str, Path, optional
        Path to folder
    timezone_netcdf : str, optional
        Timezone string of the netcdf file. UTC follows CF-conventions.
    timezone_input_files : str, optional
        Timezone string of the measurement files.
        Remember to check when measurements are taken.
        Also if summertime is used.
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
    if not file_ext == '*.xml':
        raise NotImplementedError('Only .xml files are supported for now')

    if filepathlist is None:
        filepathlist = sorted(glob.glob(os.path.join(directory, file_ext)))

        # Make sure that the list of files contains any files
        assert len(
            filepathlist) >= 1, 'No measurement files found in provided ' \
                                'directory: \n' + \
                                str(directory)

    # Make sure that the list of files contains any files
    assert len(
        filepathlist) >= 1, 'No measurement files found in provided ' \
                            'list/directory'

    device = apsensing_xml_version_check(filepathlist)

    valid_devices = ['CP320']

    if device in valid_devices:
        pass

    else:
        warnings.warn(
            'AP sensing device '
            '"{0}"'.format(device)
            + ' has not been tested.\nPlease open an issue on github'
            + ' and provide an example file')

    data_vars, coords, attrs = read_apsensing_files_routine(
        filepathlist,
        timezone_netcdf=timezone_netcdf,
        silent=silent,
        load_in_memory=load_in_memory)

    ds = DataStore(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)
    return ds


def read_sensornet_files(
        filepathlist=None,
        directory=None,
        file_ext='*.ddf',
        timezone_netcdf='UTC',
        timezone_input_files='UTC',
        silent=False,
        add_internal_fiber_length=50.,
        fiber_length=None,
        **kwargs):
    """Read a folder with measurement files. Each measurement file contains
    values for a single timestep. Remember to check which timezone
    you are working in.

    Parameters
    ----------
    filepathlist : list of str, optional
        List of paths that point the the silixa files
    directory : str, Path, optional
        Path to folder
    timezone_netcdf : str, optional
        Timezone string of the netcdf file. UTC follows CF-conventions.
    timezone_input_files : str, optional
        Timezone string of the measurement files.
        Remember to check when measurements are taken.
        Also if summertime is used.
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
            os.path.join(directory, '**', file_ext), recursive=True)

        # Make sure that the list of files contains any files
        msg = 'No measurement files found in provided directory: \n' + str(
            directory)
        assert len(filepathlist_unsorted) >= 1, msg

        # sort based on dates in filesname. A simple sorted() is not sufficient
        # as month folders do not sort well
        basenames = [os.path.basename(fp) for fp in filepathlist_unsorted]
        dates = [''.join(bn.split(' ')[2:4]) for bn in basenames]
        i_sort = np.argsort(dates)
        filepathlist = [filepathlist_unsorted[i] for i in i_sort]

        # Check measurements are all from same channel
        chno = [bn.split(' ')[1] for bn in basenames]
        assert len(
            set(chno)
        ) == 1, 'Folder contains measurements from multiple channels'

    # Make sure that the list of files contains any files
    assert len(
        filepathlist) >= 1, 'No measurement files found in provided ' \
                            'list/directory'

    ddf_version = sensornet_ddf_version_check(filepathlist)

    valid_versions = [
        'Halo DTS v1*', 'ORYX F/W v1.02 Oryx Data Collector v3*',
        'ORYX F/W v4.00 Oryx Data Collector v3*']

    valid = any([fnmatch.fnmatch(ddf_version, v_) for v_ in valid_versions])

    if valid:
        if fnmatch.fnmatch(ddf_version, 'Halo DTS v1*'):
            flip_reverse_measurements = True
        else:
            flip_reverse_measurements = False

    else:
        flip_reverse_measurements = False
        warnings.warn(
            'Sensornet .dff version '
            '"{0}"'.format(ddf_version)
            + ' has not been tested.\nPlease open an issue on github'
            + ' and provide an example file')

    data_vars, coords, attrs = read_sensornet_files_routine_v3(
        filepathlist,
        timezone_netcdf=timezone_netcdf,
        timezone_input_files=timezone_input_files,
        silent=silent,
        add_internal_fiber_length=add_internal_fiber_length,
        fiber_length=fiber_length,
        flip_reverse_measurements=flip_reverse_measurements)

    ds = DataStore(data_vars=data_vars, coords=coords, attrs=attrs, **kwargs)
    return ds


def func_fit(p, xs):
    return p[:xs, None] * p[None, xs:]


def func_cost(p, data, xs):
    fit = func_fit(p, xs)
    return np.sum((fit - data)**2)

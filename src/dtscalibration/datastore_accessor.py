import dask.array as da
import numpy as np
import xarray as xr
import yaml

from dtscalibration.datastore_utils import check_timestep_allclose


@xr.register_dataset_accessor("dts")
class DtsAccessor:
    def __init__(self, xarray_obj):
        # check xarray_obj
        # check naming convention
        assert (
            ("st" in xarray_obj.data_vars) and 
            ("ast" in xarray_obj.data_vars) and
            ("userAcquisitionTimeFW" in xarray_obj.data_vars)), \
                "xarray_obj should have st, ast, userAcquisitionTimeFW"

        # Varying acquisition times not supported. Could be in the future.
        # Should actually be moved to the estimating variance functions
        check_timestep_allclose(xarray_obj, eps=0.01)

        # cache xarray_obj
        self._obj = xarray_obj
        self.attrs = xarray_obj.attrs

        # alias commonly used variables
        self.x = xarray_obj.x
        self.time = xarray_obj.time
        self.transatt = xarray_obj.get("transatt")

        self.st = xarray_obj["st"]          # required
        self.ast = xarray_obj["ast"]        # required
        self.rst = xarray_obj.get("rst")    # None if doesn't exist
        self.rast = xarray_obj.get("rast")  # None is doesn't exist

        # alias commonly computed variables
        self.nx = self.x.size
        self.nt = self.time.size
        if self.transatt:
            self.nta = self.transatt.size
        else:
            self.nta = 0

        pass

    def __repr__(self):
        # __repr__ from xarray is used and edited.
        #   'xarray' is prepended. so we remove it and add 'dtscalibration'
        s = xr.core.formatting.dataset_repr(self._obj)
        name_module = type(self._obj).__name__
        preamble_new = "<dtscalibration.%s>" % name_module

        # Add sections to new preamble
        preamble_new += "\nSections:"
        if hasattr(self._obj, "_sections") and self.sections:
            preamble_new += "\n"
            unit = self.x.attrs.get("unit", "")

            for k, v in self.sections.items():
                preamble_new += f"    {k: <23}"

                # Compute statistics reference section timeseries
                sec_stat = f"({float(self._obj[k].mean()):6.2f}"
                sec_stat += f" +/-{float(self._obj[k].std()):5.2f}"
                sec_stat += "\N{DEGREE SIGN}C)\t"
                preamble_new += sec_stat

                # print sections
                vl = [f"{vi.start:.2f}{unit} - {vi.stop:.2f}{unit}" for vi in v]
                preamble_new += " and ".join(vl) + "\n"

        else:
            preamble_new += 18 * " " + "()\n"

        # add new preamble to the remainder of the former __repr__
        len_preamble_old = 8 + len(name_module) + 2

        # untill the attribute listing
        attr_index = s.find("Attributes:")

        # abbreviate attribute listing
        attr_list_all = s[attr_index:].split(sep="\n")
        if len(attr_list_all) > 10:
            s_too_many = ["\n.. and many more attributes. See: ds.attrs"]
            attr_list = attr_list_all[:10] + s_too_many
        else:
            attr_list = attr_list_all

        s_out = preamble_new + s[len_preamble_old:attr_index] + "\n".join(attr_list)

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
        if "_sections" not in self._obj.attrs:
            self._obj.attrs["_sections"] = yaml.dump(None)

        return yaml.load(self._obj.attrs["_sections"], Loader=yaml.UnsafeLoader)

    @sections.deleter
    def sections(self):
        self._obj.attrs["_sections"] = yaml.dump(None)

    @sections.setter
    def sections(self, value):
        msg = (
            "Not possible anymore. Instead, pass the sections as an argument to \n"
            "ds.dts.calibrate_single_ended() or ds.dts.calibrate_double_ended()."
        )
        raise NotImplementedError(msg)

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
            "st",
            "ast",
            "rst",
            "rast",
            "time",
            "timestart",
            "tmp",
            "timeend",
            "acquisitionTime",
            "x",
        ]
        int32l = [
            "filename_tstamp",
            "acquisitiontimeFW",
            "acquisitiontimeBW",
            "userAcquisitionTimeFW",
            "userAcquisitionTimeBW",
        ]

        # default variable compression
        compdata = dict(
            zlib=True, complevel=6, shuffle=False
        )  # , least_significant_digit=None

        # default coordinate compression
        compcoords = dict(zlib=True, complevel=4)

        # construct encoding dict
        encoding = {var: compdata.copy() for var in self._obj.data_vars}
        encoding.update({var: compcoords.copy() for var in self._obj.coords})

        for k, v in encoding.items():
            if k in float32l:
                v["dtype"] = "float32"

            if k in int32l:
                v["dtype"] = "int32"
                # v['_FillValue'] = -9999  # Int does not support NaN

            if np.issubdtype(self._obj[k].dtype, str) or np.issubdtype(
                self._obj[k].dtype, object
            ):
                # Compression not supported for variable length strings
                # https://github.com/Unidata/netcdf4-python/issues/1205
                v["zlib"] = False

        if time_chunks_from_key is not None:
            # obtain optimal chunk sizes in time and x dim
            if self[time_chunks_from_key].dims == ("x", "time"):
                x_chunk, t_chunk = da.ones(
                    self[time_chunks_from_key].shape,
                    chunks=(-1, "auto"),
                    dtype="float32",
                ).chunks

            elif self[time_chunks_from_key].dims == ("time", "x"):
                x_chunk, t_chunk = da.ones(
                    self[time_chunks_from_key].shape,
                    chunks=("auto", -1),
                    dtype="float32",
                ).chunks
            else:
                assert 0, "something went wrong with your Stokes dimensions"

            for k, v in encoding.items():
                # By writing and compressing the data in chunks, some sort of
                # parallism is possible.
                if self[k].dims == ("x", "time"):
                    chunks = (x_chunk[0], t_chunk[0])

                elif self[k].dims == ("time", "x"):
                    chunks = (t_chunk[0], x_chunk[0])

                elif self[k].dims == ("x",):
                    chunks = (x_chunk[0],)

                elif self[k].dims == ("time",):
                    chunks = (t_chunk[0],)

                else:
                    continue

                v["chunksizes"] = chunks

        return encoding

import dask.array as da
import numpy as np
import xarray as xr
import yaml
import scipy.stats as sst

from dtscalibration.calibration.section_utils import validate_sections
from dtscalibration.calibrate_utils import calibration_single_ended_helper
from dtscalibration.calibrate_utils import match_sections
from dtscalibration.calibrate_utils import parse_st_var
from dtscalibration.calibration.section_utils import set_sections
from dtscalibration.calibration.section_utils import set_matching_sections
from dtscalibration.datastore_utils import ParameterIndexSingleEnded
from dtscalibration.datastore_utils import get_params_from_pval_single_ended
from dtscalibration.datastore_utils import ufunc_per_section_helper
from dtscalibration.io_utils import dim_attrs


@xr.register_dataset_accessor("dts")
class DtsAccessor:
    def __init__(self, xarray_obj):
        # cache xarray_obj
        self._obj = xarray_obj
        self.attrs = xarray_obj.attrs

        # alias commonly used variables
        self.x = xarray_obj.x
        self.nx = self.x.size
        self.time = xarray_obj.time
        self.nt = self.time.size

        self.st = xarray_obj.get("st")
        self.ast = xarray_obj.get("ast")
        self.rst = xarray_obj.get("rst")    # None if doesn't exist
        self.rast = xarray_obj.get("rast")  # None is doesn't exist

        self.acquisitiontime_fw = xarray_obj.get("userAcquisitionTimeFW")
        self.acquisitiontime_bw = xarray_obj.get("userAcquisitionTimeBW")
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
    
    # noinspection PyIncorrectDocstring
    @property
    def matching_sections(self):
        """
        Define calibration sections. Each matching_section requires a reference
        temperature time series, such as the temperature measured by an
        external temperature sensor. They should already be part of the
        DataStore object.

        Please look at the example notebook on `matching_sections` if you encounter
        difficulties.

        Parameters
        ----------
        matching_sections : List[Tuple[slice, slice, bool]], optional
            Provide a list of tuples. A tuple per matching section. Each tuple
            has three items. The first two items are the slices of the sections
            that are matched. The third item is a boolean and is True if the two
            sections have a reverse direction ("J-configuration").
        Returns
        -------

        """
        if "_matching_sections" not in self._obj.attrs:
            self._obj.attrs["_matching_sections"] = yaml.dump(None)

        return yaml.load(self._obj.attrs["_matching_sections"], Loader=yaml.UnsafeLoader)

    @matching_sections.deleter
    def matching_sections(self):
        self._obj.attrs["_matching_sections"] = yaml.dump(None)

    @matching_sections.setter
    def matching_sections(self, value):
        msg = (
            "Not possible anymore. Instead, pass the matching_sections as an argument to \n"
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
    
    def ufunc_per_section(
        self,
        sections=None,
        func=None,
        label=None,
        subtract_from_label=None,
        temp_err=False,
        x_indices=False,
        ref_temp_broadcasted=False,
        calc_per="stretch",
        **func_kwargs,
    ):
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
        >>>     sections=sections, 
        >>>     func='var',
        >>>     calc_per='all',
        >>>     label='tmpf',
        >>>     temp_err=True)

        2. Calculate the variance of the residuals in the along PER\
        reference section wrt the temperature of the water baths

        >>> tmpf_var = d.ufunc_per_section(
        >>>     sections=sections, 
        >>>     func='var',
        >>>     calc_per='stretch',
        >>>     label='tmpf',
        >>>     temp_err=True)

        3. Calculate the variance of the residuals in the along PER\
        water bath wrt the temperature of the water baths

        >>> tmpf_var = d.ufunc_per_section(
        >>>     sections=sections, 
        >>>     func='var',
        >>>     calc_per='section',
        >>>     label='tmpf',
        >>>     temp_err=True)

        4. Obtain the coordinates of the measurements per section

        >>> locs = d.ufunc_per_section(
        >>>     sections=sections, 
        >>>     func=None,
        >>>     label='x',
        >>>     temp_err=False,
        >>>     ref_temp_broadcasted=False,
        >>>     calc_per='stretch')

        5. Number of observations per stretch

        >>> nlocs = d.ufunc_per_section(
        >>>     sections=sections, 
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

        >>> ix_loc = d.ufunc_per_section(sections=sections, x_indices=True)


        Note
        ----
        If `self[label]` or `self[subtract_from_label]` is a Dask array, a Dask
        array is returned else a numpy array is returned
        """
        if label is None:
            dataarray = None
        else:
            dataarray = self._obj[label]

        if x_indices:
            x_coords = self.x
            reference_dataset = None

        else:
            validate_sections(self._obj, sections)

            x_coords = None
            reference_dataset = {k: self._obj[k] for k in sections}

        out = ufunc_per_section_helper(
            x_coords=x_coords,
            sections=sections,
            func=func,
            dataarray=dataarray,
            subtract_from_dataarray=subtract_from_label,
            reference_dataset=reference_dataset,
            subtract_reference_from_dataarray=temp_err,
            ref_temp_broadcasted=ref_temp_broadcasted,
            calc_per=calc_per,
            **func_kwargs,
        )
        return out

    def calibrate_single_ended(
        self,
        sections,
        st_var,
        ast_var,
        method="wls",
        solver="sparse",
        p_val=None,
        p_var=None,
        p_cov=None,
        matching_sections=None,
        trans_att=[],
        fix_gamma=None,
        fix_dalpha=None,
        fix_alpha=None,
    ):
        r"""
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
        sections : Dict[str, List[slice]]
            Each section requires a reference temperature time series,
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
        method : {'wls',}
            Use `'wls'` for weighted least
            squares.
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
        fix_alpha : Tuple[array-like, array-like], optional
            A tuple containing two array-likes. The first array-like is the integrated
            differential attenuation of length x, and the second item is its variance.

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
    dtscalibration/python-dts-calibration/blob/main/examples/notebooks/\
    07Calibrate_single_wls.ipynb>`_

        """
        assert self.st.dims[0] == "x", "Stokes are transposed"
        assert self.ast.dims[0] == "x", "Stokes are transposed"

        # out contains the state
        out = xr.Dataset(coords={"x": self.x, "time": self.time, "trans_att": trans_att}).copy()
        out.coords["x"].attrs = dim_attrs["x"]
        out.coords["trans_att"].attrs = dim_attrs["trans_att"]

        nta = len(trans_att)

        # check and store sections and matching_sections
        validate_sections(self._obj, sections=sections)
        set_sections(out, sections)
        set_matching_sections(out, matching_sections)

        # Convert sections and matching_sections to indices
        ix_sec = self.ufunc_per_section(
            sections=sections, x_indices=True, calc_per="all"
        )
        if matching_sections:
            matching_indices = match_sections(self, matching_sections)
        else:
            matching_indices = None

        assert not np.any(self.st.isel(x=ix_sec) <= 0.0), (
            "There is uncontrolled noise in the ST signal. Are your sections"
            "correctly defined?"
        )
        assert not np.any(self.ast.isel(x=ix_sec) <= 0.0), (
            "There is uncontrolled noise in the AST signal. Are your sections"
            "correctly defined?"
        )

        if method == "wls":
            p_cov, p_val, p_var = calibration_single_ended_helper(
                self._obj,
                sections,
                st_var,
                ast_var,
                fix_alpha,
                fix_dalpha,
                fix_gamma,
                matching_indices,
                trans_att,
                solver,
            )

        elif method == "external":
            for input_item in [p_val, p_var, p_cov]:
                assert (
                    input_item is not None
                ), "Define p_val, p_var, p_cov when using an external solver"

        else:
            raise ValueError("Choose a valid method")

        # all below require the following solution sizes
        if fix_alpha:
            ip = ParameterIndexSingleEnded(
                self.nt, self.nx, nta, includes_alpha=True, includes_dalpha=False
            )
        else:
            ip = ParameterIndexSingleEnded(
                self.nt, self.nx, nta, includes_alpha=False, includes_dalpha=True
            )

        # npar = 1 + 1 + nt + nta * nt
        assert p_val.size == ip.npar
        assert p_var.size == ip.npar
        assert p_cov.shape == (ip.npar, ip.npar)

        # store calibration parameters in DataStore
        params, param_covs = get_params_from_pval_single_ended(
            ip, out.coords, p_val=p_val, p_var=p_var, p_cov=p_cov, fix_alpha=fix_alpha
        )

        tmpf = params["gamma"] / (
            (np.log(self.st / self.ast) + (params["c"] + params["talpha_fw_full"]))
            + params["alpha"]
        )

        out["tmpf"] = tmpf - 273.15
        out["tmpf"].attrs.update(dim_attrs["tmpf"])

        # tmpf_var
        deriv_dict = dict(
            T_gamma_fw=tmpf / params["gamma"],
            T_st_fw=-(tmpf**2) / (params["gamma"] * self.st),
            T_ast_fw=tmpf**2 / (params["gamma"] * self.ast),
            T_c_fw=-(tmpf**2) / params["gamma"],
            T_dalpha_fw=-self.x * (tmpf**2) / params["gamma"],
            T_alpha_fw=-(tmpf**2) / params["gamma"],
            T_ta_fw=-(tmpf**2) / params["gamma"],
        )
        deriv_ds = xr.Dataset(deriv_dict)

        var_fw_dict = dict(
            dT_dst=deriv_ds.T_st_fw**2 * parse_st_var(self.st, st_var),
            dT_dast=deriv_ds.T_ast_fw**2
            * parse_st_var(self.ast, ast_var),
            dT_gamma=deriv_ds.T_gamma_fw**2 * param_covs["gamma"],
            dT_dc=deriv_ds.T_c_fw**2 * param_covs["c"],
            dT_ddalpha=deriv_ds.T_alpha_fw**2
            * param_covs["alpha"],  # same as dT_dalpha
            dT_dta=deriv_ds.T_ta_fw**2 * param_covs["talpha_fw_full"],
            dgamma_dc=(
                2 * deriv_ds.T_gamma_fw * deriv_ds.T_c_fw * param_covs["gamma_c"]
            ),
            dta_dgamma=(
                2 * deriv_ds.T_ta_fw * deriv_ds.T_gamma_fw * param_covs["tafw_gamma"]
            ),
            dta_dc=(2 * deriv_ds.T_ta_fw * deriv_ds.T_c_fw * param_covs["tafw_c"]),
        )

        if not fix_alpha:
            # These correlations don't exist in case of fix_alpha. Including them reduces tmpf_var.
            var_fw_dict.update(
                dict(
                    dgamma_ddalpha=(
                        2
                        * deriv_ds.T_gamma_fw
                        * deriv_ds.T_dalpha_fw
                        * param_covs["gamma_dalpha"]
                    ),
                    ddalpha_dc=(
                        2
                        * deriv_ds.T_dalpha_fw
                        * deriv_ds.T_c_fw
                        * param_covs["dalpha_c"]
                    ),
                    dta_ddalpha=(
                        2
                        * deriv_ds.T_ta_fw
                        * deriv_ds.T_dalpha_fw
                        * param_covs["tafw_dalpha"]
                    ),
                )
            )

        out["var_fw_da"] = xr.Dataset(var_fw_dict).to_array(dim="comp_fw")
        out["tmpf_var"] = out["var_fw_da"].sum(dim="comp_fw")
        out["tmpf_var"].attrs.update(dim_attrs["tmpf_var"])

        out["p_val"] = (("params1",), p_val)
        out["p_cov"] = (("params1", "params2"), p_cov)

        out.update(params)
        for key, dataarray in param_covs.data_vars.items():
            out[key + "_var"] = dataarray

        return out
    

    def monte_carlo_single_ended(
        self,
        result,
        st_var,
        ast_var,
        conf_ints=[],
        mc_sample_size=100,
        da_random_state=None,
        reduce_memory_usage=False,
        mc_remove_set_flag=True):
        """The result object is what comes out of the single_ended_calibration routine)
        
        TODO: Use get_params_from_pval_single_ended() to extract parameter sets from mc
        """
        assert self.st.dims[0] == "x", "Stokes are transposed"
        assert self.ast.dims[0] == "x", "Stokes are transposed"

        if da_random_state:
            state = da_random_state
        else:
            state = da.random.RandomState()

        # out contains the state
        out = xr.Dataset(coords={"x": self.x, "time": self.time, "trans_att": result["trans_att"]}).copy()
        out.coords["x"].attrs = dim_attrs["x"]
        out.coords["trans_att"].attrs = dim_attrs["trans_att"]
        out.coords["CI"] = conf_ints

        set_sections(out, result.dts.sections)
        set_matching_sections(out, result.dts.matching_sections)

        params = out.copy()
        params.coords["mc"] = range(mc_sample_size)

        no, nt = self.st.data.shape
        nta = result["trans_att"].size

        p_val = result["p_val"].data
        p_cov = result["p_cov"].data

        npar = p_val.size

        # check number of parameters
        if npar == nt + 2 + nt * nta:
            fixed_alpha = False
        elif npar == 1 + no + nt + nt * nta:
            fixed_alpha = True
        else:
            raise Exception("The size of `p_val` is not what I expected")
        
        assert isinstance(p_val, (str, np.ndarray, np.generic))
        if isinstance(p_val, str):
            p_val = self[p_val].data

        npar = p_val.size
        p_mc = sst.multivariate_normal.rvs(mean=p_val, cov=p_cov, size=mc_sample_size)

        if fixed_alpha:
            params["alpha_mc"] = (("mc", "x"), p_mc[:, 1 : no + 1])
            params["c_mc"] = (("mc", "time"), p_mc[:, 1 + no : 1 + no + nt])
        else:
            params["dalpha_mc"] = (("mc",), p_mc[:, 1])
            params["c_mc"] = (("mc", "time"), p_mc[:, 2 : nt + 2])

        params["gamma_mc"] = (("mc",), p_mc[:, 0])
        if nta:
            params["ta_mc"] = (
                ("mc", "trans_att", "time"),
                np.reshape(p_mc[:, -nt * nta :], (mc_sample_size, nta, nt)),
            )

        rsize = (params.mc.size, params.x.size, params.time.size)

        if reduce_memory_usage:
            memchunk = da.ones(
                (mc_sample_size, no, nt), chunks={0: -1, 1: 1, 2: "auto"}
            ).chunks
        else:
            memchunk = da.ones(
                (mc_sample_size, no, nt), chunks={0: -1, 1: "auto", 2: "auto"}
            ).chunks


        # Draw from the normal distributions for the Stokes intensities
        for key_mc, sti, st_vari in zip(["r_st", "r_ast"], [self.st, self.ast], 
                                        [st_var, ast_var]):
        # for k, st_labeli, st_vari in zip(
        #     ["r_st", "r_ast"], ["st", "ast"], [st_var, ast_var]
        # ):
            # Load the mean as chunked Dask array, otherwise eats memory
            if type(sti.data) == da.core.Array:
                loc = da.asarray(sti.data, chunks=memchunk[1:])
            else:
                loc = da.from_array(sti.data, chunks=memchunk[1:])

            # Make sure variance is of size (no, nt)
            if np.size(st_vari) > 1:
                if st_vari.shape == sti.shape:
                    pass
                else:
                    st_vari = np.broadcast_to(st_vari, (no, nt))
            else:
                pass

            # Load variance as chunked Dask array, otherwise eats memory
            if type(st_vari) == da.core.Array:
                st_vari_da = da.asarray(st_vari, chunks=memchunk[1:])

            elif callable(st_vari) and type(sti.data) == da.core.Array:
                st_vari_da = da.asarray(
                    st_vari(sti).data, chunks=memchunk[1:]
                )

            elif callable(st_vari) and type(sti.data) != da.core.Array:
                st_vari_da = da.from_array(
                    st_vari(sti).data, chunks=memchunk[1:]
                )

            else:
                st_vari_da = da.from_array(st_vari, chunks=memchunk[1:])

            params[key_mc] = (
                ("mc", "x", "time"),
                state.normal(
                    loc=loc,  # has chunks=memchunk[1:]
                    scale=st_vari_da**0.5,
                    size=rsize,
                    chunks=memchunk,
                ),
            )

        ta_arr = np.zeros((mc_sample_size, no, nt))

        if nta:
            for ii, ta in enumerate(params["ta_mc"]):
                for tai, taxi in zip(ta.values, result["trans_att"].values):
                    ta_arr[ii, self.x.values >= taxi] = (
                        ta_arr[ii, self.x.values >= taxi] + tai
                    )
        params["ta_mc_arr"] = (("mc", "x", "time"), ta_arr)

        if fixed_alpha:
            params["tmpf_mc_set"] = (
                params["gamma_mc"]
                / (
                    (
                        np.log(params["r_st"])
                        - np.log(params["r_ast"])
                        + (params["c_mc"] + params["ta_mc_arr"])
                    )
                    + params["alpha_mc"]
                )
                - 273.15
            )
        else:
            params["tmpf_mc_set"] = (
                params["gamma_mc"]
                / (
                    (
                        np.log(params["r_st"])
                        - np.log(params["r_ast"])
                        + (params["c_mc"] + params["ta_mc_arr"])
                    )
                    + (params["dalpha_mc"] * params.x)
                )
                - 273.15
            )

        avg_dims = ["mc"]
        avg_axis = params["tmpf_mc_set"].get_axis_num(avg_dims)
        out["tmpf_mc_var"] = (params["tmpf_mc_set"] - result["tmpf"]).var(
            dim=avg_dims, ddof=1
        )

        if conf_ints:
            new_chunks = ((len(conf_ints),),) + params["tmpf_mc_set"].chunks[1:]

            qq = params["tmpf_mc_set"]

            q = qq.data.map_blocks(
                lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
                chunks=new_chunks,  #
                drop_axis=avg_axis,  # avg dimesnions are dropped from input arr
                new_axis=0,
            )  # The new CI dimension is added as first axis

            out["tmpf_mc"] = (("CI", "x", "time"), q)

        if not mc_remove_set_flag:
            out.update(params)

        return out

    def monte_carlo_double_ended(
        self,
        sections=None,
        p_val="p_val",
        p_cov="p_cov",
        st_var=None,
        ast_var=None,
        rst_var=None,
        rast_var=None,
        conf_ints=None,
        mc_sample_size=100,
        var_only_sections=False,
        da_random_state=None,
        mc_remove_set_flag=True,
        reduce_memory_usage=False,
        **kwargs,
    ):
        r"""
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

        Assumes sections are set.

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
        mc_remove_set_flag : bool
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

        def create_da_ta2(no, i_splice, direction="fw", chunks=None):
            """create mask array mc, o, nt"""

            if direction == "fw":
                arr = da.concatenate(
                    (
                        da.zeros((1, i_splice, 1), chunks=(1, i_splice, 1), dtype=bool),
                        da.ones(
                            (1, no - i_splice, 1),
                            chunks=(1, no - i_splice, 1),
                            dtype=bool,
                        ),
                    ),
                    axis=1,
                ).rechunk((1, chunks[1], 1))
            else:
                arr = da.concatenate(
                    (
                        da.ones((1, i_splice, 1), chunks=(1, i_splice, 1), dtype=bool),
                        da.zeros(
                            (1, no - i_splice, 1),
                            chunks=(1, no - i_splice, 1),
                            dtype=bool,
                        ),
                    ),
                    axis=1,
                ).rechunk((1, chunks[1], 1))
            return arr

        out = xr.Dataset()
        params = xr.Dataset()

        if da_random_state:
            # In testing environments
            assert isinstance(da_random_state, da.random.RandomState)
            state = da_random_state
        else:
            state = da.random.RandomState()

        if conf_ints:
            assert "tmpw", (
                "Current implementation requires you to "
                'define "tmpw" when estimating confidence '
                "intervals"
            )

        no, nt = self.st.shape
        nta = self.trans_att.size
        npar = 1 + 2 * nt + no + nt * 2 * nta  # number of parameters

        rsize = (mc_sample_size, no, nt)

        if reduce_memory_usage:
            memchunk = da.ones(
                (mc_sample_size, no, nt), chunks={0: -1, 1: 1, 2: "auto"}
            ).chunks
        else:
            memchunk = da.ones(
                (mc_sample_size, no, nt), chunks={0: -1, 1: "auto", 2: "auto"}
            ).chunks

        params.coords["mc"] = range(mc_sample_size)
        params.coords["x"] = self.x
        params.coords["time"] = self.time

        if conf_ints:
            self.coords["CI"] = conf_ints
            params.coords["CI"] = conf_ints

        assert isinstance(p_val, (str, np.ndarray, np.generic))
        if isinstance(p_val, str):
            p_val = self[p_val].values
        assert p_val.shape == (npar,), (
            "Did you set 'talpha' as "
            "keyword argument of the "
            "conf_int_double_ended() function?"
        )

        assert isinstance(p_cov, (str, np.ndarray, np.generic, bool))

        if isinstance(p_cov, bool) and not p_cov:
            # Exclude parameter uncertainty if p_cov == False
            gamma = p_val[0]
            d_fw = p_val[1 : nt + 1]
            d_bw = p_val[1 + nt : 2 * nt + 1]
            alpha = p_val[2 * nt + 1 : 2 * nt + 1 + no]

            params["gamma_mc"] = (tuple(), gamma)
            params["alpha_mc"] = (("x",), alpha)
            params["df_mc"] = (("time",), d_fw)
            params["db_mc"] = (("time",), d_bw)

            if nta:
                ta = p_val[2 * nt + 1 + no :].reshape((nt, 2, nta), order="F")
                ta_fw = ta[:, 0, :]
                ta_bw = ta[:, 1, :]

                ta_fw_arr = np.zeros((no, nt))
                for tai, taxi in zip(ta_fw.T, params.coords["trans_att"].values):
                    ta_fw_arr[params.x.values >= taxi] = (
                        ta_fw_arr[params.x.values >= taxi] + tai
                    )

                ta_bw_arr = np.zeros((no, nt))
                for tai, taxi in zip(ta_bw.T, params.coords["trans_att"].values):
                    ta_bw_arr[params.x.values < taxi] = (
                        ta_bw_arr[params.x.values < taxi] + tai
                    )

                params["talpha_fw_mc"] = (("x", "time"), ta_fw_arr)
                params["talpha_bw_mc"] = (("x", "time"), ta_bw_arr)

        elif isinstance(p_cov, bool) and p_cov:
            raise NotImplementedError("Not an implemented option. Check p_cov argument")

        else:
            # WLS
            if isinstance(p_cov, str):
                p_cov = self[p_cov].values
            assert p_cov.shape == (npar, npar)

            assert sections is not None, "Define sections"
            ix_sec = self.ufunc_per_section(
                sections=sections, x_indices=True, calc_per="all"
            )
            nx_sec = ix_sec.size
            from_i = np.concatenate(
                (
                    np.arange(1 + 2 * nt),
                    1 + 2 * nt + ix_sec,
                    np.arange(1 + 2 * nt + no, 1 + 2 * nt + no + nt * 2 * nta),
                )
            )
            iox_sec1, iox_sec2 = np.meshgrid(from_i, from_i, indexing="ij")
            po_val = p_val[from_i]
            po_cov = p_cov[iox_sec1, iox_sec2]

            po_mc = sst.multivariate_normal.rvs(
                mean=po_val, cov=po_cov, size=mc_sample_size
            )

            gamma = po_mc[:, 0]
            d_fw = po_mc[:, 1 : nt + 1]
            d_bw = po_mc[:, 1 + nt : 2 * nt + 1]

            params["gamma_mc"] = (("mc",), gamma)
            params["df_mc"] = (("mc", "time"), d_fw)
            params["db_mc"] = (("mc", "time"), d_bw)

            # calculate alpha seperately
            alpha = np.zeros((mc_sample_size, no), dtype=float)
            alpha[:, ix_sec] = po_mc[:, 1 + 2 * nt : 1 + 2 * nt + nx_sec]

            not_ix_sec = np.array([i for i in range(no) if i not in ix_sec])

            if np.any(not_ix_sec):
                not_alpha_val = p_val[2 * nt + 1 + not_ix_sec]
                not_alpha_var = p_cov[2 * nt + 1 + not_ix_sec, 2 * nt + 1 + not_ix_sec]

                not_alpha_mc = np.random.normal(
                    loc=not_alpha_val,
                    scale=not_alpha_var**0.5,
                    size=(mc_sample_size, not_alpha_val.size),
                )

                alpha[:, not_ix_sec] = not_alpha_mc

            params["alpha_mc"] = (("mc", "x"), alpha)

            if nta:
                ta = po_mc[:, 2 * nt + 1 + nx_sec :].reshape(
                    (mc_sample_size, nt, 2, nta), order="F"
                )
                ta_fw = ta[:, :, 0, :]
                ta_bw = ta[:, :, 1, :]

                ta_fw_arr = da.zeros(
                    (mc_sample_size, no, nt), chunks=memchunk, dtype=float
                )
                for tai, taxi in zip(
                    ta_fw.swapaxes(0, 2), params.coords["trans_att"].values
                ):
                    # iterate over the splices
                    i_splice = sum(params.x.values < taxi)
                    mask = create_da_ta2(no, i_splice, direction="fw", chunks=memchunk)

                    ta_fw_arr += mask * tai.T[:, None, :]

                ta_bw_arr = da.zeros(
                    (mc_sample_size, no, nt), chunks=memchunk, dtype=float
                )
                for tai, taxi in zip(
                    ta_bw.swapaxes(0, 2), params.coords["trans_att"].values
                ):
                    i_splice = sum(params.x.values < taxi)
                    mask = create_da_ta2(no, i_splice, direction="bw", chunks=memchunk)

                    ta_bw_arr += mask * tai.T[:, None, :]

                params["talpha_fw_mc"] = (("mc", "x", "time"), ta_fw_arr)
                params["talpha_bw_mc"] = (("mc", "x", "time"), ta_bw_arr)

        # Draw from the normal distributions for the Stokes intensities
        for k, st_labeli, st_vari in zip(
            ["r_st", "r_ast", "r_rst", "r_rast"],
            ["st", "ast", "rst", "rast"],
            [st_var, ast_var, rst_var, rast_var],
        ):
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

            elif callable(st_vari) and type(self[st_labeli].data) == da.core.Array:
                st_vari_da = da.asarray(
                    st_vari(self[st_labeli]).data, chunks=memchunk[1:]
                )

            elif callable(st_vari) and type(self[st_labeli].data) != da.core.Array:
                st_vari_da = da.from_array(
                    st_vari(self[st_labeli]).data, chunks=memchunk[1:]
                )

            else:
                st_vari_da = da.from_array(st_vari, chunks=memchunk[1:])

            params[k] = (
                ("mc", "x", "time"),
                state.normal(
                    loc=loc,  # has chunks=memchunk[1:]
                    scale=st_vari_da**0.5,
                    size=rsize,
                    chunks=memchunk,
                ),
            )

        for label in ["tmpf", "tmpb"]:
            if "tmpw" or label:
                if label == "tmpf":
                    if nta:
                        params["tmpf_mc_set"] = (
                            params["gamma_mc"]
                            / (
                                np.log(params["r_st"] / params["r_ast"])
                                + params["df_mc"]
                                + params["alpha_mc"]
                                + params["talpha_fw_mc"]
                            )
                            - 273.15
                        )
                    else:
                        params["tmpf_mc_set"] = (
                            params["gamma_mc"]
                            / (
                                np.log(params["r_st"] / params["r_ast"])
                                + params["df_mc"]
                                + params["alpha_mc"]
                            )
                            - 273.15
                        )
                else:
                    if nta:
                        params["tmpb_mc_set"] = (
                            params["gamma_mc"]
                            / (
                                np.log(params["r_rst"] / params["r_rast"])
                                + params["db_mc"]
                                - params["alpha_mc"]
                                + params["talpha_bw_mc"]
                            )
                            - 273.15
                        )
                    else:
                        params["tmpb_mc_set"] = (
                            params["gamma_mc"]
                            / (
                                np.log(params["r_rst"] / params["r_rast"])
                                + params["db_mc"]
                                - params["alpha_mc"]
                            )
                            - 273.15
                        )

                if var_only_sections:
                    # sets the values outside the reference sections to NaN
                    xi = self.ufunc_per_section(
                        sections=sections, x_indices=True, calc_per="all"
                    )
                    x_mask_ = [
                        True if ix in xi else False for ix in range(params.x.size)
                    ]
                    x_mask = np.reshape(x_mask_, (1, -1, 1))
                    params[label + "_mc_set"] = params[label + "_mc_set"].where(x_mask)

                # subtract the mean temperature
                q = params[label + "_mc_set"] - self[label]
                out[label + "_mc_var"] = q.var(dim="mc", ddof=1)

                if conf_ints:
                    new_chunks = list(params[label + "_mc_set"].chunks)
                    new_chunks[0] = (len(conf_ints),)
                    avg_axis = params[label + "_mc_set"].get_axis_num("mc")
                    q = params[label + "_mc_set"].data.map_blocks(
                        lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
                        chunks=new_chunks,  #
                        drop_axis=avg_axis,
                        # avg dimensions are dropped from input arr
                        new_axis=0,
                    )  # The new CI dimension is added as firsaxis

                    out[label + "_mc"] = (("CI", "x", "time"), q)

        # Weighted mean of the forward and backward
        tmpw_var = 1 / (1 / out["tmpf_mc_var"] + 1 / out["tmpb_mc_var"])

        q = (
            params["tmpf_mc_set"] / out["tmpf_mc_var"]
            + params["tmpb_mc_set"] / out["tmpb_mc_var"]
        ) * tmpw_var

        params["tmpw" + "_mc_set"] = q  #

        out["tmpw"] = (
            self["tmpf"] / out["tmpf_mc_var"] + self["tmpb"] / out["tmpb_mc_var"]
        ) * tmpw_var

        q = params["tmpw" + "_mc_set"] - self["tmpw"]
        out["tmpw" + "_mc_var"] = q.var(dim="mc", ddof=1)

        # Calculate the CI of the weighted MC_set
        if conf_ints:
            new_chunks_weighted = ((len(conf_ints),),) + memchunk[1:]
            avg_axis = params["tmpw" + "_mc_set"].get_axis_num("mc")
            q2 = params["tmpw" + "_mc_set"].data.map_blocks(
                lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
                chunks=new_chunks_weighted,  # Explicitly define output chunks
                drop_axis=avg_axis,  # avg dimensions are dropped
                new_axis=0,
                dtype=float,
            )  # The new CI dimension is added as first axis
            out["tmpw" + "_mc"] = (("CI", "x", "time"), q2)

        # Clean up the garbage. All arrays with a Monte Carlo dimension.
        if mc_remove_set_flag:
            remove_mc_set = [
                "r_st",
                "r_ast",
                "r_rst",
                "r_rast",
                "gamma_mc",
                "alpha_mc",
                "df_mc",
                "db_mc",
            ]

            for i in ["tmpf", "tmpb", "tmpw"]:
                remove_mc_set.append(i + "_mc_set")

            if nta:
                remove_mc_set.append('talpha"_fw_mc')
                remove_mc_set.append('talpha"_bw_mc')

            for k in remove_mc_set:
                if k in out:
                    del out[k]

        if not mc_remove_set_flag:
            out.update(params)

        self.update(out)
        return out

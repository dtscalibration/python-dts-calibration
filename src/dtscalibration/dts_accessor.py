import dask.array as da
import numpy as np
import xarray as xr
import yaml
import scipy.stats as sst

from dtscalibration.calibration.section_utils import validate_sections, validate_sections_definition, validate_no_overlapping_sections
from dtscalibration.calibrate_utils import calibrate_double_ended_helper
from dtscalibration.calibrate_utils import calibration_single_ended_helper
from dtscalibration.calibrate_utils import parse_st_var
from dtscalibration.calibration.section_utils import set_sections
from dtscalibration.calibration.section_utils import set_matching_sections
from dtscalibration.datastore_utils import ParameterIndexDoubleEnded
from dtscalibration.datastore_utils import ParameterIndexSingleEnded
from dtscalibration.datastore_utils import get_params_from_pval_double_ended
from dtscalibration.datastore_utils import get_params_from_pval_single_ended
from dtscalibration.datastore_utils import ufunc_per_section_helper
from dtscalibration.io.utils import dim_attrs


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

        # None if doesn't exist
        self.st = xarray_obj.get("st")
        self.ast = xarray_obj.get("ast")
        self.rst = xarray_obj.get("rst")    
        self.rast = xarray_obj.get("rast")

        self.acquisitiontime_fw = xarray_obj.get("userAcquisitionTimeFW")
        self.acquisitiontime_bw = xarray_obj.get("userAcquisitionTimeBW")
        pass

    def __repr__(self):
        # __repr__ from xarray is used and edited.
        #   'xarray' is prepended. so we remove it and add 'dtscalibration'
        s = xr.core.formatting.dataset_repr(self._obj)
        name_module = type(self._obj).__name__
        preamble_new = f"<dtscalibration.{name_module}>"

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
            if self._obj[time_chunks_from_key].dims == ("x", "time"):
                x_chunk, t_chunk = da.ones(
                    self._obj[time_chunks_from_key].shape,
                    chunks=(-1, "auto"),
                    dtype="float32",
                ).chunks

            elif self._obj[time_chunks_from_key].dims == ("time", "x"):
                x_chunk, t_chunk = da.ones(
                    self._obj[time_chunks_from_key].shape,
                    chunks=("auto", -1),
                    dtype="float32",
                ).chunks
            else:
                assert 0, "something went wrong with your Stokes dimensions"

            for k, v in encoding.items():
                # By writing and compressing the data in chunks, some sort of
                # parallism is possible.
                if self._obj[k].dims == ("x", "time"):
                    chunks = (x_chunk[0], t_chunk[0])

                elif self._obj[k].dims == ("time", "x"):
                    chunks = (t_chunk[0], x_chunk[0])

                elif self._obj[k].dims == ("x",):
                    chunks = (x_chunk[0],)

                elif self._obj[k].dims == ("time",):
                    chunks = (t_chunk[0],)

                else:
                    continue

                v["chunksizes"] = chunks

        return encoding
    
    def get_timeseries_keys(self):
        """
        Returns a list of the keys of the time series variables.
        """
        return [k for k, v in self._obj.data_vars.items() if v.dims == ("time",)]
    
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
        suppress_section_validation=False,
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
        if not suppress_section_validation:
            validate_sections_definition(sections=sections)
            validate_no_overlapping_sections(sections=sections)

        if temp_err or ref_temp_broadcasted:
            for k in sections:
                assert k in self._obj, f"{k} is not in the Dataset but is in `sections` and is required to compute temp_err"
        
        if label is None:
            dataarray = None
        else:
            dataarray = self._obj[label]

        if x_indices:
            x_coords = self.x
            reference_dataset = None

        else:
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
        # out contains the state
        out = xr.Dataset(coords={"x": self.x, "time": self.time, "trans_att": trans_att}).copy()
        out.coords["x"].attrs = dim_attrs["x"]
        out.coords["trans_att"].attrs = dim_attrs["trans_att"]

        nta = len(trans_att)

        # check and store sections and matching_sections
        validate_sections(self._obj, sections=sections)
        set_sections(out, sections)
        set_matching_sections(out, matching_sections)

        assert self.st.dims[0] == "x", "Stokes are transposed"
        assert self.ast.dims[0] == "x", "Stokes are transposed"

        ix_sec = self.ufunc_per_section(
            sections=sections, x_indices=True, calc_per="all"
        )
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
                matching_sections,
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
    

    def calibrate_double_ended(
        self,
        sections,
        st_var,
        ast_var,
        rst_var,
        rast_var,
        method="wls",
        solver="sparse",
        p_val=None,
        p_var=None,
        p_cov=None,
        trans_att=[],
        fix_gamma=None,
        fix_alpha=None,
        matching_sections=None,
        verbose=False,
    ):
        r"""
        See example notebook 8 for an explanation on how to use this function.
        Calibrate the Stokes (`ds.st`) and anti-Stokes (`ds.ast`) of the forward
        channel and from the backward channel (`ds.rst`, `ds.rast`) data to
        temperature using fiber sections with a known temperature
        (`ds.sections`) for double-ended setups. The calibrated temperature of
        the forward channel is stored under `ds.tmpf` and its variance under
        `ds.tmpf_var`, and that of the backward channel under `ds.tmpb` and
        `ds.tmpb_var`. The inverse-variance weighted average of the forward and
        backward channel is stored under `ds.tmpw` and `ds.tmpw_var`.
        In double-ended setups, Stokes and anti-Stokes intensity is measured in
        two directions from both ends of the fiber. The forward-channel
        measurements are denoted with subscript F, and the backward-channel
        measurements are denoted with subscript B. Both measurement channels
        start at a different end of the fiber and have opposite directions, and
        therefore have different spatial coordinates.
        The first processing step
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
        detector is needed. However, it is assumed in this package that the
        forward channel and backward channel are measured simultaneously, so
        that the temperature of both measurements is the same. This assumption
        holds better for short acquisition times with respect to the timescale
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
        :param mc_conf_ints:
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
            A(x) = \int_{x_1}^x{\Delta\\alpha(x')\,\mathrm{d}x'}
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
        mc_sample_size : int, optional
            If set, the variance is also computed using Monte Carlo sampling.
            The number of Monte Carlo samples drawn used to estimate the
            variance of the forward and backward channel temperature estimates
            and estimate the inverse-variance weighted average temperature.
        conf_ints : iterable object of float
            A list with the confidence boundaries that are calculated. Valid
            values are between [0, 1].
        mc_da_random_state
            For testing purposes. Similar to random seed. The seed for dask.
            Makes random not so random. To produce reproducable results for
            testing environments.
        mc_remove_set_flag : bool
            Remove the monte carlo data set, from which the CI and the
            variance are calculated.
        variance_suffix : str, optional
            String appended for storing the variance. Only used when method
            is wls.
        method : {'wls', 'external'}
            Use `'wls'` for weighted least squares.
        solver : {'sparse', 'stats'}
            Either use the homemade weighted sparse solver or the weighted
            dense matrix solver of statsmodels. The sparse solver uses much less
            memory, is faster, and gives the same result as the statsmodels
            solver. The statsmodels solver is mostly used to check the sparse
            solver. `'stats'` is the default.
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
        # out contains the state
        out = xr.Dataset(coords={"x": self.x, "time": self.time, "trans_att": trans_att}).copy()
        out.coords["x"].attrs = dim_attrs["x"]
        out.coords["trans_att"].attrs = dim_attrs["trans_att"]

        nta = len(trans_att)

        # check and store sections and matching_sections
        validate_sections(self._obj, sections=sections)
        set_sections(out, sections)
        set_matching_sections(out, matching_sections)

        # TODO: confidence intervals using the variance approximated by linear error propagation
        assert self.st.dims[0] == "x", "Stokes are transposed"
        assert self.ast.dims[0] == "x", "Stokes are transposed"
        assert self.rst.dims[0] == "x", "Stokes are transposed"
        assert self.rast.dims[0] == "x", "Stokes are transposed"

        ix_sec = self.ufunc_per_section(
            sections=sections, x_indices=True, calc_per="all"
        )
        assert not np.any(self.st.isel(x=ix_sec) <= 0.0), (
            "There is uncontrolled noise in the ST signal. Are your sections"
            "correctly defined?"
        )
        assert not np.any(self.ast.isel(x=ix_sec) <= 0.0), (
            "There is uncontrolled noise in the AST signal. Are your sections"
            "correctly defined?"
        )
        assert not np.any(self.rst.isel(x=ix_sec) <= 0.0), (
            "There is uncontrolled noise in the REV-ST signal. Are your "
            "sections correctly defined?"
        )
        assert not np.any(self.rast.isel(x=ix_sec) <= 0.0), (
            "There is uncontrolled noise in the REV-AST signal. Are your "
            "sections correctly defined?"
        )        

        if method == "wls":
            p_cov, p_val, p_var = calibrate_double_ended_helper(
                self._obj,
                sections,
                st_var,
                ast_var,
                rst_var,
                rast_var,
                fix_alpha,
                fix_gamma,
                self.nt,
                nta,
                self.nx,
                ix_sec,
                matching_sections,
                trans_att,
                solver,
                verbose,
            )

        elif method == "external":
            for input_item in [p_val, p_var, p_cov]:
                assert input_item is not None

        elif method == "external_split":
            raise ValueError("Not implemented yet")

        else:
            raise ValueError("Choose a valid method")

        # all below require the following solution sizes
        ip = ParameterIndexDoubleEnded(self.nt, self.nx, nta)

        # npar = 1 + 2 * nt + nx + 2 * nt * nta
        assert p_val.size == ip.npar
        assert p_var.size == ip.npar
        assert p_cov.shape == (ip.npar, ip.npar)

        params = get_params_from_pval_double_ended(ip, out.coords, p_val=p_val)
        param_covs = get_params_from_pval_double_ended(
            ip, out.coords, p_val=p_var, p_cov=p_cov
        )

        tmpf = params["gamma"] / (
                    np.log(self.st / self.ast)
                    + params["df"]
                    + params["alpha"]
                    + params["talpha_fw_full"]
                )
        tmpb = params["gamma"] / (
                    np.log(self.rst / self.rast)
                    + params["db"]
                    - params["alpha"]
                    + params["talpha_bw_full"]
                )
        out["tmpf"] = tmpf - 273.15
        out["tmpb"] = tmpb - 273.15

        deriv_dict = dict(
            T_gamma_fw=tmpf / params["gamma"],
            T_st_fw=-(tmpf**2) / (params["gamma"] * self.st),
            T_ast_fw=tmpf**2 / (params["gamma"] * self.ast),
            T_df_fw=-(tmpf**2) / params["gamma"],
            T_alpha_fw=-(tmpf**2) / params["gamma"],
            T_ta_fw=-(tmpf**2) / params["gamma"],
            T_gamma_bw=tmpb / params["gamma"],
            T_rst_bw=-(tmpb**2) / (params["gamma"] * self.rst),
            T_rast_bw=tmpb**2 / (params["gamma"] * self.rast),
            T_db_bw=-(tmpb**2) / params["gamma"],
            T_alpha_bw=tmpb**2 / params["gamma"],
            T_ta_bw=-(tmpb**2) / params["gamma"],
        )
        deriv_ds = xr.Dataset(deriv_dict)
        out["deriv"] = deriv_ds.to_array(dim="com2")

        var_fw_dict = dict(
            dT_dst=deriv_ds.T_st_fw**2 * parse_st_var(self.st, st_var),
            dT_dast=deriv_ds.T_ast_fw**2
            * parse_st_var(self.ast, ast_var),
            dT_gamma=deriv_ds.T_gamma_fw**2 * param_covs["gamma"],
            dT_ddf=deriv_ds.T_df_fw**2 * param_covs["df"],
            dT_dalpha=deriv_ds.T_alpha_fw**2 * param_covs["alpha"],
            dT_dta=deriv_ds.T_ta_fw**2 * param_covs["talpha_fw_full"],
            dgamma_ddf=(
                2 * deriv_ds.T_gamma_fw * deriv_ds.T_df_fw * param_covs["gamma_df"]
            ),
            dgamma_dalpha=(
                2
                * deriv_ds.T_gamma_fw
                * deriv_ds.T_alpha_fw
                * param_covs["gamma_alpha"]
            ),
            dalpha_ddf=(
                2 * deriv_ds.T_alpha_fw * deriv_ds.T_df_fw * param_covs["alpha_df"]
            ),
            dta_dgamma=(
                2 * deriv_ds.T_ta_fw * deriv_ds.T_gamma_fw * param_covs["tafw_gamma"]
            ),
            dta_ddf=(2 * deriv_ds.T_ta_fw * deriv_ds.T_df_fw * param_covs["tafw_df"]),
            dta_dalpha=(
                2 * deriv_ds.T_ta_fw * deriv_ds.T_alpha_fw * param_covs["tafw_alpha"]
            ),
        )
        var_bw_dict = dict(
            dT_drst=deriv_ds.T_rst_bw**2
            * parse_st_var(self.rst, rst_var),
            dT_drast=deriv_ds.T_rast_bw**2
            * parse_st_var(self.rast, rast_var),
            dT_gamma=deriv_ds.T_gamma_bw**2 * param_covs["gamma"],
            dT_ddb=deriv_ds.T_db_bw**2 * param_covs["db"],
            dT_dalpha=deriv_ds.T_alpha_bw**2 * param_covs["alpha"],
            dT_dta=deriv_ds.T_ta_bw**2 * param_covs["talpha_bw_full"],
            dgamma_ddb=(
                2 * deriv_ds.T_gamma_bw * deriv_ds.T_db_bw * param_covs["gamma_db"]
            ),
            dgamma_dalpha=(
                2
                * deriv_ds.T_gamma_bw
                * deriv_ds.T_alpha_bw
                * param_covs["gamma_alpha"]
            ),
            dalpha_ddb=(
                2 * deriv_ds.T_alpha_bw * deriv_ds.T_db_bw * param_covs["alpha_db"]
            ),
            dta_dgamma=(
                2 * deriv_ds.T_ta_bw * deriv_ds.T_gamma_bw * param_covs["tabw_gamma"]
            ),
            dta_ddb=(2 * deriv_ds.T_ta_bw * deriv_ds.T_db_bw * param_covs["tabw_db"]),
            dta_dalpha=(
                2 * deriv_ds.T_ta_bw * deriv_ds.T_alpha_bw * param_covs["tabw_alpha"]
            ),
        )

        out["var_fw_da"] = xr.Dataset(var_fw_dict).to_array(dim="comp_fw")
        out["var_bw_da"] = xr.Dataset(var_bw_dict).to_array(dim="comp_bw")

        out["tmpf_var"] = out["var_fw_da"].sum(dim="comp_fw")
        out["tmpb_var"] = out["var_bw_da"].sum(dim="comp_bw")

        # First estimate of tmpw_var
        out["tmpw_var" + "_approx"] = 1 / (1 / out["tmpf_var"] + 1 / out["tmpb_var"])
        out["tmpw"] = (
            (tmpf / out["tmpf_var"] + tmpb / out["tmpb_var"])
            * out["tmpw_var" + "_approx"]
        ) - 273.15

        weightsf = out["tmpw_var" + "_approx"] / out["tmpf_var"]
        weightsb = out["tmpw_var" + "_approx"] / out["tmpb_var"]

        deriv_dict2 = dict(
            T_gamma_w=weightsf * deriv_dict["T_gamma_fw"]
            + weightsb * deriv_dict["T_gamma_bw"],
            T_st_w=weightsf * deriv_dict["T_st_fw"],
            T_ast_w=weightsf * deriv_dict["T_ast_fw"],
            T_rst_w=weightsb * deriv_dict["T_rst_bw"],
            T_rast_w=weightsb * deriv_dict["T_rast_bw"],
            T_df_w=weightsf * deriv_dict["T_df_fw"],
            T_db_w=weightsb * deriv_dict["T_db_bw"],
            T_alpha_w=weightsf * deriv_dict["T_alpha_fw"]
            + weightsb * deriv_dict["T_alpha_bw"],
            T_taf_w=weightsf * deriv_dict["T_ta_fw"],
            T_tab_w=weightsb * deriv_dict["T_ta_bw"],
        )
        deriv_ds2 = xr.Dataset(deriv_dict2)

        # TODO: sigma2_tafw_tabw
        var_w_dict = dict(
            dT_dst=deriv_ds2.T_st_w**2 * parse_st_var(self.st, st_var),
            dT_dast=deriv_ds2.T_ast_w**2
            * parse_st_var(self.ast, ast_var),
            dT_drst=deriv_ds2.T_rst_w**2
            * parse_st_var(self.rst, rst_var),
            dT_drast=deriv_ds2.T_rast_w**2
            * parse_st_var(self.rast, rast_var),
            dT_gamma=deriv_ds2.T_gamma_w**2 * param_covs["gamma"],
            dT_ddf=deriv_ds2.T_df_w**2 * param_covs["df"],
            dT_ddb=deriv_ds2.T_db_w**2 * param_covs["db"],
            dT_dalpha=deriv_ds2.T_alpha_w**2 * param_covs["alpha"],
            dT_dtaf=deriv_ds2.T_taf_w**2 * param_covs["talpha_fw_full"],
            dT_dtab=deriv_ds2.T_tab_w**2 * param_covs["talpha_bw_full"],
            dgamma_ddf=2
            * deriv_ds2.T_gamma_w
            * deriv_ds2.T_df_w
            * param_covs["gamma_df"],
            dgamma_ddb=2
            * deriv_ds2.T_gamma_w
            * deriv_ds2.T_db_w
            * param_covs["gamma_db"],
            dgamma_dalpha=2
            * deriv_ds2.T_gamma_w
            * deriv_ds2.T_alpha_w
            * param_covs["gamma_alpha"],
            dgamma_dtaf=2
            * deriv_ds2.T_gamma_w
            * deriv_ds2.T_taf_w
            * param_covs["tafw_gamma"],
            dgamma_dtab=2
            * deriv_ds2.T_gamma_w
            * deriv_ds2.T_tab_w
            * param_covs["tabw_gamma"],
            ddf_ddb=2 * deriv_ds2.T_df_w * deriv_ds2.T_db_w * param_covs["df_db"],
            ddf_dalpha=2
            * deriv_ds2.T_df_w
            * deriv_ds2.T_alpha_w
            * param_covs["alpha_df"],
            ddf_dtaf=2 * deriv_ds2.T_df_w * deriv_ds2.T_taf_w * param_covs["tafw_df"],
            ddf_dtab=2 * deriv_ds2.T_df_w * deriv_ds2.T_tab_w * param_covs["tabw_df"],
            ddb_dalpha=2
            * deriv_ds2.T_db_w
            * deriv_ds2.T_alpha_w
            * param_covs["alpha_db"],
            ddb_dtaf=2 * deriv_ds2.T_db_w * deriv_ds2.T_taf_w * param_covs["tafw_db"],
            ddb_dtab=2 * deriv_ds2.T_db_w * deriv_ds2.T_tab_w * param_covs["tabw_db"],
            # dtaf_dtab=2 * deriv_ds2.T_tab_w * deriv_ds2.T_tab_w * param_covs["tafw_tabw"],
        )
        out["var_w_da"] = xr.Dataset(var_w_dict).to_array(dim="comp_w")
        out["tmpw_var"] = out["var_w_da"].sum(dim="comp_w")

        # Compute uncertainty solely due to noise in Stokes signal, neglecting parameter uncenrtainty
        tmpf_var_excl_par = (
            out["var_fw_da"].sel(comp_fw=["dT_dst", "dT_dast"]).sum(dim="comp_fw")
        )
        tmpb_var_excl_par = (
            out["var_bw_da"].sel(comp_bw=["dT_drst", "dT_drast"]).sum(dim="comp_bw")
        )
        out["tmpw_var" + "_lower"] = 1 / (1 / tmpf_var_excl_par + 1 / tmpb_var_excl_par)

        out["tmpf"].attrs.update(dim_attrs["tmpf"])
        out["tmpb"].attrs.update(dim_attrs["tmpb"])
        out["tmpw"].attrs.update(dim_attrs["tmpw"])
        out["tmpf_var"].attrs.update(dim_attrs["tmpf_var"])
        out["tmpb_var"].attrs.update(dim_attrs["tmpb_var"])
        out["tmpw_var"].attrs.update(dim_attrs["tmpw_var"])
        out["tmpw_var" + "_approx"].attrs.update(dim_attrs["tmpw_var_approx"])
        out["tmpw_var" + "_lower"].attrs.update(dim_attrs["tmpw_var_lower"])

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

        no, nt = self.st.shape
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
        result,
        st_var,
        ast_var,
        rst_var,
        rast_var,
        conf_ints,
        mc_sample_size=100,
        var_only_sections=False,
        exclude_parameter_uncertainty=False,
        da_random_state=None,
        mc_remove_set_flag=True,
        reduce_memory_usage=False
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

        TODO: Use get_params_from_pval_double_ended() to extract parameter sets from mc
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

        if da_random_state:
            # In testing environments
            assert isinstance(da_random_state, da.random.RandomState)
            state = da_random_state
        else:
            state = da.random.RandomState()

        out = xr.Dataset(coords={"x": self.x, "time": self.time, "trans_att": result["trans_att"]}).copy()
        out.coords["x"].attrs = dim_attrs["x"]
        out.coords["trans_att"].attrs = dim_attrs["trans_att"]
        out.coords["CI"] = conf_ints

        set_sections(out, result.dts.sections)
        set_matching_sections(out, result.dts.matching_sections)

        params = out.copy()  # Contains all mc sampled parameters
        params.coords["mc"] = range(mc_sample_size)

        no, nt = self.st.shape
        nta = result["trans_att"].size

        p_val = result["p_val"].data
        p_cov = result["p_cov"].data

        npar = p_val.size
        npar_valid = 1 + 2 * nt + no + nt * 2 * nta
        assert npar == npar_valid, "Inconsistent result object"

        rsize = (mc_sample_size, no, nt)

        if reduce_memory_usage:
            memchunk = da.ones(
                (mc_sample_size, no, nt), chunks={0: -1, 1: 1, 2: "auto"}
            ).chunks
        else:
            memchunk = da.ones(
                (mc_sample_size, no, nt), chunks={0: -1, 1: "auto", 2: "auto"}
            ).chunks

        if exclude_parameter_uncertainty:
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

        else:
            sections = result.dts.sections
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
        for k, sti, st_vari in zip(
            ["r_st", "r_ast", "r_rst", "r_rast"],
            [self.st, self.ast, self.rst, self.rast],
            [st_var, ast_var, rst_var, rast_var],
        ):
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
            q = params[label + "_mc_set"] - result[label]
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
            result["tmpf"] / out["tmpf_mc_var"] + result["tmpb"] / out["tmpb_mc_var"]
        ) * tmpw_var

        q = params["tmpw" + "_mc_set"] - result["tmpw"]
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

        return out
    
    def average_monte_carlo_single_ended(
        self,
        result,
        st_var,
        ast_var,
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
        mc_remove_set_flag=True,
        reduce_memory_usage=False,
    ):
        """
        Average temperatures from single-ended setups.

        Four types of averaging are implemented. Please see Example Notebook 16.


        Parameters
        ----------
        result : xr.Dataset
            The result from the `calibrate_single_ended()` method.
        st_var, ast_var : float, callable, array-like
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
            Adds "tmpw" + '_avg1' and "tmpw" + '_mc_avg1_var' to the
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
            Adds "tmpw" + '_avg2' and "tmpw" + '_mc_avg2_var' to the
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
            Adds "tmpw" + '_avgx1' and "tmpw" + '_mc_avgx1_var' to the
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
            Adds "tmpw" + '_avg2' and "tmpw" + '_mc_avg2_var' to the
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
        mc_remove_set_flag : bool
            Remove the monte carlo data set, from which the CI and the
            variance are calculated.
        reduce_memory_usage : bool
            Use less memory but at the expense of longer computation time

        Returns
        -------

        """    
        # out contains the state
        out = xr.Dataset(coords={"x": self.x, "time": self.time, "trans_att": result["trans_att"]}).copy()
        out.coords["x"].attrs = dim_attrs["x"]
        out.coords["trans_att"].attrs = dim_attrs["trans_att"]
        out.coords["CI"] = conf_ints

        mcparams = self.monte_carlo_single_ended(
            result=result,
            st_var=st_var,
            ast_var=ast_var,
            conf_ints=None,
            mc_sample_size=mc_sample_size,
            da_random_state=da_random_state,
            mc_remove_set_flag=False,
            reduce_memory_usage=reduce_memory_usage,
        )
        mcparams["tmpf"] = result["tmpf"]

        if ci_avg_time_sel is not None:
            time_dim2 = "time" + "_avg"
            x_dim2 = "x"
            mcparams.coords[time_dim2] = (
                (time_dim2,),
                mcparams["time"].sel(**{"time": ci_avg_time_sel}).data,
            )
            mcparams["tmpf_avgsec"] = (
                ("x", time_dim2),
                mcparams["tmpf"].sel(**{"time": ci_avg_time_sel}).data,
            )
            mcparams["tmpf_mc_set"] = (
                ("mc", "x", time_dim2),
                mcparams["tmpf" + "_mc_set"].sel(**{"time": ci_avg_time_sel}).data,
            )

        elif ci_avg_time_isel is not None:
            time_dim2 = "time" + "_avg"
            x_dim2 = "x"
            mcparams.coords[time_dim2] = (
                (time_dim2,),
                mcparams["time"].isel(**{"time": ci_avg_time_isel}).data,
            )
            mcparams["tmpf_avgsec"] = (
                ("x", time_dim2),
                mcparams["tmpf"].isel(**{"time": ci_avg_time_isel}).data,
            )
            mcparams["tmpf_mc_set"] = (
                ("mc", "x", time_dim2),
                mcparams["tmpf" + "_mc_set"].isel(**{"time": ci_avg_time_isel}).data,
            )

        elif ci_avg_x_sel is not None:
            time_dim2 = "time"
            x_dim2 = "x_avg"
            mcparams.coords[x_dim2] = ((x_dim2,), mcparams.x.sel(x=ci_avg_x_sel).data)
            mcparams["tmpf_avgsec"] = (
                (x_dim2, "time"),
                mcparams["tmpf"].sel(x=ci_avg_x_sel).data,
            )
            mcparams["tmpf_mc_set"] = (
                ("mc", x_dim2, "time"),
                mcparams["tmpf_mc_set"].sel(x=ci_avg_x_sel).data,
            )

        elif ci_avg_x_isel is not None:
            time_dim2 = "time"
            x_dim2 = "x_avg"
            mcparams.coords[x_dim2] = ((x_dim2,), mcparams.x.isel(x=ci_avg_x_isel).data)
            mcparams["tmpf_avgsec"] = (
                (x_dim2, time_dim2),
                mcparams["tmpf"].isel(x=ci_avg_x_isel).data,
            )
            mcparams["tmpf_mc_set"] = (
                ("mc", x_dim2, time_dim2),
                mcparams["tmpf_mc_set"].isel(x=ci_avg_x_isel).data,
            )
        else:
            mcparams["tmpf_avgsec"] = mcparams["tmpf"]
            x_dim2 = "x"
            time_dim2 = "time"

        # subtract the mean temperature
        q = mcparams["tmpf_mc_set"] - mcparams["tmpf_avgsec"]
        out["tmpf_mc" + "_avgsec_var"] = q.var(dim="mc", ddof=1)

        if ci_avg_x_flag1:
            # unweighted mean
            out["tmpf_avgx1"] = mcparams["tmpf" + "_avgsec"].mean(dim=x_dim2)

            q = mcparams["tmpf_mc_set"] - mcparams["tmpf_avgsec"]
            qvar = q.var(dim=["mc", x_dim2], ddof=1)
            out["tmpf_mc_avgx1_var"] = qvar

            if conf_ints:
                new_chunks = (len(conf_ints), mcparams["tmpf_mc_set"].chunks[2])
                avg_axis = mcparams["tmpf_mc_set"].get_axis_num(["mc", x_dim2])
                q = mcparams["tmpf_mc_set"].data.map_blocks(
                    lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
                    chunks=new_chunks,  #
                    drop_axis=avg_axis,
                    # avg dimensions are dropped from input arr
                    new_axis=0,
                )  # The new CI dim is added as firsaxis

                out["tmpf_mc_avgx1"] = (("CI", time_dim2), q)

        if ci_avg_x_flag2:
            q = mcparams["tmpf_mc_set"] - mcparams["tmpf_avgsec"]

            qvar = q.var(dim=["mc"], ddof=1)

            # Inverse-variance weighting
            avg_x_var = 1 / (1 / qvar).sum(dim=x_dim2)

            out["tmpf_mc_avgx2_var"] = avg_x_var

            mcparams["tmpf" + "_mc_avgx2_set"] = (mcparams["tmpf_mc_set"] / qvar).sum(
                dim=x_dim2
            ) * avg_x_var
            out["tmpf" + "_avgx2"] = mcparams["tmpf" + "_mc_avgx2_set"].mean(dim="mc")

            if conf_ints:
                new_chunks = (len(conf_ints), mcparams["tmpf_mc_set"].chunks[2])
                avg_axis_avgx = mcparams["tmpf_mc_set"].get_axis_num("mc")

                qq = mcparams["tmpf_mc_avgx2_set"].data.map_blocks(
                    lambda x: np.percentile(x, q=conf_ints, axis=avg_axis_avgx),
                    chunks=new_chunks,  #
                    drop_axis=avg_axis_avgx,
                    # avg dimensions are dropped from input arr
                    new_axis=0,
                    dtype=float,
                )  # The new CI dimension is added as
                # firsaxis
                out["tmpf_mc_avgx2"] = (("CI", time_dim2), qq)

        if ci_avg_time_flag1 is not None:
            # unweighted mean
            out["tmpf_avg1"] = mcparams["tmpf_avgsec"].mean(dim=time_dim2)

            q = mcparams["tmpf_mc_set"] - mcparams["tmpf_avgsec"]
            qvar = q.var(dim=["mc", time_dim2], ddof=1)
            out["tmpf_mc_avg1_var"] = qvar

            if conf_ints:
                new_chunks = (len(conf_ints), mcparams["tmpf_mc_set"].chunks[1])
                avg_axis = mcparams["tmpf_mc_set"].get_axis_num(["mc", time_dim2])
                q = mcparams["tmpf_mc_set"].data.map_blocks(
                    lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
                    chunks=new_chunks,  #
                    drop_axis=avg_axis,
                    # avg dimensions are dropped from input arr
                    new_axis=0,
                )  # The new CI dim is added as firsaxis

                out["tmpf_mc_avg1"] = (("CI", x_dim2), q)

        if ci_avg_time_flag2:
            q = mcparams["tmpf_mc_set"] - mcparams["tmpf_avgsec"]

            qvar = q.var(dim=["mc"], ddof=1)

            # Inverse-variance weighting
            avg_time_var = 1 / (1 / qvar).sum(dim=time_dim2)

            out["tmpf_mc_avg2_var"] = avg_time_var

            mcparams["tmpf" + "_mc_avg2_set"] = (mcparams["tmpf_mc_set"] / qvar).sum(
                dim=time_dim2
            ) * avg_time_var
            out["tmpf_avg2"] = mcparams["tmpf" + "_mc_avg2_set"].mean(dim="mc")

            if conf_ints:
                new_chunks = (len(conf_ints), mcparams["tmpf_mc_set"].chunks[1])
                avg_axis_avg2 = mcparams["tmpf_mc_set"].get_axis_num("mc")

                qq = mcparams["tmpf_mc_avg2_set"].data.map_blocks(
                    lambda x: np.percentile(x, q=conf_ints, axis=avg_axis_avg2),
                    chunks=new_chunks,  #
                    drop_axis=avg_axis_avg2,
                    # avg dimensions are dropped from input arr
                    new_axis=0,
                    dtype=float,
                )  # The new CI dimension is added as
                # firsaxis
                out["tmpf_mc_avg2"] = (("CI", x_dim2), qq)

        # Clean up the garbage. All arrays with a Monte Carlo dimension.
        if mc_remove_set_flag:
            remove_mc_set = [
                "r_st",
                "r_ast",
                "gamma_mc",
                "dalpha_mc",
                "c_mc",
                "x_avg",
                "time_avg",
                "mc",
                "ta_mc_arr",
            ]
            remove_mc_set.append("tmpf_avgsec")
            remove_mc_set.append("tmpf_mc_set")
            remove_mc_set.append("tmpf_mc_avg2_set")
            remove_mc_set.append("tmpf_mc_avgx2_set")
            remove_mc_set.append("tmpf_mc_avgsec_var")

            for k in remove_mc_set:
                if k in out:
                    del out[k]

        return out

    def average_monte_carlo_double_ended(
        self,
        result,
        st_var,
        ast_var,
        rst_var,
        rast_var,
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
        mc_remove_set_flag=True,
        reduce_memory_usage=False,
        **kwargs,
    ):
        """
        Average temperatures from double-ended setups.

        Four types of averaging are implemented. Please see Example Notebook 16.

        Parameters
        ----------
        result : xr.Dataset
            The result from the `calibrate_double_ended()` method.
        st_var, ast_var, rst_var, rast_var : float, callable, array-like
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
            Adds "tmpw" + '_avg1' and "tmpw" + '_mc_avg1_var' to the
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
            Adds "tmpw" + '_avg2' and "tmpw" + '_mc_avg2_var' to the
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
            Adds "tmpw" + '_avgx1' and "tmpw" + '_mc_avgx1_var' to the
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
            Adds "tmpw" + '_avg2' and "tmpw" + '_mc_avg2_var' to the
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
        mc_remove_set_flag : bool
            Remove the monte carlo data set, from which the CI and the
            variance are calculated.
        reduce_memory_usage : bool
            Use less memory but at the expense of longer computation time

        Returns
        -------

        """

        # def create_da_ta2(no, i_splice, direction="fw", chunks=None):
        #     """create mask array mc, o, nt"""

        #     if direction == "fw":
        #         arr = da.concatenate(
        #             (
        #                 da.zeros((1, i_splice, 1), chunks=(1, i_splice, 1), dtype=bool),
        #                 da.ones(
        #                     (1, no - i_splice, 1),
        #                     chunks=(1, no - i_splice, 1),
        #                     dtype=bool,
        #                 ),
        #             ),
        #             axis=1,
        #         ).rechunk((1, chunks[1], 1))
        #     else:
        #         arr = da.concatenate(
        #             (
        #                 da.ones((1, i_splice, 1), chunks=(1, i_splice, 1), dtype=bool),
        #                 da.zeros(
        #                     (1, no - i_splice, 1),
        #                     chunks=(1, no - i_splice, 1),
        #                     dtype=bool,
        #                 ),
        #             ),
        #             axis=1,
        #         ).rechunk((1, chunks[1], 1))
        #     return arr

        out = xr.Dataset(coords={"x": self.x, "time": self.time, "trans_att": result["trans_att"]}).copy()
        out.coords["x"].attrs = dim_attrs["x"]
        out.coords["trans_att"].attrs = dim_attrs["trans_att"]
        out.coords["CI"] = conf_ints

        if (ci_avg_x_flag1 or ci_avg_x_flag2) and (
            ci_avg_time_flag1 or ci_avg_time_flag2
        ):
            raise NotImplementedError(
                "Incompatible flags. Can not pick " "the right chunks"
            )

        elif not (
            ci_avg_x_flag1 or ci_avg_x_flag2 or ci_avg_time_flag1 or ci_avg_time_flag2
        ):
            raise NotImplementedError("Pick one of the averaging options")

        else:
            pass

        mcparams = self.monte_carlo_double_ended(
            result=result,
            st_var=st_var,
            ast_var=ast_var,
            rst_var=rst_var,
            rast_var=rast_var,
            conf_ints=None,
            mc_sample_size=mc_sample_size,
            da_random_state=da_random_state,
            mc_remove_set_flag=False,
            reduce_memory_usage=reduce_memory_usage,
            **kwargs,
        )

        for label in ["tmpf", "tmpb"]:
            if ci_avg_time_sel is not None:
                time_dim2 = "time" + "_avg"
                x_dim2 = "x"
                mcparams.coords[time_dim2] = (
                    (time_dim2,),
                    mcparams["time"].sel(**{"time": ci_avg_time_sel}).data,
                )
                mcparams[label + "_avgsec"] = (
                    ("x", time_dim2),
                    result[label].sel(**{"time": ci_avg_time_sel}).data,
                )
                mcparams[label + "_mc_set"] = (
                    ("mc", "x", time_dim2),
                    mcparams[label + "_mc_set"].sel(**{"time": ci_avg_time_sel}).data,
                )

            elif ci_avg_time_isel is not None:
                time_dim2 = "time" + "_avg"
                x_dim2 = "x"
                mcparams.coords[time_dim2] = (
                    (time_dim2,),
                    mcparams["time"].isel(**{"time": ci_avg_time_isel}).data,
                )
                mcparams[label + "_avgsec"] = (
                    ("x", time_dim2),
                    result[label].isel(**{"time": ci_avg_time_isel}).data,
                )
                mcparams[label + "_mc_set"] = (
                    ("mc", "x", time_dim2),
                    mcparams[label + "_mc_set"].isel(**{"time": ci_avg_time_isel}).data,
                )

            elif ci_avg_x_sel is not None:
                time_dim2 = "time"
                x_dim2 = "x_avg"
                mcparams.coords[x_dim2] = (
                    (x_dim2,),
                    mcparams.x.sel(x=ci_avg_x_sel).data,
                )
                mcparams[label + "_avgsec"] = (
                    (x_dim2, "time"),
                    result[label].sel(x=ci_avg_x_sel).data,
                )
                mcparams[label + "_mc_set"] = (
                    ("mc", x_dim2, "time"),
                    mcparams[label + "_mc_set"].sel(x=ci_avg_x_sel).data,
                )

            elif ci_avg_x_isel is not None:
                time_dim2 = "time"
                x_dim2 = "x_avg"
                mcparams.coords[x_dim2] = (
                    (x_dim2,),
                    mcparams.x.isel(x=ci_avg_x_isel).data,
                )
                mcparams[label + "_avgsec"] = (
                    (x_dim2, time_dim2),
                    result[label].isel(x=ci_avg_x_isel).data,
                )
                mcparams[label + "_mc_set"] = (
                    ("mc", x_dim2, time_dim2),
                    mcparams[label + "_mc_set"].isel(x=ci_avg_x_isel).data,
                )
            else:
                mcparams[label + "_avgsec"] = result[label]
                x_dim2 = "x"
                time_dim2 = "time"

            memchunk = mcparams[label + "_mc_set"].chunks

            # subtract the mean temperature
            q = mcparams[label + "_mc_set"] - mcparams[label + "_avgsec"]
            out[label + "_mc" + "_avgsec_var"] = q.var(dim="mc", ddof=1)

            if ci_avg_x_flag1:
                # unweighted mean
                out[label + "_avgx1"] = mcparams[label + "_avgsec"].mean(dim=x_dim2)

                q = mcparams[label + "_mc_set"] - mcparams[label + "_avgsec"]
                qvar = q.var(dim=["mc", x_dim2], ddof=1)
                out[label + "_mc_avgx1_var"] = qvar

                if conf_ints:
                    new_chunks = (len(conf_ints), mcparams[label + "_mc_set"].chunks[2])
                    avg_axis = mcparams[label + "_mc_set"].get_axis_num(["mc", x_dim2])
                    q = mcparams[label + "_mc_set"].data.map_blocks(
                        lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
                        chunks=new_chunks,  #
                        drop_axis=avg_axis,
                        # avg dimensions are dropped from input arr
                        new_axis=0,
                    )  # The new CI dim is added as firsaxis

                    out[label + "_mc_avgx1"] = (("CI", time_dim2), q)

            if ci_avg_x_flag2:
                q = mcparams[label + "_mc_set"] - mcparams[label + "_avgsec"]

                qvar = q.var(dim=["mc"], ddof=1)

                # Inverse-variance weighting
                avg_x_var = 1 / (1 / qvar).sum(dim=x_dim2)

                out[label + "_mc_avgx2_var"] = avg_x_var

                mcparams[label + "_mc_avgx2_set"] = (
                    mcparams[label + "_mc_set"] / qvar
                ).sum(dim=x_dim2) * avg_x_var
                out[label + "_avgx2"] = mcparams[label + "_mc_avgx2_set"].mean(dim="mc")

                if conf_ints:
                    new_chunks = (len(conf_ints), mcparams[label + "_mc_set"].chunks[2])
                    avg_axis_avgx = mcparams[label + "_mc_set"].get_axis_num("mc")

                    qq = mcparams[label + "_mc_avgx2_set"].data.map_blocks(
                        lambda x: np.percentile(x, q=conf_ints, axis=avg_axis_avgx),
                        chunks=new_chunks,  #
                        drop_axis=avg_axis_avgx,
                        # avg dimensions are dropped from input arr
                        new_axis=0,
                        dtype=float,
                    )  # The new CI dimension is added as
                    # firsaxis
                    out[label + "_mc_avgx2"] = (("CI", time_dim2), qq)

            if ci_avg_time_flag1 is not None:
                # unweighted mean
                out[label + "_avg1"] = mcparams[label + "_avgsec"].mean(dim=time_dim2)

                q = mcparams[label + "_mc_set"] - mcparams[label + "_avgsec"]
                qvar = q.var(dim=["mc", time_dim2], ddof=1)
                out[label + "_mc_avg1_var"] = qvar

                if conf_ints:
                    new_chunks = (len(conf_ints), mcparams[label + "_mc_set"].chunks[1])
                    avg_axis = mcparams[label + "_mc_set"].get_axis_num(
                        ["mc", time_dim2]
                    )
                    q = mcparams[label + "_mc_set"].data.map_blocks(
                        lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
                        chunks=new_chunks,  #
                        drop_axis=avg_axis,
                        # avg dimensions are dropped from input arr
                        new_axis=0,
                    )  # The new CI dim is added as firsaxis

                    out[label + "_mc_avg1"] = (("CI", x_dim2), q)

            if ci_avg_time_flag2:
                q = mcparams[label + "_mc_set"] - mcparams[label + "_avgsec"]

                qvar = q.var(dim=["mc"], ddof=1)

                # Inverse-variance weighting
                avg_time_var = 1 / (1 / qvar).sum(dim=time_dim2)

                out[label + "_mc_avg2_var"] = avg_time_var

                mcparams[label + "_mc_avg2_set"] = (
                    mcparams[label + "_mc_set"] / qvar
                ).sum(dim=time_dim2) * avg_time_var
                out[label + "_avg2"] = mcparams[label + "_mc_avg2_set"].mean(dim="mc")

                if conf_ints:
                    new_chunks = (len(conf_ints), mcparams[label + "_mc_set"].chunks[1])
                    avg_axis_avg2 = mcparams[label + "_mc_set"].get_axis_num("mc")

                    qq = mcparams[label + "_mc_avg2_set"].data.map_blocks(
                        lambda x: np.percentile(x, q=conf_ints, axis=avg_axis_avg2),
                        chunks=new_chunks,  #
                        drop_axis=avg_axis_avg2,
                        # avg dimensions are dropped from input arr
                        new_axis=0,
                        dtype=float,
                    )  # The new CI dimension is added as
                    # firsaxis
                    out[label + "_mc_avg2"] = (("CI", x_dim2), qq)

        # Weighted mean of the forward and backward
        tmpw_var = 1 / (
            1 / out["tmpf_mc" + "_avgsec_var"] + 1 / out["tmpb_mc" + "_avgsec_var"]
        )

        q = (
            mcparams["tmpf_mc_set"] / out["tmpf_mc" + "_avgsec_var"]
            + mcparams["tmpb_mc_set"] / out["tmpb_mc" + "_avgsec_var"]
        ) * tmpw_var

        mcparams["tmpw" + "_mc_set"] = q  #

        # out["tmpw"] = out["tmpw" + '_mc_set'].mean(dim='mc')
        out["tmpw" + "_avgsec"] = (
            mcparams["tmpf_avgsec"] / out["tmpf_mc" + "_avgsec_var"]
            + mcparams["tmpb_avgsec"] / out["tmpb_mc" + "_avgsec_var"]
        ) * tmpw_var

        q = mcparams["tmpw" + "_mc_set"] - out["tmpw_avgsec"]
        out["tmpw" + "_mc" + "_avgsec_var"] = q.var(dim="mc", ddof=1)

        if ci_avg_time_flag1:
            out["tmpw" + "_avg1"] = out["tmpw" + "_avgsec"].mean(dim=time_dim2)

            out["tmpw" + "_mc_avg1_var"] = mcparams["tmpw" + "_mc_set"].var(
                dim=["mc", time_dim2]
            )

            if conf_ints:
                new_chunks_weighted = ((len(conf_ints),),) + (memchunk[1],)
                avg_axis = mcparams["tmpw" + "_mc_set"].get_axis_num(["mc", time_dim2])
                q2 = mcparams["tmpw" + "_mc_set"].data.map_blocks(
                    lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
                    chunks=new_chunks_weighted,
                    # Explicitly define output chunks
                    drop_axis=avg_axis,  # avg dimensions are dropped
                    new_axis=0,
                    dtype=float,
                )  # The new CI dimension is added as
                # first axis
                out["tmpw" + "_mc_avg1"] = (("CI", x_dim2), q2)

        if ci_avg_time_flag2:
            tmpw_var_avg2 = 1 / (
                1 / out["tmpf_mc_avg2_var"] + 1 / out["tmpb_mc_avg2_var"]
            )

            q = (
                mcparams["tmpf_mc_avg2_set"] / out["tmpf_mc_avg2_var"]
                + mcparams["tmpb_mc_avg2_set"] / out["tmpb_mc_avg2_var"]
            ) * tmpw_var_avg2

            mcparams["tmpw" + "_mc_avg2_set"] = q  #

            out["tmpw" + "_avg2"] = (
                out["tmpf_avg2"] / out["tmpf_mc_avg2_var"]
                + out["tmpb_avg2"] / out["tmpb_mc_avg2_var"]
            ) * tmpw_var_avg2

            out["tmpw" + "_mc_avg2_var"] = tmpw_var_avg2

            if conf_ints:
                # We first need to know the x-dim-chunk-size
                new_chunks_weighted = ((len(conf_ints),),) + (memchunk[1],)
                avg_axis_avg2 = mcparams["tmpw" + "_mc_avg2_set"].get_axis_num("mc")
                q2 = mcparams["tmpw" + "_mc_avg2_set"].data.map_blocks(
                    lambda x: np.percentile(x, q=conf_ints, axis=avg_axis_avg2),
                    chunks=new_chunks_weighted,
                    # Explicitly define output chunks
                    drop_axis=avg_axis_avg2,  # avg dimensions are dropped
                    new_axis=0,
                    dtype=float,
                )  # The new CI dimension is added as firstax
                out["tmpw" + "_mc_avg2"] = (("CI", x_dim2), q2)

        if ci_avg_x_flag1:
            out["tmpw" + "_avgx1"] = out["tmpw" + "_avgsec"].mean(dim=x_dim2)

            out["tmpw" + "_mc_avgx1_var"] = mcparams["tmpw" + "_mc_set"].var(dim=x_dim2)

            if conf_ints:
                new_chunks_weighted = ((len(conf_ints),),) + (memchunk[2],)
                avg_axis = mcparams["tmpw" + "_mc_set"].get_axis_num(["mc", x_dim2])
                q2 = mcparams["tmpw" + "_mc_set"].data.map_blocks(
                    lambda x: np.percentile(x, q=conf_ints, axis=avg_axis),
                    chunks=new_chunks_weighted,
                    # Explicitly define output chunks
                    drop_axis=avg_axis,  # avg dimensions are dropped
                    new_axis=0,
                    dtype=float,
                )  # The new CI dimension is added as
                # first axis
                out["tmpw" + "_mc_avgx1"] = (("CI", time_dim2), q2)

        if ci_avg_x_flag2:
            tmpw_var_avgx2 = 1 / (
                1 / out["tmpf_mc_avgx2_var"] + 1 / out["tmpb_mc_avgx2_var"]
            )

            q = (
                mcparams["tmpf_mc_avgx2_set"] / out["tmpf_mc_avgx2_var"]
                + mcparams["tmpb_mc_avgx2_set"] / out["tmpb_mc_avgx2_var"]
            ) * tmpw_var_avgx2

            mcparams["tmpw" + "_mc_avgx2_set"] = q  #

            out["tmpw" + "_avgx2"] = (
                out["tmpf_avgx2"] / out["tmpf_mc_avgx2_var"]
                + out["tmpb_avgx2"] / out["tmpb_mc_avgx2_var"]
            ) * tmpw_var_avgx2

            out["tmpw" + "_mc_avgx2_var"] = tmpw_var_avgx2

            if conf_ints:
                # We first need to know the x-dim-chunk-size
                new_chunks_weighted = ((len(conf_ints),),) + (memchunk[2],)
                avg_axis_avgx2 = mcparams["tmpw" + "_mc_avgx2_set"].get_axis_num("mc")
                q2 = mcparams["tmpw" + "_mc_avgx2_set"].data.map_blocks(
                    lambda x: np.percentile(x, q=conf_ints, axis=avg_axis_avgx2),
                    chunks=new_chunks_weighted,
                    # Explicitly define output chunks
                    drop_axis=avg_axis_avgx2,  # avg dimensions are dropped
                    new_axis=0,
                    dtype=float,
                )  # The new CI dimension is added as firstax
                out["tmpw" + "_mc_avgx2"] = (("CI", time_dim2), q2)

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
                "x_avg",
                "time_avg",
                "mc",
            ]

            for i in ["tmpf", "tmpb", "tmpw"]:
                remove_mc_set.append(i + "_avgsec")
                remove_mc_set.append(i + "_mc_set")
                remove_mc_set.append(i + "_mc_avg2_set")
                remove_mc_set.append(i + "_mc_avgx2_set")
                remove_mc_set.append(i + "_mc_avgsec_var")

            if "trans_att" in mcparams and mcparams.trans_att.size:
                remove_mc_set.append('talpha"_fw_mc')
                remove_mc_set.append('talpha"_bw_mc')

            for k in remove_mc_set:
                if k in out:
                    print(f"Removed from results: {k}")
                    del out[k]

        return out

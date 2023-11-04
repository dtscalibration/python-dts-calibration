import dask.array as da
import numpy as np
import xarray as xr

from dtscalibration.calibration.section_utils import validate_no_overlapping_sections
from dtscalibration.calibration.section_utils import validate_sections_definition
from dtscalibration.datastore_utils import ufunc_per_section_helper
from dtscalibration.variance_helpers import check_allclose_acquisitiontime
from dtscalibration.variance_helpers import variance_stokes_constant_helper
from dtscalibration.variance_helpers import variance_stokes_exponential_helper
from dtscalibration.variance_helpers import variance_stokes_linear_helper


def variance_stokes_constant(st, sections, acquisitiontime, reshape_residuals=True):
    """Approximate the variance of the noise in Stokes intensity measurements
    with one value, suitable for small setups.

    * `variance_stokes_constant()` for small setups with small variations in\
    intensity. Variance of the Stokes measurements is assumed to be the same\
    along the entire fiber.

    * `variance_stokes_exponential()` for small setups with very few time\
    steps. Too many degrees of freedom results in an under estimation of the\
    noise variance. Almost never the case, but use when calibrating pre time\
    step.

    * `variance_stokes_linear()` for larger setups with more time steps.\
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
    st : DataArray
    sections : Dict[str, List[slice]]

    Returns:
    -------
    I_var : float
        Variance of the residuals between measured and best fit
    resid : array_like
        Residuals between measured and best fit

    Notes:
    -----
    * Because there are a large number of unknowns, spend time on\
    calculating an initial estimate. Can be turned off by setting to False.

    * It is often not needed to use measurements from all time steps. If\
    your variance estimate does not change when including measurements\
    additional time steps, you have included enough measurements.

    References:
    ----------
    .. [1] des Tombe, B., Schilperoort, B., & Bakker, M. (2020). Estimation
        of Temperature and Associated Uncertainty from Fiber-Optic Raman-
        Spectrum Distributed Temperature Sensing. Sensors, 20(8), 2235.
        https://doi.org/10.3390/s20082235

    Examples:
    --------
    - `Example notebook 4: Calculate variance Stokes intensity measurements\
    <https://github.com/\
    dtscalibration/python-dts-calibration/blob/main/examples/notebooks/\
    04Calculate_variance_Stokes.ipynb>`_

    TODO: Account for varying acquisition times
    """
    validate_sections_definition(sections=sections)
    validate_no_overlapping_sections(sections=sections)
    check_allclose_acquisitiontime(acquisitiontime=acquisitiontime)

    assert st.dims[0] == "x", "DataArray is transposed"

    # should maybe be per section. But then residuals
    # seem to be correlated between stretches. I don't know why.. BdT.
    data_dict = da.compute(
        ufunc_per_section_helper(sections=sections, dataarray=st, calc_per="stretch")
    )[0]

    var_I, resid = variance_stokes_constant_helper(data_dict)

    if not reshape_residuals:
        return var_I, resid

    else:
        ix_resid = ufunc_per_section_helper(
            sections=sections, x_coords=st.x, calc_per="all"
        )

        resid_sorted = np.full(shape=st.shape, fill_value=np.nan)
        resid_sorted[ix_resid, :] = resid
        resid_da = xr.DataArray(data=resid_sorted, coords=st.coords)

        return var_I, resid_da


def variance_stokes_exponential(
    st,
    sections,
    acquisitiontime,
    use_statsmodels=False,
    suppress_info=True,
    reshape_residuals=True,
):
    """Approximate the variance of the noise in Stokes intensity measurements
    with one value, suitable for small setups with measurements from only
    a few times.

    * `variance_stokes_constant()` for small setups with small variations in\
    intensity. Variance of the Stokes measurements is assumed to be the same\
    along the entire fiber.

    * `variance_stokes_exponential()` for small setups with very few time\
    steps. Too many degrees of freedom results in an under estimation of the\
    noise variance. Almost never the case, but use when calibrating pre time\
    step.

    * `variance_stokes_linear()` for larger setups with more time steps.\
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
    suppress_info : bool, optional
        Suppress print statements.
    use_statsmodels : bool, optional
        Use statsmodels to fit the exponential. If `False`, use scipy.
    reshape_residuals : bool, optional
        Reshape the residuals to the shape of the Stokes intensity
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

    Returns:
    -------
    I_var : float
        Variance of the residuals between measured and best fit
    resid : array_like
        Residuals between measured and best fit

    Notes:
    -----
    * Because there are a large number of unknowns, spend time on\
    calculating an initial estimate. Can be turned off by setting to False.

    * It is often not needed to use measurements from all time steps. If\
    your variance estimate does not change when including measurements from\
    more time steps, you have included enough measurements.

    References:
    ----------
    .. [1] des Tombe, B., Schilperoort, B., & Bakker, M. (2020). Estimation
        of Temperature and Associated Uncertainty from Fiber-Optic Raman-
        Spectrum Distributed Temperature Sensing. Sensors, 20(8), 2235.
        https://doi.org/10.3390/s20082235

    Examples:
    --------
    - `Example notebook 4: Calculate variance Stokes intensity measurements\
    <https://github.com/\
    dtscalibration/python-dts-calibration/blob/main/examples/notebooks/\
    04Calculate_variance_Stokes.ipynb>`_
    """
    validate_sections_definition(sections=sections)
    validate_no_overlapping_sections(sections=sections)
    check_allclose_acquisitiontime(acquisitiontime=acquisitiontime)

    assert st.dims[0] == "x", "Stokes are transposed"
    nt = st.coords["time"].size

    # number of reference points per section (spatial)
    len_stretch_list = []
    y_list = []  # intensities of stokes
    x_list = []  # length rel to start of section. for alpha

    for k, stretches in sections.items():
        for stretch in stretches:
            y_list.append(st.sel(x=stretch).data.T.reshape(-1))
            _x = st.coords["x"].sel(x=stretch).data.copy()
            _x -= _x[0]
            x_list.append(da.tile(_x, nt))
            len_stretch_list.append(_x.size)

    x = np.concatenate(x_list)
    y = np.concatenate(y_list)

    var_I, resid = variance_stokes_exponential_helper(
        nt, x, y, len_stretch_list, use_statsmodels, suppress_info
    )

    if not reshape_residuals:
        return var_I, resid

    else:
        # restructure the residuals, such that they can be plotted and
        # added to ds
        resid_res = []
        for leni, lenis, lenie in zip(
            len_stretch_list,
            nt * np.cumsum([0] + len_stretch_list[:-1]),
            nt * np.cumsum(len_stretch_list),
        ):
            try:
                resid_res.append(resid[lenis:lenie].reshape((leni, nt), order="F"))
            except:  # noqa: E722
                # Dask array does not support order
                resid_res.append(resid[lenis:lenie].T.reshape((nt, leni)).T)

        _resid = np.concatenate(resid_res)
        # _resid_x = self.ufunc_per_section(
        #     sections=sections, label="x", calc_per="all"
        # )
        _resid_x = ufunc_per_section_helper(
            sections=sections, dataarray=st.coords["x"], calc_per="all"
        )
        isort = np.argsort(_resid_x)
        resid_x = _resid_x[isort]  # get indices from ufunc directly
        resid = _resid[isort, :]

        ix_resid = np.array(
            [np.argmin(np.abs(ai - st.coords["x"].data)) for ai in resid_x]
        )

        resid_sorted = np.full(shape=st.shape, fill_value=np.nan)
        resid_sorted[ix_resid, :] = resid
        resid_da = xr.DataArray(data=resid_sorted, coords=st.coords)

        return var_I, resid_da


def variance_stokes_linear(
    st, sections, acquisitiontime, nbin=50, through_zero=False, plot_fit=False
):
    """Approximate the variance of the noise in Stokes intensity measurements
    with a linear function of the intensity, suitable for large setups.

    * `variance_stokes_constant()` for small setups with small variations in\
    intensity. Variance of the Stokes measurements is assumed to be the same\
    along the entire fiber.

    * `variance_stokes_exponential()` for small setups with very few time\
    steps. Too many degrees of freedom results in an under estimation of the\
    noise variance. Almost never the case, but use when calibrating pre time\
    step.

    * `variance_stokes_linear()` for larger setups with more time steps.\
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

    Notes:
    -----
    * Because there are a large number of unknowns, spend time on\
    calculating an initial estimate. Can be turned off by setting to False.

    * It is often not needed to use measurements from all time steps. If\
    your variance estimate does not change when including measurements \
    from more time steps, you have included enough measurements.

    References:
    ----------
    .. [1] des Tombe, B., Schilperoort, B., & Bakker, M. (2020). Estimation
        of Temperature and Associated Uncertainty from Fiber-Optic Raman-
        Spectrum Distributed Temperature Sensing. Sensors, 20(8), 2235.
        https://doi.org/10.3390/s20082235

    Examples:
    --------
    - `Example notebook 4: Calculate variance Stokes intensity \
    measurements <https://github.com/\
    dtscalibration/python-dts-calibration/blob/main/examples/notebooks/\
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
    validate_sections_definition(sections=sections)
    validate_no_overlapping_sections(sections=sections)
    check_allclose_acquisitiontime(acquisitiontime=acquisitiontime)

    assert st.dims[0] == "x", "Stokes are transposed"
    _, resid = variance_stokes_constant(
        sections=sections,
        st=st,
        acquisitiontime=acquisitiontime,
        reshape_residuals=False,
    )
    ix_sec = ufunc_per_section_helper(
        sections=sections, x_coords=st.coords["x"], calc_per="all"
    )

    st = st.isel(x=ix_sec).values.ravel()
    diff_st = resid.ravel()

    (
        slope,
        offset,
        st_sort_mean,
        st_sort_var,
        resid,
        var_fun,
    ) = variance_stokes_linear_helper(st, diff_st, nbin, through_zero)

    if plot_fit:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.scatter(st_sort_mean, st_sort_var, marker=".", c="black")
        plt.plot(
            [0.0, st_sort_mean[-1]],
            [var_fun(0.0), var_fun(st_sort_mean[-1])],
            c="white",
            lw=1.3,
        )
        plt.plot(
            [0.0, st_sort_mean[-1]],
            [var_fun(0.0), var_fun(st_sort_mean[-1])],
            c="black",
            lw=0.8,
        )
        plt.xlabel("intensity")
        plt.ylabel("intensity variance")

    return slope, offset, st_sort_mean, st_sort_var, resid, var_fun

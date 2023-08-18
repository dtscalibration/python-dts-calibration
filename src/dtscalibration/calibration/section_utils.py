
import numpy as np
import xarray as xr
import yaml

from dtscalibration.datastore_utils import ufunc_per_section_helper


def set_sections(ds: xr.Dataset, sections: dict[str, list[slice]]) -> xr.Dataset:
    sections_validated = None

    if sections is not None:
        sections_validated = validate_sections(ds, sections=sections)

    ds.attrs["_sections"] = yaml.dump(sections_validated)
    return ds


def validate_sections(ds: xr.Dataset, sections: dict[str, list[slice]]):
    assert isinstance(sections, dict)

    # be less restrictive for capitalized labels
    # find lower cases label
    labels = np.reshape([[s.lower(), s] for s in ds.data_vars.keys()], (-1,)).tolist()

    sections_fix = dict()
    for k, v in sections.items():
        if k.lower() in labels:
            i_lower_case = labels.index(k.lower())
            i_normal_case = i_lower_case + 1
            k_normal_case = labels[i_normal_case]
            sections_fix[k_normal_case] = v
        else:
            assert k in ds.data_vars, (
                "The keys of the "
                "sections-dictionary should "
                "refer to a valid timeserie "
                "already stored in "
                "ds.data_vars "
            )

    sections_fix_slice_fixed = dict()

    for k, v in sections_fix.items():
        assert isinstance(v, (list, tuple)), (
            "The values of the sections-dictionary " "should be lists of slice objects."
        )

        for vi in v:
            assert isinstance(vi, slice), (
                "The values of the sections-dictionary should "
                "be lists of slice objects."
            )

            assert ds.x.sel(x=vi).size > 0, (
                f"Better define the {k} section. You tried {vi}, "
                "which is not within the x-dimension"
            )

        # sorted stretches
        stretch_unsort = [slice(float(vi.start), float(vi.stop)) for vi in v]
        stretch_start = [i.start for i in stretch_unsort]
        stretch_i_sorted = np.argsort(stretch_start)
        sections_fix_slice_fixed[k] = [stretch_unsort[i] for i in stretch_i_sorted]

    # Prevent overlapping slices
    ix_sec = ufunc_per_section(
        ds, sections=sections_fix_slice_fixed, x_indices=True, calc_per="all"
    )
    assert np.unique(ix_sec).size == ix_sec.size, "The sections are overlapping"

    return sections_fix_slice_fixed


def ufunc_per_section(
    ds: xr.Dataset,
    sections,
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

    >>> tmpf_var = d.ufunc_per_section(sections, 
    >>>     func='var',
    >>>     calc_per='all',
    >>>     label='tmpf',
    >>>     temp_err=True)

    2. Calculate the variance of the residuals in the along PER\
    reference section wrt the temperature of the water baths

    >>> tmpf_var = d.ufunc_per_section(sections, 
    >>>     func='var',
    >>>     calc_per='stretch',
    >>>     label='tmpf',
    >>>     temp_err=True)

    3. Calculate the variance of the residuals in the along PER\
    water bath wrt the temperature of the water baths

    >>> tmpf_var = d.ufunc_per_section(sections, 
    >>>     func='var',
    >>>     calc_per='section',
    >>>     label='tmpf',
    >>>     temp_err=True)

    4. Obtain the coordinates of the measurements per section

    >>> locs = d.ufunc_per_section(sections, 
    >>>     func=None,
    >>>     label='x',
    >>>     temp_err=False,
    >>>     ref_temp_broadcasted=False,
    >>>     calc_per='stretch')

    5. Number of observations per stretch

    >>> nlocs = d.ufunc_per_section(sections, 
    >>>     func=len,
    >>>     label='x',
    >>>     temp_err=False,
    >>>     ref_temp_broadcasted=False,
    >>>     calc_per='stretch')

    6. broadcast the temperature of the reference sections to\
    stretch/section/all dimensions. The value of the reference\
    temperature (a timeseries) is broadcasted to the shape of self[\
    label]. The self[label] is not used for anything else.

    >>> temp_ref = d.ufunc_per_section(sections, 
    >>>     label='st',
    >>>     ref_temp_broadcasted=True,
    >>>     calc_per='all')

    7. x-coordinate index

    >>> ix_loc = d.ufunc_per_section(sections, x_indices=True)


    Note
    ----
    If `self[label]` or `self[subtract_from_label]` is a Dask array, a Dask
    array is returned else a numpy array is returned
    """
    if label is None:
        dataarray = None
    else:
        dataarray = ds[label]

    if x_indices:
        x_coords = ds.x
        reference_dataset = None
    else:
        x_coords = None
        reference_dataset = {k: ds[k] for k in sections}

    return ufunc_per_section_helper(
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

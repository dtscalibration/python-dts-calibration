"""Common utilities for reading input files."""
from contextlib import contextmanager

import pandas as pd

# Returns a dictionary with the attributes to the dimensions.
#  The keys refer to the naming used in the raw files.
# TODO: attrs for st_var and use it in parse_st_var function
_dim_attrs = {
    ("x", "distance"): dict(
        name="distance",
        description="Length along fiber",
        long_description="Starting at connector of forward channel",
        units="m",
    ),
    ("tmp", "temperature"): dict(
        name="tmp", description="Temperature calibrated by device", units=r"$^\circ$C"
    ),
    ("st",): dict(name="st", description="Stokes intensity", units="-"),
    ("ast",): dict(name="ast", description="anti-Stokes intensity", units="-"),
    ("rst",): dict(name="rst", description="reverse Stokes intensity", units="-"),
    ("rast",): dict(
        name="rast", description="reverse anti-Stokes intensity", units="-"
    ),
    ("tmpf",): dict(
        name="tmpf",
        description="Temperature estimated using the Stokes and anti-Stokes from the Forward channel",
        units=r"$^\circ$C",
    ),
    ("tmpb",): dict(
        name="tmpb",
        description="Temperature estimated using the Stokes and anti-Stokes from the Backward channel",
        units=r"$^\circ$C",
    ),
    ("tmpw",): dict(
        name="tmpw",
        description="Temperature estimated using the Stokes and anti-Stokes from both the Forward and Backward channel.",
        units=r"$^\circ$C",
    ),
    ("tmpf_var",): dict(
        name="tmpf_var",
        description="Uncertainty variance in tmpf estimated with linear-error propagation",
        units=r"$^\circ$C$^2$",
    ),
    ("tmpb_var",): dict(
        name="tmpb_var",
        description="Uncertainty variance in tmpb estimated with linear-error propagation",
        units=r"$^\circ$C$^2$",
    ),
    ("tmpw_var",): dict(
        name="tmpw_var",
        description="Uncertainty variance in tmpw estimated with linear-error propagation",
        units=r"$^\circ$C$^2$",
    ),
    ("tmpw_var_lower",): dict(
        name="tmpw_var_lower",
        description="Lower bound of uncertainty variance in tmpw estimated with linear-error propagation. "
        "Excludes the parameter uncertainties.",
        units=r"$^\circ$C$^2$",
    ),
    ("tmpw_var_approx",): dict(
        name="tmpw_var_upper",
        description="Upper bound of uncertainty variance in tmpw estimated with linear-error propagation. "
        "Excludes the correlation between tmpf and tmpb caused by alpha.",
        units=r"$^\circ$C$^2$",
    ),
    ("tmpf_mc_var",): dict(
        name="tmpf_mc_var",
        description="Uncertainty variance in tmpf estimated with Monte Carlo",
        units=r"$^\circ$C$^2$",
    ),
    ("tmpb_mc_var",): dict(
        name="tmpb_mc_var",
        description="Uncertainty variance in tmpb estimated with Monte Carlo",
        units=r"$^\circ$C$^2$",
    ),
    ("tmpw_mc_var",): dict(
        name="tmpw_mc_var",
        description="Uncertainty variance in tmpw estimated with Monte Carlo",
        units=r"$^\circ$C$^2$",
    ),
    ("acquisitionTime",): dict(
        name="acquisitionTime",
        description="Measurement duration of forward channel",
        long_description="Actual measurement duration of forward " "channel",
        units="seconds",
    ),
    ("userAcquisitionTimeFW",): dict(
        name="userAcquisitionTimeFW",
        description="Measurement duration of forward channel",
        long_description="Desired measurement duration of forward " "channel",
        units="seconds",
    ),
    ("userAcquisitionTimeBW",): dict(
        name="userAcquisitionTimeBW",
        description="Measurement duration of backward channel",
        long_description="Desired measurement duration of backward " "channel",
        units="seconds",
    ),
    ("trans_att",): dict(
        name="Locations introducing transient directional differential " "attenuation",
        description="Locations along the x-dimension introducing transient "
        "directional differential attenuation",
        long_description="Connectors introduce additional differential "
        "attenuation that is different for the forward "
        "and backward direction, and varies over time.",
        units="m",
    ),
}

# Because variations in the names exist between the different file formats. The
#   tuple as key contains the possible keys, which is expanded below.
dim_attrs = {k: v for kl, v in _dim_attrs.items() for k in kl}


@contextmanager
def open_file(path, **kwargs):
    if isinstance(path, tuple):
        # print('\nabout to open zipfile', path[0], '. from', path[1])
        # assert isinstance(path[1], zip)
        the_file = path[1].open(path[0], **kwargs)

    else:
        the_file = open(path, **kwargs)

    yield the_file
    the_file.close()


def get_xml_namespace(element):
    """Parameters
    ----------
    element

    Returns:
    -------

    """
    import re

    m = re.match("\\{.*\\}", element.tag)
    return m.group(0) if m else ""


def coords_time(
    maxTimeIndex,
    timezone_input_files=None,
    timezone_netcdf="UTC",
    dtFW=None,
    dtBW=None,
    double_ended_flag=False,
):
    """Prepares the time coordinates for the construction of DataStore
    instances with metadata.

    Parameters
    ----------
    maxTimeIndex : array-like (1-dimensional)
        Is an array with 'datetime64[ns]' timestamps of the end of the
        forward channel. If single ended this is the end of the measurement.
        If double ended this is halfway the double ended measurement.
    timezone_input_files : string, pytz.timezone, dateutil.tz.tzfile or None
        A string of a timezone that is understood by pandas of maxTimeIndex.
        If None, it is assumed that the input files are already timezone aware
    timezone_netcdf : string, pytz.timezone, dateutil.tz.tzfile or None
        A string of a timezone that is understood by pandas to write the
        netCDF to. Using UTC as default, according to CF conventions.
    dtFW : array-like (1-dimensional) of float
        The acquisition time of the Forward channel in seconds
    dtBW : array-like (1-dimensional) of float
        The acquisition time of the Backward channel in seconds
    double_ended_flag : bool
        A flag whether the measurement is double ended

    Returns:
    -------

    """
    time_attrs = {
        "time": {
            "description": "time halfway the measurement",
            "timezone": str(timezone_netcdf),
        },
        "timestart": {
            "description": "time start of the measurement",
            "timezone": str(timezone_netcdf),
        },
        "timeend": {
            "description": "time end of the measurement",
            "timezone": str(timezone_netcdf),
        },
        "timeFW": {
            "description": "time halfway the forward channel measurement",
            "timezone": str(timezone_netcdf),
        },
        "timeFWstart": {
            "description": "time start of the forward channel measurement",
            "timezone": str(timezone_netcdf),
        },
        "timeFWend": {
            "description": "time end of the forward channel measurement",
            "timezone": str(timezone_netcdf),
        },
        "timeBW": {
            "description": "time halfway the backward channel measurement",
            "timezone": str(timezone_netcdf),
        },
        "timeBWstart": {
            "description": "time start of the backward channel measurement",
            "timezone": str(timezone_netcdf),
        },
        "timeBWend": {
            "description": "time end of the backward channel measurement",
            "timezone": str(timezone_netcdf),
        },
    }

    if not double_ended_flag:
        # single ended measurement
        dt1 = dtFW.astype("timedelta64[s]")

        # start of the forward measurement
        index_time_FWstart = maxTimeIndex - dt1

        # end of the forward measurement
        index_time_FWend = maxTimeIndex

        # center of forward measurement
        index_time_FWmean = maxTimeIndex - dt1 / 2

        coords_zip = [
            ("timestart", index_time_FWstart),
            ("timeend", index_time_FWend),
            ("time", index_time_FWmean),
        ]

    else:
        # double ended measurement
        dt1 = dtFW.astype("timedelta64[s]")
        dt2 = dtBW.astype("timedelta64[s]")

        # start of the forward measurement
        index_time_FWstart = maxTimeIndex - dt1

        # end of the forward measurement
        index_time_FWend = maxTimeIndex

        # center of forward measurement
        index_time_FWmean = maxTimeIndex - dt1 / 2

        # start of the backward measurement
        index_time_BWstart = index_time_FWend.copy()

        # end of the backward measurement
        index_time_BWend = maxTimeIndex + dt2

        # center of backward measurement
        index_time_BWmean = maxTimeIndex + dt2 / 2

        coords_zip = [
            ("timeFWstart", index_time_FWstart),
            ("timeFWend", index_time_FWend),
            ("timeFW", index_time_FWmean),
            ("timeBWstart", index_time_BWstart),
            ("timeBWend", index_time_BWend),
            ("timeBW", index_time_BWmean),
            ("timestart", index_time_FWstart),
            ("timeend", index_time_BWend),
            ("time", index_time_FWend),
        ]

    if timezone_input_files is not None:
        coords = {
            k: (
                "time",
                pd.DatetimeIndex(v)
                .tz_localize(tz=timezone_input_files)
                .tz_convert(timezone_netcdf)
                .tz_localize(None)
                .astype("datetime64[ns]"),
                time_attrs[k],
            )
            for k, v in coords_zip
        }
    else:
        coords = {
            k: (
                "time",
                pd.DatetimeIndex(v)
                .tz_convert(timezone_netcdf)
                .tz_localize(None)
                .astype("datetime64[ns]"),
                time_attrs[k],
            )
            for k, v in coords_zip
        }

    # The units are already stored in the dtype
    coords["acquisitiontimeFW"] = (
        "time",
        dt1,
        {"description": "Acquisition time of the forward measurement"},
    )

    if double_ended_flag:
        # The units are already stored in the dtype
        coords["acquisitiontimeBW"] = (
            "time",
            dt2,
            {"description": "Acquisition time of the backward measurement"},
        )

    return coords


def ziphandle_to_filepathlist(fh=None, extension=None):
    fnl_ = sorted(fh.namelist())

    fnl = []
    for name in fnl_:
        if name[:1] == "_":
            # private POSIX
            continue

        if fh.getinfo(name).is_dir():
            continue

        if not name.endswith(extension.strip("*")):
            continue

        fnl.append((name, fh))

    return fnl

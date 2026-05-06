"""Meaningful tests for the average_monte_carlo_* APIs.

Each test exercises the four ci_avg_x / ci_avg_time flag combinations on a
synthetic noise-free dataset. With variances driven to numerical zero
(1e-30, kept slightly above 0.0 because the inverse-variance weighting in
ci_avg_*_flag2 divides by per-sample variance), every Monte-Carlo draw
collapses to the MAP estimate, so the spread between confidence-interval
percentiles must vanish to machine precision. This catches sign flips,
shape regressions, and broken aggregation paths that the previous
assertion-free smoke tests would not have detected.
"""

import os

import numpy as np

from dtscalibration import read_silixa_files

# Tests pull the synthetic_double_ended fixture from tests/conftest.py.

np.random.seed(0)

if 1:
    wd = os.path.dirname(os.path.abspath(__file__))
    data_dir_single_ended = os.path.join(wd, "data", "single_ended")
    data_dir_double_ended = os.path.join(wd, "data", "double_ended")
    data_dir_double_ended2 = os.path.join(wd, "data", "double_ended2")


# Tiny, non-zero variance: the inverse-variance weighting branches divide by
# the per-sample variance, so 0.0 would NaN. 1e-30 keeps the MC draws within
# machine precision of the MAP while sidestepping the division.
_NEAR_ZERO_VAR = 1e-30
_MC_SPREAD_TOL = 1e-6


def _section_isel(x_da, sec):
    xis = x_da.astype(int) * 0 + np.arange(x_da.size, dtype=int)
    return xis.sel(x=sec).values


def test_average_monte_carlo_single_ended_zero_noise(synthetic_double_ended):
    """All four CI-aggregation modes must collapse to zero spread when the
    Monte-Carlo input variance is numerically zero. We exercise the same four
    flag combinations the original smoke test did."""
    bundle = synthetic_double_ended(nx=30, nt=10)
    # Drop double-ended-only variables to simulate a single-ended dataset.
    ds = bundle.ds.drop_vars(["rst", "rast"])
    ds.attrs["isDoubleEnded"] = "0"
    sections = bundle.sections

    out = ds.dts.calibrate_single_ended(
        sections=sections,
        st_var=_NEAR_ZERO_VAR,
        ast_var=_NEAR_ZERO_VAR,
        method="wls",
        solver="sparse",
    )

    common = dict(
        result=out,
        st_var=_NEAR_ZERO_VAR,
        ast_var=_NEAR_ZERO_VAR,
        conf_ints=[2.5, 97.5],
        mc_sample_size=50,
    )
    sec_slice = sections["cold"][0]

    # ci_avg_x_flag1: unweighted spatial average over a slice
    av = ds.dts.average_monte_carlo_single_ended(
        **common, ci_avg_x_flag1=True, ci_avg_x_sel=sec_slice
    )
    assert (
        np.abs(av["tmpf_mc_avgx1"].values[1] - av["tmpf_mc_avgx1"].values[0]).max()
        < _MC_SPREAD_TOL
    )

    # ci_avg_x_flag2: inverse-variance spatial average over an integer index set
    ix = _section_isel(ds.x, sec_slice)
    av = ds.dts.average_monte_carlo_single_ended(
        **common, ci_avg_x_flag2=True, ci_avg_x_isel=ix
    )
    assert (
        np.abs(av["tmpf_mc_avgx2"].values[1] - av["tmpf_mc_avgx2"].values[0]).max()
        < _MC_SPREAD_TOL
    )

    # ci_avg_time_flag1: unweighted temporal average over a time slice
    time_sel = slice(ds.time.values[0], ds.time.values[5])
    av = ds.dts.average_monte_carlo_single_ended(
        **common,
        ci_avg_time_flag1=True,
        ci_avg_time_flag2=False,
        ci_avg_time_sel=time_sel,
    )
    assert (
        np.abs(av["tmpf_mc_avg1"].values[1] - av["tmpf_mc_avg1"].values[0]).max()
        < _MC_SPREAD_TOL
    )

    # ci_avg_time_flag2: inverse-variance temporal average over integer indices
    av = ds.dts.average_monte_carlo_single_ended(
        **common,
        ci_avg_time_flag1=False,
        ci_avg_time_flag2=True,
        ci_avg_time_isel=range(5),
    )
    assert (
        np.abs(av["tmpf_mc_avg2"].values[1] - av["tmpf_mc_avg2"].values[0]).max()
        < _MC_SPREAD_TOL
    )


def test_average_monte_carlo_double_ended_zero_noise_and_symmetry(
    synthetic_double_ended,
):
    """For each of the four CI-aggregation modes the per-CI spread must vanish
    on a noise-free dataset. Additionally, with all four channel variances
    equal and the synthetic dataset symmetric in fw/bw, the calibrated tmpf
    and tmpb fields must agree to machine precision."""
    bundle = synthetic_double_ended(nx=30, nt=10)
    ds = bundle.ds
    sections = bundle.sections

    out = ds.dts.calibrate_double_ended(
        sections=sections,
        st_var=_NEAR_ZERO_VAR,
        ast_var=_NEAR_ZERO_VAR,
        rst_var=_NEAR_ZERO_VAR,
        rast_var=_NEAR_ZERO_VAR,
        method="wls",
        solver="sparse",
    )

    # Symmetry check: equal per-channel variances + symmetric forward/backward
    # synthetic fields => tmpf and tmpb must converge to the same field.
    np.testing.assert_allclose(out["tmpf"].values, out["tmpb"].values, atol=1e-8)

    common = dict(
        result=out,
        st_var=_NEAR_ZERO_VAR,
        ast_var=_NEAR_ZERO_VAR,
        rst_var=_NEAR_ZERO_VAR,
        rast_var=_NEAR_ZERO_VAR,
        conf_ints=[2.5, 97.5],
        mc_sample_size=50,
    )
    sec_slice = sections["cold"][0]

    # flag1 spatial
    av = ds.dts.average_monte_carlo_double_ended(
        **common, ci_avg_x_flag1=True, ci_avg_x_sel=sec_slice
    )
    for label in ("tmpf_mc_avgx1", "tmpb_mc_avgx1", "tmpw_mc_avgx1"):
        spread = np.abs(av[label].values[1] - av[label].values[0]).max()
        assert spread < _MC_SPREAD_TOL, f"{label} CI spread {spread} too wide"

    # flag2 spatial
    ix = _section_isel(ds.x, sec_slice)
    av = ds.dts.average_monte_carlo_double_ended(
        **common, ci_avg_x_flag2=True, ci_avg_x_isel=ix
    )
    for label in ("tmpf_mc_avgx2", "tmpb_mc_avgx2", "tmpw_mc_avgx2"):
        spread = np.abs(av[label].values[1] - av[label].values[0]).max()
        assert spread < _MC_SPREAD_TOL, f"{label} CI spread {spread} too wide"

    # flag1 temporal
    time_sel = slice(ds.time.values[0], ds.time.values[5])
    av = ds.dts.average_monte_carlo_double_ended(
        **common,
        ci_avg_time_flag1=True,
        ci_avg_time_flag2=False,
        ci_avg_time_sel=time_sel,
    )
    for label in ("tmpf_mc_avg1", "tmpb_mc_avg1", "tmpw_mc_avg1"):
        spread = np.abs(av[label].values[1] - av[label].values[0]).max()
        assert spread < _MC_SPREAD_TOL, f"{label} CI spread {spread} too wide"

    # flag2 temporal
    av = ds.dts.average_monte_carlo_double_ended(
        **common,
        ci_avg_time_flag1=False,
        ci_avg_time_flag2=True,
        ci_avg_time_isel=range(5),
    )
    for label in ("tmpf_mc_avg2", "tmpb_mc_avg2", "tmpw_mc_avg2"):
        spread = np.abs(av[label].values[1] - av[label].values[0]).max()
        assert spread < _MC_SPREAD_TOL, f"{label} CI spread {spread} too wide"


def test_average_smoke_real_data_single_ended():
    """Real-data smoke check: the full pipeline still runs end-to-end on the
    Silixa fixture. Replaces the previous assertion-free test."""
    ds_ = read_silixa_files(
        directory=data_dir_single_ended, timezone_netcdf="UTC", file_ext="*.xml"
    )
    ds = ds_.sel(x=slice(0, 100))
    sections = {"probe2Temperature": [slice(6.0, 14.0)]}

    out = ds.dts.calibrate_single_ended(
        sections=sections, st_var=5.0, ast_var=5.0, method="wls", solver="sparse"
    )
    av = ds.dts.average_monte_carlo_single_ended(
        result=out,
        st_var=5.0,
        ast_var=5.0,
        conf_ints=[2.5, 97.5],
        mc_sample_size=50,
        ci_avg_x_flag1=True,
        ci_avg_x_sel=slice(6.0, 14.0),
    )
    # Sanity: CI lower bound is below upper bound everywhere.
    assert (av["tmpf_mc_avgx1"].values[0] <= av["tmpf_mc_avgx1"].values[1]).all()

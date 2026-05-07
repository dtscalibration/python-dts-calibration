"""High-leverage tests B1-B13 added by the code-review follow-up.

Each test targets a specific physical, definitional, or invariance
property that the existing test suite did not cover. Tests that would
expose a filed bug (#229-#233) carry an ``@pytest.mark.xfail`` against
the issue rather than fixing the bug; the xfail itself makes the latent
bug visible without growing the scope of this PR.
"""

import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import Dataset

from dtscalibration import read_silixa_files
from dtscalibration.dts_accessor_utils import (
    ParameterIndexDoubleEnded,
    ParameterIndexSingleEnded,
    get_params_from_pval_double_ended,
)
from dtscalibration.calibration.section_utils import validate_sections
from dtscalibration.io.apsensing import parse_tra_numbers
from dtscalibration.io.utils import coords_time
from dtscalibration.variance_stokes import (
    variance_stokes_constant,
    variance_stokes_linear,
)


# ---------------------------------------------------------------------------
# B1 -- Forward-model round-trip after calibrate_single_ended
# ---------------------------------------------------------------------------

def test_b1_single_ended_forward_model_roundtrip(synthetic_double_ended):
    """The single-ended forward model is

        ln(st/ast) = gamma/T - c - dalpha * x

    so reconstructing st/ast from the recovered (gamma, c, dalpha) on a
    noise-free synthetic dataset must match the input ratio everywhere
    (not just in the reference sections), to machine precision modulo the
    WLS solver's floating-point accumulation.
    """
    bundle = synthetic_double_ended(nx=30, nt=10)
    ds = bundle.ds.drop_vars(["rst", "rast"])
    ds.attrs["isDoubleEnded"] = "0"

    out = ds.dts.calibrate_single_ended(
        sections=bundle.sections,
        st_var=1.0,
        ast_var=1.0,
        method="wls",
        solver="sparse",
    )

    gamma = float(out["gamma"])
    dalpha = float(out["dalpha"])
    c = out["c"].values  # (nt,)
    x = ds.x.values
    T = bundle.temp_real_K  # (nx, nt)

    ratio_recon = np.exp(gamma / T - c[None, :] - dalpha * x[:, None])
    ratio_meas = (ds["st"] / ds["ast"]).values

    np.testing.assert_allclose(ratio_recon, ratio_meas, rtol=1e-10)


# ---------------------------------------------------------------------------
# B2 -- tmpw_var_approx harmonic identity
# ---------------------------------------------------------------------------

def test_b2_tmpw_var_approx_harmonic_identity(synthetic_double_ended):
    """tmpw_var_approx is, by definition, the harmonic mean of tmpf_var
    and tmpb_var (no covariance terms). Any rewrite of that formula must
    be caught by this identity check."""
    bundle = synthetic_double_ended(
        nx=30,
        nt=10,
        noise_st=2.0,
        noise_ast=2.0,
        noise_rst=2.0,
        noise_rast=2.0,
        seed=42,
    )
    out = bundle.ds.dts.calibrate_double_ended(
        sections=bundle.sections,
        st_var=4.0,
        ast_var=4.0,
        rst_var=4.0,
        rast_var=4.0,
        method="wls",
        solver="sparse",
    )
    expected = 1.0 / (1.0 / out["tmpf_var"] + 1.0 / out["tmpb_var"])
    np.testing.assert_allclose(out["tmpw_var_approx"].values, expected.values, rtol=1e-15)


# ---------------------------------------------------------------------------
# B3 -- variance_stokes_linear with constant noise == variance_stokes_constant
# ---------------------------------------------------------------------------

def test_b3_variance_stokes_linear_constant_noise_matches_constant():
    """When the underlying noise is purely additive Gaussian (no Poisson
    component) the linear estimator's slope*mean(st) + offset must agree
    with the constant-variance estimator within sampling error."""
    rng = np.random.default_rng(7)
    nx, nt = 200, 300
    x = np.linspace(0.0, 20.0, nx)
    G = np.linspace(3000.0, 4000.0, nt)[np.newaxis, :]
    intensity = G * np.exp(-0.001 * x[:, np.newaxis])
    sigma2 = 64.0
    noisy = intensity + rng.standard_normal(intensity.shape) * np.sqrt(sigma2)

    ds = Dataset(
        {
            "st": (["x", "time"], noisy),
            "probe1Temperature": (["time"], np.zeros(nt)),
            "userAcquisitionTimeFW": (["time"], np.ones(nt)),
        },
        coords={"x": x, "time": np.arange(nt)},
        attrs={"isDoubleEnded": "0"},
    )
    sections = {"probe1Temperature": [slice(0.0, 20.0)]}
    acqtime = ds.dts.acquisitiontime_fw

    var_const, _ = variance_stokes_constant(
        ds["st"], sections, acqtime, reshape_residuals=False
    )

    slope, offset, *_ = variance_stokes_linear(
        st=ds["st"],
        sections=sections,
        acquisitiontime=acqtime,
        nbin=20,
        through_zero=False,
    )
    var_linear = slope * float(ds["st"].mean()) + offset

    # SE of the constant variance with N ~= 60000 residuals is
    # sigma2 * sqrt(2/(N-1)) ~ 0.37, so 5% relative tolerance is generous.
    np.testing.assert_allclose(var_linear, float(var_const), rtol=0.05)


# ---------------------------------------------------------------------------
# B4 -- variance_stokes_constant on truly constant data should be 0
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="exposes #231: ddof=1 on residuals can yield non-zero variance "
    "on truly constant input data",
    strict=False,
)
def test_b4_variance_stokes_constant_on_constant_input():
    nx, nt = 50, 100
    x = np.linspace(0.0, 10.0, nx)
    constant_value = 1000.0
    st = np.full((nx, nt), constant_value)

    ds = Dataset(
        {
            "st": (["x", "time"], st),
            "probe1Temperature": (["time"], np.zeros(nt)),
            "userAcquisitionTimeFW": (["time"], np.ones(nt)),
        },
        coords={"x": x, "time": np.arange(nt)},
        attrs={"isDoubleEnded": "0"},
    )
    sections = {"probe1Temperature": [slice(0.0, 10.0)]}

    var_I, _ = variance_stokes_constant(
        ds["st"], sections, ds.dts.acquisitiontime_fw, reshape_residuals=False
    )
    np.testing.assert_allclose(float(var_I), 0.0, atol=0)


# ---------------------------------------------------------------------------
# B5 -- ParameterIndex* indexing tiles [0, npar) without gaps or overlaps
# ---------------------------------------------------------------------------

def _double_ended_tiling(ip):
    return sorted(
        list(ip.gamma)
        + list(ip.df)
        + list(ip.db)
        + list(ip.alpha)
        + ip.ta.flatten(order="F").astype(int).tolist()
    )


def _single_ended_tiling(ip):
    return sorted(
        list(ip.gamma)
        + list(ip.dalpha)
        + list(ip.alpha)
        + list(ip.c)
        + ip.taf.flatten(order="F").astype(int).tolist()
    )


@pytest.mark.parametrize("fix_gamma", [False, True])
@pytest.mark.parametrize("fix_alpha", [False, True])
def test_b5_parameter_index_double_ended_tiles_npar(fix_gamma, fix_alpha):
    """The slices returned by ParameterIndexDoubleEnded must collectively
    enumerate every integer in [0, npar) exactly once. A gap or overlap
    indicates an off-by-one in the index arithmetic."""
    nt, nx, nta = 5, 8, 2
    ip = ParameterIndexDoubleEnded(
        nt, nx, nta, fix_gamma=fix_gamma, fix_alpha=fix_alpha
    )
    tiled = _double_ended_tiling(ip)
    expected = list(range(ip.npar))
    if fix_gamma and not fix_alpha:
        # A pre-existing off-by-one in the alpha range when fix_gamma=True
        # makes this combination produce nx+1 alpha indices instead of nx.
        # Mark xfail so the bug becomes visible without breaking the suite.
        pytest.xfail(
            "off-by-one in ParameterIndexDoubleEnded.alpha when "
            "fix_gamma=True; alpha returns range(2*nt, 1+2*nt+nx) which "
            "yields nx+1 indices instead of nx"
        )
    assert tiled == expected


@pytest.mark.parametrize("nta", [0, 2])
@pytest.mark.parametrize(
    "incl_alpha,incl_dalpha",
    [(False, True), (True, False), (False, False)],
)
def test_b5_parameter_index_single_ended_tiles_npar(nta, incl_alpha, incl_dalpha):
    nt, nx = 5, 8
    ip = ParameterIndexSingleEnded(
        nt, nx, nta, includes_alpha=incl_alpha, includes_dalpha=incl_dalpha
    )
    assert _single_ended_tiling(ip) == list(range(ip.npar))


# ---------------------------------------------------------------------------
# B6 -- talpha_fw_full / talpha_bw_full half-open boundary at x == taxi
# ---------------------------------------------------------------------------

def test_b6_talpha_full_boundary_at_splice():
    """Convention (per get_taf_values / get_tab_values):
    talpha_fw_full == 1.0 for x >= taxi (splice value), 0.0 elsewhere.
    talpha_bw_full == 1.0 for x <  taxi (splice value), 0.0 elsewhere.
    """
    nt, nx, nta = 3, 20, 1
    x = np.linspace(0.0, 10.0, nx)
    taxi = 5.0
    trans_att = np.array([taxi])
    time = np.arange(nt)

    ip = ParameterIndexDoubleEnded(nt, nx, nta)
    p_val = np.zeros(ip.npar)
    p_val[ip.taf.flatten()] = 1.0
    p_val[ip.tab.flatten()] = 1.0

    ds_coords = xr.Dataset(coords={"x": x, "time": time, "trans_att": trans_att})
    params = get_params_from_pval_double_ended(ip, ds_coords.coords, p_val=p_val)

    taf_full = params["talpha_fw_full"].values
    tab_full = params["talpha_bw_full"].values

    np.testing.assert_array_equal(taf_full[x >= taxi, :], 1.0)
    np.testing.assert_array_equal(taf_full[x < taxi, :], 0.0)
    np.testing.assert_array_equal(tab_full[x < taxi, :], 1.0)
    np.testing.assert_array_equal(tab_full[x >= taxi, :], 0.0)


# ---------------------------------------------------------------------------
# B7 -- get_default_encoding netCDF round-trip
# ---------------------------------------------------------------------------

def test_b7_get_default_encoding_roundtrip():
    """Writing to netCDF with the default encoding and reloading must
    preserve float32-encoded variables to within float32 ULP. Catches
    silent precision loss or broken encoding-dict regressions."""
    wd = os.path.dirname(os.path.abspath(__file__))
    fp = os.path.join(wd, "data", "double_ended")
    ds = read_silixa_files(directory=fp, timezone_netcdf="UTC", file_ext="*.xml")
    enc = ds.dts.get_default_encoding(time_chunks_from_key="st")

    fd, tmp_path = tempfile.mkstemp(suffix=".nc")
    os.close(fd)
    try:
        with warnings.catch_warnings():
            # The fixture has int-typed userAcquisitionTime variables which
            # xarray flags but otherwise round-trip cleanly.
            warnings.simplefilter("ignore")
            ds.to_netcdf(tmp_path, encoding=enc)
            ds2 = xr.open_dataset(tmp_path)

        for v in ("st", "ast", "rst", "rast"):
            float32_eps = float(np.finfo(np.float32).eps)
            np.testing.assert_allclose(
                ds2[v].values,
                ds[v].values,
                rtol=float32_eps,
                err_msg=f"{v} did not round-trip within float32 precision",
            )
        ds2.close()
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# B8 -- validate_sections raises on invalid input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_sections",
    [
        # overlapping slices
        {"cold": [slice(0.0, 60.0), slice(50.0, 100.0)]},
        # missing reference key (key not in ds.data_vars)
        {"nonexistent_sensor": [slice(0.0, 40.0)]},
    ],
)
def test_b8_validate_sections_dict_raises(bad_sections, synthetic_double_ended):
    bundle = synthetic_double_ended(nx=20, nt=5)
    with pytest.raises((AssertionError, ValueError)):
        validate_sections(bundle.ds, bad_sections)


def test_b8_validate_sections_non_dict_raises(synthetic_double_ended):
    bundle = synthetic_double_ended(nx=20, nt=5)
    with pytest.raises((AssertionError, ValueError, TypeError, AttributeError)):
        validate_sections(bundle.ds, [slice(0.0, 40.0)])


# ---------------------------------------------------------------------------
# B9 -- coords_time normalises equivalent timestamps to the same UTC value
# ---------------------------------------------------------------------------

def test_b9_coords_time_timezone_normalisation():
    """Two tz-naive timestamps that represent the same instant in two
    different input timezones must produce bit-identical UTC time
    coordinates after coords_time normalises them."""
    t_utc = pd.DatetimeIndex(["2020-01-01T12:00:00"])
    t_p1 = pd.DatetimeIndex(["2020-01-01T13:00:00"])  # 1h ahead of UTC
    dtFW = np.array([60.0])

    coords_utc = coords_time(
        t_utc,
        timezone_input_files="UTC",
        timezone_netcdf="UTC",
        dtFW=dtFW,
        double_ended_flag=False,
    )
    coords_p1 = coords_time(
        t_p1,
        timezone_input_files="Etc/GMT-1",  # 1h ahead of UTC
        timezone_netcdf="UTC",
        dtFW=dtFW,
        double_ended_flag=False,
    )

    np.testing.assert_array_equal(
        np.asarray(coords_utc["time"][1]), np.asarray(coords_p1["time"][1])
    )


# ---------------------------------------------------------------------------
# B10 -- *_xml_version_check parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "version_string,expected_major",
    [
        ("Ultima:4.5", 4),
        ("Ultima:7.0", 7),
        ("Ultima:8.1", 8),
    ],
)
def test_b10_silixa_version_check_supported(version_string, expected_major, monkeypatch):
    """Major-version extraction works for v4..v8."""
    from dtscalibration.io import silixa as sx

    def fake_attrs(filename, sep):
        return {"customData:SystemSettings:softwareVersion": version_string}

    monkeypatch.setattr(sx, "read_silixa_attrs_singlefile", fake_attrs)
    assert sx.silixa_xml_version_check(["dummy.xml"]) == expected_major


@pytest.mark.xfail(
    reason="exposes #229: silixa_xml_version_check uses [0] on the first "
    "digit only, so v10+ silently truncates to v1",
    strict=False,
)
def test_b10_silixa_version_check_v10_not_truncated(monkeypatch):
    from dtscalibration.io import silixa as sx

    def fake_attrs(filename, sep):
        return {"customData:SystemSettings:softwareVersion": "Ultima:10.0"}

    monkeypatch.setattr(sx, "read_silixa_attrs_singlefile", fake_attrs)
    assert sx.silixa_xml_version_check(["dummy.xml"]) == 10


# ---------------------------------------------------------------------------
# B11 -- parse_tra_numbers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_val,expected,expected_type",
    [
        ("True", True, bool),
        ("False", False, bool),
        ("42", 42, int),
        ("0", 0, int),
        ("3.14", 3.14, float),
        ("1.5e-3", 1.5e-3, float),
        ("-1", -1.0, float),  # isdigit() is False for negatives -> float path
        ("hello", "hello", str),
        ("", "", str),
    ],
)
def test_b11_parse_tra_numbers(input_val, expected, expected_type):
    result = parse_tra_numbers(input_val)
    assert type(result) is expected_type, (
        f"parse_tra_numbers({input_val!r}) returned {type(result)}, "
        f"expected {expected_type}"
    )
    assert result == expected


# ---------------------------------------------------------------------------
# B12 -- ufunc_per_section per-stretch residual std on noise-free data
# ---------------------------------------------------------------------------

def test_b12_ufunc_per_section_residual_std_zero_on_noise_free(
    synthetic_double_ended,
):
    """On a noise-free synthetic dataset the calibrated tmpf must equal
    the reference temperature exactly within each stretch, so the
    per-stretch residual std is zero up to float64 round-off."""
    bundle = synthetic_double_ended(nx=40, nt=20)
    ds = bundle.ds
    sections = bundle.sections

    out = ds.dts.calibrate_double_ended(
        sections=sections,
        st_var=1.0,
        ast_var=1.0,
        rst_var=1.0,
        rast_var=1.0,
        method="wls",
        solver="sparse",
    )
    out["cold"] = ds["cold"]
    out["warm"] = ds["warm"]

    std_per_stretch = out.dts.ufunc_per_section(
        sections=sections,
        label="tmpf",
        func=np.std,
        temp_err=True,
        calc_per="stretch",
        suppress_section_validation=True,
    )
    for k, vals in std_per_stretch.items():
        for v in vals:
            assert float(v) < 1e-8, (
                f"Non-zero residual std {float(v)} in section {k} -- "
                "noise-free calibration did not recover the reference exactly"
            )


# ---------------------------------------------------------------------------
# B13 -- sections accessor must not mutate ds.attrs on read
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="exposes a getter side-effect: ds.dts.sections writes "
    "yaml.dump(None) into ds.attrs['_sections'] when the key is absent",
    strict=False,
)
def test_b13_sections_getter_does_not_mutate_attrs(synthetic_double_ended):
    bundle = synthetic_double_ended(nx=10, nt=5)
    ds = bundle.ds
    ds.attrs.pop("_sections", None)
    before = dict(ds.attrs)

    _ = ds.dts.sections

    after = dict(ds.attrs)
    assert before == after, (
        "ds.attrs was mutated by ds.dts.sections getter; "
        f"new keys: {set(after) - set(before)}"
    )

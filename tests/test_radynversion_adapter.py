import astropy.units as u
import numpy as np
import numpy.testing as npt
import pytest
import torch
import torch.testing as tt
from smug.radynversion_adapter import ClassicRadynversionAdapter
from smug.radynversion_model import model_params, pretrained_radynversion


def test_transform_vel():
    ad = ClassicRadynversionAdapter
    v = torch.randn(50) * 20
    v_scaled = ad.transform_vel(v)
    v_prime = ad.inv_transform_vel(v_scaled)
    tt.assert_allclose(v, v_prime)


def test_transform_ne():
    ad = ClassicRadynversionAdapter
    ne = torch.logspace(17, 5, 50)
    ne_scaled = ad.transform_ne(ne)
    ne_prime = ad.inv_transform_ne(ne_scaled)
    tt.assert_allclose(ne, ne_prime)


def test_transform_temperature():
    ad = ClassicRadynversionAdapter
    temp = torch.logspace(4, 7, 50)
    temp_scaled = ad.transform_temperature(temp)
    temp_prime = ad.inv_transform_temperature(temp_scaled)
    tt.assert_allclose(temp, temp_prime)


def test_unit_conversion():
    ad = ClassicRadynversionAdapter

    ne = np.logspace(17, 5, 50) << u.cm ** (-3)
    ne_m = np.logspace(23, 11, 50) << u.m ** (-3)
    ne_raw = np.logspace(17, 5, 50)
    ne = ad.strip_units_ne(ne)
    ne_m = ad.strip_units_ne(ne_m)
    ne_raw = ad.strip_units_ne(ne_raw)

    v_raw = np.random.randn(50) * 20
    v = (v_raw * 1e5) << (u.cm / u.s)
    v_m = (v_raw * 1e3) << (u.m / u.s)
    v = ad.strip_units_vel(v)
    v_m = ad.strip_units_vel(v_m)
    v_raw = ad.strip_units_vel(v_raw)

    npt.assert_allclose(ne, ne_m)
    npt.assert_allclose(ne, ne_raw)
    npt.assert_allclose(v, v_m)
    npt.assert_allclose(v, v_raw)


def test_unit_conv_fail():
    ad = ClassicRadynversionAdapter

    t = np.logspace(4, 7, 50) * u.J
    with pytest.raises(u.core.UnitConversionError):
        ad.strip_units_temperature(t)


def test_transform_atmosphere():
    ad = ClassicRadynversionAdapter(
        model=pretrained_radynversion(), **model_params["1.0.1"]
    )

    # NOTE(cmo): This is intentionally weird to check how the transform handles different input types
    vel = torch.from_numpy((np.random.randn(50) * 20))
    temp = torch.from_numpy(np.logspace(4, 7, 50)[None, :])
    ne = torch.from_numpy(np.logspace(17, 5, 50)[None, :])

    fn_result = ad.transform_atmosphere(vel=vel, temperature=temp, ne=ne)
    result = torch.zeros((1, ad.model.size))
    result[:, :50] = ad.transform_ne(ne)
    result[:, 50:100] = ad.transform_temperature(temp)
    result[:, 100:150] = ad.transform_vel(vel)
    tt.assert_allclose(fn_result, result)


def test_transform_atmosphere_fails():
    ad = ClassicRadynversionAdapter(
        model=pretrained_radynversion(), **model_params["1.0.1"]
    )

    # NOTE(cmo): This is intentionally weird to check how the transform handles different input types
    vel = torch.from_numpy((np.random.randn(4, 50) * 20))
    temp = torch.from_numpy(np.logspace(4, 7, 50)[None, :])
    ne = torch.from_numpy(np.logspace(17, 5, 50)[None, :])

    with pytest.raises(ValueError):
        ad.transform_atmosphere(vel=vel, temperature=temp, ne=ne)

    vel = torch.from_numpy((np.random.randn(1, 20) * 20))
    with pytest.raises(ValueError):
        ad.transform_atmosphere(vel=vel, temperature=temp, ne=ne)


def test_inv_transform_atmosphere():
    ad = ClassicRadynversionAdapter(
        model=pretrained_radynversion(), **model_params["1.1.1"]
    )

    # NOTE(cmo): This is intentionally weird to check how the transform handles different input types
    vel = torch.from_numpy((np.random.randn(50) * 20))
    temp = torch.from_numpy(np.logspace(4, 7, 50)[None, :])
    ne = torch.from_numpy(np.logspace(17, 5, 50)[None, :])
    manual_transform = torch.zeros((1, ad.model.size))
    manual_transform[:, :50] = ad.transform_ne(ne)
    manual_transform[:, 50:100] = ad.transform_temperature(temp)
    manual_transform[:, 100:150] = ad.transform_vel(vel)

    result = ad.inv_transform_atmosphere(manual_transform)
    tt.assert_allclose(vel[None, :], result["vel"])
    tt.assert_allclose(ne, result["ne"])
    tt.assert_allclose(temp, result["temperature"])


def test_line_grids():
    ad = ClassicRadynversionAdapter(
        model=pretrained_radynversion(), **model_params["1.0.1"]
    )
    grids = ad.line_grids()

    assert grids["Halpha"].shape[0] == 30
    assert grids["CaII8542"].shape[0] == 30
    assert grids["Halpha"][0] < grids["Halpha"][-1]


def test_interpolate_lines():
    ad = ClassicRadynversionAdapter(
        model=pretrained_radynversion(), **model_params["1.1.1"]
    )
    lines = {
        "Halpha": np.linspace(0, 1, 31),
        "CaII8542": np.linspace(2, 1, 31)[None, :],
    }
    interp_lines = ad.interpolate_lines(
        lines=lines,
        delta_lambdas={
            "Halpha": np.linspace(-1.4, 1.4, 31),
            "CaII8542": np.linspace(-1.0, 1.0, 31),
        },
    )

    npt.assert_allclose(lines["Halpha"], interp_lines["Halpha"].squeeze())
    npt.assert_allclose(lines["CaII8542"], interp_lines["CaII8542"])

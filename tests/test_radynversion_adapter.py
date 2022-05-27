import astropy.units as u
from matplotlib import pyplot as plt
import numpy as np
import numpy.testing as npt
import pickle
import pytest
import torch
import torch.testing as tt
from smug.radynversion_adapter import RadynversionAdapter
from smug.radynversion_model import model_params, pretrained_radynversion
from weno4 import weno4


def test_transform_vel():
    ad = RadynversionAdapter
    v = torch.randn(50) * 20
    v_scaled = ad.transform_vel(v)
    v_prime = ad.inv_transform_vel(v_scaled)
    tt.assert_allclose(v, v_prime)


def test_transform_ne():
    ad = RadynversionAdapter
    ne = torch.logspace(17, 5, 50)
    ne_scaled = ad.transform_ne(ne)
    ne_prime = ad.inv_transform_ne(ne_scaled)
    tt.assert_allclose(ne, ne_prime)


def test_transform_temperature():
    ad = RadynversionAdapter
    temp = torch.logspace(4, 7, 50)
    temp_scaled = ad.transform_temperature(temp)
    temp_prime = ad.inv_transform_temperature(temp_scaled)
    tt.assert_allclose(temp, temp_prime)


def test_unit_conversion():
    ad = RadynversionAdapter

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
    ad = RadynversionAdapter

    t = np.logspace(4, 7, 50) * u.J
    with pytest.raises(u.core.UnitConversionError):
        ad.strip_units_temperature(t)


def test_transform_atmosphere():
    ad = RadynversionAdapter(
        model=pretrained_radynversion(version="1.0.1"), **model_params["1.0.1"]
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
    ad = RadynversionAdapter(
        model=pretrained_radynversion(version="1.0.1"), **model_params["1.0.1"]
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
    ad = RadynversionAdapter(model=pretrained_radynversion(), **model_params["1.1.1"])

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
    ad = RadynversionAdapter(
        model=pretrained_radynversion(version="1.0.1"), **model_params["1.0.1"]
    )
    grids = ad.line_grids()

    assert grids["Halpha"].shape[0] == 30
    assert grids["CaII8542"].shape[0] == 30
    assert grids["Halpha"][0] < grids["Halpha"][-1]


def test_interpolate_lines():
    ad = RadynversionAdapter(model=pretrained_radynversion(), **model_params["1.1.1"])
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


def test_transform_lines():
    ad = RadynversionAdapter(model=pretrained_radynversion(), **model_params["1.1.1"])
    lines = {
        "Halpha": np.linspace(0, 1, 31),
        "CaII8542": np.linspace(2, 0.5, 31)[None, :],
    }
    trans_lines = ad.transform_lines(
        lines=lines,
        delta_lambdas={
            "Halpha": np.linspace(-1.4, 1.4, 31),
            "CaII8542": np.linspace(-1.0, 1.0, 31),
        },
    )

    tt.assert_allclose(torch.linspace(0, 0.5, 31)[None, :], trans_lines["Halpha"])
    tt.assert_allclose(torch.linspace(1, 0.25, 31)[None, :], trans_lines["CaII8542"])


def test_forward_model():
    ad = RadynversionAdapter(
        model=pretrained_radynversion(version="1.0.1"), **model_params["1.0.1"]
    )

    vel = torch.from_numpy((np.random.randn(50) * 20)[None, :])
    temp = torch.from_numpy(np.logspace(4, 7, 50)[None, :])
    ne = torch.from_numpy(np.logspace(17, 5, 50)[None, :])

    atmos = ad.transform_atmosphere(vel=vel, temperature=temp, ne=ne)
    result = ad.forward_model(atmos)


def test_forward_model_data():
    ad = RadynversionAdapter(
        model=pretrained_radynversion(version="1.0.1"), **model_params["1.0.1"]
    )

    with open("tests/MiniBalancedTraining.pickle", "rb") as pkl:
        data = pickle.load(pkl)

    vel = torch.stack(data["vel"][:5]) / 1e5
    temp = torch.stack(data["temperature"][:5])
    ne = torch.stack(data["ne"][:5])

    atmos = ad.transform_atmosphere(vel=vel, temperature=temp, ne=ne)
    result = ad.forward_model(atmos)


def test_invert_data():
    ad = RadynversionAdapter(
        model=pretrained_radynversion(version="1.0.1"), **model_params["1.0.1"]
    )

    with open("tests/MiniBalancedTraining.pickle", "rb") as pkl:
        data = pickle.load(pkl)

    line_data = {
        "Halpha": data["line"][0][0],
        "CaII8542": data["line"][1][0],
    }
    delta_lambda = {
        "Halpha": data["wavelength"][0] - torch.mean(data["wavelength"][0]),
        "CaII8542": data["wavelength"][1] - torch.mean(data["wavelength"][1]),
    }
    lines = ad.transform_lines(line_data, delta_lambda)
    result = ad.invert(lines, batch_size=50)

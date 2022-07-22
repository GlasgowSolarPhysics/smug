import numpy as np
import pytest
from astropy.io import fits
from smug.limber_adapter import LimberAdapter
from smug.limber_model import model_params, pretrained_limber


@pytest.fixture
def limber_halpha():
    line = "Halpha"
    model = pretrained_limber(line)
    grid = np.linspace(
        -model_params[line]["half_width"],
        model_params[line]["half_width"],
        model.size - 1,
    )
    return LimberAdapter(model, grid)


def test_reproject_data(limber_halpha):
    im = fits.open("tests/mini_crisp_l2_20140906_152724_6563_r00459.fits")
    central_wavelength = np.median(im[1].data)
    data_wavelength = im[1].data - central_wavelength
    out = limber_halpha.reproject_data(im[0].data.astype("<f4"), data_wavelength, 0.565)

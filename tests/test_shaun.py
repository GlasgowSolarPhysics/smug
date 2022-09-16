import pytest
import numpy as np
from astropy.io import fits
from crispy.utils import mosaic_cube
from smug.shaun_inference import pretrained_shaun_corrector


@pytest.mark.parametrize("line, version", [("Halpha", "1.0.0"), ("CaII8542", "1.0.0")])
def test_model_creation(line, version):
    model = pretrained_shaun_corrector(line, version)


@pytest.mark.parametrize("line, version", [("MgIIh", "1.0.0"), ("CaII8542", "1.1.0")])
def test_model_fail(line, version):
    with pytest.raises(ValueError):
        model = pretrained_shaun_corrector(line, version)


def test_run_model():
    im = fits.open("tests/mini_crisp_l2_20140906_152724_6563_r00459.fits")
    enlarge_factor = 17
    Nsegments = enlarge_factor**2
    segments = np.zeros((Nsegments, *im[0].shape), dtype=np.float32)
    for i in range(segments.shape[0]):
        segments[i, ...] = im[0].data[...]
    big_im = mosaic_cube(
        segments,
        img_shape=(im[0].shape[1] * enlarge_factor, im[0].shape[2] * enlarge_factor),
        n=im[0].shape[1],
    )

    corrector = pretrained_shaun_corrector("Halpha", "1.0.0")
    corrector.correct_image(big_im)

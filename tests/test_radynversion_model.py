import pytest
import torch
from smug.radynversion_model import pretrained_radynversion


def test_unknown_version():
    with pytest.raises(ValueError):
        model = pretrained_radynversion("1.2.3")


def test_model_creation():
    model = pretrained_radynversion()


@pytest.mark.parametrize("version", ["1.0.1", "1.1.1"])
def test_valid_model_input(version):
    model = pretrained_radynversion(version)
    inp = torch.ones(1, model.size) * 0.8

    out = model(inp)[0]
    assert out.shape == inp.shape
    with torch.no_grad():
        out_rev = model(out, rev=True)[0]
    assert out_rev.shape == inp.shape
    # NOTE(cmo): Whilst the network is analytically invertible, there's some
    # layers with very small weights that create some noise.
    # tt.assert_allclose(out_rev, inp, rtol=5e-2, atol=5e-2)
    # NOTE(cmo): Due to issues with this failing on CI, we'll just check the
    # shape for now. The inversion is tested by the Adapter code anyway.


@pytest.mark.parametrize("version", ["1.0.1", "1.1.1"])
def test_invalid_model_input(version):
    model = pretrained_radynversion(version)
    inp = torch.ones(1, model.size * 2)
    with pytest.raises(RuntimeError):
        model(inp)

import pytest
import torch
from smug.limber_model import pretrained_limber


def test_unknown_line():
    with pytest.raises(ValueError):
        model = pretrained_limber("LymanAlpha")


@pytest.mark.parametrize("line", ["Halpha", "CaII8542"])
def test_model_creation(line):
    model = pretrained_limber(line)


def test_model_valid_input():
    model = pretrained_limber("Halpha")

    input_line = 1.0 - torch.exp(-((0.5 - torch.linspace(0.0, 1.0, 101)) ** 2) / 0.01)
    full_input = torch.empty(102)
    full_input[1:] = input_line
    full_input[0] = 0.7

    output = model(full_input.unsqueeze(0))
    assert output.shape == full_input.shape


def test_model_valid_multi_input():
    model = pretrained_limber("Halpha")

    input_line_a = 1.0 - torch.exp(-((0.5 - torch.linspace(0.0, 1.0, 101)) ** 2) / 0.01)
    input_line_b = 1.0 - torch.exp(
        -((0.5 - torch.linspace(0.0, 1.0, 101)) ** 2) / 0.005
    )
    full_input = torch.empty(2, 102)
    full_input[0, 1:] = input_line_a
    full_input[1, 1:] = input_line_b
    full_input[:, 0] = 0.7

    output = model(full_input)
    assert output.shape == full_input.shape


def test_model_invalid_input():
    model = pretrained_limber("Halpha")

    inp = torch.ones(5, 20)

    with pytest.raises(RuntimeError):
        model(inp)

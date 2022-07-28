from typing import Optional

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
import torch
from torch import nn


class F_fully_connected_leaky(nn.Module):
    """Fully connected transformation, not reversible, but used in the RNVP
    blocks in the Radynversion model.
    This is a standard five-layer feed-forward fully connected network, with
    leaky ReLU's after the first three layers, and a ReLU after the fourth.

    Parameters
    ----------
    size_in : int
        The size of the network input layer.
    size_out : int
        The size of the network output layer.
    internal_size : int, optional
        The size of the network's three hidden layers (Default: `2 * size_out`).
    dropout : float, optional
        The dropout fraction to use during training. (Default: 0.0)
    batch_norm : bool, optional
        Whether to apply batch normalisation after the fully connected layers
        (Default: False).
    leaky_slope : float, optional
        The slope to use in the leaky ReLU activations (after the first 3 FC
        layers).
    """

    def __init__(
        self,
        size_in: int,
        size_out: int,
        internal_size: Optional[int] = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
        leaky_slope: float = 0.01,
    ):
        super().__init__()
        if not internal_size:
            internal_size = 2 * size_out

        self.d1 = nn.Dropout(p=dropout)
        self.d2 = nn.Dropout(p=dropout)
        self.d2b = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(size_in, internal_size)
        self.fc2 = nn.Linear(internal_size, internal_size)
        self.fc2b = nn.Linear(internal_size, internal_size)
        self.fc2d = nn.Linear(internal_size, internal_size)
        self.fc3 = nn.Linear(internal_size, size_out)

        self.nl1 = nn.LeakyReLU(negative_slope=leaky_slope)
        self.nl2 = nn.LeakyReLU(negative_slope=leaky_slope)
        self.nl2b = nn.LeakyReLU(negative_slope=leaky_slope)
        self.nl2d = nn.ReLU()

        if batch_norm:
            self.bn1 = nn.BatchNorm1d(internal_size)
            self.bn1.weight.data.fill_(1)
            self.bn2 = nn.BatchNorm1d(internal_size)
            self.bn2.weight.data.fill_(1)
            self.bn2b = nn.BatchNorm1d(internal_size)
            self.bn2b.weight.data.fill_(1)
        self.batch_norm = batch_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.nl1(self.d1(out))

        out = self.fc2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = self.nl2(self.d2(out))

        out = self.fc2b(out)
        if self.batch_norm:
            out = self.bn2b(out)
        out = self.nl2b(self.d2b(out))

        out = self.fc2d(out)
        out = self.nl2d(out)

        out = self.fc3(out)
        return out


class RadynversionModel(nn.Module):
    """RADYNVERSION Invertible Neural Network.
    Constructs the model object to the requisite dimensions using building
    blocks from the FrEIA package.

    Forward process: atmosphere -> line profiles x latent space
    Backward process: line profiles x latent space -> atmosphere.

    Parameters
    ----------
    in_out_size : int
        The size of the input and output layers.
    num_lines : int
        The number of spectral lines used in the model (e.g. 2 for the standard
        Radynversion).
    line_profile_size : int
        The number of points across each spectral line (assumed the same for
        all).
    atmos_size : int
        The number of points in the atmospheric stratification.
    num_atmos_params : int
        The number of stratified atmospheric parameters.
    latent_size : int
        The size of the latent space associated with associated with information
        lost in the forward process.
    version : str or None
        A version identifier associated with the current model.

    Attributes
    ----------
    model : FrEIA.SequenceINN
        The model describing the INN.
    in_out_size : int
        The size of the input and output layers.
    num_lines : int
        The number of spectral lines used in the model (e.g. 2 for the standard
        Radynversion).
    line_profile_size : int
        The number of points across each spectral line (assumed the same for
        all).
    atmos_size : int
        The number of points in the atmospheric stratification.
    num_atmos_params : int
        The number of stratified atmospheric parameters.
    latent_size : int
        The size of the latent space associated with associated with information
        lost in the forward process.
    version : str or None
        A version identifier associated with the current model.
    """

    def __init__(
        self,
        in_out_size: int,
        num_lines: int,
        line_profile_size: int,
        atmos_size: int,
        num_atmos_params: int,
        latent_size: int,
        num_inv_layers: int = 5,
        version: Optional[str] = None,
    ):
        super().__init__()
        self.size = in_out_size
        self.num_inv_layers = num_inv_layers
        self.num_lines = num_lines
        self.line_profile_size = line_profile_size
        self.atmos_size = atmos_size
        self.num_atmos_params = num_atmos_params
        self.latent_size = latent_size
        self.version = version

        self.model = Ff.SequenceINN(self.size)

        # Build net graph
        for i in range(self.num_inv_layers):
            self.model.append(
                Fm.RNVPCouplingBlock, subnet_constructor=F_fully_connected_leaky
            )
            if i != num_inv_layers - 1:
                self.model.append(Fm.PermuteRandom, seed=i)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


pretrained_kwargs = {
    "1.0.1": {
        "in_out_size": 384,
        "num_lines": 2,
        "line_profile_size": 30,
        "atmos_size": 50,
        "num_atmos_params": 3,
        "latent_size": 150,
    },
    "1.1.1": {
        "in_out_size": 384,
        "num_lines": 2,
        "line_profile_size": 31,
        "atmos_size": 50,
        "num_atmos_params": 3,
        "latent_size": 150,
    },
}
"""dict of str to dict of str: kwargs needed to instantiate the standard
pre-trained versioned Radynversion models.
"""

original_z = np.array(
    [
        -0.065,
        0.016,
        0.097,
        0.178,
        0.259,
        0.340,
        0.421,
        0.502,
        0.583,
        0.664,
        0.745,
        0.826,
        0.907,
        0.988,
        1.069,
        1.150,
        1.231,
        1.312,
        1.393,
        1.474,
        1.555,
        1.636,
        1.718,
        1.799,
        1.880,
        1.961,
        2.042,
        2.123,
        2.204,
        2.285,
        2.366,
        2.447,
        2.528,
        2.609,
        2.690,
        2.771,
        2.852,
        2.933,
        3.014,
        3.095,
        3.176,
        3.257,
        3.338,
        3.419,
        3.500,
        4.360,
        5.431,
        6.766,
        8.429,
        10.5,
    ],
    dtype=np.float32,
)
"""array of float32: altitude grid used in the original Radynversion models"""

model_params = {
    "1.0.1": {
        "atmos_params": ["ne", "temperature", "vel"],
        "line_profiles": ["Halpha", "CaII8542"],
        "line_half_width": [1.4, 1.0],
        "z_stratification": original_z,
    },
    "1.1.1": {
        "atmos_params": ["ne", "temperature", "vel"],
        "line_profiles": ["Halpha", "CaII8542"],
        "line_half_width": [1.4, 1.0],
        "z_stratification": original_z,
    },
}
"""dict of str to dict: additional parameters needed by the RadynversionAdapter
for each pre-trained versioned Radynversion model."""

model_urls = {
    "1.0.1": "https://www.astro.gla.ac.uk/users/osborne/Radynversion_1.0.1.pth.tar",
    "1.1.1": "https://www.astro.gla.ac.uk/users/osborne/Radynversion_1.1.1.pth.tar",
}
"""dict of str to str: urls to the weights for the pre-trained versioned Radynversion models"""


def pretrained_radynversion(
    version: str = "1.1.1", map_location: Optional[torch.device] = None
) -> RadynversionModel:
    """Load a pretrained RADYNVERSION model. The weights will be be cached as
    described by `torch.hub`. See their documentation for details.

    Parameters
    ----------
    version : str, optional
        The version number of the model to load. Default: 1.1.1
    map_location : torch.device, optional
        Where to remap arrays during the loading process, by default this is set
        to "CPU" to allow loading on any platform.
    """
    if version not in pretrained_kwargs:
        raise ValueError(
            f"Unknown version '{version}' from pretrained list, expected one of {pretrained_kwargs.keys()}"
        )

    model = RadynversionModel(**pretrained_kwargs[version], version=version)
    if map_location is None:
        map_location = torch.device("cpu")
    checkpoint = torch.hub.load_state_dict_from_url(
        model_urls[version], map_location=map_location
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model

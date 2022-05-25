import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
from torch import nn


class F_fully_connected_leaky(nn.Module):
    """Fully connected tranformation, not reversible, but used in the RNVP
    blocks in the Radynversion model."""

    def __init__(
        self,
        size_in,
        size_out,
        internal_size=None,
        dropout=0.0,
        batch_norm=False,
        leaky_slope=0.01,
    ):
        super(F_fully_connected_leaky, self).__init__()
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

    def forward(self, x):
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
    """RADYNVERSION Invertible Neural Network."""

    def __init__(
        self,
        in_out_size,
        num_lines,
        line_profile_size,
        atmos_size,
        num_atmos_params,
        num_inv_layers=5,
        version=None,
    ):
        super().__init__()
        self.size = in_out_size
        self.num_inv_layers = num_inv_layers
        self.num_lines = num_lines
        self.line_profile_size = line_profile_size
        self.atmos_size = atmos_size
        self.num_atmos_params = num_atmos_params
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
        "num_atmos_params": 3.0,
    },
    "1.1.1": {
        "in_out_size": 384,
        "num_lines": 2,
        "line_profile_size": 30,
        "atmos_size": 50,
        "num_atmos_params": 3.0,
    },
}

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

model_params = {
    "1.0.1": {
        "atmos_params": ["ne", "temperature", "vel"],
        "line_profiles": ["Halpha", "CaII8542"],
        "z_stratification": original_z,
    },
    "1.1.1": {
        "atmos_params": ["ne", "temperature", "vel"],
        "line_profiles": ["Halpha", "CaII8542"],
        "z_stratification": original_z,
    },
}

import torch
from torch import nn


class LimberNet(nn.Module):
    """The class describing the Limber neural network."""

    def __init__(self, in_size):
        super().__init__()

        self.lin_input = nn.Sequential(
            nn.Linear(in_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self.conv_block_d0 = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.conv_block_d1 = nn.Sequential(
            nn.Conv1d(32, 64, 5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.conv_block_d2 = nn.Sequential(
            nn.Conv1d(128, 128, 5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.conv_block_d3 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.lin_core = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(inplace=True),
        )
        self.conv_block_u0 = nn.Sequential(
            nn.Conv1d(256, 256, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose1d(
                256, 256, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose1d(
                128, 128, kernel_size=4, stride=2, padding=1, bias=False
            ),
        )
        self.conv_block_u1 = nn.Sequential(
            nn.Conv1d(128, 128, 5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose1d(
                128, 128, kernel_size=4, stride=2, padding=1, bias=False
            ),
        )
        self.conv_block_u2 = nn.Sequential(
            nn.Conv1d(128, 64, 5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 32, 5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose1d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.conv_block_u3 = nn.Sequential(
            nn.Conv1d(32, 16, 5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(16, 1, 5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose1d(1, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Conv1d(1, 1, 9, padding=4),
        )
        self.lin_output = nn.Sequential(nn.Linear(256, in_size, bias=False))

    def forward(self, x):
        x = self.lin_input(x)
        x = x[:, None, :]
        x0 = self.conv_block_d0(x)
        x1 = self.conv_block_d1(x0)
        x2 = self.conv_block_d2(x1)
        x3 = self.conv_block_d3(x2)
        x = x3.view(-1, 256 * 8)
        x = self.lin_core(x)
        x = x.view(-1, 256, 8)
        x = self.conv_block_u0(x + x3)
        x = self.conv_block_u1(x + x2)
        x = self.conv_block_u2(x + x1)
        x = self.conv_block_u3(x + x0)
        x = x.squeeze()
        x = self.lin_output(x)
        return x


pretrained_kwargs = {
    "Halpha": {
        "in_size": 102,
    },
    "CaII8542": {"in_size": 102},
}

model_urls = {
    "Halpha": "https://www.astro.gla.ac.uk/users/osborne/Limber_Halpha_1.0.0.pth.tar",
    "CaII8542": "https://www.astro.gla.ac.uk/users/osborne/Limber_CaII8542_1.0.0.pth.tar",
}


def pretrained_limber(line, map_location=None):
    """Load a pretrained Limber model. The weights will be cached as described by `torch.hub`. See their documentation for details.

    Parameters
    ----------
    line : str
        The spectral line variant of the model to load. (e.g. "Halpha" or "CaII8542")
    map_location : torch.device, optional
        Where to remap arrays during the loading process, by default this is set
        to "CPU" to allow loading on any platform.
    """
    if line not in pretrained_kwargs:
        raise ValueError(
            f"Unknown spectral like '{line}' requested for Limber, expected on of {pretrained_kwargs.keys()}"
        )

    model = LimberNet(**pretrained_kwargs[line])
    if map_location is None:
        map_location = torch.device("cpu")
    checkpoint = torch.hub.load_state_dict_from_url(
        model_urls[line], map_location=map_location
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model

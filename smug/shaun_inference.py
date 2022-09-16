import os
from typing import Optional

import numpy as np
import torch
from crispy.utils import mosaic, mosaic_cube, segment_cube, segmentation

from .shaun_model import Shaun


class Corrector:
    """
    This is the object to correct for seeing in observations.

    Parameters
    ----------
    ckp : Dict, containing the torch checkpoint
        The data needed to reconstruct the Shaun model.
    norm : int
        The normalisation factor used during training (e.g. 1514 for CaII8542,
        and 3145 for Halpha).
    map_location : torch.device, optional
        Where to map the model for inference.
    in_channels : int
        The number of channels of the input images.
    out_channels : int
        The number of channels of the output images.
    nef : int
        The number of base feature maps used in the first convolutional layer.
    error : float, optional
        The error on the estimates from the network. Default is None which takes the last training error from the model file.
    """

    def __init__(
        self, ckp, norm, map_location, in_channels=1, out_channels=1, nef=64, error=None
    ):
        if map_location is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = map_location
        self.model = Shaun(
            in_channels=in_channels, out_channels=out_channels, nef=nef
        ).to(self.device)

        self.norm = norm

        print(f"loading Shaun model")
        self.model.load_state_dict(ckp["model_state_dict"])
        if error is None:
            self.error = ckp["losses"]["train_l"][-1]
        else:
            self.error = error
        print("=> model loaded.")

        self.model.eval()

    def correct_image(self, img):
        """
        This class method does the correction on the images.

        Parameters
        ----------
        img : numpy.ndarray
            The image to be corrected by the network.
        """
        with torch.no_grad():
            img_shape = img.shape[-2:]
            if img.ndim == 2:
                img = segmentation(img, n=256)
                img = torch.from_numpy(img).unsqueeze(1).float().to(self.device)
                out = self.model(img / self.norm).squeeze().cpu().numpy()

                return mosaic(out, img_shape, n=256) * self.norm
            elif img.ndim == 3:
                img = segment_cube(img, n=256)
                out = np.zeros_like(img)
                for j, im in enumerate(np.rollaxis(img, 1)):
                    im = torch.from_numpy(im).unsqueeze(1).float().to(self.device)
                    out[:, j] = self.model(im / self.norm).squeeze().cpu().numpy()

                return mosaic_cube(out, img_shape, n=256) * self.norm


class SpeedyCorrector(Corrector):
    """
    This class is to do the same corrections but using the torchscript model. This is ~4x faster but limits the batch size to 16.

    N.B. Currently the torchscript models are not provided as part of smug.

    Parameters
    ----------
    model_path : str
        The path to the torchscript model.
    error : float
        The error to apply to the estimates.
    """

    def __init__(self, model_path, error=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.error = error

        if "ca8542" in model_path.lower():
            self.norm = 1514
        else:
            self.norm = 3145

        self.model.eval()


pretrained_kwargs = {
    ("CaII8542", "1.0.0"): {
        "in_channels": 1,
        "out_channels": 1,
        "nef": 64,
        "norm": 1514,
    },
    ("Halpha", "1.0.0"): {
        "in_channels": 1,
        "out_channels": 1,
        "nef": 64,
        "norm": 3145,
    },
}

model_urls = {
    "CaII8542": "https://www.astro.gla.ac.uk/users/USER-MANAGED/solar_model_weights/Shaun_CaII8542_1.0.0.pth.tar",
    "Halpha": "https://www.astro.gla.ac.uk/users/USER-MANAGED/solar_model_weights/Shaun_Halpha_1.0.0.pth.tar",
}


def pretrained_shaun_corrector(
    line: str, version: str = "1.0.0", map_location: Optional[torch.device] = None
) -> Corrector:
    """Load a pretrained Shaun model in a corrector. The weights will be cached As described by `torch.hub`. See their documentation for details.

    Parameters
    ----------
    line : str
        The spectral line to load the weights for, "Halpha" or "CaII8542"
    version : str, optional
        The version of the model to load. Default: 1.0.0
    map_location : torch.device, optional
        Where to remap arrays during the loading process, by default this is set
        to "CPU" to allow loading on any platform.
    """
    accepted_lines = [k[0] for k in pretrained_kwargs]
    accepted_versions = [k[1] for k in pretrained_kwargs]
    if version not in accepted_versions:
        raise ValueError(
            f"Unknown version '{version}' requested for Shaun, expected one of {accepted_versions}"
        )
    if line not in accepted_lines:
        raise ValueError(
            f"Unknown line '{line}' requested for Shaun, expected one of {accepted_lines}"
        )
    if (line, version) not in pretrained_kwargs:
        raise ValueError(
            f"Unknown line, version combination'{(line, version)}' requested for Shaun, expected one of {list(pretrained_kwargs.keys())}"
        )

    if map_location is None:
        map_location = torch.device("cpu")
    checkpoint = torch.hub.load_state_dict_from_url(
        model_urls[line], map_location=map_location
    )
    return Corrector(
        checkpoint, map_location=map_location, **pretrained_kwargs[(line, version)]
    )

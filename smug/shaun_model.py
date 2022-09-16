import torch.nn as nn
from torch import tanh

from .shaun_blocks import ConvBlock, ConvTranspBlock, ResBlock


class Shaun(nn.Module):
    """
    This is the base class for the neural network model for correcting for seeing.

    Parameters
    ----------
    in_channels : int
        The number of channels that the images have.
    out_channels : int
        The number of channels that the output images have.
    nef : int
        The base number of feature maps to use for the first convolutional layer.
    """

    def __init__(self, in_channels, out_channels, nef):
        super(Shaun, self).__init__()

        self.C01 = ConvBlock(in_channels, nef, kernel=7, normal="instance")

        self.C11 = ConvBlock(nef, 2 * nef, stride=2, normal="instance")

        self.C21 = ConvBlock(2 * nef, 4 * nef, stride=2, normal="instance")

        self.R1 = ResBlock(4 * nef, 4 * nef, normal="instance")
        self.R2 = ResBlock(4 * nef, 4 * nef, normal="instance")
        self.R3 = ResBlock(4 * nef, 4 * nef, normal="instance")
        self.R4 = ResBlock(4 * nef, 4 * nef, normal="instance")
        self.R5 = ResBlock(4 * nef, 4 * nef, normal="instance")
        self.R6 = ResBlock(4 * nef, 4 * nef, normal="instance")
        self.R7 = ResBlock(4 * nef, 4 * nef, normal="instance")
        self.R8 = ResBlock(4 * nef, 4 * nef, normal="instance")
        self.R9 = ResBlock(4 * nef, 4 * nef, normal="instance")

        self.C31 = ConvTranspBlock(4 * nef, 2 * nef, stride=2, normal="instance")

        self.C41 = ConvTranspBlock(2 * nef, nef, stride=2, normal="instance")

        self.C51 = ConvBlock(nef, out_channels, kernel=7, normal="instance")

    def forward(self, inp):
        C01 = self.C01(inp)

        C11 = self.C11(C01)

        C21 = self.C21(C11)

        R1 = self.R1(C21)
        R2 = self.R2(R1)
        R3 = self.R3(R2)
        R4 = self.R4(R3)
        R5 = self.R5(R4)
        R6 = self.R6(R5)
        R7 = self.R7(R6)
        R8 = self.R8(R7)
        R9 = self.R9(R8)

        C31 = self.C31(R9)

        C41 = self.C41(C31)

        C51 = self.C51(C41)

        return tanh(C51) + inp

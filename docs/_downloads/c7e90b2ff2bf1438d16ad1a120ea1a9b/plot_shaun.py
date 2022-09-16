"""
====================
Simple Shaun Example
====================
A simple example showing how to apply Shaun to correct for atmospheric seeing in a region of an SST/CRISP image.
"""

import matplotlib.pyplot as plt
import numpy as np
from crispy import CRISP
from smug.shaun_inference import pretrained_shaun_corrector

#%%
# Load data, rotate it to fill the frame, and select sub-region to work on.
data = CRISP("crisp_l2_20140906_152724_6563_r00447_3wavelengths.fits")
data.rotate_crop()

crisp_sub_region = data[:, 200:500, 200:500]

#%%
# Construct Shaun model and apply to image.
corrector = pretrained_shaun_corrector("Halpha")
shaun_im = corrector.correct_image(crisp_sub_region.data)


#%%
# Compare line-core corrected image to one with original atmospheric seeing.
fig, ax = plt.subplots(1, 2)
ax[0].imshow(crisp_sub_region[1].data, cmap="Greys_r")
ax[0].set_title("Original Data")
ax[1].imshow(shaun_im[1], cmap="Greys_r")
ax[1].set_title("Shaun Corrected")

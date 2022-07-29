"""
=====================
Simple Limber Example
=====================
A simple example showing how to apply the Limber model to correct for Limb darkening in a small SST/CRISP image.
This data is loaded using `astropy.fits`, but crispy could be used too.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from smug.limber_adapter import LimberAdapter
from smug.limber_model import model_params, pretrained_limber

#%%
# Load the Ca 8542 Angstrom data
im = fits.open("../tests/mini_crisp_l2_20140906_152724_8542_r00459.fits")


#%%
# Load pretrained limber network
line = "CaII8542"
model = pretrained_limber(line)

#%%
# Compute wavelength grid used in Limber model from provided data, and construct Adapter.
grid = np.linspace(
    -model_params[line]["half_width"],
    model_params[line]["half_width"],
    model.size - 1,
)
limber_ca = LimberAdapter(model, grid)

#%%
# Load wavelengths present in data file.
central_wavelength = np.median(im[1].data)
data_wavelength = im[1].data - central_wavelength

#%%
# Run the network to reproject the data
out = limber_ca.reproject_data(
    im[0].data.astype("<f4"),
    data_wavelength,
    mu_observed=0.565,
    reconstruct_original_shape=False,
)

#%%
# Plot the output for a pixel, note the swapped indexing as we set `reconstruct_original_shape` to False.
idx = 8
a = im[0].data.astype("<f4")
b = out
plt.plot(data_wavelength, a[:, idx, idx], label=r"$\mu=0.565$")
plt.plot(grid, b[idx, idx, 1:], label=r"$\mu=1.0$ (predicted)")
plt.xlabel(r"$\Delta\lambda$ [$\AA$]")
plt.ylabel("Intensity [DN]")
plt.title("Limber applied to Ca ɪɪ 8542 $\AA$")
plt.legend()
plt.show()

"""
===========================
Simple Radynversion Example
===========================

The following illustrates configuring and using Radynversion with some small SST/CRISP images.
To load this data we use the `crispy package <https://github.com/bionictoucan/crispy>`_.
"""
from crispy import CRISPSequence
from crispy.utils import CRISP_sequence_constructor
from matplotlib import pyplot as plt
from smug.radynversion_adapter import RadynversionAdapter
from smug.radynversion_model import model_params, pretrained_radynversion

#%%
# Plot the Halpha intensity map (see crispy's documentation for how this is used.).
files = [
    "../tests/mini_crisp_l2_20140906_152724_6563_r00459.fits",
    "../tests/mini_crisp_l2_20140906_152724_8542_r00459.fits",
]
ims = CRISPSequence(CRISP_sequence_constructor(files))
ims.list[0][7].intensity_map()
plt.show()

#%%
# Configure model, and Adapter class containing utilities to run the model and
# correctly transform the parameters.

radynversion_version = "1.1.1"
ad = RadynversionAdapter(
    model=pretrained_radynversion(version=radynversion_version),
    **model_params[radynversion_version]
)

#%%
# Run the inversion for a small slice of the data.
inv = ad.invert_dual_cubes(ims[:, :1, :3])


#%%
# Display the inversion results for a pixel
inv[:, 0, 1].plot_params(eb=True)
plt.show()

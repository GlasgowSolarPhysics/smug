## SMUG: Solar Models from the University of Glasgow

SMUG is intended to be a simple, easy to install and use, repository for the University of Glasgow solar deep learning models.
Namely, [RADYNVERSION](https://github.com/Goobley/Radynversion), Limber, and [Shaun](https://github.com/bionictoucan/shaun).
SMUG provides easy access to the pretrained models, which will be automatically downloaded when requested, as well as adapter classes to simplify their use.

### Models

The models in this package are primarily based on the PhDs of [John Armstrong](https://theses.gla.ac.uk/82866/) and [Chris Osborne](https://theses.gla.ac.uk/82584/).

- RADYNVERSION: An invertible neural network for inferring RADYN-like atmospheres from spectroscopic flare data, currently requiring data in the Hα and Ca ɪɪ 854.2 nm spectral lines. The invertibility and latent space of the model allows for determining approximate confidence intervals on the inferred atmospheric parameters, outside the constraints of statistical equilibrium (very important for the hydrogen lines). Single inversion samples are many orders of magnitude faster than those produced by synthesis-and-regression based inversion techniques.

- Limber: Convolutional autoencoder for determining the associated emission from an atmosphere when viewed at μ=1, given a limb-darkened observation at a different μ. Trained from RADYN models for Hα and Ca ɪɪ 854.2 nm, and assumes a plane-parallel geometry. A good preprocessing step for heavily inclined inversions using RADYNVERSION.

- Shaun: The Seeing AUtoeNcoder. A fully-convolutional autoencoder for correcting for atmospheric seeing in ground-based spectroscopic solar observations. Designed for use on flaring observations.

### Examples and Documentation

See https://GlasgowSolarPhysics.github.io/

### Installation

(Coming soon)
```bash
python -m pip install smug
```

or install from the repository with

```bash
python -m pip install .
```

### Tests

Tests are automatically run via GitHub actions but can be run locally by cloning the repository and running `pytest` in the root directory.

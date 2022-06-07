from setuptools import setup, find_packages

setup(
    name="smug",
    setup_requires=["setuptools_scm"],
    use_scm_version=True,
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "torch>=1.4",
        "weno4",
        "astropy",
        "FrEIA @ git+https://github.com/VLL-HD/FrEIA.git",
        "sst-crispy @ git+https://github.com/Goobley/crispy.git",
    ],
)

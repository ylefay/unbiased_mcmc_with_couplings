import sys
from setuptools import setup, find_packages

# READ README.md for long description on PyPi.
try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write(f"Failed to read README.md:\n  {e}\n")
    sys.stderr.flush()
    long_description = ""

setup(
    name="pymcmc_unbiased",
    author="Thomas Michel, Matthieu Dinot, Yvann Le Fay",
    description="Implementation of unbiased MCMC estimators using coupling (Pierre E. Jacob, John O'Leary, Yves F. Atchadé, 2019)",
    long_description=long_description,
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pytest",
        "numpy",
        "jax",
    ],
    long_description_content_type="text/markdown",
    keywords="mcmc gibbs markov chain monte carlo sampling coupling",
    license="MIT",
    license_files=("LICENSE",),
)

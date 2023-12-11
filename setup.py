from setuptools import setup
import re
import os

PACKNAME = 'snsim'
AUTHOR = 'Bastien Carreres'
EMAIL = 'carreres@cppm.in2p3.fr'
URL = 'https://github.com/bcarreres/snsim'
LICENSE = 'BSD'
DESCRIPTION = 'Package to simulate SN survey, using sncosmo SN simulation package'
VERSION = re.findall(r"__version__ = \"(.*?)\"",
                     open(os.path.join("snsim", "__init__.py")).read())[0]
setup(
    name=PACKNAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    license=LICENSE,
    packages=['snsim'],
    url=URL,
    python_requires='>=3.7',
    install_requires=[
        "pandas >= 1.2.0",
        "sncosmo >= 2.5.0",
        "numpy >= 1.13.3",
        "astropy >= 5.1.0",
        "shapely >= 1.8.0",
        "numba",
        "pyyaml",
        "sfdmap2",
        "requests",
        "geopandas"
    ]
)

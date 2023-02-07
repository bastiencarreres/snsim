from setuptools import setup
import re
import os 

version=re.findall(r"__version__ = \"(.*?)\"",
            open(os.path.join("snsim", "__init__.py")).read())[0]

if __name__ == "__main__":
    setup(version=version)

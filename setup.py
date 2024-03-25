from setuptools import setup
import re
import os

__init__path = os.path.join("snsim", "__init__.py")
version = re.findall(r"__version__ = \"(.*?)\"", open(__init__path).read())[0]

if __name__ == "__main__":
    setup(version=version)

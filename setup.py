from setuptools import setup

setup(
   name='snsim',
   version='0.1',
   author='B. Carreres',
   author_email='carreres@cppm.in2p3.fr',
   packages=['snsim'],
   url='http://github.com/bcarreres/SNSim',
   license='NONE',
   description='A package to simulate SN Ia',
   install_requires=[
       "sncosmo >= 2.5.0",
       "numpy >= 1.13.3",
       "scipy >= 0.9.0",
       "astropy >= 1.0.0" ]
)

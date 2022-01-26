# Getting started 

## Basic usage

The code is launched in a python interpreter by calling the Simulator object :

```python
from snsim import Simulator

# Initialisation
sim = Simulator('yaml_cfg_file.yml')

# Simulation
sim.simulate()
```

The result is stored in sim.sn_sample object . Simulated lc and metadata are given by :
```python
sim.sn_sample.sim_lcs.loc[i] # sn_sample.sim_lcs is a pandas.DataFrame object
sim.sn_sample.sim_lcs.meta[i] # metadata are stored in a dict in sn_sample.sim_lcs.attrs but there is a shortcut sim_lcs.meta

#  For more information :
help(sim.sn_sample)
```
The basic list of ligth-curves metadata is given in the following table :

|       zobs        |  sim_t0  |    vpec (km/s)    |         zcos          |            zpec            |                       z2cmb                       |        zCMB        |      ra (rad)      |   dec (rad)    |          sn id           |           sim_mu           |          m_sct           |
| :---------------: | :------: | :---------------: | :-------------------: | :------------------------: | :-----------------------------------------------: | :----------------: | :----------------: | :------------: | :----------------------: | :------------------------: | :----------------------: |
| Observed redshift | Peaktime | Peculiar velocity | Cosmological redshift | Peculiar velocity redshift | Contribution from our peculiar motion to redshift | CMB frame redshift | SN right ascension | SN declinaison | SN identification number | Simulated distance modulli | Coherent scattering term |

If you use SALT2/3 model you add some arguments to metadata:


|         sim_x0          |      sim_x1       |      sim_c      |               sim_mb                |
| :---------------------: | :---------------: | :-------------: | :---------------------------------: |
| Normalization parameter | Stretch parameter | color parameter | SN magnitude in restframe Bessell B |

Moreover, if you use a scattering model like G10 or C11 the random seed used is kept in the meta too.
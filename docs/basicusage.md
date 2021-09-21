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

The result is stored in sim.sn_list list which each entry is a SN object. Simulated lc and metadata are given by :
```python
sim.sn_list[i].sim_lc
sim.sn_list[i].sim_lc.meta

#  For more information :
help(snsim.SN)
```
The basic list of ligthcurves metadata is given in the following table :

|       zobs        |  sim_t0  |    vpec (km/s)    |         zcos          |            zpec            |                       z2cmb                       |        zCMB        |      ra (rad)      |   dec (rad)    |          sn id           |           sim_mu           |          m_sct           |
| :---------------: | :------: | :---------------: | :-------------------: | :------------------------: | :-----------------------------------------------: | :----------------: | :----------------: | :------------: | :----------------------: | :------------------------: | :----------------------: |
| Observed redshift | Peaktime | Peculiar velocity | Cosmological redshift | Peculiar velocity redshift | Contribution from our peculiar motion to redshift | CMB frame redshift | SN right ascension | SN declinaison | SN identification number | Simulated distance modulli | Coherent scattering term |

If you use SALT2/3 model you add some arguments to metadata:


|         sim_x0          |      sim_x1       |      sim_c      |               sim_mb                |
| :---------------------: | :---------------: | :-------------: | :---------------------------------: |
| Normalization parameter | Stretch parameter | color parameter | SN magnitude in restframe Bessell B |

Moreover, if you use a scattering model like G10 or C11 the random seed used is kept in the meta too.



## Script launch

The program can be launch with the ./sripts/launch_sim.py python script.

The script use argparse to change parameters:
```shell
>python3 launch_sim.py '/PATH/TO/YAMLFILE' -fit (optional if you want to fit) --any_config_key value (overwrite yaml configuration or add param)
```
If the config keys is a float or an int just type as :
```shell
>python3 launch_sim.py '/PATH/TO/YAMLFILE' --int_or_float_key value_nbr
```
If the config key is a dict you have to pass it like a yaml string :
```shell
>python3 launch_sim.py '/PATH/TO/YAMLFILE' --dic_key "{'key1': value1, 'key2': value2, ...}"
```
If the config key is a list you have to pass it by separate item by space :
```shell
>python3 launch_sim.py '/PATH/TO/YAMLFILE' --list_key item1 item2 item3
```
In the case of **nep_cut** key you can pass an int or pass list by typing --nep_cut multiple times, note that filter argument is optional:
```shell
#nep_cut is just an int
>python3 launch_sim.py '/PATH/TO/YAMLFILE' --nep_cut minimal_nbr_of_epoch

#Multiple cuts
>python3 launch_sim.py '/PATH/TO/YAMLFILE' --nep_cut ep_nbr1 time_inf1 time_sup1 optional_filter1 --nep_cut ep_nbr2 time_inf2 time_sup2 optional_filter2
```
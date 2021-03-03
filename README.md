# Code for simulate sn with sn cosmo
The code is in dev

## Input file :
The input file example is given by config.yml
```
data:
   obs_config_path: 'PATH/TO/OBS_FILE'
   write_path: 'PATH/TO/OUTPUT'
sn_gen:
    bands: ['band1_name','band2_name',...]
    n_sn: SN_NBR
    randseed: RANDSEED (optional)
    z_range: [z_min,z_max]
    v_cmb: VCMB (optionnal -> default = 369.82 km/s)
    M0: SN_Ia absmag
cosmology:
    Om: Omega_matter
    H0: Hubble constant
salt2_gen:
    alpha: alpha parameter of Tripp relation
    beta: beta parameter of Tripp relation
    mean_x1: x1 central value
    mean_c: c central value
    sig_x1: x1 sigma
    sig_c: c sigma
vpec_gen:
    mean_vpec: vpec central value
    sig_vpec: vpec sigma

```
## Obs file:
The obs file is in fits format and is generated with gen_obs.py
gen_obs function :
```
gen_obs(n_obs,n_epochs_b,bands,mean_depth,mjdstart,ra_list,dec_list,magsys='ab',gain=1.000)
```

# Code for simulation of SN Ia using sn cosmo
## Installation
In the setup.py directory use:
```
>python -m pip setup .
```

## Script launch
Use launch_sim.py with argparse:
```
>python3 launch_sim.py '/PATH/TO/YAMLFILE' -fit (optional if you want to fit) --any_config_keys=value (overwrite yaml configuration or add param)
```

## Input file :
The input file is a .yml with the following structure:
```
data :
    write_path: '/PATH/TO/OUTPUT'
    sim_name: 'NAME OF SIMULATION'
    band_dic: {'r':'ztfr','g':'ztfg','i':'ztfi'} #(Optional -> if bandname in db/obs file doesn't
 correpond to those in sncosmo registery)
    obs_config_path: '/PATH/TO/OBS/FILE' #(Optional -> use db_file)
    write_format: 'format' or ['format1','format2'] # Optional default pkl, fits
db_config: #(Optional -> use obs_file)
    dbfile_path: '/PATH/TO/FILE'
    db_cut: {'key1': ['conditon1','conditon2',...], 'key2':['conditon1'],...}
    zp: INSTRUMENTAL ZEROPOINT  
    ra_size: RA FIELD SIZE in DEG
    dec_size: DEC FIELD SIZE in DEG
    gain: CCD GAIN e-/ADU
sn_gen:
    n_sn: NUMBER OF SN TO GENERATE #(Optional)
    sn_rate: rate of SN/Mpc^3/year #(Optional, default=3e-5)
    rate_pw: rate = sn_rate*(1+z)^rate_pw (Optional, default=0)
    randseed: RANDSEED TO REPRODUCE SIMULATION #(Optional)
    z_range: [ZMIN,ZMAX]
    v_cmb: OUR PECULIAR VELOCITY #(Optional, default = 369.82 km/s)
    M0: SN ABSOLUT MAGNITUDE
    mag_smear: SN INTRINSIC SMEARING
    smear_mod: 'G10','C11_i' USE WAVELENGHT DEP MODEL FOR SN INT SCATTERING
cosmology:
    Om: MATTER DENSITY  
    H0: HUBBLE CONSTANT
salt_gen:
    version: 2 or 3
    salt_dir: '/PATH/TO/SALT/MODEL'  
    alpha: STRETCH CORRECTION = alpha*x1
    beta: COLOR CORRECTION = -beta*c   
    mean_x1: MEAN X1 VALUE
    mean_c: MEAN C VALUE
    sig_x1: SIGMA X1   
    sig_c: SIGMA C
 vpec_gen:
     host_file: '/PATH/TO/HOSTFILE'
     mean_vpec: MEAN SN PECULIAR VEL
     sig_vpec: SIGMA VPEC
```

* obs_file and db_file are optional but you must set one of the two!!!
* If the name of bands in the obs/db file doesn't match sncosmo bands you can use the key band_dic to translate filters names
* If you don't set the filter name item in nep_cut, the cut apply to all the band
* For wavelength dependent model, nomanclature follow arXiv:1209.2482 -> 'G10' for Guy et al. 2010 model, 'C11' or 'C11_0' for Chotard et al. model with correlation between U' and U = 0, 'C11_1' for Cor(U',U) = 1 and 'C11_2' for Cor(U',U) = -1

## Observation DataBase file:
It's a sql database file which contain cadence information. It's used to find obs epoch and their noise.

The required data keys are resumed in the next table

| expMJD | filter | fieldRA (rad) | fieldDec (rad) | fiveSigmaDepth |
| :-----------: | :-----: | :----------: | :----------: | :--------------------: |
| Obs time| Obs band | Right ascension of the obs field| Declinaison of the obs field   |  Limiting magnitude at 5 sigma |


## Obs file: (No longer usable)
The obs file is in fits format and is generated with gen_obs.py
gen_obs function :
```
gen_obs(n_obs,n_epochs_b,bands,mean_depth,mjdstart,ra_list,dec_list,magsys='ab',gain=1.000)
```

## Usage and output


```
from snsim import sn_sim

sim = sn_sim('yaml_cfg_file.yml')
sim.simulate()
```

The result is stored in sim.sim_lc table which each entry is a SN light curve. Metadata are given by
```
sim.sim_lc[i].meta
```
The list of ligth curves metadata is given in the following table

| z | t0 | x0 | x1 | c | vpec (km/s) | zcos | zpec | z2cmb | zCMB | ra (rad) | dec (rad) |  sn id  | mb | mu | msmear |
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
|  Observed redshift | Peaktime | SALT2 x0 (normalisation) parameter  | SALT2 x1 (stretch) parameter  | SALT2 c (color) parameter | Peculiar velocity  | Cosmological redshift  | Peculiar velocity redshift | CMB motion redshift | CMB frame redshift | SN right ascension   |  SN declinaison |  SN identification number | SN magnitude in restframe Bessell B | Simulated distance modulli | Coherent smear term |

# Code for simulate sn with sn cosmo
Use main.py with argparse:
```
>python3 main.py '/PATH/TO/YAMLFILE' -fit (optional if you want to fit) --any_config_keys=value (overwrite yaml configuration or add param)
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
db_config: #(Optional -> use obs_file)
    dbfile_path: '/PATH/TO/FILE'
    db_cut: {'key1': ['conditon1','conditon2',...], 'key2':['conditon1'],...}
    zp: INSTRUMENTAL ZEROPOINT  
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
cosmology:
    Om: MATTER DENSITY  
    H0: HUBBLE CONSTANT
salt2_gen:
    salt2_dir: '/PATH/TO/SALT2/MODEL'  
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

## bd file:
It's a sql database file which contain cadence information. It's used to find obs epoch and their noise.

## Obs file: (No longer usable)
The obs file is in fits format and is generated with gen_obs.py
gen_obs function :
```
gen_obs(n_obs,n_epochs_b,bands,mean_depth,mjdstart,ra_list,dec_list,magsys='ab',gain=1.000)
```

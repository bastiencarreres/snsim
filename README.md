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

NOTE : 
       - obs_file and db_file are optional but you must set one of the two!!!    
       - If the name of bands in the obs/db file doesn't match sncosmo bands
    you can use the key band_dic to translate filters names
       - If you don't set the filter name item in nep_cut, the cut apply to all the bands
       - For wavelength dependent model, nomanclature follow arXiv:1209.2482 -> Possibility are  
    'G10' for Guy et al. 2010 model, 'C11' or 'C11_0' for Chotard et al. model with correlation
    between U' and U = 0, 'C11_1' for Cor(U',U) = 1 and 'C11_2' for Cor(U',U) = -1

## Observation DataBase file:
It's a sql database file which contain cadence information. It's used to find obs epoch and their noise.

## Obs file: (No longer usable)
The obs file is in fits format and is generated with gen_obs.py
gen_obs function :
```
gen_obs(n_obs,n_epochs_b,bands,mean_depth,mjdstart,ra_list,dec_list,magsys='ab',gain=1.000)
```

# Configuration input yaml file

The input file is a .yml with the following structure:

```yaml
data:
    write_path: '/PATH/TO/OUTPUT'
    sim_name: 'NAME OF SIMULATION'
    write_format: 'format' or ['format1','format2'] #(Optional default pkl, fits)
survey_config:
    survey_file: '/PATH/TO/FILE'
    band_dic: {'r':'ztfr','g':'ztfg','i':'ztfi'} #(Optional -> if bandname in the database doesn't correpond to those in sncosmo registery)
    add_data: ['keys1', 'keys2', ...] #(Optional add survey file keys to metadata)
    db_cut: {'key1': ["conditon1","conditon2",...], 'key2':["conditon1"],...} #(Optional SQL conditions on key)
    zp: INSTRUMENTAL ZEROPOINT  #(Optional, default given by survey file)
    sig_zp: UNCERTAINTY ON ZEROPOINT #(Optional, default given by survey file)
    sig_psf: GAUSSIAN PSF SIGMA #(Optional, default given by survey file as FWHMeff)
    noise_key: [key, type] type can be 'mlim5' or 'skysigADU'                      
    ra_size: RA FIELD SIZE in DEG
    dec_size: DEC FIELD SIZE in DEG
    gain: CCD GAIN e-/ADU
    start_day: MJD NUMBER or 'YYYY-MM-DD' #(Optional, default given by survey file)
    end_day: MJD NUMBER or 'YYYY-MM-DD' #(Optional, default given by survey file)
    duration: SURVEY DURATION (DAYS) #(Optional, default given by survey file)
    sub_field: ['sub_field_file', 'sub_field_key'] # Used to divided observation in CCD quadrant for example
sn_gen:
    n_sn: NUMBER OF SN TO GENERATE #(Optional)
    duration_for_rate: FAKE DURATION ONLY USED TO GENERATE N SN (DAYS) #(Optional)
    sn_rate: rate of SN/Mpc^3/year #(Optional, default=3e-5)
    rate_pw: rate = sn_rate*(1+z)^rate_pw #(Optional, default=0)
    nep_cut: [[nep_min1,Tmin,Tmax],[nep_min2,Tmin2,Tmax2,'filter1'],...] EP CUTS #(Optional defaut >= 1 ep)
    randseed: RANDSEED TO REPRODUCE SIMULATION #(Optional default random)
    z_range: [ZMIN, ZMAX] # Cosmological redshift range
    M0: SN ABSOLUT MAGNITUDE
    mag_sct: SN INTRINSIC COHERENT SCATTERING
    sct_mod: 'G10','C11_i' USE WAVELENGHT DEP MODEL FOR SN INT SCATTERING
cosmology: # Follow astropy formalism
    Om0: MATTER DENSITY  
    H0: HUBBLE CONSTANT
cmb:
    v_cmb: OUR PECULIAR VELOCITY #(Optional, default = 369.82 km/s)
    l_cmb: GAL L OF CMB DIPOLE #(Optional, default = 271.0)            
    b_cmb: GAL B OF CMB DIPOLE #(Optional, default = 29.6)     
model_config:
    model_name: 'THE MODEL NAME' # Example : 'salt2'
    model_dir: '/PATH/TO/SALT/MODEL'  
    # Model parameters (here example for salt2)
    alpha: STRETCH CORRECTION = alpha*x1
    beta: COLOR CORRECTION = -beta*c   
    mean_x1: MEAN X1 VALUE
    mean_c: MEAN C VALUE
    sig_x1: SIGMA X1 or [SIGMA_X1_LOW, SIGMA_X1_HIGH]  
    sig_c: SIGMA C or [SIGMA_C_LOW, SIGMA_C_HIGH]
    mw_dust: MODEL_NAME #(RV = 3.1) or [MOD_NAME, RV]  #(Optional)
 vpec_dist:
     mean_vpec: MEAN SN PECULIAR VEL
     sig_vpec: SIGMA VPEC
 host_file: '/PATH/TO/HOSTFILE' # Optional
 alpha_dipole: # Experimental alpha fine structure constant dipole, optional
     coord: [RA, Dec] # Direction of the dipole
     A: A_parameter # alpha dipole = A + B * cos(theta)
     B: B_parameter  
```

  * If you set end_day and duration, duration will be ignored
  *  If the name of bands in the db file doesn't match sncosmo bands you can use the key band_dic to translate filters names
  * If you use sub_field you have to define a representation of your sub_field in a txt file.
  * If you don't set the filter name item in nep_cut, the cut apply to all the band
  * For wavelength dependent model, nomanclature follow arXiv:1209.2482 -> 'G10' for Guy et al. 2010 model, 'C11' or 'C11_0' for Chotard et al. model with correlation between U' and U = 0, 'C11_1' for Cor(U',U) = 1 and 'C11_2' for Cor(U',U) = -1
  * Note that the FWHMeff in survey file follow LSST OpSim format and is equal to 2 * sqrt(2 * ln(2)) * sig_psf
  * mw_dust available models are CCM89, OD94 and F99 (cf sncosmo documentation)


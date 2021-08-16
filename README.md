# Code for simulation of SN Ia using sn cosmo
## Installation
In the setup.py directory use:
```
>python -m pip setup .
```
## Input file :
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
* If the name of bands in the db file doesn't match sncosmo bands you can use the key band_dic to translate filters names
* If you use sub_field you have to define a representation of your sub_field in a txt file.
* If you don't set the filter name item in nep_cut, the cut apply to all the band
* For wavelength dependent model, nomanclature follow arXiv:1209.2482 -> 'G10' for Guy et al. 2010 model, 'C11' or 'C11_0' for Chotard et al. model with correlation between U' and U = 0, 'C11_1' for Cor(U',U) = 1 and 'C11_2' for Cor(U',U) = -1
* Note that the FWHMeff in survey file follow LSST OpSim format and is equal to 2 * sqrt(2 * ln(2)) * sig_psf
* mw_dust available models are CCM89, OD94 and F99 (cf sncosmo documentation)

## Observation DataBase file:
It's a sql database file which contain cadence information. It's used to find obs epoch and their noise.

The required data keys are resumed in the next table

|     expMJD     | filter   | fieldID             | fieldRA (rad)                   |  fieldDec (rad)              | zp                            | sig_zp |
| :-----------:  | :-----:  | :-----------------: | :-----------------------------: | :--------------------------: | :---------------------------: | :------: |
| Obs time in MJD| Obs band | The ID of the field | Right ascension of the obs field| Declinaison of the obs field | Zero point of the observation (Optional if given in yaml) | Uncertainty of the zeropoint (Optional if given in yaml) |

If you want to use subfield index for observation properties and not a general you have to give a .dat file that give the representation of the subfield, for example if you split your field into a 4 x 4 grid, you have to put something like that in your .dat file :

```pseudocode
ID01:ID02:ID03:ID04
ID05:ID06:ID07:ID08
ID09:ID10:ID11:ID12
ID13:ID14:ID15:ID16
```
If a sub field is not observed you should set the ID value to -1.

You can show the sub filed map by :

```python
sim.survey.fields.show_map()

```

The result is something as :

```asciiarmor
  3.50 ----------------------------------------------------------
        |  61  |  60  |  57  |  56  |  53  |  52  |  49  |  48  |
  2.63 ----------------------------------------------------------
        |  62  |  63  |  58  |  59  |  54  |  55  |  50  |  51  |
  1.75 ----------------------------------------------------------
        |  45  |  44  |  41  |  40  |  37  |  36  |  33  |  32  |
  0.87 ----------------------------------------------------------
        |  46  |  47  |  42  |  43  |  38  |  39  |  34  |  35  |
  0.00 ----------------------------------------------------------
        |  29  |  28  |  25  |  24  |  21  |  20  |  17  |  16  |
 -0.88 ----------------------------------------------------------
        |  30  |  31  |  26  |  27  |  22  |  23  |  18  |  19  |
 -1.75 ----------------------------------------------------------
        |  13  |  12  |  09  |  08  |  05  |  04  |  01  |  00  |
 -2.62 ----------------------------------------------------------
        |  14  |  15  |  10  |  11  |  06  |  07  |  02  |  03  |
 -3.50 ----------------------------------------------------------
        '      '      '      '      '      '      '      '      '
      -3.50  -2.62  -1.75  -0.88   0.00   0.87   1.75   2.63   3.50
```

Where the graduations in degrees depend on your field size.

## Host file

The host file contain coordinates and peculiar velocities to simulate SN, the needed keys are given in the next table

| redshift | ra (rad) | dec (rad) | vp_sight (km/s) |
| :-----------: | :-----: | :----------: | :----------: |
| Redshift of the host | Right ascension of the host | Declinaison of the host | Velocity along the line of sight |

## Usage and output
```python
from snsim import Simulator

sim = Simulator('yaml_cfg_file.yml')
sim.simulate()
```

The result is stored in sim.sn_list list which each entry is a SN object. Simulated lc and metadata are given by :
```python
sim.sn_list[i].sim_lc
sim.sn_list[i].sim_lc.meta

#  For more information :
help(snsim.SN)
```
The basic list of ligth curves metadata is given in the following table :

| z |  sim_t0   | vpec (km/s) | zcos | zpec | z2cmb | zCMB | ra (rad) | dec (rad) |  sn id   | sim_mu | m_smear |
| :------------:  | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
|  Observed redshift | Peaktime | Peculiar velocity  | Cosmological redshift  | Peculiar velocity redshift | Contribution from our peculiar motion to redshift | CMB frame redshift | SN right ascension   |  SN declinaison |  SN identification number | Simulated distance modulli | Coherent smear term |

If you use SALT2/3 model you add some arguments to metadata:


|         sim_x0          |      sim_x1       |      sim_c      |               sim_mb                |
| :---------------------: | :---------------: | :-------------: | :---------------------------------: |
| Normalization parameter | Stretch parameter | color parameter | SN magnitude in restframe Bessell B |

Moreover, if you use a scattering model like G10 or C11 the random seed used is stock in the meta too.

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
If the config keys is a list you have to pass it by separate item by space :
```shell
>python3 launch_sim.py '/PATH/TO/YAMLFILE' --list_key item1 item2 item3
```
In the case of nep_cut key you can pass an int or pass list by typing --nep_cut multiple times, note that filter argument is optional:
```shell
#nep_cut is just an int
>python3 launch_sim.py '/PATH/TO/YAMLFILE' --nep_cut minimal_nbr_of_epoch

#Multiple cuts
>python3 launch_sim.py '/PATH/TO/YAMLFILE' --nep_cut ep_nbr1 time_inf1 time_sup1 optional_filter1 --nep_cut ep_nbr2 time_inf2 time_sup2 optional_filter2
```
## Plot functions  

You can plot simulated lightcurves

```
sim.plot_lc(SN_ID, mag=False, zp=25. , plot_sim=True, plot_fit=False, Jy=False)
```

Plot sim :

![](readme_figures/lc_sim.png)

Plot  fit :

![](readme_figures/lc_fit.png)

Plot sim and fit:

![](readme_figures/lc_sim_fit.png)

Just the data alone :

![lc_data](readme_figures/lc_data.png)

The same plot can be show in magnitude :

![lc_mag](readme_figures/lc_mag.png)

Or in Jansky :

![lc_jy](readme_figures/lc_jy.png)

You can also plot a vpec Mollweide map

```python
sim.plot_ra_dec(plot_vpec=False, plot_fields=False, **kwarg)
```

Plot without peculiar velocities :

![](readme_figures/ra_dec_map.png)

Plot with peculiar velocities :

![](readme_figures/ra_dec_map_vpec.png)

Adding the fields :

![](readme_figures/ra_dec_fields.png)

## Fit and OpenSim class

You can direct fit after running the simulation
```python
# Fit 1 lc by id
sim.fit_lc(id)

# Fit all the lcs
sim.fit_lc()

# Write the fit
sim.write_fit()
```
Or you can open register open sim file .fits or .pkl with the open_sim class
```python
from snsim import OpenSim

sim = OpenSim('sim_file.pkl/.fits',SALT2_dir)
```

## Simulation formula

The flux in ADU is simulated following the formula :

![flux_eq](readme_figures/flux_eq.svg)

Where the magnitude in rest-frame Bessell B band m<sub>B</sub> is given by :

![](readme_figures/mb_eq.svg)

With M<sub>B</sub> the absolute magnitude of SN Ia in the Bessell B band.

The distance moduli is computed with the peculiar velocity effect :

![](readme_figures/mu_eq.svg)

Where z<sub>cosmo</sub> is the cosmological redshift, z<sub>2cmb</sub> is the redshift from our peculiar motion to the CMB frame and z<sub>vp</sub> is the redshift due to the peculiar velocity of the object.



The error is computed with the formula :

![noise_eq](readme_figures/noise_eq.svg)

Where the first term is the poissonian random noise with G the CCD gain in e<sup>-</sup> / ADU, the second term is the skynoise.

If you use limiting magnitude at 5σ, skynoise is computed as :

![skynoise_eq](readme_figures/skynoise_eq.svg)

The last term is the propagation to flux of the uncertainty on zero point determination.

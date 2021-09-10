# Code for simulation of SN Ia using sn cosmo
## Installation
In the setup.py directory use:
```
>python -m pip setup .
```

## Documentation

 The documentation is [here](https://snsim.readthedocs.io/en/main/)

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

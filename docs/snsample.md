# SNSimSample class

The SNSimSample class store simulated lightcurves.



## Fitting lightcurves

*snsim* allow to fit lightcurves :

You can direct fit after running the simulation
```python
# Fit 1 lc by id
sim.sn_sample.fit_lc(id)

# Fit all the lcs
sim.sn_sample.fit_lc()

# Write the fit
sim.sn_sample.write_fit()
```


Or you can open register sim file .fits or .pkl :

```python
from snsim import SNSimSample

sim = SNSimSample.fromFile('sim_file.pkl/.fits', SALT2_dir)

# Fit all the lcs
sim.fit_lc()

# Write the fit
sim.write_fit()

# You can acces the lcs :
sim.sim_lcs()

# Or access to parameters list:
sim.sn.get('key')  # Where 'key' is a sn parameters such as 'sim_mb', 'ra', etc... 
```

The output file of write_fit is in the same directory as the simulation and has the same name + '_fit.fits'



## SNR selection

The simulation ignored effect of selection efficiency. To introduce this effect the SNSimSample class has a SNR_select.

At this time the SNR_select function use an approximation to model SNR detection probability:


$$
P_\text{det}(SNR) = \frac{1}{1+\left(\frac{SNR_\text{mean}}{SNR}\right)^n}
$$


where $SNR_\text{mean}$ is the SNR for which $P_\text{det} = 0.5$ and n is given by a probability of detection **p** for a given **$SNR_p$** : 


$$
n = \frac{\ln\left(\frac{1 - p}{p}\right)}{\ln(SNR_\text{mean}) - \ln(SNR_p)}
$$


The function can be used as :

```python
# By default SNR_mean = 5 and P(SNR=15) = 0.99
SNSimSample.SNR_select(selec_function, SNR_mean=5, SNR_limit=[15, 0.99], randseed=np.random.randint(1000, 100000))
```



The new sample of sn can be saved with :

```python
SNSimSample.write_select(formats=['pkl', 'fits'])
```


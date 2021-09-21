# Fit and SNSimSample class

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


Or you can open register open sim file .fits or .pkl with the open_sim class

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


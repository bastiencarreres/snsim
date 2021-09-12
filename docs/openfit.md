# Fit and OpenSim class

*snsim* allow to fit lightcurves :

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

# Fit all the lcs
sim.fit_lc()

# Write the fit
sim.write_fit()
```



The output file of write_fit is in the same directory as the simulation and has the same name + '_fit.fits'

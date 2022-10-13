SNSimSample class
=================

The SNSimSample class store simulated lightcurves.

Fitting lightcurves
-------------------

*snsim* allow to fit lightcurves :

You can direct fit after running the simulation

.. code:: python

   # Fit 1 lc by id
   sim.samples[i].fit_lc(id)

   # Fit all the lcs
   sim.samples[i].fit_lc()

   # Write the fit
   sim.samples[i].write_fit()

Or you can open register sim file .fits or .pkl :

.. code:: python

   from snsim import SimSample

   sim = SimSample.fromFile('sim_file.pkl/.parquet', model_dir=None)

   # Fit all the lcs
   sim.set_fit_model(model, model_dir=None, mw_dust=None)
   sim.fit_lc()

   # Write the fit
   sim.write_fit(write_path=None)

   # You can acces the lcs :
   sim.sim_lcs # pandas.DataFrame object 

   # Or access to parameters list:
   sim.get('key')  # Where 'key' is a sn parameters such as 'sim_mb', 'ra', etc... 

The output file of write_fit is in the same directory as the simulation
and has the same name + ’_fit.fits’

You can pass
`sncosmo.fit_lc() <https://sncosmo.readthedocs.io/en/stable/api/sncosmo.fit_lc.html?highlight=fit_lc#sncosmo.fit_lc>`__
arguments to the **SNSimSample.fit_lc()** function, the only no
modifiable arguments are **data**, **model** and **vparam_names**.

Modified lcs
~~~~~~~~~~~~

**SNSimSample** as a **modifed_lcs** attribute : this a copy of sim lcs
that you can modified as you want, using selection function, and then
write as a new sim file :

.. code:: python

   SimSample.write_mod(formats=['pkl', 'parquet'])

Post Sim Tools
--------------

The post-sim-tools module contains functions to run on simulated lcs.

SNR Selection
~~~~~~~~~~~~~

The simulation ignored effect of selection efficiency. To introduce this
effect the SNSimSample class has a SNR_select.

At this time the SNR_select function use an approximation to model SNR
detection probability:

.. math::


   P_\text{det}(SNR) = \frac{1}{1+\left(\frac{SNR_\text{mean}}{SNR}\right)^n}

where :math:`SNR_\text{mean}` is the SNR for which
:math:`P_\text{det} = 0.5` and n is given by a probability of detection
**p** for a given **:math:`SNR_p`** :

.. math::


   n = \frac{\ln\left(\frac{1 - p}{p}\right)}{\ln(SNR_\text{mean}) - \ln(SNR_p)}

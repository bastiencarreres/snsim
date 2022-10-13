Configuration input yaml file
=============================

The input file is a .yml with the following structure:

.. code:: yaml

   data:
       write_path: '/PATH/TO/OUTPUT'
       sim_name: 'NAME OF SIMULATION'
       write_format: 'format' or ['format1','format2']  # Optional default pkl, parquet
   survey_config:
       survey_file: '/PATH/TO/FILE'
       band_dic: {'r':'ztfr','g':'ztfg','i':'ztfi'}  # Optional -> if bandname in the database doesn't correpond to those in sncosmo registery
       add_data: ['keys1', 'keys2', ...]  # Optional add survey file keys to metadata
       survey_cut: {'key1': ["conditon1","conditon2",...], 'key2':["conditon1"],...}  # Optional SQL conditions on key
       key_dic: {'column_name': 'new_column_name', etc}  # Optional, to change columns names
       zp: INSTRUMENTAL ZEROPOINT  # Optional, default given by survey file)
       sig_zp: UNCERTAINTY ON ZEROPOINT  # Optional, default given by survey file)
       sig_psf: GAUSSIAN PSF SIGMA  # Optional, default given by survey file as FWHMeff
       noise_key: [key, type] type can be 'mlim5' or 'skysigADU'                      
       ra_size: RA FIELD SIZE in DEG
       dec_size: DEC FIELD SIZE in DEG
       gain: CCD GAIN e-/ADU  # Optional, default given by survey file
       start_day: MJD NUMBER or 'YYYY-MM-DD'  # Optional, default given by survey file
       end_day: MJD NUMBER or 'YYYY-MM-DD'  # Optional, default given by survey file
       duration: SURVEY DURATION (DAYS)  # Optional, default given by survey file
       field_map: FIELD MAP FILE  # Optional, default is rectangle field
       fake_skynoise: [VALUE, 'add' or 'replace']  # Optional, default is no fake_skynoise
       sub_field: 'sub_field_key' # Used to divided observation in CCD quadrant for example
   sim_par:
       z_range: [ZMIN, ZMAX]
       randseed: RANDSEED TO REPRODUCE SIMULATION  # Optional
       nep_cut: [[nep_min1,Tmin,Tmax], [nep_min2,Tmin2,Tmax2,'filter1'], ...] EP CUTS # Optional
       duration_for_rate: FAKE DURATION ONLY USE TO GENERATE N OBJ  # Optional
   snia_gen:
       force_n: NUMBER OF OBJ TO GENERATE  # Optional
       sn_rate: rate of SN/Mpc^3/year or 'ptf19'  # Optional, default=3e-5
       rate_pw: rate = sn_rate*(1+z)^rate_pw  # Optional, default=0
       M0: SN ABSOLUT MAGNITUDE
       mag_sct: SN INTRINSIC COHERENT SCATTERING
       sct_model: 'SCATTERING_MODEL_NAME' USE WAVELENGHT DEP MODEL FOR SN INT SCATTERING
       mod_fcov: True or False # Use the covariance of simulation model to scatter flux Optional, default = False
       model_config:
           model_name: 'THE MODEL NAME' #  Example : 'salt2'
           model_dir: '/PATH/TO/SALT/MODEL'
           # Model parameters : here example for salt
           alpha: STRETCH CORRECTION = alpha*x1
           beta: COLOR CORRECTION = -beta*c
           dist_x1: [MEAN X1, SIGMA X1] or [MEAN X1, SIGMA_X1_LOW, SIGMA_X1_HIGH] or 'N21'
           dist_c: [MEAN C, SIGMA C] or [SIGMA_C_LOW, SIGMA_C_HIGH]
   cosmology:  # Follow astropy formalism
       Om0: MATTER DENSITY  
       H0: HUBBLE CONSTANT
   cmb:
       v_cmb: OUR PECULIAR VELOCITY  # Optional, default = 369.82 km/s
       l_cmb: GAL L OF CMB DIPOLE  # Optional, default = 264.021            
       b_cmb: GAL B OF CMB DIPOLE  # Optional, default = 48.253   
   mw_dust: # Optional
       model: MOD_NAME
       rv: Rv # Optional, default Rv = 3.1
   vpec_dist: # Optional
        mean_vpec: MEAN SN PECULIAR VEL
        sig_vpec: SIGMA VPEC
   host: # Optional 
        host_file: '/PATH/TO/HOSTFILE' 
        distrib: 'as_sn' or 'as_host' or 'mass_weight' # Optional, default = 'as_sn'
        key_dic: {'column_name': 'new_column_name', ...}  # Optional, to change columns names
   dipole:  # Experimental dipole, optional
        coord: [RA, Dec]  # Direction of the dipole
        # alpha dipole = A + B * cos theta
        A: A_parameter  
        B: B_parameter

data
----

This section of the yaml file only contains information about output
files of the simulation :

-  **write_path** is the path to the output directory. *type* : str

-  **sim_name** is the simulation file name. *type* : str

-  **write_format** is the desired output format(s), only **parquet** or
   **pkl** are available. *type* : str or list(str). *Optional* :
   default is **parquet and pkl**. Note that **parquet** working only if
   you have **pyarrow** and **json** python modules installed.

survey_config
-------------

This section contains informations about the survey configuration :

-  **survey_file** is the path to the SQL database or the CSV file that
   describe observations.
-  **ra_size** is the Right Ascension size of the field in DEG.
-  **dec_size** is the Declinaison size of the field in DEG.
-  **noise_key** is a list that contains the **key** used in the SQL
   database for the noise, and the type of noise : **skysigADU** if it’s
   directly the sky noise in **ADU** units, **mlim5** if it’s the
   limiting magnitude at 5 :math:`\sigma`. *type* : list(str).
-  **key_dic** is a dictionary to use if you use csv file in order to
   change columns names.
-  **gain** is the gain of the CCD in :math:`e^-` / ADU. *type* : float.
   *Optional* : If not set, gain is taken in the SQL database.
-  **zp** is a constant zero point to use in simulation. *type* : float.
   *Optional* : If not set, zero point is taken in the SQL database.
-  **sig_zp** is the error on zero point. *type* : float. *Optional* :
   If not set, this parameter is taken in the SQL database.
-  **sig_psf** is the PSF scale. *type* : float. *Optional* : If not
   set, the PSF is taken in the SQL database, to following LSST OpSim
   structure the PSF in the database is take has the **FWHM**
   (:math:`FWHM = 2 \sqrt{2 \ln(2)} \sigma_{PSF}`).
-  **start_day** is the starting day in **MJD** or in formated str
   **‘YYYY-MM-DD’**. *type* : float or str. *Optional* : default is the
   first day of the SQL database.
-  **end_day** same as **start_day** but for the end of the survey.
   *type* : float or str. *Optional* : default is the last day of the
   SQL database.
-  **duration** : instead of setting an **end_day** you can specify a
   duration in **days**. *type* : float. *Optional* : the **duration**
   is ignored if an **end_day** is configured.
-  **field_map** is a file that describe the field geometry, more
   information `here <obsfile.md>`__. *type* : str. *Optional* : default
   is a rectangle ra_size :math:`\times` dec_size field.
-  **sub_field** correspond to the sub_field key of the database, it’s
   allow to have a database with observations indexed by subfield and
   not by field. *type* : str. *Optional* : If you don’t use a database
   with subfields, however the code will run but all subfields
   observations will be take into account.
-  **band_dic** is a dictionnary that map bands names in the database to
   bands names in *sncosmo* . *type* dic. *Optional*
-  **survey_cut** is used to put cuts on the SQL query of the
   observations, it’s a dictionary : {‘key1’:
   [“conditon1”,“conditon2”,…], ‘key2’:[“conditon1”],…} where keys are
   any database keys and condition are str SQL queries. *type* : dic.
   *Optional*
-  **add_data** is a list of database key that you want to retrieve in
   lightcurves tables. *type* : list(str). *Optional*
-  **fake_skynoise** allow to add or replace the skynoise term. The fake
   skynoise is multiply by the **PSF** if there is one given. This is a
   list : [VALUE, ‘add’ or ‘replace’] the VALUE is the skynoise value in
   ADU, if you use ‘add’ the fake_skynoise is added to skynoise from the
   SQL database, else, if you use ‘replace’ the skynoise from SQL
   database is just ignored. Note that if you set **fake_skynoise** with
   ‘replace’ option and **sig_psf** = 0, the skynoise is exactly the
   **fake_skynoise** value. *type* : list(float, str). *Optional*
   default is no **fake_skynoise**

sim_par
-------

-  **z_range** cosmological redshift range in which generate obj. *type*
   : list(float).
-  **randseed** the randseed used to produce the simulation. *type* :
   int. *Optional* : default is random.
-  **duration_for_rate** allow to use a different duration for the
   survey and the number of SN, it must be in **days**. *type* : float.
   *Optional*
-  **nep_cut** is a filter function to only generate SN with a minimum
   number of epochs. It can be just a number or you can specify
   different requirements for each band. *type* int or list. *Optional*

astrobj_gen
-----------

Here we present how to generate different astrobj : each astrobj
configuration us represented by a yaml section named astrobj_gen.
Available astrobj are : \* SNIa (Future implementaiton for new astrobj)

Common properties
~~~~~~~~~~~~~~~~~

Common properties to all astro obj

-  **force_n** force the number of SN to generate. *type* int.
   *Optional*
-  **rate** is the rate of SN in units of SN/Mpc\ :math:`^3`/year.
   *type* : float or str. *Optional* : default value is
   :math:`3 \times 10^{-5}\ SN.Mpc^{-3}.year^{-1}` .
-  **rate_pw** give an evolution of the rate with redshift as
   :math:`r_v(z) = (1+z)^{rate_pw} r_v(0)`. *type* float. *Optional* :
   default is 0.
-  **mod_fcov** use or not the simulation model covariance to scatter
   flux. *type* : boolean. *Optional* : default is False.

Flux covariance come from **sncosmo.Model.bandfluxcov()** and is apply
using :

.. code:: python

   flux += np.random.multivariate_normal(np.zeros(len(fluxcov)),
                                         fluxcov,
                                         check_valid='ignore',
                                         method='eigh')

-  **model_config** contains parameters of the model used to simulated
   SN Ia light curves.

   -  **model_name** give the name of your model.
   -  **model_dir** give the path to the model files. *type* : str.
      *Optional* : if not given, use **model_name** as *sncosmo*
      built-in source.

snia_gen
~~~~~~~~

This section concern the type Ia supernovae properties.

-  **M0** is the absolute magnitude of Supernovae in rest-frame Bessell
   B band. *type* : float or str.

   Possibilities are :

   -  Directly give a float value
   -  Give ‘jla’ : use the `JLA <https://arxiv.org/abs/1401.4064>`__
      best fit value :math:`M_0 = -19.05` for :math:`H_0 = 70` km/s/Mpc.
      :math:`M_0` is rescale in function of the :math:`H_0` set in
      cosmology.

-  **mag_sct** the SN Ia coherent intrinsic scattering. For each SN
   :math:`M_0 \rightarrow M_0 + \sigma_M`. *type* : float.

-  **rate**

   Additional possibilities are:

   -  Give ‘ptf19’ : use the
      `PTF19 <https://arxiv.org/abs/1903.08580>`__ SN Ia rate
      :math:`r_v = 2.43 \times10^{-5} \ SN.Mpc^{-3}.year^{-1}` for
      :math:`H_0 = 70` km/s/Mpc. :math:`r_v` is rescale in function of
      the :math:`H_0` set in cosmology.

   Note that the rate is used to generate the redshift distribution.

-  **sct_mod** a model of wavelength dependant scattering. Follow
   nomanclature of `Kessler et
   al. 2012 <https://arxiv.org/abs/1209.2482>`__. *type* : str.
   *Optional*

   Possibilities are:

   -  **‘G10’** for `Guy et
      al. 2010 <https://arxiv.org/abs/1010.4743>`__ model.
   -  **‘C11’** or **‘C11_0’** for `Chotard et
      al. 2011 <https://arxiv.org/abs/1103.5300>`__ model with
      correlation between U’ and U = 0, **‘C11_1’** for Cor(U’,U) = 1
      and **‘C11_2’** for Cor(U’,U) = -1.

-  Available model for **model_config**:

   -  all sncosmo **salt** models.

Salt 2 / 3
^^^^^^^^^^

-  **alpha** correspond to the stretch correction in Tripp relation :
   :math:`\alpha x_1`. *type* float.

-  **beta** correspond to the color correction in Tripp relation :
   :math:`\beta c`. *type* : float.

-  **dist_x1** represents the parameters of the stretch’s distribution.
   *type* : list(float) or str.

   Possibilities are:

   -  [MEAN, SIGMA] for gaussian distribution.
   -  [MEAN, SIGMA-, SIGMA+] for asymmetric gaussian distribution.
   -  ‘N21’ to use the distribution of `Nicolas et
      al. 2021 <https://arxiv.org/abs/2005.09441>`__

-  **dist_c** represents the parameters of the color’s distribution.
   *type* : list(float) .

   Possibilities are:

   -  [MEAN, SIGMA] for gaussian distribution.

   -  [MEAN, SIGMA-, SIGMA+] for asymmetric gaussian distribution.

mw_dust
-------

The model of Milky Way dust to apply. *Optional* : not set, no dust.

-  **model** the name of the MW dust to use. *type* : str. Possibilities
   are :

   -  **CCM89**

   -  **OD94**

   -  **F99**

-  **rv** MW :math:`R_V` value. *type* : float. *Optional* : default
   :math:`R_v=3.1`.

For more information go to the *sncosmo* documentation.

cosmology
---------

This section is about the cosmological model used in the simulation.

The first way of use is to just write the parameters following the
`astropy.cosmology.w0waCDM <https://docs.astropy.org/en/stable/api/astropy.cosmology.w0waCDM.html#astropy.cosmology.FlatLambdaCDM>`__
parameters names. At least you need to give the Hubble constant : **H0**
and the matter density at z=0 : **Om0**. If you don’t give any other
parameters the Universe is assumed flat with a cosmological constant.

The second way is to use the key **name** and load one of built-in
astropy cosmological model:

​ Possibilities are:

-  **‘planck18’**
-  **‘planck15’**
-  **‘planck13’**
-  **‘wmap9’**
-  **‘wmap7’**
-  **‘wmap5’**

cmb *optional*
--------------

This section set the CMB reference frame. Defaults values come from
`Planck18 <https://arxiv.org/pdf/1807.06205.pdf>`__

-  **v_cmb** is our peculiar velocity in the CMB frame in km/s. *type* :
   float. *Optional* : default is 620 km/s
-  **l_cmb** is the galactic longitude of the CMB dipole. *type* :
   float. *Optional* : default is 264.021 deg
-  **b_cmb** is the galactic longitude of the CMB dipole. *type* :
   float. *Optional* : default is 48.253 deg

vpec_dist *optional*
--------------------

This section describe the distribution of peculiar velocities. Peculiar
velocities are taken from a gaussian distribution.

Default is all vpec = 0.

-  **mean_vpec** is the mean of the gaussian distribution. *type* float
-  **sig_vpec** is the scale of the gaussian distribution. *type* float

host *optional*
---------------

The host configuration to place SN in host, see `here <hostfile.md>`__.

-  **host_file** is the path to the host_file, used to generate SN in
   hosts. *type* str

-  **key_dic** is a dictionary to change column name in order to
   correspond to what is needed (*cf* `host file doc <hostfile.md>`__)

-  **distrib** is the distribution to use for redshift. *type* str.

   The possibilities are:

   -  ‘as_sn’ : the simulation use the sn rate to generate redshifts
      distribution
   -  ‘as_host’ : the simulation use the host distribution to generate
      redshifts
   -  ‘mass_weight’ : host mass weight the distribution to generate
      redshifts as :math:`w_i = \frac{m_i}{\sum_i m_i}`

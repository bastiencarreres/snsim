AstrObj
=========

The AstrObj abstract class
---------------------------

`AstrObj` is an abstract class of objects that snsim used to model transients.

The basics attributes of an AstrObj are:

+--------------------+--------------------------------------------------+
| :code:`ID`         | Identification number, set to 0 by default       |
+--------------------+--------------------------------------------------+
| :code:`ra`         | Right ascension [rad]                            |
+--------------------+--------------------------------------------------+
| :code:`dec`        | Declination [rad]                                |
+--------------------+--------------------------------------------------+
| :code:`zcos`       | Cosmological redshift                            |
+--------------------+--------------------------------------------------+
| :code:`vpec`       | Peculiar velocities [km/s]                       |
+--------------------+--------------------------------------------------+
| :code:`zpcmb`      | Redshift contribution from the CMB dipole motion |
+--------------------+--------------------------------------------------+
| :code:`como_dist`  | Comoving distance [Mpc]                          |
+--------------------+--------------------------------------------------+

Derived properties can be called:


* The peculiar redshift :code:`zpec`: :math:`z_p = v_p / c`
* The redshift corrected from CMB dipole :code:`zCMB`:  :math:`z_\mathrm{CMB} = (1 + z_\mathrm{cos}) (1 + z_p) - 1`
* The observed redshift :code:`zobs`: :math:`z_\mathrm{obs} = (1 + z_\mathrm{cos}) (1 + z_{p,\mathrm{CMB}}) (1 + z_p) - 1`
* The distance modulus :code:`mu`: :math:`\mu = 5 \log_{10}((1 + z_\mathrm{cos}) (1 + z_{p,\mathrm{CMB}}) (1 + z_p)^2 r(z_\mathrm{cos}))`


Any model will add attributes depanding on what is needed to models the transient.
These new attributes are added through the definition of :code:`_obj_attrs`.

During the :code:`__init__` call, 
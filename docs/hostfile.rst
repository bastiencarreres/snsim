Host file
=========

The host file contain coordinates and peculiar velocities to simulate
SN, the needed keys are given in the next table

+--------------------+----------+------------------+------------------+
| redshift           | ra (rad) | dec (rad)        | v_radial (km/s)  |
+====================+==========+==================+==================+
| Redshift of the    | Right    | Declinaison of   | Velocity along   |
| host               | a        | the host         | the line of      |
|                    | scension |                  | sight            |
|                    | of the   |                  |                  |
|                    | host     |                  |                  |
+--------------------+----------+------------------+------------------+

Note that you can used the redshift distribution of your host to
generate SN redshift distribution.

+----------------------------------+
| mass                             |
+==================================+
| The host mass in arbitrary units |
+----------------------------------+

Additionally you can use a **mass** columns to weight the distribution
with host masses : :math:`w_i = \frac{m_i}{\sum_i m_i}`

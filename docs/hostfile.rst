Host file
=========

The host file contain coordinates and peculiar velocities to simulate
SN, the needed keys are given in the next table

+-----------------------+-----------------+------------------+----------------+
|         zcos          |      ra (rad)   | dec (rad)        | vpec (km/s)    |
+=======================+=================+==================+================+
| Cosmological redshift | Right ascension | Declinaison of   | Velocity along |
| of the host           | of the host     | the host         | the line of    |
|                       |                 |                  | sight          |
+-----------------------+-----------------+------------------+----------------+

Note that you can used the redshift distribution of your host to
generate SN redshift distribution.

+----------------------------------+
| mass                             |
+==================================+
| The host mass in arbitrary units |
+----------------------------------+

Additionally you can use a **mass** columns to weight the distribution
with host masses : :math:`w_i = \frac{m_i}{\sum_i m_i}`

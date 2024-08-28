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

Extra columns are necessary if you want to use the `distrib` configuration key to draw hosts.

As example,

+------------------------------------------+------------------------------------------+
| sm                                       | sfr                                      |
+==========================================+==========================================+
| The host stellar mass in arbitrary units | The host sfr in arbitrary units          |
+------------------------------------------+------------------------------------------+

can be used for `"mass"`, `"mass_sfr"` or `"sfr"` `distrib` options.


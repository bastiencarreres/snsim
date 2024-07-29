Observation database file
=========================

It’s a **Comma-separated values** (.csv) or **parquet** (.parquet)
file which contain observations informations. It’s used to find obs
epoch and their noise.

The SQL Table inside the SQL DataBase must be named **Summary**.

Required keys
-------------

The required data keys are resumed in the next table :

+-----------+-----------+-----------+-----------+-----------+-----------+
| expMJD    | filter    | fieldID   | fieldRA   | fieldDec  | noise_key |
|           |           |           | (rad)     | (rad)     |           |
+===========+===========+===========+===========+===========+===========+
| Obs time  | Obs band  | The ID of | Right     | De        | The       |
| in MJD    |           | the field | ascension | clinaison | column    |
|           |           |           | of the    | of the    | you want  |
|           |           |           | obs field | obs field | to use as |
|           |           |           |           |           | noise in  |
|           |           |           |           |           | the       |
|           |           |           |           |           | s         |
|           |           |           |           |           | imulation |
+-----------+-----------+-----------+-----------+-----------+-----------+

**noise_key** has to be defined in the `configuration yaml
file <./configfile.md>`__

If you use **csv** file you can define a **key_dic** to change columns
name to corresponds to what is needed.

You can set a different zero point and its error for each observation by
setting the two additional columns:

+-----------------------------------+-----------------------------------+
| zp                                | sig_zp                            |
+===================================+===================================+
| Zero point of the observation     | Uncertainty of the zeropoint      |
| (Optional if given in yaml)       | (Optional if given in yaml)       |
+-----------------------------------+-----------------------------------+

In addition you can take into account the variation of the PSF as the
**F**\ ull **W**\ idth at **H**\ alf **M**\ aximum 
:math:`FWHM = 2 \sqrt{2 \log(2)} \sigma_\mathrm{psf}`

+-------------------------------------------+
| fwhm_psf                                  |
+===========================================+
| The Full Width at Half Maximum of the PSF |
+-------------------------------------------+

And you can set a different gain for each observation by giving the
**gain** column :

+------------------------+
| gain                   |
+========================+
| The CCD gain in e-/ADU |
+------------------------+

Subfields
---------

If you want to use subfield index for observation properties or just set
the geometry of the field, you have to give a .dat file that give the
representation of the subfield, for example if you split your field into
a 4 x 4 grid, you have to put something like that in your .dat file :

.. code:: pseudocode

   ID01:ID02:ID03:ID04
   ID05:ID06:ID07:ID08
   ID09:ID10:ID11:ID12
   ID13:ID14:ID15:ID16

If a sub field is not observed you should set the ID value to -1.

In addition, you can add space between subfield by adding a header
(begin line with %) that defines some “space-symbols”: 

.. code:: pseudocode 

   % #:ra:0.13 
   % @:dec:0.13

   ID01:ID02:#:ID03:ID04 
   ID05:ID06:#:ID07:ID08 
   @ 
   ID09:ID10:#:ID11:ID12
   ID13:ID14:#:ID15:ID16 

In the previous example the symbol # is used
has a ra space of 0.13 degrees and the @ is used has a dec space of 0.13
degrees.

You can show the sub filed map by :

.. code:: python

   sim.survey.show_map()

.. figure:: _static/show_map.png
   :alt: show_map

   show_map

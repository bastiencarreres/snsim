Formula used in the simulation
==============================

SN Ia flux and distance moduli
------------------------------

The flux of a SN Ia in a band **b** at the obs-frame phase **p** is
simulate by *sncosmo* with the following formula :

.. math::


   F_b(p) = \frac{1}{1+z}\frac{1}{4\pi d_L^2}\int_0^{+\infty} \phi_b\left(\frac{\lambda}{1+z}, \frac{p}{1+z}\right)T_b\left(\lambda\right)\frac{\lambda}{hc} d\lambda

where :math:`\mathbf{\phi_b}` is the restframe flux density and
:math:`\mathbf{\lambda}` the obs-frame wavelength.

The flux is re-scaled in **ADU** units by applying the following factor:

.. math::


   F_b^{ADU} = 10^{-0.4 m_B} 10^{\left(ZP_{obs} - ZP_{AB}\right)}

The observed magnitude is given by the absolute magnitude
:math:`\mathbf{M_B}` and the distance moduli :math:`\mathbf{\mu}` :

.. math::


   m_B = M_B + \mu

In the simulation **:math:`\mu`** is computed as:

.. math::


   \mu = 5 \log\left((1+z_{vp})^2 (1+z_{2cmb}) (1+z_{cos})r(z_{cos})\right) + 25

with :

-  :math:`\mathbf{z_{cos}}` the cosmological redshift
-  :math:`\mathbf{z_{vp}}` the redshift due to the peculiar velocity of the
   SN / CMB
-  :math:`\mathbf{z_{2cmb}}` the redshift due to our peculiar motion / CMB
-  :math:`\mathbf{r(z)}` the comoving distance

Noise formula
-------------

The flux error is computed as :

.. math::


   \sigma^2_F = \frac{F}{G} + \sigma_{skynoise}^2 + \left(\frac{\ln(10)}{2.5}F\sigma_{zp}\right)^2

The first term is the Poisson noise with **G** the gain in $ e^- $ /
ADU.

The second term is the noise from sky flux. If there is a PSF given this
term take into account the PSF by applying :

.. math::


   \sigma_{skynoise}^2  *= 4\pi\sigma_{PSF}^2

If you use limiting magnitude at 5σ, sky-noise is computed as :

.. math::


   \sigma_{skynoise} = \frac{1}{5}10^{0.4\left(ZP - m_{5\sigma}\right)}

The last term is the propagation of the zero point incertitude.

# Plots 

## Light-curves

You can plot light-curves directly after simulation or using the **OpenSim** module.

```python
sim.plot_lc(SN_ID, mag=False, zp=25. , plot_sim=True, plot_fit=False, Jy=False)
```

Plot sim :

![](_static/lc_sim.png)

Plot  fit :

![](_static/lc_fit.png)



The same plot can be show in magnitude :

![lc_mag](_static/lc_mag.png)

Or in Jansky :

![lc_jy](_static/lc_jy.png)



## Mollweide map

You can plot the  directly after simulation or using the **OpenSim** module. But the **plot_field** option doesn't work with **OpenSim**.

You can pass **kwargs** for *matplotlib* scatter function.

```python
sim.plot_ra_dec(plot_vpec=False, plot_fields=False, **kwarg)
```

Plot without peculiar velocities :

![](_static/ra_dec_map.png)

Plot with peculiar velocities :

![](_static/ra_dec_map_vpec.png)

Adding the fields :

![](_static/ra_dec_fields.png)